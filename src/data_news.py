"""News ingestion (Finnhub) and validation."""

from __future__ import annotations

import argparse
import hashlib
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from .utils import get_logger, save_parquet

LOG = get_logger("data_news")

NEWS_COLUMNS = ["ticker", "headline", "published_utc", "source", "url_hash"]


class NewsAuthError(RuntimeError):
    """Raised when the news provider rejects the API key."""


class NewsFetchError(RuntimeError):
    """Raised when the news provider fails irrecoverably."""


class TokenBucket:
    """Small synchronous token bucket (Finnhub free tier: 60 requests/minute)."""

    def __init__(self, rate_per_minute: int = 55, capacity: int = 5):
        self.rate = max(rate_per_minute, 1) / 60.0
        self.capacity = max(capacity, 1)
        self.tokens = float(self.capacity)
        self.updated = time.monotonic()

    def take(self) -> None:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
        self.updated = now
        if self.tokens < 1.0:
            sleep_for = (1.0 - self.tokens) / self.rate
            LOG.debug("Rate limiter sleeping %.2fs", sleep_for)
            time.sleep(sleep_for)
            self.tokens = 0.0
            self.updated = time.monotonic()
        else:
            self.tokens -= 1.0


def _headline_hash(headline: str) -> str:
    normalized = " ".join(str(headline).lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _records_to_df(records: list[dict[str, Any]], ticker: str) -> pd.DataFrame:
    rows = []
    for item in records:
        headline = (item.get("headline") or "").strip()
        if not headline:
            continue
        url = item.get("url") or item.get("id") or ""
        rows.append(
            {
                "ticker": ticker.upper(),
                "headline": headline,
                "published_utc": pd.to_datetime(item.get("datetime"), unit="s", utc=True),
                "source": str(item.get("source") or ""),
                "url_hash": hashlib.sha256(str(url).encode("utf-8")).hexdigest()[:16]
                if url else _headline_hash(headline),
            }
        )
    return pd.DataFrame(rows, columns=NEWS_COLUMNS)


def _cache_path(cache_dir: str, ticker: str) -> Path:
    return Path(cache_dir) / f"news_{ticker.upper()}.parquet"


def _load_cached_news(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        df["published_utc"] = pd.to_datetime(df["published_utc"], utc=True)
        return df
    except Exception as exc:  # pragma: no cover - corrupt cache recovery
        LOG.warning("Ignoring unreadable news cache %s: %s", path, exc)
        return None


def _date_chunks(start: date, end: date, days: int = 90) -> Iterable[tuple[date, date]]:
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=days - 1), end)
        yield cursor, chunk_end
        cursor = chunk_end + timedelta(days=1)


def _fetch_finnhub_chunk(
    ticker: str,
    start: date,
    end: date,
    api_key: str,
    bucket: TokenBucket,
) -> list[dict[str, Any]]:
    bucket.take()
    try:
        resp = requests.get(
            "https://finnhub.io/api/v1/company-news",
            params={
                "symbol": ticker,
                "from": start.isoformat(),
                "to": end.isoformat(),
                "token": api_key,
            },
            timeout=45,
        )
    except requests.RequestException as exc:
        raise NewsFetchError(f"Finnhub request failed for {ticker}: {exc}") from exc

    if resp.status_code in {401, 403}:
        raise NewsAuthError(
            "Finnhub rejected the API key (HTTP 401/403). Set FINNHUB_API_KEY in .env."
        )
    if resp.status_code == 429:
        LOG.warning("Finnhub rate limit hit for %s; backing off 60s", ticker)
        time.sleep(60)
        return _fetch_finnhub_chunk(ticker, start, end, api_key, bucket)
    if resp.status_code >= 400:
        raise NewsFetchError(
            f"Finnhub HTTP {resp.status_code} for {ticker}: {resp.text[:200]}"
        )
    payload = resp.json()
    if isinstance(payload, dict):
        message = payload.get("error") or payload.get("message") or str(payload)
        raise NewsFetchError(f"Finnhub error for {ticker}: {message}")
    if not isinstance(payload, list):
        raise NewsFetchError(f"Unexpected Finnhub payload for {ticker}: {type(payload)}")
    return payload


def _fetch_one_finnhub(
    ticker: str,
    start: str,
    end: str,
    api_key: str,
    cache_dir: str,
    bucket: TokenBucket,
) -> pd.DataFrame:
    cache = _cache_path(cache_dir, ticker)
    cached = _load_cached_news(cache)
    end_date = pd.Timestamp(end).date()

    if cached is not None and not cached.empty:
        cached = cached[cached["published_utc"] <= pd.Timestamp(end, tz="UTC")]
        last = cached["published_utc"].max() if not cached.empty else None
        if last is not None and last.date() >= end_date - timedelta(days=2):
            LOG.info("Using cached news for %s (%d headlines through %s)",
                     ticker, len(cached), last.date())
            return cached
        fetch_start_date = last.date() + timedelta(days=1) if last is not None else \
            pd.Timestamp(start).date()
    else:
        fetch_start_date = pd.Timestamp(start).date()

    if fetch_start_date > end_date:
        return cached if cached is not None else pd.DataFrame(columns=NEWS_COLUMNS)

    frames: list[pd.DataFrame] = []
    empty_chunks = 0
    for chunk_start, chunk_end in _date_chunks(fetch_start_date, end_date):
        LOG.info("Fetching Finnhub news for %s %s..%s", ticker, chunk_start, chunk_end)
        records = _fetch_finnhub_chunk(ticker, chunk_start, chunk_end, api_key, bucket)
        if not records:
            empty_chunks += 1
        frames.append(_records_to_df(records, ticker))
    fresh = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=NEWS_COLUMNS)
    if empty_chunks:
        LOG.warning(
            "%s: %d Finnhub chunk(s) returned no headlines. The free tier only covers "
            "roughly the last year and may have coverage gaps.",
            ticker, empty_chunks,
        )

    combined = pd.concat([cached, fresh], ignore_index=True) if cached is not None and \
        not cached.empty else fresh
    if combined.empty:
        combined = pd.DataFrame(columns=NEWS_COLUMNS)
    combined = combined.drop_duplicates(subset=["ticker", "url_hash"], keep="last")
    combined = combined.sort_values(["ticker", "published_utc"]).reset_index(drop=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    save_parquet(combined, cache)
    return combined


def fetch_news(
    tickers: list[str],
    start: str,
    end: str,
    api_key: str,
    cache_dir: str,
    *,
    source: str = "finnhub",
    news_file: str | None = None,
    rate_per_minute: int = 55,
) -> pd.DataFrame:
    """Fetch raw headlines per ticker with pagination, caching and a token bucket.

    ``source="local"`` reads ``news_file`` (tests and offline demo runs).
    """
    if source == "local":
        if not news_file:
            raise NewsFetchError("news_file is required for local news source")
        df = pd.read_parquet(news_file) if not str(news_file).endswith(".csv") \
            else pd.read_csv(news_file)
        if "published_utc" not in df.columns:
            raise NewsFetchError("local news file must have a published_utc column")
        df["published_utc"] = pd.to_datetime(df["published_utc"], utc=True)
        df["ticker"] = df["ticker"].astype(str).str.upper()
        df = df[df["ticker"].isin([t.upper() for t in tickers])]
        return df.reset_index(drop=True)

    if not api_key:
        raise NewsAuthError(
            "FINNHUB_API_KEY is not set. Put it in .env, or use data.news_source: "
            "'local' with data.news_file for an offline demo/test run."
        )
    bucket = TokenBucket(rate_per_minute=rate_per_minute)
    frames = [
        _fetch_one_finnhub(t.upper(), start, end, api_key, cache_dir, bucket)
        for t in tickers
    ]
    panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=NEWS_COLUMNS)
    return panel.sort_values(["ticker", "published_utc"]).reset_index(drop=True)


def validate_news(
    df: pd.DataFrame,
    *,
    end: str | date | None = None,
    market_tz: str = "America/New_York",
) -> pd.DataFrame:
    """Apply §5.2 validation: parse, dedupe, drop empties, convert to ET."""
    missing = [c for c in NEWS_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"news panel missing columns: {missing}")
    if df.empty:
        out = df.copy()
        out["published_utc"] = pd.to_datetime(out["published_utc"], utc=True)
        out["published_et"] = out["published_utc"].dt.tz_convert(market_tz)
        return out

    out = df.copy()
    out["ticker"] = out["ticker"].astype(str).str.upper()
    out["headline"] = out["headline"].astype("string").fillna("").str.strip()

    # Accept epoch seconds, ISO strings, and tz-aware/naive datetimes.
    published = out["published_utc"]
    if pd.api.types.is_numeric_dtype(published):
        out["published_utc"] = pd.to_datetime(published, unit="s", utc=True, errors="coerce")
    else:
        parsed = pd.to_datetime(published, utc=True, errors="coerce", format="mixed")
        out["published_utc"] = parsed

    bad_ts = out["published_utc"].isna()
    if bad_ts.any():
        LOG.warning("Dropping %d headlines with unparseable timestamps", int(bad_ts.sum()))
        out = out.loc[~bad_ts]

    empty = out["headline"].str.len() == 0
    if empty.any():
        LOG.warning("Dropping %d empty/whitespace headlines", int(empty.sum()))
        out = out.loc[~empty]

    source = out["source"].astype("string").fillna("")
    url_hash = out["url_hash"].astype("string").fillna("")
    computed = out["headline"].map(_headline_hash)
    out["url_hash"] = url_hash.where(url_hash.str.len() > 0, computed)
    out["source"] = source

    before = len(out)
    out = out.drop_duplicates(subset=["ticker", "url_hash"], keep="first")
    if len(out) != before:
        LOG.warning("Removed %d duplicate headlines", before - len(out))

    if end is not None:
        end_date = pd.Timestamp(end).date()
        out["published_et"] = out["published_utc"].dt.tz_convert(market_tz)
        after_end = out["published_et"].dt.date > end_date
        if after_end.any():
            LOG.warning("Dropping %d headlines published after dataset end %s",
                        int(after_end.sum()), end_date)
            out = out.loc[~after_end]

    out["published_et"] = out["published_utc"].dt.tz_convert(market_tz)
    out = out.sort_values(["ticker", "published_utc"]).reset_index(drop=True)
    return out


def main(argv: Iterable[str] | None = None) -> None:
    from .pipeline import run

    parser = argparse.ArgumentParser(description="Ingest and validate daily news headlines")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    run(args.config, stage="news", force=args.force)


if __name__ == "__main__":  # pragma: no cover
    main()
