"""Price ingestion and validation (canonical adjusted OHLCV panel)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from .utils import get_logger, save_parquet

LOG = get_logger("data_prices")

PRICE_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]
_OHLC = ["open", "high", "low", "close"]


class PriceValidationError(ValueError):
    """Raised when a price panel violates the canonical schema or rules."""


def _normalize_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Flatten yfinance's single/multi-index columns into the canonical names."""
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance uses (Price, Ticker); keep the first level.
        df = df.copy()
        df.columns = [str(c[0]) for c in df.columns]
    mapping = {
        "Open": "open", "High": "high", "Low": "low", "Close": "close",
        "Adj Close": "adj_close", "AdjClose": "adj_close", "Volume": "volume",
        "open": "open", "high": "high", "low": "low", "close": "close",
        "adj_close": "adj_close", "volume": "volume",
    }
    df = df.rename(columns=mapping)
    df = df.loc[:, ~df.columns.duplicated()]
    df["ticker"] = ticker.upper()
    if "date" not in df.columns:
        df = df.reset_index().rename(columns={"Date": "date", "index": "date", "Datetime": "date"})
    dates = pd.to_datetime(df["date"], errors="coerce")
    if getattr(dates.dt, "tz", None) is not None:
        # Keep the provider's wall-clock trading day, then store it tz-naive.
        dates = dates.dt.tz_localize(None)
    df["date"] = dates.dt.normalize()
    missing = [c for c in PRICE_COLUMNS if c not in df.columns]
    if missing:
        raise PriceValidationError(f"{ticker}: downloaded data missing columns {missing}")
    return df[PRICE_COLUMNS]


def _download_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    # yfinance's end is exclusive; add one day so `end` is inclusive.
    end_inclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    LOG.info("Downloading %s from %s to %s via yfinance", ticker, start, end)
    raw = yf.download(
        ticker,
        start=start,
        end=end_inclusive,
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        raise PriceValidationError(
            f"yfinance returned no rows for {ticker} ({start}..{end}); "
            "check the ticker and your network/certificate setup"
        )
    return _normalize_columns(raw, ticker)


def _download_alphavantage(ticker: str, start: str, end: str, api_key: str) -> pd.DataFrame:
    import requests

    LOG.info("Downloading %s via Alpha Vantage", ticker)
    resp = requests.get(
        "https://www.alphavantage.co/query",
        params={
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": ticker,
            "outputsize": "full",
            "apikey": api_key,
        },
        timeout=60,
    )
    resp.raise_for_status()
    payload = resp.json()
    if "Time Series (Daily)" not in payload:
        raise PriceValidationError(
            f"Alpha Vantage did not return daily data for {ticker}: "
            f"{payload.get('Note') or payload.get('Information') or list(payload)[:3]}"
        )
    rows = []
    for day, values in payload["Time Series (Daily)"].items():
        rows.append(
            {
                "date": day,
                "open": values.get("1. open"),
                "high": values.get("2. high"),
                "low": values.get("3. low"),
                "close": values.get("4. close"),
                "adj_close": values.get("5. adjusted close"),
                "volume": values.get("6. volume"),
            }
        )
    df = pd.DataFrame(rows)
    df["ticker"] = ticker.upper()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
    return df[PRICE_COLUMNS]


def _cache_path(cache_dir: str, ticker: str) -> Path:
    return Path(cache_dir) / f"prices_{ticker.upper()}.parquet"


def _fetch_one(
    ticker: str,
    start: str,
    end: str,
    cache_dir: str,
    source: str,
    api_key: str | None,
) -> pd.DataFrame:
    cache = _cache_path(cache_dir, ticker)
    cached = None
    if cache.exists():
        try:
            cached = pd.read_parquet(cache)
            cached["date"] = pd.to_datetime(cached["date"]).dt.normalize()
            cached = cached[(cached["date"] >= pd.Timestamp(start))]
            cached = cached[cached["date"] <= pd.Timestamp(end)]
        except Exception as exc:  # pragma: no cover - corrupt cache recovery
            LOG.warning("Ignoring unreadable price cache %s: %s", cache, exc)
            cached = None

    if cached is not None and not cached.empty:
        last = cached["date"].max()
        if last >= pd.Timestamp(end) - pd.Timedelta(days=4):
            LOG.info("Using cached prices for %s (%d rows through %s)",
                     ticker, len(cached), last.date())
            return cached

    fetch_start = start
    if cached is not None and not cached.empty:
        fetch_start = (cached["date"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if fetch_start > end:
            return cached

    if source == "alphavantage":
        if not api_key:
            raise PriceValidationError(
                "ALPHAVANTAGE_API_KEY is not set; required for price_source=alphavantage"
            )
        fresh = _download_alphavantage(ticker, fetch_start, end, api_key)
    else:
        try:
            fresh = _download_yfinance(ticker, fetch_start, end)
        except Exception as exc:
            # Spec §2 fallback: Alpha Vantage when yfinance fails and a key exists.
            if api_key:
                LOG.warning("yfinance failed for %s (%s); falling back to Alpha Vantage",
                            ticker, exc)
                fresh = _download_alphavantage(ticker, fetch_start, end, api_key)
            else:
                raise

    if cached is not None and not cached.empty and not fresh.empty:
        combined = pd.concat([cached, fresh], ignore_index=True)
    elif cached is not None and not cached.empty:
        combined = cached
    else:
        combined = fresh
    combined = combined.drop_duplicates(subset=["ticker", "date"], keep="last")
    combined = combined.sort_values("date").reset_index(drop=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    save_parquet(combined, cache)
    return combined


def fetch_prices(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: str,
    *,
    source: str = "yfinance",
    price_file: str | None = None,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Return the canonical price panel, caching one Parquet per ticker.

    ``source="local"`` reads ``price_file`` instead of hitting the network (used by
    tests and offline demo runs). The yfinance path refetches only the missing tail.
    """
    if source == "local":
        if not price_file:
            raise PriceValidationError("price_file is required for local price source")
        df = pd.read_parquet(price_file)
        if "date" not in df.columns:
            df = df.reset_index().rename(columns={"index": "date"})
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["ticker"] = df["ticker"].astype(str).str.upper()
        df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
        df = df[df["ticker"].isin([t.upper() for t in tickers])]
        return df.sort_values(["ticker", "date"]).reset_index(drop=True)

    frames = []
    for ticker in tickers:
        frames.append(
            _fetch_one(ticker, start, end, cache_dir, source, api_key)
        )
        # Be polite to free data providers.
        if source == "yfinance":
            time.sleep(0.2)
    panel = pd.concat(frames, ignore_index=True)
    panel = panel[(panel["date"] >= pd.Timestamp(start)) & (panel["date"] <= pd.Timestamp(end))]
    return panel.sort_values(["ticker", "date"]).reset_index(drop=True)


def validate_prices(
    df: pd.DataFrame,
    *,
    max_missing_ratio: float = 0.05,
    calendar: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Apply §5.1 rules and return a clean panel.

    - required columns and dtypes
    - reject rows with ``adj_close <= 0`` or null OHLC
    - duplicate ``(ticker, date)`` is an error
    - warn (do not fill) when a ticker misses >5% of the panel calendar
    """
    missing_cols = [c for c in PRICE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise PriceValidationError(f"price panel missing columns: {missing_cols}")
    if df.empty:
        raise PriceValidationError("price panel is empty")

    out = df.copy()
    out["ticker"] = out["ticker"].astype(str).str.upper()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    for col in (*_OHLC, "adj_close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    dup = out.duplicated(subset=["ticker", "date"], keep=False)
    if dup.any():
        examples = out.loc[dup, ["ticker", "date"]].head(5).to_dict("records")
        raise PriceValidationError(f"duplicate (ticker, date) rows, e.g. {examples}")

    bad_ohlc = out[_OHLC].isna().any(axis=1)
    if bad_ohlc.any():
        LOG.warning("Dropping %d rows with null OHLC", int(bad_ohlc.sum()))
        out = out.loc[~bad_ohlc]

    no_adj = out["adj_close"].isna() | (out["adj_close"] <= 0)
    if no_adj.any():
        LOG.warning("Dropping %d rows with null/non-positive adj_close", int(no_adj.sum()))
        out = out.loc[~no_adj]

    bad_date = out["date"].isna()
    if bad_date.any():
        LOG.warning("Dropping %d rows with unparseable dates", int(bad_date.sum()))
        out = out.loc[~bad_date]

    if out.empty:
        raise PriceValidationError("no valid price rows remain after validation")

    out["volume"] = out["volume"].fillna(0).clip(lower=0)
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)

    calendar_days = (
        pd.DatetimeIndex(pd.to_datetime(calendar)).normalize().drop_duplicates()
        if calendar is not None else out["date"].drop_duplicates().sort_values()
    )
    for ticker, group in out.groupby("ticker", sort=True):
        if len(calendar_days):
            ratio = 1.0 - len(group) / len(calendar_days)
            if ratio > max_missing_ratio:
                LOG.warning(
                    "Ticker %s is missing %.1f%% of trading days (%d/%d); "
                    "gaps are kept as gaps (no forward fill)",
                    ticker, ratio * 100, len(calendar_days) - len(group), len(calendar_days),
                )
    return out


def load_prices(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def main(argv: Iterable[str] | None = None) -> None:
    from .pipeline import run

    parser = argparse.ArgumentParser(description="Ingest and validate daily prices")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    run(args.config, stage="prices", force=args.force)


if __name__ == "__main__":  # pragma: no cover
    main()
