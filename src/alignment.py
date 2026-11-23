"""The single owner of the 16:00 ET news-alignment rule (G1).

No other module may re-implement the eligibility comparison. A headline is
eligible for trading day *T* iff its publication timestamp is strictly before
the market close of *T* (16:00 ET by default). Headlines are assigned to exactly
one trading day: the first trading day whose close is strictly after publication.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from .utils import get_logger

LOG = get_logger("alignment")


def _parse_close_time(close_time: str) -> time:
    try:
        hour, minute = (int(x) for x in str(close_time).split(":"))
        return time(hour=hour, minute=minute)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"close_time must be HH:MM, got {close_time!r}") from exc


def market_close(day: str | datetime | pd.Timestamp, *, market_tz: str = "America/New_York",
                 close_time: str = "16:00") -> datetime:
    """Return an aware datetime for the market close on ``day`` (ET)."""
    ts = pd.Timestamp(day)
    naive = datetime.combine(ts.date(), _parse_close_time(close_time))
    return naive.replace(tzinfo=ZoneInfo(market_tz))


def _ensure_utc(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit="s", utc=True, errors="coerce")
    parsed = pd.to_datetime(series, utc=True, errors="coerce", format="mixed")
    return parsed


def eligible_headlines_for_day(
    headlines: pd.DataFrame,
    day: str | datetime | pd.Timestamp,
    *,
    prev_close: datetime | None = None,
    market_tz: str = "America/New_York",
    close_time: str = "16:00",
) -> pd.DataFrame:
    """Return headlines assigned to ``day``.

    A headline is included when ``prev_close <= t_pub < t_close(day)``.
    ``prev_close`` is the previous trading day's close, so each headline is
    assigned to exactly one trading day and headlines after 16:00 ET roll into
    the next trading day. If ``prev_close`` is omitted, the window starts at
    00:00 ET on ``day`` (useful for the unit-level cutoff test).
    """
    if headlines.empty:
        return headlines.copy()

    out = headlines.copy()
    out["published_utc"] = _ensure_utc(out["published_utc"])
    cutoff = market_close(day, market_tz=market_tz, close_time=close_time)
    cutoff_utc = pd.Timestamp(cutoff).tz_convert("UTC")

    if prev_close is None:
        start = pd.Timestamp(datetime.combine(pd.Timestamp(day).date(), time(0, 0))).tz_localize(
            ZoneInfo(market_tz)
        )
    else:
        start = pd.Timestamp(prev_close)
        if start.tzinfo is None:
            start = start.tz_localize(ZoneInfo(market_tz))
    start_utc = start.tz_convert("UTC")

    mask = (out["published_utc"] >= start_utc) & (out["published_utc"] < cutoff_utc)
    return out.loc[mask].copy()


def assign_headlines_to_trading_days(
    news: pd.DataFrame,
    trading_days: list[str] | list[pd.Timestamp] | pd.Series,
    *,
    market_tz: str = "America/New_York",
    close_time: str = "16:00",
) -> pd.DataFrame:
    """Assign each headline to the next trading day whose close is after it.

    Returns a copy of ``news`` with a ``trading_day`` column. Headlines after the
    final trading day's close are omitted.
    """
    if news.empty:
        out = news.copy()
        out["trading_day"] = pd.Series(dtype="datetime64[ns]")
        return out
    if trading_days is None or len(trading_days) == 0:
        raise ValueError("trading_days must not be empty")

    days = pd.DatetimeIndex(pd.to_datetime(list(trading_days))).normalize().sort_values()
    days = days.drop_duplicates()
    first_lower = market_close(
        days[0] - pd.Timedelta(days=4), market_tz=market_tz, close_time=close_time
    )

    pieces = []
    for ticker, group in news.groupby("ticker", sort=True):
        group = group.copy()
        group["published_utc"] = _ensure_utc(group["published_utc"])
        for idx, day in enumerate(days):
            prev = first_lower if idx == 0 else market_close(
                days[idx - 1], market_tz=market_tz, close_time=close_time
            )
            eligible = eligible_headlines_for_day(
                group,
                day,
                prev_close=prev,
                market_tz=market_tz,
                close_time=close_time,
            )
            if not eligible.empty:
                eligible["trading_day"] = pd.Timestamp(day)
                pieces.append(eligible)

    if not pieces:
        out = news.iloc[0:0].copy()
        out["trading_day"] = pd.Series(dtype="datetime64[ns]")
        return out

    out = pd.concat(pieces, ignore_index=True)
    # Runtime guard: every assigned headline must precede that day's close (G1).
    closes_utc = pd.to_datetime(
        [market_close(d, market_tz=market_tz, close_time=close_time)
         for d in out["trading_day"]],
        utc=True,
    )
    published_utc = pd.to_datetime(out["published_utc"], utc=True)
    assert (published_utc.to_numpy() < closes_utc.to_numpy()).all(), \
        "G1 violated: a headline was assigned to a day whose close precedes publication"
    return out.sort_values(["ticker", "trading_day", "published_utc"]).reset_index(drop=True)
