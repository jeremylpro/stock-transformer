from __future__ import annotations

import pandas as pd

from src.alignment import assign_headlines_to_trading_days, eligible_headlines_for_day, market_close


def _news(rows):
    return pd.DataFrame(
        [
            {
                "ticker": ticker,
                "headline": headline,
                "published_utc": pd.Timestamp(when, tz="America/New_York").tz_convert("UTC"),
                "source": "test",
                "url_hash": str(i),
            }
            for i, (ticker, headline, when) in enumerate(rows)
        ]
    )


def test_alignment_cutoff_includes_1559_excludes_1600_and_1601():
    day = "2024-01-08"  # Monday
    news = _news(
        [
            ("AAPL", "before close", "2024-01-08 15:59"),
            ("AAPL", "exactly close", "2024-01-08 16:00"),
            ("AAPL", "after close", "2024-01-08 16:01"),
        ]
    )
    eligible = eligible_headlines_for_day(news, day)
    assert eligible["headline"].tolist() == ["before close"]
    assert (eligible["published_utc"] < pd.Timestamp(market_close(day)).tz_convert("UTC")).all()


def test_weekend_news_assignment():
    days = pd.bdate_range("2024-01-05", "2024-01-09")  # Fri, Mon, Tue
    news = _news(
        [
            ("MSFT", "saturday story", "2024-01-06 10:00"),
            ("MSFT", "sunday story", "2024-01-07 12:00"),
            ("MSFT", "monday morning", "2024-01-08 09:00"),
        ]
    )
    assigned = assign_headlines_to_trading_days(news, days)
    mapping = assigned.set_index("headline")["trading_day"].dt.date.astype(str).to_dict()
    assert mapping == {
        "saturday story": "2024-01-08",
        "sunday story": "2024-01-08",
        "monday morning": "2024-01-08",
    }


def test_headlines_after_close_roll_to_next_trading_day():
    days = pd.bdate_range("2024-01-08", "2024-01-10")
    news = _news(
        [
            ("AAPL", "close", "2024-01-08 16:00"),
            ("AAPL", "after close", "2024-01-08 16:01"),
            ("AAPL", "next morning", "2024-01-09 09:30"),
        ]
    )
    assigned = assign_headlines_to_trading_days(news, days)
    mapping = assigned.set_index("headline")["trading_day"].dt.date.astype(str).to_dict()
    assert mapping == {
        "close": "2024-01-09",
        "after close": "2024-01-09",
        "next morning": "2024-01-09",
    }


def test_far_future_headline_never_assigned():
    days = pd.bdate_range("2024-01-08", "2024-01-12")
    news = _news([("AAPL", "future", "2024-06-01 10:00")])
    assigned = assign_headlines_to_trading_days(news, days)
    assert assigned.empty


def test_every_assigned_headline_precedes_its_day_close():
    days = pd.bdate_range("2024-01-05", "2024-01-12")
    news = _news(
        [
            ("AAPL", "fri after close", "2024-01-05 16:30"),
            ("AAPL", "sat", "2024-01-06 11:00"),
            ("AAPL", "mon before close", "2024-01-08 15:59"),
        ]
    )
    assigned = assign_headlines_to_trading_days(news, days)
    for _, row in assigned.iterrows():
        close = market_close(row["trading_day"]).astimezone(row["published_utc"].tzinfo)
        assert row["published_utc"] < pd.Timestamp(close)
