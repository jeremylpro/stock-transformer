from __future__ import annotations

import pandas as pd

from src.data_news import validate_news


def _row(ticker, headline, when, url_hash=""):
    return {
        "ticker": ticker,
        "headline": headline,
        "published_utc": when,
        "source": "test",
        "url_hash": url_hash,
    }


def test_news_dedup_and_tz_across_dst_boundary():
    # US DST begins 2021-03-14 at 02:00 local: 06:59 UTC is 01:59 EST,
    # 07:00 UTC is 03:00 EDT.
    news = pd.DataFrame(
        [
            _row("aapl", "Apple wins a ruling", pd.Timestamp("2021-03-14 06:59", tz="UTC")),
            _row("AAPL", "Apple wins a ruling", pd.Timestamp("2021-03-14 06:59", tz="UTC")),
            _row("AAPL", "Apple opens a new store", pd.Timestamp("2021-03-14 07:00", tz="UTC")),
            _row("AAPL", "   ", pd.Timestamp("2021-03-14 08:00", tz="UTC")),
        ]
    )
    out = validate_news(news)
    assert len(out) == 2
    times = out.set_index("headline")["published_et"]
    assert str(times["Apple wins a ruling"]) == "2021-03-14 01:59:00-05:00"
    assert str(times["Apple opens a new store"]) == "2021-03-14 03:00:00-04:00"
    assert out["published_utc"].dt.tz is not None


def test_news_dedup_on_url_hash():
    news = pd.DataFrame(
        [
            _row("AAPL", "First headline", pd.Timestamp("2024-01-02 14:00", tz="UTC"), "abc"),
            _row("AAPL", "Different headline", pd.Timestamp("2024-01-02 15:00", tz="UTC"), "abc"),
            _row("MSFT", "Same hash other ticker", pd.Timestamp("2024-01-02 15:00", tz="UTC"), "abc"),
        ]
    )
    out = validate_news(news)
    assert len(out) == 2
    assert set(out["ticker"]) == {"AAPL", "MSFT"}


def test_news_end_date_filter():
    news = pd.DataFrame(
        [
            _row("AAPL", "Before end", pd.Timestamp("2024-01-02 14:00", tz="UTC")),
            _row("AAPL", "After end", pd.Timestamp("2024-02-01 14:00", tz="UTC")),
        ]
    )
    out = validate_news(news, end="2024-01-31")
    assert out["headline"].tolist() == ["Before end"]


def test_news_accepts_epoch_seconds():
    news = pd.DataFrame(
        [_row("AAPL", "Epoch headline", pd.Timestamp("2024-01-02 14:00", tz="UTC").timestamp())]
    )
    out = validate_news(news)
    assert out.loc[0, "published_utc"] == pd.Timestamp("2024-01-02 14:00", tz="UTC")


def test_scrub_secret_hides_api_token():
    from src.data_news import _scrub_secret

    text = "HTTPSConnectionPool: /company-news?symbol=AAPL&token=secret123&from=2024-01-01"
    scrubbed = _scrub_secret(text)
    assert "secret123" not in scrubbed
    assert "token=***" in scrubbed


def test_news_chunks_stay_under_provider_result_cap():
    from datetime import date

    from src.data_news import NEWS_CHUNK_DAYS, _date_chunks

    chunks = list(_date_chunks(date(2026, 1, 1), date(2026, 3, 15)))
    assert chunks[0] == (date(2026, 1, 1), date(2026, 1, 7))
    assert chunks[-1][1] == date(2026, 3, 15)
    assert all((b - a).days + 1 <= NEWS_CHUNK_DAYS for a, b in chunks)
    for (_, prev_end), (next_start, _) in zip(chunks, chunks[1:]):
        assert (next_start - prev_end).days == 1
