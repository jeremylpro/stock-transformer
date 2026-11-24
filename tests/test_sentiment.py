from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.sentiment import daily_sentiment, score_headlines, vader_scores


def _news(rows):
    return pd.DataFrame(
        [
            {
                "ticker": ticker,
                "headline": headline,
                "published_utc": pd.Timestamp(when, tz="America/New_York").tz_convert("UTC"),
                "source": "test",
                "url_hash": str(i),
                "sent_score": score,
            }
            for i, (ticker, headline, when, score) in enumerate(rows)
        ]
    )


def test_sentiment_null_vs_zero(prices):
    news = _news([("AAPL", "good news", "2024-01-03 10:00", 0.5)])
    daily = daily_sentiment(news, prices)
    aapl = daily[daily["ticker"] == "AAPL"].set_index("date")
    no_news_day = aapl.index[0]
    row = aapl.loc[no_news_day]
    assert row["sent_count"] == 0
    assert np.isnan(row["sent_mean"])
    assert np.isnan(row["sent_std"])
    assert np.isnan(row["sent_lag1"])

    story_day = pd.Timestamp("2024-01-03")
    story = aapl.loc[story_day]
    assert story["sent_count"] == 1
    assert story["sent_mean"] == pytest.approx(0.5)
    assert np.isnan(story["sent_std"])  # std undefined for one headline


def test_sentiment_lag1_uses_previous_trading_day(prices):
    news = _news([("AAPL", "good news", "2024-01-03 10:00", 0.5)])
    daily = daily_sentiment(news, prices)
    aapl = daily[daily["ticker"] == "AAPL"].sort_values("date").set_index("date")
    assert aapl.loc[pd.Timestamp("2024-01-04"), "sent_lag1"] == pytest.approx(0.5)


def test_after_close_headline_goes_to_next_day(prices):
    news = _news([("AAPL", "after close", "2024-01-03 16:01", 0.9)])
    daily = daily_sentiment(news, prices)
    aapl = daily[daily["ticker"] == "AAPL"].sort_values("date").set_index("date")
    assert aapl.loc[pd.Timestamp("2024-01-03"), "sent_count"] == 0
    assert aapl.loc[pd.Timestamp("2024-01-04"), "sent_mean"] == pytest.approx(0.9)


def test_score_headlines_vader_range():
    scores = score_headlines(
        ["Company beats estimates", "Company misses estimates badly"], backend="vader"
    )
    assert list(scores.columns) == [
        "headline_idx", "prob_positive", "prob_negative", "prob_neutral", "sent_score",
    ]
    assert scores["sent_score"].between(-1, 1).all()
    assert scores.loc[0, "sent_score"] > scores.loc[1, "sent_score"]
    assert scores["sent_score"].equals(vader_scores(
        ["Company beats estimates", "Company misses estimates badly"]
    )["sent_score"])


def test_score_headlines_empty():
    out = score_headlines([], backend="vader")
    assert out.empty
    assert "sent_score" in out.columns


def test_select_device_returns_a_torch_device():
    import torch

    from src.sentiment import _select_device

    device = _select_device()
    assert isinstance(device, torch.device)
    assert device.type in {"cpu", "cuda", "mps"}
