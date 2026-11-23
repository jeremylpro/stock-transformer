"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


def make_prices(
    tickers: dict[str, float] | None = None,
    days: int = 80,
    start: str = "2024-01-02",
    seed: int = 3,
) -> pd.DataFrame:
    tickers = tickers or {"AAPL": 100.0, "MSFT": 200.0, "GOOGL": 150.0}
    dates = pd.bdate_range(start, periods=days)
    frames = []
    for i, (ticker, base) in enumerate(tickers.items()):
        rng = np.random.default_rng(seed + i)
        close = base * np.exp(np.cumsum(rng.normal(0.0005, 0.012, size=len(dates))))
        open_ = close * (1 + rng.normal(0, 0.001, size=len(dates)))
        frames.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "date": dates,
                    "open": open_,
                    "high": np.maximum(open_, close) * 1.004,
                    "low": np.minimum(open_, close) * 0.996,
                    "close": close,
                    "adj_close": close,
                    "volume": rng.integers(1_000_000, 3_000_000, size=len(dates)),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def prices() -> pd.DataFrame:
    return make_prices()


@pytest.fixture
def zero_sentiment(prices: pd.DataFrame) -> pd.DataFrame:
    dates = prices[["ticker", "date"]].drop_duplicates().sort_values(["ticker", "date"])
    return pd.DataFrame(
        {
            "ticker": dates["ticker"].to_numpy(),
            "date": dates["date"].to_numpy(),
            "sent_mean": np.nan,
            "sent_std": np.nan,
            "sent_count": 0,
            "sent_lag1": np.nan,
            "sent_vader_mean": np.nan,
        }
    )
