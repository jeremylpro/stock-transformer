from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.labels import build_labels


def _prices(rows):
    dates = pd.bdate_range("2024-01-02", periods=len(rows))
    return pd.DataFrame(
        {
            "ticker": "AAPL",
            "date": dates,
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": [1.0] * len(rows),   # deliberately not used for returns
            "adj_close": rows,
            "volume": 1000,
        }
    )


def test_labels_use_adj_close_and_zero_return_is_down():
    prices = _prices([100.0, 110.0, 110.0, 99.0])
    labels = build_labels(prices)
    assert len(labels) == 3
    assert labels.loc[0, "log_ret_next"] == pytest.approx(np.log(110 / 100))
    assert labels.loc[0, "y_next_up"] == 1
    assert labels.loc[1, "log_ret_next"] == 0.0
    assert labels.loc[1, "y_next_up"] == 0
    assert labels.loc[2, "y_next_up"] == 0


def test_labels_require_adj_close_column(prices):
    with pytest.raises(AssertionError, match="adj_close"):
        build_labels(prices.drop(columns=["adj_close"]))


def test_labels_drop_final_row_per_ticker(prices):
    labels = build_labels(prices)
    for ticker, group in labels.groupby("ticker"):
        ticker_prices = prices[prices["ticker"] == ticker]
        assert group["date"].max() < ticker_prices["date"].max()
        assert group["log_ret_next"].notna().all()
