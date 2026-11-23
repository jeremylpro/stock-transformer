from __future__ import annotations

import pandas as pd
import pytest

from src.data_prices import PriceValidationError, _normalize_columns, validate_prices


def test_price_validation_drops_bad_rows_and_keeps_adj_close(prices):
    bad = prices.copy()
    bad.loc[0, "adj_close"] = 0.0
    bad.loc[1, "open"] = None
    bad.loc[2, "adj_close"] = -5.0
    clean = validate_prices(bad)
    assert len(clean) == len(bad) - 3
    assert (clean["adj_close"] > 0).all()
    assert not clean[["open", "high", "low", "close"]].isna().any().any()
    assert "adj_close" in clean.columns


def test_price_validation_rejects_duplicates(prices):
    dup = pd.concat([prices, prices.iloc[[0]]], ignore_index=True)
    with pytest.raises(PriceValidationError, match="duplicate"):
        validate_prices(dup)


def test_price_validation_rejects_missing_columns(prices):
    with pytest.raises(PriceValidationError, match="missing columns"):
        validate_prices(prices.drop(columns=["adj_close"]))


def test_price_validation_missing_days_are_not_filled(prices):
    aapl_dates = prices.loc[prices["ticker"] == "AAPL", "date"].sort_values()
    removed = set(aapl_dates.iloc[5:15])
    partial = prices[~((prices["ticker"] == "AAPL") & prices["date"].isin(removed))]
    clean = validate_prices(partial)
    assert not clean.loc[clean["ticker"] == "AAPL", "date"].isin(removed).any()
    assert len(clean) == len(partial)


def test_normalize_columns_accepts_naive_and_aware_indices():
    index = pd.bdate_range("2024-01-02", periods=3)
    frame = pd.DataFrame(
        {
            "Open": [1.0, 2.0, 3.0],
            "High": [1.0, 2.0, 3.0],
            "Low": [1.0, 2.0, 3.0],
            "Close": [1.0, 2.0, 3.0],
            "Adj Close": [1.0, 2.0, 3.0],
            "Volume": [100, 100, 100],
        },
        index=index,
    )
    naive = _normalize_columns(frame, "aapl")
    aware = _normalize_columns(
        frame.set_index(index.tz_localize("America/New_York")), "aapl"
    )
    for out in (naive, aware):
        assert out["ticker"].eq("AAPL").all()
        assert out["date"].tolist() == list(index)
        assert "adj_close" in out.columns
