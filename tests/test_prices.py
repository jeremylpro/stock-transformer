from __future__ import annotations

import pandas as pd
import pytest

from src.data_prices import PriceValidationError, validate_prices


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
    # Drop a chunk of AAPL dates and confirm they stay missing (no forward fill).
    aapl_dates = prices.loc[prices["ticker"] == "AAPL", "date"].sort_values()
    removed = set(aapl_dates.iloc[5:15])
    partial = prices[~((prices["ticker"] == "AAPL") & prices["date"].isin(removed))]
    clean = validate_prices(partial)
    assert not clean.loc[clean["ticker"] == "AAPL", "date"].isin(removed).any()
    assert len(clean) == len(partial)
