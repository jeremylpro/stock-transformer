from __future__ import annotations

import inspect

import pandas as pd
import pytest

from src.config import SplitsConfig
from src.splits import chronological_split, split_dates


def _panel(tickers=("AAPL", "MSFT", "GOOGL"), days=100):
    dates = pd.bdate_range("2024-01-02", periods=days)
    rows = []
    for ticker in tickers:
        for d in dates:
            rows.append({"ticker": ticker, "date": d, "x": 1.0})
    return pd.DataFrame(rows)


def test_splits_are_chronological_disjoint_and_embargoed():
    cfg = SplitsConfig(
        method="chronological", train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
        embargo_days=1,
    )
    panel = _panel()
    train, val, test = chronological_split(panel, cfg)
    train_d = pd.DatetimeIndex(sorted(train["date"].unique()))
    val_d = pd.DatetimeIndex(sorted(val["date"].unique()))
    test_d = pd.DatetimeIndex(sorted(test["date"].unique()))

    assert train_d.max() < val_d.min() < test_d.min()
    assert train_d.intersection(val_d).empty
    assert val_d.intersection(test_d).empty
    assert train_d.intersection(test_d).empty

    _, _, _, info = split_dates(panel["date"].unique(), cfg)
    all_dates = sorted(panel["date"].unique())
    train_end = all_dates.index(info.train_end.to_datetime64())
    val_start = all_dates.index(info.val_start.to_datetime64())
    val_end = all_dates.index(info.val_end.to_datetime64())
    test_start = all_dates.index(info.test_start.to_datetime64())
    assert val_start - train_end - 1 == cfg.embargo_days
    assert test_start - val_end - 1 == cfg.embargo_days


def test_all_tickers_for_a_date_stay_in_the_same_split():
    cfg = SplitsConfig("chronological", 0.6, 0.2, 0.2, 1)
    panel = _panel()
    train, val, test = chronological_split(panel, cfg)
    for split, expected in ((train, 3), (val, 3), (test, 3)):
        counts = split.groupby("date")["ticker"].nunique()
        assert (counts == expected).all()


def test_split_function_exposes_no_shuffle_parameter():
    sig = inspect.signature(chronological_split)
    assert "shuffle" not in sig.parameters
    assert set(sig.parameters) == {"df", "cfg"}


def test_too_few_dates_rejected():
    cfg = SplitsConfig("chronological", 0.6, 0.2, 0.2, 1)
    with pytest.raises(ValueError):
        split_dates(pd.bdate_range("2024-01-02", periods=4), cfg)
