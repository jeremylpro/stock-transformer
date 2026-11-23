from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import BacktestConfig
from src.backtest import run_backtest


def _bars(days=14):
    dates = pd.bdate_range("2024-01-02", periods=days)
    price = 100.0 + np.arange(days)
    return dates, price


def test_backtest_fill_mode_is_next_bar_open():
    cfg = BacktestConfig(
        entry_threshold=0.55, commission_bps=10, slippage_bps=5, fill="next_bar_open"
    )
    assert cfg.trade_on_close is False


def test_backtest_enters_at_next_open_and_charges_costs():
    dates, price = _bars()
    prices = pd.DataFrame(
        {
            "ticker": "AAPL",
            "date": dates,
            "open": price,
            "high": price + 1,
            "low": price - 1,
            "close": price,
            "adj_close": price,
            "volume": 10_000,
        }
    )
    test = prices.iloc[1:13].copy()
    test["y_next_up"] = 1
    p_up = np.array([0.8, 0.8, 0.2, 0.2, 0.8, 0.8, 0.2, 0.2, 0.8, 0.8, 0.2, 0.2])
    cfg = BacktestConfig(0.55, 10, 5, "next_bar_open")
    result = run_backtest(test, p_up, cfg, prices=prices, featureset="combined")
    payload = result.to_dict()
    assert payload["n_trades"] > 0
    first = payload["trades"][0]
    assert pd.Timestamp(first["entry_time"]).normalize() == dates[2]
    # Entry at the open of the next bar.
    assert first["entry_price"] == float(prices.loc[2, "open"])
    assert payload["net"]["cumulative_return"] < payload["gross"]["cumulative_return"]
    assert "buy_and_hold_net" in payload
    assert payload["costs"]["trade_on_close"] is False


def test_backtest_all_cash_when_no_signals():
    dates, price = _bars()
    prices = pd.DataFrame(
        {
            "ticker": "AAPL", "date": dates, "open": price, "high": price + 1,
            "low": price - 1, "close": price, "adj_close": price, "volume": 1000,
        }
    )
    test = prices.iloc[:10].copy()
    test["y_next_up"] = 1
    cfg = BacktestConfig(0.55, 10, 5, "next_bar_open")
    result = run_backtest(test, np.full(10, 0.2), cfg, prices=prices)
    assert result.payload["n_trades"] == 0
    assert result.payload["net"]["cumulative_return"] == 0.0
