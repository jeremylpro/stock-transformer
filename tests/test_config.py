from __future__ import annotations

import copy

import pytest
import yaml

from src.config import ConfigError, config_from_dict, load_config


def _base() -> dict:
    return {
        "universe": {"tickers": ["AAPL", "MSFT", "GOOGL"], "benchmark": "SPY"},
        "date_range": {"start": "2021-01-01", "end": "2025-06-30"},
        "data": {"price_source": "yfinance", "news_source": "finnhub"},
        "splits": {"train_ratio": 0.6, "val_ratio": 0.2, "test_ratio": 0.2},
        "backtest": {"fill": "next_bar_open"},
    }


def test_load_main_config():
    cfg = load_config("config.yaml")
    assert 3 <= len(cfg.universe.tickers) <= 5
    assert cfg.splits.method == "chronological"
    assert cfg.backtest.trade_on_close is False


def test_split_ratios_must_sum_to_one():
    raw = _base()
    raw["splits"]["test_ratio"] = 0.3
    with pytest.raises(ConfigError, match="sum to 1.0"):
        config_from_dict(raw)


def test_more_than_five_tickers_rejected():
    raw = _base()
    raw["universe"]["tickers"] = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"]
    with pytest.raises(ConfigError, match="3-5"):
        config_from_dict(raw)


def test_same_bar_fill_rejected():
    raw = _base()
    raw["backtest"]["fill"] = "same_bar_close"
    with pytest.raises(ConfigError, match="next_bar_open"):
        config_from_dict(raw)


def test_local_source_requires_file():
    raw = _base()
    raw["data"]["price_source"] = "local"
    with pytest.raises(ConfigError, match="price_file"):
        config_from_dict(raw)


def test_unknown_sentiment_backend_rejected():
    raw = _base()
    raw["sentiment"] = {"backend": "magic"}
    with pytest.raises(ConfigError, match="backend"):
        config_from_dict(raw)
