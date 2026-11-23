from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.pipeline import run


def _smoke_config(fixtures_dir: Path, tmp_path: Path) -> dict:
    prices = pd.read_parquet(fixtures_dir / "smoke_prices.parquet")
    start = str(pd.to_datetime(prices["date"]).min().date())
    end = str(pd.to_datetime(prices["date"]).max().date())
    return {
        "universe": {
            "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
            "benchmark": "SPY",
        },
        "date_range": {"start": start, "end": end},
        "data": {
            "price_source": "local",
            "news_source": "local",
            "market_tz": "America/New_York",
            "close_time": "16:00",
            "cache_dir": str(tmp_path / "data" / "raw"),
            "price_file": str(fixtures_dir / "smoke_prices.parquet"),
            "news_file": str(fixtures_dir / "smoke_news.parquet"),
        },
        "sentiment": {
            "model": "ProsusAI/finbert",
            "batch_size": 16,
            "max_tokens": 64,
            "include_vader_baseline": True,
            "backend": "vader",
        },
        "features": {
            "return_lags": 5,
            "vol_windows": [5, 20],
            "winsorize_quantiles": [0.01, 0.99],
            "indicators": {
                "sma_windows": [10, 50],
                "ema_window": 20,
                "rsi_window": 14,
                "macd": {"fast": 12, "slow": 26, "signal": 9},
                "atr_window": 14,
                "bb_window": 20,
                "obv_window": 20,
            },
            "sentiment": {"include_lag1": True, "min_headlines_per_day": 1},
        },
        "splits": {
            "method": "chronological",
            "train_ratio": 0.6,
            "val_ratio": 0.2,
            "test_ratio": 0.2,
            "embargo_days": 1,
        },
        "model": {
            "xgb": {
                "n_estimators": 20,
                "max_depth": 3,
                "learning_rate": 0.1,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "early_stopping_rounds": 5,
            },
            "rf": {"n_estimators": 20, "max_depth": 4, "min_samples_leaf": 3},
            "random_seed": 42,
        },
        "backtest": {
            "entry_threshold": 0.5,
            "commission_bps": 10,
            "slippage_bps": 5,
            "fill": "next_bar_open",
        },
        "reporting": {"output_dir": "reports"},
    }


def test_pipeline_smoke_end_to_end(monkeypatch, tmp_path, fixtures_dir):
    cfg = _smoke_config(fixtures_dir, tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    run(str(config_path), stage="all", force=True)

    required = [
        "data/raw/prices.parquet",
        "data/raw/news.parquet",
        "data/processed/sentiment_daily.parquet",
        "data/processed/features.parquet",
        "data/processed/dataset.parquet",
        "artifacts/predictions_test.parquet",
        "artifacts/imputation_combined.json",
        "artifacts/imputation_tech_only.json",
        "artifacts/imputation_sent_only.json",
        "reports/splits.json",
        "reports/metrics.json",
        "reports/summary.md",
        "reports/backtest_combined.json",
        "reports/backtest_tech_only.json",
        "reports/backtest_sent_only.json",
        "reports/equity_combined.png",
        "reports/feature_importance_combined_xgb.png",
        "reports/run_manifest.json",
    ]
    for rel in required:
        assert (tmp_path / rel).exists(), f"missing artifact: {rel}"

    model_files = sorted(
        p for p in (tmp_path / "artifacts" / "models").glob("*")
        if p.suffix in {".json", ".joblib"} and not p.name.endswith(".manifest.json")
    )
    assert len(model_files) == 6

    import json

    metrics = json.loads((tmp_path / "reports" / "metrics.json").read_text())
    expected_models = {
        "combined_xgb", "combined_rf", "tech_only_xgb", "tech_only_rf",
        "sent_only_xgb", "sent_only_rf",
    }
    assert set(metrics["models"]) == expected_models
    assert set(metrics["baselines"]) == {"majority", "random"}

    # All six models must be scored on exactly the same held-out rows.
    n_samples = {m["test"]["n_samples"] for m in metrics["models"].values()}
    assert n_samples == {metrics["dataset"]["n_test"]}
    assert metrics["dataset"]["n_test"] > 0

    summary = (tmp_path / "reports" / "summary.md").read_text()
    assert "combined model" in summary
    assert "technical-only" in summary and "sentiment-only" in summary
    assert "OFFLINE / DEMO DATA WARNING" in summary

    daily = pd.read_parquet(tmp_path / "data" / "processed" / "sentiment_daily.parquet")
    no_news = daily[daily["sent_count"] == 0]
    assert not no_news.empty
    assert no_news["sent_mean"].isna().all()
    assert daily["date"].notna().all()

    backtest = json.loads((tmp_path / "reports" / "backtest_combined.json").read_text())
    assert backtest["costs"]["trade_on_close"] is False
    assert backtest["period"]["n_test_rows"] == metrics["dataset"]["n_test"]
    assert "net" in backtest and "gross" in backtest and "buy_and_hold_net" in backtest
