from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config
from src.features import FORBIDDEN_PREFIXES, build_features


def _zero_sentiment(prices: pd.DataFrame) -> pd.DataFrame:
    keys = prices[["ticker", "date"]].drop_duplicates().sort_values(["ticker", "date"])
    return pd.DataFrame(
        {
            "ticker": keys["ticker"].to_numpy(),
            "date": keys["date"].to_numpy(),
            "sent_mean": np.nan,
            "sent_std": np.nan,
            "sent_count": 0,
            "sent_lag1": np.nan,
            "sent_vader_mean": np.nan,
        }
    )


def test_no_target_in_features(prices, zero_sentiment):
    cfg = load_config("config.yaml")
    feats = build_features(prices, zero_sentiment, cfg.features)
    forbidden = [c for c in feats.columns if c.startswith(FORBIDDEN_PREFIXES)]
    assert forbidden == []
    assert "log_ret_next" not in feats.columns
    assert "y_next_up" not in feats.columns


def test_expected_feature_columns(prices, zero_sentiment):
    cfg = load_config("config.yaml")
    feats = build_features(prices, zero_sentiment, cfg.features)
    expected = {
        "ret_lag_1", "ret_lag_2", "ret_lag_3", "ret_lag_4", "ret_lag_5",
        "ret_cum_5", "vol_5", "vol_20", "ret_z_20",
        "ta_sma_ratio_10", "ta_sma_ratio_50", "ta_ema_ratio_20", "ta_rsi_14",
        "ta_macd_line", "ta_macd_signal", "ta_macd_hist", "ta_atr_norm_14",
        "ta_bb_pct_b_20", "ta_obv_z_20", "ta_vol_z_20",
        "sent_mean", "sent_std", "sent_count", "sent_lag1", "sent_vader_mean",
    }
    assert expected.issubset(set(feats.columns))


def test_features_do_not_change_when_future_prices_change(fixtures_dir):
    prices = pd.read_csv(fixtures_dir / "leakage_prices.csv", parse_dates=["date"])
    cfg = load_config("config.yaml")
    full = build_features(prices, _zero_sentiment(prices), cfg.features)

    cutoff = pd.Timestamp("2024-04-01")
    truncated_prices = prices[prices["date"] < cutoff].copy()
    truncated = build_features(
        truncated_prices, _zero_sentiment(truncated_prices), cfg.features
    )

    feature_cols = [c for c in full.columns if c not in ("ticker", "date")]
    left = full[full["date"] < cutoff].sort_values(["ticker", "date"]).reset_index(drop=True)
    right = truncated.sort_values(["ticker", "date"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        left[["ticker", "date"] + feature_cols],
        right[["ticker", "date"] + feature_cols],
        check_dtype=False,
    )


def test_sentiment_count_is_log1p_and_no_news_stays_nan(prices, zero_sentiment):
    cfg = load_config("config.yaml")
    feats = build_features(prices, zero_sentiment, cfg.features)
    assert (feats["sent_count"] == 0).all()       # log1p(0) == 0
    assert feats["sent_mean"].isna().all()
