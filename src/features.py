"""Feature engineering: lagged log returns, technical indicators, sentiment."""

from __future__ import annotations

import argparse
from typing import Iterable

import numpy as np
import pandas as pd

from .config import FeaturesConfig
from .utils import get_logger

LOG = get_logger("features")

FEATURE_PREFIXES = ("ret_", "vol_", "ta_", "sent_")
FORBIDDEN_PREFIXES = ("y_", "log_ret_next", "log_ret_fwd")


def _add_return_features(g: pd.DataFrame, cfg: FeaturesConfig) -> pd.DataFrame:
    adj_close = g["adj_close"]
    r = np.log(adj_close / adj_close.shift(1))
    out = g.copy()
    for lag in range(1, cfg.return_lags + 1):
        out[f"ret_lag_{lag}"] = r.shift(lag - 1)
    out["ret_cum_5"] = r.rolling(5).sum()
    for window in cfg.vol_windows:
        out[f"vol_{window}"] = r.rolling(window).std()
    if "vol_20" in out.columns:
        denom = out["vol_20"].where(out["vol_20"] > 0)
        out["ret_z_20"] = r / denom
    return out


def _add_technical_features(g: pd.DataFrame, cfg: FeaturesConfig) -> pd.DataFrame:
    from ta.momentum import RSIIndicator
    from ta.trend import EMAIndicator, MACD, SMAIndicator
    from ta.volatility import AverageTrueRange, BollingerBands
    from ta.volume import OnBalanceVolumeIndicator

    ind = cfg.indicators
    out = g.copy()
    close = out["adj_close"]

    # Adjust OHLC by the same factor as adj_close so splits/dividends do not
    # distort intraday-range indicators.
    factor = (out["adj_close"] / out["close"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    adj_high = out["high"] * factor
    adj_low = out["low"] * factor

    for window in ind.sma_windows:
        sma = SMAIndicator(close=close, window=window, fillna=False).sma_indicator()
        out[f"ta_sma_ratio_{window}"] = close / sma.where(sma > 0) - 1.0
    ema = EMAIndicator(close=close, window=ind.ema_window, fillna=False).ema_indicator()
    out["ta_ema_ratio_20"] = close / ema.where(ema > 0) - 1.0

    out["ta_rsi_14"] = RSIIndicator(close=close, window=ind.rsi_window, fillna=False).rsi()

    macd = MACD(
        close=close,
        window_slow=ind.macd_slow,
        window_fast=ind.macd_fast,
        window_sign=ind.macd_signal,
        fillna=False,
    )
    out["ta_macd_line"] = macd.macd()
    out["ta_macd_signal"] = macd.macd_signal()
    out["ta_macd_hist"] = macd.macd_diff()

    atr = AverageTrueRange(
        high=adj_high, low=adj_low, close=close, window=ind.atr_window, fillna=False
    ).average_true_range()
    out["ta_atr_norm_14"] = atr / close.replace(0, np.nan)

    bb = BollingerBands(close=close, window=ind.bb_window, window_dev=2, fillna=False)
    upper = bb.bollinger_hband()
    lower = bb.bollinger_lband()
    width = (upper - lower).replace(0, np.nan)
    out["ta_bb_pct_b_20"] = (close - lower) / width

    obv = OnBalanceVolumeIndicator(close=close, volume=out["volume"], fillna=False)
    obv_values = obv.on_balance_volume()
    obv_mean = obv_values.rolling(ind.obv_window).mean()
    obv_std = obv_values.rolling(ind.obv_window).std().replace(0, np.nan)
    out["ta_obv_z_20"] = (obv_values - obv_mean) / obv_std

    vol_mean = out["volume"].rolling(ind.obv_window).mean()
    vol_std = out["volume"].rolling(ind.obv_window).std().replace(0, np.nan)
    out["ta_vol_z_20"] = (out["volume"] - vol_mean) / vol_std
    return out


def build_features(
    prices: pd.DataFrame,
    sentiment_daily: pd.DataFrame,
    cfg: FeaturesConfig,
) -> pd.DataFrame:
    """Return one row per ``(ticker, date)`` with all §7 features.

    Only shifted/rolling-backward operations are used, so no feature depends on
    information after 16:00 ET of ``date``. Targets are asserted absent (G2).
    """
    required = {"ticker", "date", "open", "high", "low", "close", "adj_close", "volume"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"prices missing required columns: {sorted(missing)}")

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    sent_cols = [
        "ticker", "date", "sent_mean", "sent_std", "sent_count", "sent_lag1",
        "sent_vader_mean",
    ]
    sent = sentiment_daily.copy()
    for col in sent_cols:
        if col not in sent.columns:
            if col in {"ticker", "date", "sent_count"}:
                raise ValueError(f"sentiment_daily missing required column {col!r}")
            sent[col] = np.nan
    sent = sent[sent_cols].copy()
    sent["date"] = pd.to_datetime(sent["date"]).dt.normalize()
    sent["ticker"] = sent["ticker"].astype(str).str.upper()
    if sent.duplicated(subset=["ticker", "date"]).any():
        raise ValueError("sentiment_daily has duplicate (ticker, date) rows")

    frames = []
    for ticker, group in prices.groupby("ticker", sort=True):
        group = group.sort_values("date").reset_index(drop=True)
        if len(group) < 2:
            LOG.warning("Skipping %s: fewer than 2 price rows", ticker)
            continue
        enriched = _add_technical_features(_add_return_features(group, cfg), cfg)
        frames.append(enriched)
    if not frames:
        raise ValueError("no ticker has enough price history to build features")

    features = pd.concat(frames, ignore_index=True)
    features = features.merge(sent, on=["ticker", "date"], how="left", validate="one_to_one")

    # Missing sentiment rows mean "no eligible headlines" for that day.
    features["sent_count"] = features["sent_count"].fillna(0).astype(int)
    features["sent_count"] = np.log1p(features["sent_count"].astype(float))
    if not cfg.sentiment.include_lag1:
        features = features.drop(columns=["sent_lag1"])

    feature_cols = [c for c in features.columns if c.startswith(FEATURE_PREFIXES)]
    if not feature_cols:
        raise ValueError("no feature columns were produced")

    forbidden = [c for c in features.columns if c.startswith(FORBIDDEN_PREFIXES)]
    if forbidden:
        raise AssertionError(f"G2 violated: target-like columns in features: {forbidden}")

    meta = ["ticker", "date"]
    features = features[meta + feature_cols]
    return features.sort_values(["ticker", "date"]).reset_index(drop=True)


def main(argv: Iterable[str] | None = None) -> None:
    from .pipeline import run

    parser = argparse.ArgumentParser(description="Build the feature matrix")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    run(args.config, stage="features", force=args.force)


if __name__ == "__main__":  # pragma: no cover
    main()
