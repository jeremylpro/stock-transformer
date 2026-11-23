"""Typed configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when config.yaml is missing required keys or has invalid values."""


def _as_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{where} must be a mapping, got {type(value).__name__}")
    return value


def _get(mapping: Mapping[str, Any], key: str, default: Any = None, where: str = "config") -> Any:
    if key not in mapping:
        if default is _REQUIRED:
            raise ConfigError(f"missing required key: {where}.{key}")
        return default
    return mapping[key]


_REQUIRED = object()


def _int(value: Any, where: str, *, minimum: int | None = None) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{where} must be an integer, got {value!r}") from exc
    if minimum is not None and out < minimum:
        raise ConfigError(f"{where} must be >= {minimum}, got {out}")
    return out


def _float(value: Any, where: str, *, minimum: float | None = None,
           maximum: float | None = None) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{where} must be a number, got {value!r}") from exc
    if minimum is not None and out < minimum:
        raise ConfigError(f"{where} must be >= {minimum}, got {out}")
    if maximum is not None and out > maximum:
        raise ConfigError(f"{where} must be <= {maximum}, got {out}")
    return out


@dataclass(frozen=True)
class UniverseConfig:
    tickers: tuple[str, ...]
    benchmark: str


@dataclass(frozen=True)
class DateRangeConfig:
    start: str
    end: str


@dataclass(frozen=True)
class DataConfig:
    price_source: str
    news_source: str
    market_tz: str
    close_time: str
    cache_dir: str
    price_file: str | None
    news_file: str | None


@dataclass(frozen=True)
class SentimentConfig:
    model: str
    batch_size: int
    max_tokens: int
    include_vader_baseline: bool
    backend: str


@dataclass(frozen=True)
class IndicatorsConfig:
    sma_windows: tuple[int, ...]
    ema_window: int
    rsi_window: int
    macd_fast: int
    macd_slow: int
    macd_signal: int
    atr_window: int
    bb_window: int
    obv_window: int


@dataclass(frozen=True)
class FeatureSentimentConfig:
    include_lag1: bool
    min_headlines_per_day: int


@dataclass(frozen=True)
class FeaturesConfig:
    return_lags: int
    vol_windows: tuple[int, ...]
    winsorize_quantiles: tuple[float, float]
    indicators: IndicatorsConfig
    sentiment: FeatureSentimentConfig


@dataclass(frozen=True)
class SplitsConfig:
    method: str
    train_ratio: float
    val_ratio: float
    test_ratio: float
    embargo_days: int


@dataclass(frozen=True)
class XGBConfig:
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    early_stopping_rounds: int


@dataclass(frozen=True)
class RFConfig:
    n_estimators: int
    max_depth: int
    min_samples_leaf: int


@dataclass(frozen=True)
class ModelConfig:
    xgb: XGBConfig
    rf: RFConfig
    random_seed: int


@dataclass(frozen=True)
class BacktestConfig:
    entry_threshold: float
    commission_bps: float
    slippage_bps: float
    fill: str

    @property
    def round_trip_cost_rate(self) -> float:
        return (self.commission_bps + self.slippage_bps) / 10_000.0

    @property
    def trade_on_close(self) -> bool:
        """backtesting.py fills next bar open when this is False (G6)."""
        return not (self.fill == "next_bar_open")


@dataclass(frozen=True)
class ReportingConfig:
    output_dir: str


@dataclass(frozen=True)
class Config:
    universe: UniverseConfig
    date_range: DateRangeConfig
    data: DataConfig
    sentiment: SentimentConfig
    features: FeaturesConfig
    splits: SplitsConfig
    model: ModelConfig
    backtest: BacktestConfig
    reporting: ReportingConfig
    source_path: str = field(default="config.yaml")

    @property
    def all_tickers(self) -> tuple[str, ...]:
        """Universe plus benchmark (benchmark is used for calendar diagnostics)."""
        return tuple(dict.fromkeys((*self.universe.tickers, self.universe.benchmark)))


def _load_indicators(raw: Mapping[str, Any]) -> IndicatorsConfig:
    macd = _as_mapping(raw.get("macd", {}), "features.indicators.macd")
    sma = tuple(_int(x, "features.indicators.sma_windows", minimum=2)
                for x in _get(raw, "sma_windows", [10, 50], "features.indicators"))
    if not sma:
        raise ConfigError("features.indicators.sma_windows must not be empty")
    return IndicatorsConfig(
        sma_windows=sma,
        ema_window=_int(_get(raw, "ema_window", 20, "features.indicators"),
                        "features.indicators.ema_window", minimum=2),
        rsi_window=_int(_get(raw, "rsi_window", 14, "features.indicators"),
                        "features.indicators.rsi_window", minimum=2),
        macd_fast=_int(_get(macd, "fast", 12, "features.indicators.macd"),
                       "features.indicators.macd.fast", minimum=2),
        macd_slow=_int(_get(macd, "slow", 26, "features.indicators.macd"),
                       "features.indicators.macd.slow", minimum=3),
        macd_signal=_int(_get(macd, "signal", 9, "features.indicators.macd"),
                         "features.indicators.macd.signal", minimum=2),
        atr_window=_int(_get(raw, "atr_window", 14, "features.indicators"),
                        "features.indicators.atr_window", minimum=2),
        bb_window=_int(_get(raw, "bb_window", 20, "features.indicators"),
                       "features.indicators.bb_window", minimum=2),
        obv_window=_int(_get(raw, "obv_window", 20, "features.indicators"),
                        "features.indicators.obv_window", minimum=2),
    )


def config_from_dict(raw: Mapping[str, Any], *, source_path: str = "config.yaml") -> Config:
    raw = _as_mapping(raw, "config")

    universe_raw = _as_mapping(_get(raw, "universe", _REQUIRED, "config"), "universe")
    tickers_raw = _get(universe_raw, "tickers", _REQUIRED, "universe")
    if not isinstance(tickers_raw, (list, tuple)) or not tickers_raw:
        raise ConfigError("universe.tickers must be a non-empty list")
    tickers = tuple(dict.fromkeys(str(t).strip().upper() for t in tickers_raw if str(t).strip()))
    if not tickers:
        raise ConfigError("universe.tickers must contain at least one ticker")
    if len(tickers) < 3 or len(tickers) > 5:
        raise ConfigError(
            "universe.tickers must contain 3-5 large-cap tickers"
        )
    benchmark = str(_get(universe_raw, "benchmark", "SPY", "universe")).strip().upper()

    dr_raw = _as_mapping(_get(raw, "date_range", _REQUIRED, "config"), "date_range")
    start = str(_get(dr_raw, "start", _REQUIRED, "date_range"))
    end = str(_get(dr_raw, "end", _REQUIRED, "date_range"))
    if start >= end:
        raise ConfigError(f"date_range.start ({start}) must be before date_range.end ({end})")

    data_raw = _as_mapping(_get(raw, "data", _REQUIRED, "config"), "data")
    price_source = str(_get(data_raw, "price_source", "yfinance", "data")).lower()
    news_source = str(_get(data_raw, "news_source", "finnhub", "data")).lower()
    if price_source not in {"yfinance", "alphavantage", "local"}:
        raise ConfigError("data.price_source must be one of yfinance|alphavantage|local")
    if news_source not in {"finnhub", "local"}:
        raise ConfigError("data.news_source must be one of finnhub|local")
    price_file = _get(data_raw, "price_file", None, "data")
    news_file = _get(data_raw, "news_file", None, "data")
    if price_source == "local" and not price_file:
        raise ConfigError("data.price_file is required when data.price_source == 'local'")
    if news_source == "local" and not news_file:
        raise ConfigError("data.news_file is required when data.news_source == 'local'")

    sentiment_raw = _as_mapping(_get(raw, "sentiment", {}), "sentiment")
    backend = str(_get(sentiment_raw, "backend", "auto", "sentiment")).lower()
    if backend not in {"auto", "finbert", "vader"}:
        raise ConfigError("sentiment.backend must be one of auto|finbert|vader")

    features_raw = _as_mapping(_get(raw, "features", {}), "features")
    q = _get(features_raw, "winsorize_quantiles", [0.01, 0.99], "features")
    if not isinstance(q, (list, tuple)) or len(q) != 2:
        raise ConfigError("features.winsorize_quantiles must be [low, high]")
    q_low = _float(q[0], "features.winsorize_quantiles[0]", minimum=0.0, maximum=1.0)
    q_high = _float(q[1], "features.winsorize_quantiles[1]", minimum=0.0, maximum=1.0)
    if q_low >= q_high:
        raise ConfigError("features.winsorize_quantiles[0] must be < [1]")

    sent_feat_raw = _as_mapping(_get(features_raw, "sentiment", {}), "features.sentiment")
    features = FeaturesConfig(
        return_lags=_int(_get(features_raw, "return_lags", 5, "features"),
                         "features.return_lags", minimum=1),
        vol_windows=tuple(_int(x, "features.vol_windows", minimum=2)
                          for x in _get(features_raw, "vol_windows", [5, 20], "features")),
        winsorize_quantiles=(q_low, q_high),
        indicators=_load_indicators(
            _as_mapping(_get(features_raw, "indicators", {}), "features.indicators")
        ),
        sentiment=FeatureSentimentConfig(
            include_lag1=bool(_get(sent_feat_raw, "include_lag1", True, "features.sentiment")),
            min_headlines_per_day=_int(
                _get(sent_feat_raw, "min_headlines_per_day", 1, "features.sentiment"),
                "features.sentiment.min_headlines_per_day", minimum=0),
        ),
    )
    if not features.vol_windows:
        raise ConfigError("features.vol_windows must not be empty")

    splits_raw = _as_mapping(_get(raw, "splits", {}), "splits")
    train_ratio = _float(_get(splits_raw, "train_ratio", 0.6, "splits"),
                         "splits.train_ratio", minimum=0.0, maximum=1.0)
    val_ratio = _float(_get(splits_raw, "val_ratio", 0.2, "splits"),
                       "splits.val_ratio", minimum=0.0, maximum=1.0)
    test_ratio = _float(_get(splits_raw, "test_ratio", 0.2, "splits"),
                        "splits.test_ratio", minimum=0.0, maximum=1.0)
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ConfigError("splits train/val/test ratios must sum to 1.0")
    method = str(_get(splits_raw, "method", "chronological", "splits"))
    if method != "chronological":
        raise ConfigError("splits.method must be 'chronological' (shuffling is forbidden)")
    splits = SplitsConfig(
        method=method,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        embargo_days=_int(_get(splits_raw, "embargo_days", 1, "splits"),
                          "splits.embargo_days", minimum=0),
    )

    model_raw = _as_mapping(_get(raw, "model", {}), "model")
    xgb_raw = _as_mapping(_get(model_raw, "xgb", {}), "model.xgb")
    rf_raw = _as_mapping(_get(model_raw, "rf", {}), "model.rf")
    model = ModelConfig(
        xgb=XGBConfig(
            n_estimators=_int(_get(xgb_raw, "n_estimators", 500, "model.xgb"),
                              "model.xgb.n_estimators", minimum=1),
            max_depth=_int(_get(xgb_raw, "max_depth", 4, "model.xgb"),
                           "model.xgb.max_depth", minimum=1),
            learning_rate=_float(_get(xgb_raw, "learning_rate", 0.03, "model.xgb"),
                                 "model.xgb.learning_rate", minimum=0.0),
            subsample=_float(_get(xgb_raw, "subsample", 0.8, "model.xgb"),
                             "model.xgb.subsample", minimum=0.0, maximum=1.0),
            colsample_bytree=_float(_get(xgb_raw, "colsample_bytree", 0.8, "model.xgb"),
                                    "model.xgb.colsample_bytree", minimum=0.0, maximum=1.0),
            early_stopping_rounds=_int(
                _get(xgb_raw, "early_stopping_rounds", 50, "model.xgb"),
                "model.xgb.early_stopping_rounds", minimum=1),
        ),
        rf=RFConfig(
            n_estimators=_int(_get(rf_raw, "n_estimators", 500, "model.rf"),
                              "model.rf.n_estimators", minimum=1),
            max_depth=_int(_get(rf_raw, "max_depth", 6, "model.rf"),
                           "model.rf.max_depth", minimum=1),
            min_samples_leaf=_int(_get(rf_raw, "min_samples_leaf", 20, "model.rf"),
                                  "model.rf.min_samples_leaf", minimum=1),
        ),
        random_seed=_int(_get(model_raw, "random_seed", 42, "model"),
                         "model.random_seed", minimum=0),
    )

    bt_raw = _as_mapping(_get(raw, "backtest", {}), "backtest")
    fill = str(_get(bt_raw, "fill", "next_bar_open", "backtest"))
    if fill != "next_bar_open":
        raise ConfigError("backtest.fill must be 'next_bar_open' (G6)")
    backtest = BacktestConfig(
        entry_threshold=_float(_get(bt_raw, "entry_threshold", 0.55, "backtest"),
                               "backtest.entry_threshold", minimum=0.0, maximum=1.0),
        commission_bps=_float(_get(bt_raw, "commission_bps", 10, "backtest"),
                              "backtest.commission_bps", minimum=0.0),
        slippage_bps=_float(_get(bt_raw, "slippage_bps", 5, "backtest"),
                            "backtest.slippage_bps", minimum=0.0),
        fill=fill,
    )

    reporting_raw = _as_mapping(_get(raw, "reporting", {}), "reporting")
    reporting = ReportingConfig(
        output_dir=str(_get(reporting_raw, "output_dir", "reports", "reporting"))
    )

    return Config(
        universe=UniverseConfig(tickers=tickers, benchmark=benchmark),
        date_range=DateRangeConfig(start=start, end=end),
        data=DataConfig(
            price_source=price_source,
            news_source=news_source,
            market_tz=str(_get(data_raw, "market_tz", "America/New_York", "data")),
            close_time=str(_get(data_raw, "close_time", "16:00", "data")),
            cache_dir=str(_get(data_raw, "cache_dir", "data/raw", "data")),
            price_file=str(price_file) if price_file else None,
            news_file=str(news_file) if news_file else None,
        ),
        sentiment=SentimentConfig(
            model=str(_get(sentiment_raw, "model", "ProsusAI/finbert", "sentiment")),
            batch_size=_int(_get(sentiment_raw, "batch_size", 32, "sentiment"),
                            "sentiment.batch_size", minimum=1),
            max_tokens=_int(_get(sentiment_raw, "max_tokens", 64, "sentiment"),
                            "sentiment.max_tokens", minimum=8),
            include_vader_baseline=bool(
                _get(sentiment_raw, "include_vader_baseline", True, "sentiment")),
            backend=backend,
        ),
        features=features,
        splits=splits,
        model=model,
        backtest=backtest,
        reporting=reporting,
        source_path=source_path,
    )


def load_config(path: str | Path = "config.yaml") -> Config:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    with open(p, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return config_from_dict(raw, source_path=str(p))
