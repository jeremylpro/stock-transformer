"""Model training, prediction and fitted preprocessing (G4)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import numpy as np
import pandas as pd

from .config import ModelConfig
from .utils import ensure_parent, get_logger, read_json, write_json

LOG = get_logger("model")

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "combined": ("ret_", "vol_", "ta_", "sent_"),
    "tech_only": ("ret_", "vol_", "ta_"),
    "sent_only": ("sent_",),
}
# VADER is a diagnostic baseline, never a model input.
EXCLUDED_FEATURES = {"sent_vader_mean"}


def feature_columns(df: pd.DataFrame, featureset: str) -> list[str]:
    """Select model columns by prefix, excluding target-like and baseline columns."""
    if featureset not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {featureset!r}")
    prefixes = FEATURE_SETS[featureset]
    cols = [
        c for c in df.columns
        if c.startswith(prefixes) and c not in EXCLUDED_FEATURES
    ]
    if not cols:
        raise ValueError(f"feature set {featureset!r} selected no columns")
    return cols


@dataclass
class FeatureTransformer:
    """Winsorization + median imputation fitted on the training split only."""

    feature_names: list[str]
    lower: dict[str, float] = field(default_factory=dict)
    upper: dict[str, float] = field(default_factory=dict)
    medians: dict[str, float] = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        feature_names: list[str],
        quantiles: tuple[float, float] = (0.01, 0.99),
    ) -> "FeatureTransformer":
        lower: dict[str, float] = {}
        upper: dict[str, float] = {}
        medians: dict[str, float] = {}
        for col in feature_names:
            values = pd.to_numeric(df[col], errors="coerce")
            if values.notna().any():
                lo = float(values.quantile(quantiles[0]))
                hi = float(values.quantile(quantiles[1]))
                median = float(values.median())
            else:
                lo, hi, median = np.nan, np.nan, 0.0
            lower[col] = lo
            upper[col] = hi
            medians[col] = median
        return cls(feature_names=list(feature_names), lower=lower, upper=upper, medians=medians)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.reindex(columns=self.feature_names).copy()
        for col in self.feature_names:
            values = pd.to_numeric(out[col], errors="coerce").astype(float)
            lo, hi = self.lower.get(col, np.nan), self.upper.get(col, np.nan)
            if not np.isnan(lo) and not np.isnan(hi):
                values = values.clip(lower=lo, upper=hi)
            median = self.medians.get(col, 0.0)
            values = values.fillna(median if not np.isnan(median) else 0.0)
            out[col] = values
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "winsorize_lower": self.lower,
            "winsorize_upper": self.upper,
            "medians": self.medians,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureTransformer":
        return cls(
            feature_names=list(payload["feature_names"]),
            lower={k: float(v) for k, v in payload.get("winsorize_lower", {}).items()},
            upper={k: float(v) for k, v in payload.get("winsorize_upper", {}).items()},
            medians={k: float(v) for k, v in payload.get("medians", {}).items()},
        )

    def save(self, path: str | Path) -> Path:
        return write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "FeatureTransformer":
        return cls.from_dict(read_json(path))


@dataclass
class TrainedModel:
    algo: str
    featureset: str
    feature_names: list[str]
    model: Any

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return predict_proba(self.model, X)

    def save(self, directory: str | Path) -> Path:
        p = Path(directory)
        p.mkdir(parents=True, exist_ok=True)
        if self.algo == "xgb":
            path = ensure_parent(p / f"{self.featureset}_{self.algo}.json")
            self.model.save_model(path)
        else:
            path = ensure_parent(p / f"{self.featureset}_{self.algo}.joblib")
            joblib.dump(self.model, path)
        return path


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_val: pd.DataFrame,
    y_val: pd.Series | np.ndarray | None,
    algo: str,
    cfg: ModelConfig,
    *,
    featureset: str = "combined",
) -> TrainedModel:
    """Train XGBoost (early stopping on validation AUC) or RandomForest."""
    algo = str(algo).lower()
    if algo not in {"xgb", "rf"}:
        raise ValueError(f"unsupported algo {algo!r}")
    y_train = np.asarray(y_train).astype(int)
    if len(np.unique(y_train)) < 2:
        raise ValueError("training labels contain a single class; cannot train a classifier")

    feature_names = list(X_train.columns)
    if algo == "xgb":
        from xgboost import XGBClassifier

        xcfg = cfg.xgb
        model = XGBClassifier(
            n_estimators=xcfg.n_estimators,
            max_depth=xcfg.max_depth,
            learning_rate=xcfg.learning_rate,
            subsample=xcfg.subsample,
            colsample_bytree=xcfg.colsample_bytree,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=cfg.random_seed,
            n_jobs=4,
            early_stopping_rounds=xcfg.early_stopping_rounds
            if y_val is not None and len(np.unique(np.asarray(y_val))) > 1 else None,
        )
        if y_val is not None and len(np.unique(np.asarray(y_val))) > 1:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, np.asarray(y_val).astype(int))],
                verbose=False,
            )
        else:
            model.fit(X_train, y_train, verbose=False)
    else:
        from sklearn.ensemble import RandomForestClassifier

        rcfg = cfg.rf
        model = RandomForestClassifier(
            n_estimators=rcfg.n_estimators,
            max_depth=rcfg.max_depth,
            min_samples_leaf=rcfg.min_samples_leaf,
            random_state=cfg.random_seed,
            n_jobs=-1,
            class_weight=None,
        )
        model.fit(X_train, y_train)

    return TrainedModel(
        algo=algo, featureset=featureset, feature_names=feature_names, model=model
    )


def predict_proba(model: Any, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Return P(y=1), independent of the estimator's class ordering."""
    estimator = model.model if isinstance(model, TrainedModel) else model
    if hasattr(estimator, "predict_proba"):
        probs = estimator.predict_proba(X)
    else:  # pragma: no cover - all supported estimators have predict_proba
        raise TypeError("model does not implement predict_proba")
    classes = list(getattr(estimator, "classes_", [0, 1]))
    if 1 in classes:
        return np.asarray(probs)[:, classes.index(1)]
    return np.zeros(len(X), dtype=float)


def load_model(path: str | Path, algo: str, featureset: str,
               feature_names: list[str]) -> TrainedModel:
    if algo == "xgb":
        from xgboost import XGBClassifier

        model = XGBClassifier()
        model.load_model(str(path))
    else:
        model = joblib.load(path)
    return TrainedModel(
        algo=algo, featureset=featureset, feature_names=feature_names, model=model
    )


def _main(argv: Iterable[str] | None = None) -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Train models")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    from .pipeline import run

    run(args.config, stage="train", force=args.force)


if __name__ == "__main__":  # pragma: no cover
    _main()
