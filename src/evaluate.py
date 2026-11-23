"""Evaluation metrics, comparisons and feature importance."""

from __future__ import annotations

import argparse
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .utils import get_logger

LOG = get_logger("evaluate")


def _safe_auc(y_true: np.ndarray, p_up: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, p_up))


def evaluate(y_true: Iterable[int], p_up: Iterable[float]) -> dict[str, Any]:
    """Return AUC, accuracy, Brier, log loss, sample count and positive rate."""
    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

    y = np.asarray(list(y_true)).astype(int)
    p = np.asarray(list(p_up), dtype=float)
    if len(y) == 0:
        raise ValueError("cannot evaluate zero samples")
    if len(y) != len(p):
        raise ValueError("y_true and p_up differ in length")
    if np.isnan(p).any():
        raise ValueError("p_up contains NaN")

    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    result: dict[str, Any] = {
        "auc": _safe_auc(y, p),
        "accuracy": float(accuracy_score(y, (p >= 0.5).astype(int))),
        "brier": float(brier_score_loss(y, clipped)),
        "log_loss": float(log_loss(y, clipped, labels=[0, 1])),
        "n_samples": int(len(y)),
        "positive_rate": float(y.mean()),
    }
    if len(np.unique(y)) < 2:
        LOG.warning("AUC is undefined because y_true contains a single class")
    return result


def majority_baseline(y_true: Iterable[int]) -> dict[str, Any]:
    """Always predict the training-majority class (here: the test positive rate)."""
    y = np.asarray(list(y_true)).astype(int)
    if len(y) == 0:
        raise ValueError("cannot evaluate zero samples")
    rate = float(y.mean())
    pred = 1 if rate >= 0.5 else 0
    return {
        "auc": 0.5,
        "accuracy": float((y == pred).mean()),
        "brier": float(((y - rate) ** 2).mean()),
        "log_loss": float(-np.mean(
            y * np.log(max(rate, 1e-7)) + (1 - y) * np.log(max(1 - rate, 1e-7))
        )),
        "n_samples": int(len(y)),
        "positive_rate": rate,
        "predicted_class": pred,
    }


def random_baseline(y_true: Iterable[int], *, seed: int = 42) -> dict[str, Any]:
    """Random probabilities; AUC is ~0.5 by construction (sanity floor)."""
    y = np.asarray(list(y_true)).astype(int)
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, 1, size=len(y))
    return evaluate(y, p)


def feature_importance(model: Any, feature_names: list[str]) -> pd.DataFrame:
    """Gain importance for XGBoost, impurity importance for RandomForest."""
    estimator = model.model if hasattr(model, "model") else model
    algo = getattr(model, "algo", None)
    if algo == "xgb" or hasattr(estimator, "get_booster"):
        booster = estimator.get_booster()
        scores = booster.get_score(importance_type="gain")
        values = [float(scores.get(name, 0.0)) for name in feature_names]
    elif hasattr(estimator, "feature_importances_"):
        values = [float(v) for v in estimator.feature_importances_]
    else:
        raise TypeError("model does not expose feature importance")
    out = pd.DataFrame({"feature": feature_names, "importance": values})
    total = out["importance"].sum()
    out["importance_normalized"] = out["importance"] / total if total > 0 else 0.0
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def _main(argv: Iterable[str] | None = None) -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Evaluate saved test predictions")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(argv)
    cfg_path = args.config
    from .config import load_config

    cfg = load_config(cfg_path)
    pred_path = "artifacts/predictions_test.parquet"
    df = pd.read_parquet(pred_path)
    metrics = {}
    for (featureset, algo), group in df.groupby(["featureset", "algo"]):
        metrics[f"{featureset}_{algo}"] = evaluate(group["y_next_up"], group["p_up"])
    from .utils import write_json

    write_json(f"{cfg.reporting.output_dir}/metrics_eval.json", metrics)


if __name__ == "__main__":  # pragma: no cover
    _main()
