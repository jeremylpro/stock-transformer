"""Human-readable report generation (reports/summary.md) and plots."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .utils import get_logger, read_json

LOG = get_logger("reporting")

MODEL_ORDER = [
    "combined_xgb", "combined_rf",
    "tech_only_xgb", "tech_only_rf",
    "sent_only_xgb", "sent_only_rf",
]
FEATURE_SETS = ["combined", "tech_only", "sent_only"]


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if value != value:  # NaN
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


def _metrics_table(metrics: dict[str, Any]) -> str:
    lines = [
        "| Model | Test AUC | Accuracy | Brier | Log loss | N | Positive rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in MODEL_ORDER:
        row = metrics.get("models", {}).get(name)
        if not row:
            continue
        test = row.get("test", {})
        lines.append(
            f"| {name} | {_fmt(test.get('auc'))} | {_fmt(test.get('accuracy'))} | "
            f"{_fmt(test.get('brier'))} | {_fmt(test.get('log_loss'))} | "
            f"{test.get('n_samples', 'n/a')} | {_fmt(test.get('positive_rate'))} |"
        )
    for name, label in (("majority", "majority-class"), ("random", "random (seeded draw)")):
        row = metrics.get("baselines", {}).get(name)
        if not row:
            continue
        lines.append(
            f"| {label} | {_fmt(row.get('auc'))} | {_fmt(row.get('accuracy'))} | "
            f"{_fmt(row.get('brier'))} | {_fmt(row.get('log_loss'))} | "
            f"{row.get('n_samples', 'n/a')} | {_fmt(row.get('positive_rate'))} |"
        )
    return "\n".join(lines)


def _importance_table(metrics: dict[str, Any], model: str = "combined_xgb",
                      top: int = 15) -> str:
    rows = metrics.get("models", {}).get(model, {}).get("feature_importance", [])[:top]
    if not rows:
        return "_Feature importance unavailable._"
    lines = ["| Rank | Feature | Gain importance (normalized) |", "|---:|---|---:|"]
    for i, row in enumerate(rows, 1):
        lines.append(
            f"| {i} | {row['feature']} | {_fmt(row.get('importance_normalized'), 4)} |"
        )
    return "\n".join(lines)


def _backtest_table(backtests: dict[str, dict[str, Any]]) -> str:
    lines = [
        "| Feature set | Net cum. return | Gross cum. return | Net Sharpe | "
        "Max drawdown | Hit rate | Long days | Trades | Buy & hold (net) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fs in FEATURE_SETS:
        bt = backtests.get(fs)
        if not bt:
            continue
        net = bt.get("net", {})
        gross = bt.get("gross", {})
        bh = bt.get("buy_and_hold_net", {})
        lines.append(
            f"| {fs} | {_fmt(100 * net.get('cumulative_return', float('nan')), 2)}% | "
            f"{_fmt(100 * gross.get('cumulative_return', float('nan')), 2)}% | "
            f"{_fmt(net.get('annualized_sharpe'))} | "
            f"{_fmt(100 * net.get('max_drawdown', float('nan')), 2)}% | "
            f"{_fmt(bt.get('hit_rate'))} | {bt.get('n_long_days', 'n/a')} | "
            f"{bt.get('n_trades', 'n/a')} | "
            f"{_fmt(100 * bh.get('cumulative_return', float('nan')), 2)}% |"
        )
    return "\n".join(lines) if len(lines) > 2 else "_No backtests available._"


def _comparison_sentence(metrics: dict[str, Any], algo: str = "xgb") -> str:
    models = metrics.get("models", {})
    combined = models.get(f"combined_{algo}", {}).get("test", {}).get("auc")
    tech = models.get(f"tech_only_{algo}", {}).get("test", {}).get("auc")
    sent = models.get(f"sent_only_{algo}", {}).get("test", {}).get("auc")
    if combined is None or tech is None or sent is None:
        return ("The combined-vs-baseline comparison could not be computed because a model "
                "or its test metric is missing.")
    d_tech = combined - tech
    d_sent = combined - sent
    verdict = "outperformed both single-modality baselines" if d_tech > 0 and d_sent > 0 \
        else "did not outperform both single-modality baselines"
    return (
        f"On the identical held-out test split, the combined model ({algo.upper()}) "
        f"{verdict}: its AUC was {combined:.4f}, versus {tech:.4f} for technical-only "
        f"(delta {d_tech:+.4f}) and {sent:.4f} for sentiment-only (delta {d_sent:+.4f})."
    )


def _interpretation(auc: float | None) -> str:
    if auc is None or auc != auc:
        return "Test AUC is undefined, so no signal can be claimed."
    if auc < 0.50:
        return ("Test AUC is below 0.50: the model does no better than chance on this "
                "split; this is reported as a negative result.")
    if auc <= 0.53:
        return ("Test AUC is in [0.50, 0.53]: there is no evidence of exploitable signal; "
                "this is reported as a negative result.")
    if auc <= 0.58:
        return ("Test AUC is in (0.53, 0.58]: a plausibly weak signal. Check stability "
                "across tickers and feature importance before drawing conclusions.")
    if auc <= 0.60:
        return ("Test AUC is in (0.58, 0.60]: weak-to-moderate signal; treat with caution "
                "given the single chronological split.")
    return ("Test AUC is above 0.60: this is a red flag, not a win. Audit every §8 "
            "guardrail (especially news alignment and target leakage) before proceeding.")


def build_summary(
    metrics: dict[str, Any],
    backtests: dict[str, dict[str, Any]],
    *,
    output_path: str | Path,
) -> Path:
    meta = metrics.get("metadata", {})
    splits = metrics.get("splits", {})
    dataset = metrics.get("dataset", {})
    is_demo = meta.get("price_source") == "local" or meta.get("news_source") == "local"

    lines: list[str] = []
    lines.append("# Stock Direction Predictor MVP - Results Summary")
    lines.append("")
    lines.append(f"Generated: {meta.get('generated_utc', 'unknown')}")
    lines.append(f"Universe: {', '.join(meta.get('tickers', [])) or 'unknown'}")
    lines.append(
        f"Price source: `{meta.get('price_source', 'unknown')}` | "
        f"News source: `{meta.get('news_source', 'unknown')}` | "
        f"Sentiment backend: `{meta.get('sentiment_backend', 'unknown')}`"
    )
    date_range = meta.get("date_range", {})
    lines.append(
        f"Config: `{meta.get('config_path', 'unknown')}` | "
        f"Requested range: {date_range.get('start', 'unknown')}.."
        f"{date_range.get('end', 'unknown')}"
    )
    lines.append("")
    if is_demo:
        lines.append("> **OFFLINE / DEMO DATA WARNING.** At least one input comes from a "
                     "local fixture or synthetic demo file. These numbers exercise the "
                     "pipeline and are **not** evidence about markets. Re-run with a "
                     "FINNHUB_API_KEY for real results.")
        lines.append("")
    lines.append("## 1. Chronological splits")
    lines.append("")
    lines.append("| Split | Start | End | Trading dates |")
    lines.append("|---|---|---|---:|")
    for key in ("train", "validation", "test"):
        part = splits.get(key, {})
        n = splits.get(f"n_{'val' if key == 'validation' else key}_dates", "n/a")
        lines.append(f"| {key} | {part.get('start', 'n/a')} | {part.get('end', 'n/a')} | {n} |")
    lines.append("")
    lines.append(
        f"Embargo: {splits.get('embargo_days', 'n/a')} trading day(s) between splits. "
        "Splits are chronological; no shuffling occurs anywhere in the training path."
    )
    lines.append("")
    lines.append(
        f"Dataset: {dataset.get('n_rows', 'n/a')} labeled rows "
        f"({dataset.get('n_train', 'n/a')} train / {dataset.get('n_val', 'n/a')} val / "
        f"{dataset.get('n_test', 'n/a')} test). Rows dropped for missing targets before "
        f"splitting: {dataset.get('target_nan_dropped', 'n/a')}."
    )
    lines.append("")

    lines.append("## 2. Test metrics (all models on the identical held-out rows)")
    lines.append("")
    lines.append(_metrics_table(metrics))
    lines.append("")
    lines.append(
        "The random baseline has an AUC floor of 0.50; a single seeded draw can deviate "
        "above or below it."
    )
    lines.append("")

    lines.append("## 3. Combined vs. single-modality comparison")
    lines.append("")
    lines.append(_comparison_sentence(metrics, "xgb"))
    lines.append("")
    lines.append(_comparison_sentence(metrics, "rf"))
    lines.append("")
    comparisons = metrics.get("comparisons", {})
    if comparisons:
        lines.append("| Comparison (XGB, test AUC) | Delta |")
        lines.append("|---|---:|")
        for label, value in comparisons.items():
            lines.append(f"| {label} | {_fmt(value, 4)} |")
        lines.append("")

    lines.append("## 4. Top 15 feature importances - combined XGB")
    lines.append("")
    lines.append(_importance_table(metrics, "combined_xgb", 15))
    lines.append("")

    lines.append("## 5. Out-of-sample long/cash backtest (test period only)")
    lines.append("")
    lines.append(_backtest_table(backtests))
    lines.append("")
    lines.append(
        "Signals are generated at the close of day T and filled at the open of T+1. "
        "Net results charge commission and slippage on every position change. "
        "The backtest is an evaluation aid, not a tradeable strategy. "
        "Any position still open at the end of the test window is marked to market "
        "at the final close."
    )
    if backtests and all(
        int(bt.get("n_long_days", 0)) == 0 for bt in backtests.values()
    ):
        lines.append("")
        lines.append(
            "No model produced a probability at or above the entry threshold, so the "
            "strategy stayed in cash for the entire test period."
        )
    lines.append("")

    combined_auc = metrics.get("models", {}).get("combined_xgb", {}).get("test", {}).get("auc")
    lines.append("## 6. Interpretation")
    lines.append("")
    lines.append(_interpretation(combined_auc))
    lines.append("")

    lines.append("## 7. Leakage guardrails")
    lines.append("")
    lines.append("| ID | Guardrail | Status |")
    lines.append("|---|---|---|")
    lines.append("| G1 | 16:00 ET news cutoff owned by `alignment.eligible_headlines_for_day` | "
                 "Enforced + unit tested |")
    lines.append("| G2 | No target columns in the feature matrix | Enforced + unit tested |")
    lines.append("| G3 | Chronological splits, no shuffle API | Enforced + unit tested |")
    lines.append("| G4 | Winsorize/impute fitted on train only | Enforced + unit tested |")
    lines.append("| G5 | Returns use `adj_close` | Enforced + unit tested |")
    lines.append("| G6 | Next-bar-open backtest fills (`trade_on_close=False`) | "
                 "Enforced + config tested |")
    lines.append("| G7 | Missing-target rows dropped before splitting (count logged) | Enforced |")
    lines.append("")

    lines.append("## 8. Known limitations")
    lines.append("")
    for limitation in (
        "Small universe of current large caps; survivorship and selection bias.",
        "Headline-only sentiment discards article bodies, entity disambiguation and source "
        "credibility.",
        "Finnhub's free tier has a limited historical window and coverage gaps; "
        "`sent_count` measures availability, not quality.",
        "A single chronological split gives high-variance AUC estimates; bootstrapped "
        "confidence intervals are post-MVP work.",
        "No market-regime features (VIX, rates, sector returns) and no cross-sectional "
        "normalization across tickers.",
        "The long/cash equal-weight backtest ignores market impact, borrow costs, taxes "
        "and dividend reinvestment details; it is not a tradeable strategy.",
    ):
        lines.append(f"- {limitation}")
    lines.append("")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def save_importance_plot(metrics: dict[str, Any], path: str | Path,
                         model: str = "combined_xgb", top: int = 15) -> Path | None:
    rows = metrics.get("models", {}).get(model, {}).get("feature_importance", [])[:top]
    if not rows:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["feature"] for r in reversed(rows)]
    values = [r.get("importance_normalized", 0.0) for r in reversed(rows)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels, values, color="#2c6e9b")
    ax.set_xlabel("Normalized gain importance")
    ax.set_title(f"Top {len(labels)} features - {model}")
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def write_report(output_dir: str | Path) -> Path:
    out = Path(output_dir)
    metrics = read_json(out / "metrics.json")
    backtests = {}
    for fs in FEATURE_SETS:
        p = out / f"backtest_{fs}.json"
        if p.exists():
            backtests[fs] = read_json(p)
    path = build_summary(metrics, backtests, output_path=out / "summary.md")
    save_importance_plot(metrics, out / "feature_importance_combined_xgb.png")
    LOG.info("Wrote %s", path)
    return path


def _main(argv: Iterable[str] | None = None) -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Build reports/summary.md")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args(argv)
    write_report(args.output_dir)


if __name__ == "__main__":  # pragma: no cover
    _main()
