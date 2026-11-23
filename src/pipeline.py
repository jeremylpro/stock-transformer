"""End-to-end batch pipeline with per-stage artifacts and caching."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from . import backtest as backtest_mod
from . import data_news, data_prices, evaluate, features, labels, model, reporting, sentiment, splits
from .config import Config, load_config
from .utils import (
    artifact_is_current,
    ensure_parent,
    get_logger,
    read_json,
    save_parquet,
    setup_logging,
    sha256_file,
    utc_now_iso,
    write_json,
    write_manifest,
)

LOG = get_logger("pipeline")

STAGES = ("prices", "news", "sentiment", "features", "train", "backtest")
RAW = Path("data/raw")
PROCESSED = Path("data/processed")
ARTIFACTS = Path("artifacts")
MODELS_DIR = ARTIFACTS / "models"

PRICES_PATH = RAW / "prices.parquet"
BENCHMARK_PATH = RAW / "benchmark.parquet"
NEWS_PATH = RAW / "news.parquet"
SENTIMENT_PATH = PROCESSED / "sentiment_daily.parquet"
SENTIMENT_META_PATH = PROCESSED / "sentiment_meta.json"
NEWS_SCORED_PATH = PROCESSED / "news_scored.parquet"
FEATURES_PATH = PROCESSED / "features.parquet"
LABELS_PATH = PROCESSED / "labels.parquet"
DATASET_PATH = PROCESSED / "dataset.parquet"
PREDICTIONS_PATH = ARTIFACTS / "predictions_test.parquet"


def _config_hash(config_path: str | Path) -> str:
    return sha256_file(config_path)


def _fresh(artifact: Path, cfg_hash: str, inputs: Sequence[str | Path], force: bool) -> bool:
    if force:
        return False
    return artifact_is_current(artifact, config_hash=cfg_hash, inputs=inputs)


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:  # pragma: no cover
        return
    load_dotenv()


def _stage_prices(cfg: Config, cfg_hash: str, force: bool) -> None:
    LOG.info("=== Stage: prices ===")
    inputs = [cfg.source_path] + ([cfg.data.price_file] if cfg.data.price_source == "local" else [])
    if _fresh(PRICES_PATH, cfg_hash, [i for i in inputs if i], force):
        LOG.info("prices artifact is current; skipping (use --force to rebuild)")
        return

    tickers = list(cfg.universe.tickers)
    source = cfg.data.price_source
    api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
    panel = data_prices.fetch_prices(
        tickers, cfg.date_range.start, cfg.date_range.end, cfg.data.cache_dir,
        source=source, price_file=cfg.data.price_file, api_key=api_key,
    )

    benchmark = None
    if source != "local":
        try:
            benchmark = data_prices.fetch_prices(
                [cfg.universe.benchmark], cfg.date_range.start, cfg.date_range.end,
                cfg.data.cache_dir, source=source, price_file=None, api_key=api_key,
            )
        except Exception as exc:  # benchmark is diagnostic only
            LOG.warning("Could not fetch benchmark %s: %s", cfg.universe.benchmark, exc)

    calendar = None
    if benchmark is not None and not benchmark.empty:
        calendar = pd.DatetimeIndex(pd.to_datetime(benchmark["date"]).unique())
        save_parquet(benchmark, BENCHMARK_PATH)
        write_manifest(BENCHMARK_PATH, stage="prices", config_hash=cfg_hash,
                       inputs=inputs, extra={"ticker": cfg.universe.benchmark})

    clean = data_prices.validate_prices(panel, calendar=calendar)
    save_parquet(clean, PRICES_PATH)
    write_manifest(PRICES_PATH, stage="prices", config_hash=cfg_hash, inputs=inputs,
                   extra={"n_rows": len(clean), "tickers": sorted(clean["ticker"].unique())})
    LOG.info("prices: %d rows for %s", len(clean), sorted(clean["ticker"].unique()))


def _stage_news(cfg: Config, cfg_hash: str, force: bool) -> None:
    LOG.info("=== Stage: news ===")
    inputs = [cfg.source_path] + ([cfg.data.news_file] if cfg.data.news_source == "local" else [])
    if _fresh(NEWS_PATH, cfg_hash, [i for i in inputs if i], force):
        LOG.info("news artifact is current; skipping")
        return

    api_key = os.environ.get("FINNHUB_API_KEY", "")
    raw = data_news.fetch_news(
        list(cfg.universe.tickers), cfg.date_range.start, cfg.date_range.end,
        api_key, cfg.data.cache_dir, source=cfg.data.news_source,
        news_file=cfg.data.news_file,
    )
    clean = data_news.validate_news(raw, end=cfg.date_range.end, market_tz=cfg.data.market_tz)
    save_parquet(clean, NEWS_PATH)
    write_manifest(NEWS_PATH, stage="news", config_hash=cfg_hash, inputs=inputs,
                   extra={"n_headlines": len(clean)})
    LOG.info("news: %d unique headlines", len(clean))


def _headline_scores(
    news: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, str]:
    """Score unique headlines, reusing cached scores where the backend matches."""
    unique = news[["headline"]].drop_duplicates().reset_index(drop=True)
    cache = None
    if NEWS_SCORED_PATH.exists():
        try:
            cache = pd.read_parquet(NEWS_SCORED_PATH)
        except Exception as exc:  # pragma: no cover
            LOG.warning("Ignoring unreadable sentiment score cache: %s", exc)

    requested = cfg.sentiment.backend
    backend_hint = requested
    if cache is not None and not cache.empty and requested == "auto":
        backends = set(cache["backend"].dropna().unique())
        if len(backends) == 1:
            cached_backend = next(iter(backends))
            if cached_backend in {"finbert", "vader"}:
                backend_hint = cached_backend

    known = pd.DataFrame()
    if cache is not None and not cache.empty and backend_hint in {"finbert", "vader"}:
        known = cache[cache["backend"] == backend_hint].copy()
        if backend_hint != requested:
            LOG.info("Reusing cached %s sentiment scores", backend_hint)

    missing = unique[~unique["headline"].isin(known["headline"])] if not known.empty else unique
    scored_frames = []
    actual_backend = backend_hint
    if not missing.empty:
        LOG.info("Scoring %d headlines with backend=%s (cached: %d)",
                 len(missing), requested, len(known))
        fresh = sentiment.score_headlines(
            missing["headline"].tolist(),
            model_name=cfg.sentiment.model,
            batch_size=cfg.sentiment.batch_size,
            max_tokens=cfg.sentiment.max_tokens,
            backend=requested,
        )
        actual_backend = fresh.attrs.get("backend", requested)
        fresh = fresh.copy()
        fresh["headline"] = missing["headline"].to_numpy()
        fresh["backend"] = actual_backend
        fresh["model"] = cfg.sentiment.model if actual_backend == "finbert" else "vader"
        scored_frames.append(fresh)

    if not known.empty:
        known = known.copy()
        known["backend"] = backend_hint
        scored_frames.append(known)

    if not scored_frames:
        raise RuntimeError("no sentiment scores were produced")

    scores = pd.concat(scored_frames, ignore_index=True)
    if actual_backend != backend_hint:
        LOG.info("Resolved sentiment backend %s -> %s", backend_hint, actual_backend)
        if not known.empty:
            # The cached rows were produced by a different scorer, so they are
            # not comparable with the fresh batch. Rescore them instead of
            # silently dropping them (which would leave those headlines NaN).
            LOG.info("Rescoring %d cached headlines with %s", len(known), actual_backend)
            rescored = sentiment.score_headlines(
                known["headline"].tolist(),
                model_name=cfg.sentiment.model,
                batch_size=cfg.sentiment.batch_size,
                max_tokens=cfg.sentiment.max_tokens,
                backend=actual_backend,
            )
            rescored = rescored.copy()
            rescored["headline"] = known["headline"].to_numpy()
            rescored["backend"] = actual_backend
            rescored["model"] = (
                cfg.sentiment.model if actual_backend == "finbert" else "vader"
            )
            scores = pd.concat(
                [scores[scores["backend"] == actual_backend], rescored],
                ignore_index=True,
            )
        else:
            scores = scores[scores["backend"] == actual_backend].copy()

    if cfg.sentiment.include_vader_baseline:
        need_vader = scores["vader_score"].isna() if "vader_score" in scores else \
            pd.Series(True, index=scores.index)
        if need_vader.any():
            vader = sentiment.vader_scores(
                scores.loc[need_vader, "headline"].tolist()
            )["sent_score"].to_numpy()
            scores.loc[need_vader, "vader_score"] = vader
    else:
        scores["vader_score"] = np.nan

    score_cols = [
        "headline", "prob_positive", "prob_negative", "prob_neutral",
        "sent_score", "vader_score", "backend", "model",
    ]
    scores = scores.drop_duplicates("headline", keep="last")[score_cols]
    save_parquet(scores, NEWS_SCORED_PATH)
    return scores, actual_backend


def _stage_sentiment(cfg: Config, cfg_hash: str, force: bool) -> None:
    LOG.info("=== Stage: sentiment ===")
    inputs = [cfg.source_path, NEWS_PATH, PRICES_PATH]
    if _fresh(SENTIMENT_PATH, cfg_hash, inputs, force):
        LOG.info("sentiment artifact is current; skipping")
        return

    prices = pd.read_parquet(PRICES_PATH)
    news = pd.read_parquet(NEWS_PATH)
    if news.empty:
        LOG.warning("No news headlines available; sentiment will be NaN everywhere")
        scores = pd.DataFrame(columns=["headline", "sent_score", "vader_score"])
        backend = "none"
    else:
        scores, backend = _headline_scores(news, cfg)
        news = news.merge(
            scores[["headline", "sent_score", "vader_score"]],
            on="headline", how="left", validate="many_to_one",
        )

    daily = sentiment.daily_sentiment(
        news,
        prices,
        market_tz=cfg.data.market_tz,
        close_time=cfg.data.close_time,
        include_vader_baseline=cfg.sentiment.include_vader_baseline,
    )
    save_parquet(daily, SENTIMENT_PATH)
    zero_days = int((daily["sent_count"] == 0).sum())
    min_headlines = cfg.features.sentiment.min_headlines_per_day
    if min_headlines > 1:
        thin_days = int(
            ((daily["sent_count"] > 0) & (daily["sent_count"] < min_headlines)).sum()
        )
        if thin_days:
            LOG.warning(
                "%d ticker-days have headlines below min_headlines_per_day=%d",
                thin_days, min_headlines,
            )
    meta = {
        "backend": backend,
        "n_headlines": int(len(news)),
        "n_days": int(len(daily)),
        "n_zero_news_days": zero_days,
        "mean_sentiment": float(daily["sent_mean"].mean())
        if daily["sent_mean"].notna().any() else None,
    }
    write_json(SENTIMENT_META_PATH, meta)
    write_manifest(SENTIMENT_PATH, stage="sentiment", config_hash=cfg_hash, inputs=inputs,
                   extra=meta)
    LOG.info(
        "sentiment: backend=%s, %d headlines, %.1f%% ticker-days with no news",
        backend, len(news), 100 * zero_days / max(len(daily), 1),
    )


def _stage_features(cfg: Config, cfg_hash: str, force: bool) -> None:
    LOG.info("=== Stage: features ===")
    inputs = [cfg.source_path, PRICES_PATH, SENTIMENT_PATH]
    if _fresh(FEATURES_PATH, cfg_hash, inputs, force):
        LOG.info("features artifact is current; skipping")
        return

    prices = pd.read_parquet(PRICES_PATH)
    daily = pd.read_parquet(SENTIMENT_PATH)
    feats = features.build_features(prices, daily, cfg.features)
    save_parquet(feats, FEATURES_PATH)
    write_manifest(FEATURES_PATH, stage="features", config_hash=cfg_hash, inputs=inputs,
                   extra={"n_rows": len(feats), "n_features": int(
                       sum(c.startswith(features.FEATURE_PREFIXES) for c in feats.columns))})
    LOG.info("features: %d rows, %d columns", len(feats), feats.shape[1])


def _merge_dataset() -> pd.DataFrame:
    feats = pd.read_parquet(FEATURES_PATH)
    prices = pd.read_parquet(PRICES_PATH)
    labs = labels.build_labels(prices)
    price_cols = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]
    dataset = feats.merge(labs, on=["ticker", "date"], how="inner", validate="one_to_one")
    dataset = dataset.merge(
        prices[price_cols], on=["ticker", "date"], how="left", validate="one_to_one"
    )
    if dataset[["open", "high", "low", "close", "adj_close"]].isna().any().any():
        raise ValueError("dataset is missing price bars for some labeled rows")
    dataset["date"] = pd.to_datetime(dataset["date"]).dt.normalize()
    return dataset.sort_values(["ticker", "date"]).reset_index(drop=True)


def _stage_train(cfg: Config, cfg_hash: str, force: bool) -> dict[str, Any]:
    LOG.info("=== Stage: train ===")
    inputs = [cfg.source_path, FEATURES_PATH, PRICES_PATH]
    metrics_path = Path(cfg.reporting.output_dir) / "metrics.json"
    if _fresh(metrics_path, cfg_hash, inputs, force):
        LOG.info("training artifacts are current; skipping")
        return read_json(metrics_path)

    dataset = _merge_dataset()
    save_parquet(dataset, DATASET_PATH)
    n_before = len(dataset)
    dataset = dataset[dataset["log_ret_next"].notna()].copy()
    dropped = n_before - len(dataset)
    LOG.info("G7: dropped %d rows with missing targets before splitting", dropped)

    train_dates, val_dates, test_dates, split_info = splits.split_dates(
        dataset["date"].unique(), cfg.splits
    )
    train = dataset[dataset["date"].isin(train_dates)].copy()
    val = dataset[dataset["date"].isin(val_dates)].copy()
    test = dataset[dataset["date"].isin(test_dates)].copy()
    if train.empty or val.empty or test.empty:
        raise ValueError("one of the chronological splits is empty")

    reports_dir = Path(cfg.reporting.output_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    write_json(reports_dir / "splits.json", split_info.to_dict())

    predictions: list[pd.DataFrame] = []
    model_metrics: dict[str, Any] = {}
    trained: dict[str, model.TrainedModel] = {}
    transformers: dict[str, model.FeatureTransformer] = {}

    for fs_name in model.FEATURE_SETS:
        cols = model.feature_columns(train, fs_name)
        transformer = model.FeatureTransformer.fit(
            train, cols, cfg.features.winsorize_quantiles
        )
        transformers[fs_name] = transformer
        imputation_path = transformer.save(ARTIFACTS / f"imputation_{fs_name}.json")
        write_manifest(imputation_path, stage="train", config_hash=cfg_hash,
                       inputs=inputs, extra={"featureset": fs_name})
        X_train = transformer.transform(train)
        X_val = transformer.transform(val)
        X_test = transformer.transform(test)

        for algo in ("xgb", "rf"):
            key = f"{fs_name}_{algo}"
            LOG.info("Training %s (%d features, %d train rows)", key, len(cols), len(train))
            trained_model = model.train_model(
                X_train, train["y_next_up"], X_val, val["y_next_up"],
                algo, cfg.model, featureset=fs_name,
            )
            p_val = trained_model.predict_proba(X_val)
            p_test = trained_model.predict_proba(X_test)
            val_metrics = evaluate.evaluate(val["y_next_up"], p_val)
            test_metrics = evaluate.evaluate(test["y_next_up"], p_test)
            importance = evaluate.feature_importance(trained_model, cols)
            model_path = trained_model.save(MODELS_DIR)
            write_manifest(model_path, stage="train", config_hash=cfg_hash,
                           inputs=inputs,
                           extra={"featureset": fs_name, "algo": algo})
            trained[key] = trained_model

            model_metrics[key] = {
                "featureset": fs_name,
                "algo": algo,
                "features": cols,
                "validation": val_metrics,
                "test": test_metrics,
                "feature_importance": importance.head(30).to_dict("records"),
            }
            predictions.append(
                pd.DataFrame(
                    {
                        "ticker": test["ticker"].to_numpy(),
                        "date": test["date"].to_numpy(),
                        "y_next_up": test["y_next_up"].to_numpy(),
                        "log_ret_next": test["log_ret_next"].to_numpy(),
                        "p_up": p_test,
                        "featureset": fs_name,
                        "algo": algo,
                    }
                )
            )

    predictions_df = pd.concat(predictions, ignore_index=True)
    save_parquet(predictions_df, PREDICTIONS_PATH)
    write_manifest(PREDICTIONS_PATH, stage="train", config_hash=cfg_hash,
                   inputs=inputs, extra={"n_rows": len(predictions_df)})

    y_test = test["y_next_up"].to_numpy()
    baselines = {
        "majority": evaluate.majority_baseline(y_test),
        "random": evaluate.random_baseline(y_test, seed=cfg.model.random_seed),
    }
    comparisons: dict[str, float] = {}
    for algo in ("xgb", "rf"):
        combined = model_metrics[f"combined_{algo}"]["test"]["auc"]
        for other in ("tech_only", "sent_only"):
            value = model_metrics[f"{other}_{algo}"]["test"]["auc"]
            if combined == combined and value == value:
                comparisons[f"combined_minus_{other}_{algo}_auc"] = float(combined - value)
    combined_xgb_auc = model_metrics["combined_xgb"]["test"]["auc"]
    if combined_xgb_auc == combined_xgb_auc:
        comparisons["combined_minus_majority_auc"] = float(
            combined_xgb_auc - baselines["majority"]["auc"]
        )
        comparisons["combined_minus_random_auc"] = float(
            combined_xgb_auc - baselines["random"]["auc"]
        )
    baselines["random"]["auc_floor"] = 0.5

    sentiment_meta = read_json(SENTIMENT_META_PATH) if SENTIMENT_META_PATH.exists() else {}
    metrics = {
        "metadata": {
            "generated_utc": utc_now_iso(),
            "config_path": cfg.source_path,
            "config_hash": cfg_hash,
            "tickers": list(cfg.universe.tickers),
            "price_source": cfg.data.price_source,
            "news_source": cfg.data.news_source,
            "sentiment_backend": sentiment_meta.get("backend", cfg.sentiment.backend),
            "date_range": {"start": cfg.date_range.start, "end": cfg.date_range.end},
            "random_seed": cfg.model.random_seed,
        },
        "splits": split_info.to_dict(),
        "dataset": {
            "n_rows": int(len(dataset)),
            "n_train": int(len(train)),
            "n_val": int(len(val)),
            "n_test": int(len(test)),
            "target_nan_dropped": int(dropped),
            "n_tickers": int(dataset["ticker"].nunique()),
            "test_positive_rate": float(test["y_next_up"].mean()),
        },
        "sentiment": sentiment_meta,
        "models": model_metrics,
        "baselines": baselines,
        "comparisons": comparisons,
    }
    write_json(metrics_path, metrics)
    write_manifest(metrics_path, stage="train", config_hash=cfg_hash, inputs=inputs,
                   extra={"models": list(model_metrics)})
    LOG.info("Wrote %s with %d models", metrics_path, len(model_metrics))
    return metrics


def _stage_backtest(cfg: Config, cfg_hash: str, force: bool) -> None:
    LOG.info("=== Stage: backtest ===")
    reports_dir = Path(cfg.reporting.output_dir)
    targets = [reports_dir / f"backtest_{fs}.json" for fs in model.FEATURE_SETS]
    inputs = [cfg.source_path, PREDICTIONS_PATH, reports_dir / "splits.json"]
    if not force and all(
        artifact_is_current(p, config_hash=cfg_hash, inputs=inputs) for p in targets
    ):
        LOG.info("backtest artifacts are current; skipping")
        return

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"{PREDICTIONS_PATH} not found; run --stage train before backtest"
        )
    prices = pd.read_parquet(PRICES_PATH)
    dataset = pd.read_parquet(DATASET_PATH)
    split_info = read_json(reports_dir / "splits.json")
    test_start = pd.Timestamp(split_info["test"]["start"])
    test_end = pd.Timestamp(split_info["test"]["end"])
    test = dataset[
        (pd.to_datetime(dataset["date"]) >= test_start)
        & (pd.to_datetime(dataset["date"]) <= test_end)
    ].copy()
    if test.empty:
        raise ValueError("test split is empty; cannot backtest")

    predictions = pd.read_parquet(PREDICTIONS_PATH)
    for fs_name in model.FEATURE_SETS:
        preds = predictions[
            (predictions["featureset"] == fs_name) & (predictions["algo"] == "xgb")
        ][["ticker", "date", "p_up"]]
        merged = test.merge(
            preds, on=["ticker", "date"], how="inner", validate="one_to_one"
        ).sort_values(["ticker", "date"]).reset_index(drop=True)
        if merged.empty:
            raise ValueError(f"no test predictions for {fs_name}")
        result = backtest_mod.run_backtest(
            merged, merged["p_up"].to_numpy(), cfg.backtest,
            prices=prices, featureset=fs_name, algo="xgb",
        )
        backtest_mod.save_backtest(result, cfg.reporting.output_dir, featureset=fs_name)
        write_manifest(
            reports_dir / f"backtest_{fs_name}.json", stage="backtest",
            config_hash=cfg_hash, inputs=inputs,
            extra={"featureset": fs_name, "algo": "xgb"},
        )
        write_manifest(
            reports_dir / f"equity_{fs_name}.png", stage="backtest",
            config_hash=cfg_hash, inputs=inputs,
            extra={"featureset": fs_name, "algo": "xgb"},
        )

    reporting.write_report(cfg.reporting.output_dir)
    write_manifest(reports_dir / "summary.md", stage="report", config_hash=cfg_hash,
                   inputs=inputs)
    importance_plot = reports_dir / "feature_importance_combined_xgb.png"
    if importance_plot.exists():
        write_manifest(importance_plot, stage="report", config_hash=cfg_hash,
                       inputs=inputs)
    write_json(
        reports_dir / "run_manifest.json",
        {
            "generated_utc": utc_now_iso(),
            "config_hash": cfg_hash,
            "artifacts": sorted(str(p) for p in (
                list(RAW.glob("*.parquet")) + list(PROCESSED.glob("*.parquet"))
                + list(reports_dir.glob("*.json")) + list(reports_dir.glob("*.png"))
                + list(reports_dir.glob("summary.md")) + list(MODELS_DIR.glob("*"))
            )),
        },
    )
    LOG.info("Wrote backtests and %s", reports_dir / "summary.md")


def run(config_path: str = "config.yaml", *, stage: str = "all", force: bool = False) -> None:
    """Execute stages in order, skipping current artifacts unless ``force``."""
    if stage not in (*STAGES, "all"):
        raise ValueError(f"unknown stage {stage!r}; choose from {(*STAGES, 'all')}")
    _load_env()
    log_path = setup_logging()
    cfg = load_config(config_path)
    cfg_hash = _config_hash(config_path)
    LOG.info("Loaded %s (hash %s); log: %s", config_path, cfg_hash[:12], log_path)

    for directory in (RAW, PROCESSED, ARTIFACTS, MODELS_DIR, Path(cfg.reporting.output_dir)):
        directory.mkdir(parents=True, exist_ok=True)

    requested = STAGES if stage == "all" else (stage,)
    for name in requested:
        if name == "prices":
            _stage_prices(cfg, cfg_hash, force)
        elif name == "news":
            _stage_news(cfg, cfg_hash, force)
        elif name == "sentiment":
            _stage_sentiment(cfg, cfg_hash, force)
        elif name == "features":
            _stage_features(cfg, cfg_hash, force)
        elif name == "train":
            _stage_train(cfg, cfg_hash, force)
        elif name == "backtest":
            _stage_backtest(cfg, cfg_hash, force)
    LOG.info("Pipeline finished (stage=%s)", stage)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stock direction predictor pipeline")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--stage", default="all", choices=(*STAGES, "all"))
    parser.add_argument("--force", action="store_true",
                        help="rebuild artifacts even if they are current")
    args = parser.parse_args(argv)
    run(args.config, stage=args.stage, force=args.force)


if __name__ == "__main__":  # pragma: no cover
    main()
