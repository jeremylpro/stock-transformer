"""Headline sentiment scoring (FinBERT preferred, VADER baseline)."""

from __future__ import annotations

import argparse
from typing import Iterable

import numpy as np
import pandas as pd

from .alignment import assign_headlines_to_trading_days
from .utils import get_logger

LOG = get_logger("sentiment")

SENTIMENT_COLUMNS = [
    "headline_idx",
    "prob_positive",
    "prob_negative",
    "prob_neutral",
    "sent_score",
]

_FINBERT_CACHE: dict[str, tuple[object, object]] = {}
_VADER = None


def vader_scores(headlines: list[str]) -> pd.DataFrame:
    """Score headlines with VADER. ``sent_score`` is the compound score."""
    global _VADER
    if _VADER is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        _VADER = SentimentIntensityAnalyzer()
    rows = []
    for idx, text in enumerate(headlines):
        scores = _VADER.polarity_scores(str(text))
        rows.append(
            {
                "headline_idx": idx,
                "prob_positive": float(scores["pos"]),
                "prob_negative": float(scores["neg"]),
                "prob_neutral": float(scores["neu"]),
                "sent_score": float(scores["compound"]),
            }
        )
    return pd.DataFrame(rows, columns=SENTIMENT_COLUMNS)


def _load_finbert(model_name: str):
    if model_name in _FINBERT_CACHE:
        return _FINBERT_CACHE[model_name]
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    LOG.info("Loading FinBERT model %s (CPU)", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    _FINBERT_CACHE[model_name] = (tokenizer, model)
    return tokenizer, model


def finbert_scores(
    headlines: list[str],
    *,
    model_name: str = "ProsusAI/finbert",
    batch_size: int = 32,
    max_tokens: int = 64,
) -> pd.DataFrame:
    """Score headlines with FinBERT.

    ``sent_score = P(positive) - P(negative)`` in [-1, 1]. Neutral probability is
    retained for diagnostics but is not a model feature.
    """
    import torch

    tokenizer, model = _load_finbert(model_name)
    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}

    rows: list[dict[str, float | int]] = []
    with torch.no_grad():
        for start in range(0, len(headlines), batch_size):
            batch = [str(h) for h in headlines[start:start + batch_size]]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_tokens,
                return_tensors="pt",
            )
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            for offset, prob in enumerate(probs):
                probs_by_label = {
                    label: float(prob[i]) for i, label in id2label.items() if i < len(prob)
                }
                pos = probs_by_label.get("positive", 0.0)
                neg = probs_by_label.get("negative", 0.0)
                neu = probs_by_label.get("neutral", 0.0)
                total = pos + neg + neu
                if total > 0:
                    pos, neg, neu = pos / total, neg / total, neu / total
                rows.append(
                    {
                        "headline_idx": start + offset,
                        "prob_positive": pos,
                        "prob_negative": neg,
                        "prob_neutral": neu,
                        "sent_score": pos - neg,
                    }
                )
    return pd.DataFrame(rows, columns=SENTIMENT_COLUMNS)


def score_headlines(
    headlines: list[str],
    model_name: str = "ProsusAI/finbert",
    batch_size: int = 32,
    *,
    max_tokens: int = 64,
    backend: str = "auto",
) -> pd.DataFrame:
    """Return ``headline_idx, prob_positive, prob_negative, prob_neutral, sent_score``.

    ``backend="auto"`` uses FinBERT when transformers+torch are importable and the
    model can be loaded, otherwise falls back to VADER with a warning.
    """
    headlines = [str(h) for h in headlines]
    if not headlines:
        return pd.DataFrame(columns=SENTIMENT_COLUMNS)
    backend = str(backend).lower()
    if backend not in {"auto", "finbert", "vader"}:
        raise ValueError("backend must be one of auto|finbert|vader")

    if backend == "vader":
        result = vader_scores(headlines)
        result.attrs["backend"] = "vader"
        return result
    try:
        result = finbert_scores(
            headlines,
            model_name=model_name,
            batch_size=batch_size,
            max_tokens=max_tokens,
        )
        result.attrs["backend"] = "finbert"
        return result
    except Exception as exc:
        if backend == "finbert":
            raise
        LOG.warning(
            "FinBERT unavailable (%s: %s); falling back to VADER. Install torch and "
            "transformers for the preferred scorer.",
            type(exc).__name__, exc,
        )
        result = vader_scores(headlines)
        result.attrs["backend"] = "vader"
        return result


def _score_missing(news_df: pd.DataFrame) -> pd.Series:
    """Score rows lacking ``sent_score`` with the configured auto backend."""
    scores = pd.Series(np.nan, index=news_df.index, dtype=float)
    missing = news_df["sent_score"].isna() if "sent_score" in news_df else pd.Series(
        True, index=news_df.index
    )
    if missing.any():
        result = score_headlines(news_df.loc[missing, "headline"].tolist())
        scores.loc[missing] = result["sent_score"].to_numpy()
    return scores


def daily_sentiment(
    news_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    *,
    market_tz: str = "America/New_York",
    close_time: str = "16:00",
    include_vader_baseline: bool = True,
) -> pd.DataFrame:
    """Apply §5.3 alignment, then aggregate per ``(ticker, trading_day)``.

    Days with no eligible headlines get ``sent_count=0`` and ``sent_mean=NaN``
    (never 0.0), so "no news" is distinguishable from neutral news.
    """
    if prices_df.empty:
        raise ValueError("prices_df is empty")
    for col in ("ticker", "date"):
        if col not in prices_df.columns:
            raise ValueError(f"prices_df missing required column {col!r}")
    for col in ("ticker", "headline", "published_utc"):
        if col not in news_df.columns:
            raise ValueError(f"news_df missing required column {col!r}")

    prices = prices_df[["ticker", "date"]].copy()
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    trading_days = prices["date"].drop_duplicates().sort_values().tolist()
    tickers = sorted(prices["ticker"].unique())

    if news_df.empty:
        assigned = news_df.copy()
        assigned["trading_day"] = pd.Series(dtype="datetime64[ns]")
    else:
        news = news_df.copy()
        news["ticker"] = news["ticker"].astype(str).str.upper()
        if "sent_score" not in news.columns:
            news["sent_score"] = _score_missing(news)
        else:
            news["sent_score"] = pd.to_numeric(news["sent_score"], errors="coerce")
            if news["sent_score"].isna().any():
                news.loc[news["sent_score"].isna(), "sent_score"] = _score_missing(news).loc[
                    news["sent_score"].isna()
                ]
        if include_vader_baseline and "vader_score" not in news.columns:
            LOG.info("Computing VADER baseline for %d headlines", len(news))
            news["vader_score"] = vader_scores(news["headline"].tolist())["sent_score"].to_numpy()

        # Keep only news for tickers in the price universe.
        news = news[news["ticker"].isin(tickers)]
        assigned = assign_headlines_to_trading_days(
            news, trading_days, market_tz=market_tz, close_time=close_time
        )

    if assigned.empty:
        grouped = pd.DataFrame(
            columns=["ticker", "date", "sent_mean", "sent_std", "sent_count",
                     "sent_vader_mean"]
        )
    else:
        agg = {
            "sent_mean": ("sent_score", "mean"),
            "sent_std": ("sent_score", "std"),
            "sent_count": ("sent_score", "size"),
        }
        if "vader_score" in assigned.columns:
            agg["sent_vader_mean"] = ("vader_score", "mean")
        grouped = (
            assigned.groupby(["ticker", "trading_day"], sort=True)
            .agg(**agg)
            .reset_index()
            .rename(columns={"trading_day": "date"})
        )

    # Reindex over every (ticker, trading day) so no-news days are explicit.
    full_index = pd.MultiIndex.from_product(
        [tickers, trading_days], names=["ticker", "date"]
    )
    grouped = grouped.set_index(["ticker", "date"]).reindex(full_index).reset_index()
    if "sent_count" not in grouped.columns:
        grouped["sent_count"] = 0
    grouped["sent_count"] = grouped["sent_count"].fillna(0).astype(int)
    if "sent_vader_mean" not in grouped.columns:
        grouped["sent_vader_mean"] = np.nan

    # A day with zero headlines must keep NaN, not zero.
    grouped.loc[grouped["sent_count"] == 0, "sent_mean"] = np.nan
    grouped.loc[grouped["sent_count"] < 2, "sent_std"] = np.nan

    grouped = grouped.sort_values(["ticker", "date"]).reset_index(drop=True)
    grouped["sent_lag1"] = grouped.groupby("ticker", sort=False)["sent_mean"].shift(1)

    return grouped[
        ["ticker", "date", "sent_mean", "sent_std", "sent_count", "sent_lag1",
         "sent_vader_mean"]
    ]


def main(argv: Iterable[str] | None = None) -> None:
    from .pipeline import run

    parser = argparse.ArgumentParser(description="Score headlines and aggregate daily sentiment")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    run(args.config, stage="sentiment", force=args.force)


if __name__ == "__main__":  # pragma: no cover
    main()
