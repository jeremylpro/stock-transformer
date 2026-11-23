from __future__ import annotations

import pandas as pd

from src import pipeline, sentiment
from src.config import load_config


def _fake_scores(headlines, *, backend="auto", **kwargs):
    resolved = "vader" if backend == "vader" else "finbert"
    frame = pd.DataFrame(
        {
            "headline_idx": range(len(headlines)),
            "prob_positive": 0.6,
            "prob_negative": 0.2,
            "prob_neutral": 0.2,
            "sent_score": 0.4 if resolved == "finbert" else -0.4,
        }
    )
    frame.attrs["backend"] = resolved
    return frame


def _fake_vader(headlines):
    return pd.DataFrame({"sent_score": [0.0] * len(headlines)})


def test_backend_change_rescores_cached_headlines(monkeypatch, tmp_path):
    """A scorer change must rescore the cache, not drop cached headlines."""
    cache_path = tmp_path / "news_scored.parquet"
    pd.DataFrame(
        {
            "headline": ["a", "b"],
            "sent_score": [-0.4, -0.4],
            "vader_score": [0.0, 0.0],
            "backend": ["vader", "vader"],
            "model": ["vader", "vader"],
        }
    ).to_parquet(cache_path, index=False)
    monkeypatch.setattr(pipeline, "NEWS_SCORED_PATH", cache_path)
    monkeypatch.setattr(sentiment, "score_headlines", _fake_scores)
    monkeypatch.setattr(sentiment, "vader_scores", _fake_vader)

    cfg = load_config("config.yaml")
    assert cfg.sentiment.backend == "auto"
    news = pd.DataFrame({"headline": ["a", "b", "c"]})
    scores, backend = pipeline._headline_scores(news, cfg)

    assert backend == "finbert"
    assert set(scores["headline"]) == {"a", "b", "c"}
    assert scores["backend"].eq("finbert").all()
    assert scores["sent_score"].notna().all()
    assert scores.loc[scores["headline"].isin(["a", "b"]), "sent_score"].eq(0.4).all()
    saved = pd.read_parquet(cache_path)
    assert set(saved["headline"]) == {"a", "b", "c"}
