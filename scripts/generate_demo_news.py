"""Create synthetic demo headlines from a price panel.

This exists only so the pipeline can be exercised without a Finnhub key. The
headlines are generated independently of returns, so the resulting metrics have
no economic meaning. They are written to ``data/demo/news.parquet`` and used by
``config.demo.yaml``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
TZ = "America/New_York"

POSITIVE = [
    "{t} beats quarterly revenue expectations",
    "Analysts raise {t} price target after strong demand",
    "{t} announces a larger share buyback",
    "{t} launches a product that draws record preorders",
    "{t} margins expand on cost discipline",
    "{t} wins a multi-year enterprise contract",
    "{t} reports accelerating cloud growth",
    "{t} guidance points to a stronger second half",
    "{t} adds a new market to its platform",
    "{t} free cash flow tops consensus",
]
NEGATIVE = [
    "{t} guidance cut as demand weakens",
    "Regulators open a probe into {t} accounting",
    "{t} loses a key executive",
    "{t} faces a patent lawsuit from a rival",
    "{t} margins compress on rising input costs",
    "{t} recalls a product after quality complaints",
    "Analysts downgrade {t} on slowing growth",
    "{t} subscriber churn rises unexpectedly",
    "{t} delays a flagship launch",
    "{t} warns about foreign-exchange headwinds",
]
NEUTRAL = [
    "{t} schedules its quarterly earnings call",
    "{t} to present at an industry conference",
    "{t} declares an unchanged dividend",
    "{t} announces its annual shareholder meeting date",
    "{t} begins trading on a new exchange segment",
    "{t} completes a routine debt refinancing",
    "{t} appoints a new board member",
    "{t} publishes its annual sustainability report",
    "{t} holds an investor day next month",
    "{t} updates its corporate governance page",
]


def main(seed: int = 123) -> None:
    prices = pd.read_parquet(PROJECT / "data" / "raw" / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    for ticker, group in prices.groupby("ticker", sort=True):
        for day in sorted(group["date"].unique()):
            day = pd.Timestamp(day)
            # Weekend stories roll into the next trading day; keep a few.
            if rng.random() < 0.12:
                when = day + pd.Timedelta(days=int(rng.integers(1, 3)),
                                          hours=int(rng.integers(9, 18)))
                text = f"{ticker}: weekend commentary on the sector"
                rows.append(_row(ticker, text, when, rng))
            if rng.random() < 0.25:
                continue
            for _ in range(int(rng.integers(1, 3))):
                bucket = rng.choice(["positive", "negative", "neutral"], p=[0.34, 0.33, 0.33])
                template = rng.choice({"positive": POSITIVE, "negative": NEGATIVE,
                                       "neutral": NEUTRAL}[str(bucket)])
                text = str(template).format(t=ticker)
                hour = int(rng.choice([7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]))
                minute = int(rng.choice([0, 10, 20, 30, 40, 50, 59]))
                when = day + pd.Timedelta(hours=hour, minutes=minute)
                rows.append(_row(ticker, text, when, rng))

    news = pd.DataFrame(rows)
    news["published_utc"] = pd.to_datetime(news["published_utc"], utc=True)
    news = news.drop_duplicates(subset=["ticker", "url_hash"]).sort_values(
        ["ticker", "published_utc"]
    ).reset_index(drop=True)
    out = PROJECT / "data" / "demo" / "news.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    news.to_parquet(out, index=False)
    print(f"wrote {len(news)} synthetic headlines to {out}")
    print("WARNING: synthetic data; do not interpret any resulting metrics.")


def _row(ticker: str, text: str, when: pd.Timestamp, rng: np.random.Generator) -> dict:
    unique = f"{ticker}|{when.isoformat()}|{text}|{rng.integers(0, 1_000_000)}"
    return {
        "ticker": ticker,
        "headline": text,
        "published_utc": when.tz_localize(TZ).tz_convert("UTC"),
        "source": "synthetic-demo",
        "url_hash": hashlib.sha256(unique.encode()).hexdigest()[:16],
    }


if __name__ == "__main__":
    main()
