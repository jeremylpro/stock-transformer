"""Generate deterministic fixtures used by the test suite.

Run from the project root:  python scripts/generate_fixtures.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
TZ = "America/New_York"


def _price_panel(tickers: dict[str, float], start: str, days: int,
                 seed: int) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=days)
    frames = []
    for i, (ticker, base) in enumerate(tickers.items()):
        rng = np.random.default_rng(seed + i)
        returns = rng.normal(0.0004, 0.011, size=len(dates))
        close = base * np.exp(np.cumsum(returns))
        open_ = close * (1 + rng.normal(0, 0.002, size=len(dates)))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, size=len(dates))))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, size=len(dates))))
        volume = rng.integers(500_000, 5_000_000, size=len(dates))
        frames.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "date": dates,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "adj_close": close,
                    "volume": volume,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _headline_row(ticker: str, headline: str, when_et: pd.Timestamp,
                  source: str) -> dict[str, object]:
    url = f"https://example.test/{ticker}/{hashlib.sha256(headline.encode()).hexdigest()}"
    return {
        "ticker": ticker,
        "headline": headline,
        "published_utc": when_et.tz_convert("UTC"),
        "source": source,
        "url_hash": hashlib.sha256(url.encode()).hexdigest()[:16],
    }


def leakage_fixtures() -> None:
    prices = _price_panel(
        {"AAPL": 180.0, "MSFT": 380.0, "GOOGL": 140.0}, "2024-01-02", 90, seed=7
    )
    # A deliberate future spike at 2024-04-01; features before it must not change.
    spike = (prices["ticker"] == "AAPL") & (prices["date"] == pd.Timestamp("2024-04-01"))
    for col in ("open", "high", "low", "close", "adj_close"):
        prices.loc[spike, col] = prices.loc[spike, col] * 3.0
    prices.to_csv(OUT / "leakage_prices.csv", index=False)

    rows = [
        _headline_row("AAPL", "Apple tops earnings expectations",
                      pd.Timestamp("2024-02-05 15:59", tz=TZ), "wire"),
        _headline_row("AAPL", "Apple hit by regulatory probe",
                      pd.Timestamp("2024-02-05 16:01", tz=TZ), "wire"),
        _headline_row("AAPL", "Apple product event announced",
                      pd.Timestamp("2024-02-05 16:00", tz=TZ), "wire"),
        _headline_row("MSFT", "Microsoft cloud demand remains strong",
                      pd.Timestamp("2024-02-03 10:00", tz=TZ), "paper"),
        _headline_row("GOOGL", "Alphabet ad sales soften in Europe",
                      pd.Timestamp("2024-02-04 12:00", tz=TZ), "paper"),
        # Far beyond the price panel: must never be eligible for any trading day.
        _headline_row("AAPL", "Apple launches a rocket to the moon",
                      pd.Timestamp("2024-12-31 23:00", tz=TZ), "future"),
        _headline_row("AAPL", "Duplicate headline text",
                      pd.Timestamp("2024-03-01 10:00", tz=TZ), "wire"),
        _headline_row("AAPL", "Duplicate headline text",
                      pd.Timestamp("2024-03-01 11:00", tz=TZ), "other"),
        _headline_row("AAPL", "   ", pd.Timestamp("2024-03-02 10:00", tz=TZ), "wire"),
    ]
    pd.DataFrame(rows).to_csv(OUT / "leakage_news.csv", index=False)


def smoke_fixtures() -> None:
    tickers = {"AAPL": 180.0, "MSFT": 380.0, "GOOGL": 140.0, "AMZN": 160.0, "NVDA": 600.0}
    prices = _price_panel(tickers, "2024-01-02", 60, seed=21)
    prices.to_parquet(OUT / "smoke_prices.parquet", index=False)

    templates = {
        "positive": [
            "Company beats quarterly profit expectations",
            "Analysts upgrade shares on strong demand",
            "New product launch draws record preorders",
            "Revenue growth accelerates for a third quarter",
            "Board approves a larger share buyback",
        ],
        "negative": [
            "Regulators open an investigation into accounting",
            "Guidance cut as demand weakens",
            "Key executive unexpectedly resigns",
            "Lawsuit alleges patent infringement",
            "Margins compress on rising input costs",
        ],
        "neutral": [
            "Company to present at an industry conference",
            "Quarterly earnings scheduled for next week",
            "Board declares an unchanged dividend",
            "Shares begin trading on a new exchange segment",
            "Annual shareholder meeting date announced",
        ],
    }
    rng = np.random.default_rng(99)
    dates = pd.bdate_range("2024-01-02", periods=60)
    rows = []
    for ticker in tickers:
        for day in dates:
            n = int(rng.integers(0, 3))
            for _ in range(n):
                kind = rng.choice(["positive", "negative", "neutral"], p=[0.35, 0.35, 0.30])
                headline = f"{ticker}: {rng.choice(templates[kind])}"
                hour = int(rng.choice([8, 9, 10, 11, 13, 14, 15, 16, 17]))
                minute = int(rng.choice([0, 15, 30, 45, 59]))
                when = pd.Timestamp(day) + pd.Timedelta(hours=hour, minutes=minute)
                rows.append(_headline_row(ticker, headline, when.tz_localize(TZ), "demo-wire"))
            if rng.random() < 0.10:
                when = pd.Timestamp(day) + pd.Timedelta(days=1, hours=10)
                rows.append(
                    _headline_row(
                        ticker,
                        f"{ticker}: Weekend market commentary",
                        when.tz_localize(TZ),
                        "demo-wire",
                    )
                )
    news = pd.DataFrame(rows).drop_duplicates(subset=["ticker", "url_hash"])
    news.to_parquet(OUT / "smoke_news.parquet", index=False)
    print(f"wrote {len(prices)} price rows and {len(news)} headlines to {OUT}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    leakage_fixtures()
    smoke_fixtures()
