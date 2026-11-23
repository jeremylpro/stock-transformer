"""Next-day labels derived from adjusted close (G5)."""

from __future__ import annotations

import argparse
from typing import Iterable

import numpy as np
import pandas as pd

from .utils import get_logger

LOG = get_logger("labels")


def build_labels(prices: pd.DataFrame) -> pd.DataFrame:
    """Return ``ticker, date, log_ret_next, y_next_up``.

    ``log_ret_next = ln(adj_close[T+1] / adj_close[T])`` and
    ``y_next_up = 1 if log_ret_next > 0 else 0``. A zero-return day is labeled 0
    (DOWN). The final row per ticker has no target and is dropped.
    """
    if "adj_close" not in prices.columns:
        raise AssertionError(
            "G5 violated: build_labels requires an 'adj_close' column"
        )
    required = {"ticker", "date", "adj_close"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"prices missing required columns: {sorted(missing)}")

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    frames = []
    for ticker, group in prices.groupby("ticker", sort=True):
        g = group.sort_values("date").reset_index(drop=True)
        next_adj = g["adj_close"].shift(-1)
        log_ret_next = np.log(next_adj / g["adj_close"])
        labels = pd.DataFrame(
            {
                "ticker": g["ticker"],
                "date": g["date"],
                "log_ret_next": log_ret_next,
                "y_next_up": (log_ret_next > 0).astype("float"),
            }
        )
        # The last row per ticker has no next-day price and is dropped.
        labels = labels.iloc[:-1] if len(labels) else labels
        labels["y_next_up"] = labels["y_next_up"].astype(int)
        frames.append(labels)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["ticker", "date", "log_ret_next", "y_next_up"]
    )
    out = out[out["log_ret_next"].notna()].reset_index(drop=True)
    LOG.info("Built %d labels (%d rows dropped for missing next-day price)",
             len(out), len(prices) - len(out))
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def main(argv: Iterable[str] | None = None) -> None:
    from .pipeline import run

    parser = argparse.ArgumentParser(description="Build next-day direction labels")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    run(args.config, stage="train", force=args.force)


if __name__ == "__main__":  # pragma: no cover
    main()
