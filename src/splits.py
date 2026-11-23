"""Chronological train/validation/test splits with an embargo (G3)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .config import SplitsConfig
from .utils import get_logger

LOG = get_logger("splits")


@dataclass(frozen=True)
class SplitInfo:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train_dates: int
    n_val_dates: int
    n_test_dates: int
    embargo_days: int

    def to_dict(self) -> dict[str, object]:
        return {
            "train": {"start": str(self.train_start.date()), "end": str(self.train_end.date())},
            "validation": {"start": str(self.val_start.date()),
                           "end": str(self.val_end.date())},
            "test": {"start": str(self.test_start.date()), "end": str(self.test_end.date())},
            "n_train_dates": self.n_train_dates,
            "n_val_dates": self.n_val_dates,
            "n_test_dates": self.n_test_dates,
            "embargo_days": self.embargo_days,
        }


def split_dates(
    dates: Iterable[pd.Timestamp], cfg: SplitsConfig
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex, SplitInfo]:
    """Split sorted unique dates with an embargo day between adjacent splits."""
    unique = pd.DatetimeIndex(pd.to_datetime(list(dates))).normalize().drop_duplicates()
    unique = unique.sort_values()
    n = len(unique)
    if n < 6:
        raise ValueError(f"need at least 6 unique trading dates to split, got {n}")

    n_train = int(np.floor(n * cfg.train_ratio))
    n_val = int(np.floor(n * cfg.val_ratio))
    embargo = int(cfg.embargo_days)

    # Leave room for both embargo gaps and at least one test date.
    if n_train < 1:
        n_train = 1
    if n_val < 1:
        n_val = 1
    while n_train + n_val + 2 * embargo >= n and n_train > 1:
        n_train -= 1
    while n_train + n_val + 2 * embargo >= n and n_val > 1:
        n_val -= 1
    if n_train + n_val + 2 * embargo >= n:
        raise ValueError(
            f"not enough dates ({n}) for train={n_train}, val={n_val}, "
            f"embargo={embargo} on each side"
        )

    train = unique[:n_train]
    val_start = n_train + embargo
    val = unique[val_start:val_start + n_val]
    test_start = val_start + n_val + embargo
    test = unique[test_start:]

    if len(val) == 0 or len(test) == 0:
        raise ValueError("chronological split produced an empty validation or test set")

    info = SplitInfo(
        train_start=train[0], train_end=train[-1],
        val_start=val[0], val_end=val[-1],
        test_start=test[0], test_end=test[-1],
        n_train_dates=len(train), n_val_dates=len(val), n_test_dates=len(test),
        embargo_days=embargo,
    )
    LOG.info(
        "Split dates: train %s..%s (%d), val %s..%s (%d), test %s..%s (%d), embargo %d",
        train[0].date(), train[-1].date(), len(train),
        val[0].date(), val[-1].date(), len(val),
        test[0].date(), test[-1].date(), len(test), embargo,
    )
    return train, val, test, info


def chronological_split(
    df: pd.DataFrame, cfg: SplitsConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split on unique dates, not rows, so every ticker stays together.

    The function deliberately exposes no shuffle parameter (G3).
    """
    if "date" not in df.columns:
        raise ValueError("df must contain a 'date' column")
    dates = pd.to_datetime(df["date"]).dt.normalize()
    train_dates, val_dates, test_dates, _ = split_dates(dates.unique(), cfg)
    train = df.loc[dates.isin(train_dates)].copy()
    val = df.loc[dates.isin(val_dates)].copy()
    test = df.loc[dates.isin(test_dates)].copy()
    return train, val, test


def _main(argv: Iterable[str] | None = None) -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Inspect chronological splits")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(argv)
    from .config import load_config

    cfg = load_config(args.config)
    # Only to show boundaries; the real data is split inside the pipeline.
    dates = pd.bdate_range(cfg.date_range.start, cfg.date_range.end)
    _, _, _, info = split_dates(dates, cfg.splits)
    from .utils import write_json

    write_json("reports/splits.json", info.to_dict())


if __name__ == "__main__":  # pragma: no cover
    _main()
