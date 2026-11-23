"""Cost-aware long/cash backtest using backtesting.py (next-bar-open fills, G6)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .utils import ensure_parent, get_logger, write_json

LOG = get_logger("backtest")

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    featureset: str
    payload: dict[str, Any]
    equity: pd.DataFrame

    def to_dict(self) -> dict[str, Any]:
        return self.payload


def _backtest_algo() -> Any:
    from backtesting import Backtest, Strategy

    class LongCash(Strategy):
        """Long at next open when the signal says UP, cash otherwise."""

        def init(self) -> None:
            # Do not cache self.data.X here: data arrays are length-sliced per bar.
            pass

        def next(self) -> None:
            value = self.data.Signal[-1]
            threshold = self.data.Threshold[-1]
            if value is None or np.isnan(value):
                value = 0.0
            if value >= threshold:
                if not self.position:
                    self.buy()
            elif self.position:
                self.position.close()

    return Backtest, LongCash


def _adjusted_bars(prices_ticker: pd.DataFrame) -> pd.DataFrame:
    """Build adjusted OHLC bars (raw OHLC scaled to adj_close)."""
    g = prices_ticker.copy()
    g["date"] = pd.to_datetime(g["date"]).dt.normalize()
    g = g.sort_values("date").drop_duplicates("date", keep="last")
    factor = (g["adj_close"] / g["close"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    bars = pd.DataFrame(
        {
            "Open": g["open"].to_numpy() * factor.to_numpy(),
            "High": g["high"].to_numpy() * factor.to_numpy(),
            "Low": g["low"].to_numpy() * factor.to_numpy(),
            "Close": g["adj_close"].to_numpy(),
            "Volume": g["volume"].to_numpy(),
        },
        index=pd.DatetimeIndex(g["date"].to_numpy(), name="Date"),
    )
    return bars[~bars.index.duplicated(keep="last")]


def _equity_stats(equity: pd.Series) -> dict[str, float]:
    equity = equity.dropna().astype(float)
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return {
            "cumulative_return": 0.0,
            "annualized_sharpe": float("nan"),
            "max_drawdown": 0.0,
        }
    returns = equity.pct_change().dropna()
    std = returns.std(ddof=1)
    sharpe = float(returns.mean() / std * np.sqrt(TRADING_DAYS)) if std and std > 0 else 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return {
        "cumulative_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "annualized_sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
    }


def _run_single_ticker(
    bars: pd.DataFrame,
    signals: pd.Series,
    *,
    threshold: float,
    cost_rate: float,
) -> tuple[pd.Series, list[dict[str, Any]]]:
    Backtest, LongCash = _backtest_algo()
    data = bars.copy()
    data["Signal"] = signals.reindex(data.index).fillna(0.0).astype(float)
    data["Threshold"] = float(threshold)
    if len(data) < 2:
        raise ValueError("backtest needs at least 2 bars per ticker")

    bt = Backtest(
        data,
        LongCash,
        cash=100_000.0,
        commission=cost_rate,
        exclusive_orders=True,
        trade_on_close=False,   # G6: market orders fill at the next bar's open
        finalize_trades=True,
    )
    stats = bt.run()
    curve = stats["_equity_curve"]["Equity"].copy()
    trades = stats["_trades"]
    trade_rows: list[dict[str, Any]] = []
    if isinstance(trades, pd.DataFrame) and not trades.empty:
        for _, row in trades.iterrows():
            trade_rows.append(
                {
                    "entry_time": str(row.get("EntryTime")),
                    "exit_time": str(row.get("ExitTime")),
                    "entry_price": float(row.get("EntryPrice", np.nan)),
                    "exit_price": float(row.get("ExitPrice", np.nan)),
                    "pnl": float(row.get("PnL", np.nan)),
                    "return_pct": float(row.get("ReturnPct", np.nan)),
                }
            )
    return curve, trade_rows


def run_backtest(
    test_df: pd.DataFrame,
    p_up: np.ndarray,
    cfg: BacktestConfig,
    *,
    prices: pd.DataFrame | None = None,
    featureset: str = "combined",
    algo: str = "xgb",
) -> BacktestResult:
    """Simulate long/cash per ticker, then equal-weight the equity curves.

    Signals are generated at the close of T and executed at the open of T+1.
    ``prices`` (the full price panel) is optional; when supplied, one bar after
    the final test date is included so the last signal can fill.
    """
    required = {"ticker", "date", "y_next_up", "open", "high", "low", "close",
                "adj_close", "volume"}
    missing = required - set(test_df.columns)
    if missing:
        raise ValueError(f"test_df missing columns for backtest: {sorted(missing)}")
    if len(test_df) != len(p_up):
        raise ValueError("test_df and p_up differ in length")

    test = test_df.copy()
    test["date"] = pd.to_datetime(test["date"]).dt.normalize()
    test["ticker"] = test["ticker"].astype(str).str.upper()
    test["p_up"] = np.asarray(p_up, dtype=float)
    test = test.sort_values(["ticker", "date"]).reset_index(drop=True)

    source_prices = prices if prices is not None else test
    source_prices = source_prices.copy()
    source_prices["date"] = pd.to_datetime(source_prices["date"]).dt.normalize()
    source_prices["ticker"] = source_prices["ticker"].astype(str).str.upper()

    curves_net: dict[str, pd.Series] = {}
    curves_gross: dict[str, pd.Series] = {}
    buyhold: dict[str, pd.Series] = {}
    per_ticker: dict[str, Any] = {}
    all_trades_net: list[dict[str, Any]] = []
    long_mask = test["p_up"] >= cfg.entry_threshold

    for ticker, group in test.groupby("ticker", sort=True):
        group = group.sort_values("date")
        ticker_prices = source_prices[source_prices["ticker"] == ticker].sort_values("date")
        if ticker_prices.empty:
            raise ValueError(f"no prices for ticker {ticker}")

        first_date = group["date"].min()
        last_date = group["date"].max()
        calendar = source_prices[source_prices["ticker"] == ticker]["date"].sort_values()
        # Include one bar before the first test signal so next() can act on the
        # first test day's signal (backtesting.py does not call next() on bar 0).
        prior = calendar[calendar < first_date]
        bar_start = prior.iloc[-1] if len(prior) else first_date
        after = calendar[calendar > last_date]
        end_date = after.iloc[0] if len(after) else last_date
        bars = _adjusted_bars(
            ticker_prices[(ticker_prices["date"] >= bar_start) &
                          (ticker_prices["date"] <= end_date)]
        )
        if bars.empty:
            raise ValueError(f"no adjusted bars for {ticker} in backtest window")

        signals = pd.Series(0.0, index=bars.index)
        for _, row in group.iterrows():
            if row["date"] in signals.index:
                signals.loc[row["date"]] = float(row["p_up"])

        net_curve, trades = _run_single_ticker(
            bars, signals, threshold=cfg.entry_threshold,
            cost_rate=cfg.round_trip_cost_rate,
        )
        gross_curve, _ = _run_single_ticker(
            bars, signals, threshold=cfg.entry_threshold, cost_rate=0.0,
        )
        curves_net[ticker] = net_curve / net_curve.iloc[0]
        curves_gross[ticker] = gross_curve / gross_curve.iloc[0]

        base_close = bars["Close"] / bars["Close"].iloc[0]
        buyhold[ticker] = base_close

        ticker_hits = group.loc[
            group["p_up"] >= cfg.entry_threshold, "y_next_up"
        ]
        per_ticker[ticker] = {
            "n_signals": int((group["p_up"] >= cfg.entry_threshold).sum()),
            "hit_rate": float(ticker_hits.mean()) if len(ticker_hits) else None,
            "n_trades": len(trades),
            "net": _equity_stats(curves_net[ticker]),
            "gross": _equity_stats(curves_gross[ticker]),
            "buy_and_hold": _equity_stats(buyhold[ticker]),
        }
        for t in trades:
            t["ticker"] = ticker
        all_trades_net.extend(trades)

    def _combine(curves: dict[str, pd.Series]) -> pd.Series:
        frame = pd.DataFrame(curves).sort_index()
        frame = frame.ffill().bfill()
        return frame.mean(axis=1)

    net_portfolio = _combine(curves_net)
    gross_portfolio = _combine(curves_gross)
    bh_portfolio = _combine(buyhold)

    # Buy-and-hold net of a single entry cost per ticker.
    bh_net_portfolio = bh_portfolio * (1.0 - cfg.round_trip_cost_rate)

    hits = test.loc[long_mask, "y_next_up"]
    payload = {
        "featureset": featureset,
        "algo": algo,
        "entry_threshold": cfg.entry_threshold,
        "costs": {
            "commission_bps": cfg.commission_bps,
            "slippage_bps": cfg.slippage_bps,
            "round_trip_cost_rate": cfg.round_trip_cost_rate,
            "fill": cfg.fill,
            "trade_on_close": cfg.trade_on_close,
        },
        "period": {
            "start": str(test["date"].min().date()),
            "end": str(test["date"].max().date()),
            "n_test_rows": int(len(test)),
        },
        "n_tickers": int(test["ticker"].nunique()),
        "n_long_days": int(long_mask.sum()),
        "hit_rate": float(hits.mean()) if len(hits) else None,
        "n_trades": int(len(all_trades_net)),
        "n_position_changes": int(len(all_trades_net) * 2),
        "trades": all_trades_net[:200],
        "gross": _equity_stats(gross_portfolio),
        "net": _equity_stats(net_portfolio),
        "buy_and_hold_gross": _equity_stats(bh_portfolio),
        "buy_and_hold_net": _equity_stats(bh_net_portfolio),
        "per_ticker": per_ticker,
    }
    equity = pd.DataFrame(
        {
            "net": net_portfolio,
            "gross": gross_portfolio,
            "buy_and_hold": bh_portfolio,
        }
    )
    equity.index.name = "date"

    LOG.info(
        "%s/%s backtest: net return %.2f%%, gross %.2f%%, trades %d, hit rate %s",
        featureset, algo,
        100 * payload["net"]["cumulative_return"],
        100 * payload["gross"]["cumulative_return"],
        payload["n_trades"],
        f"{payload['hit_rate']:.3f}" if payload["hit_rate"] is not None else "n/a",
    )
    return BacktestResult(featureset=featureset, payload=payload, equity=equity)


def save_backtest(result: BacktestResult, output_dir: str, *, featureset: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / f"backtest_{featureset}.json", result.to_dict())

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(result.equity.index, result.equity["net"], label="Long/cash net of costs")
    ax.plot(result.equity.index, result.equity["gross"], label="Long/cash gross",
            linestyle="--", alpha=0.7)
    ax.plot(result.equity.index, result.equity["buy_and_hold"],
            label="Buy & hold (gross)", linestyle=":", alpha=0.8)
    ax.set_title(f"Out-of-sample equity curve - {featureset}")
    ax.set_ylabel("Growth of $1")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = ensure_parent(out / f"equity_{featureset}.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _main(argv: Iterable[str] | None = None) -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Run the out-of-sample backtest")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    from .pipeline import run

    run(args.config, stage="backtest", force=args.force)


if __name__ == "__main__":  # pragma: no cover
    _main()
