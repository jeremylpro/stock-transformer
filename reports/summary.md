# Stock Direction Predictor MVP - Results Summary

Generated: 2026-11-23T06:21:14.751777+00:00
Universe: AAPL, MSFT, GOOGL, AMZN
Price source: `yfinance` | News source: `local` | Sentiment backend: `finbert`

> **OFFLINE / DEMO DATA WARNING.** At least one input comes from a local fixture or synthetic demo file. These numbers exercise the pipeline and are **not** evidence about markets. Re-run with a FINNHUB_API_KEY for real results.

## 1. Chronological splits

| Split | Start | End | Trading dates |
|---|---|---|---:|
| train | 2021-01-04 | 2023-09-08 | 675 |
| validation | 2023-09-12 | 2024-08-02 | 225 |
| test | 2024-08-06 | 2025-06-27 | 224 |

Embargo: 1 trading day(s) between splits. Splits are chronological; no shuffling occurs anywhere in the training path.

Dataset: 4504 labeled rows (2700 train / 900 val / 896 test). Rows dropped for missing targets before splitting: 0.

## 2. Test metrics (all models on the identical held-out rows)

| Model | Test AUC | Accuracy | Brier | Log loss | N | Positive rate |
|---|---:|---:|---:|---:|---:|---:|
| combined_xgb | 0.4887 | 0.5391 | 0.2492 | 0.6916 | 896 | 0.5413 |
| combined_rf | 0.5067 | 0.5201 | 0.2508 | 0.6948 | 896 | 0.5413 |
| tech_only_xgb | 0.5032 | 0.5413 | 0.2490 | 0.6912 | 896 | 0.5413 |
| tech_only_rf | 0.5117 | 0.5268 | 0.2503 | 0.6938 | 896 | 0.5413 |
| sent_only_xgb | 0.4976 | 0.5357 | 0.2492 | 0.6915 | 896 | 0.5413 |
| sent_only_rf | 0.5189 | 0.5301 | 0.2503 | 0.6938 | 896 | 0.5413 |
| majority-class | 0.5000 | 0.5413 | 0.2483 | 0.6897 | 896 | 0.5413 |
| random (seeded draw) | 0.4835 | 0.4844 | 0.3434 | 1.0282 | 896 | 0.5413 |

The random baseline has an AUC floor of 0.50; a single seeded draw can deviate above or below it.

## 3. Combined vs. single-modality comparison

On the identical held-out test split, the combined model (XGB) did not outperform both single-modality baselines: its AUC was 0.4887, versus 0.5032 for technical-only (delta -0.0145) and 0.4976 for sentiment-only (delta -0.0090).

On the identical held-out test split, the combined model (RF) did not outperform both single-modality baselines: its AUC was 0.5067, versus 0.5117 for technical-only (delta -0.0050) and 0.5189 for sentiment-only (delta -0.0122).

| Comparison (XGB, test AUC) | Delta |
|---|---:|
| combined_minus_tech_only_xgb_auc | -0.0145 |
| combined_minus_sent_only_xgb_auc | -0.0090 |
| combined_minus_tech_only_rf_auc | -0.0050 |
| combined_minus_sent_only_rf_auc | -0.0122 |
| combined_minus_majority_auc | -0.0113 |
| combined_minus_random_auc | 0.0052 |

## 4. Top 15 feature importances - combined XGB

| Rank | Feature | Gain importance (normalized) |
|---:|---|---:|
| 1 | ta_sma_ratio_50 | 0.0532 |
| 2 | ta_ema_ratio_20 | 0.0516 |
| 3 | ret_lag_2 | 0.0507 |
| 4 | sent_mean | 0.0503 |
| 5 | ta_sma_ratio_10 | 0.0501 |
| 6 | ta_macd_line | 0.0465 |
| 7 | ta_obv_z_20 | 0.0463 |
| 8 | ta_rsi_14 | 0.0449 |
| 9 | ret_cum_5 | 0.0442 |
| 10 | ret_z_20 | 0.0432 |
| 11 | sent_std | 0.0421 |
| 12 | ta_vol_z_20 | 0.0406 |
| 13 | ret_lag_5 | 0.0402 |
| 14 | sent_lag1 | 0.0393 |
| 15 | ret_lag_3 | 0.0386 |

## 5. Out-of-sample long/cash backtest (test period only)

| Feature set | Net cum. return | Gross cum. return | Net Sharpe | Max drawdown | Hit rate | Long days | Trades | Buy & hold (net) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| combined | 0.00% | 0.00% | 0.0000 | 0.00% | n/a | 0 | 0 | 18.20% |
| tech_only | 0.00% | 0.00% | 0.0000 | 0.00% | n/a | 0 | 0 | 18.20% |
| sent_only | 0.06% | 1.42% | 0.0415 | -2.18% | 0.5556 | 18 | 18 | 18.20% |

Signals are generated at the close of day T and filled at the open of T+1. Net results charge commission and slippage on every position change. The backtest is an evaluation aid, not a tradeable strategy. Any position still open at the end of the test window is marked to market at the final close.

## 6. Interpretation

Test AUC is below 0.50: the model does no better than chance on this split; this is reported as a negative result.

## 7. Leakage guardrails

| ID | Guardrail | Status |
|---|---|---|
| G1 | 16:00 ET news cutoff owned by `alignment.eligible_headlines_for_day` | Enforced + unit tested |
| G2 | No target columns in the feature matrix | Enforced + unit tested |
| G3 | Chronological splits, no shuffle API | Enforced + unit tested |
| G4 | Winsorize/impute fitted on train only | Enforced + unit tested |
| G5 | Returns use `adj_close` | Enforced + unit tested |
| G6 | Next-bar-open backtest fills (`trade_on_close=False`) | Enforced + config tested |
| G7 | Missing-target rows dropped before splitting (count logged) | Enforced |

## 8. Known limitations

- Small universe of current large caps; survivorship and selection bias.
- Headline-only sentiment discards article bodies, entity disambiguation and source credibility.
- Finnhub's free tier has a limited historical window and coverage gaps; `sent_count` measures availability, not quality.
- A single chronological split gives high-variance AUC estimates; bootstrapped confidence intervals are post-MVP work.
- No market-regime features (VIX, rates, sector returns) and no cross-sectional normalization across tickers.
- The long/cash equal-weight backtest ignores market impact, borrow costs, taxes and dividend reinvestment details; it is not a tradeable strategy.
