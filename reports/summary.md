# Stock Direction Predictor MVP - Results Summary

Generated: 2026-11-23T07:31:36.715600+00:00
Universe: AAPL, MSFT, GOOGL, AMZN
Price source: `yfinance` | News source: `finnhub` | Sentiment backend: `finbert`
Config: `config.finnhub.yaml` | Requested range: 2025-12-01..2026-11-20

## 1. Chronological splits

| Split | Start | End | Trading dates |
|---|---|---|---:|
| train | 2025-12-01 | 2026-07-02 | 147 |
| validation | 2026-07-07 | 2026-09-14 | 49 |
| test | 2026-09-16 | 2026-11-19 | 47 |

Embargo: 1 trading day(s) between splits. Splits are chronological; no shuffling occurs anywhere in the training path.

Dataset: 980 labeled rows (588 train / 196 val / 188 test). Rows dropped for missing targets before splitting: 0.

## 2. Test metrics (all models on the identical held-out rows)

| Model | Test AUC | Accuracy | Brier | Log loss | N | Positive rate |
|---|---:|---:|---:|---:|---:|---:|
| combined_xgb | 0.4812 | 0.4840 | 0.2505 | 0.6942 | 188 | 0.5000 |
| combined_rf | 0.4655 | 0.4734 | 0.2594 | 0.7125 | 188 | 0.5000 |
| tech_only_xgb | 0.4668 | 0.4681 | 0.2505 | 0.6942 | 188 | 0.5000 |
| tech_only_rf | 0.4353 | 0.4574 | 0.2639 | 0.7219 | 188 | 0.5000 |
| sent_only_xgb | 0.5264 | 0.5053 | 0.2498 | 0.6927 | 188 | 0.5000 |
| sent_only_rf | 0.5527 | 0.5372 | 0.2497 | 0.6929 | 188 | 0.5000 |
| majority-class | 0.5000 | 0.5000 | 0.2500 | 0.6931 | 188 | 0.5000 |
| random (seeded draw) | 0.5045 | 0.4947 | 0.3273 | 0.9618 | 188 | 0.5000 |

The random baseline has an AUC floor of 0.50; a single seeded draw can deviate above or below it.

## 3. Combined vs. single-modality comparison

On the identical held-out test split, the combined model (XGB) did not outperform both single-modality baselines: its AUC was 0.4812, versus 0.4668 for technical-only (delta +0.0143) and 0.5264 for sentiment-only (delta -0.0453).

On the identical held-out test split, the combined model (RF) did not outperform both single-modality baselines: its AUC was 0.4655, versus 0.4353 for technical-only (delta +0.0302) and 0.5527 for sentiment-only (delta -0.0872).

| Comparison (XGB, test AUC) | Delta |
|---|---:|
| combined_minus_tech_only_xgb_auc | 0.0143 |
| combined_minus_sent_only_xgb_auc | -0.0453 |
| combined_minus_tech_only_rf_auc | 0.0302 |
| combined_minus_sent_only_rf_auc | -0.0872 |
| combined_minus_majority_auc | -0.0188 |
| combined_minus_random_auc | -0.0234 |

## 4. Top 15 feature importances - combined XGB

| Rank | Feature | Gain importance (normalized) |
|---:|---|---:|
| 1 | ta_macd_hist | 0.0566 |
| 2 | ret_lag_4 | 0.0546 |
| 3 | ta_rsi_14 | 0.0496 |
| 4 | ret_lag_5 | 0.0477 |
| 5 | ret_lag_3 | 0.0475 |
| 6 | sent_lag1 | 0.0442 |
| 7 | ta_obv_z_20 | 0.0441 |
| 8 | vol_5 | 0.0439 |
| 9 | ta_bb_pct_b_20 | 0.0438 |
| 10 | sent_std | 0.0433 |
| 11 | ta_macd_signal | 0.0433 |
| 12 | ret_z_20 | 0.0412 |
| 13 | ret_lag_1 | 0.0410 |
| 14 | ta_macd_line | 0.0409 |
| 15 | ta_sma_ratio_50 | 0.0409 |

## 5. Out-of-sample long/cash backtest (test period only)

| Feature set | Net cum. return | Gross cum. return | Net Sharpe | Max drawdown | Hit rate | Long days | Trades | Buy & hold (net) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| combined | 0.00% | 0.00% | 0.0000 | 0.00% | n/a | 0 | 0 | -5.66% |
| tech_only | 0.00% | 0.00% | 0.0000 | 0.00% | n/a | 0 | 0 | -5.66% |
| sent_only | 0.00% | 0.00% | 0.0000 | 0.00% | n/a | 0 | 0 | -5.66% |

Signals are generated at the close of day T and filled at the open of T+1. Net results charge commission and slippage on every position change. The backtest is an evaluation aid, not a tradeable strategy. Any position still open at the end of the test window is marked to market at the final close.

No model produced a probability at or above the entry threshold, so the strategy stayed in cash for the entire test period.

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
