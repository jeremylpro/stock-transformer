# Product Design Document: Stock Direction Predictor MVP

---

## 1. Product Overview

### 1.1 Summary
A minimal viable product (MVP) that predicts the **next-day direction (UP/DOWN)** of a small set of large-cap stocks by combining **technical indicators** and **news sentiment**. The MVP is designed to quickly and honestly validate whether fusing price-based and news-based signals adds measurable predictive value over either signal alone.

### 1.2 Overall Direction
Build a clean, leak-free, interpretable baseline before adding complexity. The MVP prioritizes:
- **Narrow scope**: few stocks, daily frequency, binary direction.
- **Data integrity**: strict timestamp alignment, adjusted prices, no look-ahead bias.
- **Simple fusion**: concatenate technical and sentiment features into a tabular model.
- **Rigorous evaluation**: chronological splits, benchmark comparisons, and a basic backtest with costs.

### 1.3 Goals
- Predict next-day UP/DOWN direction for 3–5 large-cap, highly liquid stocks.
- Use daily adjusted OHLCV data and daily news headlines.
- Incorporate **log returns** as both lagged features and the basis for the prediction target.
- Compare a combined model against technical-only and sentiment-only baselines.
- Produce a simple, cost-aware backtest on out-of-sample data.

### 1.4 Non-Goals (for MVP)
- Intraday prediction or high-frequency trading.
- Exact price or return forecasting (regression is optional later).
- Full article NLP, topic modeling, or event extraction.
- Deep learning architectures (LSTM, Transformers for price sequences).
- Broad market coverage or small-cap/illiquid stocks.
- Live trading or production-grade serving infrastructure.

---

## 2. Target Users & Use Cases

**Primary user:** The builder/researcher validating the signal.  
**Secondary users:** Early stakeholders reviewing whether the approach merits further investment.

**Key use cases:**
- Run a daily pipeline that ingests prices and news, computes features, and outputs a direction prediction per stock.
- Inspect feature importance to understand whether sentiment contributes beyond technicals.
- Review out-of-sample AUC and a simple long/cash backtest to judge practical value.

---

## 3. MVP Scope Summary

| Dimension | MVP Decision |
|---|---|
| **Universe** | 3–5 large-cap, liquid stocks (e.g., AAPL, MSFT, GOOGL) |
| **Frequency** | Daily |
| **Prediction target** | Next-day direction: \(y_{T+1}=1\) if next-day log return \(r_{T+1}>0\), else 0 |
| **Return definition** | Daily log return \(r_t=\ln(P_t/P_{t-1})\) using adjusted close |
| **Price data** | Adjusted daily OHLCV |
| **News data** | Daily news headlines only |
| **Technical features** | 5–7 indicators + 3–5 lagged log returns |
| **Sentiment features** | Daily average headline sentiment (FinBERT or VADER) + optional 1-day lag |
| **Model** | XGBoost or Random Forest on concatenated features |
| **Validation** | Chronological split / TimeSeriesSplit |
| **Metrics** | AUC-ROC, accuracy; benchmark vs. technical-only and sentiment-only |
| **Backtest** | Simple long/cash simulation with transaction costs and slippage |
| **Guardrails** | No look-ahead bias, no random shuffling, adjusted prices only |

---

## 4. Main Features

### 4.1 Data Ingestion
- **Price data**: Fetch daily OHLCV via `yfinance` or equivalent. Use **adjusted close** for all return calculations. Store raw and adjusted series.
- **News data**: Fetch daily headlines from a news API (e.g., NewsAPI). Store headline text, source, and publication timestamp.
- **Alignment rule**: For a prediction made at the close of day \(T\), only news published **before the close of day \(T\)** may be used. Any later news is excluded from features for that prediction.

### 4.2 Feature Engineering

**Log return features (required):**
- Lagged daily log returns: \(r_T, r_{T-1}, r_{T-2}, r_{T-3}\) (3–5 lags).
- Optional: 5-day cumulative log return.
- Optional: 5-day or 20-day rolling volatility of log returns.
- Optional: volatility-normalized return \(r_T / \sigma_{20,T}\).
- Winsorize extreme lagged log returns during training (e.g., 1st/99th percentile).

**Technical indicators (5–7 total):**
- Trend: SMA or EMA.
- Momentum: RSI and MACD.
- Volatility: ATR or Bollinger Bands.
- Volume: OBV or volume moving average.

**Sentiment features:**
- Score each headline with FinBERT (preferred) or VADER.
- Aggregate to a **daily average sentiment score** per stock.
- Optionally include a **1-day lagged sentiment score** to capture delayed reactions.

**Target construction:**
- Compute next-day log return \(r_{T+1}\).
- Binary label: \(y_{T+1}=1\) if \(r_{T+1}>0\), else 0.
- Ensure the target is never used as a feature.

### 4.3 Modeling
- **Approach**: Early fusion—concatenate technical, log-return, and sentiment features into a single feature vector.
- **Models**: XGBoost or Random Forest. Prefer XGBoost for tabular performance and feature importance.
- **Baselines** (mandatory for comparison):
  - Technical-only model (technical + log-return features).
  - Sentiment-only model.
  - Combined model.
- **Interpretability**: Report feature importance to assess whether sentiment contributes meaningfully.

### 4.4 Evaluation
- **Splitting**: Chronological train/validation/test split or `TimeSeriesSplit`. **Never shuffle** time-ordered data.
- **Primary metric**: AUC-ROC.
- **Secondary metrics**: Accuracy; optionally precision/recall at a chosen threshold.
- **Benchmarking**: Combined model must be compared against technical-only and sentiment-only baselines on the same out-of-sample period.
- **Expectations**: Be skeptical of accuracy above ~55–60%; investigate for leakage.

### 4.5 Backtest (Minimal)
- Simulate a simple strategy: go long if predicted UP, stay in cash if predicted DOWN.
- Use **out-of-sample test data only**.
- Convert log returns to simple returns for P&L: \(\text{simple return}=e^{r}-1\).
- Include **transaction costs** (e.g., 0.1% per trade) and slippage assumptions.
- Report cumulative return, Sharpe ratio, and hit rate with and without costs.

### 4.6 Guardrails & Quality Checks
- **No look-ahead bias**: Features at time \(T\) may only use information available before the close of \(T\). Explicitly verify that \(r_{T+1}\) and future news are excluded.
- **No random shuffling** in any train/test split.
- **Adjusted prices only** for return computation.
- **Survivorship bias awareness**: Note limitations of using current large caps; document as a known MVP limitation.
- **Zero-return handling**: Define direction consistently; document how zero-return days are labeled.
- **Data validation**: Check for missing days, duplicate headlines, and timestamp parsing errors.

---

## 5. High-Level Architecture

```
[Price API] ──┐
              ├──> [Data Ingestion & Alignment] ──> [Feature Store]
[News API] ───┘                                      │
                                                      ▼
                                          [Feature Engineering]
                                          - Log returns (lags)
                                          - Technical indicators
                                          - Sentiment scores
                                                      │
                                                      ▼
                                          [Model Training & Eval]
                                          - XGBoost / RF
                                          - Baselines
                                          - Chronological split
                                                      │
                                                      ▼
                                          [Backtest & Reporting]
                                          - Long/cash simulation
                                          - Costs & slippage
                                          - AUC, accuracy, Sharpe
```

**Suggested stack**: Python, `pandas`, `scikit-learn`, `xgboost`, `transformers` (FinBERT), `yfinance`, NewsAPI. Jupyter notebooks for exploration; refactor into a few `.py` scripts for the pipeline. Optional FastAPI endpoint is out of scope for the first iteration.

---

## 6. Success Metrics

**MVP is successful if:**
- The pipeline runs end-to-end reproducibly for 3–5 stocks on daily data.
- The combined model is evaluated on a strictly out-of-sample period with no detected leakage.
- The combined model shows **measurable improvement** (e.g., higher AUC-ROC) over at least one single-modality baseline, or the result is clearly documented as negative.
- A cost-aware backtest is produced and reported honestly.

**MVP is not required to:**
- Beat the market or achieve high accuracy.
- Generalize beyond the chosen stocks and time period.

---

## 7. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Look-ahead bias in news alignment | Strict timestamp rule; code-level assertions that feature timestamps < prediction time |
| Overfitting | Simple models, time-series splits, skepticism toward high accuracy |
| Sentiment model noise | Use headlines only, aggregate daily, compare against sentiment-only baseline |
| Data quality issues | Adjusted prices, missing-day checks, duplicate headline checks |
| Misleading backtest | Include transaction costs and slippage; test out-of-sample only |

---

## 8. Out of Scope / Future Iterations
- Regression on next-day log return instead of binary classification.
- Additional feature fusion (late fusion, attention over headlines).
- Sequence models (LSTM, temporal CNN) for price/return histories.
- Broader universe and sector-neutral evaluation.
- Live prediction serving and monitoring.

---

This document defines the MVP direction and main features. Implementation should follow the guardrails and evaluation discipline above; any deviation (e.g., adding deep learning or expanding the universe) should be treated as a post-MVP iteration.
