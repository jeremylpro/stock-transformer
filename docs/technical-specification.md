# Technical Specification: Stock Direction Predictor MVP

**Target:** A single-machine, batch-mode Python pipeline that produces a validated next-day direction model plus an honest backtest.

---

## 1. Scope

### 1.1 In Scope
- Batch daily pipeline: ingest prices + headlines → compute features → train models → evaluate → backtest → report.
- 3–5 configurable US large-cap tickers.
- Binary classification of next-day direction derived from next-day log return.
- Three models trained and compared: combined, technical-only, sentiment-only.
- Chronological validation only; a cost-aware long/cash backtest on the holdout period.

### 1.2 Out of Scope
- Real-time or intraday operation, order execution, brokerage integration.
- Serving/API layer, databases, cloud deployment, container orchestration.
- Deep learning models, sequence models, alternative data.
- Regression on return magnitude (interface may allow it later; not implemented).

### 1.3 Operating Assumptions
- Single developer machine, CPU only, ≤ 16 GB RAM.
- FinBERT inference on CPU acceptable for ≤ 5 tickers × ~3 years of headlines.
- Pipeline runs manually or via a daily cron job; no concurrency requirements.
- "Market close" is defined as **16:00 ET** on each trading day.

---

## 2. Technology Stack

| Layer | Component | Version | Rationale |
|---|---|---|---|
| Runtime | Python | 3.10+ | FinBERT/torch compatibility |
| Dataframes | `pandas`, `numpy` | 2.2.x, 1.26.x | — |
| Price data | `yfinance` | 0.2.36 | Free, no key, adjusted OHLCV |
| Price fallback | Alpha Vantage free tier | — | 25 req/day if `yfinance` fails |
| News data | Finnhub `/company-news` | free tier | 60 calls/min; raw headlines |
| Sentiment | `transformers` + `torch`, model `ProsusAI/finbert` | 4.47.x, 2.5.x | FinBERT > VADER on financial text |
| Sentiment baseline | `vaderSentiment` | latest | Cheap sanity check |
| Technical indicators | `ta` | 0.11.x | RSI/MACD/ATR/Bollinger/OBV |
| Modeling | `scikit-learn`, `xgboost` | 1.4.x, latest | Tabular direction prediction |
| Backtesting | `backtesting.py` | latest | Next-bar-open fills by default |
| Config/secrets | `python-dotenv` | 1.0.x | API key handling |
| Testing | `pytest` | 8.x | Leakage and alignment tests |
| Exploration | Jupyter | — | Stage-level prototyping |

**Explicitly excluded:** `backtrader` (unmaintained since 2023), `vectorbt` (same-bar close fill default), any serving framework.

---

## 3. System Architecture

```
                ┌──────────────────────┐
                │   config.yaml         │
                │   .env (API keys)     │
                └──────────┬───────────┘
                           │
        ┌──────────────────┴───────────────────┐
        │                                      │
┌───────▼────────┐                   ┌─────────▼────────┐
│ data_prices.py │                   │  data_news.py    │
│  yfinance →     │                   │ Finnhub →        │
│  adjusted OHLCV │                   │ raw headlines    │
└───────┬────────┘                   └─────────┬────────┘
        │                                      │
        │                            ┌─────────▼────────┐
        │                            │  sentiment.py    │
        │                            │ FinBERT scoring  │
        │                            └─────────┬────────┘
        │                                      │
        └──────────────┬───────────────────────┘
                       ▼
              ┌─────────────────┐
              │  alignment.py   │  ← leakage-critical boundary
              │  daily join,    │
              │  16:00 ET cutoff│
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │  features.py    │  returns, TA, sentiment agg
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │  labels.py      │  y = 1{r_{T+1} > 0}
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │  splits.py      │  chronological train/val/test
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │  model.py       │  XGB + RF × 3 feature sets
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │  evaluate.py    │  AUC, acc, Brier, importances
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │  backtest.py    │  long/cash + costs
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │  reports/       │  metrics.json, plots, summary.md
              └─────────────────┘
```

All intermediate artifacts (raw data, features, labels, model files) are written to `data/` and `artifacts/` as Parquet/JSON so each stage is independently re-runnable.

---

## 4. Configuration

Single `config.yaml`; secrets in `.env`.

```yaml
universe:
  tickers: ["AAPL", "MSFT", "GOOGL", "AMZN"]
  benchmark: "SPY"

date_range:
  start: "2021-01-01"
  end: "2025-06-30"

data:
  price_source: "yfinance"          # yfinance | alphavantage
  news_source: "finnhub"
  market_tz: "America/New_York"
  close_time: "16:00"
  cache_dir: "data/raw"

sentiment:
  model: "ProsusAI/finbert"
  batch_size: 32
  max_tokens: 64
  include_vader_baseline: true

features:
  return_lags: 5
  vol_windows: [5, 20]
  winsorize_quantiles: [0.01, 0.99]
  indicators:
    sma_windows: [10, 50]
    ema_window: 20
    rsi_window: 14
    macd: {fast: 12, slow: 26, signal: 9}
    atr_window: 14
    bb_window: 20
    obv_window: 20
  sentiment:
    include_lag1: true
    min_headlines_per_day: 1

splits:
  method: "chronological"
  train_ratio: 0.60
  val_ratio: 0.20
  test_ratio: 0.20

model:
  xgb:
    n_estimators: 500
    max_depth: 4
    learning_rate: 0.03
    subsample: 0.8
    colsample_bytree: 0.8
    early_stopping_rounds: 50
  rf:
    n_estimators: 500
    max_depth: 6
    min_samples_leaf: 20
  random_seed: 42

backtest:
  entry_threshold: 0.55       # go long if p_up >= threshold
  commission_bps: 10          # 0.10% per trade
  slippage_bps: 5
  fill: "next_bar_open"

reporting:
  output_dir: "reports"
```

---

## 5. Data Specifications

### 5.1 Price Data

**Source:** `yfinance.download(tickers, start, end, auto_adjust=False)`; use the `Adj Close` column for all returns and keep raw `Close` for reference.

**Daily price record (canonical schema):**

| Field | Type | Notes |
|---|---|---|
| `ticker` | str | |
| `date` | date | Trading day, ET |
| `open`, `high`, `low`, `close` | float | Raw prices |
| `adj_close` | float | Split/dividend adjusted; **source of truth for returns** |
| `volume` | int | |

**Validation rules:**
- Reject rows with `adj_close <= 0` or null OHLC.
- Warn if a ticker has > 5% missing trading days versus the benchmark calendar; forward-fill is **not** permitted for price (drop the day instead, and record the gap).
- Duplicate `(ticker, date)` rows are an error.

### 5.2 News Data

**Source:** Finnhub `/company-news?symbol={T}&from={d0}&to={d1}`.

**Raw headline record:**

| Field | Type | Notes |
|---|---|---|
| `ticker` | str | |
| `headline` | str | Title only for MVP |
| `published_utc` | datetime (UTC) | From API `datetime` epoch |
| `source` | str | |
| `url_hash` | str | Dedup key |

**Validation rules:**
- Parse epoch → UTC → convert to `America/New_York`.
- Deduplicate on `(ticker, url_hash)`; if `url_hash` missing, dedupe on normalized headline text.
- Drop empty or whitespace-only headlines.
- Headlines with `published_utc` after the dataset end date are discarded.

### 5.3 The Alignment Rule (leakage-critical)

> A headline is eligible for the feature set of trading day *T* if and only if its publication timestamp is **strictly before 16:00 ET on day *T***.

Formally, with `t_pub` in ET and `t_close(T) = date(T) 16:00 ET`:

```
eligible(T, h) = t_pub(h) < t_close(T)
```

Implementation notes:
- Headlines published between 16:00 ET and midnight belong to the **next** trading day's feature set, not the current one.
- Weekends/holidays: headlines published Saturday/Sunday are assigned to the next trading day.
- This rule must be implemented in exactly one function (`alignment.eligible_headlines_for_day`) and unit-tested; no other module may re-implement it.

---

## 6. Module Specifications

Each module exposes pure functions over DataFrames plus a `main()`/CLI entry point, and writes its output artifact to disk.

### 6.1 `data_prices.py`

```python
def fetch_prices(tickers: list[str], start: str, end: str,
                 cache_dir: str) -> pd.DataFrame:
    """Returns canonical price schema. Caches per-ticker Parquet;
    refetches only the missing tail."""

def validate_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Applies §5.1 validation rules; raises PriceValidationError."""
```

**Artifact:** `data/raw/prices.parquet`

### 6.2 `data_news.py`

```python
def fetch_news(tickers: list[str], start: str, end: str,
               api_key: str, cache_dir: str) -> pd.DataFrame:
    """Finnhub pull with per-ticker pagination and caching.
    Respects 60 req/min via a simple token-bucket limiter."""

def validate_news(df: pd.DataFrame) -> pd.DataFrame:
    """Applies §5.2 validation rules; dedupes; converts to ET."""
```

**Artifact:** `data/raw/news.parquet`

### 6.3 `sentiment.py`

```python
def score_headlines(headlines: list[str],
                    model_name: str = "ProsusAI/finbert",
                    batch_size: int = 32) -> pd.DataFrame:
    """Returns columns: headline_idx, prob_positive, prob_negative,
    prob_neutral, sent_score where
    sent_score = prob_positive - prob_negative  (range [-1, 1])."""

def daily_sentiment(news_df: pd.DataFrame,
                    prices_df: pd.DataFrame) -> pd.DataFrame:
    """Applies §5.3 alignment, then aggregates per (ticker, trading_day):
    sent_mean, sent_std, sent_count, and optionally sent_lag1.
    Days with 0 eligible headlines produce sent_count=0 and NaN mean
    (NOT 0.0) so the model can distinguish 'neutral' from 'no news'."""
```

**Design note:** `sent_score = P(positive) − P(negative)` is used because it is signed, bounded, and interpretable. Neutral probability is retained for diagnostics but not fed to the model.

**Artifact:** `data/processed/sentiment_daily.parquet`

### 6.4 `features.py`

```python
def build_features(prices: pd.DataFrame,
                   sentiment_daily: pd.DataFrame,
                   cfg: FeatureConfig) -> pd.DataFrame:
    """Returns one row per (ticker, date) with all features from §7.
    Guarantees no column depends on information after 16:00 ET of `date`."""
```

### 6.5 `labels.py`

```python
def build_labels(prices: pd.DataFrame) -> pd.DataFrame:
    """Returns columns: ticker, date, log_ret_next, y_next_up.
    log_ret_next = ln(adj_close[T+1] / adj_close[T]);
    y_next_up = 1 if log_ret_next > 0 else 0.
    The final row per ticker has NaN target and is dropped."""
```

### 6.6 `splits.py`

```python
def chronological_split(df: pd.DataFrame, cfg: SplitConfig
                        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits on unique dates (not rows) so all tickers for a date
    stay in the same split. Returns (train, val, test)."""
```

**Rules:**
- Split boundaries are computed on **sorted unique trading dates** across the universe, then applied to all rows.
- An **embargo** of 1 trading day is inserted between splits to prevent target/feature overlap bleeding across the boundary.
- Random shuffling anywhere in the training path is a hard error (`pytest` enforces).

### 6.7 `model.py`

```python
FEATURE_SETS = {
    "combined":   ["ret_*", "ta_*", "sent_*"],
    "tech_only":  ["ret_*", "ta_*"],
    "sent_only":  ["sent_*"],
}

def train_model(X_train, y_train, X_val, y_val,
                algo: str, cfg: ModelConfig) -> TrainedModel:
    """XGBoost with early stopping on validation AUC, or RandomForest.
    NaN handling: XGBoost native; RF via median imputation fit on train only."""

def predict_proba(model, X) -> np.ndarray:
    """Returns P(y=1)."""
```

Six models are trained: `{combined, tech_only, sent_only} × {xgb, rf}`.

**Artifacts:** `artifacts/models/{featureset}_{algo}.json` (XGB) or `.joblib` (RF), plus `artifacts/imputation_{featureset}.json`.

### 6.8 `evaluate.py`

```python
def evaluate(y_true, p_up) -> dict:
    """Returns {"auc", "accuracy", "brier", "log_loss",
                "n_samples", "positive_rate"}."""

def feature_importance(model, feature_names) -> pd.DataFrame:
    """Gain importance for XGB; impurity importance for RF."""
```

**Required comparisons (all on the same test rows):**

| Comparison | Purpose |
|---|---|
| combined vs. tech_only | Does sentiment add value? |
| combined vs. sent_only | Do technicals add value? |
| any model vs. majority-class baseline | Is the model better than always guessing UP? |
| any model vs. random baseline (AUC = 0.5) | Sanity floor |

**Artifact:** `reports/metrics.json`

### 6.9 `backtest.py`

```python
def run_backtest(test_df: pd.DataFrame, p_up: np.ndarray,
                 cfg: BacktestConfig) -> BacktestResult:
    """Long/cash rule on a per-ticker basis, then equal-weight
    portfolio aggregation. Uses backtesting.py with next-bar-open fills."""
```

**Strategy rules (exact):**
- Signal generated at close of day *T* using `p_up(T)`.
- If `p_up >= entry_threshold`: target position = long at **open of T+1**.
- Else: target position = cash at **open of T+1**.
- Costs: `commission_bps` + `slippage_bps` applied per position change.
- No leverage, no shorting, no overlapping positions per ticker.
- Convert log returns to simple returns only in the P&L layer: `simple = exp(r) - 1`.

**Reported metrics:** cumulative return (gross and net), annualized Sharpe (net), hit rate, number of trades, max drawdown, and the same for a buy-and-hold benchmark.

**Artifact:** `reports/backtest_{featureset}.json` + equity curve PNG.

### 6.10 `pipeline.py`

```python
def run(config_path: str) -> None:
    """Executes stages in order, skipping stages whose artifacts are
    newer than their inputs (unless --force)."""
```

CLI: `python -m src.pipeline --config config.yaml [--stage prices|news|sentiment|features|train|backtest|all] [--force]`

---

## 7. Feature Definitions

All features are computed per ticker on the trading-day index. `r_t = ln(adj_close_t / adj_close_{t-1})`.

### 7.1 Return features (`ret_*`)

| Name | Definition |
|---|---|
| `ret_lag_1` … `ret_lag_5` | `r_{t}`, `r_{t-1}`, …, `r_{t-4}` |
| `ret_cum_5` | `Σ_{i=0..4} r_{t-i}` |
| `vol_5`, `vol_20` | Rolling std of `r` over 5 / 20 days |
| `ret_z_20` | `r_t / vol_20` (guard: `vol_20 > 0`) |

### 7.2 Technical features (`ta_*`)

| Name | Definition |
|---|---|
| `sma_ratio_10`, `sma_ratio_50` | `adj_close_t / SMA_n(t) − 1` |
| `ema_ratio_20` | `adj_close_t / EMA_20(t) − 1` |
| `rsi_14` | `ta.momentum.RSIIndicator(14)` |
| `macd_line`, `macd_signal`, `macd_hist` | `ta.trend.MACD(12, 26, 9)` |
| `atr_norm_14` | `ATR(14) / adj_close_t` |
| `bb_pct_b_20` | `(close − lower) / (upper − lower)` |
| `obv_z_20` | z-score of `OBV` over 20 days |
| `vol_z_20` | z-score of `volume` over 20 days |

Ratios and z-scores are preferred over raw levels so features are comparable across tickers and price regimes.

### 7.3 Sentiment features (`sent_*`)

| Name | Definition |
|---|---|
| `sent_mean` | Mean `sent_score` over eligible headlines for day *T*; NaN if `sent_count == 0` |
| `sent_std` | Std of `sent_score`; NaN if `sent_count < 2` |
| `sent_count` | Number of eligible headlines (`log1p`-transformed for modeling) |
| `sent_lag1` | `sent_mean` from the previous trading day |
| `sent_vader_mean` | VADER compound mean (baseline comparison only) |

### 7.4 Target (never a feature)

| Name | Definition |
|---|---|
| `log_ret_next` | `ln(adj_close_{T+1} / adj_close_T)` |
| `y_next_up` | `1 if log_ret_next > 0 else 0` |

### 7.5 Preprocessing

- Winsorize **features only** (not the target) at `[0.01, 0.99]` quantiles fit on the **training set** and applied to val/test.
- Median imputation fit on train only, applied to val/test. `sent_count == 0` is preserved via the imputed value plus the count feature, so "no news" remains distinguishable from "neutral news" where the model has signal to do so.
- No scaling required for tree models; skip StandardScaler entirely.

---

## 8. Leakage Guardrails (enforced, not aspirational)

These are implemented as runtime assertions and `pytest` tests:

| ID | Guardrail | Enforcement |
|---|---|---|
| G1 | No feature may use data with timestamp ≥ `t_close(T)` | `alignment.eligible_headlines_for_day` is the only news-filtering implementation; unit test constructs a headline at 15:59 ET and 16:01 ET and asserts only the former is included |
| G2 | Target column excluded from feature matrix | `build_features` output columns are checked against a forbidden-prefix list (`y_`, `log_ret_next`, `log_ret_fwd`) |
| G3 | Chronological split only | `splits.py` exposes no shuffle parameter; a test asserts split date ranges are strictly ordered and non-overlapping |
| G4 | Fitted transforms (winsorize, impute) never see val/test | Functions accept `fit_df` explicitly; test asserts test-set quantiles differ from those applied |
| G5 | Adjusted close used for all returns | `build_labels` asserts the source column is named `adj_close` |
| G6 | Backtest fills at next bar's open | `backtesting.py` configured with `trade_on_close=False`; asserted in config test |
| G7 | No NaN target leakage through `dropna` | Rows dropped for missing targets are dropped **before** splitting, and the row-count drop is logged |

---

## 9. Evaluation Protocol

1. Run `splits.chronological_split` on the full panel; record split date ranges in `reports/splits.json`.
2. Fit winsorization and imputation on train only.
3. Train all six models with a fixed `random_seed`.
4. Score once on val (for XGB early stopping only) and **once** on test.
5. Compute metrics for all six models plus majority-class and random baselines.
6. Run the backtest for the three feature sets using the XGB predictions (RF backtest optional).
7. Emit `reports/summary.md` containing: split ranges, metrics table, top-15 feature importances for `combined_xgb`, and the net-of-cost backtest table.

**Interpretation policy (documented in the report):**
- Test AUC in `[0.50, 0.53]` → no evidence of signal; report as negative result.
- Test AUC in `(0.53, 0.58]` → plausible weak signal; check stability across tickers and feature importance before drawing conclusions.
- Test AUC `> 0.60` → treat as a **red flag**, not a win; audit §8 guardrails before proceeding.

---

## 10. Testing Plan

| Test | Type | Asserts |
|---|---|---|
| `test_price_validation` | unit | Bad rows rejected; adjusted close used |
| `test_news_dedup_and_tz` | unit | Duplicates removed; UTC→ET conversion correct across a DST boundary |
| `test_alignment_cutoff` | unit | 15:59 ET included, 16:01 ET excluded (G1) |
| `test_weekend_news_assignment` | unit | Saturday headlines assigned to Monday |
| `test_no_target_in_features` | unit | Forbidden columns absent (G2) |
| `test_splits_are_chronological` | unit | Ordered, disjoint, embargo applied (G3) |
| `test_transforms_fit_on_train` | unit | Val/test do not influence fit (G4) |
| `test_labels_use_adj_close` | unit | Correct column, correct formula, zero-return convention (G5) |
| `test_backtest_fill_mode` | config | `trade_on_close=False` (G6) |
| `test_pipeline_smoke` | integration | Tiny fixture (5 tickers, 60 days) runs end-to-end and produces all artifacts |
| `test_sentiment_null_vs_zero` | unit | No-news day yields NaN mean + count 0, not 0.0 |

**Fixtures:** a 3-ticker, 90-day synthetic dataset with hand-constructed leakage traps (a future headline, a future price spike) lives in `tests/fixtures/`.

---

## 11. Reproducibility & Logging

- All randomness seeded from `config.model.random_seed`.
- Every artifact stores a `manifest.json` with: config hash, input artifact hashes, library versions, timestamp.
- Logging via stdlib `logging` to `logs/pipeline_{date}.log`, with INFO for stage boundaries and WARNING for every validation drop (with counts).
- `reports/summary.md` is the single human-readable deliverable.

---

## 12. Directory Structure

```
project/
├── config.yaml
├── .env                      # FINNHUB_API_KEY, ALPHAVANTAGE_API_KEY
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── config.py             # pydantic/dataclass config loader + validation
│   ├── data_prices.py
│   ├── data_news.py
│   ├── sentiment.py
│   ├── alignment.py          # sole owner of the 16:00 ET rule
│   ├── features.py
│   ├── labels.py
│   ├── splits.py
│   ├── model.py
│   ├── evaluate.py
│   ├── backtest.py
│   ├── reporting.py
│   └── pipeline.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_sentiment_check.ipynb
│   └── 03_feature_diagnostics.ipynb
├── tests/
│   ├── fixtures/
│   └── test_*.py
├── data/
│   ├── raw/
│   └── processed/
├── artifacts/
│   ├── models/
│   └── imputation/
└── reports/
```

---

## 13. Dependencies (`requirements.txt`, pinned)

```
pandas==2.2.0
numpy==1.26.3
yfinance==0.2.36
requests==2.31.0
python-dotenv==1.0.1
ta==0.11.0
transformers==4.47.1
torch==2.5.1
vaderSentiment==3.3.2
scikit-learn==1.4.0
xgboost==2.0.3
backtesting==0.6.0
pyyaml==6.0.1
pytest==8.0.0
matplotlib==3.8.2
pyarrow==15.0.0
```

---

## 14. Acceptance Criteria

The MVP is technically complete when all of the following hold:

1. `python -m src.pipeline --config config.yaml` runs end-to-end on a clean machine from a fresh clone (given `.env` keys) without manual intervention.
2. All tests in §10 pass.
3. `reports/metrics.json` contains metrics for all six models plus both baselines on the identical test split.
4. `reports/summary.md` states, in one explicit sentence, whether the combined model outperformed both single-modality baselines on test AUC, and by how much.
5. A cost-aware backtest result exists for all three feature sets on test data only.
6. The leakage audit in §8 is satisfied by passing tests, not by inspection.
7. Results are reported honestly, including the negative-result case; no accuracy claims above 60% are accepted without a written leakage audit.

---

## 15. Known Limitations (to be stated in the report)

- Small universe, single market, current large caps → survivorship and selection bias.
- Headline-only sentiment discards article body, entity disambiguation, and source credibility.
- Finnhub free tier may have headline coverage gaps; `sent_count` captures availability but not coverage quality.
- Single chronological split yields high-variance AUC estimates; bootstrapped confidence intervals are a documented post-MVP item.
- No market-regime features (VIX, rates, sector returns); no cross-sectional normalization across tickers.
- Backtest is single-asset long/cash, equal-weight; it is an **evaluation aid**, not a tradeable strategy.

---

## 16. Implementation Milestones

| Milestone | Deliverable | Definition of done |
|---|---|---|
| M1 | Data layer | `prices.parquet` + `news.parquet` + validation tests pass |
| M2 | Alignment + sentiment | `sentiment_daily.parquet`, G1 test green |
| M3 | Features + labels | Full feature matrix, G2/G5 tests green |
| M4 | Splits + models | Six models trained, `metrics.json` populated |
| M5 | Backtest + report | `summary.md` with honest comparison and limitations |
| M6 | Hardening | Integration smoke test, full leakage suite green, README complete |

Each milestone is independently reviewable; no milestone may begin before the previous milestone's tests pass.
