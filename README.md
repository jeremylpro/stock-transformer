# Stock Direction Predictor MVP

A single-machine, batch-mode Python pipeline that predicts next-day UP/DOWN
direction for a small universe of liquid US large caps by fusing technical
indicators, lagged log returns and news sentiment, then evaluates the result
honestly with chronological splits and a cost-aware long/cash backtest.

This implementation follows `docs/product-design.md` and
`docs/technical-specification.md`.

---

## What is implemented

- **Prices** (`src/data_prices.py`): yfinance adjusted OHLCV with per-ticker
  Parquet caching and tail refetch; Alpha Vantage fallback; strict validation
  (positive `adj_close`, no null OHLC, no duplicate `(ticker, date)`, missing-day
  warnings, no forward filling).
- **News** (`src/data_news.py`): Finnhub `/company-news` with short-window
  pagination (7 days, because Finnhub caps a response at 250 articles), a
  token-bucket rate limiter, per-ticker caching, timestamp parsing and
  deduplication. API tokens are scrubbed from error text. A `local` source
  exists for tests and offline demos.
- **Alignment** (`src/alignment.py`): the single implementation of the
  16:00 ET cutoff. A headline belongs to the first trading day whose close is
  strictly after publication, so after-close and weekend stories roll forward.
- **Sentiment** (`src/sentiment.py`): FinBERT (`ProsusAI/finbert`) preferred,
  VADER fallback and baseline. `sent_score = P(positive) - P(negative)`.
  No-news days keep `sent_mean = NaN` and `sent_count = 0` (never 0.0).
- **Features** (`src/features.py`): 5 lagged log returns, 5-day cumulative
  return, 5/20-day volatility, volatility-normalized return, SMA/EMA ratios,
  RSI, MACD, normalized ATR, Bollinger %B, OBV z-score and volume z-score.
- **Labels** (`src/labels.py`): `log_ret_next = ln(adj_close[T+1] / adj_close[T])`,
  `y_next_up = 1` only when the next log return is strictly positive.
- **Splits** (`src/splits.py`): chronological train/validation/test split on
  unique dates with a one-trading-day embargo. No shuffle parameter exists.
- **Models** (`src/model.py`): six models — `{combined, tech_only, sent_only} ×
  {XGBoost, RandomForest}`. Winsorization and median imputation are fit on train
  only and persisted to `artifacts/imputation_*.json`.
- **Evaluation** (`src/evaluate.py`): AUC, accuracy, Brier, log loss, sample
  count, positive rate, feature importance, majority-class and random baselines.
- **Backtest** (`src/backtest.py`): per-ticker long/cash with `backtesting.py`,
  next-bar-open fills (`trade_on_close=False`), commission + slippage on every
  position change, equal-weight portfolio aggregation, buy-and-hold benchmark.
- **Pipeline** (`src/pipeline.py`): per-stage CLI, artifact manifests, logging,
  and skip-if-current behavior (`--force` rebuilds).
- **Report** (`src/reporting.py`): `reports/summary.md`, metrics JSON, split
  ranges, top-15 importances, net-of-cost backtest table and known limitations.

All 56 tests in `tests/` pass, including the leakage suite from §10.

---

## Quickstart

Requires Python 3.12 (the pinned stack predates 3.13) on macOS, Linux or Windows.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and set FINNHUB_API_KEY (and optionally ALPHAVANTAGE_API_KEY)

python -m src.pipeline --config config.yaml
```

`config.yaml` keeps the specification's 2021-2025 window, which needs a paid
Finnhub plan. With a free-tier key, use `config.finnhub.yaml` instead (see
"Real Finnhub run" below); it covers the most recent year.

The first run downloads prices, headlines and the FinBERT weights (~440 MB),
then writes:

```
data/raw/prices.parquet
data/raw/news.parquet
data/processed/sentiment_daily.parquet
data/processed/features.parquet
data/processed/dataset.parquet
artifacts/models/*.json|*.joblib
artifacts/imputation_*.json
artifacts/predictions_test.parquet
reports/splits.json
reports/metrics.json
reports/backtest_{combined,tech_only,sent_only}.json
reports/equity_*.png
reports/feature_importance_combined_xgb.png
reports/summary.md
logs/pipeline_<date>.log
```

Every artifact has a sidecar `*.manifest.json` with the config hash, input
hashes, library versions and timestamp.

### Stage-by-stage

```bash
python -m src.pipeline --config config.yaml --stage prices
python -m src.pipeline --config config.yaml --stage news
python -m src.pipeline --config config.yaml --stage sentiment
python -m src.pipeline --config config.yaml --stage features
python -m src.pipeline --config config.yaml --stage train
python -m src.pipeline --config config.yaml --stage backtest
python -m src.pipeline --config config.yaml --force   # rebuild everything
```

Stages are skipped when their artifact manifest still matches the config and
inputs. `--force` ignores that check.

### Tests

```bash
pytest -q
```

The integration smoke test runs a 5-ticker, 60-day fixture through every stage
with VADER, so it needs no network access.

---

## Configuration

The main knobs mirror the technical specification: universe, date range, data
sources, sentiment backend, feature windows, split ratios/embargo, model
hyperparameters, backtest costs and report directory. See `config.yaml`.

Additional, backwards-compatible options:

| Key | Values | Purpose |
|---|---|---|
| `data.price_source` | `yfinance`, `alphavantage`, `local` | `local` reads `data.price_file` (tests/demos) |
| `data.news_source` | `finnhub`, `local` | `local` reads `data.news_file` |
| `sentiment.backend` | `auto`, `finbert`, `vader` | `auto` tries FinBERT and falls back to VADER |
| `splits.embargo_days` | integer, default `1` | Gap between adjacent splits |

`ALPHAVANTAGE_API_KEY` is only used when `price_source` is `alphavantage` or as a
fallback if yfinance fails. `FINNHUB_API_KEY` is required for `news_source:
finnhub`.

### Demo mode (no Finnhub key)

`config.demo.yaml` uses **real yfinance prices** but **synthetic headlines**
generated by `scripts/generate_demo_news.py`. It exercises every code path,
including FinBERT, without an API key:

```bash
python -m src.pipeline --config config.demo.yaml --stage prices
python scripts/generate_demo_news.py
python -m src.pipeline --config config.demo.yaml
```

The generated `reports/summary.md` carries an explicit **OFFLINE / DEMO DATA
WARNING**. Do not interpret its metrics as market evidence: the synthetic
sentiment is independent of returns by construction.

### Real Finnhub run (free tier)

Put the key in a git-ignored `finnhub-api-key.txt` (one line, no quotes) or in
`.env` as `FINNHUB_API_KEY=...`, then:

```bash
export FINNHUB_API_KEY="$(tr -d '\r\n' < finnhub-api-key.txt)"
python -m src.pipeline --config config.finnhub.yaml --force
```

The free tier only returns roughly the last year of company news; requests for
2021–2025 return HTTP 200 with an empty list. `config.yaml` deliberately keeps
the specification's 2021-01-01..2025-06-30 window, so a free-tier key cannot
populate it. `config.finnhub.yaml` is identical except for
`date_range: 2025-12-01..2026-11-20`, which is fully covered by the free tier.
The API also caps a company-news response at 250 articles, so ingestion pages in
7-day windows; a 90-day window silently loses all but the last few days.

### Results included in this repository

The checked-in `reports/` directory is the **real-news free-tier run** from
`config.finnhub.yaml`:

- 47,996 deduplicated Finnhub headlines, scored with FinBERT (`auto` resolved to
  FinBERT); 17.8% of ticker-days have no eligible headline and keep `sent_mean`
  as NaN.
- 980 labeled rows, split chronologically 588 / 196 / 188 (train / validation /
  test) with a one-day embargo.
- Test AUC: combined XGB **0.4812**, technical-only XGB 0.4668, sentiment-only
  XGB 0.5264. The combined model did not beat both single-modality baselines,
  so the report states a negative result.
- No model reached the 0.55 entry threshold, so all three long/cash backtests
  stayed in cash. The report records zero trades rather than hiding it.

This is a small, recent, single-split sample, so the AUC estimates are noisy.
The synthetic demo (`config.demo.yaml`) remains available for offline plumbing
tests and is clearly marked with an `OFFLINE / DEMO DATA WARNING`.

### Cloud training (Google Colab / Kaggle)

`notebooks/04_colab_kaggle_training.ipynb` runs the full pipeline on a fresh
Linux GPU runtime, using `requirements-colab.txt` and Colab/Kaggle secrets for
the Finnhub key. FinBERT automatically uses CUDA when available. See
`docs/colab_kaggle.md` for setup, runtime estimates, resuming and
troubleshooting.

---

## Leakage guardrails

| ID | Guardrail | Implementation |
|---|---|---|
| G1 | No feature uses data at/after 16:00 ET of day T | `alignment.eligible_headlines_for_day`; every assignment is asserted against the close |
| G2 | Targets never enter the feature matrix | Prefix assertion in `build_features` |
| G3 | Chronological splits only | `chronological_split`, no shuffle parameter, ordered/disjoint tests |
| G4 | Transforms fit on train only | `FeatureTransformer.fit(train, ...)`; test data is never passed to `fit` |
| G5 | Returns always use `adj_close` | `build_labels` asserts the column name |
| G6 | Backtest fills at next bar open | `trade_on_close=False`; asserted in config and trade-entry tests |
| G7 | Missing targets dropped before splitting | Count logged by the pipeline |

The news rule has a dedicated test that includes 15:59 ET and excludes 16:00 and
16:01 ET, plus weekend roll-forward, DST conversion and far-future headline
tests. A fixture with a future price spike is used to prove that features before
the spike are byte-for-byte unchanged when future prices are removed.

---

## Evaluation and interpretation policy

- Primary metric: test AUC-ROC on a strictly held-out chronological test period.
- Secondary: accuracy at 0.5, Brier score, log loss.
- Required comparisons: combined vs. technical-only, combined vs. sentiment-only,
  plus majority-class and random (AUC floor 0.5) baselines.
- Test AUC in `[0.50, 0.53]` → no evidence of signal; report as negative.
- Test AUC in `(0.53, 0.58]` → plausible weak signal; check stability.
- Test AUC above `0.60` → treat as a leakage red flag, not a win.

The current checked-in run is a negative result (combined XGB test AUC 0.4812,
sentiment-only XGB 0.5264), and the report says so explicitly. This is the
honest outcome for a one-year, four-ticker sample with headline-only sentiment.

---

## Known limitations

- Small universe of current large caps → survivorship and selection bias.
- Headline-only sentiment discards article bodies, entity resolution and source
  credibility.
- Finnhub's free tier covers only about one year of history and caps each
  response at 250 articles; the real run therefore uses a shorter window and a
  188-row test set. The specification's 2021-2025 window needs a paid Finnhub
  plan.
- One chronological split gives a high-variance AUC estimate; bootstrap
  confidence intervals are post-MVP.
- No market-regime or cross-sectional features.
- The long/cash equal-weight backtest is an evaluation aid, not a tradeable
  strategy. At the default `entry_threshold: 0.55`, a model may produce no entry
  signals at all; the report records that as zero trades rather than hiding it.

---

## Stack deviations from §13 (Intel Mac)

The specification pins versions that are not installable on the target machine
(Intel macOS, Python 3.12). The deviations are limited to packaging:

| Spec pin | Installed | Reason |
|---|---|---|
| `yfinance==0.2.36` | `1.8.0` | 0.2.36 cannot authenticate against Yahoo's current endpoints |
| `torch==2.5.1` | `2.2.2` | 2.5.x has no macOS x86_64 wheels; 2.2.2 is the last one, and FinBERT works |
| `requests==2.31.0` | `2.34.2` | required by current yfinance |

Everything else follows the specified stack (`pandas 2.2`, `numpy 1.26`,
`scikit-learn 1.4`, `xgboost 2.0.3`, `backtesting 0.6.0`, `ta 0.11`, FinBERT via
`transformers 4.47.1`).

If `torch` or `transformers` is unavailable, `sentiment.backend: auto` falls
back to VADER and the report records the backend that actually ran.

---

## Directory map

```
config.yaml                     main configuration (Finnhub, spec window)
config.finnhub.yaml             free-tier real run (last ~1 year)
config.demo.yaml                real prices + synthetic headlines
src/
  config.py                     typed YAML loader and validation
  data_prices.py                yfinance/Alpha Vantage ingestion + validation
  data_news.py                  Finnhub ingestion + validation
  sentiment.py                  FinBERT/VADER scoring and daily aggregation
  alignment.py                  sole owner of the 16:00 ET rule
  features.py                   return/technical/sentiment features
  labels.py                     next-day direction labels
  splits.py                     chronological split + embargo
  model.py                      preprocessing, XGB/RF training
  evaluate.py                   metrics, baselines, importance
  backtest.py                   next-bar-open long/cash backtest
  reporting.py                  summary.md and plots
  pipeline.py                   stage orchestration and manifests
tests/                          56 unit/integration/leakage/notebook tests
tests/fixtures/                 deterministic leakage and smoke datasets
scripts/generate_fixtures.py    regenerate test fixtures
scripts/generate_demo_news.py   synthetic demo headlines
notebooks/                      exploration notebooks
```
