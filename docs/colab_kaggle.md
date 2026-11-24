# Training on Google Colab or Kaggle

The notebook `notebooks/04_colab_kaggle_training.ipynb` runs the full pipeline on
a fresh Linux runtime:

```
prices -> Finnhub news -> 16:00 ET alignment -> FinBERT sentiment
       -> features -> chronological splits -> six models
       -> cost-aware backtest -> reports
```

It is the same code as a local run. The only differences are the installation
file (`requirements-colab.txt`) and the use of Colab/Kaggle secrets for the
Finnhub key.

---

## 1. Prerequisites

- A Finnhub API key. Free keys cover roughly the last year; the specification's
  2021-2025 window needs a paid plan.
- A GitHub URL for this repository, or the project uploaded as a zip.
- A GPU runtime is strongly recommended for FinBERT.

Files used by the notebook:

| File | Purpose |
|---|---|
| `notebooks/04_colab_kaggle_training.ipynb` | End-to-end training notebook |
| `requirements-colab.txt` | Linux/GPU dependency pins |
| `config.finnhub.yaml` | Free-tier window (`2025-12-01..2026-11-20`) |
| `config.yaml` | Specification window (`2021-01-01..2025-06-30`), paid plan |

---

## 2. Google Colab

1. Open <https://colab.research.google.com>.
2. Choose **File > Upload notebook** and select
   `notebooks/04_colab_kaggle_training.ipynb`, or open the repository in Colab
   from GitHub.
3. Choose **Runtime > Change runtime type > T4 GPU** and save.
4. Add the key:
   - Click the key icon in the left sidebar.
   - Add a secret named `FINNHUB_API_KEY`.
   - Paste the key as the value and enable notebook access.
5. In the clone cell, set `REPO_URL` to your repository.
6. Choose **Runtime > Run all**.
7. The last cell creates `stock-transformer-outputs.zip` and downloads it.

Colab disconnects after idle periods. The virtualenv and project are lost when
the runtime is recycled. Use the optional Google Drive cell to copy
`reports/` and `artifacts/` somewhere persistent.

### Private repositories

Use a short-lived token in the clone URL, for example:

```python
REPO_URL = "https://<TOKEN>@github.com/your-org/stock-transformer.git"
```

Do not save a notebook containing a real token. Alternatively, upload the
project as a zip, unzip it in `/content`, and set `PROJECT_DIR` to the unzipped
directory.

---

## 3. Kaggle

1. Create a new notebook and choose **File > Import notebook**, then upload
   `notebooks/04_colab_kaggle_training.ipynb`.
2. Open **Settings**:
   - **Accelerator:** GPU T4 x2 or P100.
   - **Internet:** On (required for GitHub, Yahoo Finance and Finnhub).
3. Add the key under **Add-ons > Secrets**:
   - Name: `FINNHUB_API_KEY`
   - Value: your key
   - Enable the secret for this notebook.
4. Set `REPO_URL` in the clone cell.
5. Choose **Run all**.
6. The zip is written to `/kaggle/working/stock-transformer-outputs.zip`.
   Download it from the notebook's **Output** tab.

Kaggle keeps `/kaggle/working` for the session. The virtualenv is under
`/kaggle/working/st-venv`, so it survives a notebook restart within the same
session, but not a new session.

---

## 4. Choose the configuration

| Config | Date range | Finnhub plan | Notes |
|---|---|---|---|
| `config.finnhub.yaml` | 2025-12-01..2026-11-20 | Free | About one year, ~48k headlines |
| `config.yaml` | 2021-01-01..2025-06-30 | Paid | Specification window |

Free-tier Finnhub requests for 2021-2025 return HTTP 200 with an empty list.
That is why the notebook defaults to `config.finnhub.yaml`. The report records
the config path and requested range, so there is no ambiguity about which window
was used.

The notebook's `STAGE` variable accepts `all`, `prices`, `news`, `sentiment`,
`features`, `train`, or `backtest`. For a first run leave it at `all`.

---

## 5. Expected runtime and resources

These are rough numbers for the free-tier window:

| Phase | GPU runtime | CPU runtime |
|---|---|---|
| Install venv + dependencies | 3-6 min | 3-6 min |
| Prices | < 1 min | < 1 min |
| Finnhub news (7-day pages) | 4-6 min | 4-6 min |
| FinBERT on ~35k unique headlines | 2-5 min | ~30 min |
| Features + six models + backtest | 1-2 min | 1-2 min |
| **Total** | **~10-20 min** | **~45-60 min** |

Disk use is dominated by PyTorch and the FinBERT weights (several GB). The
runtime's default disk is sufficient on Colab and Kaggle.

FinBERT selects the device automatically:

- `cuda` when an NVIDIA GPU is visible,
- `mps` on Apple Silicon (not relevant for Colab/Kaggle),
- `cpu` otherwise.

The report records the sentiment backend (`finbert` or `vader`) but not the
device. If CUDA is unavailable, expect the CPU timings above.

---

## 6. Outputs

After a successful run, the project directory contains:

```
reports/summary.md                         human-readable report
reports/metrics.json                       all six models + baselines
reports/splits.json                        chronological split boundaries
reports/backtest_{combined,tech_only,sent_only}.json
reports/equity_*.png
reports/feature_importance_combined_xgb.png
artifacts/models/*.json|*.joblib           six trained models
artifacts/imputation_*.json                train-fitted preprocessing
artifacts/predictions_test.parquet
data/processed/features.parquet
data/processed/dataset.parquet
data/processed/sentiment_daily.parquet
data/raw/prices.parquet
data/raw/news.parquet
```

The download cell packages `reports/`, `artifacts/`, `data/processed/`,
`data/raw/prices.parquet` and `data/raw/news.parquet`.

To copy results back into the local repository, unzip the archive over the
project root. The report and models are then available to the local notebooks
and README.

---

## 7. Resuming and stage-by-stage runs

Within one session:

1. Rerun the environment and secret cells.
2. Set `STAGE = "train"` to reuse existing features, or `STAGE = "backtest"` to
   reuse predictions.
3. Set `FORCE = False` to let the pipeline skip artifacts whose manifests are
   still current.

After a runtime restart, the venv and project are gone. Rerun the clone and
install cells first. If you copied the project to Google Drive, copy it back or
point `PROJECT_DIR` at the Drive copy.

---

## 8. Reproducibility notes

- `config.model.random_seed` controls Python, NumPy, XGBoost and RandomForest
  randomness.
- Winsorization and median imputation are fit on the training split only and
  stored in `artifacts/imputation_*.json`.
- Every artifact has a sidecar manifest with the config hash, input hashes,
  library versions and UTC timestamp.
- The Colab/Kaggle environment pins the same Python packages as the local
  project, but uses a Linux CUDA build of PyTorch. Tiny numerical differences
  are possible; the chronological split and evaluation rows are identical for
  the same input data.
- Re-running `news` can change the dataset because Finnhub's recent window
  advances. Use the cached `data/raw/news.parquet` and manifests when you need a
  fixed snapshot.

---

## 9. Security

- Store the key in Colab/Kaggle Secrets, not in a cell.
- `finnhub-api-key.txt` and `.env` are git-ignored locally.
- The pipeline scrubs `token=...` from Finnhub request errors.
- Do not commit the generated `stock-transformer-outputs.zip`; it can contain
  raw headlines and model artifacts.

---

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Set REPO_URL ...` | Edit the clone cell with your repository URL, or upload the project and set `PROJECT_DIR`. |
| `git clone` asks for credentials | Use a short-lived token in the URL or upload a zip; do not commit the token. |
| `FINNHUB_API_KEY` missing | Add the Colab/Kaggle secret and enable notebook access; or enter it at the hidden prompt. |
| HTTP 401/403 from Finnhub | The key is invalid, revoked, or not enabled for this notebook. |
| HTTP 429 from Finnhub | The rate limiter backs off automatically; wait and rerun the news stage. |
| Zero headlines | A free key cannot read 2021-2025. Use `config.finnhub.yaml`. |
| `torch.cuda.is_available()` is `False` | The runtime has no GPU, or the GPU accelerator is not enabled. FinBERT falls back to CPU. |
| CUDA out of memory | Lower `sentiment.batch_size` to 16 or 32 in the config and rerun `sentiment`. |
| `pip` dependency errors | Make sure the notebook uses the venv Python from the install cell, not the runtime Python. Delete `/content/st-venv` or `/kaggle/working/st-venv` and rerun the install cell. |
| Yahoo `CertificateVerifyError` warning | Non-fatal; yfinance continues without the cookie. |
| `No module named src` | The working directory is not the project root. Rerun the clone cell, which calls `os.chdir`. |
| Colab disconnected | Rerun clone, install and secret cells. Copy results to Drive next time. |
| Kaggle cannot reach GitHub/Finnhub | Enable **Internet** in the notebook settings. |
| Report still shows old results | The pipeline skips current artifacts. Set `FORCE = True` or delete the relevant `reports/`/`data/` files. |

---

## 11. Relationship to the local workflow

The local macOS workflow is documented in the main `README.md` and uses
`requirements.txt`. This notebook is an execution wrapper: it does not change
the model, the leakage rules or the report format. The same `src/` code runs in
both places.
