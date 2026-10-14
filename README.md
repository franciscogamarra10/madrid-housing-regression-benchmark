# Madrid Housing Price Regression

### Custom feature pipelines, Skrub, AutoGluon, and FLAML — an empirical comparison

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

End-to-end benchmark on 17,393 residential listings from Madrid. The notebook compares four feature-engineering strategies and three AutoML frameworks, and evaluates them not just on R² and RMSE but on the metrics that decide whether a model ships: artifact size, unit latency, p95 latency, batch throughput, and training time.

---

## TL;DR

- **Winner**: FLAML `DataTransformer` + default LightGBM. RMSE **25,129**, artifact **1.31 MB**, unit latency **23.5 ms**, throughput **16,445 rows/s**.
- **Runner-up**: AutoGluon `best_quality` after 10 minutes of training. RMSE **27,808**, artifact **165 MB**, unit latency **449 ms**, throughput **755 rows/s**.
- **Why the lighter pipeline wins**: the FLAML transformer compresses 43 columns into 34. AutoGluon expands the same 43 into 449, most of them auto-generated text n-grams appearing in 2–5 rows each. On 17k rows those columns are noise.
- **The counterintuitive part**: the smaller pipeline wins on accuracy *and* on every deployment axis simultaneously. It is not a speed-versus-quality trade-off.

---

## Methodology (the part that generalises)

The interesting result is not which framework won. It is the workflow that produced the answer.

1. Run two or three feature transformers on the same raw data.
2. Train the *same* default LightGBM on each output.
3. Compare hold-out RMSE, artifact size, and unit latency.
4. Deploy the smallest pipeline that meets the accuracy bar.

Skip HPO, bagging, and stacking until the feature space is settled. The gains live in the transformer, not in the model zoo.

This workflow is dataset-agnostic. The result is not. On text-rich or multi-modal data, AutoGluon's expansion can win. On small tabular sets with high-cardinality categoricals and informative missingness, a compress-and-ordinal strategy (FLAML-style) tends to dominate. The notebook exists so the numbers decide, not the brand.

---

## Key findings

### 1. The feature transformer dominates every other decision

Same LightGBM, same hyperparameters, same train/test split. Only the transformer changes.

| Feature transformer | Output columns | RMSE | Notes |
|---|---|---|---|
| Custom (`NLPFeatureExtractor2` + target encoding + binning) | ~150 | ~35,000 | High-cardinality text expands without pruning |
| AutoGluon `AutoMLPipelineFeatureGenerator` | 449 | 32,699 | 356 of those are auto text n-grams |
| Skrub `TableVectorizer` | ~60 | ~35,000 | Competitive, but slower to fit |
| **FLAML `DataTransformer`** | **34** | **25,129** | Compresses. Preserves MNAR signal. |

The winning transformer produces fewer columns than the raw input. AutoGluon produces 13× more. The 356 n-grams it generates from `title`, `raw_address`, `street_name`, and `neighborhood_id` are near-constant on this dataset — the tree spends splits filtering them before it reaches signal.

FLAML does the opposite. It encodes text columns ordinally, collapses rare categories, and leaves NaN in place. On a dataset this size, fewer clean features beat more features.

### 2. Missingness is MNAR — imputation destroys signal

30 of 32 columns with NaN correlate with the target at |effect| ≥ 0.5 SD. Missingness is predictive, not noise. LightGBM and CatBoost learn an optimal NaN split direction per feature during training; explicit `_isna` flags add a redundant column for information the tree already extracts. Imputation (mean, median, or AutoGluon's numeric fill) erases the signal entirely.

`Flagnan3` — the flag-plus-impute transformer in the custom pipeline — is the right pattern for models without native NaN support (Ridge, SVR, KNN, vanilla MLPs). For GBDTs on MNAR-dominant data, it is overhead. The custom pipeline was tested both with and without it; the version without wins.

### 3. HPO does not beat the default

Two independent HPO experiments failed to improve on `LGBMRegressor(n_estimators=1200, learning_rate=0.1)`:

- **Optuna**: 50 trials of TPE with median pruning across a 10-dimensional search space. Best OOF RMSE matched the default. The surface near the default is flat.
- **FLAML's internal search**: 60 seconds of CFO + BlendSearch. Extracting only the `DataTransformer` and pairing it with a default LightGBM reproduces FLAML `best_quality` exactly — same R², same RMSE. The search added zero.

The practical takeaway: check whether the defaults are already at the optimum before spending compute on search. On this dataset they are.

### 4. Ensembling is a rounding exercise when one model dominates

CatBoost receives 1–10% weight in the OOF-optimized blend. It moves RMSE from 32,699 to 33,053 — slightly worse. Stacking with RidgeCV converges to nearly identical results. When one model owns >90% of the weight, "ensemble" is a euphemism for "one model plus noise".

The pattern held across the other projects in this series (credit default, diamond pricing). The blend only helps when there is real model diversity to exploit, and that requires each base model to be strong on its own. On this dataset, LightGBM dominates.

### 5. AutoML frameworks are not interchangeable

AutoGluon `best_quality` and FLAML are both strong AutoML frameworks. On this dataset they produce very different artifacts from similar accuracy:

| Framework | Budget | Model selected | R² | RMSE |
|---|---|---|---|---|
| AutoGluon `best_quality` | 600 s | 106-model ensemble, 8-fold bagging, dynamic stacking | 0.9989 | 27,808 |
| FLAML | 60 s | Single LightGBM | **0.9991** | **25,129** |

The heavy framework loses on both accuracy and deployment. Its cost does not buy anything here — the extra 10 minutes of compute goes into bagging and stacking that the feature space does not need.

### 6. Neural networks did not close the gap

Four neural architectures were trained on the same preprocessed features, with early stopping and best-checkpoint restoration:

| Model | Framework | RMSE |
|---|---|---|
| TabNet | PyTorch (`pytorch-tabnet`) | ~265,000 |
| Keras MLP (4 layers, width = n_features) | TensorFlow / Keras | ~128,000 |
| Keras Tuner (RandomSearch, 15 trials) | TensorFlow / Keras | ~131,000 |
| PyTorch MLP (same topology as Keras) | PyTorch | ~149,000 |
| **LightGBM on the same features** | LightGBM | **25,129** |

The neural nets trail the boosted trees by an order of magnitude — not by a small margin. TabNet underperformed even its baseline, and the two Keras variants (fixed architecture and tuner-searched) landed within noise of each other. More architecture search did not help.

Three reasons for the gap on this dataset:

1. **Sample size.** 17k rows is small for deep learning and generous for GBDTs. Neural nets need more data to learn the interaction structure that boosted trees extract from the split mechanism.
2. **Heterogeneous features.** A mix of continuous, discrete, categorical, and text columns is the natural habitat of tree ensembles. Neural nets need careful embedding design to handle that mix, and even then the advantage usually appears only at >100k rows.
3. **Informative missingness.** Tabular neural nets have no native NaN mechanism. Whatever imputation is used before the network loses the MNAR signal that GBDTs exploit through their native NaN split direction.

This is consistent with the broader literature: on tabular regression with <50k rows and mixed feature types, boosted trees beat neural networks by a large margin, and it is not close. Adding a neural net to the ensemble did not improve the result either — the weight assigned by any meta-learner converges to zero.

The practical implication: on this class of problem, spending time on neural architecture is a net loss. The same hours invested in trying a different feature transformer return 10,000+ RMSE. The order of operations matters.

---

## Deployment trade-off

The accuracy gap between the two best pipelines is 0.0002 R². The deployment gap is two orders of magnitude.

| Metric | LGB + FLAML FE | AutoGluon `best_quality` | Ratio |
|---|---|---|---|
| R² | **0.9991** | 0.9989 | — |
| RMSE | **25,129** | 27,808 | — |
| MAE | 10,699 | **6,051** | AutoGluon wins |
| MAPE | 1.63% | **0.64%** | AutoGluon wins |
| Artifact size | **1.31 MB** | 165 MB | **126× smaller** |
| Unit latency (p50) | **23.5 ms** | 449 ms | **19× faster** |
| Unit latency (p95) | **29.4 ms** | 660 ms | **22× faster** |
| Batch throughput | **16,445 rows/s** | 755 rows/s | **22× higher** |
| Training time | **~15 s** | ~600 s | **40× faster** |

### Why each of these numbers matters in production

**1.31 MB fits everywhere — including Lambda, at a fraction of the cost.** AWS Lambda packages cap at 250 MB unzipped, so both models technically fit. But Lambda pricing scales with memory tier and execution time. A 165 MB artifact needs a larger tier to load and runs 19× longer per invocation. The 1.31 MB bundle fits in the smallest tier. For edge nodes or embedded inference, the difference is hard: megabytes of budget, not gigabytes.

**23.5 ms fits inside an interactive SLA.** At p95, 29.4 ms. Both leave headroom under a 100 ms budget that most pricing and risk APIs operate within. AutoGluon's 449 ms p50 does not — its p95 of 660 ms exceeds the timeout of any web-facing service.

**16,445 rows/s means batch scoring is not a scheduling problem.** 1M predictions complete in about one minute. 100M in about 1.7 hours. AutoGluon at 755 rows/s takes 22 minutes and 37 hours respectively. That is the difference between a nightly job that finishes before morning and one that bleeds into the next business day.

**15-second retraining makes iterative development possible.** Every feature change, every drift alarm, every threshold adjustment becomes a rebuild. With AutoGluon, each cycle costs 10 minutes of compute. With the bundle, it costs less than a coffee break.

### The choice is not accuracy versus speed

The common framing is "heavier model wins accuracy, lighter model wins deployment". On this dataset that framing is wrong. The lighter pipeline wins on R² **and** RMSE **and** size **and** latency **and** throughput **and** training time. AutoGluon retains real advantages on MAE and MAPE — those matter for scenarios where the typical error matters more than the tail — but they do not offset a 126× size difference and a 19× latency difference in most production contexts.

---

## Repository contents

| File / Folder | Purpose |
|---|---|
| `regression_madrid_housing.ipynb` | Full experiment notebook — EDA, FE, models, deployment audit |
| `loaders.py` | Data loading, column cleaning, `drop_empty_and_constant` |
| `pandas_transformers.py` | Custom transformers: NLP extractors, target encoders, outlier flags |
| `polars_transformers.py` | Polars variants for the same transformers |
| `dashboards2.py` | Classification, correlation plots, EDA diagnostics |
| `viviendas_Madrid.xlsx` | Source data (sheet `Datos_Históricos`) |
| `madrid_final_bundle.joblib` | Production artifact — FLAML transformer + LightGBM |

---

## Quick start

### Colab or local

```bash
pip install feature-engine optuna catboost skrub "flaml[automl]" "autogluon.tabular" lightgbm shap joblib
```

1. Place the four custom modules (`loaders.py`, `pandas_transformers.py`, `polars_transformers.py`, `dashboards2.py`) and `viviendas_Madrid.xlsx` in the working directory.
2. Open `regression_madrid_housing.ipynb` and run top-to-bottom (CPU is enough; ~15–20 min total, dominated by the AutoGluon 10-minute budget).
3. The production bundle is written as `madrid_final_bundle.joblib`.

### Load the production bundle

```python
import joblib

bundle = joblib.load("madrid_final_bundle.joblib")

def predict_pipeline(df, bundle):
    X = df[bundle["feature_names_in"]].copy()
    X_fe = bundle["transformer"].transform(X)
    # encode categoricals to integer codes if needed
    return bundle["model"].predict(X_fe)

preds = predict_pipeline(new_df, bundle)
```

---




## Licence

MIT. Use freely; attribution appreciated.
