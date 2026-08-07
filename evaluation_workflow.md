# Evaluation Workflow

This document describes the evaluation pipeline of the DeepMechanisticModels codebase. The pipeline runs after model training and produces per-cell-line predicted trajectories, latent embeddings, inferred parameters, sensitivity analyses, and aggregated RMSE summaries. Execution order is orchestrated by the `Snakefile`.

---

## Pipeline Overview

```
process_data ──► select_features ──► train (estimate_parameters)
                                          │
pretrain_per_sample ─┐                    ▼
pretrain_average ────┤             evaluate_training
                     │                    │
                     ▼                    │
            evaluate_references           │
                     │                    │
                     │    evaluate_regressors
                     │         │          │
                     ▼         ▼          ▼
                       evaluate_all
                           │
                           ▼
                       report_all
```

---

## Scripts in Execution Order

### 1. `evaluate_training.py`

**Purpose:** Evaluate a single trained DMM for one hyperparameter configuration (context, features, samples, job).

**Invoked by:** Snakefile rule `evaluate_training`; called once per `{scan_attributes}` combination.

**Inputs:**
- Trained model checkpoint (`.eqx` file) loaded via `evaluation_utils.load_model()`
- Input features loaded via `process_features_and_setup_models()`
- PEtAb problem for computing simulations

**Outputs (per dataset = train, val):**

| Output | Path template (`common.py`) | Description |
|--------|---------------------------|-------------|
| Simulation residuals | `EVALUATION_TRAINING` | Per-row: `res`, `sim`, `obs`, `sample`, `observable`, `condition`, `time` + scan attributes |
| Latent embeddings | `EVALUATION_EMBEDDING` | Per-cell-line latent space coordinates |
| Parameter deviations | `EVALUATION_PARAMETER_DEVIATIONS` | Cell-line-specific parameter deviations from the average model |
| Full parameters | `EVALUATION_FULL_PARAMETERS` | Complete inferred parameter vectors per cell line |
| Parameter sensitivities | `EVALUATION_SENSITIVITY_PARAMS` | RMSE change when zeroing or isolating each ODE parameter deviation |
| Latent sensitivities | `EVALUATION_SENSITIVITY_LATENT` | RMSE change when zeroing each latent dimension |

**Key functions:**
- `evaluate_training()` — runs AMICI simulations via `dmm.analysis.evaluate_simulations()` and extracts embeddings/parameters via `evaluation_utils.get_embedding_and_params_df()`
- Sensitivity analysis section — uses `sensitivity.set_param_to_zero()`, `sensitivity.activate_single_param()`, and `sensitivity.zero_latent_direction()` to compute per-parameter and per-latent-dimension RMSE perturbations

**File layout:**
```
eval/{model}/{data}/training/{dataset}/{config}.csv
eval/{model}/{data}/embeddings/{dataset}/{config}.csv
eval/{model}/{data}/trained_param_dev/{dataset}/{config}.csv
eval/{model}/{data}/trained_parameters/{dataset}/{config}.csv
eval/{model}/{data}/sensitivity_params/{dataset}/{config}.csv
eval/{model}/{data}/sensitivity_latent/{dataset}/{config}.csv
```

where `{config}` is the `__`-joined scan attributes (context, features, n_hidden, ..., job, n_epoch, inflater_bound).

---

### 2. `evaluate_reference.py`

**Purpose:** Evaluate baseline/reference models that do **not** use the DMM encoder.

**Invoked by:** Snakefile rule `evaluate_references`; called once per `{model, data, samples}`.

**Reference models:**

| Model | Function | Description |
|-------|----------|-------------|
| `avg_model` | `evaluate_average_model()` | Simulates the ODE with the average (pretraining) parameters for all cell lines |
| `per_sample` | `evaluate_pretraining_per_sample()` | Simulates the ODE with per-cell-line pretrained parameters |

**Output:** Same `process_simulation` format as training (columns: `res`, `sim`, `obs`, `sample`, `observable`, `condition`, `time`).

**File layout:**
```
eval/{model}/{data}/references/{samples}_{mode}_{dataset}.csv
```

---

### 3. `evaluate_regressors.py`

**Purpose:** Train and evaluate linear regressors (linreg, lasso, elasticnet) that predict ODE simulation outputs from input features.

**Invoked by:** Snakefile rule `evaluate_regressors`; called once per `{model, data, context, features, samples}`.

**Workflow:**
1. Load input features (same as DMM)
2. Load output features (cytof dynamic measurements)
3. Train a `sklearn` pipeline via `regressor_training.train_pipeline()`
4. Predict on train/val sets
5. Call `dmm.analysis.process_simulation()` to compute residuals

**Output:** Same `process_simulation` format, plus `features` column.

**File layout:**
```
eval/{model}/{data}/regressors/{context}__{samples}__{mode}__{features}__{dataset}.csv
```

---

### 4. `evaluate_all.py`

**Purpose:** Aggregate all per-run evaluation CSVs (DMMs, references, regressors) into unified DataFrames. This is the central aggregation script.

**Invoked by:** Snakefile rule `evaluate_all`; called once per `{model, data}` (figure-level).

**Workflow:**

1. **Generate run configurations** via `generate_run_configs()` — produces the full grid of hyperparameter combinations grouped by CV split
2. **Load per-run CSVs** — iterates over `(samples, dataset)` pairs and reads:
   - DMM training evaluations (`EVALUATION_TRAINING`)
   - Latent embeddings (`EVALUATION_EMBEDDING`)
   - Parameter deviations (`EVALUATION_PARAMETER_DEVIATIONS`)
   - Full parameters (`EVALUATION_FULL_PARAMETERS`)
   - Parameter sensitivities (`EVALUATION_SENSITIVITY_PARAMS`) — if available
   - Latent sensitivities (`EVALUATION_SENSITIVITY_LATENT`) — if available
   - References (`EVALUATION_REFERENCE`) — avg_model, per_sample
   - Regressors (`EVALUATION_REGRESSOR`) — linreg, lasso, elasticnet
3. **Concatenate** all method types into a single `df`, tagging each with a `ref` column (`"DMM"`, `"avg_model"`, `"sample"`, `"linreg"`, `"lasso"`, `"elasticnet"`)
4. **Save aggregated outputs:**

| Output file | Contents |
|------------|----------|
| `trajectories_{figure}.csv` | Raw predicted trajectories (`sim` column) for all methods, without `obs`/`res` |
| `embeddings_{figure}.csv` | Concatenated latent embeddings across all configs |
| `param_devs_{figure}.csv` | Concatenated parameter deviations |
| `sensitivity_params_{figure}.csv` | Concatenated parameter sensitivities (if available) |
| `sensitivity_latent_{figure}.csv` | Concatenated latent sensitivities (if available) |

5. **RMSE aggregation** via `evaluation_utils.aggregate_and_log()`:
   - Computes RMSE per DMM config, per reference, and per cell-line/condition/observable
   - Identifies best regressor per context
   - Merges train/val results, selects top-N jobs per config
   - Determines best DMM configuration per context
   - Saves: `evaluate_all_{figure}.csv`, `by_cl_cond_obs_{figure}.csv`, `unified_dmm_rmse_train_test.csv`, `top1_best_dmm_{figure}.csv`, `top_{N}_best_dmm_with_refs.{split_label}.csv`

**File layout:**
```
eval/{model}/{data}/{filename}.csv
```

---

### 5. `report_all.py`

**Purpose:** Load aggregated results and produce final performance visualizations (plots, W&B logging).

**Invoked by:** Snakefile rule `report_all`; called after `evaluate_all`.

**Inputs:** Reads `evaluate_all_{figure}.csv`, `by_cl_cond_obs_{figure}.csv`, and `param_devs_{figure}.csv` from `EVALUATE_ALL_CSVS`.

---

## Key Supporting Modules

### `common.py`
Defines all file path templates (`EVALUATION_TRAINING`, `EVALUATION_REFERENCE`, etc.), directory paths (`evaluations_dir`, `fig_dir`, `results_dir`), cell line lists, and the `scan_attributes` list that parameterizes each run.

### `evaluation_utils.py`
- `load_model(conf, pypesto_subproblem)` — loads a trained `.eqx` model checkpoint
- `get_embedding_and_params_df(...)` — extracts latent embeddings and parameter DataFrames from a DMM
- `aggregate_and_log(df, conf, ...)` — performs RMSE aggregation, model selection, and saves result CSVs
- `get_measurements_and_obervables(conf)` — loads PEtAb measurement/observable tables
- Helper functions for avg_model simulation and per-sample pretrain processing

### `sensitivity.py`
- `set_param_to_zero(model, param)` — zeroes one parameter deviation in the output sparsity mask
- `activate_single_param(model, param)` — activates only one parameter deviation (all others zeroed)
- `zero_latent_direction(model, idx, dim)` — zeroes one row of the encoder weight matrix
- `classify_param(name)` — classifies parameters by receptor group (EGFR, ERBB2, EGFR+ERBB2, Other)
- `compute_sensitivities(...)` — high-level function (used in notebooks) that loops over contexts, splits, and jobs

### `dmm/analysis.py`
- `process_simulation(...)` — converts AMICI simulation + measurement DataFrames into per-row evaluation records with columns: `res`, `sim`, `obs`, `sample`, `observable`, `condition`, `time`
- `evaluate_simulations(...)` — runs AMICI simulations for all cell lines in a dataset, calling `process_simulation` for each

### `dmm/config_options.py`
- `Conf` dataclass — central configuration object with all hyperparameters
- `scan_attributes` — list of `Conf` fields that are varied across runs (context, features, n_hidden, depth, dropout_rate, nn_init_scale, regularization params, job, n_epoch, inflater_bound)

### `training_configuration.py`
- `CONTEXTS_FEATURES_BY_FIGURE` — which (context, features) pairs to evaluate per figure
- `PARAMS_TO_SCAN` — hyperparameter grid per figure
- `SPLITS_BY_FIGURE` — CV splits per figure
- `SELECT_CENTRAL_VALUES_BY_FIGURE` — whether to restrict to central hyperparameter values

### `generate_run_configs.py`
- `generate_run_configs(...)` — generates the full Cartesian product of hyperparameter configurations

---

## Data Flow Diagram (Columns)

All simulation-based evaluation scripts produce DataFrames via `process_simulation()` with these core columns:

```
res          — residual (obs - sim)
sim          — simulated/predicted value
obs          — observed measurement
sample       — cell line identifier (e.g. "cMCF7")
observable   — measured quantity (e.g. phospho-protein)
condition    — experimental condition (stimulus + time)
time         — time point
```

Plus method-specific metadata:
- DMMs: all `scan_attributes` (context, features, n_hidden, depth, job, regularization params, ...)
- References: none (added during aggregation)
- Regressors: `features` column

During aggregation in `evaluate_all.py`, additional columns are added:
- `ref` — method identifier (`"DMM"`, `"avg_model"`, `"sample"`, `"linreg"`, `"lasso"`, `"elasticnet"`)
- `dataset` — `"train"` or `"val"`
- `samples` — CV split identifier (e.g. `"0of5"`)
