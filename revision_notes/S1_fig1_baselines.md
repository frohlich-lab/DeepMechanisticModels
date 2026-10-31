# S1: Fig. 1C baselines, paired tests, noise floor, Methods 8

Branch `claude/revision-fig1-baselines-e0894d`. Fills placeholders in the responses
to R2 major 1 and R3.10, and in the revised Results paragraph on Fig. 1C.

**Status (2026-10-31)**
- Noise floor: done. The numbers below are final.
- Methods 8 check: done. See section 4.
- Fig. 1C and the paired tests: the code is done and was tested end to end on
  synthetic data in the `evaluate_all` layout. **X, Y and the test statistics are
  still open.** They need `trajectories_figure1a.csv` and
  `by_cl_cond_obs_figure1a.csv` from `eval/EGFR_MAPK__logobs/dream_cytof/` on NEMO.
  Those files are not on the laptop, and `ssh crick` times out from it.

Code
- `figures_paper/baseline_comparison.py` (new): replicate pairing and noise floor,
  per-combo RMSE of every method on matched measurements, paired tests, and the
  Fig. 1C box plot. `python -m figures_paper.baseline_comparison` prints all of
  it, and skips the tests while `eval/` is missing.
- `figures_paper/figure_1.ipynb`: the RMSE panel of Fig. 1C now shows DMM train,
  DMM test, the negative control, linear regression, lasso, elastic net and the
  positive control, plus the noise floor as a dashed line. New cells compute the
  noise floor and the paired tests and write `fig1c_noise_floor.csv` and
  `fig1c_paired_tests.csv` (under `notebook_output/data/` when run through
  `run_notebooks.py`). The new `HELDOUT` switch is explained in section 3a.
  Outputs are cleared because none could be regenerated here.

---

## 1. Measurement noise floor (R2 major 1, Results "[value]")

### Data and scale
These are the measurements the mechanistic models are scored on: the PEtab
measurement table `data/EGFR_MAPK__logobs__dream_cytof__measurements.tsv`, restricted
to `measurementType == "cytof"` and the four model observables in
`..._observables.tsv` (pERK_Y204, pMEK_S222, pERBB2_Y1248 (p-HER2),
pRPS6KA1_S380 (p-p90RSK)). The conditions are EGF, iEGFR and iMEK; full medium is
dropped, as it is not scored. The lines are the 63 modelled cell lines (58 training
and 5 test). The values are row for row identical to `data/cytof.csv`: the per-well
mean of the arcsinh-transformed single-cell intensities, which is the scale on which
residuals and RMSEs are computed. That gives 6,793 measurements.

### How replicates are paired
A replicate pair is two measurements of the same cell line, condition, observable
and time point. Every cell line was measured on a single `date`, so there are no
between-day replicates. Pairs arise in two ways:

1. **Time courses A vs. B** (`time_course` column). The two time courses of each
   cell line and condition are interleaved: EGF A has 0, 5.5, 9, 13, 23 and 40 min;
   EGF B has 0, 7, 9, 17, 30 and 60 min; the inhibitor courses have 0/9/13/40 and
   0/7/17/60 min. They share only the **0-min point under every condition and the
   9-min point under EGF**. Each shared point with exactly one A and one B
   measurement gives one pair: 899 pairs from 61 cell lines. MDAMB157 and MDAMB436
   are not in this group (see 2).
2. **Replicate sets.** A few lines were measured as two complete sets: the same date
   and the same time-course label, but different files (fileID; the pipeline drops
   this column, and the code notes that MDAMB157 "has double the amount of fileIDs
   (biological replicates)"). Within one time course, the two values at each time
   point form a pair, at every time point: 215 pairs from MDAMB157 and MDAMB436, plus
   single points of CAL148 (iMEK, 0 min) and MCF12A (iEGFR, 0 min). Where a point has
   A, A, B and B, which A goes with which B is unknown. Those points therefore
   contribute only the within-time-course (set) pairs, never A-B pairs.

### Estimator
For a pair, E[(x1 − x2)^2] = 2σ^2. The noise SD of a single measurement is therefore
**σ = sqrt(mean(d^2) / 2)**, pooled over pairs. This is the RMSE that a model
predicting every expected value exactly would still incur against individual
measurements. It is directly comparable to the per-combo RMSEs in Fig. 1C, which are
also computed against individual replicate measurements. The RMSE between the two
replicates themselves is √2·σ, given for completeness.

### Results

| Pairs | n pairs | cell lines | σ (noise floor) | replicate-vs-replicate RMSE |
|---|---:|---:|---:|---:|
| **All** | 1,114 | 63 | **0.26** | 0.37 |
| Time courses A vs. B | 899 | 61 | 0.23 | 0.33 |
| Replicate sets | 215 | 4 | 0.36 | 0.51 |
| Test lines only (all A vs. B) | 80 | 5 | 0.22 | 0.32 |
| Training lines only | 1,034 | 58 | 0.27 | 0.38 |

By observable (A vs. B): pERK 0.25, pMEK 0.17, p-HER2 0.25, p-p90RSK 0.26. By time
point (A vs. B): EGF 0 min 0.25, EGF 9 min 0.20, iEGFR 0 min 0.26, iMEK 0 min 0.22.
The replicate sets show no trend with time (0.19 to 0.53 across time points). The
baseline points are therefore not systematically noisier or quieter than the
stimulated ones.

**Recommended value: 0.26.** For comparison, the DMM's held-out median RMSE is 0.43
in the current figure and the training median is 0.34.

Caveats the authors should know about:
- The A/B time courses come from one experiment on one day. Their agreement (0.23)
  leaves out experiment-to-experiment variability. The only lines measured twice
  in full (MDAMB157 and MDAMB436, plus HCC2157 and MCF10F, which are not modelled)
  show **set-level offsets** of up to about 0.7 that persist across the whole time
  course. For example, MDAMB436 is shifted by about 0.6 in pERK, pMEK and p-p90RSK.
  This variability exists but cannot be estimated for most lines, so 0.26 is best
  read as a lower bound on the error a perfect model would make on a newly measured
  cell line. Adding the three lines in the data that are not modelled (HCC2157,
  MCF10F, MDAkb2) gives σ = 0.34 for replicate sets (371 pairs, 6 lines) and 0.23
  for A vs. B (911 pairs, 62 lines).
- Please confirm what A and B are experimentally (independent cultures, or splits
  of one culture?). The text below calls them "replicate time courses" so that it
  does not claim more than the data show. Change it to "biological replicate time
  courses" if that is accurate.
- Taking the median over per-combo σ would be biased low: each combo has only one
  or two pairs, and that median is 0.16. Use the pooled value.

### Paste-ready text
R2 major 1 (replaces "[RESULT]"):
> …we report the measurement noise floor estimated from replicate measurements.
> Pooling 1,114 pairs of replicate measurements (same cell line, condition,
> observable and time point; time courses A and B, which share the unstimulated and
> the 9-min EGF time point, and two cell lines measured in two complete replicate
> sets) across all 63 cell lines, the standard deviation of a single measurement is
> 0.26 on the scale on which RMSEs are computed. This is the RMSE that a model
> predicting every expected signal exactly would still incur. Replicate time courses
> from the same experiment agree more closely (0.23; 899 pairs, 61 cell lines) than
> complete replicate sets (0.36; 215 pairs), so this floor is, if anything, an
> underestimate for a newly measured cell line. Test-set prediction errors (median
> RMSE [0.xx]) thus lie [0.xx] above the noise floor, and the positive control (per-
> cell-line fits, [0.xx]) lies [close to / 0.xx above] it.

Revised Results: "…and the measurement noise floor (**0.26**, Fig. 1C)."

Fig. 1C legend addition:
> Dashed line: measurement noise floor, the standard deviation of a single
> measurement estimated from replicate pairs (0.26; Methods).

Methods (new sentence, wherever RMSE is defined):
> The measurement noise floor was estimated from pairs of measurements of the same
> cell line, condition, observable and time point (replicate time courses sharing a
> time point, and complete replicate sets) as σ = (Σ d_i² / 2n)^½, the expected RMSE
> of a model that predicts the mean signal exactly.

---

## 2. Fig. 1C and paired tests (R2 major 1, R3.10, Results X/Y) — numbers pending

### What is compared
- **DMM**: the base CyTOF model (`EGFR_MAPK__logobs`, context `cytof_init`,
  `RFE_8_permute`, central hyperparameters, figure config `figure1a`). Per-combo
  RMSE is the median over random seeds, as in the current figure.
- **Negative control** `avg_model`: one mechanistic parameter set fitted to all
  training lines of the split.
- **Positive control** `sample`: per-cell-line fit (in-sample by construction).
- **Linear, lasso and elastic-net regression** (`linreg`, `lasso`, `elasticnet`)
  on the DMM's input features (section 4).
- Unit: one (cell line × condition × observable) combo. With `HELDOUT = "test"`,
  that is 5 test lines × 3 conditions × 4 observables = 60 combos.

### Matched scoring
The pipeline (`evaluate_regressors.py` → `aggregate_and_log`) scores the
regressors against **their own training-style targets**: replicate means on a
gap-filled time grid. At 0 min and at 9 min (EGF) the regressors are therefore
compared with one averaged value, while the mechanistic models are compared with
two replicates. The regressors are also scored at filled-in time points. That
setup favours the regressors and makes the pairs not quite paired.
`matched_combo_rmse` scores every method against the **same individual
measurement rows**, matching regressor predictions on time point. The notebook
prints the fraction of measurements with a regressor prediction, which should be
1.0 on the test set. Use the matched numbers. The as-aggregated ones are printed
as a sensitivity check.

### Tests
- Two-sided Wilcoxon signed-rank tests over combos, DMM vs. each of the five
  baselines (`scipy.stats.wilcoxon`; normal approximation for n > 50),
  Holm-corrected across the five. Reported: n, median RMSE of each method, median
  ΔRMSE (DMM − baseline, so negative favours the DMM), the % difference of median
  RMSEs (X and Y), the number of combos in which the DMM is lower, W, p and Holm p.
- Robustness: the combos of one cell line are not independent. `p_cell_lines`
  repeats each comparison on one mean ΔRMSE per cell line, using an exact sign-flip
  test. With 5 test lines its smallest attainable p is 0.0625, so report it as
  "lower in k/5 test lines" rather than as a p-value.

### How to get the numbers
1. Copy `eval/EGFR_MAPK__logobs/dream_cytof/trajectories_figure1a.csv` and
   `by_cl_cond_obs_figure1a.csv` from NEMO into this checkout's
   `eval/EGFR_MAPK__logobs/dream_cytof/`.
2. Run `./venv/bin/python run_notebooks.py figure_1`, or
   `./venv/bin/python -m figures_paper.baseline_comparison`, which prints both
   held-out definitions.
3. Paste the printed lines into the templates below.

### Paste-ready templates (fill from `fig1c_paired_tests.csv`)
Results, revised: "Prediction errors on the test set (median RMSE [DMM val
median], red) were only moderately higher than on the training set ([DMM train
median], blue), indicating limited overfitting. Test errors were [pct_lower vs
avg_model]% lower than those of a mechanistic model fitted to data averaged across
cell lines (negative control) and [comparable to (Holm p ≥ 0.05) / pct_lower vs
elasticnet% lower than] those of an elastic-net regression baseline, while
remaining above the error of mechanistic models fitted to each cell line
individually (positive control, [sample median]) and the measurement noise floor
(0.26, Fig. 1C)."

R3.10: "…paired statistical tests across (cell line × condition × observable)
combinations (two-sided Wilcoxon signed-rank tests, n = [60] test-set combinations
from 5 test cell lines, Holm-corrected across baselines): DMM vs. elastic net,
median ΔRMSE = [median_delta] (p = [p_holm]); vs. lasso, [..]; vs. linear
regression, [..]; vs. the population-average mechanistic model, [..]. The DMM had
the lower RMSE in [dmm_lower]/[n] combinations and in [cell_lines_dmm_lower]/5 test
cell lines relative to elastic net."

Results (R3.10): "DMM predictions were [comparable to / more accurate than] those
of regularized linear regression models trained on the same features (Fig. 1C;
median ΔRMSE vs. elastic net [..], Wilcoxon signed-rank p = [..], n = [60])."

Choose "comparable to" if the Holm-adjusted p ≥ 0.05 for both lasso and elastic
net, or if the sign is not consistent across test lines.

---

## 3. Flags for the authors

### a. The current "test" box is not the test set
The "test" box in Fig. 1C (median 0.43; train 0.34) and the train-vs-test numbers in
`figure_1.ipynb` used every split's held-out line. That is **10 lines (120 combos)**:
the 5 test lines (split `all`) plus the 5 CV-validation lines (BT20, HCC1500, HCC2185,
MCF7, UACC3199), each predicted by the CV model that held it out. The saved notebook
output confirms this: "Range … (120 combos)", and n_val = 10 per observable and
condition. Those CV lines informed the hyperparameter choice. Calling the pooled set
"the test set" is therefore inaccurate and mildly optimistic.

The notebook now has `HELDOUT = "test"` (default): only split `all`, so the test
box is the 5 test lines and the train box is the training fit of the same models.
`HELDOUT = "cv+test"` restores the old pooling. **The 0.43/0.34 in the drafted text
will change** under the default. Please decide which definition the paper uses and
update the Results numbers accordingly. `python -m figures_paper.baseline_comparison`
prints the tests for both. If the 10-line set is kept, call it "held-out cell lines
(5 test lines and 5 cross-validation lines)". In that case n = 120 combos and the
cell-line-level sign-flip test can reach p ≈ 0.002.

### b. The pipeline scores the regressors on averaged targets
See section 2 (matched scoring). The Methods text should say that all methods are
scored against the same individual measurements, which is true for the numbers this
code produces.

### c. Legend label
`figure_config.MODEL_LABELS["elasticnet"]` is "linear regression", so the elastic
net trajectories in Fig. 1 were labelled "linear regression". Now that OLS appears
as "Linear reg.", the notebook overrides the label locally to "Elastic net".
`figure_config.py` (S4's file) is unchanged.

### d. Regressor outputs may predate the CV-fold change
See section 4: confirm that the regressor CSVs on NEMO were regenerated after
`evaluate_regressors.py` switched to the DMM folds.

---

## 4. Methods 8 check: "regression baselines predict the measured CyTOF dynamics directly from the DMM's input features with the same CV folds as DMM tuning"

Verdict: **true for the current code, with qualifications that the Methods should
state.** Whether the evaluation outputs on NEMO were produced by this version of the
code cannot be checked from the laptop.

What the code does (`evaluate_regressors.py`, `regressor_training.py`,
`dmm/feature_selection.py`):

- **Inputs: the DMM's input features. True.** They come from the same selected-feature
  file as the DMM for that context, feature set and CV split (`FEATURES_OUTFILE`,
  read with `process_features`). Feature selection is therefore per split, on that
  split's training lines. Inside the pipeline the inputs are standardized
  (`StandardScaler`), and missing values are imputed with `KNNImputer`
  (k = 5, the default).
- **Outputs: measured CyTOF dynamics, directly. True, with processing.** The targets
  are the `cytof_dynamic` contextualization. There is one output per model observable
  (pERK, pMEK, p-HER2, p-p90RSK) × condition (EGF, iEGFR, iMEK) × time point. Each is
  **averaged over replicates** (A/B time courses and replicate sets) and **aligned to a
  common time grid** by `harmonise_cytof_dynamic`: 13 min is filled from 12 or 14,
  17 min from 15 or 16 (inhibitors), 60 from 40, 40 from 35 (EGF), and 7, 13 and 40 min
  are interpolated linearly. Outputs missing in ≥ 50 % of training lines are dropped,
  and the remaining missing training targets are KNN-imputed. All outputs are fitted
  **jointly as one multi-output model**. `MultiTaskLassoCV` and `MultiTaskElasticNetCV`
  select one common set of features for all outputs. There is no mechanistic model in
  the loop.
- **Same CV folds as DMM tuning. True in the current code.** α (the penalty strength;
  20 candidates) is chosen with `PredefinedSplit` over the DMM's CV folds. Each fold
  validates on one CV line (BT20, HCC1500, HCC2185, MCF7 or UACC3199), and all other
  training lines are always used for fitting. The model is then refitted on all
  training lines of the split. The split `all` that predicts the test lines uses 5
  folds; each CV split uses the 4 CV lines it trains on. Test lines never enter the
  fit or the α selection. Ordinary least squares has no hyperparameter. The
  elastic-net mixing parameter is **fixed at l1_ratio = 0.5** (scikit-learn default)
  and not tuned. The draft should not say both are tuned.
- **Caveat: provenance of the outputs.** `build_pipeline` and `train_pipeline`
  still default to `cv=5` (unshuffled 5-fold `KFold` over the training lines). Any
  outputs written before `evaluate_regressors.py` started passing the DMM folds used
  that default. A comment in `evaluate_reference.py` refers to a
  "regressor-fairness note that used to sit here", which suggests the change is
  recent, and git history (a single squashed commit) cannot tell. Check the modification times of
  `eval/EGFR_MAPK__logobs/dream_cytof/regressors/*__lasso__*` and `*__elasticnet__*`
  on NEMO against the date of the change. If they are older, rerun the
  `evaluate_regressors` rule (CPU only, minutes per split) and then `evaluate_all`
  for `figure1a` before quoting X, Y or p.
- **Scoring.** The pipeline scores the regressors against the averaged, gap-filled
  targets (3b). The new code scores them against the individual measurements.

Suggested Methods 8 text (US spelling):
> Regression baselines. To test whether the mechanistic structure adds predictive
> value beyond a direct statistical mapping, we trained ordinary least-squares,
> multi-task lasso and multi-task elastic-net regression models (scikit-learn) that
> predict the measured CyTOF dynamics directly from the same input features as the
> DMM (same feature set and cross-validation split). Inputs were standardized and
> missing values imputed by k-nearest neighbors (k = 5). The targets were the
> replicate-averaged phospho-signals of the four model observables under each
> condition and time point, aligned to a common time grid (missing time points
> filled from the nearest measured or by linear interpolation), and were fitted
> jointly as one multi-output model. The multi-task penalties select a common set of
> features across outputs. The regularization strength (20 candidate values; elastic
> net mixing parameter fixed at 0.5) was chosen by cross-validation over the same
> folds used to tune the DMM, each holding out one of the five cross-validation cell
> lines, and the model was refitted on all training cell lines. Test cell lines were
> not used for fitting or model selection. For comparison with the DMM, regression
> predictions were scored against the same individual replicate measurements as the
> mechanistic models.
