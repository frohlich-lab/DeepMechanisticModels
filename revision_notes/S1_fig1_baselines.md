# S1: Fig. 1C baselines, paired tests, noise floor, Methods 8

Branch `claude/revision-fig1-baselines-e0894d`. This note fills the placeholders in
the responses to R2 major 1 and R3.10, and in the revised Results paragraph on
Fig. 1C.

**Status (2026-10-31): all numbers are in.** They come from the NEMO aggregates
`eval/EGFR_MAPK__logobs/dream_cytof/{trajectories,by_cl_cond_obs}_figure1a.csv`,
dated 7–8 April 2026.

**Read this first.** On the five test cell lines the DMM is **not more accurate**
than regularized regression on the same features. Its median per-combo RMSE is
0.36, against 0.34 for elastic net and 0.34 for lasso. The difference is not
significant after Holm correction. The regressions are numerically ahead in 3–4 of
the 5 test lines. The supported wording is therefore "comparable to", not "more
accurate than". The added value of the DMM has to rest on its mechanistic,
interpretable parameters and not on prediction accuracy. The advantage over the
negative control is also weak on the 5 test lines alone: 18% lower median,
p = 0.04 unadjusted, 3/5 lines, driven by CAMA1. It is clear on the 10 held-out
lines that the old figure pooled. Section 3a covers which set to call "test".

Code
- `figures_paper/baseline_comparison.py` (new): replicate pairing and noise floor;
  per-combo RMSE of every method on matched measurements; paired tests; the Fig. 1C
  box plot. `python -m figures_paper.baseline_comparison` prints everything below,
  for both held-out definitions, in about 20 s.
- `figures_paper/figure_1.ipynb`: the RMSE panel of Fig. 1C now shows DMM train, DMM
  test, the negative control, linear regression, lasso, elastic net and the positive
  control, with the noise floor as a dashed line. New cells compute the noise floor
  and the paired tests and write `fig1c_noise_floor.csv` and `fig1c_paired_tests.csv`
  (under `notebook_output/data/` when run through `run_notebooks.py`). The new
  `HELDOUT` switch is described in section 3a. It runs cleanly through
  `run_notebooks.py figure_1` on the NEMO aggregates (about 1 min). Committed outputs
  are cleared.

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
residuals and RMSEs are computed. That gives 6,793 measurements, a median of 112 per
cell line.

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
   this column, and `cytof/data.py` notes that MDAMB157 "has double the amount of
   fileIDs (biological replicates)"). Within one time course, the two values at each
   time point form a pair, at every time point: 215 pairs from MDAMB157 and MDAMB436,
   plus single points of CAL148 (iMEK, 0 min) and MCF12A (iEGFR, 0 min). Where a
   point has A, A, B and B, which A goes with which B is unknown. Those points
   therefore contribute only the within-time-course (set) pairs, never A-B pairs.

### Estimators
- **σ, the SD of a single measurement.** For a pair, E[(x1 − x2)^2] = 2σ^2, so
  σ = sqrt(mean(d^2) / 2), pooled over pairs. This is the RMSE, pooled over all
  measurements, that a model predicting every expected value exactly would still
  incur. The replicate-vs-replicate RMSE is √2·σ.
- **The floor on the scale of the Fig. 1C boxes.** The boxes are per-combo RMSEs and
  are summarized by their median, so the dashed line uses the same statistic.
  `combo_noise_floor` gives every held-out combo as many residuals as it has
  measurements, drawn with replacement from the replicate noise (d/√2), and takes the
  median per-combo RMSE (1,000 draws). The result is **0.245** (95% of draws
  0.224–0.266) for the 60 test combos. It is the same for the 120 combos of the
  10-line set and for the training combos.

### Results

| Pairs | n pairs | cell lines | σ | robust σ (MAD) | replicate RMSE |
|---|---:|---:|---:|---:|---:|
| **All** | 1,114 | 63 | **0.26** | 0.23 | 0.37 |
| Time courses A vs. B | 899 | 61 | 0.23 | 0.19 | 0.33 |
| Replicate sets | 215 | 4 | 0.36 | 0.47 | 0.51 |
| Test lines only (all A vs. B) | 80 | 5 | 0.22 | 0.22 | 0.32 |
| Training lines only | 1,034 | 58 | 0.27 | 0.23 | 0.38 |

By observable (A vs. B): pERK 0.25, pMEK 0.17, p-HER2 0.25, p-p90RSK 0.26. By time
point (A vs. B): EGF 0 min 0.25, EGF 9 min 0.20, iEGFR 0 min 0.26, iMEK 0 min 0.22.
The replicate sets show no trend with time (0.19 to 0.53 across time points).

**Values to quote:** σ = 0.26 as the noise floor, and 0.25 as the dashed line in
Fig. 1C (box scale).

**The positive control reaches the floor but does not beat it.** Its median per-combo
test RMSE is 0.22, slightly below the 0.25 line, but its pooled RMS over all test
residuals is 0.265, the same as σ = 0.263. The per-cell-line fits are in-sample: each
estimates up to 52 parameters (44 kinetic, 8 observable scale/offset) from about 110
measurements of that same line, so they absorb part of the replicate noise. Say this
in the legend; otherwise a reviewer will ask how a model can beat the noise.

Caveats:
- The A/B time courses come from one experiment on one day. Their agreement (0.23)
  leaves out experiment-to-experiment variability. The lines measured twice in full
  (MDAMB157 and MDAMB436, plus HCC2157 and MCF10F, which are not modelled) show
  **set-level offsets** of up to about 0.7 that persist across the whole time course.
  For example, MDAMB436 is shifted by about 0.6 in pERK, pMEK and p-p90RSK. So 0.26 is
  best read as a lower bound on the error a perfect model would make on a newly
  measured line. Adding the three lines in the data that are not modelled (HCC2157,
  MCF10F, MDAkb2) gives σ = 0.34 for replicate sets (371 pairs, 6 lines) and 0.23 for
  A vs. B (911 pairs, 62 lines).
- Please confirm what A and B are experimentally (independent cultures, or splits of
  one culture?). The text below says "replicate time courses". Change it to
  "biological replicate time courses" only if that is accurate.

### Paste-ready text
R2 major 1 (replaces "[RESULT]"; numbers for `HELDOUT = "test"`):
> …we report the measurement noise floor estimated from replicate measurements.
> Pooling 1,114 pairs of replicate measurements (same cell line, condition,
> observable and time point; time courses A and B, which share the unstimulated and
> the 9-min EGF time point, and two cell lines measured in two complete replicate
> sets) across all 63 cell lines, the standard deviation of a single measurement is
> 0.26 on the scale on which RMSEs are computed. This is the error a model
> predicting every expected signal exactly would still incur. On the scale of
> Fig. 1C (median RMSE per cell line, condition and observable), this corresponds to
> 0.25 (dashed line). Replicate time courses from the same experiment agree more
> closely (0.23; 899 pairs, 61 cell lines) than complete replicate sets (0.36; 215
> pairs), so this floor is, if anything, an underestimate for a newly measured cell
> line. Test-set errors of the DMM (median RMSE 0.36) thus lie about 0.11 above the
> noise floor. Mechanistic models fitted to each cell line individually (positive
> control) reach it (median 0.22; pooled RMSE 0.27 vs. 0.26), as expected for in-sample
> fits.

Revised Results: "…and the measurement noise floor (**0.25**, Fig. 1C)."

Fig. 1C legend addition:
> Dashed line: measurement noise floor, the median RMSE expected for a model that
> predicts every mean signal exactly, given the variability between replicate
> measurements (0.25; Methods). Positive-control fits are in-sample and can reach
> this floor.

Methods (new sentences, wherever RMSE is defined):
> The measurement noise floor was estimated from pairs of measurements of the same
> cell line, condition, observable and time point (replicate time courses sharing a
> time point, and complete replicate sets; 1,114 pairs). The standard deviation of a
> single measurement is σ = (Σ d_i² / 2n)^½ = 0.26. For comparison with per-
> combination RMSEs, we resampled residuals from the replicate differences (d/√2)
> for each test combination and report the median RMSE over 1,000 resamples (0.25).

---

## 2. Fig. 1C and paired tests (R2 major 1, R3.10, Results X/Y)

### Set-up
- **DMM**: the base CyTOF model (`EGFR_MAPK__logobs`, context `cytof_init`,
  `RFE_8_permute`, central hyperparameters, figure config `figure1a`, 10 random
  seeds). Per-combo RMSE is the median over seeds, as in the current figure.
- **Negative control** `avg_model`: one mechanistic parameter set fitted to all
  training lines of the split.
- **Positive control** `sample`: per-cell-line fit (in-sample by construction).
- **Linear, lasso and elastic-net regression** on the DMM's input features
  (section 4).
- **Matched scoring.** All methods are scored on the same individual replicate
  measurements; regressor predictions are matched on time point, with 100%
  coverage. The pipeline itself scores regressors against replicate-averaged,
  gap-filled targets (3b), and that view is reported below as a sensitivity check.
  For the DMM, the matched scoring reproduces the pipeline exactly: with the old
  pooling the medians are 0.428 and 0.336, as in the old figure.
- **Tests.** Two-sided Wilcoxon signed-rank over (cell line × condition × observable)
  combos, Holm-corrected across the five baselines. ΔRMSE = DMM − baseline, so
  negative values favour the DMM. "% lower" compares the median RMSEs (the boxes).
  Because combos of one line are not independent, the table also gives the number of
  lines in which the DMM's mean ΔRMSE favours it, and an exact sign-flip p over cell
  lines (with 5 lines, its smallest attainable value is 0.0625).

### Results: test set (5 test lines, split `all`, 60 combos) — notebook default

DMM median RMSE: test 0.36, train 0.34 (651 combos). Welch t-test train vs. test
p = 0.034.

| Baseline | median RMSE | DMM vs. baseline (median) | median ΔRMSE | DMM lower in (combos) | DMM better (lines) | p | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Negative control | 0.44 | **18% lower** | −0.020 | 33/60 | 3/5 | 0.039 | 0.13 |
| Linear regression | 0.35 | 3% higher | +0.030 | 24/60 | 2/5 | 0.21 | 0.21 |
| Lasso | 0.34 | 7% higher | +0.019 | 20/60 | 1/5 | 0.032 | 0.13 |
| Elastic net | 0.34 | **6% higher** | +0.022 | 22/60 | 2/5 | 0.062 | 0.13 |
| Positive control | 0.22 | 62% higher | +0.124 | 7/60 | 0/5 | 1.2e-9 | 6e-9 |

Per test line, mean ΔRMSE of the DMM:
- vs. elastic net: CAL120 +0.04, CAMA1 −0.06, HCC1143 +0.11, KPL1 −0.00, ZR75B +0.02.
- vs. negative control: CAL120 −0.04, CAMA1 −0.43, HCC1143 −0.08, KPL1 +0.04,
  ZR75B +0.04.

The DMM's advantage over the negative control on the test set comes mostly from
CAMA1.

As aggregated by the pipeline (regressors scored on averaged targets), the
regressors look slightly better still: elastic net 0.33 (DMM 10% higher, p = 0.19),
lasso 0.33 (p = 0.12), linear regression 0.33 (p = 0.34).

### Results: 10 held-out lines (5 test + 5 CV-validation, 120 combos) — old Fig. 1C pooling

DMM median RMSE: held-out 0.43, train 0.34.

| Baseline | median RMSE | DMM vs. baseline | median ΔRMSE | DMM lower in (combos) | lines | p | Holm p | sign-flip p (lines) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Negative control | 0.53 | **20% lower** | −0.139 | 82/120 | 8/10 | 1.3e-9 | 5e-9 | 0.012 |
| Linear regression | 0.41 | 5% higher | +0.009 | 54/120 | 6/10 | 0.64 | 0.64 | 0.82 |
| Lasso | 0.40 | 6% higher | +0.012 | 51/120 | 4/10 | 0.18 | 0.55 | 0.47 |
| Elastic net | 0.41 | **5% higher** | +0.012 | 51/120 | 5/10 | 0.29 | 0.57 | 0.55 |
| Positive control | 0.22 | 92% higher | +0.137 | 11/120 | 0/10 | 2.6e-19 | 1e-18 | 0.002 |

### Paste-ready text
Use one held-out definition throughout. Version A uses the 5 test lines, which is
the notebook default and what "test set" means elsewhere in the paper. Version B
keeps the 10-line pooling, which needs the wording change in 3a.

**Results, revised (version A):**
> Prediction errors on the test set (median RMSE 0.36, red) were only moderately
> higher than on the training set (0.34, blue), indicating limited overfitting. Test
> errors were 18% lower than those of a mechanistic model fitted to data averaged
> across cell lines (negative control; median 0.44) and comparable to those of an
> elastic-net regression baseline (0.34), while remaining above the error of
> mechanistic models fitted to each cell line individually (positive control, 0.22)
> and the measurement noise floor (0.25, Fig. 1C).

**Results, revised (version B):**
> Prediction errors on held-out cell lines (five test and five cross-validation lines;
> median RMSE 0.43, red) were higher than on the training set (0.34, blue). They were
> 20% lower than those of a mechanistic model fitted to data averaged across cell
> lines (negative control; 0.53; lower in 8 of 10 held-out lines) and comparable to
> those of an elastic-net regression baseline (0.41), while remaining above the error
> of mechanistic models fitted to each cell line individually (positive control,
> 0.22) and the measurement noise floor (0.25, Fig. 1C).

Under version A, "only moderately higher" (0.36 vs. 0.34) is accurate. Under version
B, 0.43 vs. 0.34 is a 27% increase, so "only moderately" is a stretch; I dropped it.

**R3.10 response (replaces "[RESULT: …]", version A):**
> …paired statistical tests across (cell line × condition × observable) combinations
> (two-sided Wilcoxon signed-rank tests over the 60 test-set combinations of the five
> test cell lines, Holm-corrected across five baselines). Prediction errors of the
> DMM did not differ significantly from those of regularized linear regression
> trained on the same features (DMM vs. elastic net: median ΔRMSE = +0.02, p = 0.06,
> adjusted p = 0.13; vs. lasso: +0.02, p = 0.03, adjusted p = 0.13; vs. ordinary
> least squares: +0.03, p = 0.21). They were lower than those of the population-
> average mechanistic model (median ΔRMSE = −0.02, p = 0.04, adjusted p = 0.13; 33
> of 60 combinations) and higher than those of per-cell-line fits (+0.12,
> p = 1·10⁻⁹). With only five test cell lines, these combination-level tests treat
> conditions and observables of one cell line as independent and should be read as
> descriptive.

(Version B: elastic net +0.01, p = 0.29, adjusted p = 0.57; lasso +0.01, p = 0.18;
OLS +0.01, p = 0.64; negative control −0.14, p = 1·10⁻⁹, lower in 8/10 lines,
cell-line sign-flip p = 0.01; n = 120.)

**Results sentence for R3.10:**
> DMM predictions were comparable in accuracy to those of regularized linear
> regression models trained on the same features (Fig. 1C; median RMSE 0.36 vs. 0.34
> for elastic net, Wilcoxon signed-rank p = 0.06, n = 60 test combinations), while
> providing cell-line-specific kinetic parameters that the regression models lack.

The second clause is my suggestion for where the argument has to go. Adjust it to
what the paper actually shows (Figs. 2–4).

---

## 3. Flags for the authors

### a. The old "test" box was not the test set
The old Fig. 1C "test" box (median 0.43; train 0.34) and the train-vs-test numbers in
`figure_1.ipynb` used every split's held-out line: **10 lines (120 combos)**, namely
the 5 test lines (split `all`) plus the 5 CV-validation lines (BT20, HCC1500, HCC2185,
MCF7, UACC3199), each predicted by the CV model that held it out. The CV lines
informed the hyperparameter choice, so calling the pooled set "the test set" is
inaccurate. The notebook default is now `HELDOUT = "test"`: split `all` only, with
the test box from the 5 test lines and the train box from the same models' fits.
`HELDOUT = "cv+test"` restores the old pooling. Under the default:
- the per-observable/condition train-vs-test tests (section 2 of the notebook)
  have n_val = 5 and none is significant after FDR correction;
- the representative trajectories now all come from test lines. Q5 and Q95 are the
  same combos as before; Q50 is now ZR75B / EGF / pERK.

### b. The pipeline scores the regressors on averaged targets
`evaluate_regressors.py` scores the regressors against replicate means on a
gap-filled time grid. At 0 min and at 9 min (EGF) they are compared with one averaged
value, while the mechanistic models are compared with two replicates. That makes the
regressors look about 0.01 better (elastic net 0.33 instead of 0.34). All numbers
above use matched scoring.

### c. Legend label
`figure_config.MODEL_LABELS["elasticnet"]` is "linear regression", so the elastic net
trajectories in Fig. 1 were labelled "linear regression". Now that OLS appears as
"Linear reg.", the notebook overrides the label locally to "Elastic net".
`figure_config.py` (S4's file) is unchanged.

### d. Regressor outputs may predate the CV-fold change: unverified
The aggregates date from 7–8 April 2026. See section 4. Whether these regressors used
the DMM folds or scikit-learn's default 5-fold split cannot be told from the
aggregates. To check, refit split `all` both ways and compare with the stored
predictions. That needs only `features/EGFR_MAPK__logobs/dream_cytof/{train,val}/
cytof_init__all__RFE_8_permute.csv` from NEMO; only `RFE_15` is on the laptop.
Alternatively, rerun `evaluate_regressors` for `figure1a` (CPU only). The choice
only moves α, so a large change is unlikely, but the Methods should describe the
procedure that produced the numbers.

---

## 4. Methods 8 check: "regression baselines predict the measured CyTOF dynamics directly from the DMM's input features with the same CV folds as DMM tuning"

Verdict: **true for the current code, with qualifications that the Methods should
state.** Whether the April 2026 outputs used the same CV folds is unverified (3d).

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
- **Caveat: provenance of the outputs.** `build_pipeline` and `train_pipeline` still
  default to `cv=5` (unshuffled 5-fold `KFold` over the training lines). Any outputs
  written before `evaluate_regressors.py` started passing the DMM folds used that
  default. A comment in `evaluate_reference.py` refers to a "regressor-fairness note
  that used to sit here", which suggests the change is recent. Git history (a single
  squashed commit) cannot tell, and the aggregates date from April 2026. See 3d for
  how to check.
- **Scoring.** The pipeline scores the regressors against the averaged, gap-filled
  targets (3b); the Fig. 1C numbers above score them against the individual
  measurements.

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

(If 3d shows that the April outputs used the default 5-fold split, replace "over the
same folds used to tune the DMM, each holding out one of the five cross-validation
cell lines" with "by five-fold cross-validation over the training cell lines", or
rerun.)
