# S1: Fig. 1C baselines, paired tests, noise floor, Methods 8

Branch `claude/revision-fig1-baselines-e0894d`. This note fills the placeholders in
the responses to R2 major 1 and R3.10, in the revised Results paragraph on Fig. 1C,
and in Methods 8.

**Status: final (2026-10-31), rechecked 2026-11-24 on the post-fix files.** The
numbers were first computed on the NEMO aggregates
`eval/EGFR_MAPK__logobs/dream_cytof/{trajectories,by_cl_cond_obs}_figure1a.csv` of
7–8 April 2026. They were rechecked on the post-HVG-fix files of 19 November 2026
(PR #21 rerun of `figure1a`), using `python -m figures_paper.baseline_comparison` and
`run_notebooks.py figure_1`. Every number in sections 1–3 is identical at full
precision, because the rerun left the `cytof_init` rows unchanged. Section 3b
depends only on those rows and on the April `cytof_init` feature files, so it is
unchanged too.

One decision is left to the authors: the Methods 8 variant (section 5). It decides
whether the regression numbers stay as they are (variant 1, the numbers below) or
are rerun.

## Summary

- **Test set = the 5 test lines** (CAL120, CAMA1, HCC1143, KPL1, ZR75B), predicted by
  the models trained on all 58 training lines (FF's decision, 2026-10-31). This is the
  default in `figure_1.ipynb` (`HELDOUT = "test"`) and in Fig. 1C.
- **The draft's train/test RMSEs (0.34/0.43) came from the old pooling.** That
  pooling mixed the 5 test lines with the 5 CV-validation lines; section 4c has the
  details. Corrected values: **train 0.34, test 0.36** (median RMSE per cell line ×
  condition × observable).
- **The DMM is comparable to regularized regression, not better.** On the test set
  its median is 0.36, against 0.34 for elastic net and 0.34 for lasso. That is 6–7%
  higher, not significant after Holm correction (p = 0.13). The regressions are
  numerically ahead in 3–4 of the 5 test lines. "More accurate than" is not
  supported. The DMM's added value has to rest on its mechanistic, interpretable
  parameters.
- **Negative control:** the DMM is 18% lower (0.36 vs. 0.44), p = 0.04 unadjusted
  (Holm 0.13), better in 3 of 5 lines. Most of this comes from CAMA1.
- **Noise floor:** the SD of a single measurement is σ = 0.26. On the box scale of
  Fig. 1C it is 0.25. The positive control (0.22) reaches it because those fits are
  in-sample.
- **Methods 8 does not describe the reported numbers.** The April lasso and
  elastic-net models chose their penalty by scikit-learn's default 5-fold split, not
  by the DMM's CV folds. This was verified by exact reproduction. With the DMM folds
  the conclusion is the same (section 3).

Code
- `figures_paper/baseline_comparison.py` (new): replicate pairing and noise floor;
  per-combo RMSE of every method on the same measurements; paired tests; the
  Fig. 1C box plot. `python -m figures_paper.baseline_comparison` prints all numbers
  below in about 20 s.
- `figures_paper/figure_1.ipynb`: the RMSE panel of Fig. 1C shows DMM train, DMM
  test, the negative control, linear regression, lasso, elastic net and the positive
  control, with the noise floor as a dashed line. It writes `fig1c_noise_floor.csv`
  and `fig1c_paired_tests.csv` (to `notebook_output/data/` via `run_notebooks.py`).
  `run_notebooks.py figure_1` passes on the NEMO aggregates in about 1 min.
  Committed outputs are cleared.

---

## 1. Paste-ready text (5 test lines)

Numbers are for the April regressors (Methods 8, variant 1). If the regressors are
rerun on the DMM folds (variant 2), replace the regression numbers with those in
section 3b.

### Results paragraph (replaces the drafted Fig. 1C sentences)
> Prediction errors on the test set (median RMSE 0.36, red) were only moderately
> higher than on the training set (0.34, blue), indicating limited overfitting. Test
> errors were 18% lower than those of a mechanistic model fitted to data averaged
> across cell lines (negative control; median 0.44) and comparable to those of an
> elastic-net regression baseline (0.34), while remaining above the error of
> mechanistic models fitted to each cell line individually (positive control, 0.22)
> and the measurement noise floor (0.25, Fig. 1C).

The R3.10 sentence in the Results:
> DMM predictions were comparable in accuracy to those of regularized linear
> regression models trained on the same features (Fig. 1C; median RMSE 0.36 vs. 0.34
> for elastic net, Wilcoxon signed-rank p = 0.06, n = 60 test combinations), while
> providing cell-line-specific kinetic parameters that the regression models lack.

The second clause is my suggestion for where the argument has to go. Adjust it to
what Figs. 2–4 show.

### Fig. 1C legend
> RMSE per cell line, condition and observable for the training set (blue; 58 cell
> lines, 651 combinations) and the test set (red and baselines; five held-out cell
> lines, 60 combinations), each DMM value being the median over 10 random
> initializations. Test cell lines were used neither for training nor for model
> selection. Negative control: mechanistic model with parameters fitted to all
> training cell lines jointly. Positive control: mechanistic model fitted to each cell
> line individually (in-sample, and therefore able to reach the noise floor). Linear
> reg., lasso, elastic net: regression from the DMM's input features to the measured
> signals. Dashed line: measurement noise floor, the median RMSE expected for a model
> that predicts every mean signal exactly, given the variability between replicate
> measurements (0.25; Methods). Red ticks: quantiles of the trajectories shown.

### R2 major 1 (replaces "[RESULT]")
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
> control) reach it (median 0.22; pooled RMSE 0.27 vs. 0.26), as expected for
> in-sample fits.

Please confirm what A and B are experimentally (independent cultures, or splits of
one culture?). The text says "replicate time courses"; change it to "biological
replicate time courses" only if that is accurate.

### R3.10 response (replaces "[RESULT: e.g. DMM vs elastic net, …]")
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

### Methods: noise floor (new sentences, wherever RMSE is defined)
> The measurement noise floor was estimated from pairs of measurements of the same
> cell line, condition, observable and time point (replicate time courses sharing a
> time point, and complete replicate sets; 1,114 pairs). The standard deviation of a
> single measurement is σ = (Σ d_i² / 2n)^½ = 0.26. For comparison with per-
> combination RMSEs, we resampled residuals from the replicate differences (d/√2)
> for each test combination and report the median RMSE over 1,000 resamples (0.25).

### Methods 8: regression baselines
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
> net mixing parameter fixed at 0.5) was chosen by {CV CLAUSE}, and the model was
> refitted on all training cell lines. Test cell lines were not used for fitting or
> model selection. For comparison with the DMM, regression predictions were scored
> against the same individual replicate measurements as the mechanistic models.

{CV CLAUSE} (section 5):
- Variant 1, the April numbers as they are: "five-fold cross-validation over the
  training cell lines".
- Variant 2, after rerunning the regressors: "cross-validation over the same folds
  used to tune the DMM, each holding out one of the five cross-validation cell lines".

---

## 2. Test-set results (5 test lines, split `all`, 60 combinations)

Set-up
- **DMM**: base CyTOF model (`EGFR_MAPK__logobs`, context `cytof_init`,
  `RFE_8_permute`, central hyperparameters, figure config `figure1a`). Per-combo RMSE
  is the median over 10 random seeds.
- **Negative control** `avg_model`: one parameter set fitted to all 58 training lines.
  **Positive control** `sample`: per-cell-line fit (in-sample). **Regressions**:
  section 5.
- **Scoring**: every method is scored on the same individual replicate measurements.
  Regression predictions are matched on time point, with 100% coverage. For the DMM
  this reproduces the pipeline exactly: under the old pooling, the medians are 0.428
  and 0.336, as in the old figure.
- **Tests**: two-sided Wilcoxon signed-rank tests over combinations, Holm-corrected
  across the five baselines. ΔRMSE = DMM − baseline (negative favours the DMM).
  "% lower" compares median RMSEs, i.e. the boxes. "Lines" counts the test lines in
  which the DMM's mean ΔRMSE favours it. A cell-line-level sign-flip test cannot go
  below p = 0.0625 with 5 lines.

DMM median RMSE: **test 0.36, train 0.34** (651 combinations). Welch t-test, train
vs. test: p = 0.034.

| Baseline | median RMSE | DMM vs. baseline | median ΔRMSE | DMM lower (combos) | DMM better (lines) | p | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Negative control | 0.44 | **18% lower** | −0.020 | 33/60 | 3/5 | 0.039 | 0.13 |
| Linear regression | 0.35 | 3% higher | +0.030 | 24/60 | 2/5 | 0.21 | 0.21 |
| Lasso | 0.34 | 7% higher | +0.019 | 20/60 | 1/5 | 0.032 | 0.13 |
| Elastic net | 0.34 | **6% higher** | +0.022 | 22/60 | 2/5 | 0.062 | 0.13 |
| Positive control | 0.22 | 62% higher | +0.124 | 7/60 | 0/5 | 1.2e-9 | 6e-9 |

Mean ΔRMSE per test line:
- vs. elastic net: CAL120 +0.04, CAMA1 −0.06, HCC1143 +0.11, KPL1 −0.00, ZR75B +0.02.
- vs. negative control: CAL120 −0.04, CAMA1 −0.43, HCC1143 −0.08, KPL1 +0.04,
  ZR75B +0.04.

Per observable and condition, none of the train-vs-test differences is significant
after FDR correction (Mann–Whitney U, n = 5 test lines). The representative
trajectories all come from test lines: Q5 CAL120 / +iRTK / pMEK, Q50 ZR75B / EGF /
pERK, Q95 ZR75B / +iRTK / pERK.

---

## 3. Sensitivity checks (none changes the conclusion)

### a. Scoring the regressions as the pipeline does
`evaluate_regressors.py` scores the regressions against replicate means on a
gap-filled time grid, not against the individual measurements the DMM is scored on.
Scored that way, they look about 0.01 better: elastic net 0.33 (DMM 10% higher,
p = 0.19), lasso 0.33 (p = 0.12), linear regression 0.33 (p = 0.34).

### b. Regressors refitted on the DMM's CV folds (Methods 8, variant 2)
The current code selects α over the DMM folds. That gives a smaller α (lasso 0.040
instead of 0.171; elastic net 0.039 instead of 0.165). Test set:

| Baseline | median RMSE | DMM vs. baseline | median ΔRMSE | DMM lower (combos) | lines | p | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Lasso (DMM folds) | 0.36 | 2% higher | +0.029 | 24/60 | 2/5 | 0.13 | 0.38 |
| Elastic net (DMM folds) | 0.35 | 2% higher | +0.023 | 24/60 | 2/5 | 0.13 | 0.38 |

Linear regression, the controls and the DMM do not change. The negative-control Holm
p becomes 0.16. Under variant 2, replace the regression numbers in section 1 with
these: elastic net 0.35, median ΔRMSE +0.02, p = 0.13, adjusted p = 0.38; lasso
+0.03, p = 0.13, adjusted p = 0.38. These numbers are from a local refit of split
`all`. A NEMO rerun with the current code should reproduce them.

### c. Old pooling (10 held-out lines: 5 test + 5 CV-validation lines, 120 combinations)
This is what the draft's 0.43 came from. It pools the 5 test lines with the 5
CV-validation lines (BT20, HCC1500, HCC2185, MCF7, UACC3199), each predicted by the
CV model that held it out. Those lines informed the hyperparameter choice. Under this
pooling, the DMM's median is 0.43 (train 0.34). It is 20% below the negative control
(0.53; p = 1·10⁻⁹; better in 8/10 lines, cell-line sign-flip p = 0.01) and comparable
to elastic net (0.41; DMM 5% higher, p = 0.29), lasso (0.40; p = 0.18) and linear
regression (0.41; p = 0.64). Set `HELDOUT = "cv+test"` in the notebook to reproduce
it.

---

## 4. Measurement noise floor: details

### a. Data and scale
These are the measurements the mechanistic models are scored on: the PEtab
measurement table `data/EGFR_MAPK__logobs__dream_cytof__measurements.tsv`, restricted
to `measurementType == "cytof"` and the four model observables (pERK_Y204, pMEK_S222,
pERBB2_Y1248 (p-HER2), pRPS6KA1_S380 (p-p90RSK)). The conditions are EGF, iEGFR and
iMEK; full medium is not scored. The lines are the 63 modelled cell lines. The values
are row for row identical to `data/cytof.csv`: per-well means of arcsinh-transformed
single-cell intensities, the scale on which RMSEs are computed. That gives 6,793
measurements, a median of 112 per cell line.

### b. How replicates are paired
A pair is two measurements of the same cell line, condition, observable and time
point. Every cell line was measured on one `date`, so there are no between-day
replicates.
1. **Time courses A vs. B** (`time_course`). These are interleaved: EGF A has 0, 5.5,
   9, 13, 23 and 40 min; EGF B has 0, 7, 9, 17, 30 and 60 min; the inhibitor courses
   have 0/9/13/40 and 0/7/17/60 min. They share only the **0-min point under every
   condition and the 9-min point under EGF**. Each shared point with exactly one A
   and one B gives one pair: 899 pairs, 61 lines.
2. **Replicate sets.** MDAMB157 and MDAMB436 were measured as two complete sets: same
   date and time-course label, different files (`cytof/data.py`: MDAMB157 "has double
   the amount of fileIDs (biological replicates)"). Within a time course, the two
   values at each time point form a pair. That gives 215 pairs, plus single points of
   CAL148 (iMEK, 0 min) and MCF12A (iEGFR, 0 min). At points with A, A, B and B, which
   A goes with which B is unknown, so these give only within-time-course pairs.

### c. Estimators
- σ = sqrt(mean(d²)/2), pooled over pairs: the SD of a single measurement, i.e. the
  pooled RMSE that exact prediction of every expected value would still incur.
- Box scale: every test combination gets as many residuals as it has measurements,
  drawn with replacement from d/√2. The value is the median per-combination RMSE over
  1,000 draws (`combo_noise_floor`): **0.245** (95% of draws 0.224–0.266). It is the
  same for the training combinations and for the 10-line pooling.

### d. Results

| Pairs | n pairs | cell lines | σ | robust σ (MAD) | replicate RMSE |
|---|---:|---:|---:|---:|---:|
| **All** | 1,114 | 63 | **0.26** | 0.23 | 0.37 |
| Time courses A vs. B | 899 | 61 | 0.23 | 0.19 | 0.33 |
| Replicate sets | 215 | 4 | 0.36 | 0.47 | 0.51 |
| Test lines only (all A vs. B) | 80 | 5 | 0.22 | 0.22 | 0.32 |
| Training lines only | 1,034 | 58 | 0.27 | 0.23 | 0.38 |

By observable (A vs. B): pERK 0.25, pMEK 0.17, p-HER2 0.25, p-p90RSK 0.26. By time
point (A vs. B): EGF 0 min 0.25, EGF 9 min 0.20, iEGFR 0 min 0.26, iMEK 0 min 0.22.
The replicate sets show no trend with time (0.19–0.53).

The positive control's median per-combination test RMSE (0.22) lies slightly below
the 0.25 line. Its pooled RMS (0.265) equals σ (0.263). The per-line fits estimate up
to 52 parameters (44 kinetic, 8 observable scale/offset) from about 110 measurements
of the same line, so they absorb part of the replicate noise.

Caveats:
- The A/B time courses come from one experiment on one day, so they leave out
  experiment-to-experiment variability. The lines measured as two complete sets show
  set-level offsets of up to about 0.7 that persist over the whole time course
  (MDAMB436: about 0.6 in pERK, pMEK and p-p90RSK). The floor of 0.26 is therefore a
  lower bound for a newly measured line.
- Adding the three lines in the data that are not modelled (HCC2157, MCF10F,
  MDAkb2) gives σ = 0.34 for replicate sets (371 pairs, 6 lines) and 0.23 for A vs. B.

---

## 5. Methods 8 check

Drafted claim: "the regression baselines predict the measured CyTOF dynamics directly
from the DMM's input features with the same CV folds as DMM tuning".

Verdict: **inputs and outputs are as drafted, with qualifications. "Same CV folds as
DMM tuning" is false for the reported numbers.** Only the current code uses the DMM
folds.

- **Inputs: the DMM's input features. True.** They come from the same selected-feature
  file as the DMM for that context, feature set and split (`FEATURES_OUTFILE`, read
  with `process_features`). Feature selection is per split, on its training lines.
  The pipeline standardizes the inputs (`StandardScaler`) and imputes them with
  `KNNImputer` (k = 5).
- **Outputs: measured CyTOF dynamics, directly. True, with processing.** The targets
  are the `cytof_dynamic` contextualization: one output per observable (pERK, pMEK,
  p-HER2, p-p90RSK) × condition (EGF, iEGFR, iMEK) × time point. They are **averaged
  over replicates** and **aligned to a common time grid** (`harmonise_cytof_dynamic`):
  13 min from 12 or 14, 17 min from 15 or 16 (inhibitors), 60 from 40, 40 from 35
  (EGF), and 7, 13 and 40 min interpolated linearly. Outputs missing in ≥ 50% of
  training lines are dropped; remaining missing training targets are KNN-imputed.
  There is **one joint multi-output model**; the multi-task penalties select a common
  feature set. No mechanistic model is involved.
- **CV folds. False for the reported numbers.** α (20 candidates) is chosen by
  `MultiTaskLassoCV` / `MultiTaskElasticNetCV`, and the model is refitted on all
  training lines. The current `evaluate_regressors.py` passes `PredefinedSplit` over
  the DMM folds: each fold validates on one CV line, and split `all` uses all 5. The
  April outputs, however, were made with `cv=5`, unshuffled 5-fold `KFold` over the
  58 training lines (table below). Test lines enter neither scheme. OLS has no
  hyperparameter. The elastic-net mixing parameter is **fixed at l1_ratio = 0.5**,
  not tuned.
- **Scoring.** The pipeline scores against the averaged, gap-filled targets
  (section 3a). The numbers in this note score against individual measurements.

**CV check.** I refitted the split-`all` regressions as `evaluate_regressors.py` does,
from the April feature files
(`features/EGFR_MAPK__logobs/dream_cytof/{train,val}/cytof_init__all__RFE_8_permute.csv`)
and the PEtab tables. Each was compared with the stored predictions in
`trajectories_figure1a.csv` (5,195 training and 460 test values per model):

| Model | CV | selected α | max abs. difference to stored |
|---|---|---:|---:|
| Linear regression | (none) | – | 4e-15 |
| Lasso | default `cv=5` | 0.171 | **3e-15** |
| Lasso | DMM folds | 0.040 | 0.27 |
| Elastic net | default `cv=5` | 0.165 | **4e-15** |
| Elastic net | DMM folds | 0.039 | 0.28 |

The exact match of linear regression confirms that inputs, targets and preprocessing
were reconstructed correctly. The CV splits were not checked; they ran the same code
at the same time.

**Decision for the authors.**
- **Variant 1:** keep the April numbers and describe the 5-fold split. No rerun is
  needed. However, the repository code (DMM folds) no longer reproduces the paper's
  regression numbers unless `evaluate_regressors.py` goes back to `cv=5`.
- **Variant 2 (recommended, if a cluster run is acceptable):** rerun
  `evaluate_regressors` for all `figure1a` splits on NEMO (CPU only, minutes per
  split), then `evaluate_all --figure=figure1a` and `figure_1.ipynb`. Expected
  numbers are in section 3b. Code, Methods and numbers then agree.

---

## 6. Other notes
- **Legend label.** `figure_config.MODEL_LABELS["elasticnet"]` is "linear
  regression", so the elastic-net trajectories were labelled "linear regression".
  With OLS now shown as "Linear reg.", `figure_1.ipynb` overrides the label locally
  to "Elastic net". `figure_config.py` (S4's file) is unchanged.
- **Notebook data.** `figure_1.ipynb` now also reads the PEtab measurement and
  observable tables in `data/`.
