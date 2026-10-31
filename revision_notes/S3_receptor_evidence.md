# S3: receptor levels, model-independent evidence (Reviewer 2, major point 2)

Branch `claude/revision-receptor-evidence-b04f76`.

- Code: `figures_paper/receptor_evidence.py`
- Figure notebook: `figures_paper/figure_S_receptor_evidence.ipynb` (panels A–D, saved as `figS_receptor_panel{A,B,C,D}.pdf`)

All numbers below come from that notebook. Data: `EGFR_MAPK__logobs__dream_cytof__*.tsv` (the PEtab tables the encoder features are built from), 63 cell lines (`cytof.get_samples`).

## Flags: read these first

1. **ERBB2 vs. phospho-ERBB2: the draft's "neither … correlate" is too strong.** With baseline p-ERBB2 defined as the encoder defines it (n = 63, 14 of them KNN-imputed), neither correlation is significant: transcript r = 0.17 (P = 0.18), protein r = 0.23 (P = 0.07). With only the cell lines whose baseline was measured, both correlations are weak but nominally significant: transcript r = 0.32 (P = 0.026, n = 49), protein r = 0.32 (P = 0.026, n = 48). These r values mean receptor level accounts for at most about 10 % of the variance in p-ERBB2. Neither level correlates with the EGF-induced change in p-ERBB2 (r = −0.03 and 0.11). Corrected wording is below.
2. **The CyTOF panel has no phospho-EGFR or phospho-ERBB3 channel.** The only phospho-receptor readout is `p.HER2` (ERBB2 Y1248, model observable `pERBB2_Y1248`). The planned EGFR-vs-pEGFR and ERBB3-vs-pERBB3 contrasts therefore cannot be computed. As context I used EGF-induced responses (maximum over the EGF time course minus baseline) of the four model observables. EGFR transcript is the positive contrast there: r = 0.47–0.59 with all four responses, q ≤ 0.001.
3. **Two other receptor levels show univariate associations, so "receptor levels other than EGFR transcripts are not predictive" needs qualifying.**
   - ERBB3 transcript correlates negatively with EGF-induced ΔpMEK, ΔpERK and Δp90RSK: r = −0.47, −0.43 and −0.38, q ≤ 0.009.
     - ERBB3 and EGFR transcripts are themselves anticorrelated: r = −0.51, ρ = −0.65, n = 63.
     - Adjusted for EGFR transcript, the partial r is −0.30 (ΔpMEK, P = 0.02), −0.25 (ΔpERK, P = 0.048) and −0.12 (Δp90RSK, P = 0.35). All Spearman partials have P ≥ 0.12.
   - EGFR protein correlates with EGF-induced ΔpERBB2 (r = 0.45, q = 0.007, n = 46). It does not correlate with the downstream responses (|r| ≤ 0.21, q ≥ 0.30).
   - ERBB2 transcript and protein show no association with any of the five readouts (|r| ≤ 0.23, q ≥ 0.23).
4. **The HVG prefilter in `select_features.py` removes an arbitrary 20 % of features.** `prepare_inputs_for_context` mean-centers the inputs before `get_selected_features` calls `get_hvg`. The "drop the 20 % lowest means" step therefore compares floating-point residuals of zero (|mean| < 3e-15). For the receptors this means, before any random forest sees them:
   - EGFR protein (variance rank 128–152 of ~6,180) is removed in 3 of 6 splits.
   - ERBB2 protein (variance rank 25–26) is removed in 5 of 6 splits.
   - ERBB2 and ERBB3 transcripts would be excluded anyway: their variance ranks are 1,034–1,076 and 1,528–1,573 of 16,562, outside the 501 kept (panel D).

   I did not change `select_features.py`, because it is out of scope. The artifact affects every `HVGRFE_*` feature set (the transcriptomics and proteomics contexts, Fig. 1 and later). It also makes the HVG set depend on floating-point details: 9 of the 10 saved `HVGRFE` selections fall inside my replicated HVG sets, but proteomics/BT20 has 1 of its 10 selected features outside. Whether to fix this and rerun is the authors' decision; see the follow-up below.
6. **Random-forest importance does not single out EGFR transcript.** On all features of a modality, every receptor level ranks mid-distribution, and its rank varies about tenfold across splits (section 2a). The RF evidence supports "receptor levels are not among the most important features". It does not support "EGFR transcript is, the others are not". In the pipeline's own first round (the HVG set), EGFR transcript is the only receptor present in all splits (median rank 130.5 of 501). That is mainly a statement about the HVG prefilter (flag 4).
5. **The random-forest targets are the CyTOF dynamics, not kinetic parameters.** `select_features.py` fits the RF to the KNN-imputed, centered `cytof_dynamic` table (the four model observables × 3 conditions × time points, 96 columns). The response's phrase "random-forest models used for feature selection (Methods, Section 5)" is accurate. Methods §5 should name this target if it does not already.

## 1. Receptor level vs. phospho-ERBB2 across cell lines

Baseline p-ERBB2 comes from `load_data("cytof_init", …)`: the EGF-condition, t = 0 value, replicate mean. The 14 cell lines without a measured t = 0 value get KNN-imputed values, as the encoder sees them. The only difference: here the imputer is fitted on all 63 lines, whereas the encoder fits it on each split's training lines. For measured cell lines the imputed and raw values are identical (asserted in code). Receptor levels are the replicate means of the PEtab transcriptomics and proteomics tables: the values mechanistic integration rescales by, on log2 scale.

| Receptor level | Baseline p-ERBB2 (encoder definition): n | r (P) | ρ (P) | Measured baseline only: n | r (P) | ρ (P) |
|---|---|---|---|---|---|---|
| ERBB2 transcript | 63 | 0.17 (0.18) | 0.19 (0.14) | 49 | 0.32 (0.026) | 0.35 (0.013) |
| ERBB2 protein | 62 | 0.23 (0.068) | 0.15 (0.25) | 48 | 0.32 (0.026) | 0.24 (0.10) |
| EGFR transcript | 63 | 0.12 (0.35) | 0.09 (0.50) | 49 | 0.12 (0.43) | 0.06 (0.70) |
| EGFR protein | 59 | 0.15 (0.26) | 0.22 (0.087) | 46 | 0.07 (0.64) | 0.18 (0.25) |
| ERBB3 transcript | 63 | 0.14 (0.26) | −0.04 (0.74) | 49 | 0.24 (0.094) | 0.02 (0.90) |
| ERBB3 protein | 33 | −0.16 (0.37) | −0.20 (0.26) | 26 | −0.22 (0.27) | −0.20 (0.34) |

Restricting to the 58 training cell lines of split `all` gives the same picture: ERBB2 transcript r = 0.19 (P = 0.16), ERBB2 protein r = 0.24 (P = 0.069).

Transcript vs. protein agreement:

| Receptor | n | r (P) | ρ (P) |
|---|---|---|---|
| ERBB2 | 62 | 0.77 (4e-13) | 0.69 (4e-10) |
| EGFR | 59 | 0.44 (5e-4) | 0.54 (1e-5) |
| ERBB3 | 33 | 0.28 (0.11) | 0.25 (0.17) |

ERBB3 protein is quantified in only 33 of the 63 lines.

### Context: EGF-induced responses (panel B)

Pearson r; Benjamini–Hochberg q is computed over all 30 pairs. Δ = maximum over the EGF time course minus t = 0, from the harmonised `cytof_dynamic` table without imputation. ΔpERBB2 therefore has n = 49 (fewer where a receptor level is missing).

| Receptor level | Baseline pERBB2 | ΔpERBB2 | ΔpMEK | ΔpERK | Δp90RSK |
|---|---|---|---|---|---|
| EGFR transcript | 0.12 | **0.59** (q = 1e-4) | **0.48** (q = 6e-4) | **0.47** (q = 6e-4) | **0.57** (q = 3e-5) |
| ERBB2 transcript | 0.17 | −0.03 | −0.10 | −0.05 | −0.18 |
| ERBB3 transcript | 0.14 | −0.25 | **−0.47** (q = 6e-4) | **−0.43** (q = 0.002) | **−0.38** (q = 0.009) |
| EGFR protein | 0.15 | **0.45** (q = 0.007) | 0.13 | −0.08 | 0.21 |
| ERBB2 protein | 0.23 | 0.11 | 0.02 | −0.00 | −0.13 |
| ERBB3 protein | −0.16 | 0.10 | 0.21 | 0.10 | −0.07 |

Bold: q < 0.05. Spearman agrees on every bold cell. It additionally flags ERBB3 transcript vs. ΔpERBB2 (ρ = −0.45, q = 0.005).

## 2. Random-forest importance (panel C)

This replicates the first RFE round of `select_features.py` (`HVGRFE_*` branch):

- **Targets:** the training lines' `cytof_dynamic` table, KNN-imputed and centered (96 columns).
- **Inputs:** the modality's features that pass the `load_data` missingness filter (< 30 % missing), KNN-imputed, mean-centered, then standardized.
- **Estimator:** `RandomForestRegressor(max_features=0.8)` with 100 trees and impurity importance. That is the pipeline's rule for more than 500 features.
- **Splits:** the five CV splits of Fig. 1 (BT20, HCC1500, HCC2185, MCF7, UACC3199; 57 training lines each) and split `all` (58 lines; Figs. 2–4).

Rank = 1 + the number of features with higher importance. Percentile = the share of features with lower importance; the median feature of a modality sits at 50.

### 2a. Among all features of the modality (panel C)

Importance is averaged over 5 seeds (42, the pipeline's seed, plus 43–46).

| Receptor level | Features | Median rank (range) over 6 splits | Median percentile (range) | Splits with importance 0 |
|---|---|---|---|---|
| EGFR transcript | 16,562 | 5,508 (1,010–10,047) | 67 (0–94) | 2 (HCC1500, MCF7) |
| ERBB2 transcript | 16,562 | 4,946 (1,120–9,935) | 70 (0–93) | 1 (UACC3199) |
| ERBB3 transcript | 16,562 | 2,997 (590–10,026) | 82 (0–96) | 1 (BT20) |
| EGFR protein | ~6,180 | 2,920 (297–4,123) | 53 (33–95) | 0 |
| ERBB2 protein | ~6,180 | 2,438 (1,302–5,419) | 61 (12–79) | 0 |
| ERBB3 protein | n/a | not ranked: quantified in only 33 of 63 lines, removed by the missingness filter in every split | n/a | n/a |

Share of single forests (one split × one seed, 30 per modality) in which a feature is never used for a split:

| Modality | All features | EGFR | ERBB2 | ERBB3 |
|---|---|---|---|---|
| Transcripts | 81 % | 80 % | 67 % | 63 % |
| Proteins | 57 % | 53 % | 53 % | n/a |

Per-split ranks are in the notebook (`ranks`).

### 2b. The pipeline's own first round (HVG set, 501 features, seed 42; panel D)

| Receptor level | Ranked among the 501 HVG candidates | Excluded before ranking |
|---|---|---|
| EGFR transcript | 6 of 6 splits; rank 30–362 (median 130.5) | none |
| ERBB2 transcript | none | variance step, 6 of 6 |
| ERBB3 transcript | none | variance step, 6 of 6 |
| EGFR protein | 3 of 6 (BT20 57, MCF7 118, UACC3199 408) | mean step, 3 of 6 (HCC1500, HCC2185, all) |
| ERBB2 protein | 1 of 6 (all: 342) | mean step, 5 of 6 |
| ERBB3 protein | none | missingness filter, 6 of 6 |

No receptor level appears in any saved pipeline selection: the `transcriptomics … HVGRFE_15`, `proteomics … HVGRFE_10` and `multimodal … RFE_10` sets for the five CV splits, in `features/` of the main checkout. Selections for split `all` and for the sizes in `BEST_FEATURE_SETS` (`HVGRFE_12`, `HVGRFE_8`) are not on this machine, so I could not check those.

## Paste-ready text

US spelling. "Fig. S_" is the new figure. Panels: A (scatters), B (correlation heatmap), C (RF percentiles), D (first-round status).

### Response to Reviewer 2, major point 2

Replace the ERBB2 sentence ("For ERBB2, neither transcript nor protein levels correlate … [RESULT …]") with:

> For ERBB2, transcript and protein levels were at most weakly correlated with baseline phospho-ERBB2 levels across cell lines (Pearson r = 0.17, P = 0.18, n = 63 for transcript; r = 0.23, P = 0.07, n = 62 for protein; r = 0.32, P = 0.03 for both when restricted to the 49 and 48 cell lines with a measured, non-imputed baseline; new Fig. S_A), and neither was correlated with the EGF-induced change in phospho-ERBB2 (r = −0.03 and r = 0.11, P ≥ 0.46), whereas EGFR transcript levels were (r = 0.59, P < 10⁻⁴; Fig. S_B).

Replace the random-forest clause ("in the random-forest models used for feature selection (Methods, Section 5), receptor transcript and protein levels [RESULT …]") with:

> in random forests trained as for feature selection (Methods, Section 5) on all features of the respective modality, receptor transcript and protein levels ranked in the middle of the importance distribution (median rank across the six data splits: 5,508, 4,946 and 2,997 of 16,562 transcripts for EGFR, ERBB2 and ERBB3; 2,920 and 2,438 of about 6,180 proteins for EGFR and ERBB2; ERBB3 protein was quantified in only 33 of 63 cell lines and therefore not considered; Fig. S_C).

If the response also says that receptor levels other than EGFR transcripts "are not predictive" in general, flag 3 contradicts that at the univariate level. This more precise, numbers-only version could replace it:

> Apart from EGFR transcript levels, receptor levels showed no consistent association with the measured signaling dynamics: ERBB2 transcript and protein levels were not associated with baseline or EGF-induced phospho-ERBB2, phospho-MEK, phospho-ERK or phospho-p90RSK (|r| ≤ 0.23, q ≥ 0.23); EGFR protein levels correlated with the EGF-induced phospho-ERBB2 response (r = 0.45) but not with downstream responses (|r| ≤ 0.21); and the negative correlation of ERBB3 transcript levels with EGF-induced downstream responses (r = −0.38 to −0.47) was attenuated after adjusting for EGFR transcript levels (partial r = −0.12 to −0.30), with which they are anticorrelated (r = −0.51; Fig. S_B).

Avoid "none of the receptor levels was selected by the feature selection". It is true for the saved CV-split selections, but EGFR and ERBB2 protein were mostly not candidates, because of the HVG artifact (flag 4).

### Results

Replace "neither ERBB2 transcript nor protein levels correlated with phospho-ERBB2 levels across cell lines ([r = …], Fig. S_), and receptor levels [ranked … in random-forest feature importance] (Fig. S_)" with:

> ERBB2 transcript and protein levels were at most weakly correlated with baseline phospho-ERBB2 levels across cell lines (r = 0.17 and 0.23, P ≥ 0.07; Fig. S_A) and not correlated with its EGF-induced change (r = −0.03 and 0.11; Fig. S_B), and in random forests trained on all transcripts or proteins, receptor levels ranked in the middle of the feature-importance distribution (median percentile 53–82 across data splits; Fig. S_C)

### Figure legend (draft)

> **Figure S_. Receptor levels and phospho-signaling across cell lines.** (A) Receptor transcript (top) and protein (bottom) levels versus baseline phospho-ERBB2 (CyTOF, ERBB2 Y1248; EGF condition, t = 0) across cell lines. Open circles, baseline value KNN-imputed as for the encoder input. Dashed line, least-squares fit; r, Pearson; ρ, Spearman. (B) Pearson correlation of receptor levels with baseline phospho-ERBB2 and with the EGF-induced response (maximum over the EGF time course minus baseline) of the four model observables. \*, Benjamini–Hochberg q < 0.05 over the 30 pairs. (C) Random-forest feature importance of receptor levels for the CyTOF signaling dynamics, trained as in feature selection (Methods, Section 5) on all features of the respective modality, shown as the percentage of features with lower importance (mean over five seeds) for each data split; bars, median; open symbols, importance zero. ERBB3 protein was quantified in fewer than 70 % of cell lines and excluded. (D) Fate of receptor levels in the first round of the feature-selection pipeline per data split: rank among the 501 most variable features, or the filter step that excluded them.

Panel D's "mean step" category describes the artifact in flag 4. If the authors fix the prefilter, D must be regenerated, or dropped and the legend cut to A–C.

### Methods (suggested addition to Section 5)

> To assess receptor levels independently of the DMM, we correlated EGFR, ERBB2 and ERBB3 transcript and protein levels (replicate means) across cell lines with baseline phospho-ERBB2 (CyTOF ERBB2 Y1248, EGF condition, t = 0, preprocessed as the CyTOF encoder input) and with the EGF-induced response of the four model observables (maximum over the EGF time course minus baseline), using Pearson and Spearman correlation with Benjamini–Hochberg correction. Random-forest importances were computed with the estimator, targets and preprocessing of the first feature-selection round, but on all features of the respective modality without the variance prefilter, averaging impurity-based importances over five random seeds per data split.

## For S6 (workflow)

- No change to `run_notebooks.py` is needed. Its current main-branch version discovers committed notebooks via `git ls-files figures_paper/*.ipynb`, and S6's branch also finds uncommitted ones. The notebook is not under a "(not in paper)" directory, so `--paper-only` includes it.
- Runtime: the first run refits 72 random forests in one cell. That took 23 min under `taskpolicy -c utility` with `n_jobs = 3`. The result is cached as parquet under `eval/EGFR_MAPK__logobs/dream_cytof/receptor_evidence/`; parquet rather than CSV because `run_notebooks.py` redirects `to_csv`. Later runs take under a minute. This fits the default 1800 s cell timeout, but only just on a busy machine.
- Inputs: only the PEtab tables `data/EGFR_MAPK__logobs__dream_cytof__{measurements,observables}.tsv`, i.e. the `process_data` output for model `EGFR_MAPK__logobs`. No new figure configuration is needed. The replication check also reads `features/EGFR_MAPK__logobs/dream_cytof/train/*HVGRFE*` when present and skips it otherwise.
- S6's note asks whether S3 owns the PXD017199 / `figure_S7` scripts. It does not: this task did not touch that analysis.

## Follow-up (not done here)

- **HVG prefilter.** In `select_features.py`, either apply the mean step before centering or drop it, then rerun the `HVGRFE_*` feature selection and the affected figures. This changes `select_features.py` behavior, so it needs the authors' decision. It also touches S2's feature scan.

## Reproduce

```bash
taskpolicy -c utility ./venv/bin/python run_notebooks.py figure_S_receptor_evidence
```
