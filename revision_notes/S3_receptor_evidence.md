# S3: receptor levels, model-independent evidence (Reviewer 2, major point 2)

Branch `claude/revision-receptor-evidence-b04f76`.

- Code: `figures_paper/receptor_evidence.py`
- Figure notebook: `figures_paper/figure_S_receptor_evidence.ipynb` (panels A–E, saved as `figS_receptor_panel{A,B,C,D,E}.pdf`)

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
4. **The HVG prefilter bug is fixed; this analysis uses the fix.** `prepare_inputs_for_context` mean-centered the inputs before `get_hvg`, so its "drop the 20 % lowest means" step compared floating-point residuals of zero.
   - The fix is commit 50bd9e1 (fix B: no mean step, keep the 501 most variable features). It is on branch `claude/objective-rubin-5b7968` and not yet on `main`. The figure1c feature scan was rerun with it on NEMO; see `revision_notes/hvg_prefilter.md` on that branch.
   - `receptor_evidence.get_hvg` copies the fixed version.
   - Before the fix, EGFR protein (variance rank 128–152 of ~6,180) was dropped in 3 of 6 splits and ERBB2 protein (variance rank 25–26) in 5 of 6. With the fix, both are candidates in all 6.
   - ERBB2 and ERBB3 transcripts are still not candidates. Their variance ranks are 1,034–1,076 and 1,528–1,573 of 16,562, outside the 501 kept. ERBB3 protein is still removed by the missingness filter. Of the six receptor measures, only three therefore reach the pipeline's random forest (panel D).
5. **Random-forest importance does not single out EGFR transcript.**
   - On all features of a modality, every receptor level ranks mid-distribution, and its rank varies about tenfold across splits (section 2a).
   - Within the pipeline's first round (501 HVG candidates, section 2b), EGFR transcript, EGFR protein and ERBB2 protein all rank mid-pack (median 176.5, 212.5 and 315.5 of 501).
   - A single forest, which is what the pipeline uses, can put the same receptor anywhere from about rank 20 to 480, depending on its seed.
   - In the full RFE, as run with the fix for the Fig. 1C scan (section 2c), no receptor level is in any selection of 24 or fewer features in any of the five CV splits. That includes the final models' sizes, 12 transcripts and 8 proteins. EGFR protein in HCC1500 lasts longest: it is kept down to 28 features.
   - Split `all` (final models) has no post-fix pipeline files yet. An exact replication of the pipeline's RFE in its container finds no receptor level in any of its selections of 256 or fewer features (section 2d). The final feature sets regenerated with the fix will therefore contain none.
   - How deep a receptor gets is not robust. Outside the container, with another scikit-learn version and platform, the depths change. The conclusion holds either way.
   - The RF evidence therefore supports "receptor levels are not among the most important features". It does not support "EGFR transcript is, the others are not".
6. **The random-forest targets are the CyTOF dynamics, not kinetic parameters.** `select_features.py` fits the RF to the KNN-imputed, centered `cytof_dynamic` table (the four model observables × 3 conditions × time points, 96 columns). The response's phrase "random-forest models used for feature selection (Methods, Section 5)" is accurate. Methods §5 should name this target if it does not already.

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

### 2b. The pipeline's own first round (fixed HVG set, 501 features; panel D)

Importance is averaged over the same 5 seeds. Seed 42 alone is the pipeline's forest, but single-forest ranks are too noisy to report: for example, EGFR transcript in split `all` ranges from 20 to 234 across seeds, and EGFR protein in BT20 from 41 to 422.

| Receptor level | Ranked among the 501 HVG candidates (BT20, HCC1500, HCC2185, MCF7, UACC3199, all) | Median | Excluded before ranking |
|---|---|---|---|
| EGFR transcript | 199, 343, 154, 216, 125, 48 | 176.5 | none |
| EGFR protein | 139, 46, 334, 173, 329, 252 | 212.5 | none |
| ERBB2 protein | 380, 219, 198, 382, 281, 350 | 315.5 | none |
| ERBB2 transcript | not ranked | n/a | not among the 501 most variable, 6 of 6 |
| ERBB3 transcript | not ranked | n/a | not among the 501 most variable, 6 of 6 |
| ERBB3 protein | not ranked | n/a | missingness filter, 6 of 6 |

Before the fix (first version of this note), EGFR protein was a candidate in 3 of 6 splits and ERBB2 protein in 1 of 6.

### 2c. Survival through the full RFE (Fig. 1C feature-count scan; panel E)

Source: the `figure1c` feature files regenerated with the fix on NEMO, dated 2 Nov 2026. These are `features/EGFR_MAPK__logobs/dream_cytof/train/{transcriptomics,proteomics}__<split>__HVGRFE_<N>_permute.csv` for N = 4 … 4,096, 170 files in all.

- All 170 selections lie inside the locally replicated HVG sets.
- The 10 selections with N ≥ 512 hold all 501 candidates, and they match the replicated sets exactly.
- The scan covers the five CV splits only.

The table gives the smallest selection that still contains the receptor, in number of features. Every larger selection contains it too.

| Receptor level | BT20 | HCC1500 | HCC2185 | MCF7 | UACC3199 | Median |
|---|---|---|---|---|---|---|
| EGFR transcript | 501 | 256 | 64 | 256 | 256 | 256 |
| EGFR protein | 501 | 28 | 501 | 128 | 256 | 256 |
| ERBB2 protein | 501 | 501 | 256 | 501 | 128 | 501 |
| ERBB2 transcript, ERBB3 transcript, ERBB3 protein | never candidates (section 2b) | | | | | |

How to read the table:

- 501 means the receptor is already gone from the 256-feature selection.
- The scanned sizes between 32 and 501 are 64, 128 and 256. So 256 means "kept at 256, gone at 128".

Findings:

- No receptor level is in any selection of 24 or fewer features in any split. That includes the final models' sizes in `BEST_FEATURE_SETS`: 12 transcripts and 8 proteins.
- EGFR protein in HCC1500 lasts longest. It is kept down to 28 of 501 features and gone at 24.
- The multimodal selections (`multimodal__<split>__RFE_<N>_permute`, N ≤ 32) concatenate the same omics selections; all 100 checked are identical. pEGFR therefore appears only in HCC1500, at N = 28 and 32.
- **Split `all` is not part of the scan.** It holds the final models of Figs. 2–4. Its feature files on NEMO predate the fix: `transcriptomics__all__HVGRFE_12`, `proteomics__all__HVGRFE_8` and the multimodal `RFE_8` date from Apr 2026, and the others from Dec 2025–Feb 2026. None of them contains a receptor level, but they were drawn from the buggy prefilter. Section 2d covers split `all` with the fix.

The older CV-split selections in `features/` of the main checkout (`HVGRFE_15`, `HVGRFE_10`, multimodal `RFE_10`) also predate the fix. None of them contains a receptor level.

### 2d. Split `all` (final models): exact replication in the pipeline container

Split `all`'s feature files on NEMO predate the fix, so I reran the pipeline's RFE for all six splits at sizes 4–256.

- **How.** `receptor_evidence.rfe_path` copies the `HVGRFE_*` loop of `select_features.py`. I ran it on the fixed HVG candidates inside the pipeline's container on NEMO: SLURM job 62741011, scikit-learn 1.7.1, numpy 2.3.1, x86_64.
  - The job read the shared checkout's data.
  - It wrote only to `/nemo/lab/froehlichf/home/users/frohlif/s3_rfe_container/`, which holds the script, log and `rfe_selections_container.csv`.
- **The replication is exact.** For the five CV splits, all 130 selections (sizes 4–256) are identical to the pipeline's `figure1c` files, including the feature order.
- **Split `all` result.** EGFR transcript, EGFR protein and ERBB2 protein are among the 501 candidates, but none is in any selection of 256 or fewer features. Once the final models' feature sets (12 transcripts, 8 proteins) are regenerated with the fix, they will contain no receptor level.
- **The regenerated final sets will differ from the current pre-fix files.** Only 7 of 12 transcripts and 4 of 8 proteins stay the same. The sets with the fix, in order of importance:
  - Transcripts: GJA5, UCP2, CYP4B1, ZG16B, CLIP4, TINAGL1, COL5A1, LXN, PLS3, MYEOV, C1orf116, FAM83A.
  - Proteins: BAZ1A, CMBL, MT1X, PSMB10, ELAPOR1, PLS3, IL18, ALDH1A3.

**Outside the container, the RFE does not reproduce.** I ran the same code locally with scikit-learn 1.8.0 and numpy 2.4.4 on arm64. Only 7 of the 130 CV-split selections matched.

- The median overlap is 75 %. Per selection it ranges from 50 to 100 %, and it is 75–83 % at sizes 8–12.
- The receptor depths changed too:
  - Locally, EGFR protein in HCC1500 is kept down to 64 features rather than 28.
  - Locally, all three receptors in split `all` are in the 256-feature selection.
- Neither run has a receptor level in any selection of 24 or fewer features.
- Only the container run counts. The local sensitivity is one more sign that single-forest RFE depths are not robust (section 2b).

## Paste-ready text

US spelling. "Fig. S_" is the new figure. Panels: A (scatters), B (correlation heatmap), C (RF percentiles), D (first-round status), E (survival through the full RFE).

### Response to Reviewer 2, major point 2

Replace the ERBB2 sentence ("For ERBB2, neither transcript nor protein levels correlate … [RESULT …]") with:

> For ERBB2, transcript and protein levels were at most weakly correlated with baseline phospho-ERBB2 levels across cell lines (Pearson r = 0.17, P = 0.18, n = 63 for transcript; r = 0.23, P = 0.07, n = 62 for protein; r = 0.32, P = 0.03 for both when restricted to the 49 and 48 cell lines with a measured, non-imputed baseline; new Fig. S_A), and neither was correlated with the EGF-induced change in phospho-ERBB2 (r = −0.03 and r = 0.11, P ≥ 0.46), whereas EGFR transcript levels were (r = 0.59, P < 10⁻⁴; Fig. S_B).

Replace the random-forest clause ("in the random-forest models used for feature selection (Methods, Section 5), receptor transcript and protein levels [RESULT …]") with:

> in random forests trained as for feature selection (Methods, Section 5) on all features of the respective modality, receptor transcript and protein levels ranked in the middle of the importance distribution (median rank across the six data splits: 5,508, 4,946 and 2,997 of 16,562 transcripts for EGFR, ERBB2 and ERBB3; 2,920 and 2,438 of about 6,180 proteins for EGFR and ERBB2; ERBB3 protein was quantified in only 33 of 63 cell lines and therefore not considered; Fig. S_C).

Optionally, continue with the pipeline's own recursive feature elimination (sections 2c and 2d):

> Within the feature-selection pipeline itself, which applies recursive feature elimination to the 501 most variable transcripts or proteins, ERBB2 and ERBB3 transcript levels were not among the most variable features and ERBB3 protein levels did not pass the missingness filter. EGFR transcript, EGFR protein and ERBB2 protein levels were eliminated before the feature sets reached 24 features in each of the five cross-validation splits (median smallest feature set retaining them: 256, 256 and 501 features; Fig. S_D,E) and before they reached 256 features in the data split used for the final models, whose feature sets comprise 12 transcripts and 8 proteins.

The last clause cites the container replication (section 2d). It is exact, but the pipeline's own split-`all` files are regenerated only when Figs. 2–4 are rerun with the fix. If the response should cite only pipeline output, end the sentence after "Fig. S_D,E".

If the response also says that receptor levels other than EGFR transcripts "are not predictive" in general, flag 3 contradicts that at the univariate level. This more precise, numbers-only version could replace it:

> Apart from EGFR transcript levels, receptor levels showed no consistent association with the measured signaling dynamics: ERBB2 transcript and protein levels were not associated with baseline or EGF-induced phospho-ERBB2, phospho-MEK, phospho-ERK or phospho-p90RSK (|r| ≤ 0.23, q ≥ 0.23); EGFR protein levels correlated with the EGF-induced phospho-ERBB2 response (r = 0.45) but not with downstream responses (|r| ≤ 0.21); and the negative correlation of ERBB3 transcript levels with EGF-induced downstream responses (r = −0.38 to −0.47) was attenuated after adjusting for EGFR transcript levels (partial r = −0.12 to −0.30), with which they are anticorrelated (r = −0.51; Fig. S_B).

"None of the receptor levels was selected by the feature selection" now holds for the five CV splits, at every selection size of 24 or fewer features (section 2c). It also holds for split `all`, which holds the final models of Figs. 2–4: in the exact container replication, no receptor level is in any selection of 256 or fewer features (section 2d).

### Results

Replace "neither ERBB2 transcript nor protein levels correlated with phospho-ERBB2 levels across cell lines ([r = …], Fig. S_), and receptor levels [ranked … in random-forest feature importance] (Fig. S_)" with:

> ERBB2 transcript and protein levels were at most weakly correlated with baseline phospho-ERBB2 levels across cell lines (r = 0.17 and 0.23, P ≥ 0.07; Fig. S_A) and not correlated with its EGF-induced change (r = −0.03 and 0.11; Fig. S_B), and in random forests trained on all transcripts or proteins, receptor levels ranked in the middle of the feature-importance distribution (median percentile 53–82 across data splits; Fig. S_C)

### Figure legend (draft)

> **Figure S_. Receptor levels and phospho-signaling across cell lines.** (A) Receptor transcript (top) and protein (bottom) levels versus baseline phospho-ERBB2 (CyTOF, ERBB2 Y1248; EGF condition, t = 0) across cell lines. Open circles, baseline value KNN-imputed as for the encoder input. Dashed line, least-squares fit; r, Pearson; ρ, Spearman. (B) Pearson correlation of receptor levels with baseline phospho-ERBB2 and with the EGF-induced response (maximum over the EGF time course minus baseline) of the four model observables. \*, Benjamini–Hochberg q < 0.05 over the 30 pairs. (C) Random-forest feature importance of receptor levels for the CyTOF signaling dynamics, trained as in feature selection (Methods, Section 5) on all features of the respective modality, shown as the percentage of features with lower importance (mean over five seeds) for each data split; bars, median; open symbols, importance zero. ERBB3 protein was quantified in fewer than 70 % of cell lines and excluded. (D) Receptor levels in the first round of the feature-selection pipeline per data split: importance rank (mean over five seeds) among the 501 most variable features of the modality, or the filter that excluded them. (E) Smallest feature set in the recursive feature elimination (feature-count scan of Fig. 1C) that still contained the receptor level, per cross-validation split; every larger feature set contained it too. Dashed lines, feature-set size of the final models.

Panels D and E use the fixed HVG prefilter (commit 50bd9e1). They are consistent with the pipeline only once that fix is on `main`.

### Methods (suggested addition to Section 5)

> To assess receptor levels independently of the DMM, we correlated EGFR, ERBB2 and ERBB3 transcript and protein levels (replicate means) across cell lines with baseline phospho-ERBB2 (CyTOF ERBB2 Y1248, EGF condition, t = 0, preprocessed as the CyTOF encoder input) and with the EGF-induced response of the four model observables (maximum over the EGF time course minus baseline), using Pearson and Spearman correlation with Benjamini–Hochberg correction. Random-forest importances were computed with the estimator, targets and preprocessing of the first feature-selection round, but on all features of the respective modality without the variance prefilter, averaging impurity-based importances over five random seeds per data split. How far receptor levels were retained by the recursive feature elimination was read from the feature sets of the feature-count scan (Fig. 1C).

## For S6 (workflow)

- No change to `run_notebooks.py` is needed. Its current main-branch version discovers committed notebooks via `git ls-files figures_paper/*.ipynb`, and S6's branch also finds uncommitted ones. The notebook is not under a "(not in paper)" directory, so `--paper-only` includes it.
- Runtime: the first run refits 120 random forests in one cell: about 25 min under `taskpolicy -c utility` with `n_jobs = 3` (the 60 all-feature fits took 23 min, the 60 first-round fits 1.5 min). The result is cached as parquet under `eval/EGFR_MAPK__logobs/dream_cytof/receptor_evidence/`; parquet rather than CSV because `run_notebooks.py` redirects `to_csv`. Later runs take under a minute. This fits the default 1800 s cell timeout, but only just on a busy machine.
- Inputs: only the PEtab tables `data/EGFR_MAPK__logobs__dream_cytof__{measurements,observables}.tsv`, i.e. the `process_data` output for model `EGFR_MAPK__logobs`. No new figure configuration is needed.
- Optional inputs: the `figure1c` feature files `features/EGFR_MAPK__logobs/dream_cytof/train/{transcriptomics,proteomics}__<split>__HVGRFE_<N>_permute.csv`. The replication check and panel E read them; without them, both are skipped. In the Snakefile, the notebook therefore belongs after the `figure1c` feature selection. Scan sizes, splits and final feature-set sizes are imported from `training_configuration.py` (`CONTEXTS_FEATURES_1C`, `SPLITS_1C`, `BEST_FEATURE_SETS`).
- S6's note asks whether S3 owns the PXD017199 / `figure_S7` scripts. It does not: this task did not touch that analysis.

## Follow-up

- **HVG prefilter.** Fixed in commit 50bd9e1 on `claude/objective-rubin-5b7968`; the figure1c scan was rerun on NEMO. That branch has to reach `main` before this PR's panels D and E match the code on `main`.
- **Split `all` feature files predate the fix.** On NEMO, the final models' feature sets for Figs. 2–4 (`*__all__HVGRFE_12/8`, multimodal `RFE_8`) were selected before the fix. They need regenerating before Figs. 2–4 are rerun with the fix. Section 2d predicts the outcome: no receptor level, and 5 of 12 transcripts and 4 of 8 proteins change.

## Reproduce

Panel E needs the regenerated `figure1c` feature files locally. Copy them from NEMO; the files are gitignored:

```bash
rsync -a --include='transcriptomics__*__HVGRFE_*_permute.csv' --include='proteomics__*__HVGRFE_*_permute.csv' --exclude='*' login008.nemo.thecrick.org:/nemo/lab/froehlichf/home/shared/DeepMechanisticModels/features/EGFR_MAPK__logobs/dream_cytof/train/ features/EGFR_MAPK__logobs/dream_cytof/train/
```

This also copies older selections that predate the fix: CV-split sizes 5, 15, 25 and 30, and split `all`. Panel E reads only the scan sizes, so it ignores them. The replication check lists them as well, but it cannot reliably tell them apart:

- Every older proteomics selection has features outside the fixed HVG set.
- Many older transcript selections lie entirely inside it, for example `transcriptomics__all__HVGRFE_12`.

Judge these files by date.

```bash
taskpolicy -c utility ./venv/bin/python run_notebooks.py figure_S_receptor_evidence
```

The container replication (section 2d) is not in the notebook. It has to run inside the pipeline container on NEMO to be exact. The files are in `/nemo/lab/froehlichf/home/users/frohlif/s3_rfe_container/`:

- `rfe_container.py` holds `load_petab_tables`, `get_hvg`, `rf_inputs` and `rfe_path`, copied verbatim from `receptor_evidence.py` with `inspect.getsource`, plus a driver that runs one process per split and modality.
- `rfe_container.sbatch` runs it on 12 CPUs in `.snakemake/singularity/7f04b812f71f5e2337af6e13df44ead6.simg`, with `PYTHONPATH` set to the shared checkout. It takes about 6 min.

To rerun:

```bash
ssh login008.nemo.thecrick.org 'cd /nemo/lab/froehlichf/home/users/frohlif/s3_rfe_container && sbatch rfe_container.sbatch'
```

Outside the container, `rfe_path` runs but does not reproduce the pipeline's selections (section 2d):

```python
from receptor_evidence import get_hvg, load_petab_tables, rf_inputs, rfe_path

inputs, output = rf_inputs(load_petab_tables(), "all", "proteomics")
rfe_path(get_hvg(inputs), output, [4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 64, 128, 256])
```
