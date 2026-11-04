# S2: Feature-count scan (Reviewer 1, Methods 2)

## Bottom line

This note is updated for the HVG-prefilter fix: the figure1c rerun of
3 Nov 2026 (commit 50bd9e1).

**What still holds:**

- For RNAseq, MassSpec, MOSA and multimodal, small feature sets (4–12
  features per modality) generalize better than large ones (64 features up to
  all features).
- The size of that effect is modest: on average 4–8 % lower validation
  error, consistent in 4 of 5 folds (3 of 3 for MOSA).
- CyTOF is flat.

**What no longer holds:**

- "Increased markedly … in all modalities".
- "6–12 best".
- The selected RNAseq (12) and MassSpec (8) sizes are no longer near the
  scan's optimum. They now perform on par with using all features.

**Decision for the user:** whether to keep the selected sizes or re-pick them.
Re-picking means retraining the main-text models. Section 5 has wording for
both choices.

**Data and code:**

- Data: `eval/EGFR_MAPK__logobs/dream_cytof/evaluate_all_figure1c.csv` from
  the NEMO rerun, dated 3 Nov 2026 23:35.
- The April 2026 file predates the fix. It is backed up on NEMO under
  `pre_hvg_fix_2026-11-02/`.
- `select_features.py` on this branch carries the same `get_hvg` change as
  50bd9e1 (`claude/objective-rubin-5b7968`).

## 1. Deliverables

- **`figures_paper/figure_S_feature_count_scan.ipynb`** is committed with its
  outputs. It plots validation RMSE against the number of input features on a
  log axis, one line per modality.
  - Panel A shows absolute validation RMSE.
  - Panel B shows the paired per-fold difference to the selected size.
  - Stars mark the sizes in `BEST_FEATURE_SETS`, so they move automatically
    if those sizes are re-picked.
  - Open circles mark `all` features.
  - A dotted line marks the number of training cell lines per fold (57).
- The notebook computes the candidate counts in section 3 itself, using the
  pipeline's own `load_data` and `get_hvg`.
- **Outputs:**
  - `figure_S_feature_count_scan.pdf` and `.png`.
  - `figure_S_feature_count_scan_summary.csv`: one row per modality × size,
    for cross-checking Table S4.
  - **Table S4 must be regenerated:** its omics and multimodal rows predate
    the fix.

## 2. What the scan is

- The config key is `figure1c` (`CONTEXTS_FEATURES_1C` in
  `training_configuration.py`). Despite the name, this is the feature-count
  scan behind Table S4, not paper Fig. 1C.
- Sizes scanned: N ∈ {4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 64, 128, 256, 512,
  1024, 2048, 4096}, plus `all`.
  - CyTOF and multimodal stop at N = 32 (37 CyTOF markers).
  - MOSA stops at N = 128, plus `all` = 200 latent dimensions.
- Selection method:
  - RFE uses a random forest. It drops 20 % of features per step, ranked by
    permutation importance, or by impurity importance when more than 500
    features remain.
  - HVGRFE (MassSpec, RNAseq) first keeps the 501 most variable features of
    the training lines, with no expression filter (after the fix), then runs
    RFE.
  - Selection is fitted on the training lines of each fold only.
- Training setup: base model `EGFR_MAPK__logobs` with the central ML
  hyperparameters, which are identical within each modality.
- Folds: 5 CV folds, each holding out one line (BT20, HCC1500, HCC2185, MCF7,
  UACC3199), × 10 random starts.
  - MOSA has only 3 usable folds, because HCC2185 and UACC3199 have no MOSA
    embedding.
- Selected sizes (`BEST_FEATURE_SETS`, left unchanged): CyTOF RFE_8,
  MassSpec HVGRFE_8, RNAseq HVGRFE_12, MOSA RFE_6, multimodal RFE_8.

### The number of encoder inputs is not always N

1. **HVGRFE never sees more than 501 features.** For N ≥ 401 the RFE loop
   does not run, so HVGRFE_512 … HVGRFE_4096 are one model: the notebook
   checks that their RMSE is identical for every fold and start.
   - In practice, the omics RFE scan covers 4–501 features.
   - Larger sets exist only as the `all` runs: 16,562 genes and about 6,181
     proteins.
2. **Multimodal RFE_N means 3N inputs:** N each from CyTOF, MassSpec and
   RNAseq. The selected multimodal model therefore has 24 inputs, and the scan
   covers 12–96.
3. **`all`** means every feature that passes the missing-value filter, with no
   RFE.

## 3. Candidate features per modality

The steps follow `select_features.prepare_inputs_for_context`, on the training
lines of each fold:

1. Pivot the PEtab measurement table.
2. Drop features missing in ≥ 30 % of lines (`load_data`).
3. Apply KNN imputation.
4. Mean-centre.
5. Apply the HVG prefilter (`get_hvg`, HVGRFE only).

| Modality | Raw | Training lines per fold | After missing-value filter = `all` | Enter RFE | Selected |
|---|---|---|---|---|---|
| RNAseq | 16,562 genes, complete in all 64 lines | 57 | **16,562** | **501** (most variable) | 12 |
| MassSpec | 7,877 proteins (3,751 measured in all 67 lines) | 57 | **6,177–6,187** (median 6,181) | **501** (most variable) | 8 |
| CyTOF | 37 markers. `data/cytof.csv` has 38 feature IDs, but one is `glob_cellID`, a cell index | 57 (58 in the final model) | **37** | 37 | 8 |
| MOSA | 200 latent dimensions | 45–46 (lines with a MOSA embedding) | **200** | 200 | 6 |
| Multimodal | 37 + 6,181 + 16,562 = 22,780 | 57 | — | 37 + 501 + 501 = **1,039** | 8 per modality = 24 |

**Which number to quote:**

- "Up to 16,562 per modality" is correct both raw and after filtering.
- For MassSpec, quote ~6,200 after filtering, or 7,877 raw. Do not quote 3,751.
- n = 58 is right for the final models. Each CV fold trained on 57 lines.

**HVG prefilter, after the fix:**

- It now keeps the 501 most variable features of the training lines.
- The old low-mean step acted on mean-centred data, so it excluded features
  effectively at random (185–212 of the 501 most variable proteins per fold).
  It is gone.
- Suggested Methods phrase: "prefiltered to the ~500 most variable features of
  the training cell lines".
- The candidate counts above do not change, because the prefilter still keeps
  501 features. Only which features they are changes.

## 4. Do the data support the draft claims?

RMSE values are the mean over folds of the median over 10 starts. Δ values are
paired per fold. "Worse" counts the folds in which the larger set had higher
RMSE.

### Best, selected and largest size

| Modality | Lowest RMSE at | Selected (RMSE) | All/largest (RMSE) | Largest vs best | Largest vs selected | Trend ρ (p) |
|---|---|---|---|---|---|---|
| RNAseq | **4** (0.609) | 12 (0.774) | all 16,562 (0.761) | +25 %, worse in 5/5 | **−2 %**, worse in 2/5 | 0.58 (0.02) |
| MassSpec | **10** (0.741) | 8 (0.788) | all ~6,181 (0.805) | +9 %, worse in 5/5 | **+2 %**, worse in 3/5 | 0.38 (0.16) |
| Multimodal | **6/modality = 18** (0.566) | 24 (0.588) | 96 (0.620) | +10 %, worse in 4/5 | +5 %, worse in 4/5 | 0.82 (0.004) |
| MOSA | **6** (0.583) | 6 (0.583) | all 200 (0.641) | +10 %, worse in 3/3 | +10 %, worse in 3/3 | 0.51 (0.08) |
| CyTOF | **16** (0.536) | 8 (0.552) | all 37 (0.546) | +2 %, worse in 1/5 | −1 %, worse in 1/5 | −0.34 (0.31) |

The MOSA and CyTOF results are identical to the April run; MassSpec, RNAseq
and multimodal changed with the fix.

### Averaged over size bands

Picking the single best of 15 noisy sizes on the same folds is optimistic.
Averaging over bands of sizes is a more robust summary.

| Modality | N ≤ 12 | N = 16–32 | N ≥ 64 incl. `all` | Large vs small | Folds large worse |
|---|---|---|---|---|---|
| RNAseq | 0.702 | 0.689 | 0.760 | **+8 %** | 4/5 |
| MassSpec | 0.772 | 0.817 | 0.805 | **+4 %** | 4/5 |
| Multimodal (large = 24–32/modality, 72–96 inputs) | 0.584 | 0.625 (16–20) | 0.626 | **+7 %** | 4/5 |
| MOSA | 0.608 | 0.685 | 0.640 | **+5 %** | 3/3 |
| CyTOF (large = `all` 37) | 0.567 | 0.561 | 0.546 | −4 % | 2/5 |

For comparison, the April run (before the fix) gave MassSpec +15 %,
RNAseq +8 % and multimodal +8.5 %.

### Verdict

- **"Small feature sets (6–12) yielded the lowest error."**
  - The lowest-error sizes are now 4–10 per modality: RNAseq 4, MassSpec 10,
    multimodal 6, MOSA 6. CyTOF is flat.
  - "Small sets are better than large ones" still holds as a band average.
  - But the selected RNAseq 12 is at a local peak of its curve, and MassSpec 8
    is about average. **Neither is the low point any more.**
- **"Validation error increased markedly with the number of input features in
  all modalities"** is not supported.
  - On average, large sets are 4–8 % worse than small sets for RNAseq,
    MassSpec, MOSA and multimodal, consistently across folds.
  - They are 9–25 % worse than the best size.
  - The trend is clear only for multimodal (ρ = 0.82). It is weak for RNAseq
    and MOSA, absent for MassSpec, and reversed (but flat) for CyTOF.
  - Write "modestly" or give the numbers; avoid "markedly".
- **Why the selected sizes moved:** at those sizes the change is driven by the
  new feature selections for some folds, not by the random starts.
  - HCC1500: MassSpec HVGRFE_8 0.53 → 1.19, RNAseq HVGRFE_12 0.60 → 0.87.
  - MCF7: RNAseq HVGRFE_12 0.68 → 0.88.
- **Consistency with the main text:** the figure1a/1b/3/3b feature sets were
  not regenerated with the fix as part of this run. Until they are, the
  RNAseq, MassSpec and multimodal main-text models use different feature sets
  than the stars in this figure.

## 5. Proposed wording (paste-ready, US spelling)

### Methods sentence, valid for both options

> Because the number of available features (up to 16,562 per modality)
> vastly exceeds the number of training cell lines (n = 58), we treated the
> number of selected features as a hyperparameter and scanned it from 4
> features up to the complete feature set of each modality (Table S4,
> Fig. S_). For transcriptomics, proteomics, MOSA and the multimodal encoder,
> small feature sets (4–12 features per modality) had on average 4–8% lower
> validation error than large feature sets (64 features up to all available
> features), whereas for CyTOF (37 markers) validation error varied little
> with the number of selected markers.

### Option A: keep the selected sizes (8 proteins, 12 transcripts, 8 per modality); no retraining

Response to Reviewer 1:

> We treated the number of input features as a hyperparameter and scanned it
> from 4 features up to the complete feature set of each modality (Table S4,
> Fig. S_). Small feature sets generalized better: for transcriptomics,
> proteomics, MOSA and the multimodal encoder, models with 4–12 features per
> modality had on average 4–8% lower cross-validation error than models with
> 64 or more features. For transcriptomics, proteomics and MOSA, the best
> small feature set outperformed the complete feature set in every
> cross-validation fold. Differences among small
> feature sets were within the variability across folds, and for CyTOF
> (37 markers) the number of selected markers had little effect.

Caveats for option A:

- Fig. S_ shows openly that RNAseq 12 and MassSpec 8 perform on par with all
  features, and that RNAseq 4 and MassSpec 10 did better. A reviewer may ask
  why those sizes were used. A sentence such as "the sizes used for the
  main-text models were fixed before the final scan and lie within the range
  of small feature sets" would be honest.
- Only the band-average claim holds. Do not write "the selected sizes
  minimized validation error".

### Option B: re-pick the sizes to the scan's best (RNAseq 4, MassSpec 10, multimodal 6/modality; MOSA 6 and CyTOF 8 unchanged); requires retraining the main-text models

Response to Reviewer 1:

> We treated the number of input features as a hyperparameter and scanned it
> from 4 features up to the complete feature set of each modality (Table S4,
> Fig. S_). For transcriptomics, proteomics, MOSA and the multimodal encoder,
> the lowest cross-validation error was obtained with 4–10 features per
> modality, and the largest feature sets (up to all available features)
> increased validation error by 9–25% relative to these small feature sets. For CyTOF (37 markers), the
> number of selected markers had little effect.

Caveats for option B:

- The best sizes are minima of noisy curves picked on the same five folds
  (fold SEM 0.02–0.12).
- RNAseq 4 genes (0.609) is the extreme low point. It beat 12 genes in 4/5
  folds, but its neighbors (6 and 8 genes) are 10–16 % worse.
- Before committing to option B, it may be worth checking how stable the
  selected genes are across folds.

### Optional clarifications (both options)

- "…for proteomics and transcriptomics, recursive feature elimination was
  applied to the ~500 most variable features of the training cell lines, and
  larger feature sets were evaluated by using all features that passed the
  missing-value filter…"
- "…the multimodal encoder combines the selected features of CyTOF,
  proteomics and transcriptomics (3 × N inputs)…"
- Avoid "scanned from 4 to 4,096": the omics RFE runs saturate at 501
  features.

### Legend for Fig. S_

> Validation error as a function of the number of input features. DMMs (base
> EGFR/MAPK model) were trained on feature sets of increasing size, selected
> by recursive feature elimination on the training cell lines of each
> cross-validation fold. For proteomics and transcriptomics, selection was
> applied to the 501 most variable features. Open circles show all features
> passing the missing-value filter. Multimodal models combine the same number
> of selected features from CyTOF, proteomics and transcriptomics; the x-axis
> shows the total. Stars mark the feature-set sizes used in the main text. The
> dotted line marks the number of training cell lines per fold (57).
> (A) Validation RMSE on the held-out cell line, shown as the median over 10
> random starts, then the mean ± SEM over 5 cross-validation folds (3 for
> MOSA). (B) Per-fold difference in validation RMSE to the selected
> feature-set size (mean ± SEM over folds).

## 6. For S6 (workflow) and data

- `run_notebooks.py` needs no change. It discovers tracked notebooks under
  `figures_paper/` via `git ls-files`, and its `savefig`/`to_csv` redirects
  catch this notebook's bare filenames.
- The notebook needs two inputs:
  - The **3 Nov 2026** `eval/EGFR_MAPK__logobs/dream_cytof/evaluate_all_figure1c.csv`.
    The main checkout's `eval/` still has the April file; a local copy of the
    new one is in the shared side-session notes folder.
  - The PEtab tables in `data/`.
- It also needs the Cellosaurus file in `features/`, downloaded on first use,
  for the MOSA cell-line mapping.
- Runtime is a few minutes.
