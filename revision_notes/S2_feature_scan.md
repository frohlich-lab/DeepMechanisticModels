# S2: Feature-count scan (Reviewer 1, Methods 2)

## Bottom line

This note is updated for the HVG-prefilter fix and for re-pinning the
feature-set sizes (option B, the user's choice on 4 Nov 2026).

- **Corrected scan:** the figure1c scan with the corrected HVG prefilter
  (commit 50bd9e1; rerun of 3 Nov 2026) puts the lowest validation error at
  small feature sets:
  - 4 genes
  - 10 proteins
  - 6 features per modality for multimodal
  - 6 MOSA dimensions
- **Re-pinned sizes:** `BEST_FEATURE_SETS` now uses these sizes (commit
  677d833, cherry-picked here). CyTOF stays at 8 markers, because its curve
  is flat.
- **Size of the effect:** compared with these sizes, the largest feature sets
  (up to all features) are 9–25 % worse. Averaged over size ranges, small
  sets beat large ones by 4–8 %.
- **Draft claims:** "increased markedly … in all modalities" and "6–12 best"
  are replaced in section 5.

**Data and code:**

- Data: `eval/EGFR_MAPK__logobs/dream_cytof/evaluate_all_figure1c.csv` from
  the NEMO rerun of 3 Nov 2026, 23:35.
- The April 2026 file predates the fix. It is backed up on NEMO under
  `pre_hvg_fix_2026-11-02/`.
- This branch carries both commits from `claude/objective-rubin-5b7968`:
  - The `get_hvg` change of 50bd9e1, with identical content.
  - 677d833 itself (re-pinned sizes), cherry-picked.

## 1. Deliverables

- **`figures_paper/figure_S_feature_count_scan.ipynb`** is committed with its
  outputs. It plots validation RMSE against the number of input features on a
  log axis, one line per modality.
  - Panel A shows absolute validation RMSE.
  - Panel B shows the paired per-fold difference to the selected size.
  - Stars mark the sizes in `BEST_FEATURE_SETS`, now the re-pinned ones.
  - Open circles mark `all` features.
  - A dotted line marks the number of training cell lines per fold (57).
- **Outputs:**
  - `figure_S_feature_count_scan.pdf` and `.png`.
  - `figure_S_feature_count_scan_summary.csv`: one row per modality × size.
    **Regenerate Table S4 from it**, since its omics and multimodal rows
    predate the fix.

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
    the training lines, then runs RFE.
  - Selection is fitted on the training lines of each fold only.
- Training setup: base model `EGFR_MAPK__logobs` with the central ML
  hyperparameters, which are identical within each modality.
- Folds: 5 CV folds, each holding out one line (BT20, HCC1500, HCC2185, MCF7,
  UACC3199), × 10 random starts.
  - MOSA has only 3 usable folds, because HCC2185 and UACC3199 have no MOSA
    embedding.
- Selected sizes (`BEST_FEATURE_SETS`):

  | Modality | Spec | Size |
  |---|---|---|
  | RNAseq | HVGRFE_4 | 4 genes (was 12) |
  | MassSpec | HVGRFE_10 | 10 proteins (was 8) |
  | Multimodal | RFE_6 | 6 per modality = 18 inputs (was 8 per modality = 24) |
  | CyTOF | RFE_8 | 8 markers (unchanged) |
  | MOSA | RFE_6 | 6 dimensions (unchanged) |

### The number of encoder inputs is not always N

1. **HVGRFE never sees more than 501 features.** HVGRFE_512 … HVGRFE_4096
   are one model: the notebook checks that their RMSE is identical for every
   fold and start.
   - In practice, the omics RFE scan covers 4–501 features.
   - Larger sets exist only as the `all` runs: 16,562 genes and about 6,181
     proteins.
2. **Multimodal RFE_N means 3N inputs:** N each from CyTOF, MassSpec and
   RNAseq. The selected multimodal model therefore has 18 inputs, and the scan
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
| RNAseq | 16,562 genes, complete in all 64 lines | 57 | **16,562** | **501** (most variable) | 4 |
| MassSpec | 7,877 proteins (3,751 measured in all 67 lines) | 57 | **6,177–6,187** (median 6,181) | **501** (most variable) | 10 |
| CyTOF | 37 markers. `data/cytof.csv` has 38 feature IDs, but one is `glob_cellID`, a cell index | 57 (58 in the final model) | **37** | 37 | 8 |
| MOSA | 200 latent dimensions | 45–46 (lines with a MOSA embedding) | **200** | 200 | 6 |
| Multimodal | 37 + 6,181 + 16,562 = 22,780 | 57 | — | 37 + 501 + 501 = **1,039** | 6 per modality = 18 |

**Which number to quote:**

- "Up to 16,562 per modality" is correct both raw and after filtering.
- For MassSpec, quote ~6,200 after filtering, or 7,877 raw. Do not quote 3,751.
- n = 58 is right for the final models. Each CV fold trained on 57 lines.

**HVG prefilter, after the fix:**

- It keeps the 501 most variable features of the training lines.
- The old low-mean step acted on mean-centred data, so it excluded features
  effectively at random. It is gone.
- Suggested Methods phrase: "prefiltered to the ~500 most variable features of
  the training cell lines".

## 4. What the data show

RMSE values are the mean over folds of the median over 10 starts. Δ values are
paired per fold. "Worse" counts the folds in which the larger set had higher
RMSE.

| Modality | Selected = lowest RMSE | All/largest (RMSE) | Largest vs selected | Trend ρ (p) |
|---|---|---|---|---|
| RNAseq | 4 genes (0.609) | all 16,562 (0.761) | **+25 %**, worse in 5/5 | 0.58 (0.02) |
| MassSpec | 10 proteins (0.741) | all ~6,181 (0.805) | **+9 %**, worse in 5/5 | 0.38 (0.16) |
| Multimodal | 6/modality = 18 inputs (0.566) | 96 inputs (0.620) | **+10 %**, worse in 4/5 | 0.82 (0.004) |
| MOSA | 6 (0.583) | all 200 (0.641) | **+10 %**, worse in 3/3 | 0.51 (0.08) |
| CyTOF | 8 (0.552); lowest is 16 (0.536) | all 37 (0.546) | −1 %, worse in 1/5 | −0.34 (0.31) |

**Averaged over size bands.** This is robust to picking the single best of 15
noisy sizes on the same folds.

| Modality | N ≤ 12 | N = 16–32 | N ≥ 64 incl. `all` | Large vs small | Folds large worse |
|---|---|---|---|---|---|
| RNAseq | 0.702 | 0.689 | 0.760 | **+8 %** | 4/5 |
| MassSpec | 0.772 | 0.817 | 0.805 | **+4 %** | 4/5 |
| Multimodal (large = 24–32/modality, 72–96 inputs) | 0.584 | 0.625 (16–20) | 0.626 | **+7 %** | 4/5 |
| MOSA | 0.608 | 0.685 | 0.640 | **+5 %** | 3/3 |
| CyTOF (large = `all` 37) | 0.567 | 0.561 | 0.546 | −4 % | 2/5 |

### Verdict on the draft claims

- **"Small feature sets (6–12) yielded the lowest error"** becomes 4–10
  features per modality. CyTOF is the exception: there the number of markers
  hardly matters.
- **"Validation error increased markedly with the number of input features in
  all modalities"** is not supported as written.
  - Relative to the selected sizes, the largest sets are 9–25 % worse, in
    every fold for RNAseq and MassSpec.
  - Averaged over size ranges, the increase is 4–8 %.
  - The trend is monotone only for multimodal. CyTOF is flat.
  - Give numbers instead of "markedly", and treat CyTOF separately.
- **The selected sizes are, by construction, the minima of noisy curves**
  picked on the same five folds (fold SEM 0.02–0.12), so their RMSE is
  somewhat optimistic.
  - RNAseq 4 genes is the extreme point: it beat 12 genes in 4/5 folds, but
    6 and 8 genes are 10–16 % worse.
  - The band averages are the safer basis for the general claim.

## 5. Proposed wording (paste-ready, US spelling)

**Response to Reviewer 1** (replaces the bracketed sentence):

> We treated the number of input features as a hyperparameter and scanned it
> from 4 features up to the complete feature set of each modality (Table S4,
> Fig. S_). For transcriptomics, proteomics, MOSA and the multimodal encoder,
> the lowest cross-validation error was obtained with 4–10 features per
> modality, and the largest feature sets (up to all available features)
> increased validation error by 9–25% relative to these small feature sets.
> For CyTOF (37 markers), the number of selected markers had little effect.

**Methods** (replaces the draft sentence):

> Because the number of available features (up to 16,562 per modality)
> vastly exceeds the number of training cell lines (n = 58), we treated the
> number of selected features as a hyperparameter and scanned it from 4
> features up to the complete feature set of each modality (Table S4,
> Fig. S_). For transcriptomics, proteomics, MOSA and the multimodal encoder,
> validation error was lowest for small feature sets (4 genes, 10 proteins,
> 6 MOSA dimensions, and 6 features per modality for the multimodal encoder),
> and models trained on large feature sets overfitted: validation error was
> on average 4–8% higher for large than for small feature sets, and 9–25%
> higher for the largest sets than for the selected ones. For CyTOF (37 markers), validation error
> varied little with the number of selected markers, and we used 8 markers.

Optional clarifications:

- "…for proteomics and transcriptomics, recursive feature elimination was
  applied to the ~500 most variable features of the training cell lines, and
  larger feature sets were evaluated by using all features that passed the
  missing-value filter…"
- "…the multimodal encoder combines 6 selected features each from CyTOF,
  proteomics and transcriptomics (18 inputs)…"
- Avoid "scanned from 4 to 4,096": the omics RFE runs saturate at 501
  features.

**Legend for Fig. S_:**

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

**What changed vs option A (keep 8/12/8, not chosen):**

- Under the fixed prefilter, the old sizes (RNAseq 12, MassSpec 8,
  multimodal 8 per modality) performed on par with all features (−2 %, +2 %,
  +5 %).
- So only the band-average claim would have held, and the figure would have
  shown the selected RNAseq size on a local peak.
- Re-pinning avoids that, at the cost of retraining the main-text models.

### Consistency to check before submission

- All main-text RNAseq, MassSpec and multimodal models must be retrained with
  the fixed prefilter and the new sizes, under the figure1a, figure1b, figure3
  and figure3b configs. Until then, those panels use feature sets that differ
  from this figure's stars.
  - The figure1b ML scan is rerunning on NEMO.
- Any text, table or figure that mentions "8 proteins", "12 transcripts" or
  "8 features per modality" for multimodal (24 inputs) needs updating.
- Table S4 needs regenerating from the summary CSV.

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
