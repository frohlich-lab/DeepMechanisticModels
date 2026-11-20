# S2: Feature-count scan (Reviewer 1, Methods 2)

## Bottom line

This note is updated for three changes:

- the HVG-prefilter fix;
- the re-pinned feature-set sizes (option B, the user's choice on 4 Nov 2026);
- RNAseq at 6 latent dimensions instead of 7 (re-picked on 18 Nov 2026).

What the final scan (`figure1c`, finished 19 Nov 2026) shows:

- **Lowest validation error at small feature sets:**
  - 4 genes (validation RMSE 0.595)
  - 10 proteins (0.741)
  - 6 features per modality for multimodal (0.566)
  - 6 MOSA dimensions (0.583)
- **These are the selected sizes** in `BEST_FEATURE_SETS`. CyTOF stays at 8
  markers, because its curve is flat.
- **Size of the effect:** compared with the selected sizes, the largest
  feature sets (up to all features) are 9–27 % worse. Averaged over size
  ranges, small sets beat large ones by 4–10 %.
- **Draft claims:** "increased markedly … in all modalities" and "6–12 best"
  are replaced in section 5.
- **Table S4:** regenerated from `training_configuration.py` (section 6).

**Data and code:**

- Data: `eval/EGFR_MAPK__logobs/dream_cytof/evaluate_all_figure1c.csv` in
  the main checkout, synced from NEMO on 19 Nov 2026.
  - RNAseq was rerun at 6 latent dimensions.
  - CyTOF, MassSpec, multimodal and MOSA are identical to the 3 Nov file.
  - The notebook checks that every scan run uses the context's final
    hyperparameters.
- This branch carries the same changes as PR 21 (`claude/objective-rubin-5b7968`):
  - the `get_hvg` change of 50bd9e1 (same content);
  - 677d833, re-pinned feature-set sizes (cherry-picked);
  - 70ad8de, RNAseq latent dimensions 7 → 6 (cherry-picked).

## 1. Deliverables

- **`figures_paper/figure_S_feature_count_scan.ipynb`** makes Fig. S8 and
  Table S4. It is committed with its outputs.
  - Fig. S8A: validation RMSE against the number of input features, on a log
    axis, one line per modality.
  - Fig. S8B: paired per-fold difference to the selected size.
  - Stars mark the selected sizes (`BEST_FEATURE_SETS`). Open circles mark
    `all` features.
  - A dotted line marks the number of training cell lines per fold (57).
- **Outputs:**
  - `figure_S_feature_count_scan.pdf` and `.png` (Fig. S8).
  - `figure_S_feature_count_scan_summary.csv`: one row per modality × size.
  - `final_hyperparameters_per_context.csv` (Table S4).

## 2. What the scan is

- The config key is `figure1c` (`CONTEXTS_FEATURES_1C` in
  `training_configuration.py`). Despite the name, this is the feature-count
  scan behind Table S4 and Fig. S8, not paper Fig. 1C.
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
- Each modality is trained with its final ML hyperparameters (Table S4).
- Folds: 5 CV folds, each holding out one line (BT20, HCC1500, HCC2185, MCF7,
  UACC3199), × 10 random starts.
  - MOSA has only 3 usable folds, because HCC2185 and UACC3199 have no MOSA
    embedding.

### The number of encoder inputs is not always N

1. **HVGRFE never sees more than 501 features.** HVGRFE_512 … HVGRFE_4096
   are one model: the notebook checks that their RMSE is identical for every
   fold and start.
   - In practice, the omics RFE scan covers 4–501 features.
   - Larger sets exist only as the `all` runs: 16,562 genes and about 6,181
     proteins.
2. **Multimodal RFE_N means 3N inputs:** N each from CyTOF, MassSpec and
   RNAseq. The selected multimodal model has 18 inputs, and the scan covers
   12–96.
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
| RNAseq | 4 genes (0.595) | all 16,562 (0.756) | **+27 %**, worse in 5/5 | 0.58 (0.02) |
| MassSpec | 10 proteins (0.741) | all ~6,181 (0.805) | **+9 %**, worse in 5/5 | 0.38 (0.16) |
| Multimodal | 6/modality = 18 inputs (0.566) | 96 inputs (0.620) | **+10 %**, worse in 4/5 | 0.82 (0.004) |
| MOSA | 6 (0.583) | all 200 (0.641) | **+10 %**, worse in 3/3 | 0.51 (0.08) |
| CyTOF | 8 (0.552); lowest is 16 (0.536) | all 37 (0.546) | −1 %, worse in 1/5 | −0.34 (0.31) |

**Averaged over size bands.** This is robust to picking the single best of 15
noisy sizes on the same folds.

| Modality | N ≤ 12 | N = 16–32 | N ≥ 64 incl. `all` | Large vs small | Folds large worse |
|---|---|---|---|---|---|
| RNAseq | 0.696 | 0.684 | 0.765 | **+10 %** | 4/5 |
| MassSpec | 0.772 | 0.817 | 0.805 | **+4 %** | 4/5 |
| Multimodal (large = 24–32/modality, 72–96 inputs) | 0.584 | 0.625 (16–20) | 0.626 | **+7 %** | 4/5 |
| MOSA | 0.608 | 0.685 | 0.640 | **+5 %** | 3/3 |
| CyTOF (large = `all` 37) | 0.567 | 0.561 | 0.546 | −4 % | 2/5 |

### Verdict on the draft claims

- **"Small feature sets (6–12) yielded the lowest error"** no longer matches
  the selected sizes: 4 genes, 10 proteins, 6 per modality (multimodal),
  6 MOSA dimensions and 8 CyTOF markers. For CyTOF the number of markers
  hardly matters.
- **"Validation error increased markedly with the number of input features in
  all modalities"** is not supported as written.
  - Relative to the selected sizes, the largest sets are 9–27 % worse, in
    every fold for RNAseq and MassSpec.
  - Averaged over size ranges, the increase is 4–10 %.
  - The trend is monotone only for multimodal. CyTOF is flat.
  - Give numbers instead of "markedly", and treat CyTOF separately.
- **The selected sizes are, by construction, the minima of noisy curves**
  picked on the same five folds (fold SEM 0.02–0.12), so their RMSE is
  somewhat optimistic.
  - RNAseq 4 genes is the most pronounced minimum. It beats every other size
    in at least 3/5 folds, and in 4–5/5 for every size except 10 genes.
  - The next-best RNAseq sizes are 20–28 genes (0.670, +13 %). 6 and 8 genes
    are 15–20 % worse than 4 genes.
  - The band averages are the safer basis for the general claim.

## 5. Proposed wording (paste-ready, US spelling)

**Response to Reviewer 1** (replaces the bracketed sentence):

> We treated the number of input features as a hyperparameter and scanned it
> from 4 features up to the complete feature set of each modality (Table S4,
> Fig. S8). For transcriptomics, proteomics, MOSA and the multimodal encoder,
> the lowest cross-validation error was obtained with small feature sets
> (4 genes, 10 proteins, 6 MOSA dimensions, and 6 features per modality for
> the multimodal encoder), and the largest feature sets (up to all available
> features) increased validation error by 9–27% relative to these. For CyTOF
> (37 markers), the number of selected markers had little effect, and we used
> 8 markers.

**Methods** (replaces the draft sentence):

> Because the number of available features (up to 16,562 per modality)
> vastly exceeds the number of training cell lines (n = 58), we treated the
> number of selected features as a hyperparameter and scanned it from 4
> features up to the complete feature set of each modality (Table S4,
> Fig. S8). For transcriptomics, proteomics, MOSA and the multimodal encoder,
> validation error was lowest for small feature sets (4 genes, 10 proteins,
> 6 MOSA dimensions, and 6 features per modality for the multimodal encoder),
> and models trained on large feature sets overfitted: validation error was
> on average 4–10% higher for large than for small feature sets, and 9–27%
> higher for the largest sets than for the selected ones. For CyTOF
> (37 markers), validation error varied little with the number of selected
> markers, and we used 8 markers.

Optional clarifications:

- "…for proteomics and transcriptomics, recursive feature elimination was
  applied to the ~500 most variable features of the training cell lines, and
  larger feature sets were evaluated by using all features that passed the
  missing-value filter…"
- "…the multimodal encoder combines 6 selected features each from CyTOF,
  proteomics and transcriptomics (18 inputs)…"
- Avoid "scanned from 4 to 4,096": the omics RFE runs saturate at 501
  features.

**Legend for Fig. S8:**

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
  multimodal 8 per modality) performed on par with all features.
- So only the band-average claim would have held, and the figure would have
  shown the selected RNAseq size on a local peak.
- Re-pinning avoids that, at the cost of retraining the main-text models.

## 6. Table S4: final hyperparameters per input modality

Generated by the notebook from `training_configuration.py` (the pipeline's own
`generate_run_configs`), written to `final_hyperparameters_per_context.csv`.

| | CyTOF | RNAseq | MassSpec | Multimodal | MOSA |
|---|---|---|---|---|---|
| Feature set | RFE_8 | HVGRFE_4 | HVGRFE_10 | RFE_6 | RFE_6 |
| Features per modality | 8 | 4 | 10 | 6 | 6 |
| Encoder inputs | 8 | 4 | 10 | 18 | 6 |
| Candidate features entering selection | 37 | 501 | 501 | 1,039 | 200 |
| Latent dimensions | 8 | 6 | 5 | 6 | 6 |
| Hidden layers | 1 | 1 | 1 | 0 | 0 |
| Dropout rate | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| L1 penalty on parameter deviations (until epoch 200) | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| Validation RMSE at selected size (mean ± SEM) | 0.552 ± 0.051 | 0.595 ± 0.041 | 0.741 ± 0.064 | 0.566 ± 0.068 | 0.583 ± 0.033 (3 folds) |

**Shared by all modalities:**

- L2 penalty on parameter deviations: 0.
- Parameter deviation bound: 3. Deviations pass through 3·tanh(x/3).
- Weight initialization scale: 0.1.
- Reconstruction loss weight: 0.01.
- 500 epochs.
- After epoch 200, the L1 penalty is lifted but the sparsity pattern is kept.

**How each value was chosen:**

- Latent dimensions, hidden layers, dropout and L1 come from the figure1b
  scan. RNAseq's latent dimensions were re-picked from 7 to 6 on 18 Nov.
  - 6 beat 7 in all five CV folds.
  - The figure1b rerun of 19 Nov confirmed depth 1, dropout 0.1 and L1 0.1;
    L1 between 0 and 0.01 ties with 0.1.
- The feature sets come from this scan.

### Consistency to check before submission

- All main-text RNAseq, MassSpec and multimodal models must use the fixed
  prefilter and the new sizes, and RNAseq must use 6 latent dimensions
  (figure1a, figure1b, figure3 and figure3b configs).
- Any text, table or figure that mentions "8 proteins", "12 transcripts",
  "8 features per modality" for multimodal (24 inputs), or 7 latent
  dimensions for RNAseq, needs updating.

## 7. For S6 (workflow) and data

- `run_notebooks.py` needs no change. It discovers tracked notebooks under
  `figures_paper/` via `git ls-files`, and its `savefig`/`to_csv` redirects
  catch this notebook's bare filenames.
- The notebook needs two inputs:
  - The 19 Nov 2026 `eval/EGFR_MAPK__logobs/dream_cytof/evaluate_all_figure1c.csv`,
    now in the main checkout's `eval/`. An older file fails the notebook's
    hyperparameter check for RNAseq.
  - The PEtab tables in `data/`.
- It also needs the Cellosaurus file in `features/`, downloaded on first use,
  for the MOSA cell-line mapping.
- Runtime is a few minutes.
