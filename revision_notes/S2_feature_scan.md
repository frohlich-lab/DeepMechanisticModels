# S2: Feature-count scan (Reviewer 1, Methods 2)

> **Status: PRELIMINARY.** The candidate-feature counts (section 3) are final.
> The scan results (sections 4–5) come from the cached outputs of
> `figures_paper/model analysis (not in paper)/feature_scan.ipynb`, an earlier
> read of the same evaluation file. They do not include the `all` runs for
> CyTOF, MassSpec or RNAseq. They will be replaced once
> `eval/EGFR_MAPK__logobs/dream_cytof/evaluate_all_figure1c.csv` is copied
> from NEMO and the new notebook has run on it.

## 1. Deliverables

- `figures_paper/figure_S_feature_count_scan.ipynb` is the new supplementary
  figure: validation RMSE vs number of input features, with one line per
  modality.
  - Panel A shows absolute validation RMSE.
  - Panel B shows the paired per-fold difference to the selected size.
  - Stars mark the selected sizes. Open circles mark `all` features.
  - A dotted line marks the number of training cell lines per fold (57).
  - Outputs: `figure_S_feature_count_scan.pdf/.png` and
    `figure_S_feature_count_scan_summary.csv`, the per-size table for
    cross-checking Table S4.
- The notebook computes the candidate counts in section 3 itself, from
  `data/` and the pipeline's own `load_data` and `get_hvg`.

## 2. What the scan is

- The config key is `figure1c` (`CONTEXTS_FEATURES_1C` in
  `training_configuration.py`). Despite the name, this is the
  feature-count scan behind Table S4, not paper Fig. 1C.
- Sizes scanned: N ∈ {4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 64, 128, 256, 512,
  1024, 2048, 4096}, plus `all`.
  - CyTOF and multimodal stop at N = 32 because there are 37 CyTOF markers.
  - MOSA stops at N = 128, plus `all` = 200 latent dimensions.
- Selection method:
  - RFE uses a random forest. It drops 20 % of features per step, ranked by
    permutation importance, or by impurity importance when more than 500
    features remain.
  - HVGRFE (MassSpec, RNAseq) first applies a highly-variable-feature
    prefilter, then runs RFE.
  - Selection is fitted on the training lines of each fold only.
- Training setup: base model `EGFR_MAPK__logobs` with the central ML
  hyperparameters.
- Folds: 5 CV folds, each holding out one line (BT20, HCC1500, HCC2185, MCF7,
  UACC3199), with 5 random starts each.
  - MOSA has only 3 usable folds, because HCC2185 and UACC3199 have no MOSA
    embedding.
- Selected sizes (`BEST_FEATURE_SETS`): CyTOF RFE_8, MassSpec HVGRFE_8,
  RNAseq HVGRFE_12, MOSA RFE_6, multimodal RFE_8.

### The number of encoder inputs is not always N

These three points matter for the wording of the response:

1. **HVGRFE never sees more than 501 features.** The HVG prefilter keeps the
   501 most variable features. For every N ≥ 401 the RFE loop does not run,
   and the final step keeps min(N, 501). So HVGRFE_512 … HVGRFE_4096 are the
   same 501-feature model. The cached results confirm this: RNAseq and
   MassSpec have identical RMSE for N = 512 to 4096.
   - In practice, the RFE scan for the omics modalities covers 4–501
     features.
   - Larger sets exist only as the `all` runs: 16,562 genes and about 6,181
     proteins.
   - "Scanned from 4 to 4,096" describes the configuration, not the inputs.
2. **Multimodal RFE_N means 3N inputs:** N CyTOF markers, N proteins
   (HVGRFE) and N genes (HVGRFE). I checked this against cached feature files:
   multimodal RFE_10 has 30 columns. The selected multimodal model (RFE_8)
   therefore has **24 inputs**, 8 per modality. The multimodal scan covers
   12–96 inputs.
3. **`all`** means every feature that passes the missing-value filter, with no
   RFE.

## 3. Candidate features per modality (final)

The steps follow `select_features.prepare_inputs_for_context`, on the training
lines of each CV fold:

1. Pivot the PEtab measurement table.
2. Drop features missing in ≥ 30 % of lines (`load_data`).
3. Apply KNN imputation.
4. Mean-centre.
5. Apply the HVG prefilter (`get_hvg`, HVGRFE only).

Values are identical across the 5 folds unless a range is given.

| Modality | Raw | Lines per fold (training) | After missing-value filter = `all` | Enter RFE (HVGRFE) | Selected |
|---|---|---|---|---|---|
| RNAseq | 16,562 genes (complete in all 64 lines in the table) | 57 | **16,562** | **501** | 12 |
| MassSpec | 7,877 proteins (3,751 measured in all 67 lines). 7,872–7,874 are seen in a fold's 57 training lines, of which ~3,810 are complete | 57 | **6,177–6,187** (median 6,181) | **501** | 8 |
| CyTOF | 37 markers. `data/cytof.csv` has 38 feature IDs, but one is `glob_cellID`, a cell index | 57 (58 in the final `all` split) | **37** | 37 (plain RFE) | 8 |
| MOSA | 200 latent dimensions | 45–46 (lines with a MOSA embedding) | **200** | 200 (plain RFE) | 6 |
| Multimodal | 37 + 6,181 + 16,562 = 22,780 | 57 | — | 37 + 501 + 501 = **1,039** | 8 per modality = 24 |

**Which number to quote for "available features":**

- **16,562** (RNAseq) is both the raw count and the post-filter count. The
  RNAseq data have no missing values, so the claim "up to 16,562 per
  modality" holds either way.
- For MassSpec, quote **~6,200 after filtering** or 7,877 raw. Do not quote
  3,751: that is the complete-case count, which the pipeline does not use.
- The number that actually enters supervised selection is **501** for the
  omics modalities.

### Caveat on the HVG prefilter (please check the Methods wording)

`get_hvg` is meant to "remove 20 % of features with the lowest mean, then keep
the top 500 by variance". However, `prepare_inputs_for_context` mean-centres
the data *before* calling it. As a result:

- The mean filter acts on floating-point residuals (|mean| < 3e-15) and
  removes 20 % of features effectively at random.
- For MassSpec the removal is biased toward high-variance features. In the
  BT20 fold it excluded 212 of the 501 most variable proteins. For RNAseq it
  excluded 34.
- So the HVG set is "the ~500 most variable features among a pseudo-random
  80 %". It is also 501 features, not 500, because of an off-by-one in the
  threshold.

Fixing this would mean rerunning feature selection and all downstream
training, so I am not proposing that. But if the Methods describe a
mean-expression filter, the text does not match what ran. Safer wording would
be: "prefiltered to the ~500 most variable features".

## 4. Do the data support the draft claims? (preliminary)

The table shows the mean over folds of the median over 5 starts. The
`largest` column is the largest set in the cached output, excluding the
`all` runs.

| Modality | Lowest RMSE at | Selected | RMSE best → selected | Largest tested | RMSE there | Δ vs best |
|---|---|---|---|---|---|---|
| CyTOF | **16** (0.536) | 8 (0.552) | +3 % | 32 | 0.562 | +5 %; whole range 4–32 is 0.536–0.582 and N = 4 is worst |
| MassSpec | 8 (0.669) | 8 | — | 501 | 0.807 | +21 % (peak +26 % at 128) |
| RNAseq | 12 (0.657) | 12 | — | 501 | 0.784 | +19 % (peak +21 % at 256) |
| Multimodal | 8/modality = 24 inputs (0.533) | same | — | 32/modality = 96 inputs | 0.591 | +11 % (peak +16 % at 72 inputs) |
| MOSA | 6 (0.583) | 6 | — | 200 (`all`) | 0.641 | +10 % (peak +19 % at 16); 3 folds |

- **"Small feature sets (6–12) yielded the lowest error"** holds for
  MassSpec, RNAseq and MOSA, and for multimodal counted per modality.
  - It does not hold for CyTOF, where 16 markers was marginally best. 8 was
    second, within 0.015 RMSE.
  - It also does not hold for multimodal counted as total inputs (24).
- **"Validation error increased markedly with the number of input features in
  all modalities"** holds for the omics-derived modalities: MassSpec, RNAseq,
  multimodal and MOSA, with increases of 10–26 % over the best size.
  - It does **not** hold for CyTOF. With only 37 markers there is little room
    to grow. RMSE varies by less than 10 % across 4–32 markers, with no
    monotone trend, and the smallest set is the worst.
  - The CyTOF `all` (37) point is not in the cached output; the final figure
    will show it.
  - The MOSA curve is not monotone: 200 dimensions beat 16–32.

## 5. Proposed wording (to be confirmed with the final data)

**Response to Reviewer 1** (replaces the bracketed sentence):

> We treated the number of input features as a hyperparameter and scanned it
> from 4 features up to the full feature set of each modality (Table S4,
> Fig. S_). For all modalities, small feature sets (6–16 features per
> modality) yielded the lowest cross-validation error. For the omics
> modalities, validation error increased by 10–26 % when larger feature sets
> were used.

**Methods** (replaces the draft sentence):

> Because the number of available features (up to 16,562 per modality)
> vastly exceeds the number of training cell lines (n = 58), we treated the
> number of selected features as a hyperparameter. For transcriptomics,
> proteomics, MOSA and the multimodal encoder, models trained on large feature
> sets overfitted, and validation error increased markedly with the number of
> input features (Table S4, Fig. S_). For CyTOF (37 markers), validation error
> varied little with feature-set size. In all modalities, the lowest
> validation error was obtained with 6–16 features per modality.

Optional clarifications, if a reviewer could check the inputs:

- "…for proteomics and transcriptomics, recursive feature elimination was
  applied to the ~500 most variable features of the training lines; larger
  feature sets were evaluated by using all features…"
- "…the multimodal encoder combines the selected features of CyTOF,
  proteomics and transcriptomics (3 × 8 = 24 inputs)…"

Do not write "scanned from 4 to 4,096" without qualification, because the
omics RFE runs saturate at 501. Either say "from 4 to all available features"
or report the actual maxima (RNAseq 16,562, MassSpec ~6,200, MOSA 200,
CyTOF 37).

**Draft legend for Fig. S_:**

> Validation error as a function of the number of input features. DMMs (base
> EGFR/MAPK model) were trained on feature sets of increasing size, selected
> by recursive feature elimination on the training cell lines of each fold.
> For proteomics and transcriptomics, selection was applied after a
> highly-variable-feature prefilter (501 features). Open circles show all
> features passing the missing-value filter. Multimodal models combine the
> same number of selected features from CyTOF, proteomics and
> transcriptomics; the x-axis shows the total. (A) Validation RMSE on the
> held-out cell line, as the median over 5 random starts, then the mean ± SEM
> over 5 cross-validation folds (3 for MOSA). (B) Per-fold difference to the
> selected feature-set size (stars). The dotted line marks the number of
> training cell lines per fold (57).

## 6. For S6 (workflow)

- `run_notebooks.py` needs no change. It discovers tracked notebooks under
  `figures_paper/` via `git ls-files`, and its `savefig`/`to_csv` redirects
  catch this notebook's bare filenames.
- The notebook needs two inputs:
  - `eval/EGFR_MAPK__logobs/dream_cytof/evaluate_all_figure1c.csv`, from the
    Snakefile `evaluate_all` rule with `FIGURE=figure1c`.
  - The PEtab tables in `data/`.
- It also needs the Cellosaurus download already used by other notebooks,
  because of the MOSA cell-line mapping.
- Runtime is a few minutes, mostly spent loading the 240 MB measurement table
  and pivoting it per fold.
