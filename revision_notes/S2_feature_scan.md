# S2: Feature-count scan (Reviewer 1, Methods 2)

**Bottom line:** the data support the draft argument for RNAseq, MassSpec,
MOSA and multimodal, but not for CyTOF. With CyTOF, validation error barely
depends on the number of markers, and all 37 markers do as well as the
selected 8.

Two draft phrases need changing:

- "Increased markedly … in all modalities" needs a CyTOF exception.
- "Scanned from 4 to 4,096" needs rephrasing, because the omics RFE runs
  stop at 501 features.

Corrected wording is in section 5.

- **Data:** `eval/EGFR_MAPK__logobs/dream_cytof/evaluate_all_figure1c.csv`
  (NEMO, dated 7 Apr 2026), which contains every configured run.
- **Notebook:** `figures_paper/figure_S_feature_count_scan.ipynb`, committed
  with its outputs.

## 1. Deliverables

- **`figures_paper/figure_S_feature_count_scan.ipynb`** is the new
  supplementary figure: validation RMSE against the number of input features
  on a log axis, one line per modality.
  - Panel A shows absolute validation RMSE.
  - Panel B shows the paired per-fold difference to the selected size.
  - Stars mark the selected sizes. Open circles mark `all` features.
  - A dotted line marks the number of training cell lines per fold (57).
- The notebook computes the candidate counts in section 3 itself, using the
  pipeline's own `load_data` and `get_hvg`.
- **Outputs**, written to `notebook_output/figures/` and
  `notebook_output/data/` under `run_notebooks.py`, or to the working
  directory when run by hand:
  - `figure_S_feature_count_scan.pdf` and `.png`.
  - `figure_S_feature_count_scan_summary.csv`: one row per modality × size,
    for cross-checking Table S4.

## 2. What the scan is

- The config key is `figure1c` (`CONTEXTS_FEATURES_1C` in
  `training_configuration.py`). Despite the name, this is the feature-count
  scan behind Table S4, not paper Fig. 1C.
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
  hyperparameters. These are identical within each modality across the whole
  scan.
- Folds: **5 CV folds**, each holding out one line (BT20, HCC1500, HCC2185,
  MCF7, UACC3199), with **10 random starts each**.
  - MOSA has only 3 usable folds, because HCC2185 and UACC3199 have no MOSA
    embedding.
- Selected sizes (`BEST_FEATURE_SETS`): CyTOF RFE_8, MassSpec HVGRFE_8,
  RNAseq HVGRFE_12, MOSA RFE_6, multimodal RFE_8.

### The number of encoder inputs is not always N

1. **HVGRFE never sees more than 501 features.** The HVG prefilter keeps 501
   features. For N ≥ 401 the RFE loop does not run, and the final step keeps
   min(N, 501). So HVGRFE_512 … HVGRFE_4096 are the same model: the notebook
   checks that their RMSE is identical for every fold and start.
   - In practice, the omics RFE scan covers **4–501 features**.
   - Larger sets exist only as the `all` runs: 16,562 genes and about 6,181
     proteins.
2. **Multimodal RFE_N means 3N inputs:** N CyTOF markers, N proteins and
   N genes. Cached feature files confirm this: multimodal RFE_10 has 30
   columns. The selected multimodal model therefore has **24 inputs**, and the
   scan covers 12–96.
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
| RNAseq | 16,562 genes, complete in all 64 lines | 57 | **16,562** | **501** (HVG) | 12 |
| MassSpec | 7,877 proteins (3,751 measured in all 67 lines) | 57 | **6,177–6,187** (median 6,181) | **501** (HVG) | 8 |
| CyTOF | 37 markers. `data/cytof.csv` has 38 feature IDs, but one is `glob_cellID`, a cell index | 57 (58 in the final model) | **37** | 37 | 8 |
| MOSA | 200 latent dimensions | 45–46 (lines with a MOSA embedding) | **200** | 200 | 6 |
| Multimodal | 37 + 6,181 + 16,562 = 22,780 | 57 | — | 37 + 501 + 501 = **1,039** | 8 per modality = 24 |

**Which number to quote:**

- **16,562** (RNAseq) is both the raw and the post-filter count, because the
  RNAseq data have no missing values. "Up to 16,562 per modality" is correct
  as written.
- For MassSpec, quote ~6,200 after filtering or 7,877 raw. Do not quote 3,751:
  that complete-case count is not used anywhere in the pipeline.
- **501** is what actually enters supervised selection for the omics
  modalities.
- n = 58 is right for the final models. Each CV fold trained on 57 lines.

### Caveat on the HVG prefilter (check the Methods wording)

`get_hvg` is meant to "remove 20 % of features with the lowest mean, then keep
the top 500 by variance". However, `prepare_inputs_for_context` mean-centres
the data *before* calling it. As a result:

- The mean filter acts on floating-point residuals (|mean| < 3e-15) and
  removes 20 % of features effectively at random.
- For MassSpec the removal is biased toward high-variance features. In the
  BT20 fold it excluded 212 of the 501 most variable proteins. For RNAseq it
  excluded 34.
- So the HVG set is "the ~500 most variable features among a pseudo-random
  80 %", and it has 501 features, not 500, because of an off-by-one in the
  threshold.

Fixing this would mean rerunning selection and all downstream training, so
I am not proposing it. But if the Methods describe a mean-expression filter,
the text does not match what ran. Safer wording would be: "prefiltered to the
~500 most variable features".

## 4. Do the data support the draft claims?

RMSE values are the mean over folds of the median over 10 starts. Δ is the
paired per-fold difference to the selected size. "Worse" counts the folds in
which the larger set had higher RMSE than the selected size.

| Modality | Lowest RMSE at | Selected size (RMSE) | All features (RMSE, Δ vs selected) | Worst size (Δ vs selected) | Folds worse with all/largest |
|---|---|---|---|---|---|
| RNAseq | **12** (0.657) | 12 (0.657) | 16,562 (0.777, **+18 %**) | 256 (+21 %) | 5/5 |
| MassSpec | **8** (0.669) | 8 (0.669) | 6,181 (0.805, **+20 %**) | 128 (+26 %) | 4/5 |
| MOSA | **6** (0.583) | 6 (0.583) | 200 (0.641, **+10 %**) | 16 (+19 %) | 3/3 |
| Multimodal | **8/modality = 24** (0.533) | 24 (0.533) | largest run is 96 inputs (0.591, **+11 %**) | 72 (+16 %) | 4/5 |
| CyTOF | **16** (0.536) | 8 (0.552) | 37 (0.546, **−1 %**) | 4 (+5 %) | 1/5 |

- **"Small feature sets (6–12) yielded the lowest error"** is true for
  RNAseq, MassSpec, MOSA and multimodal (counted per modality).
  - It is not true for CyTOF: 16 markers were marginally best (−3 % vs 8,
    better in only 2/5 folds).
  - All 37 CyTOF markers were also slightly better than the selected 8
    (−1 %, better in 4/5 folds).
  - Every CyTOF size lies within 8.4 % of the best, well inside the
    fold-to-fold spread (SEM ≈ 0.05).
- **"Validation error increased markedly with the number of input features in
  all modalities"** is true for RNAseq and MassSpec (+18–20 % with all
  features; up to +26 %).
  - For MOSA and multimodal the increase is moderate: +10–11 % at the largest
    set, with a peak of +16–19 % at intermediate sizes.
  - It is **not** true for CyTOF: no trend, Spearman ρ = −0.34, p = 0.31.
- The curves are not monotone.
  - Very small sets (4 features) are also worse for every modality, which
    gives a U-shape.
  - For the omics modalities, error rises from ~12 to ~128–256 features and
    then plateaus: 501 HVGs and all 16,562 genes perform alike.
  - MOSA is noisy, with only 3 folds; its 200-dimension set beats 16–32.
- **Possible reviewer question:** why 8 CyTOF markers, when 16 or all 37 did
  as well or slightly better in the scan? The honest answer is that for CyTOF
  the choice barely matters. If the paper says 8 was "optimal" for CyTOF,
  soften it.

## 5. Proposed wording (paste-ready, US spelling)

**Response to Reviewer 1** (replaces the bracketed sentence):

> We treated the number of input features as a hyperparameter and scanned it
> from 4 features up to the complete feature set of each modality (Table S4,
> Fig. S_). For transcriptomics, proteomics, MOSA and the multimodal encoder,
> small feature sets (6–12 features per modality) yielded the lowest
> cross-validation error, and the largest feature sets increased validation
> error by 10–20%. For CyTOF, which comprises only 37 markers,
> validation error was largely insensitive to the number of selected markers,
> and the 8 selected markers performed on par with all 37.

**Methods** (replaces the draft sentence):

> Because the number of available features (up to 16,562 per modality)
> vastly exceeds the number of training cell lines (n = 58), we treated the
> number of selected features as a hyperparameter (Table S4, Fig. S_). For
> transcriptomics, proteomics, MOSA and the multimodal encoder, validation
> error was lowest for 6–12 features per modality, and models trained on
> large feature sets overfitted: validation error was 10–20% higher for the
> largest feature sets tested (up to all available features). For CyTOF (37 markers), validation error
> varied by less than 10% across feature-set sizes.

Optional clarifications, if a reviewer could check the inputs:

- "…for proteomics and transcriptomics, recursive feature elimination was
  applied to the ~500 most variable features of the training cell lines, and
  larger feature sets were evaluated by using all features that passed the
  missing-value filter…"
- "…the multimodal encoder combines the selected features of CyTOF,
  proteomics and transcriptomics (3 × 8 = 24 inputs)…"

Avoid "scanned from 4 to 4,096" without qualification: the omics RFE runs
saturate at 501 features. Either say "from 4 to all available features", or
give the actual maxima: RNAseq 16,562, MassSpec ~6,200, MOSA 200, CyTOF 37.

**Legend for Fig. S_:**

> Validation error as a function of the number of input features. DMMs (base
> EGFR/MAPK model) were trained on feature sets of increasing size, selected
> by recursive feature elimination on the training cell lines of each
> cross-validation fold. For proteomics and transcriptomics, selection was
> applied after a highly-variable-feature prefilter (501 features). Open
> circles show all features passing the missing-value filter. Multimodal
> models combine the same number of selected features from CyTOF, proteomics
> and transcriptomics; the x-axis shows the total. Stars mark the feature-set
> sizes used in the main text. The dotted line marks the number of training
> cell lines per fold (57). (A) Validation RMSE on the held-out cell line,
> shown as the median over 10 random starts, then the mean ± SEM over 5
> cross-validation folds (3 for MOSA). (B) Per-fold difference in validation
> RMSE to the selected feature-set size (mean ± SEM over folds).

## 6. For S6 (workflow)

- `run_notebooks.py` needs no change. It discovers tracked notebooks under
  `figures_paper/` via `git ls-files`, and its `savefig`/`to_csv` redirects
  catch this notebook's bare filenames.
- The notebook needs two inputs:
  - `eval/EGFR_MAPK__logobs/dream_cytof/evaluate_all_figure1c.csv`, from the
    Snakefile `evaluate_all` rule with `FIGURE=figure1c`.
  - The PEtab tables in `data/`.
- It also needs the Cellosaurus file in `features/`, which is downloaded on
  first use, for the MOSA cell-line mapping.
- Runtime is a few minutes, mostly spent loading the 240 MB measurement table
  and pivoting it per fold.
