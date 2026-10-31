# S5: Figure 3/4 items for Reviewer 1

Session S5, branch `claude/revision-fig3-fig4-83b06f`. This note covers Reviewer 1 Fig. 1 point 1 ([k] of 44), Fig. 3B (P2 outliers), Fig. 3E–G (font color), Fig. 4C (color key) and the gene set enrichment wording.

Code changes on this branch:

- `figures_paper/figure_3.ipynb`
  - A new diagnostic cell after Panel B gives the per-replicate k of 44 and the P2-outlier table (items 1 and 2).
  - The composite figure (Fig. 3) gets a key for feature-name color next to the bar-color key.
  - The gene set enrichment section now describes what the code does. Two misleading strings are fixed: "Found N enriched terms (FDR<0.25)" and the dotplot title.
  - A new cell repeats the enrichment against the measured features as background.
- `figures_paper/figure_4.ipynb`
  - The composite figure (Fig. 4) now draws the Fig. 4C color key, in the free corner below the row labels.
  - `plot_panel_snp_heatmap` takes `legend_kws` so the key can be placed there.

Neither notebook was re-executed. Both need the `eval/` CSVs from NEMO. The Fig. 4C key and the Fig. 3 legend row were checked in mock renders with the real plotting code and the same layout. Stored outputs of the enrichment cells still show the old print text until the notebooks are re-run.

## Data caveat for items 1 and 2

**The numbers in items 1 and 2 are a proxy, not the Fig. 3 models.**

- **What Fig. 3 uses.** Fig. 3 reads `eval/EGFR_MAPK__logobs_tegfr_aggavg/dream_cytof/{sensitivity_params,param_devs}_figure1a.csv` (`samples == "all"`). These files come from the `all` split, trained with the current config for `cytof_init`: n_hidden = 8, depth = 1 (`res/…/all__cytof_init__RFE_8_permute__8__1__…`).
- **What is local.** None of these files are on this Mac, and `ssh crick` times out. Locally there are 60 older models of the same pipeline (January 2026): the `all_plus_missingtx` split (61 training and 5 test cell lines) plus five LOOCV splits, 10 replicates each.
- **How they differ.** The local models use n_hidden = 4 and depth = 0. All other hyperparameters are the same: L1 = 0.1 on Δ_c until epoch 200, GMM sparsification, dropout 0.1, inflater bound 3, 500 epochs.
- **How I computed them.** Masks were read from the `.eqx` files. Ablation ΔRMSE was recomputed with the repository's `sensitivity.set_param_to_zero` and `rmse`, run against the main checkout's data and compiled AMICI model.

**To get the real numbers**, run one of these:

1. `figure_3.ipynb` with the NEMO `eval/` files. The new cell after Panel B prints k per replicate, how often each parameter was retained, and a per-replicate P2 table: ΔRMSE, train/test RMSE, which of P1–P6 were kept, Tukey-outlier flags, the P2–P3 correlation, and how well each replicate's P2 profile agrees with the other replicates.
2. The snippet below, from the repository root on NEMO. It needs only numpy and the `.eqx` files.

```python
# Count cell-line-variable parameters (output sparsity mask) per replicate
# straight from the serialized .eqx files (no AMICI/pyPESTO needed).
import json
from pathlib import Path

import numpy as np

RES = Path("res/EGFR_MAPK__logobs_tegfr_aggavg/dream_cytof")
PRETRAIN = Path("pretrain/EGFR_MAPK__logobs_tegfr_aggavg/dream_cytof")
names = [c[4:] for c in next(PRETRAIN.glob("c*.csv")).open().readline().strip().split(",")
         if c.startswith("MED_")]  # the 44 deviation parameters, in model order

for f in sorted(RES.glob("all__cytof_init__*.eqx")):
    with f.open("rb") as fh:
        hdr = json.loads(fh.readline())
        leaves = []
        while fh.peek(1):
            leaves.append(np.lib.format.read_array(fh))
    # learned_global_kin_params is the only 1-D leaf longer than 44; it is followed by the
    # input mask (n_input_features entries) and the 44-entry output sparsity mask
    i = next(j for j, a in enumerate(leaves) if a.ndim == 1 and a.size > 44)
    n_in = hdr["n_input_features"]
    mask = np.array([a.item() for a in leaves[i + 1 + n_in:i + 1 + n_in + 44]])
    kept = [p for p, m in zip(names, mask) if m]
    print(hdr["conf"]["job"], int(mask.sum()), kept)
```

## 1. "[k] of 44" (Reviewer 1, Fig. 1 point 1)

**Proxy result (`all_plus_missingtx` split, 10 replicates):**

- k per replicate (jobs 0–9): 16, 17, 18, 16, 15, 15, 16, 20, 16, 24.
- **k = 16 of 44 (median), range 15–24.**
- Across all 60 local models (including the 50 LOOCV-split models): median 18, range 13–24.
- Dropped parameters have Δ_c = 0 exactly for every cell line. Retained ones vary across the training cell lines with a standard deviation of 0.04–1.2 (median 0.24) in the model's log10 parameter scale. So "retained" and "cell-line-variable" coincide.
- Retained in all 10 replicates:
  - P1 (ERK → p90RSK, `RPS6KA1_p_S380_ERK__Y204_p_kw`)
  - P3 (p.EGFR → p.ERBB2, `ERBB2_p_Y1248_EGFR__Y1173_p_kw`)
  - P6 (iMEK ⊣ p.ERK, `ERK_p_Y204_iMEK_0_kw`)
- Retained in 9 of 10: P2 (`ERBB2_p_Y1248_ERBB2_kw`, dropped in job 7) and `ERK_p_Y204_TOPK__Y74_p_kw`.
- Retained in 8 of 10: `EGFR_deg_kdeg` and `RPS6KA1_p_S380_bact`.
- Retained in 7 of 10: P4 (`MEK_p_S222_EGFR__Y1173_p_kw`), `ERK_p_Y204_MEK__S222_p_kw` and `MEK_p_S222_ERBB2__Y1248_p_kw`. P5 (`MEK_p_S222_ERK__Y204_p_kw`) is retained in 6 of 10.
- In total, 37 of the 44 parameters are retained in at least one replicate. The variable set therefore differs between replicates, apart from a stable core.

**Real-model evidence already in the stored outputs** (executed `figure_3.ipynb`): only **6 of 44** parameters have an ablation effect on the training RMSE that differs significantly from zero across the 10 replicates (one-sample t-test, BH-adjusted p < 0.05). These are P1–P6 in Fig. 3B.

**Implication for the wording.** If the real models resemble the proxy, "a small subset" overstates the sparsity. Roughly a third of the parameters stay variable, though only a handful matter for the fit. Suggested text, with numbers to confirm from NEMO:

> "…the L1 penalty on Δ_c, followed by GMM-based sparsification, restricts cell-line variability to a subset of the kinetic parameters (here [median k] of 44, range [min–max] across the ten training replicates), of which only six have a significant effect on model fit when ablated (Fig. 3B)."

If the real k turns out to be small, for example ≤ 10, keep "only a small subset (here k of 44)".

## 2. Fig. 3B: P2 outlier replicates

### What the published panel shows

These values were read off the vector PDF of the Fig. 3 composite (`notebook_output/figures/figure_3__composite_…pdf`), at 800 dpi.

- **The P2 outliers are high, not low.** Two training points lie at ΔRMSE ≈ 0.09 and 0.13 (the rest are about 0.02–0.045). Two test points lie at ≈ 0.25 and 0.31 (the rest are about 0.02–0.075). These are most likely the same two replicates.
- **This argues against the draft's explanation (a)** ("P2 not retained, so ablation has no effect"). That explanation would produce points at exactly 0, not at the top.

The analogous mechanism that does fit is **missing compensation**. In some replicates, the GMM sparsification drops the parameters that co-vary with P2 (Fig. 3C: P3 ρ = 1.0, P4 ρ = 0.8). P2 then carries the whole ERBB2 axis alone, and ablating it costs more.

### Proxy test (10 local replicates; ablation recomputed)

| job | k | P2 ΔRMSE train | P2 ΔRMSE test | P2 | P3 | P4 | P5 | RMSE train | RMSE test | P2 profile ρ vs other replicates |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 16 | 0.003 | 0.018 | ✓ | ✓ | ✓ | ✓ | 0.483 | 0.516 | 0.98 |
| 1 | 17 | **0.030** | **0.038** | ✓ | ✓ | – | – | 0.477 | 0.495 | 0.99 |
| 2 | 18 | 0.007 | 0.020 | ✓ | ✓ | ✓ | ✓ | 0.462 | 0.480 | 1.00 |
| 3 | 16 | **0.027** | **0.032** | ✓ | ✓ | – | – | 0.478 | 0.491 | 0.99 |
| 4 | 15 | 0.014 | 0.033 | ✓ | ✓ | ✓ | ✓ | 0.473 | 0.455 | 0.96 |
| 5 | 15 | **0.032** | **0.046** | ✓ | ✓ | – | – | 0.481 | 0.492 | 0.98 |
| 6 | 16 | 0.003 | 0.016 | ✓ | ✓ | ✓ | ✓ | 0.477 | 0.512 | 0.99 |
| 7 | 20 | 0 (dropped) | 0 (dropped) | – | ✓ | ✓ | ✓ | 0.477 | 0.544 | n/a |
| 8 | 16 | 0.009 | 0.018 | ✓ | ✓ | ✓ | – | 0.486 | 0.490 | 0.98 |
| 9 | 24 | −0.000 | 0.002 | ✓ | ✓ | ✓ | ✓ | 0.474 | 0.524 | 0.93 |

✓ = parameter retained by the sparsification; – = dropped.

1. **The size of the P2 effect depends on which correlated parameters were retained.** The three replicates with the largest P2 ablation effect (jobs 1, 3, 5) are exactly those in which P4 (p.EGFR → p.MEK) and P5 were dropped.
   - Train ΔRMSE is 0.027–0.032 there, against −0.000–0.014 in the six replicates that retained P2 and P4 (Mann–Whitney p = 0.024). Test ΔRMSE is 0.032–0.046 against 0.002–0.033 (p = 0.048).
   - Where both are retained, P4 co-varies with P2 across cell lines (median Spearman ρ = 0.71), and P3 even more so (ρ = 0.92).
   - So P4 can partly take over P2's role when P2 is ablated. Where P4 is absent, it cannot.
   - The size of the P2 effect also tracks how variable P2 is across cell lines (Spearman ρ = 0.83 between P2 test ΔRMSE and the SD of P2 across cell lines, p = 0.003).
2. **The draft's explanation (a), taken literally, occurs once but gives a low point.** In job 7, P2 was dropped, so its ablation effect is exactly 0. In job 9, k = 24, and many retained parameters absorb the ablation (effect ≈ 0). The published panel has no P2 points at 0, so in the Fig. 3 models P2 was presumably retained in all 10 replicates.
3. **There is no evidence for different local optima (b).**
   - The training RMSE of jobs 1, 3 and 5 (0.477–0.481) is within the range of the others (0.462–0.486). Their test RMSE (0.491–0.495) is within the range too (0.455–0.544).
   - P2 ablation effect and training RMSE are unrelated (Spearman ρ = 0.03).
   - Each replicate's P2 deviation profile across cell lines matches the median profile of the other replicates (Spearman ρ = 0.93–1.00). The same holds for P1 (ρ = 0.95–0.99). The replicates therefore infer the same cell-line pattern and differ only in how the effect is shared among correlated parameters.
4. **Caveat: the proxy shows no Tukey outliers of the published size.** Its largest P2 test ΔRMSE is 0.046, against 0.25–0.31 in Fig. 3B. The mechanism therefore still has to be confirmed on the Fig. 3 models (n_hidden = 8, depth = 1). The new diagnostic cell in `figure_3.ipynb` does exactly this.

### Paste-ready text (fill the brackets after running the diagnostic cell on the NEMO files)

**If the real outliers lack P3 or P4 and have normal RMSE** (the outcome the proxy predicts):

> Response: "We have examined the outlying replicates and found that they are not failed fits. Their training and test errors lie within the range of the other replicates, and the cell-line pattern of their inferred P2 deviations agrees with that of the other replicates (Spearman ρ ≥ [x]). Instead, the magnitude of the P2 ablation effect depends on which of the parameters that co-vary with P2 (Fig. 3C) were retained by the sparsification. In the two outlying replicates, [P4 / P3] was not retained, so P2 alone accounts for this axis of cell-line variability and its ablation costs more. Where these parameters are retained, they partly compensate for the ablation of P2. We now note this in the figure legend."

> Legend sentence (Fig. 3B): "Two replicates show markedly larger effects of ablating P2 (ΔRMSE [0.09 and 0.13] on training and [0.25 and 0.31] on test cell lines). In these replicates, the sparsification did not retain the co-varying parameter [P4] (C), so that P2 alone carries this axis of variability; their training and test errors are within the range of the other replicates."

**If the outliers instead show a clearly worse training RMSE, or a P2 profile that disagrees with the consensus (ρ well below the others)**, use (b):

> "…found that the two outlying replicates converged to a different local optimum, with [higher training error / a different distribution of the ERBB2 axis across parameters]. The remaining eight replicates agree closely."

**If neither applies**, report it plainly ("…found no systematic difference in fit quality or in the retained parameters…"), and keep the legend sentence descriptive.

## 3. Fig. 3F: what the font color encodes

**The draft is wrong.** The draft says the font color "indicates genes associated with the corresponding biological process in Fig. 3G". In the code (`draw_corr_barplot` in `figures_paper/de_corr_plots.py`, called from the composite cell of `figure_3.ipynb`), there is no biological-process grouping for Fig. 3.

What the code actually draws:

- A feature name is drawn in the color of its parameter (P1 green, P2 orange, the same colors as the P labels in Fig. 3B/C/E) when its correlation is significant: Pearson correlation, Benjamini–Hochberg FDR < 0.05 within each parameter × data type. Otherwise the name is grey.
- This duplicates the bar shading. Dark red or blue bars mean FDR < 0.05; light bars mean FDR ≥ 0.05.
- Dashed vertical lines in the parameter color mark the correlation of the canonical pathway genes: ERBB2 for P2, and RPS6KA1, RPS6KA2 and RPS6KA3 for P1. They are drawn only for RNA-seq and proteomics.
- Labels colored by biological process exist only in **Fig. 4**. There, the p90RSK-gap volcano plots color labels by cytoskeletal dynamics, endo/exocytosis, or calcium sensing/dynamics (`_P90RSK_GENE_GROUPS`). That figure has no key for these colors either.
- The draft sentence may have been meant for Fig. 4, or for a version of Fig. 3 edited by hand in Illustrator. Please check the submitted figure file.

**Figure change.** The bottom legend of the Fig. 3 composite now has two extra entries: "Aa Aa" in the P2/P1 colors labeled "FDR<0.05 (name)", and a grey "Aa" labeled "FDR≥0.05 (name)".

**Legend line (paste-ready):**

> (F) Top ten features ranked by absolute Pearson correlation with the inferred deviations of P2 (top) and P1 (bottom) across cell lines, for RNA-seq, mass-spectrometry proteomics and CyTOF baseline data. Bars are dark if the correlation is significant (Benjamini–Hochberg FDR < 0.05) and light otherwise (red, positive; blue, negative). Significant features are also labeled in the color of the respective parameter, and non-significant features in grey. Dashed vertical lines indicate the correlation of the canonical pathway members ERBB2 (P2) and RPS6KA1–3 (P1).

**Response wording (replaces the draft clause):** "…the font color of the feature names marks correlations that are significant at FDR < 0.05, shown in the color of the respective parameter, with non-significant features in grey. We have added this to the figure key and the legend."

## 4. Fig. 4C: color key

**The draft is partly wrong.**

- Correct as drafted: full saturation means the variant is reported by ≥ 2 sources; pale means a single source; black means no mutation data.
- Not correct: "each color indicates the affected gene". The fill color encodes the **variant class**, taken from `_MUT_TYPE_COLORS` in `figure_4.ipynb`: missense `#2166ac` blue, nonsense red, frameshift green, in-frame purple, splice `#e08214` orange, other grey. The gene and the amino-acid change are given by the row label.
- Among the nine variants shown, only missense (blue) and splice (orange, MET Δex14 in HCC2218) occur.

Sources counted:

- cBioPortal CCLE (Broad, 2019)
- Sanger Cell Model Passports (COSMIC cell lines)
- Marcotte et al. (2016) Table S1C exome sequencing (`figures_paper/mmc2.xlsx`)

Black cells mark the 12 cell lines that have no data in either cBioPortal or Cell Model Passports. They are not drawn in the KRAS G12/13, MET N375S and BRAF V600E rows, because the Marcotte exome table covers those genes, and I checked that it covers all 12 of these cell lines.

Hits per stored notebook output:

| Variant | Cell lines [sources] |
|---|---|
| RAC1 N39S | CAL51 [cbio+cmp] |
| RAC1 P29S | MDAMB157 [cbio] |
| RHOA G17E | CAL148 [cbio+cmp], BT474 [cbio+cmp] |
| KRAS G12/13 | MDAMB231 [cbio+cmp+mmc2], MPE600 [mmc2] |
| MET N375S | UACC893 [mmc2], HCC2218 [mmc2] |
| MET Δex14 | HCC2218 [cbio] |
| BRAF V600E | MDAMB361 [cbio], DU4475 [cbio+cmp] |
| RAF1 E478K | CAL148 [cbio] |
| MAP2K1 E203K | CAL148 [cbio+cmp] |

**Figure change.** The composite now draws the key (Missense, Splice, Single source, No mutation data) in the empty corner below the row labels and left of the cell-line labels. It was checked in a mock render with the real plotting function and the composite layout.

**Legend line (paste-ready):**

> (C) Selected hotspot variants (rows) across cell lines (columns), grouped by ΔRMSE-gap and training-error subgroups (color strips as in A and B). Fill color indicates the variant class (blue, missense; orange, splice site); the affected gene and amino-acid change are given by the row label. Fully saturated cells mark variants reported by at least two independent sources (cBioPortal CCLE, Cell Model Passports and Marcotte et al., 2016); pale cells mark variants reported by a single source. Black cells mark cell lines without mutation data in cBioPortal or Cell Model Passports. For KRAS, MET N375S and BRAF V600E, these cell lines are covered by the exome data of Marcotte et al. (2016).

**Response wording:** "We have added a color key to Fig. 4C. The fill color indicates the variant class (missense or splice site), and the affected gene is given by the row label. Full saturation indicates variants confirmed by at least two independent sources, pale coloring indicates single-source variants, and black indicates cell lines without mutation data."

## 5. Gene set enrichment: what was tested and found

What the notebook section "Gene Set Enrichment — MSigDB Signaling Pathways" (`figure_3.ipynb`) actually does. It differs from its own markdown in several points, now corrected.

- **Foreground.** Features whose **Pearson correlation** with the parameter deviation is significant (BH FDR < 0.05, `corr_results`, the same values as Fig. 3F). It does **not** use DE genes, and it does not split up- and down-regulated features. Runs with fewer than five features are skipped.
- **Test.** Over-representation analysis via Enrichr (`gseapy.enrichr`, gseapy 1.1.13) against 12 libraries:
  - MSigDB Hallmark 2020 and Oncogenic Signatures
  - Reactome 2024, KEGG 2026, WikiPathways 2024
  - GO BP 2023 and 2025, GO MF 2023, GO CC 2023
  - CellMarker 2024, Pfam 2019, InterPro 2019
- **Correction.** Adjusted p-values are BH within each library.
- **Background.** None is passed: `background=all_genes if PANEL_CORR_TYPES == ["transcriptomics"] else None`, and the notebook runs three data types. So Enrichr uses its genome-wide default rather than the measured features.
- **"517 terms at FDR < 0.25" is a mislabel.** `enr.results` lists every term with at least one overlapping gene, without any filter. Of the 517 terms, 23 have adjusted P < 0.05. Ten of these match the "signaling" keyword filter.

What qualified for testing:

| Data type | P1 (ERK → p90RSK) | P2 (ERBB2 autoactivation) |
|---|---|---|
| RNA-seq | 2 genes (NCDN, MAP3K8): skipped | 0: skipped |
| Proteomics | **13 proteins**: tested | 0: skipped |
| CyTOF | not applicable (epitope labels) | not applicable |

The 13 proteins are PLS1, MFHAS1, NCDN, NAA16, CDR2L, MRPL27, MAPKAPK5, RPS6KA3, FLJ45252, ANXA11, BAG5, MRPL3 and NHP2.

Top terms for proteomics/P1 with Enrichr adjusted P < 0.05 (per-library BH, genome background). Each term is driven by one to four proteins:

| Library | Term | Overlap | Adj. P | Genes |
|---|---|---|---|---|
| InterPro | BAG domain | 1/5 | 0.026 | BAG5 |
| Pfam / InterPro | Ribosomal L7Ae | 1/11 | 0.029 | NHP2 |
| GO MF | Calcium-dependent protein kinase activity | 1/7 | 0.033 | MAPKAPK5 |
| GO MF | MAP kinase kinase activity | 1/14 | 0.033 | MAPKAPK5 |
| GO MF | RNA binding | 4/1411 | 0.033 | MRPL3, MRPL27, NHP2, ANXA11 |
| GO MF | Cysteine-type endopeptidase inhibitor activity | 1/19 | 0.033 | RPS6KA3 |
| GO MF | Ubiquitin protein ligase binding | 2/271 | 0.033 | MFHAS1, BAG5 |
| GO MF | Protein Ser/Thr kinase activity | 2/342 | 0.033 | RPS6KA3, MAPKAPK5 |
| Reactome | Mitochondrial translation initiation/termination | 2/91 | 0.035 | MRPL3, MRPL27 |

Robustness check (new cell in the notebook; run here on the same 13 proteins):

- **Test.** Hypergeometric test against the **6,775 quantified proteins** (≤ 50 % missing across the 63 cell lines), on MSigDB C2 canonical pathways (`data/msigdb_c2cp_2025.1.Hs.json`, 3,063 sets with 5–500 measured members), with BH correction across all sets.
- **Top hits:**
  - BIOCARTA MAPK pathway: MAPKAPK5 and RPS6KA3, p = 0.004
  - Reactome RSK activation: RPS6KA3, p = 0.010
  - Reactome mitochondrial translation: MRPL3 and MRPL27, p = 0.012
- **Result: no set survives correction (minimum FDR = 1.0).** The same holds with a genome-like background (all 13,868 C2:CP genes).

**Why it was not informative (for the response):** For three of the four omics × parameter combinations, too few features pass the correlation FDR threshold for an over-representation test. In the one testable list (13 proteins correlated with P1), the nominal hits rest on one or two proteins each. They are either direct members of the modeled pathway (RPS6KA3, MAPKAPK5), which is expected, or generic housekeeping functions (mitochondrial ribosome, BAG domain). None of them survives correction for multiple testing across gene sets when the measured proteome is used as background.

**Suggested response wording:** "Standard over-representation analysis of the features significantly correlated with each parameter (Enrichr; MSigDB, Reactome, KEGG, WikiPathways and GO libraries) was not informative. Only the proteomic correlates of P1 (13 proteins) were numerous enough to test. The resulting terms were each supported by one or two proteins, mostly direct pathway members such as RPS6KA3 and MAPKAPK5, and none remained significant against the measured proteome as background with correction across gene sets. We therefore report the individual correlates (Fig. 3F) rather than pathway-level enrichments."

**Corrected Methods sentence (paste-ready):**

> For each data type and parameter, features whose Pearson correlation with the inferred parameter deviation was significant (Benjamini–Hochberg FDR < 0.05; at least five features required) were tested for over-representation with Enrichr (gseapy 1.1.13) against the MSigDB Hallmark and Oncogenic Signatures, Reactome, KEGG, WikiPathways, Gene Ontology, CellMarker, Pfam and InterPro libraries, with Benjamini–Hochberg correction within each library. Results were checked with a hypergeometric test against all quantified features as background on MSigDB C2 canonical pathways (v2025.1; gene sets with 5–500 measured members), with Benjamini–Hochberg correction across all gene sets.

**Recommended follow-up (not done here, because it changes results):** In the Enrichr call, pass `background=all_genes` for every data type instead of only when `PANEL_CORR_TYPES == ["transcriptomics"]`. If you do, the Methods sentence above can drop the "checked with" clause. A rank-based GSEA (prerank on the full correlation profile) would be the natural test for "correlation profiles", but it needs `param_devs_figure1a.csv` from NEMO.
