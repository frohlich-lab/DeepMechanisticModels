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

**Verified on the real data.** Both notebooks were executed end to end with `run_notebooks.py figure_3` (332 s) and `run_notebooks.py figure_4` (298 s), using the NEMO `eval/`/`res/` files and the main checkout's `data/`, linked temporarily into the worktree. Both finished without errors.

- In the Fig. 3 composite, the new label-color key sits in the bottom legend row without overlaps. The rest of the figure is unchanged from the published version.
- In the Fig. 4 composite, the 4C key sits in the free corner below the row labels. The SNP hit list is unchanged: 13 hits, 12 cell lines without mutation data.
- The new diagnostic cell and the robustness-check cell produce the numbers below.

The repository copies of the notebooks keep their old stored outputs. Re-run `run_notebooks.py` to refresh them.

## Data used for items 1 and 2

These numbers come from the Fig. 3 models themselves.

- The NEMO outputs are synced into the main checkout: `eval/EGFR_MAPK__logobs_tegfr_aggavg/dream_cytof/{sensitivity_params,param_devs}_figure1a.csv` (context `cytof_init`, samples `all`), and the 10 models `res/…/all__cytof_init__RFE_8_permute__8__1__…__{0..9}__500__3.eqx` (n_hidden = 8, depth = 1).
- Each replicate covers 58 training and 5 test cell lines.
- k was obtained two independent ways, which agree exactly for every replicate:
  - from the output sparsity mask in each `.eqx` file;
  - from `sensitivity_params`. A dropped parameter leaves the model unchanged when zeroed, so both its zeroed and activated ΔRMSE are exactly 0.
- The table in item 2 is the output of the new diagnostic cell in `figure_3.ipynb`, run on these files.

An earlier pass on older local models (n_hidden = 4, depth = 0) is superseded. It had predicted the same mechanism for the P2 outliers.

To recount from the `.eqx` files without AMICI or pyPESTO, run this from the repository root:

```python
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

**Result: k = 16 of 44 (median; mean 16.1), range 10–21 across the 10 training replicates.**

- k per replicate (jobs 0–9): 17, 15, 19, 21, 10, 17, 20, 15, 14, 13.
- Retained parameters vary across the 63 cell lines with SD 0.07–1.10 (median 0.26) in the model's log10 parameter scale. Dropped parameters have Δ_c = 0 exactly.
- Retained in all 10 replicates:
  - P1 (ERK → p90RSK, `RPS6KA1_p_S380_ERK__Y204_p_kw`)
  - P2 (ERBB2 autoactivation, `ERBB2_p_Y1248_ERBB2_kw`)
- Retained in 9 of 10:
  - P3 (p.EGFR → p.ERBB2, `ERBB2_p_Y1248_EGFR__Y1173_p_kw`; dropped in job 9)
  - P6 (iMEK ⊣ p.ERK, `ERK_p_Y204_iMEK_0_kw`; dropped in job 0)
- Retained in 8 of 10: `ERK_p_Y204_MEK__S222_p_kw`.
- Retained in 7 of 10:
  - P4 (`MEK_p_S222_EGFR__Y1173_p_kw`)
  - P5 (`MEK_p_S222_ERK__Y204_p_kw`)
  - `ERBB2_p_Y1248_serum_0_kw`
  - `ERBB2_p_Y1248_iEGFR_0_kw`
  - `MEK_p_S222_ERBB2__Y1248_p_kw`
- 34 of 44 parameters are retained in at least one replicate. Ten are never retained:
  - `EGFR_recycle_kcat`, `ERBB3_dp_Y1289_kcat`, `ERK_dp_Y204_kcat`, `MEK_dp_S222_kcat`, `RPS6KA1_dp_S380_kcat`
  - `ERBB2_p_Y1248_ERBB3__Y1289_p_kw`, `MEK_p_S222_iMEK_0_kw`
  - `TOPK_p_Y74_bact`, `TOPK_p_Y74_EGFR__Y1173_p_kw`, `TOPK_p_Y74_ERBB3__Y1289_p_kw`
- Only **6 of 44** parameters (P1–P6) have an ablation effect on the training RMSE that differs significantly from zero across replicates (one-sample t-test, BH-adjusted p < 0.05; Fig. 3B).

**Wording.** About a third of the parameters stay variable, so "a small subset" is generous. Suggested text:

> "…the L1 penalty on Δ_c, followed by GMM-based sparsification, restricts cell-line variability to a subset of the kinetic parameters (here 16 of 44; range 10–21 across the ten training replicates), of which only six have a significant effect on model fit when ablated (Fig. 3B)."

## 2. Fig. 3B: P2 outlier replicates

**The outlying replicates are jobs 7 and 9.** They are Tukey outliers in both the training and the test set, and match the two isolated points in the published panel.

| job | k | P2 ΔRMSE train | P2 ΔRMSE test | P3 | P4 | P5 | P6 | RMSE train | RMSE test | ρ(P2, P3) | P2 profile ρ vs other replicates |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 17 | 0.022 | 0.030 | ✓ | ✓ | ✓ | – | 0.422 | 0.447 | 0.97 | 0.96 |
| 1 | 15 | 0.046 | 0.075 | ✓ | ✓ | ✓ | ✓ | 0.383 | 0.428 | 0.97 | 0.98 |
| 2 | 19 | 0.031 | 0.048 | ✓ | ✓ | ✓ | ✓ | 0.397 | 0.444 | 0.96 | 0.98 |
| 3 | 21 | 0.038 | 0.057 | ✓ | ✓ | ✓ | ✓ | 0.386 | 0.437 | 0.93 | 0.99 |
| 4 | 10 | 0.040 | 0.051 | ✓ | – | ✓ | ✓ | 0.424 | 0.481 | 0.96 | 0.88 |
| 5 | 17 | 0.043 | 0.071 | ✓ | ✓ | ✓ | ✓ | 0.382 | 0.445 | 0.90 | 0.99 |
| 6 | 20 | 0.023 | 0.035 | ✓ | ✓ | ✓ | ✓ | 0.401 | 0.424 | 0.99 | 0.97 |
| **7** | 15 | **0.087** | **0.249** | ✓ (SD 0.31) | – | – | ✓ | 0.386 | 0.416 | 0.81 | 0.97 |
| 8 | 14 | 0.018 | 0.022 | ✓ | ✓ | – | ✓ | 0.411 | 0.454 | 1.00 | 0.95 |
| **9** | 13 | **0.126** | **0.311** | – | – | – | ✓ | 0.383 | 0.402 | n/a | 0.90 |

P1 and P2 are retained in all replicates (not shown). ✓ = retained; – = dropped by the sparsification.

Findings:

1. **(a) as drafted is ruled out.** P2 is retained in all 10 replicates, so there is no replicate in which its ablation "has no effect". The outliers are replicates with unusually *large* effects: train ΔRMSE 0.087 and 0.126 against 0.018–0.046, and test ΔRMSE 0.249 and 0.311 against 0.022–0.075.
2. **The variant of (a) that does hold is that the co-varying parameters were not retained.** Jobs 7 and 9 are the only two replicates in which both P4 (p.EGFR → p.MEK) and P5 (p.ERK ⊣ p.MEK) were dropped. Job 4 lacks P4 but keeps P5 and a strongly variable P3; job 8 lacks P5 only. P3 (p.EGFR → p.ERBB2) co-varies most strongly with P2 (Fig. 3C ρ = 1.0; within replicates ρ = 0.90–1.00):
   - Job 9 is the only replicate that dropped P3 altogether.
   - In job 7, P3 is retained but almost invariant across cell lines (SD 0.31 against 0.56–1.10 in the other replicates), and it is the least correlated with P2 (ρ = 0.81).

   In both replicates, P2 alone therefore carries this axis of cell-line variability. Its ablation cannot be absorbed by correlated parameters, and the error rises more. P2's own spread is not larger in the outliers (SD 0.27 and 0.34 against 0.26–0.38).
3. **(b), a different or worse local optimum, is ruled out.**
   - Both outliers are among the *best* fits. Their test RMSE (0.402 and 0.416) is the lowest and second lowest of the ten replicates (others 0.424–0.481). Their training RMSE (0.383 and 0.386) ranks 2nd and 4th of ten (others 0.382–0.424).
   - Their P2 deviation profiles across cell lines agree with the median profile of the other replicates (Spearman ρ = 0.90 and 0.97, against 0.88–0.99 among the rest).

**Response (paste-ready):**

> "We have examined the two outlying replicates and found that they are not failed fits. They have the lowest test errors of the ten replicates (RMSE 0.40 and 0.42, compared with 0.42–0.48) and training errors within the range of the others, and the cell-line pattern of their inferred P2 deviations agrees with that of the other replicates (Spearman ρ = 0.90 and 0.97). Instead, they are the two replicates in which the sparsification removed the parameters that co-vary with P2 (Fig. 3C): P4 (p.EGFR → p.MEK) and P5 (p.ERK ⊣ p.MEK) were dropped in both, and P3 (p.EGFR → p.ERBB2) was dropped in one and nearly invariant across cell lines in the other. In these replicates, P2 alone accounts for this axis of cell-line variability, so its ablation cannot be compensated by correlated parameters and increases the error more. We now note this in the figure legend."

**Legend sentence for Fig. 3B (paste-ready):**

> "In two of the ten replicates, ablation of P2 increases the RMSE markedly more (training 0.09 and 0.13; test 0.25 and 0.31). In these replicates, the parameters that co-vary with P2 (P3–P5; C) were largely removed by the sparsification, so their contribution cannot compensate for the ablation; both replicates fit the data as well as the others."

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
