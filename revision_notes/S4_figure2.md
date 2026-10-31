# S4: Figure 2 and the EGFR supplementary figures

Responses to Reviewer 1 (Fig. 2A, 2B, 2D, old Figs. S7–S8), Reviewer 4 (point 2) and Reviewer 3 (points 11–12).

Supplementary numbering follows the plan: old S7 → **Fig. S1** (EGFR/ERBB2 transcript vs. protein), old S8 → **Fig. S2** (EGFR peptide heat map). The two new supplementary figures are called **Fig. S-PCA** and **Fig. S-pERK** below; give them numbers when the order is fixed.

## Status

| Item | State |
|---|---|
| 1. Fig. 2 redesign (legends above every sub-panel, direct box labels, palette, CVD check) | Done; run on the NEMO `eval/` outputs. |
| 2. Fig. 2D relabel + minima; Table S4 check | Done. **The draft's minima statement doesn't hold** (F2). Table S4 is correct for CyTOF only (F1). |
| 3. New Fig. S-PCA (4 combinations) | Done (`figure_2.ipynb`, section "Figure S — Latent embeddings…"). |
| 4. Fig. 2B counts | Confirmed from the plotted embeddings: **30 luminal / 22 basal / 6 normal-like**. |
| 5. Fig. S1 highlight + threshold | Done. |
| 6. Fig. S2 column order | Fixed (`figure_S7.ipynb`). |
| 7. Optional Fig. S-pERK (R3.12) | Done. |

Both notebooks are committed with outputs from the real data (`eval/` synced from NEMO on 31 Oct 2026). `figure_2.ipynb` keeps its text statistics identical to the submitted run, e.g. CyTOF DMM RMSE 0.5456 and AUROC 0.859 → 0.744, so the synced `eval/` is the data behind the submission.

## Files changed

- `figures_paper/figure_config.py`: adds new constants only, so existing figures are unchanged. The constants are `OKABE_ITO`, `CVD_CONTEXT_COLORS`, `CVD_RECEPTOR_COLORS`, `CVD_MODEL_STYLES`, `EMBEDDING_LABELS` ("Normal" → "Normal-like"), and `EGFR_DISCORDANT_CELLS`/`EGFR_DISCORDANT_COLOR`, the shared highlight set for Figs. S1/S2.
- `figures_paper/figure_2.ipynb`: changes listed below.
  - New panel renderers. They pick the same rows as the old `barplots` functions, so only the drawing changes.
  - The composite is laid out as A (three sub-panels) / B + C / D + E, which matches the paper's letters.
  - New sections: an architecture printout, the Fig. 2D minima check, the alternative 2D in the old layout, Fig. S-PCA, Fig. S-pERK, Fig. S1 with the highlight, the threshold check and the CVD check.
- `figures_paper/figure_S7.ipynb` (Fig. S2): changes listed below.
  - Sorted by EGFR protein, with a bar strip above the heat map that shows the sort key.
  - Same highlight set as Fig. S1.
  - Peptide-support table.
  - Raw condition `184A11` renamed to `184A1`.

## Findings

### F1. Latent dimensions used for training (Table S4)

- `training_configuration.py` sets context-specific central values, which `generate_run_configs` applies to every figure run (`SELECT_CENTRAL_VALUES_BY_FIGURE` is True for figures 1a, 2, 3, 4 and 5):

  | input context | n_hidden | depth |
  |---|---|---|
  | CyTOF (`cytof_init`) | **8** | 1 |
  | Multimodal | 6 | 0 |
  | RNAseq | 7 | 1 |
  | MassSpec | 5 | 1 |
  | MOSA | 6 | 0 (global default) |
  | CyTOF + feature augmentation (`cytof_init_plus_*`, Fig. 2A right) | **4** | **0**, the global defaults, because these contexts have no override |

- The locally saved downstream models (`res/EGFR_MAPK__logobs_tegfr_aggavg/dream_cytof/*.eqx`) are 60 files: 6 splits (all, BT20, HCC1500, HCC2185, MCF7, UACC3199) × 10 jobs. All have `n_hidden = 4, depth = 0` and all are dated **2 Jan 2026**. These are the global defaults from before the context-specific overrides, so they are stale local copies and **not** the final runs.
- **Confirmed from the evaluation tables** (`print_architectures`, cell 4): every figure 1a/2/3 run uses exactly the values in the table above. CyTOF is 8/1 for all models, including the receptor integrations, and the feature-augmentation contexts are 4/0. The n_hidden scan (figure3b) varies n_hidden from 1 to 8 at depth 1 (CyTOF) and depth 0 (multimodal).
- The parent session also synced the final Fig. 3 models, `res/…/all__cytof_init__RFE_8_permute__8__1__*.eqx`, which are n_hidden 8, depth 1, 10 runs.
- **Conclusion:** "8 latent dimensions" is correct for the CyTOF-input DMMs, including the downstream CyTOF analyses. It is not correct for the other inputs. If Table S4 gives a single value, change it to the per-context values above.
- **Caveat for Fig. 2A, right (feature augmentation):** the "CyTOF only" boxes use n_hidden 8 with one hidden layer. The "+ L/B", "+ intrinsic", "+ EGFR tx" and "+ EGFR prot" boxes use n_hidden 4 with no hidden layer, because those contexts never got the CyTOF override. This confounds the comparison and is confirmed by the evaluation tables. With no new runs allowed, state it in Methods/Table S4 at least.

### F2. Fig. 2D: local and global minima

- The submitted panel hard-coded the black rings at n_hidden = 4 (DMM) and 3 (+ EGFR tx). They were not computed. The new panel computes and marks two things for each model: the **first interior local minimum** (open circle) and the **lowest test RMSE** (star). The cell below the panel prints both, for the pooled curve and for each input context separately.
- The real data, as mean test RMSE ± SEM over 10 training runs (the alternative panel reproduces the submitted panel exactly, so these are the same data):

  | n_hidden | DMM pooled | + EGFR tx pooled | DMM CyTOF | + EGFR tx CyTOF |
  |---|---|---|---|---|
  | 1 | 0.517 ± 0.004 | 0.474 ± 0.003 | 0.547 ± 0.005 | 0.480 ± 0.005 |
  | 2 | 0.512 ± 0.004 | 0.467 ± 0.005 | 0.548 ± 0.007 | 0.481 ± 0.005 |
  | 3 | 0.499 ± 0.005 | **0.451 ± 0.005** | 0.525 ± 0.008 | **0.465 ± 0.006** |
  | 4 | 0.493 ± 0.005 | 0.453 ± 0.005 | 0.523 ± 0.007 | 0.472 ± 0.008 |
  | 5 | 0.484 ± 0.007 | 0.450 ± 0.005 | 0.515 ± 0.006 | 0.461 ± 0.009 |
  | 6 | **0.479 ± 0.004** | 0.442 ± 0.004 | **0.512 ± 0.005** | 0.449 ± 0.006 |
  | 7 | 0.487 ± 0.007 | 0.442 ± 0.006 | 0.520 ± 0.009 | 0.444 ± 0.005 |
  | 8 | 0.481 ± 0.007 | 0.435 ± 0.006 | 0.504 ± 0.009 | 0.439 ± 0.007 |

  Bold marks the first interior local minimum.
- **The draft's "first local minimum at 4 for the base DMM" is wrong.** Test RMSE falls monotonically from 1 to 6, so the first local minimum is **6**. This holds in every grouping (pooled, CyTOF, multimodal). The submitted panel's ring at 4 was hard-coded.
- **"3 with EGFR transcript integration" holds** (pooled and CyTOF), but the dip is 0.002–0.007, about one SEM.
- **"Global minimum 8 for both" holds for CyTOF input** (DMM 0.504, + EGFR tx 0.439). It does not hold for the pooled curve that the submitted panel shows: there the DMM is lowest at 6 (0.479 vs. 0.481 at 8, well within the SEM). For multimodal alone, DMM is lowest at 6 and + EGFR tx at 8.
- **Overall:** every local minimum is a dip of at most about one SEM. The robust statement is that test RMSE decreases with the number of latent dimensions and flattens from about 6, with the lowest values at 6–8. Recommended response and legend text is below.
- **Recommendation:** restrict 2D to CyTOF input (`NHIDDEN_CONTEXTS = ["cytof_init"]`, one line). Then the panel matches the models it justifies (Table S4: CyTOF 8), and "lowest at 8 for both" is true. The committed figure still pools both inputs, as submitted.
- **Wording:** 8 is the largest value scanned (`LATENT_DIMS["range"]` = 1…8). It is the lowest test RMSE *within the scanned range*, not a demonstrated global minimum. Suggested text: "lowest test RMSE within the scanned range (1–8)".
- **Pooling:** the scan (`figure3b`) covers CyTOF *and* multimodal input. The submitted panel averaged both per training run, and the new one does the same by default (`NHIDDEN_CONTEXTS = None`). The multimodal models were trained with 6, not 8.
- The panel now plots test RMSE against "number of latent dimensions (n_hidden)" directly. The submitted train-vs-test layout is kept as an alternative (`fig2_panelD_trainval.pdf`) with a size legend titled "number of latent dimensions (n_hidden)".
- Side note: the size handles that `barplots.render_nhidden_trainval` returns have twice the marker area of the plotted points. The notebook rebuilds them, and `barplots.py` is untouched.

### F3. Fig. 2B counts: 30 / 22 / 6, not 30 / 23 / 5

- Code path: `load_marcotte_subtypes` → `prepare_pca_embeddings`. Marcotte `subtype_intrinsic` maps LuminalA/LuminalB/HER2 to luminal and Basal/CL to basal, and DU4475 is forced to basal. For the 58 training lines this gives **30 luminal, 22 basal, 6 normal-like** (184A1, 184B5, HDQP1, MACLS2, MCF10A, MCF12A).
- Confirmed on the plotted embeddings (real `eval/`): all four DMM / DMM + EGFR tx × CyTOF / multimodal embeddings contain the 58 training lines, 30/22/6.
- The old legend's 30/23/5 is exactly what you get if HDQP1 is counted as basal. HDQP1 is basal in the Tognetti annotation and normal-like in Marcotte, so the legend probably mixed the two sources.
- The notebook prints the counts under panel B and in the text-statistics cell. The panel legends now show them as well, e.g. "Luminal (n=30)".

### F4. Fig. 2B left panel shows DMM + EGFR tx, not the base DMM

- The submitted composite plotted `EGFR_MAPK__logobs_tegfr_aggavg` in the multimodal context but titled it "DMM (Multimodal)". The old stand-alone panel cell and the text-statistics cell used the base model instead.
- Fig. 2B is meant to stay unchanged, so the data stay as submitted (`PANEL_B_EMBEDDINGS`). The title is now derived from the plotted model: "DMM + EGFR tx (Multimodal)".
- If the base DMM was intended, change the first entry of `PANEL_B_EMBEDDINGS` to `"EGFR_MAPK__logobs"`. Fig. S-PCA shows both versions either way.

### F5. Fig. 2C uses different subtype labels from 2B

- The classifiers in 2C use the Tognetti et al. luminal/basal labels (`common.subtypes_tognetti`). The 2B colors use Marcotte.
- Of the 58 lines, 48 have a Tognetti luminal/basal label (**25 luminal, 23 basal**). Six have no Tognetti label (AU565, EFM19, HCC2218, LY2, MACLS2, MDAMB436) and four are normal.
- Two lines differ between the sources: HCC1954 (Marcotte luminal/HER2, Tognetti basal) and HDQP1 (Marcotte normal-like, Tognetti basal).
- The notebook prints all of this under panel C. Add the label source and n to the 2C legend (see below).

### F6. Fig. S1: the "log2FC > −2 = high protein" threshold

- The highlighted set (`HIGHLIGHT_CELLS` in the old S2 notebook) is exactly the lines with **no EGFR transcript (log-FPKM < 0, i.e. FPKM < 1) and EGFR protein log2FC > −2** in `data/proteomics.csv`. All 10 are in Fig. S1.
- The old ad-hoc filter (transcript < 2) also caught OCUBM, which was never highlighted.
- "High protein" is not defensible, for three reasons:
  - The median EGFR log2FC across the 58 cancer lines is −1.14, and 41/58 lie above −2.
  - −2 is the 27th percentile of all 63 lines.
  - Among transcript-positive lines (log-FPKM ≥ 0), the protein quartiles are −1.98 and −0.02.
- What *is* defensible: among the 14 transcript-negative lines, protein values split into two groups. Four lie at ≤ −2.64 (MDAMB453, ZR75B, KPL1, DU4475) and ten at ≥ −1.95. **Any threshold between −2.64 and −1.95 selects the same ten lines**, so −2 is a cut in that gap, not a claim about "high" protein. These ten lines lack transcript but have protein estimates in the range of transcript-expressing lines.
- The peptide data undercut those protein estimates. Nine of the ten have only **1–3 of the 40 EGFR peptides** detected, versus a median of 25 in the other lines. The MSstats estimate in `figures_paper/egfr_log2fc.csv` places those nine at **−6.7 to −5.1 log2FC**, among the lowest of all lines. BT483 is the exception: 31 peptides, MSstats +2.13, so it has genuine protein without transcript.
- **Recommendation:** keep the highlight but define it as transcript–protein discordance (wording below), and drop "high protein". Optionally say that the protein estimates of nine of the ten lines rest on one to three peptides. That may also explain why protein integration helps less than transcript integration, but that's the authors' call.
- Legend fix: the old S1 caption says the protein values are "log2 fold-change relative to the cross-cell-line mean". That is not true of `data/proteomics.csv`: the EGFR mean across lines is −1.27, and the per-protein means have median −0.29. **State the actual reference** from the data provenance. The new caption avoids the claim.
- The old caption's counts (n = 56; 28 luminal, 22 basal) are stale. The plotted set is **n = 54: 27 luminal, 21 basal, 6 normal-like.** The axis label typo "log-FKPM" is fixed to "log-FPKM".

### F7. Fig. S2: column order

- The columns *were* sorted, but by the MSstats log2FC in `egfr_log2fc.csv` (each line vs. the normal-like lines), not by the protein value shown in Fig. S1. The five normal-like reference lines have no MSstats value, so they were appended unsorted after the EGFR-highest line (MDAMB468), which made the order look broken.
- On the MSstats scale, the nine peptide-poor highlighted lines sit in the EGFR-low third of the columns, while Fig. S1 places them above −2. So the two figures disagreed.
- **Fix:** columns are now sorted by EGFR protein log2FC from `data/proteomics.csv`, the same value as the Fig. S1 protein axis, with all 63 lines including the normal-like ones. A bar strip above the heat map shows the sort key, with the highlighted lines in red and a dotted line at −2. `SORT_BY = "msstats"` restores the old key.
- In the new order, the highlighted lines sit among EGFR-expressing lines yet show almost empty columns. This is the visual point of S1/S2.
- The raw PXD017199 condition "184A11" (files `180501_MT_175_184A11_A–C`) is renamed to 184A1 so it matches the other tables. Please confirm it is 184A1.
- In these lines the detected peptides are mainly 38–53 (ectodomain) and 1062–1068 (C-terminal tail). UACC812 and ZR7530 show only 1062–1068.

### F8. Palette and color-vision-deficiency check

- New palette (Okabe & Ito 2008):
  - Input contexts: CyTOF blue `#0072B2`, RNAseq orange `#E69F00`, MassSpec bluish green `#009E73`, Multimodal vermilion `#D55E00`, MOSA light gray `#BBBBBB`.
  - Receptors: EGFR reddish purple `#CC79A7`, ERBB2 sky blue `#56B4E9`, ERBB3 yellow `#F0E442`.
  - Protein (and "est") boxes are hatched on a light face. Linear regression is black.
  - Subtype colors (2B) and EGFR-bin colors (2E) are unchanged.
- Check: Machado et al. (2009) simulation at severity 100 via `colorspacious`, minimum pairwise CAM02-UCS ΔE between colors sharing a panel (notebook cell "Color-vision-deficiency check"):

  | panel | normal | deutan | protan | tritan |
  |---|---|---|---|---|
  | A-i modalities | 21.6 | 15.2 | 18.9 | 11.7 |
  | A-ii mechanistic integration | 24.6 | 10.3 | 8.0* | 9.6* |
  | A-iii feature augmentation | 44.2 | 28.5 | 14.0 | 47.3 |
  | B subtypes | 49.9 | 23.8 | 20.6 | 31.3 |
  | C contexts | 62.4 | 56.2 | 49.1 | 59.4 |
  | D models | 44.2 | 28.5 | 14.0 | 47.3 |
  | E groups | 58.3 | 45.9 | 38.7 | 61.0 |
  | *submitted A-i (tab20)* | 26.1 | 5.2 | **1.6** | 10.8 |
  | *submitted A-ii (tab20)* | 27.0 | **4.3** | 5.3 | 14.1 |

  \* These are the lightened faces of two *hatched* (protein) boxes. Those boxes also differ in hatch color and carry direct labels. Among filled colors, the minimum is 11.7.
- In the submitted palette, RNAseq vs. MassSpec was nearly identical for protanopes (ΔE 1.6), and MassSpec vs. Multimodal was poor for deuteranopes (5.2).
- Same-receptor transcript/protein pairs share a hue on purpose and are told apart by the hatch, so they are excluded from the minimum.
- `colorspacious` is not in the project venv. The cell skips itself with a message if the package is missing (`pip install colorspacious`).

### F9. R3.12: pERK by subtype

These numbers come from real data (measured CyTOF).
- Grouping by Marcotte luminal/basal largely reproduces the EGFR-transcript split.
  - 19 of 22 basal lines are above the mean EGFR transcript level.
  - 21 of 30 luminal lines are below it.
  - All 6 normal-like lines are above it.
- Basal lines show a higher pERK peak and higher sustained pERK than luminal lines, just as EGFR-high lines do.
- The simulated overlay in that figure needs `eval/`.

## Legend text (paste-ready, US spelling)

**Figure 2** (panel parts; keep your title sentence)

> **(A)** Test-set RMSE (five held-out cell lines) of DMMs with different inputs. Left: base DMM trained on each input data modality. Middle: base DMM and DMMs in which EGFR, ERBB2 or ERBB3 transcript (filled boxes) or protein (hatched boxes) levels rescale the abundance of the corresponding receptor ("mechanistic integration"), for CyTOF and multimodal input; linear regression is shown for comparison. Right: base DMM, DMM + EGFR est and DMM + EGFR tx with CyTOF input alone or augmented with luminal/basal subtype, intrinsic subtype, EGFR transcript or EGFR protein as additional input features. Boxes show the distribution over ten independent training runs and are labeled individually; linear regression is a single fit (bar); dash-dotted and dotted lines mark the negative (population-average model) and positive (per-cell-line fits) controls. **(B)** First two principal components of the latent embeddings of the 58 training cell lines for DMM + EGFR tx with multimodal (left) and CyTOF (right) input, colored by intrinsic subtype (Marcotte et al.; luminal, n = 30; basal, n = 22; normal-like, n = 6). **(C)** Luminal/basal classification from the latent embeddings: cross-validated AUROC of three classifiers (symbols) for the base DMM (x axis) versus DMM + EGFR tx (filled symbols) or DMM + EGFR est (open symbols) with CyTOF or multimodal input (colors); labels from Tognetti et al. (48 cell lines; 25 luminal, 23 basal). **(D)** Test RMSE (mean ± SEM over ten training runs) versus the number of latent dimensions (n_hidden) for the base DMM and DMM + EGFR tx, averaged over CyTOF and multimodal input. Open circles mark the first local minimum (n_hidden = 6 and 3, respectively) and stars the lowest test RMSE within the scanned range (n_hidden = 6 and 8). **(E)** EGF-induced pERK (CyTOF; mean ± SEM across cell lines) in cell lines with EGFR transcript levels below (n = 24) or above (n = 34) the mean (2.68 log-FPKM); dashed lines: DMM + EGFR tx simulations (CyTOF input), averaged over training runs and cell lines. Colors follow a color-blind-safe palette (Okabe and Ito, 2008) checked under simulated deuteranopia, protanopia and tritanopia.

(D) as committed (both inputs pooled). If you restrict the panel to CyTOF (recommended, F2), use instead: "…for the base DMM and DMM + EGFR tx with CyTOF input. Open circles mark the first local minimum (n_hidden = 6 and 3, respectively) and stars the lowest test RMSE within the scanned range (n_hidden = 8 for both)."

**Figure S-PCA (new)**

> **Figure S? | Latent embeddings with and without EGFR transcript integration.** First two principal components of the latent embeddings (latent coordinates concatenated over training runs) of the 58 training cell lines for the base DMM (left) and DMM + EGFR tx (right), trained on CyTOF (top) or multimodal (bottom) input, colored by intrinsic subtype (Marcotte et al.; luminal, n = 30; basal, n = 22; normal-like, n = 6). AUROC: mean luminal/basal classification performance of the three classifiers in Figure 2C.

**Figure S1** (old S7)

> **Figure S1 | EGFR and ERBB2 transcript and protein levels across the training cell lines.** Pairwise scatter plots of EGFR and ERBB2 at the transcript (log-FPKM, RNA-seq) and protein (log₂FC, mass spectrometry) level for the n = 54 training cell lines with all four measurements (luminal, n = 27; basal, n = 21; normal-like, n = 6; Marcotte et al. intrinsic subtypes). (Top left) EGFR transcript versus protein. (Top right) ERBB2 transcript versus protein. (Bottom left) EGFR versus ERBB2 transcript. (Bottom right) EGFR versus ERBB2 protein. Pearson r is given in each panel. Red circles mark the ten cell lines without detectable EGFR transcript (log-FPKM < 0, i.e. FPKM < 1) whose EGFR protein estimate nonetheless lies above −2 log₂FC, within the range of transcript-expressing lines (dotted lines, top left): BT483, EFM19, HCC1500, LY2, MCF7, MDAMB134VI, MDAMB415, MFM223, UACC812 and ZR7530. Any threshold between −2.64 and −1.95 selects the same lines. The same lines are highlighted in Figure S2.

Add the protein reference ("log₂FC relative to …") once you've confirmed it (F6). Pearson r is shown in the panels; the value for EGFR transcript vs. protein is 0.37.

**Figure S2** (old S8)

> **Figure S2 | EGFR peptide intensities across 63 breast cell lines (PXD017199).** log₁₀ intensities of the 40 EGFR peptides (rows, ordered by sequence position; ● present in all isoforms, ◆ absent in isoform 2, ▲ canonical isoform only; gray, not detected) across cell lines (columns). Columns are sorted by EGFR protein log₂FC (bar plot on top; same values as the protein axis of Figure S1), from lowest to highest. Red: cell lines highlighted in Figure S1. In nine of these ten lines only one to three EGFR peptides are detected (median of the other lines: 25), whereas BT483 has 31.

**Figure S-pERK (optional, R3.12)**

> **Figure S? | EGF-induced pERK by EGFR transcript level and by subtype.** EGF-induced pERK (CyTOF; mean ± SEM across cell lines) in the training cell lines grouped by EGFR transcript level (left; as Figure 2E) or by luminal/basal subtype (right; Marcotte et al.; luminal, n = 30; basal, n = 22; normal-like lines omitted). Dashed lines: DMM + EGFR tx simulations (CyTOF input). The two groupings largely coincide: 19 of 22 basal lines but only 9 of 30 luminal lines have above-mean EGFR transcript levels.

## Paste-ready sentences for the response

**Placeholder "[checked for color-vision deficiency]":**
> … a color-blind-safe palette (Okabe and Ito, 2008), which we checked for color-vision deficiency by simulating deuteranopia, protanopia and tritanopia (Machado et al., 2009); colors that share a panel remain distinguishable under all three (minimum CAM02-UCS distance between filled colors 11.7, compared with 1.6 in the original figure).

**R1, Fig. 2A / R4.2 (readability):**
> We redesigned Figure 2. Each sub-panel now has its own legend placed directly above it, every box in Figure 2A is labeled on the x axis, and the panels follow a color-blind-safe palette in which each input modality and each integrated receptor has a distinct hue, with protein-level integrations hatched.

**R1, Fig. 2B:**
> We corrected the subtype counts in the legend (30 luminal, 22 basal and 6 normal-like training cell lines, intrinsic subtypes from Marcotte et al.) and now show them in the panel legend. The new Figure S? shows the latent embeddings for all four combinations of input data (CyTOF, multimodal) and EGFR transcript integration.

**R1, Fig. 2D.** The draft's "4 / 3 / 8" statement must change (F2). Wording that the data support, assuming 2D is restricted to CyTOF input:
> We relabeled the axis "number of latent dimensions (n_hidden)" and now plot the test RMSE directly against it, marking the first local minimum and the lowest value of each curve. With CyTOF input, the test RMSE decreased with the number of latent dimensions and was lowest at eight, the largest value tested, for both the base DMM and the DMM with EGFR transcript integration; earlier local minima (six and three latent dimensions, respectively) were within one standard error of the neighboring values. We therefore used eight latent dimensions for the CyTOF-input DMMs (Table S4).

If 2D stays pooled over both inputs, use instead: "…was lowest at six (base DMM) and eight (with EGFR transcript integration) latent dimensions, with values at six to eight within one standard error…".

**R1, Figs. S7–S8 → S1–S2:**
> We now highlight the same ten cell lines in both figures and define them explicitly: they lack detectable EGFR transcript (log-FPKM < 0, i.e. FPKM < 1), yet their EGFR protein estimate lies above −2 log₂FC, within the range of transcript-expressing lines; any cut-off between −2.64 and −1.95 selects the same lines. The columns of the peptide heat map (Figure S2) are now sorted by the same EGFR protein values, shown as a bar plot above the heat map. The heat map shows that the protein estimates of nine of these ten lines rest on one to three of the 40 EGFR peptides, whereas the other lines have a median of 25 detected peptides.

**R3.12** (if the optional figure is included):
> We added Figure S?, which stratifies EGF-induced pERK by luminal/basal subtype instead of EGFR transcript level. Basal cell lines show a higher and more sustained pERK response than luminal lines, mirroring the EGFR-transcript stratification of Figure 2E; the two groupings largely overlap (19 of 22 basal but 9 of 30 luminal lines have above-mean EGFR transcript levels).

## Hand-over

- **S6 (`run_notebooks.py`):** no new notebook. Fig. S-PCA and Fig. S-pERK are new sections in `figure_2.ipynb`.
  - New figure outputs: `fig2_panelA.pdf` (now the three A sub-panels), `fig2_panelB…E.pdf` in the paper's letters (the old F/G files are gone), `fig2_panelD_trainval.pdf`, `figS_latent_pca_cytof_multimodal.pdf`, `figS_perk_by_subtype.pdf` and `fig2_cvd_check.png`.
  - `figure_S7.ipynb` now imports `embeddings`/`figure_config` and reads `data/proteomics.csv` and `data/transcriptomics.csv`.
  - Optional dependency: `colorspacious`, for the CVD cell only.
- **S5 and others:** `figure_config.py` only gained constants, so existing ones are unchanged. Adopting `CVD_CONTEXT_COLORS` in Figs. 1/3/4 would make context colors consistent across the paper; that's an author decision. `barplots.render_nhidden_trainval` size-legend handles are 2× the plotted marker area (F2); `barplots.py` was not touched.
- **Author decisions left:**
  1. Pool 2D over both inputs or restrict it to CyTOF (F2).
  2. Left 2B panel: keep DMM + EGFR tx, or switch to the base DMM (F4).
  3. How to handle the feature-augmentation architecture mismatch (F1).
  4. The reference of the proteomics log2FC (F6).
  5. Whether to include Fig. S-pERK.
- **Data checks done:** each DMM box in 2A is one test RMSE per training run (10 runs). Linear regression is one deterministic value per context, duplicated across model files, so it is drawn as a bar.
