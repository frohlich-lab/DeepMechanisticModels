# HVG prefilter in `select_features.py`: fix and impact

Branch `claude/objective-rubin-5b7968`.

- **Analysis script:** `revision_notes/hvg_prefilter_impact.py`.
  - It imports `rf_inputs`, `get_hvg` and `hvg_diagnostics` from
    `figures_paper/receptor_evidence.py` on branch
    `claude/revision-receptor-evidence-b04f76`. So it runs once that branch is
    merged, or with that file on `PYTHONPATH`.
  - It takes a few minutes on the laptop. Everything below was computed on
    this laptop, not on the cluster.
- **Implemented: fix B, no expression filter (section 1).** The low-mean step
  is gone, and the prefilter keeps the 501 most variable features of each
  omics context.
  - FF chose it over fix C, which would add the Fig. 4 DE filter for RNAseq
    (median log2 expression > −2.5). Fix C was committed first (9142ba2) and
    then replaced.
  - Fix C would remove 74–78 of the 501 most variable transcripts: subtype
    markers that are on in a minority of lines.
- **Checked end to end.** Running the patched `select_features.py` with
  `--features=HVG_all` reproduces the script's fix-B sets exactly, for BT20
  and `all`, in both contexts.
- **Regenerated so far (section 4):**
  - The `figure1c` feature-count scan was rerun with fix B on NEMO.
  - Its new minima re-pinned `BEST_FEATURE_SETS`: 4 genes, 10 proteins and
    6 features per modality for multimodal.
  - `figure1b` is rerunning on those sizes.
  - `figure1a`, `figure3` and `figure3b` are still on the old sizes.

## Summary

- **The bug.** `prepare_inputs_for_context` mean-centers the omics inputs
  before `get_hvg` runs. The first step of `get_hvg` ("drop the 20 % of
  features with the lowest mean") therefore ranks floating-point residuals
  (|mean| < 4e-15).
- **It is not random noise. It is biased.** The rounding residuals scale with
  the size of the values.
  - For MassSpec, the removal rate rises from 3 % in the lowest-variance
    quintile to 35 % in the highest. In each split, 185–212 of the 501 most
    variable proteins are removed (about 100 would be expected at random).
  - Most affected are VIM (variance rank 1, kept in 1 of 6 splits), AGR2 (4,
    1/6), CD44 (8, 2/6), ERBB2 (25, 1/6) and EGFR (≈130, 3/6). 24 of the 50
    most variable proteins are missing from the current set in at least 3 of
    6 splits.
  - For RNAseq the bias runs the other way: 30–38 of the 501 most variable
    genes are removed. Lost genes include VIM (rank 10–13, kept in 4/6),
    MGP (rank 19–23, kept in 1/6) and FOXA1 (rank 144–171, kept in 4/6).
- **Why no expression filter.**
  - MassSpec values are relative quantities. Their mean is not an expression
    level, and it is strongly anti-correlated with variance (ρ = −0.92). Any
    low-mean filter therefore removes the most variable proteins: fix A would
    remove 491–496 of the top 501.
  - For RNAseq, an expression filter is meaningful (fix C), but it removes
    subtype markers that are on in a minority of lines, such as PGR, KRT6B,
    GABRP and AGR3. The supervised RFE that follows decides which candidates
    carry signal.
- **What fix B changes.**
  - MassSpec: 289–316 of 501 candidates stay the same (≈40 % change).
  - RNAseq: 463–471 of 501 stay the same (≈7 % change). The 30–38
    transcripts the bug removed come back, and as many drop out from just
    below the cut (ranks 502–538, including GRB7).
  - Every saved MassSpec selection contains 2–5 proteins that would no longer
    be candidates. So do 3 of 10 RNAseq selections.
  - All `proteomics`/`transcriptomics` `HVGRFE_*` feature sets and all
    `multimodal` `RFE_*` feature sets must be regenerated, with every model
    trained on them (section 3).

## 1. The fixes compared

| | Fix A: mean step on uncentered data | **Fix B: no mean step (implemented)** | Fix C: per modality |
|---|---|---|---|
| MassSpec | 20 % lowest uncentered means removed, then top 501 by variance | top 501 by variance | same as B |
| RNAseq | 20 % lowest uncentered means removed (cut ≈ −2.49) | top 501 by variance | median > −2.5 over the training lines, then top 501 by variance |
| Top-501 features removed by the filter | MassSpec 491–496; RNAseq 0 | 0 | MassSpec 0; RNAseq 74–78 |
| Consistent with Fig. 4 DE prefilter | no | MassSpec yes (no cutoff there); RNAseq no | yes |
| Platform dependence | none, except a feature that sits exactly on the 20 % cut | none | none |

**Fix B code** (`select_features.py`, `get_hvg`): the three lines of the mean
step are replaced by a comment.

```python
    # No low-mean filter: inputs arrive mean-centered, and proteomics is
    # relative quantification, so feature means carry no expression level.
    if isinstance(top_n, int):
        # Keep top N features with the highest variance
```

- Every other path (`RFE_*`, `all`, curated, MSIGDB, `genes:`, subtype) is
  unchanged.
- The off-by-one is untouched: `top_n=500` keeps 501 features. Fixing it
  would change every set by one more feature, and the Methods can simply say
  "~500".
- The variance ranking cannot flip between platforms: the relative gap
  between the 501st and 502nd variance is at least 2e-5 in every split.
- MassSpec candidates carry somewhat more imputed values: median 7 % missing,
  and 39 % of them more than 10 % missing, against 5 % and 31–36 % now
  (splits BT20 and `all`). Lower-mean proteins are measured in fewer lines
  (ρ = −0.61). The supervised RFE that follows should drop features whose
  signal is mostly imputed.

### Fix C, for the record

Fix C keeps transcripts with median log2 expression > −2.5 over the training
lines, as the Fig. 4 DE analysis does
(`embeddings.compute_marker_parameter_correlations`), and gives MassSpec no
expression filter. The code is in commit 9142ba2.

The transcripts that fail the filter (≈27 % of all, ≈15 % of the 501 most
variable) are expressed above −2.5 in fewer than half of the training lines.
Among the most variable, they are mostly genes that are on in a minority of
lines:

- AGR3 (rank 73–109), CLEC3A, MMP7, TFAP2B, CDH13 and ANKRD30A;
- the stromal/ECM genes COL1A2, COL3A1, COL4A1, CXCL12 and A2M;
- the cancer-testis genes MAGEA2B/A3/A6/A12, CSAG1 and CTAG2;
- the basal markers GABRP (216–257) and KRT6B (222–229);
- PGR (451–524) and, in 4 of 6 splits, SCGB2A1 (383–398).

With ~57 training lines, a gene that is on in 15–25 lines could still carry
signal for the random forest. That trade-off decided against fix C. Fix C
shares 423–427 RNAseq candidates with fix B and 419–422 with the current sets.

### Fix A, for the record

Fix A applies the mean step to the uncentered data. It matches the original
code comment, but fails for MassSpec:

- Per-line medians lie between −0.46 and −0.04, and per-protein means between
  −5.6 and 0.
- 494 of the 501 most variable proteins (split `all`) lie in the lowest-mean
  20 %.
- ERBB2, EGFR, GRB7, KRT5/14/17 and VIM would be excluded in every split.
- Fix A shares only 5–10 MassSpec candidates with fix B.

For RNAseq, fix A equals fix B in all 6 splits: the cut at ≈ −2.49 removes
none of the top 501.

## 2. Candidate sets before and after

All 6 splits, both omics contexts. Each set has 501 features. "Current" is
`get_hvg` on centered inputs as the pipeline runs it, replicated on this
laptop. It reproduces 19 of 20 saved cluster selections (see below).

| Split | Context | Candidates | Top-501 removed by current mean step | **Current ∩ B** | Current ∩ C | B ∩ C | Current ∩ A |
|---|---|---:|---:|---:|---:|---:|---:|
| BT20 | RNAseq | 16,562 | 34 | 467 | 420 | 425 | 467 |
| HCC1500 | RNAseq | 16,562 | 30 | 471 | 422 | 426 | 471 |
| HCC2185 | RNAseq | 16,562 | 38 | 463 | 420 | 423 | 463 |
| MCF7 | RNAseq | 16,562 | 35 | 466 | 419 | 423 | 466 |
| UACC3199 | RNAseq | 16,562 | 35 | 466 | 420 | 425 | 466 |
| all | RNAseq | 16,562 | 35 | 466 | 419 | 427 | 466 |
| BT20 | MassSpec | 6,179 | 212 | 289 | 289 | 501 | 55 |
| HCC1500 | MassSpec | 6,177 | 207 | 294 | 294 | 501 | 59 |
| HCC2185 | MassSpec | 6,182 | 206 | 295 | 295 | 501 | 55 |
| MCF7 | MassSpec | 6,187 | 209 | 292 | 292 | 501 | 55 |
| UACC3199 | MassSpec | 6,181 | 185 | 316 | 316 | 501 | 44 |
| all | MassSpec | 6,161 | 199 | 302 | 302 | 501 | 56 |

### Notable genes, MassSpec (number of splits in the set, out of 6)

| Protein | Variance rank | Current | Fix B | Fix A |
|---|---:|---:|---:|---:|
| VIM | 1 | 1 | 6 | 0 |
| AGR2 | 4 | 1 | 6 | 0 |
| CD44 | 8 | 2 | 6 | 0 |
| KRT5 | 12–17 | 3 | 6 | 0 |
| KRT7 | 13–17 | 4 | 6 | 0 |
| ERBB2 | 25–26 | 1 (`all` only) | 6 | 0 |
| KRT14 | 31–36 | 2 | 6 | 0 |
| GRB7 | 75–92 | 3 | 6 | 0 |
| EGFR | 128–152 | 3 (BT20, MCF7, UACC3199) | 6 | 0 |
| ERBB3 | — | removed by the 30 % missingness filter in every split | | |

- Fix B also gains the PAM50/IHC proteins MLPH (rank ≈265), NUF2 (≈165) and
  PHGDH (≈155) in the splits where they are currently missing.
- It loses ANLN (rank 744–796), CEP55 (619–662) and TYMS (584–694). These
  were in the current set only because higher-variance proteins had been
  removed. Overall, the 199 proteins lost in split `all` have variance ranks
  502–820.

### Notable genes, RNAseq (fix B)

- EGFR (rank 265–316) is in every set in every split. ERBB2 (rank
  ≈1,035–1,076) and ERBB3 (≈1,530–1,570) are in none, because of the variance
  step.
- **Gained** vs current, split `all` (35 genes, ranks 13–472): VIM (13),
  S100A9 (21), MGP (22), MT1E (39), GPX1 (44), CLDN1 (83), ALDH1A3 (175) and
  others. Across splits, VIM, MGP and FOXA1 are now candidates in all 6
  (currently 4, 1 and 4).
- **Lost** vs current, split `all` (35 genes, ranks 502–538): the genes just
  below the cut, which were in the current set only because higher-variance
  genes had been removed. They include GRB7 (rank 508–531; currently in all 6
  splits, now in none; GRB7 protein is gained in all 6), MAGEC2, MAGEA4, LTF
  and DAB2.
- The minority-expressed genes that fix C would remove stay in, as they are
  now: AGR3, GABRP, KRT6B, SCGB2A1 and, in 4 of 6 splits, PGR (currently 5).

### Saved selections

The checked files are the cluster selections in the main checkout's
`features/EGFR_MAPK__logobs/dream_cytof/train/`. They are older sizes
(`proteomics HVGRFE_10`, `transcriptomics HVGRFE_15`, `multimodal RFE_10`), not
the current `BEST_FEATURE_SETS`.

- **Current replication.** 19 of 20 selections lie inside the replicated
  sets. The exception is SP100 in BT20 MassSpec, a floating-point difference,
  as S3 found.
- **Fix B.**
  - All 10 MassSpec selections (5 `HVGRFE_10`, 5 multimodal) contain 2–5
    proteins that are no longer candidates. RDH10 and TAP2 are the most
    frequent.
  - 3 of 10 RNAseq selections do too: CYP4B1 in HCC1500 (`HVGRFE_15` and
    multimodal), and DAB2, PSCA and TMPRSS2 in HCC2185 `HVGRFE_15`.
  - The other selections will most likely change as well, because the random
    forest's ranking depends on the whole candidate pool.

Per-split tables are in `eval/EGFR_MAPK__logobs/dream_cytof/hvg_prefilter/`
(`summary.csv`, `notable_genes.csv`, `receptors.csv`, `saved_selections.csv`,
`membership.csv`) when the script is run with its default output directory.

## 3. What needs regenerating

### Feature files

Paths have the form
`features/<model>/dream_cytof/{train,val}/<context>__<split>__<features>.csv`.

- **Affected:**
  - `proteomics__*__HVGRFE_*`
  - `transcriptomics__*__HVGRFE_*`
  - `multimodal__*__RFE_*`, because `build_context_feature_recipe` turns
    multimodal `RFE_N` into CyTOF `RFE_N` + proteomics `HVGRFE_N` +
    transcriptomics `HVGRFE_N`.
  - `HVG_all` and multimodal `best_RFE_*`, also affected but not used by any
    current configuration.
- **Not affected:**
  - `cytof_init` `RFE_*`, MOSA `RFE_*`, and any `all` feature set (it returns
    before the prefilter).
  - Curated and MSIGDB sets, `cytof_init_plus_*` (`genes:` specs), and
    `subtype_*`.
  - The CyTOF third of a multimodal selection is unaffected by itself, but
    the multimodal file is rewritten as a whole.

### Configurations in `training_configuration.py`

The figure keys below are configuration keys, not paper figure numbers. The
sizes are those before the re-pin. Since section 4, the entries read
proteomics `HVGRFE_10`, transcriptomics `HVGRFE_4` and multimodal `RFE_6`.

| Config key | Affected entries | Feature sets | Downstream |
|---|---|---|---|
| `figure1a` | proteomics `HVGRFE_8`, transcriptomics `HVGRFE_12`, multimodal `RFE_8`; 2 pathways (`logobs`, `logobs_tegfr_aggavg`) × 6 splits | 36 | training, `evaluate_training`, `evaluate_regressors` (elastic net / lasso / OLS on the same features), `evaluate_all_figure1a.csv` |
| `figure1b` | the same three contexts; `logobs`; 5 CV splits; ML hyperparameter scans | 15 | every scan point of those three contexts |
| `figure1c` (feature-count scan, Table S4) | proteomics and transcriptomics `HVGRFE_N` (17 sizes each), multimodal `RFE_N` (N ≤ 32, 10 sizes); 5 CV splits | 220 | 10 starts per fold (S2); `evaluate_all_figure1c.csv` |
| `figure3` | multimodal `RFE_8`; 7 pathways; split `all` | 7 | training and evaluation |
| `figure3b` | multimodal `RFE_8`; 2 pathways; split `all`; `n_hidden` scan | 2 | every `n_hidden` point |
| `default` | proteomics, transcriptomics, multimodal × 8 pathways | — | not built by the `paper` target (S6), so it can be skipped |
| `figure2`, `figure4`, `figure5`, `figure5b`, `figure6` | none (CyTOF or `cytof_init_plus_*` only) | 0 | — |

Feature files do not depend on the figure key, so the configurations share
sets: `figure1b`, `figure3b` and 15 of the `figure1c` sets are also in
`figure1a`. That makes **246 distinct selection jobs** (36 + 205 + 5).

**Snakemake caution (for S6):** `select_features.py` is an input of every
`select_features` job. Editing it makes even the unaffected CyTOF and MOSA
jobs look outdated under the default mtime trigger, along with everything
trained on them. Before launching, check with a dry run (`-n`) that only the
affected contexts are scheduled. For example, delete the affected feature
files and leave `mtime` out of `--rerun-triggers`.

### Notebooks and panels

| Notebook | Panel | Why |
|---|---|---|
| `figure_1.ipynb` | none as currently written | reads `figure1a`, but its panels filter to `selected_context = "cytof_init"` (also on S1's branch) |
| `figure_2.ipynb` | A "Input Modalities" | MassSpec, RNAseq and Multimodal boxes (`figure1a`) |
| | B "RTK Integrations" | multimodal rows of `figure3` |
| | D "Latent Embeddings", left | `logobs` multimodal |
| | E "Subtype Classification" | uses embeddings of all contexts in `figure1a` |
| | G `n_hidden` scan | `figure3b`: `render_nhidden_trainval` does not filter by context, so it averages CyTOF and multimodal runs |
| | composite stats cell (tEGFR vs DMM on multimodal) | |
| | Panels C and F, MOSA diagnostic | not affected |
| `figure_3.ipynb` | E "Annotation classifier summary" | `ANNOTATION_CONTEXT = "multimodal"` (`figure1a`); other panels use `cytof_init` |
| `figure_5.ipynb` | `plot_figure3b_nhidden` panel | `figure3b`, pooled over contexts as above; `figure4`/`figure6` parts not affected |
| `figure_4.ipynb` | none | `figure5`, CyTOF only; omics are used only as marker data |
| `figure_S_feature_count_scan.ipynb` (S2 branch) | MassSpec, RNAseq and multimodal curves (Table S4) | `figure1c`; the CyTOF and MOSA curves are not affected |
| `figure_S_receptor_evidence.ipynb` (S3 branch) | the "HVG candidates" RF importances and `first_round_status` | `receptor_evidence.get_hvg`/`rf_inputs` copy the buggy prefilter and need fix B; their "excluded by mean step" category goes away |

### Text

- Suggested Methods wording for fix B:

  > Before recursive feature elimination, omics features were prefiltered on
  > the training cell lines of each split, retaining the ~500 most variable
  > transcripts and the ~500 most variable proteins.

- If a reviewer asks why there is no expression filter: proteomics values
  are relative, and for transcripts an expression filter would exclude
  subtype markers that are on in a minority of lines. The DE analysis of
  Fig. 4 does use one (median log2 expression > −2.5), so the two filters
  differ by design.
- If the pipeline is **not** rerun, the Methods must describe what actually
  ran. That is hard to state honestly: "a variance prefilter applied to a
  pseudo-random 80 % of features, biased against high-variance proteins".

## 4. Reruns with fix B (NEMO, November 2026)

### `figure1c` feature-count scan (done, 4 Nov)

- **What was rerun:** the 220 affected selections and their trainings
  (5 CV folds × 10 starts).
  - CyTOF, MOSA and proteomics `all` results are byte-identical to April.
  - The transcriptomics `all` models were retrained because their model files
    were missing, so those numbers differ only by training noise.
- **Failures:** 111 trainings first failed on Singularity "no loop devices
  available", an infrastructure problem, and were rerun. No failure was in
  the code.
- **Files:** the new
  `eval/EGFR_MAPK__logobs/dream_cytof/evaluate_all_figure1c.csv` is dated
  3 Nov. The April files are in `pre_hvg_fix_2026-11-02/` next to it.
- **Results:** S2's measure (validation RMSE, median over 10 starts, mean over
  folds), April → now:

| Modality | Lowest RMSE at | Old selected size | Largest set vs selected |
|---|---|---|---|
| RNAseq | 12 (0.657) → **4 (0.609)** | 12: 0.657 → 0.774 | +18 % → −2 % (all 16,562) |
| MassSpec | 8 (0.669) → **10 (0.741)** | 8: 0.669 → 0.788 | +20 % → +2 % (all ~6,181) |
| Multimodal | 24 inputs (0.533) → **18 (0.566)** | 24: 0.533 → 0.588 | +11 % → +5 % (96 inputs) |
| CyTOF, MOSA | unchanged | unchanged | unchanged |

- **The HCC1500 fold drives much of the change at the old sizes.** With
  8 proteins, all 10 starts went from 0.46–0.64 to 1.04–1.32. The new feature
  selection for that fold generalises badly, so it is not noise from the
  random starts.
- **Wording:** S2's note (PR 16) has the updated scan figure and wording.

### Re-pinned `BEST_FEATURE_SETS` (677d833)

- **New sizes:** transcriptomics `HVGRFE_4`, proteomics `HVGRFE_10`,
  multimodal `RFE_6` (6 per modality, 18 inputs). These are the scan's new
  minima.
- **Unchanged:** CyTOF `RFE_8` and MOSA `RFE_6`. Fix B does not affect them.
  CyTOF's scan minimum (16) is within noise of 8.

### `figure1b` ML hyperparameter scan (running)

- **What runs:** 3,000 new trainings, for the three re-pinned feature sets:
  20 configurations × 5 folds × 10 starts each.
  - The central configuration is shared with `figure1c` and reused.
  - CyTOF and MOSA reuse their April models.
- **Launch and resume:** `sbatch estimationCluster_figure1b_revision.sh` on
  NEMO.
- **April results:** in `pre_hvg_fix_2026-11-02/`.
- **Next step:** the central values in `LATENT_DIMS_BY_CONTEXT` and
  `NETWORK_DEPTH_BY_CONTEXT` were tuned on the old sizes. Revisit them when
  the scan finishes.

### Still to do

- **Main-text configurations** `figure1a`, `figure3` and `figure3b` use the
  re-pinned sizes but have not been rerun. Then their notebook panels
  (section 3) need updating.
- **NEMO checkout state:** `select_features.py` is backdated (mtime
  2025-12-19), so Snakemake does not rerun the unaffected selections. A
  configuration that still has old affected feature files needs those files
  moved aside first, otherwise they are reused.
