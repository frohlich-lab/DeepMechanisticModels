# HVG prefilter in `select_features.py`: fix and impact

Branch `claude/objective-rubin-5b7968`.

- **Analysis script:** `revision_notes/hvg_prefilter_impact.py`.
  - It imports `rf_inputs`, `get_hvg` and `hvg_diagnostics` from
    `figures_paper/receptor_evidence.py` on branch
    `claude/revision-receptor-evidence-b04f76`. So it runs once that branch is
    merged, or with that file on `PYTHONPATH`.
  - It takes a few minutes on the laptop. Everything below was computed on
    this laptop, not on the cluster.
- **Implemented: fix C, a per-modality prefilter (section 1).** FF chose it
  after comparing it with the DE analysis in `figure_4.ipynb`.
  - Proteomics: no expression filter. The prefilter keeps the 501 most
    variable proteins.
  - Transcriptomics: keep transcripts with median log2 expression > −2.5 over
    the training lines, then the 501 most variable. This is the same filter
    the Fig. 4 DE analysis uses (`embeddings.compute_marker_parameter_correlations`).
- **Checked end to end.** Running the patched `select_features.py` with
  `--features=HVG_all` reproduces the script's fix-C sets exactly, for BT20
  and `all`, in both contexts.
- **Not yet done.** No feature set has been regenerated (section 3).

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
    genes are removed. Lost genes include VIM (rank 10–13, kept in 4/6) and
    FOXA1 (rank 156–171, missing in 2/6).
- **Why the fix differs by modality.**
  - MassSpec values are relative quantities. Their mean is not an expression
    level, and it is strongly anti-correlated with variance (ρ = −0.92). Any
    low-mean filter therefore removes the most variable proteins: fix A would
    remove 491–496 of the top 501.
  - RNAseq is log2 expression with a floor at −3.32. There, "not expressed"
    is meaningful, and the Fig. 4 median cut defines it.
- **What fix C changes.**
  - MassSpec: 289–316 of 501 candidates stay the same (≈40 % change).
  - RNAseq: 419–422 of 501 stay the same (≈16 % change).
  - Every saved MassSpec selection contains 2–5 proteins that would no longer
    be candidates. So does 1 of 10 RNAseq selections.
  - All `proteomics`/`transcriptomics` `HVGRFE_*` feature sets and all
    `multimodal` `RFE_*` feature sets must be regenerated, with every model
    trained on them (section 3).

## 1. The fixes compared

| | Fix A: mean step on uncentered data | Fix B: no mean step | **Fix C: per modality (implemented)** |
|---|---|---|---|
| MassSpec | 20 % lowest uncentered means removed, then top 501 by variance | top 501 by variance | same as B |
| RNAseq | 20 % lowest uncentered means removed (cut ≈ −2.49) | top 501 by variance | median > −2.5 over the training lines, then top 501 by variance |
| Top-501 features removed by the filter | MassSpec 491–496; RNAseq 0 | 0 | MassSpec 0; RNAseq 74–78 |
| Consistent with Fig. 4 DE prefilter | no | no for RNAseq | yes |
| Platform dependence | none, except a feature that sits exactly on the 20 % cut | none | none |

**Fix C code** (`select_features.py`):

```python
# Expression filter of the HVG prefilter, as in the DE analysis of Fig. 4:
# keep transcripts with median log2 expression > -2.5 over the training cell
# lines. Proteomics is relative quantification and gets no expression filter.
HVG_MIN_MEDIAN = {"transcriptomics": -2.5}
```

- `get_hvg(input_df, top_n, expressed=None)`:
  - The low-mean step is gone.
  - When given, `expressed` (a boolean Series) restricts the input before the
    variance step.
- `prepare_inputs_for_context`:
  - computes `expressed = input_train.median() > HVG_MIN_MEDIAN[context]` on
    the imputed, **uncentered** training inputs;
  - passes it to `get_hvg` (`HVG_all`) and to `get_selected_features`
    (`HVGRFE_*`).
- Every other path (`RFE_*`, `all`, curated, MSIGDB, `genes:`, subtype) is
  unchanged.
- The off-by-one is untouched: `top_n=500` keeps 501 features. Fixing it
  would change every set by one more feature, and the Methods can simply say
  "~500".

### What the RNAseq median filter removes

The transcripts that fail the filter (≈27 % of all, ≈15 % of the 501 most
variable) are expressed above −2.5 in fewer than half of the training lines.

- Among the most variable, they are mostly genes that are on in a minority of
  lines:
  - AGR3 (rank ≈78), CLEC3A, MMP7, TFAP2B, CDH13 and ANKRD30A;
  - the stromal/ECM genes COL1A2, COL3A1, COL4A1, CXCL12 and A2M;
  - the cancer-testis genes MAGEA2B/A3/A6/A12, CSAG1 and CTAG2;
  - the basal markers GABRP and KRT6B.
- PGR (rank ≈450–525) and SCGB2A1 (≈385) are also removed.
- This is the trade-off of fix C. Minority-expressed subtype markers can no
  longer be candidates. With ~57 training lines, a gene that is on in 15–25
  lines could still carry signal for the random forest.

The MassSpec cost is shared with fix B. Candidates carry somewhat more imputed
values: median 7 % missing, and 39 % of them more than 10 % missing, against
5 % and 31–36 % now (splits BT20 and `all`). Lower-mean proteins are measured
in fewer lines (ρ = −0.61). The supervised RFE that follows should drop
features whose signal is mostly imputed.

### Fix A, for the record

Fix A applies the mean step to the uncentered data. It matches the original
code comment, but fails for MassSpec:

- Per-line medians lie between −0.46 and −0.04, and per-protein means between
  −5.6 and 0.
- 494 of the 501 most variable proteins (split `all`) lie in the lowest-mean
  20 %.
- ERBB2, EGFR, GRB7, KRT5/14/17 and VIM would be excluded in every split.
- Fix A shares only 5–10 MassSpec candidates with fix B/C.

For RNAseq, fix A equals fix B in all 6 splits: the cut at ≈ −2.49 removes
none of the top 501.

## 2. Candidate sets before and after

All 6 splits, both omics contexts. Each set has 501 features. "Current" is
`get_hvg` on centered inputs as the pipeline runs it, replicated on this
laptop. It reproduces 19 of 20 saved cluster selections (see below).

| Split | Context | Candidates | Top-501 removed by current mean step | Current ∩ C | B ∩ C | Current ∩ B | Current ∩ A |
|---|---|---:|---:|---:|---:|---:|---:|
| BT20 | RNAseq | 16,562 (12,157 expressed) | 34 | 420 | 425 | 467 | 467 |
| HCC1500 | RNAseq | 16,562 (12,146) | 30 | 422 | 426 | 471 | 471 |
| HCC2185 | RNAseq | 16,562 (12,150) | 38 | 420 | 423 | 463 | 463 |
| MCF7 | RNAseq | 16,562 (12,143) | 35 | 419 | 423 | 466 | 466 |
| UACC3199 | RNAseq | 16,562 (12,140) | 35 | 420 | 425 | 466 | 466 |
| all | RNAseq | 16,562 (12,147) | 35 | 419 | 427 | 466 | 466 |
| BT20 | MassSpec | 6,179 | 212 | 289 | 501 | 289 | 55 |
| HCC1500 | MassSpec | 6,177 | 207 | 294 | 501 | 294 | 59 |
| HCC2185 | MassSpec | 6,182 | 206 | 295 | 501 | 295 | 55 |
| MCF7 | MassSpec | 6,187 | 209 | 292 | 501 | 292 | 55 |
| UACC3199 | MassSpec | 6,181 | 185 | 316 | 501 | 316 | 44 |
| all | MassSpec | 6,161 | 199 | 302 | 501 | 302 | 56 |

### Notable genes, MassSpec (number of splits in the set, out of 6; fix C = fix B)

| Protein | Variance rank | Current | Fix C | Fix A |
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

- Fix C also gains the PAM50/IHC proteins MLPH (rank ≈265), NUF2 (≈165) and
  PHGDH (≈155) in the splits where they are currently missing.
- It loses ANLN (rank 744–796), CEP55 (619–662) and TYMS (584–694). These
  were in the current set only because higher-variance proteins had been
  removed. Overall, the 199 proteins lost in split `all` have variance ranks
  502–820.

### Notable genes, RNAseq (fix C)

- EGFR (rank 265–316) is in every set in every split. ERBB2 (rank
  ≈1,035–1,076) and ERBB3 (≈1,530–1,570) are in none, because of the variance
  step.
- **Gained** vs current, split `all` (82 genes, ranks 13–593):
  - VIM (13), S100A9 (21), MGP (22), MT1E (39), GPX1 (44), CLDN1 (83),
    ALDH1A3 (175) and others;
  - FOXA1 (rank 156–171), now in all 6 splits (currently 4);
  - genes just below rank 501. GRB7 (rank ≈510–531) stays in, as it is now.
- **Lost** vs current, split `all` (82 genes): the minority-expressed genes
  listed in section 1, including AGR3, MMP7, TFAP2B, COL1A2, CXCL12, GABRP,
  KRT6B and PGR.

### Saved selections

The checked files are the cluster selections in the main checkout's
`features/EGFR_MAPK__logobs/dream_cytof/train/`. They are older sizes
(`proteomics HVGRFE_10`, `transcriptomics HVGRFE_15`, `multimodal RFE_10`), not
the current `BEST_FEATURE_SETS`.

- **Current replication.** 19 of 20 selections lie inside the replicated
  sets. The exception is SP100 in BT20 MassSpec, a floating-point difference,
  as S3 found.
- **Fix C.**
  - All 10 MassSpec selections (5 `HVGRFE_10`, 5 multimodal) contain 2–5
    proteins that are no longer candidates. RDH10 and TAP2 are the most
    frequent.
  - 1 of 10 RNAseq selections does too: CST1 in HCC1500 `HVGRFE_15`. For
    comparison, 3 of 10 would under fix B.
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

The figure keys below are configuration keys, not paper figure numbers.

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
| `figure_S_receptor_evidence.ipynb` (S3 branch) | the "HVG candidates" RF importances and `first_round_status` | `receptor_evidence.get_hvg`/`rf_inputs` copy the buggy prefilter and need fix C; their "excluded by mean step" category becomes "not expressed" (RNAseq only) |

### Text

- Suggested Methods wording for fix C:

  > Before recursive feature elimination, omics features were prefiltered on
  > the training cell lines of each split. Transcripts were restricted to
  > those with median log2 expression above −2.5 (the expression filter used
  > for differential expression), and the ~500 most variable transcripts and
  > the ~500 most variable proteins were retained.

- Proteomics gets no expression filter because its values are relative.
  This could be stated briefly if a reviewer asks.
- If the pipeline is **not** rerun, the Methods must describe what actually
  ran. That is hard to state honestly: "a variance prefilter applied to a
  pseudo-random 80 % of features, biased against high-variance proteins".
