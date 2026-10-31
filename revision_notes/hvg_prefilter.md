# HVG prefilter in `select_features.py`: fix and impact

Branch `claude/objective-rubin-5b7968`. Analysis script:
`revision_notes/hvg_prefilter_impact.py`. It imports `rf_inputs`, `get_hvg` and
`hvg_diagnostics` from `figures_paper/receptor_evidence.py` on branch
`claude/revision-receptor-evidence-b04f76`, so it runs once that branch is
merged (or with that file on `PYTHONPATH`). It takes a few minutes on the
laptop. Everything below is computed on this laptop, not on the cluster.

`select_features.py` is **unchanged**. The fix waits for a decision (section 1).

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
- **Recommendation: drop the mean step (fix B).**
  - For RNAseq, fixes A and B give the same set in all 6 splits.
  - For MassSpec, fix A would remove 491–496 of the 501 most variable
    proteins, ERBB2 and EGFR included.
- **Impact of fix B on the candidate sets.**
  - MassSpec: only 289–316 of 501 candidates stay the same (≈40 % change).
  - RNAseq: 463–471 of 501 stay the same (≈7 % change).
  - Every saved MassSpec selection contains 2–5 proteins that would no longer
    be candidates. So do 3 of 10 RNAseq selections.
  - All `proteomics`/`transcriptomics` `HVGRFE_*` feature sets and all
    `multimodal` `RFE_*` feature sets must be regenerated, with every model
    trained on them (section 3).

## 1. The two fixes and the trade-off

**Fix A: compute the mean step on the uncentered imputed inputs.** This is
what the code comment intends. `prepare_inputs_for_context` already has the
training means (`mean_train`); pass them through:

```diff
-from typing import Union
+from typing import Optional, Union
@@ def get_hvg(
-    input_df: pd.DataFrame, top_n: Union[int, float] = 500
+    input_df: pd.DataFrame,
+    top_n: Union[int, float] = 500,
+    means: Optional[pd.Series] = None,
 ) -> pd.DataFrame:
-    # remove 20% of features with the lowest mean:
-    means = np.mean(input_df, axis=0)
+    # remove 20% of features with the lowest mean; pass the uncentered
+    # means if input_df is mean-centered
+    if means is None:
+        means = np.mean(input_df, axis=0)
+    means = means.reindex(input_df.columns)
@@ def get_selected_features(
     cv=None,
+    feature_means: Optional[pd.Series] = None,
 ):
@@
-            input_data = get_hvg(input_data)
+            input_data = get_hvg(input_data, means=feature_means)
@@ def prepare_inputs_for_context(
+    mean_train = None
     if "subtype" not in subconf.context:
@@
-            input_train = get_hvg(input_train)
+            input_train = get_hvg(input_train, means=mean_train)
@@
                 cv=None,
+                feature_means=mean_train,
             )
```

**Fix B: drop the mean step.** The prefilter keeps the 501 most variable
features:

```diff
 def get_hvg(
     input_df: pd.DataFrame, top_n: Union[int, float] = 500
 ) -> pd.DataFrame:
-    # remove 20% of features with the lowest mean:
-    means = np.mean(input_df, axis=0)
-    threshold = np.percentile(means, 20)
-    input_df = input_df.loc[:, means >= threshold]
     if isinstance(top_n, int):
```

Both fixes leave the off-by-one alone: `top_n=500` keeps 501 features. Fixing
it would change every set by one more feature, and the Methods can simply say
"~500".

| | Fix A: mean on uncentered data | Fix B: no mean step |
|---|---|---|
| Code change | ≈15 lines across 3 functions | delete 4 lines |
| Matches the code comment / a "low-expression filter" in the Methods | yes | no; Methods must say "the ~500 most variable features" (as S2 already proposed) |
| RNAseq candidates | identical to B in all 6 splits | identical to A |
| MassSpec candidates | removes 491–496 of the 501 most variable proteins; shares only 5–10 candidates with B; ERBB2, EGFR, GRB7, KRT5/14/17 and VIM excluded in every split | the 501 most variable proteins |
| Missing values among MassSpec candidates (median; share > 10 % missing; splits BT20 and `all`) | 0 %; 10 % | 7 %; 39 % (current: 5 %; 31–36 %; all candidates: 0 %; 17 %) |
| Platform dependence | none, except a feature that sits exactly on the 20 % cut (split `all`, MassSpec) | none: the 501st and 502nd variances differ by ≥ 2e-5 (relative) |

Why fix A fails for MassSpec:

- The two omics tables are on different scales.
  - RNAseq is log2 expression with a floor at −3.32 (= log2 0.1). There, a low
    mean really does mean "not expressed".
  - The 20 % cut sits at −2.49, close to that floor. None of the 501 most
    variable genes falls below it, so for RNAseq the mean step is redundant.
- MassSpec values are relative: per-line medians lie between −0.46 and −0.04,
  and per-protein means between −5.6 and 0.
  - On this scale, mean and variance are almost perfectly anti-correlated
    (Spearman ρ = −0.92).
  - 494 of the 501 most variable proteins (split `all`) lie in the
    lowest-mean 20 %.
  - So fix A would turn the HVG filter into a "least variable of the
    variable" filter.

The cost of fix B: MassSpec candidates carry somewhat more imputed values.
Lower-mean proteins are measured in fewer lines (ρ = −0.61 between mean and
missing fraction). The RFE that follows is supervised, so features whose signal
is mostly imputed should drop out. If that is a concern, a missingness cap on
the candidates would address it directly. A mean cut does not address it.

**Recommended: fix B.**

## 2. Candidate sets before and after

All 6 splits, both omics contexts. Each set has 501 features. "Current" is
`get_hvg` on centered inputs as the pipeline runs it, replicated on this
laptop. It reproduces 19 of 20 saved cluster selections (see below).

| Split | Context | Candidates | Top-501 removed by current mean step | Current ∩ B | Current ∩ A | A ∩ B |
|---|---|---:|---:|---:|---:|---:|
| BT20 | RNAseq | 16,562 | 34 | 467 | 467 | 501 |
| HCC1500 | RNAseq | 16,562 | 30 | 471 | 471 | 501 |
| HCC2185 | RNAseq | 16,562 | 38 | 463 | 463 | 501 |
| MCF7 | RNAseq | 16,562 | 35 | 466 | 466 | 501 |
| UACC3199 | RNAseq | 16,562 | 35 | 466 | 466 | 501 |
| all | RNAseq | 16,562 | 35 | 466 | 466 | 501 |
| BT20 | MassSpec | 6,179 | 212 | 289 | 55 | 10 |
| HCC1500 | MassSpec | 6,177 | 207 | 294 | 59 | 10 |
| HCC2185 | MassSpec | 6,182 | 206 | 295 | 55 | 5 |
| MCF7 | MassSpec | 6,187 | 209 | 292 | 55 | 10 |
| UACC3199 | MassSpec | 6,181 | 185 | 316 | 44 | 8 |
| all | MassSpec | 6,161 | 199 | 302 | 56 | 7 |

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

- Fix B also gains PAM50/IHC proteins MLPH (rank ≈265), NUF2 (≈165) and
  PHGDH (≈155) in the splits where they are currently missing.
- Fix B loses ANLN (rank 744–796), CEP55 (619–662) and TYMS (584–694). These
  were in the current set only because higher-variance proteins had been
  removed. Overall, the 199 proteins lost in split `all` have variance ranks
  502–820.
- Fix A keeps only MKI67 (rank ≈1,400) and NDC80 (≈1,040) from the PAM50/IHC
  lists.

### Notable genes, RNAseq (fix A = fix B)

- EGFR (rank 265–316) is in every set in every split. ERBB2 (rank
  ≈1,035–1,076) and ERBB3 (≈1,530–1,570) are in none, because of the variance
  step.
- **Gained** (split `all`): VIM (13), S100A9 (21), MGP (22), MT1E (39), GPX1
  (44), CLDN1 (83), C3 (114), TRIM29 (156), ALDH1A3 (175), COL12A1 (371) and
  others: 35 genes, ranks 13–472. FOXA1 (rank 156–171) is gained in HCC2185
  and UACC3199.
- **Lost** (split `all`): 35 genes of rank 502–538, among them GRB7 (531),
  DAB2, ETV5 and CD70. PGR (rank 524) is lost in HCC1500.

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
  - 3 of 10 RNAseq selections do too: HCC1500 has CYP4B1 in both files, and
    HCC2185 `HVGRFE_15` has DAB2, PSCA and TMPRSS2.
  - The other selections will most likely change as well, because the random
    forest's ranking depends on the whole candidate pool.
- **Fix A.** 9–10 of 10 selected proteins drop out of every MassSpec
  selection.

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
| `figure_S_receptor_evidence.ipynb` (S3 branch) | the "HVG candidates" RF importances and `first_round_status` | `receptor_evidence.get_hvg`/`rf_inputs` copy the buggy prefilter; they would need the same fix, and their "excluded by mean step" category disappears under fix B |

### Text

- With fix B, the Methods should describe the prefilter as "the ~500 most
  variable features (training lines only)", with no mean-expression step.
- With fix A, the Methods can keep a low-mean filter, but the MassSpec
  candidates would then exclude ERBB2, EGFR and most keratins.
- If the pipeline is **not** rerun, the Methods must describe what actually
  ran. That is hard to state honestly: "a variance prefilter applied to a
  pseudo-random 80 % of features, biased against high-variance proteins".
