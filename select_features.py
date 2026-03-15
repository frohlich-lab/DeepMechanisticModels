import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Union

import fire
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cell_line_annotations import onehot_intrinsic, onehot_lb
from common import FEATURES_OUTFILE, Wildcards, training_samples, val_samples
from dmm.config_options import Conf
from dmm.feature_selection import (
    build_preprocessor,
    load_data,
    preprocess_mosa_latent,
)
from training_configuration import DROP_HER2_FROM_FEATURES
from util import load_petab_base_files

# Path for caching MSigDB GMT data
MSIGDB_CACHE_PATH = (
    Path(__file__).parent / "data" / "msigdb_c2cp_2025.1.Hs.json"
)


def load_msigdb_gmt() -> dict[str, list[str]]:
    """Load MSigDB GMT data from local cache, fetching from web if not cached."""
    if MSIGDB_CACHE_PATH.exists():
        with open(MSIGDB_CACHE_PATH, "r") as f:
            return json.load(f)

    # Fetch from web and cache locally
    from gseapy import Msigdb

    msig = Msigdb()
    gmt = msig.get_gmt(category="c2.cp", dbver="2025.1.Hs")

    # Ensure cache directory exists
    MSIGDB_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MSIGDB_CACHE_PATH, "w") as f:
        json.dump(gmt, f)

    return gmt


def get_feature_importances(model, X, y, method="auto"):
    """
    Get feature importances for RandomForest or ElasticNet/LogisticRegression models.

    Parameters:
    - model: fitted sklearn model
    - method: 'auto', 'coef', 'tree', 'permutation'
    - X: input features (optional, needed for permutation importance)
    - y: target variable (optional, needed for permutation importance)

    Returns:
    - feature_importances: np.ndarray of shape (n_features,)
    """
    if method == "tree":
        return model.named_steps["regressor"].feature_importances_

    elif method == "permute":
        if X is None or y is None:
            raise ValueError(
                "X and y must be provided for permutation importance."
            )
        result = permutation_importance(
            model, X, y, n_repeats=10, random_state=42
        )
        return result.importances_mean

    else:
        raise ValueError(f"Unknown method: {method}")


def get_hvg(
    input_df: pd.DataFrame, top_n: Union[int, float] = 2500
) -> pd.DataFrame:
    # remove 20% of features with the lowest mean:
    means = np.mean(input_df, axis=0)
    threshold = np.percentile(means, 20)
    input_df = input_df.loc[:, means >= threshold]
    if isinstance(top_n, int):
        # Keep top N features with the highest variance
        var_threshold = sorted(np.nanvar(input_df, axis=0), reverse=True)[
            top_n
        ]
        input_df = input_df.loc[
            :, np.nanvar(input_df, axis=0) >= var_threshold
        ]
    elif isinstance(top_n, float):
        perc = 100 * top_n
        # Keep top 50% features with the highest variance (default)
        var_threshold = np.percentile(np.nanvar(input_df, axis=0), perc)
        input_df = input_df.loc[
            :, np.nanvar(input_df, axis=0) >= var_threshold
        ]
    return input_df


def get_selected_features(
    input_data,
    output_data,
    context: str,
    features: str,
    features_all: list,
    cv=None,
):
    if features == "all":
        return features_all

    curated_features = {
        # commonly used IHC markers in breast cancer
        "IHC": [
            # https://doi.org/10.1371/journal.pmed.1000279
            "ERBB2",  # HER2/neu
            "EGFR",  # epidermal growth factor receptor
            "KRT5",  # keratin 5;
            "KRT6A",  # keratin 6A;
            "KRT6B",  # keratin 6B; filtered in prot
            "PGR",  # progesterone receptor; filtered in prot
            "ESR1",  # estrogen receptor; filtered in prot
            "MKI67",  # Ki-67
            "TP53",  # p53;
            # https://doi.org/10.1038/s41379-020-00697-3
            "GATA3",
            "SOX10",
            "SCGB2A2",  # mammaglobin 1 (MMGB)
            "SCGB2A1",  # mammaglobin 2, missing in proteomics
            "KRT7",  # keratin 7
            "PIP",  # GCDFP-15, gross cystic disease fluid protein
            "CDX2",  # missing in proteomics
            "KRT20",  # missing in proteomics
            "NKX2-1",  # thyroid transcription factor 1 (TTF-1), missing in transcriptomics
        ],
        # Breast cancer stem cell markers
        # https://doi.org/10.3390/cancers12123765
        "CSC": [
            "ALDH1A1",  # filtered in transcriptomics
            "ALDH1A2",  # filtered in proteomics, transcriptomics
            "ALDH1A3",  # filtered in transcriptomics
            "ALDH1B1",  # filtered in transcriptomics
            "ALDH1L1",  # filtered in proteomics, transcriptomics
            "ALDH1L2",  # filtered in transcriptomics
            "ABCG2",  # filtered in proteomics
            "LGR5",  # missing in proteomics, filtered in transcriptomics
            # "SSEA3", glycoprotein, not a gene, not in proteomics/transcriptomics
            "CD70",  # filtered in proteomics
            "PROCR",  # filtered in proteomics, transcriptomics
            "CD44",
            "CD24",  # missing in proteomics/transcriptomics
            "CD133",  # missing in proteomics/transcriptomics
            "EPCAM",
            "ITGA6",  # CD49f
            "THY1",  # CD90  # filtered in proteomics
            "ITGB3",  # CD61  # filtered in proteomics
            "MUC1",
            "FGD2",  # GD2, missing in proteomics
            "NECTIN4",  # missing in transcriptomics, filtered in proteomics
        ],
        # MAPK Pathway Activity Score
        # https://doi.org/10.1038/s41698-018-0051-4
        "MPAS": [
            "SPRY2",  # missing in proteomics
            "SPRY4",
            "ETV4",  # missing in proteomics
            "ETV5",  # missing in proteomics
            "DUSP4",
            "DUSP6",  # missing in proteomics
            "CCND1",
            "EPHA2",
            "EPHA4",
        ],
        # compensatory resistance signature
        # https://doi.org/10.1158/0008-5472.CAN-09-1577 Fig 4
        "CompRes": [
            "IL6",  # missing in proteomics
            "CD274",
            "G0S2",  # missing in proteomics
            "STAC,"  # missing in proteomics
            "COL5A1",
            "COL12A1",
            "SERPINE1",
            "CRIM1",
            "LOX",
            "GPR176",  # missing in proteomics
            "FZD2",
            "BASP1",
            "CLU",
        ],
        # MEK functional activation
        # https://doi.org/10.1158/0008-5472.CAN-09-1577 Fig 4
        "MEKFA": [
            "ZNF106",  # ZFP106, missing in proteomics
            "PROS1",  # missing in proteomics
            "LZTS1",  # missing in proteomics
            "KANK1",  # ANKRD15
            "TRIB2",  # missing in proteomics
            "DUSP4",
            "ETV4",  # missing in proteomics
            "ETV6",  # missing in proteomics
            "DUSP6",  # missing in proteomics
            "PHLDA1",
            "SPRY2",
            "ELF1",
            "LGALS3",
            "FXYD5",  # missing in proteomics
            "S100A6",
            "SERPINB1",
            "SLCO4A1",  # missing in proteomics
            "MAP2K3",
        ],
        # PAM50 gene signature
        # https://doi.org/10.1200/JCO.2008.18.1370 Fig A2
        # transcriptomics 50, proteomics 31
        "PAM50": [
            # basal-like, missing: KNTC2 (alias NDC80)
            "FOXC1",
            "MIA",
            "NDC80",
            "CEP55",
            "ANLN",
            "MELK",
            "GPR160",
            "TMEM45B",
            "ESR1",
            "FOXA1",
            # her2
            "ERBB2",
            "GRB7",
            "FGFR4",
            "BLVRA",
            "BAG1",
            "CDC20",
            "CCNE1",
            "ACTR3B",
            "MYC",
            "SFRP1",
            # normal-like
            "KRT14",
            "KRT17",
            "KRT5",
            "MLPH",
            "CCNB1",
            "CDC6",
            "TYMS",
            "UBE2T",
            "RRM2",
            "MMP11",
            # luminal B, missing: ORC6L (alias ORC6), PGR
            # PGR filtered out in preprocessing
            "CXXC5",
            "ORC6",
            "MDM2",
            "KIF2C",
            "PGR",
            "MKI67",
            "BCL2",
            "EGFR",
            "PHGDH",
            "CDH3",
            # luminal A, missing: CDCA1 (alias NUF2)
            "NAT1",
            "SLC39A6",
            "MAPT",
            "UBE2C",
            "PTTG1",
            "EXO1",
            "CENPF",
            "NUF2",
            "MYBL2",
            "BIRC5",
        ],
    }

    if features in curated_features or features.startswith("MSIGDB_"):
        if features in curated_features:
            gene_list = curated_features[features]
        elif features.startswith("MSIGDB_"):
            gene_set = "_".join(features.split("_")[1:])
            gene_sets = {
                "KEGG_ERBB": "KEGG_ERBB_SIGNALING_PATHWAY",
                "KEGG_MAPK": "KEGG_MAPK_SIGNALING_PATHWAY",
                "KEGG_EGFR": "KEGG_MEDICUS_REFERENCE_EGF_EGFR_RAS_ERK_SIGNALING_PATHWAY",
                "KEGG_RTK": "KEGG_MEDICUS_REFERENCE_RTK_PLCG_ITPR_SIGNALING_PATHWAY",
                "KEGG_ERK": "KEGG_MEDICUS_REFERENCE_GF_RTK_RAS_ERK_SIGNALING_PATHWAY",
                "BIOCARTA_MAPK": "BIOCARTA_MAPK_PATHWAY",
                "BIOCARTA_EGF": "BIOCARTA_EGF_PATHWAY",
                "BIOCARTA_ERK": "BIOCARTA_ERK_PATHWAY",
                "BIOCARTA_RAS": "BIOCARTA_RAS_PATHWAY",
                "BIOCARTA_P38": "BIOCARTA_P38MAPK_PATHWAY",
                "PID_ERBB_DOWNSTREAM": "PID_ERBB1_DOWNSTREAM_PATHWAY",
                "PID_ERBB_INTERN": "PID_ERBB1_INTERNALIZATION_PATHWAY",
                "PID_ERBB_PROXIMAL": "PID_ERBB1_RECEPTOR_PROXIMAL_PATHWAY",
                "PID_ERBB": "PID_ERBB2_ERBB3_PATHWAY",
                "PID_RAS": "PID_RAS_PATHWAY",
                "PID_MAPK": "PID_MAPK_TRK_PATHWAY",
                "PID_P38_DOWNSTREAM": "PID_P38_ALPHA_BETA_DOWNSTREAM_PATHWAY",
                "PID_P38": "PID_P38_MKK3_6PATHWAY",
                "REACTOME_EGFR_CANCER_VARIANTS": "REACTOME_CONSTITUTIVE_SIGNALING_BY_LIGAND_RESPONSIVE_EGFR_CANCER_VARIANTS",
                "REACTOME_EGFR_DOWNREGULATION": "REACTOME_EGFR_DOWNREGULATION",
                "REACTOME_EGFR": "REACTOME_SIGNALING_BY_EGFR",
                "REACTOME_EGFR_CANCER": "REACTOME_SIGNALING_BY_EGFR_IN_CANCER",
                "REACTOME_ERBB2_OVEREXPRESSION": "REACTOME_CONSTITUTIVE_SIGNALING_BY_OVEREXPRESSED_ERBB2",
                "REACTOME_ERBB2": "REACTOME_SIGNALING_BY_ERBB2",
                "REACTOME_ERBB2_CANCER": "REACTOME_SIGNALING_BY_ERBB2_IN_CANCER",
                "REACTOME_ERK_TARGETS": "REACTOME_ERK_MAPK_TARGETS",
                "REACTOME_ERK": "REACTOME_SIGNALLING_TO_ERKS",
                "REACTOME_MAPK": "REACTOME_MAPK1_MAPK3_SIGNALING",
                "REACTOME_MAPK_CANCER": "REACTOME_ONCOGENIC_MAPK_SIGNALING",
                "REACTOME_P38": "REACTOME_P38MAPK_EVENTS",
                "WP_EGFR": "WP_EGFEGFR_SIGNALING",
                "WP_EGFR_RESISTANCE": "WP_EGFR_TYROSINE_KINASE_INHIBITOR_RESISTANCE",
                "WP_MAPK": "WP_MAPK_SIGNALING",
                "WP_P38": "WP_P38_MAPK_SIGNALING",
            }
            gmt = load_msigdb_gmt()
            gene_list = gmt[gene_sets[gene_set]]

        return [g for g in input_data.columns if g in gene_list]

    elif features.startswith(("RFE_", "HVGRFE_")):
        output_data = output_data.loc[input_data.index, :]
        reduce_factor = 0.80
        # drop nans, this shouldnt do anything
        input_data = input_data.dropna(axis=1, how="any")
        if features.startswith("HVG") and context in [
            "proteomics",
            "transcriptomics",
        ]:
            input_data = get_hvg(input_data)

        output_data -= output_data.mean(axis=0)  # center output data

        n_features = int(features.split("_")[1])
        method = features.split("_")[2]
        random_state = 42  # For reproducibility
        estimator = RandomForestRegressor(
            random_state=random_state,
            max_features=reduce_factor,
        )
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("regressor", estimator)]
        )
        while input_data.shape[1] * reduce_factor > n_features:
            pipeline = pipeline.fit(input_data, output_data)
            y_pred = pipeline.predict(input_data)
            rmse = np.sqrt(np.mean(np.square(output_data.values - y_pred)))
            importances = get_feature_importances(
                pipeline,
                input_data,
                output_data,
                method=method if input_data.shape[1] <= 500 else "tree",
            )

            n_features_target = int(np.ceil(len(importances) * reduce_factor))
            if n_features_target == input_data.shape[1]:
                n_features_target -= 1  # reduce by at least one feature
            indices = np.argsort(importances)[::-1][:n_features_target]
            input_data = input_data.iloc[:, indices]
            print(
                f"Reduced features to: {input_data.shape[1]:>5} ({rmse:.2f})",
            )
        # Fit the final model with the selected features
        pipeline = pipeline.fit(input_data, output_data)
        importances = get_feature_importances(
            pipeline, input_data, output_data, method=method
        )
        indices = np.argsort(importances)[::-1][:n_features]
        return input_data.columns[indices]

    preprocessor = build_preprocessor(features, input_data, output_data, cv=cv)
    preprocessor = preprocessor.fit(input_data, output_data)

    return preprocessor.feature_names_in_[
        preprocessor.steps[-1][1].get_support()
    ]


def build_context_feature_recipe(conf) -> list:
    """
    Returns a list of (context, feature_spec) describing the desired stack.

    Supports dynamic patterns like:
      - cytof_init_plus_tEGFR                 -> cytof_init + transcriptomics [EGFR]
      - cytof_init_plus_pEGFR                 -> cytof_init + proteomics [EGFR]
      - cytof_init_plus_tEGFR_tERBB2          -> cytof_init + transcriptomics [EGFR, ERBB2]
      - cytof_init_plus_tEGFR_pERBB2          -> cytof_init + transcriptomics [EGFR] + proteomics [ERBB2]

    The suffix after `cytof_init_plus_` is underscore-separated tokens of the
    form `[t|p]<GENE>`, where `t` = transcriptomics, `p` = proteomics.

    Keeps existing 'multimodal' behaviors (including 'optimal' and 'best_RFE_*').
    Falls back to single-context otherwise.
    """
    ctx = conf.context

    # Generalized "cytof_init_plus_" parser
    if isinstance(ctx, str) and ctx.startswith("cytof_init_plus_"):
        suffix = ctx[len("cytof_init_plus_") :]
        tokens = [tok for tok in suffix.split("_") if tok]

        genes_by_modality = {"transcriptomics": [], "proteomics": []}
        use_intr = False
        use_lb = False
        for tok in tokens:
            if tok in ("intr", "lb"):
                use_intr = use_intr or (tok == "intr")
                use_lb = use_lb or (tok == "lb")
            else:
                m = re.fullmatch(r"([tp])([A-Za-z0-9_]+)", tok)
                if not m:
                    # Ignore unknown tokens; switch to ValueError if you prefer strictness
                    continue
                which, gene = m.groups()
                modality = "transcriptomics" if which == "t" else "proteomics"
                genes_by_modality[modality].append(gene)

        recipe = [("cytof_init", conf.features)]
        if use_intr:
            recipe.append(("subtype_intr", "onehot"))
        if use_lb:
            recipe.append(("subtype_lb", "onehot"))
        if genes_by_modality["transcriptomics"]:
            recipe.append(
                (
                    "transcriptomics",
                    "genes:" + ",".join(genes_by_modality["transcriptomics"]),
                )
            )
        if genes_by_modality["proteomics"]:
            recipe.append(
                (
                    "proteomics",
                    "genes:" + ",".join(genes_by_modality["proteomics"]),
                )
            )
        return recipe

    # Existing multimodal semantics
    if ctx == "multimodal":
        if conf.features == "optimal":
            return [
                ("cytof_init", "RFE_10_permute"),
                ("proteomics", "HVGRFE_20_permute"),
                ("transcriptomics", "HVGRFE_15_permute"),
            ]
        if isinstance(conf.features, str) and conf.features.startswith(
            "best_RFE_"
        ):
            # HVG prefilter then global RFE on concatenated inputs
            return [
                ("cytof_init", "all"),
                ("proteomics", "HVG_all"),
                ("transcriptomics", "HVG_all"),
            ]
        # Plain multimodal, add HVG on omics unless already requested
        return [
            ("cytof_init", conf.features),
            (
                "proteomics",
                "HVG" + conf.features
                if "HVG" not in conf.features
                else conf.features,
            ),
            (
                "transcriptomics",
                "HVG" + conf.features
                if "HVG" not in conf.features
                else conf.features,
            ),
        ]

    # Single-context fallback
    return [(conf.context, conf.features)]


def prefix_for_context(ctx: str) -> str:
    """Prefix non-cytof contexts to avoid column name clashes when concatenating."""
    if ctx == "proteomics":
        return "p"
    elif ctx == "transcriptomics":
        return "t"
    else:
        return ""  # cytof / subtypes / MOSA


def parse_feature_spec(feature_spec: str):
    """
    Extend the feature spec language with a lightweight 'genes:' option:
      - 'genes:EGFR' or 'genes:EGFR,ERBB2'
    Falls back to native behavior otherwise.
    """
    if isinstance(feature_spec, str) and feature_spec.startswith("genes:"):
        genes = [
            g.strip()
            for g in feature_spec.split(":", 1)[1].split(",")
            if g.strip()
        ]
        return {"kind": "genes", "genes": genes}
    return {"kind": "native", "spec": feature_spec}


def prepare_inputs_for_context(
    subconf,
    samples_train_split,
    samples_val_split,
    feature_spec_str,
    petab_base_files,
    maybe_output_train,
    do_prefix=False,
):
    """
    Unified loader → imputer → mean centering → (optional HVG prefilter) → selection.
    Supports:
      - native specs (all existing: all / RFE_* / HVGRFE_* / curated / MSIGDB_*)
      - 'HVG_all' quick prefilter (keeps all HVG without supervised step) - used to perform feature selection on
        concatenated multimodal contexts.
      - 'genes:...' direct selection by gene symbols (e.g., genes:EGFR)
    """
    spec = parse_feature_spec(feature_spec_str)

    # Load inputs for context
    if subconf.context == "MOSA":
        input_train, input_val, features_all = preprocess_mosa_latent(
            subconf, samples_train_split, samples_val_split
        )
    elif subconf.context == "subtype_intr":
        input_train = onehot_intrinsic(samples_train_split)
        input_val = onehot_intrinsic(samples_val_split)
        features_all = input_train.columns.tolist()
    elif subconf.context == "subtype_lb":
        input_train = onehot_lb(samples_train_split)
        input_val = onehot_lb(samples_val_split)
        features_all = input_train.columns.tolist()
    else:
        input_train, features_all, _, imputer = load_data(
            contextualization=subconf.context,
            samples=samples_train_split,
            features=None,
            **petab_base_files,
        )
        input_val, _, _, _ = load_data(
            contextualization=subconf.context,
            samples=samples_val_split,
            features=features_all,
            imputer=imputer,
            **petab_base_files,
        )

    if "subtype" not in subconf.context:
        # Impute missing input values
        imputer_input = KNNImputer()
        filled = imputer_input.fit_transform(input_train)
        input_train = pd.DataFrame(
            filled,
            index=input_train.index,
            columns=input_train.columns,
        )
        # Mean-center
        mean_train = input_train.mean()
        input_train -= mean_train

        if len(input_val):
            input_val = pd.DataFrame(
                imputer_input.transform(input_val),
                index=input_val.index,
                columns=input_val.columns,
            )

            # Mean-center using training mean
            input_val -= mean_train

    # Feature selection
    if spec["kind"] == "genes":
        # Keep only the requested genes present in this context
        requested = set(spec["genes"])
        selected = [c for c in input_train.columns if c in requested]
        input_train = input_train[selected]
        input_val = input_val[selected]
    else:
        native = spec["spec"]
        # For subtype pseudo-contexts, the features are already one-hot-encoded
        if subconf.context in ("subtype_intr", "subtype_lb"):
            selected = input_train.columns.tolist()
            # ensure val has same columns as train
            input_val = input_val.reindex(columns=selected, fill_value=0.0)
        elif native == "HVG_all":
            # Quick unsupervised HVG prefilter, then keep all remaining
            input_train = get_hvg(input_train)
            selected = input_train.columns.tolist()
            input_val = input_val.reindex(columns=selected, fill_value=0.0)
        else:
            if DROP_HER2_FROM_FEATURES and subconf.context.startswith(
                "cytof_init"
            ):
                if "p.HER2" in input_train.columns:
                    input_train.drop(columns=["p.HER2"], inplace=True)

            # Supervised (or curated/MSIG) selection reusing existing function
            selected = get_selected_features(
                input_data=input_train,
                output_data=maybe_output_train,
                context=subconf.context,
                features=native,
                features_all=input_train.columns.tolist(),
                cv=None,
            )
            input_train = input_train[selected]
            input_val = input_val[selected]

    # Optional prefixing to avoid duplication across contexts
    if do_prefix:
        prefix = prefix_for_context(subconf.context)
        if prefix:
            input_train.columns = [f"{prefix}{c}" for c in input_train.columns]
            input_val.columns = [f"{prefix}{c}" for c in input_val.columns]
            selected = input_train.columns.tolist()
    else:
        selected = input_train.columns.tolist()

    return input_train, input_val, selected


conf = fire.Fire(Conf)
petab_base_files = load_petab_base_files(conf)
del petab_base_files["condition_table"]

if (conf.context == "MOSA") and ("EVSAT" == conf.samples):
    raise ValueError(f"{conf.context} not available for CV split")

recipe = build_context_feature_recipe(conf)

samples_train = training_samples(Wildcards(conf.data, conf.samples))
samples_val = val_samples(Wildcards(conf.data, conf.samples))

# Preload output (cytof_dynamic) once for the split (used by supervised selectors)
output_train_raw, _, _, _ = load_data(
    contextualization="cytof_dynamic",
    samples=samples_train,
    features=None,
    **petab_base_files,
)
imputer_out = KNNImputer()
output_train_imputed = pd.DataFrame(
    imputer_out.fit_transform(output_train_raw),
    index=output_train_raw.index,
    columns=output_train_raw.columns,
)

# Detect global-RFE pattern: 'multimodal' + 'best_RFE_N_permute'
perform_global_rfe = (
    conf.context == "multimodal"
    and isinstance(conf.features, str)
    and conf.features.startswith("best_RFE_")
)

# Prepare each (context, feature_spec)
parts_train, parts_val = [], []
for ctx, feat_spec in recipe:
    subconf = replace(conf, context=ctx, features=feat_spec)
    tr, va, sel = prepare_inputs_for_context(
        subconf=subconf,
        samples_train_split=samples_train,
        samples_val_split=samples_val,
        feature_spec_str=feat_spec,
        petab_base_files=petab_base_files,
        maybe_output_train=output_train_imputed,
        do_prefix=conf.context == "multimodal" or len(recipe) > 1,
    )
    print(f"Selected features for context {ctx}: {sel}")
    parts_train.append(tr)
    parts_val.append(va)

# Concatenate across recipe (if multiple)
Xtr = (
    pd.concat(parts_train, axis=1) if len(parts_train) > 1 else parts_train[0]
)
Xva = pd.concat(parts_val, axis=1) if len(parts_val) > 1 else parts_val[0]

# If requested, run global RFE on the concatenated inputs
if perform_global_rfe:
    N = int(conf.features.split("_")[-2])
    selected_features = get_selected_features(
        input_data=Xtr,
        output_data=output_train_imputed,
        context="multimodal",
        features=f"RFE_{N}_permute",
        features_all=Xtr.columns.tolist(),
        cv=None,
    )
    print(selected_features)
    Xtr = Xtr[selected_features]
    Xva = Xva[selected_features]

# Save once per dataset
for dataset, mat in (("train", Xtr), ("val", Xva)):
    outfile = FEATURES_OUTFILE.format_map(
        dict(**conf.__dict__, dataset=dataset)
    )
    Path(outfile).parent.mkdir(exist_ok=True, parents=True)
    print(f"Saving {dataset} data for split {conf.samples} to {outfile}")
    mat.to_csv(outfile)
