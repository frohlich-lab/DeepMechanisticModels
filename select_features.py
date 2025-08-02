from dataclasses import dataclass, replace
from pathlib import Path

import fire
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import FEATURES_OUTFILE, Wildcards, training_samples, val_samples
from dmm.feature_selection import (
    build_preprocessor,
    load_data,
    preprocess_mosa_latent,
)
from training_configuration import SPLITS
from util import load_petab_base_files


@dataclass(init=True)
class MinimalConf(dict):
    model: str
    data: str
    context: str
    features: str
    samples: str


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
            list = curated_features[features]
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
            from gseapy import Msigdb

            msig = Msigdb()
            gmt = msig.get_gmt(category="c2.cp", dbver="2025.1.Hs")
            list = gmt[gene_sets[gene_set]]

        return [g for g in input_data.columns if g in list]

    elif features.startswith("RFE_") or features.startswith("HVGRFE_"):
        reduce_factor = 0.80
        # drop nans, this shouldnt do anything
        input_data = input_data.dropna(axis=1, how="any")
        if features.startswith("HVG") and context in [
            "proteomics",
            "transcriptomics",
        ]:
            # remove 20% of features with lowest mean:
            means = np.mean(input_data, axis=0)
            threshold = np.percentile(means, 20)
            input_data = input_data.loc[:, means >= threshold]
            # Keep top 500 features with highest variance
            var_threshold = sorted(
                np.nanvar(input_data, axis=0), reverse=True
            )[500]
            input_data = input_data.loc[
                :, np.nanvar(input_data, axis=0) >= var_threshold
            ]

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
                pipeline, input_data, output_data, method=method
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


conf = fire.Fire(MinimalConf)
petab_base_files = load_petab_base_files(conf)
del petab_base_files["condition_table"]

if (conf.context == "MOSA") and ("4of5" == conf.samples):
    raise ValueError(f"{conf.context} not available for CV split")

samples_train = {
    split: sorted(training_samples(Wildcards(conf.data, split)))
    for split in sorted(SPLITS)
}
samples_val = {
    split: sorted(val_samples(Wildcards(conf.data, split)))
    for split in sorted(SPLITS)
}


# Handle multimodality
contexts = []
multimodal_dfs = {}
if conf.context == "multimodal":
    contexts = ["cytof_init", "proteomics", "transcriptomics"]
else:
    contexts = [conf.context]

features_dict = {}
if conf.features == "optimal":
    # Hardcoded optimal feature selection methods
    features_dict = {
        "cytof_init": "RFE_10_permute",
        "proteomics": "HVGRFE_20_permute",
        "transcriptomics": "HVGRFE_15_permute",
    }
else:
    features_dict = {context: conf.features for context in contexts}

for context in contexts:
    subconf = replace(conf, context=context, features=features_dict[context])

    input_parts = []
    output_parts = []
    features_all = None
    all_indices = []
    split_indices = []

    if subconf.context == "MOSA":
        input_train, input_val, features_all = preprocess_mosa_latent(
            subconf, samples_train[conf.samples], samples_val[conf.samples]
        )
    else:
        input_train, features_all = load_data(
            contextualization=context,
            samples=samples_train[conf.samples],
            features=None,
            **petab_base_files,
        )
        input_val, _ = load_data(
            contextualization=context,
            samples=samples_val[conf.samples],
            features=features_all,
            **petab_base_files,
        )

    imputer_input = KNNImputer()
    filled = imputer_input.fit_transform(input_train)
    input_train = pd.DataFrame(
        filled,
        index=input_train.index,
        columns=input_train.columns,
    )
    input_val = pd.DataFrame(
        imputer_input.transform(input_val),
        index=input_val.index,
        columns=input_val.columns,
    )

    mean_train = input_train.mean()
    input_train -= mean_train
    input_val -= mean_train

    output_train, features_output_train = load_data(
        contextualization="cytof_dynamic",
        samples=samples_train[conf.samples],
        features=None,
        **petab_base_files,
    )
    imputer_output = KNNImputer()
    filled = imputer_output.fit_transform(output_train)
    output_train = pd.DataFrame(
        filled,
        index=output_train.index,
        columns=output_train.columns,
    )

    selected_features = get_selected_features(
        input_data=input_train,
        output_data=output_train,
        context=subconf.context,
        features=subconf.features,
        features_all=features_all,
        cv=None,
    )
    print(
        f"Selected {len(selected_features)} features for split {conf.samples} for {subconf.context}: {selected_features}"
    )
    # Transform and save per split
    for dataset, inputs in zip(("train", "val"), (input_train, input_val)):
        outfile = FEATURES_OUTFILE.format_map(
            dict(**subconf.__dict__, dataset=dataset)
        )
        Path(outfile).parent.mkdir(exist_ok=True, parents=True)
        print(f"Preprocessing {dataset} data for split {conf.samples}...")
        df_inputs = pd.DataFrame(
            inputs[selected_features].values,
            index=inputs.index,
            columns=[
                col
                if isinstance(col, str)
                else "#".join([str(level) for level in col])
                for col in selected_features
            ],
        )
        if not (conf.context == "multimodal"):
            print(
                f"Saving {dataset} data for split {conf.samples} to {outfile}"
            )
            df_inputs.to_csv(outfile)
        else:
            if dataset not in multimodal_dfs:
                multimodal_dfs[dataset] = []
            multimodal_dfs[dataset].append(df_inputs)

if conf.context == "multimodal":
    for dataset in ["train", "val"]:
        outfile = FEATURES_OUTFILE.format_map(
            dict(**conf.__dict__, dataset=dataset)
        )
        concat_df = pd.concat(multimodal_dfs[dataset], axis=1)
        print(f"Saving {dataset} data for split {conf.samples} to {outfile}")
        concat_df.to_csv(outfile)
