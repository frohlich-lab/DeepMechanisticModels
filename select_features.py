from dataclasses import dataclass, replace
from pathlib import Path

import fire
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import KNNImputer
from sklearn.model_selection import PredefinedSplit

from common import FEATURES_OUTFILE, Wildcards, test_samples, training_samples
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
    features_selection: str


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
    if features.startswith("HVG"):
        # Build and fit per-split preprocessor on training data only
        n_features = int(features.replace("HVG", ""))
        selector = VarianceThreshold(
            threshold=sorted(np.nanvar(input_data, axis=0), reverse=True)[
                min(n_features, input_data.shape[1] - 1)
            ]
        )
        selector = selector.fit(input_data)
    else:
        preprocessor = build_preprocessor(
            features, input_data, output_data, cv=cv
        )
        preprocessor = preprocessor.fit(input_data, output_data)
        selector = preprocessor.steps[-1][1]

    return selector.feature_names_in_[selector.get_support()]


conf = fire.Fire(MinimalConf)
petab_base_files = load_petab_base_files(conf)
del petab_base_files["condition_table"]

if (conf.context == "MOSA") and ("4of5" in SPLITS):
    raise ValueError(f"{conf.context} not available for CV split")

samples_train = {
    split: sorted(training_samples(Wildcards(conf.data, split)))
    for split in sorted(SPLITS)
}
samples_val = {
    split: sorted(test_samples(Wildcards(conf.data, split)))
    for split in sorted(SPLITS)
}

inputs_dict = {split: {} for split in sorted(SPLITS)}
outputs_dict = {split: {} for split in sorted(SPLITS)}

for context in conf.context.split("+"):
    subconf = replace(conf, context=context)

    input_parts = []
    output_parts = []
    features_all = None
    all_indices = []
    split_indices = []

    for split in sorted(SPLITS):
        if subconf.context == "MOSA":
            input_train, input_val, features_all = preprocess_mosa_latent(
                subconf, samples_train[split], samples_val[split]
            )
        else:
            input_train, features_all = load_data(
                contextualization=context,
                samples=samples_train[split],
                features=None,
                **petab_base_files,
            )
            input_val, _ = load_data(
                contextualization=context,
                samples=samples_val[split],
                features=features_all,
                **petab_base_files,
            )

        output_train, features_output_train = load_data(
            contextualization="cytof_dynamic",
            samples=samples_train[split],
            features=None,
            **petab_base_files,
        )
        output_val, _ = load_data(
            contextualization="cytof_dynamic",
            samples=samples_val[split],
            features=features_output_train,
            **petab_base_files,
        )

        imputer = KNNImputer()
        imputer.fit(output_train)

        output_train_filled = pd.DataFrame(
            imputer.transform(output_train),
            index=output_train.index,
            columns=output_train.columns,
        )
        output_val_filled = pd.DataFrame(
            imputer.transform(output_val),
            index=output_val.index,
            columns=output_val.columns,
        )

        # Concatenate and record indices for PredefinedSplit
        n_before = len(all_indices)
        input_parts.extend([input_train, input_val])
        inputs_dict[split] = {
            "train": input_train,
            "val": input_val,
        }
        output_parts.extend([output_train_filled, output_val_filled])
        outputs_dict[split] = {
            "train": output_train_filled,
            "val": output_val_filled,
        }
        all_indices.extend(
            input_train.index.tolist() + input_val.index.tolist()
        )

        train_idx = list(range(n_before, n_before + len(input_train)))
        val_idx = list(
            range(
                n_before + len(input_train),
                n_before + len(input_train) + len(input_val),
            )
        )
        split_indices.append((train_idx, val_idx))

    # Combine everything
    input_all = pd.concat(input_parts)
    output_all = pd.concat(output_parts)

    assert (
        input_all.shape[0] == output_all.shape[0]
    ), "Mismatched rows between inputs and outputs"
    assert all(
        input_all.index == output_all.index
    ), "Mismatched indices between inputs and outputs"

    if conf.features_selection == "across_cv":
        # Create PredefinedSplit for feature selection across all CV splits
        test_fold = [-1] * len(input_all)
        for cv_split, (_, val_idx) in zip(sorted(SPLITS), split_indices):
            for i in val_idx:
                test_fold[i] = int(cv_split.split("of")[0])
        cv = PredefinedSplit(test_fold)

        selected_features = get_selected_features(
            input_data=input_all,
            output_data=output_all,
            context=subconf.context,
            features=conf.features,
            features_all=features_all,
            cv=cv,
        )
        print(
            f"Selected {len(selected_features)} features shared across splits for {subconf.context}: {selected_features}"
        )
        selected_features_dict = {
            split: selected_features for split in sorted(SPLITS)
        }

    elif conf.features_selection == "per_cv":
        selected_features_dict = {}
        for split in sorted(SPLITS):
            input_train_split = inputs_dict[split]["train"]
            output_train_split = outputs_dict[split]["train"]

            selected_features = get_selected_features(
                input_data=input_train_split,
                output_data=output_train_split,
                context=subconf.context,
                features=conf.features,
                features_all=features_all,
                cv=None,
            )
            print(
                f"Selected {len(selected_features)} features for split {split} for {subconf.context}: {selected_features}"
            )

            selected_features_dict[split] = selected_features
    else:
        raise ValueError(
            f"Unknown feature selection method {conf.features_selection}"
        )

    # Transform and save per split
    for split in sorted(SPLITS):
        if subconf.context == "MOSA":
            input_train, input_val, _ = preprocess_mosa_latent(
                subconf, samples_train, samples_val
            )
        else:
            input_train = inputs_dict[split]["train"]
            input_val = inputs_dict[split]["val"]

        for dataset, inputs in zip(("train", "val"), (input_train, input_val)):
            outfile = FEATURES_OUTFILE.format_map(
                dict(**subconf.__dict__, dataset=dataset, samples=split)
            )
            Path(outfile).parent.mkdir(exist_ok=True, parents=True)
            print(
                f"Preprocessing {dataset} data for split {split} to {outfile}"
            )
            df_inputs = pd.DataFrame(
                inputs[selected_features_dict[split]].values,
                index=inputs.index,
                columns=selected_features_dict[split],
            )
            df_inputs.to_csv(outfile)
