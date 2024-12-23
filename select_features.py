from dataclasses import dataclass, replace
from pathlib import Path

import fire
import pandas as pd

from common import FEATURES_OUTFILE, Wildcards, test_samples, training_samples
from dmm.feature_selection import build_preprocesser, load_data, preprocess_mosa_latent
from sklearn.impute import KNNImputer
from util import load_petab_base_files


@dataclass(init=True)
class MinimalConf(dict):
    model: str
    data: str
    context: str
    features: str
    samples: str


conf = fire.Fire(MinimalConf)

petab_base_files = load_petab_base_files(conf)
del petab_base_files["condition_table"]

if conf.context == "MOSA" and conf.samples == "4of5":
    raise ValueError(f"{conf.context} not available for CV split {conf.samples}")

samples_train = training_samples(Wildcards(conf.data, conf.samples))
samples_val = test_samples(Wildcards(conf.data, conf.samples))

# Modified to handle multiple context in case of multimodal learning (context1+context2+...+contextN).
# For now, this simply repeats the same procedure for each context independently.
for context in conf.context.split("+"):
    # Replace the multi-context with single sub-context
    subconf = replace(conf, context=context)

    if subconf.context == "MOSA":
        input_train, input_val, features_train = preprocess_mosa_latent(subconf, samples_train, samples_val)
    else:
        input_train, features_train = load_data(
            contextualization=context,
            samples=samples_train,
            features=None,
            **petab_base_files,
        )
        input_val, _ = load_data(
            contextualization=context,
            samples=samples_val,
            features=features_train,
            **petab_base_files,
        )

    output_train, targets_train = load_data(
        contextualization="cytof_dynamic",
        samples=samples_train,
        features=None,
        **petab_base_files,
    )

    preprocessor = build_preprocesser(conf.features, input_train, output_train)
    # Need to impute missing ERBB2 data in output
    output_train_filled = KNNImputer().fit_transform(output_train)
    preprocessor = preprocessor.fit(input_train, output_train_filled)

    if conf.features == "all":
        features = features_train
    else:
        selector = preprocessor.steps[-1][1]
        features = preprocessor.feature_names_in_[selector.get_support()]


    print(
        f"selected {len(features)} features for {conf.features} feature selection on {conf.context} data: {features}"
    )

    # TODO @GiacomoFabrini - make sure to check this when changing into val and adding untouched test
    for dataset, inputs in zip(("train", "val"), (input_train, input_val)):
        outfile = FEATURES_OUTFILE.format_map(
            dict(**subconf.__dict__, dataset=dataset)
        )
        Path(outfile).parent.mkdir(exist_ok=True, parents=True)
        print(f"preprocessing {dataset} data to {outfile}")
        df_inputs = pd.DataFrame(inputs[features].values, index=inputs.index, columns=features)
        df_inputs.to_csv(outfile)
