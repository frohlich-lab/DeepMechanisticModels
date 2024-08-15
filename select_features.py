from dataclasses import dataclass
from pathlib import Path

import fire
import pandas as pd

from common import FEATURES_OUTFILE, Wildcards, test_samples, training_samples
from dmm.feature_selection import build_preprocesser, load_data
from util import load_petab_base_files


@dataclass
class RegressionConf(dict):
    model: str
    data: str
    context: str
    features: str
    samples: str


conf = fire.Fire(RegressionConf)

petab_base_files = load_petab_base_files(conf)
del petab_base_files["condition_table"]

samples_train = training_samples(Wildcards(conf.data, conf.samples))
samples_val = test_samples(Wildcards(conf.data, conf.samples))

input_train, features_train = load_data(
    contextualization=conf.context,
    samples=samples_train,
    features=None,
    **petab_base_files,
)
input_val, _ = load_data(
    contextualization=conf.context,
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
preprocessor = preprocessor.fit(input_train, output_train)

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
        dict(**conf.__dict__, dataset=dataset)
    )
    Path(outfile).parent.mkdir(exist_ok=True, parents=True)
    print(f"preprocessing {dataset} data to {outfile}")
    df_inputs = pd.DataFrame(inputs[features].values, index=inputs.index, columns=features)
    df_inputs.to_csv(outfile)
