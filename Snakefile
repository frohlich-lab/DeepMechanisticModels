import datetime
import json
import os
import itertools as itt
import re
import urllib.request

from common import (
    PER_SAMPLE_OUTFILE_PARS,
    TRAINED_MODEL,
    per_sample_pretraining_train, per_sample_pretraining_test, tpl_petab_file,
    EVALUATION_TRAINING, EVALUATION_EMBEDDING, EVALUATION_PARAMETER_DEVIATIONS, EVALUATION_FULL_PARAMETERS,
    EVALUATION_REFERENCE, EVALUATION_REGRESSOR,
    MEASUREMENTS_FILE, FEATURES_OUTFILE, EVALUATE_ALL_CSVS,
    SafeDict,
    data_dir,
    fig_dir
)
from generate_run_configs import generate_run_configs
from pathlib import Path
from training_configuration import (
    DATASETS,
    CONTEXTS_FEATURES_BY_FIGURE,
    SPLITS_BY_FIGURE,
    PATHWAYS_BY_FIGURE,
    SELECT_CENTRAL_VALUES_BY_FIGURE,
    PARAMS_TO_SCAN
)
from dmm.config_options import scan_attributes

basedir = Path(os.getcwd())
dmm_dir = basedir / 'dmm'
cytof_dir = basedir / 'cytof'

# Get config arguments from CLI
N_STARTS = int(config.get("num_starts", "5"))
STARTS = [str(i) for i in range(N_STARTS)]

DATE_TAG = str(datetime.date.today())

FIGURE = str(config.get("figure", "default"))

# Figure configurations (training_configuration.py) that the paper is built
# from; targets `paper`, `paper_results` and `figures` cover all of them.
# Restrict with e.g. --config figures=figure2,figure3
PAPER_FIGURES = (
    str(config["figures"]).split(",")
    if "figures" in config
    else [figure for figure in CONTEXTS_FEATURES_BY_FIGURE if figure != "default"]
)

singularity: "docker://fabfroehlich/generic_parameter_estimation:main"

# Passed through to jobs when set. Neither is required: synapseclient falls
# back to ~/.synapseConfig, and W&B only needs a key with WANDB_MODE=online.
envvars:
    *[var for var in ("SYNAPSE_AUTH_TOKEN", "WANDB_API_KEY") if var in os.environ]

wildcard_constraints:
    figure="|".join(map(re.escape, CONTEXTS_FEATURES_BY_FIGURE)),


rule load_data:
    input:
        script='load_data.py',
        data_code=cytof_dir / 'data.py',
    output:
        cytof='data/cytof.csv',
        proteomics='data/proteomics.csv',
        transcriptomics='data/transcriptomics.csv'
    resources:
        mem="8GB",  # tried on cluster and process_data was OOM killed
        runtime="60m",
        nodes=1,
        threads=1
    shell:
        'python3 {input.script}'

rule process_data:
    input:
        script='process_data.py',
        model_code=dmm_dir / 'mechanistic_model.py',
        data_code2=cytof_dir / 'data.py',
        pathways=cytof_dir / 'pathways.py',
        cytof=rules.load_data.output.cytof,
        proteomics=rules.load_data.output.proteomics,
        transcriptomics=rules.load_data.output.transcriptomics,
    output:
        datafiles=expand(
            tpl_petab_file,
            model='{model}',
            data='{data}',
            file=['measurements', 'conditions', 'observables']
        )
    resources:
        mem="8GB",  # tried on cluster and process_data was OOM killed
        runtime="1h",
        nodes=1,
        threads=1
    shell:
        'python3 {input.script} ' + ' '.join(
            f'--{arg}={{wildcards.{arg}}}'
            for arg in ('model', 'data')
        )

rule compile_mechanistic_model:
    input:
        script='compile_model.py',
        model_code=rules.process_data.input.model_code,
        pathways=rules.process_data.input.pathways,
        data=rules.process_data.output.datafiles,
        petab= dmm_dir / 'petab_subproblem.py',
        mechanistic_model= dmm_dir / 'mechanistic_model.py',
    output:
        model= basedir / 'cytof' / 'amici_models' / '{model}_{data}_petab' / '{model}' / '{model}.py'
    resources:
        mem="8GB",
        runtime="1h",
        nodes=1,
        threads=1
    shell:
        'python3 {input.script} ' + ' '.join(
            f'--{arg}={{wildcards.{arg}}}'
            for arg in ('model', 'data')
        )


rule pretrain_per_sample:
    input:
        script='pretrain_per_sample.py',
        pretraining_code=dmm_dir / 'pretraining.py',
        model=rules.compile_mechanistic_model.output.model,
        data=rules.process_data.output.datafiles
    output:
        pretraining=PER_SAMPLE_OUTFILE_PARS
    resources:
        mem="2GB",
        runtime="6h",
        nodes=1,
        threads=2
    shell:
        'python3 {input.script} ' + ' '.join(
            f'--{arg}={{wildcards.{arg}}}'
            for arg in ('model', 'data', 'sample')
        ) + ' --threads={resources.threads}'


rule pretrain_average_model:
    input:
        script='pretrain_average.py',
        pretraining_code=dmm_dir / 'pretraining.py',
        model=rules.compile_mechanistic_model.output.model,
        data=rules.process_data.output.datafiles
    output:
        pretraining=PER_SAMPLE_OUTFILE_PARS.format_map(SafeDict(sample='model_average_{samples}'))
    resources:
        mem="2GB",
        runtime="24h",
        nodes=1,
        threads=2
    shell:
        'python3 {input.script} ' + ' '.join(
            f'--{arg}={{wildcards.{arg}}}'
            for arg in ('model', 'data', 'samples')
        ) + ' --threads={resources.threads}'

rule select_features:
    input:
        script='select_features.py',
        code= dmm_dir / 'feature_selection.py',
        data=MEASUREMENTS_FILE,
    output:
        data=[
            FEATURES_OUTFILE.format_map(SafeDict(dataset=dataset))
            for dataset in ['train', 'val']
        ]
    resources:
        mem="4GB",
        runtime="10h",
        nodes=1,
        threads=1
    shell:
        'python3 {input.script} ' + ' '.join(
            f'--{arg}={{wildcards.{arg}}}'
            for arg in ('model', 'data', 'context', 'features', 'samples')
        )

rule evaluate_references:
    input:
        script='evaluate_reference.py',
        pretrain_per_sample_test=per_sample_pretraining_test,
        pretrain_per_sample_train=per_sample_pretraining_train,
        pretrain_average=rules.pretrain_average_model.output.pretraining
    output:
        csv=[
            EVALUATION_REFERENCE.format_map(SafeDict(dataset=dataset, mode=mode))
            for dataset, mode in itt.product(['train', 'val'], ['per_sample', 'avg_model'])
        ]
    retries: 1
    resources:
        mem="8GB",
        runtime="1h",
        nodes=1,
        threads=1
    shell:
        'python3 {input.script} ' + ' '.join(
        f'--{arg}={{wildcards.{arg}}}'
        for arg in ('model','data','samples')
        ) + ' --n_starts={N_STARTS}'

rule estimate_parameters:
    input:
        script = 'train.py',
        training =dmm_dir / 'training.py',
        data=MEASUREMENTS_FILE,
        model=rules.compile_mechanistic_model.output.model,
        features=rules.select_features.output.data,
        pretrain=rules.pretrain_average_model.output.pretraining,
    output:
        model=TRAINED_MODEL
    retries: 3
    resources:
        mem="8GB",
        runtime="24h",
        nodes=1,
        threads=2
    shell:
        'python3 {input.script} ' + ' '.join(
            f'--{arg}={{wildcards.{arg}}}'
            for arg in scan_attributes
        ) + ' --threads={resources.threads} --date_tag={DATE_TAG} --figure={FIGURE}'

rule evaluate_training:
    input:
        script='evaluate_training.py',
        code=dmm_dir / 'analysis.py',
        training=rules.estimate_parameters.output.model
    output:
        csv=[
            [
                path_format.format_map(SafeDict(dataset=dataset))
                for dataset in ['train', 'val']
            ]
            for path_format in [
                EVALUATION_TRAINING, EVALUATION_EMBEDDING, EVALUATION_PARAMETER_DEVIATIONS, EVALUATION_FULL_PARAMETERS
            ]
        ]
    retries: 1
    resources:
        mem="16GB",
        runtime="90min",
        nodes=1,
        threads=1
    shell:
        'python3 {input.script} ' + ' '.join(
            f'--{arg}={{wildcards.{arg}}}'
            for arg in scan_attributes
        )

rule evaluate_regressors:
    input:
        script='evaluate_regressors.py',
        selected_features=[
            FEATURES_OUTFILE.format_map(SafeDict(
                model='{model}',
                data='{data}',
                context='{context}',
                features='{features}',
                samples='{samples}',
                dataset=dataset
            ))
            for dataset in ['train', 'val']
        ]
    output:
        csv=[
            EVALUATION_REGRESSOR.format_map(
                SafeDict(
                    dataset=dataset,
                    mode=mode,
                )
            )
            for dataset, mode in itt.product(
                ['train', 'val'],
                ['linreg', 'lasso', 'elasticnet'],
            )
        ]
    retries: 1
    resources:
        mem="8GB",
        runtime="1h",
        nodes=1,
        threads=1
    shell:
        'python3 {input.script} ' + ' '.join(
        f'--{arg}={{wildcards.{arg}}}'
        for arg in ('model', 'data', 'context', 'features', 'samples')
        )


def evaluate_all_training(wildcards):
    return [
        y
        for x in rules.evaluate_training.output.csv
        for hyperparam_configuration in generate_run_configs(
            contexts_features=CONTEXTS_FEATURES_BY_FIGURE[wildcards.figure],
            n_starts=N_STARTS,
            select_central_values=SELECT_CENTRAL_VALUES_BY_FIGURE[wildcards.figure],
            params_to_scan=PARAMS_TO_SCAN[wildcards.figure],
            splits=SPLITS_BY_FIGURE[wildcards.figure],
        )
        for y in expand(
            x.format_map(SafeDict(**hyperparam_configuration)),
            model=wildcards.model, data=wildcards.data  # dataset is defined in evaluate_training rule
        )
    ]


def evaluate_all_references(wildcards):
    return expand(
        rules.evaluate_references.output.csv,
        model=wildcards.model, data=wildcards.data,
        samples=SPLITS_BY_FIGURE[wildcards.figure],
    ) + [
        y
        for context, features in CONTEXTS_FEATURES_BY_FIGURE[wildcards.figure]
        for y in expand(
            rules.evaluate_regressors.output.csv,
            model=wildcards.model, data=wildcards.data,
            samples=SPLITS_BY_FIGURE[wildcards.figure],
            context=context, features=features
        )
    ]


# {figure} is a wildcard here (rather than the FIGURE config) so that one
# invocation can evaluate several figure configurations, see `paper` below.
rule evaluate_all:
    input:
        script='evaluate_all.py',
        training=evaluate_all_training,
        reference=evaluate_all_references,
    output:
        csv=[
            EVALUATE_ALL_CSVS.format_map(SafeDict(filename=filename))
            for filename in (
                'evaluate_all_{figure}',
            )
        ]
    threads: 4
    resources:
        mem="96GB",
        runtime="8h",
        nodes=1,
    shell:
        'python3 {input.script} ' + ' '.join(
            f'--{arg}={{wildcards.{arg}}}'
            for arg in ('model', 'data')
        ) + ' --n_starts={N_STARTS} --figure={wildcards.figure}'


# Regular train_and_evaluate
rule report_all:
    input:
        script='report_all.py',
        evaluation=rules.evaluate_all.output.csv,
    output:
        performance=fig_dir / '{model}' / '{data}' / 'performance_{figure}.pdf'
    resources:
        mem="8GB",
        runtime="4h",
        nodes=1,
        threads=1
    shell:
        'python3 {input.script} ' + ' '.join(
            f'--{arg}={{wildcards.{arg}}}'
            for arg in ('model', 'data')
        ) + ' --figure={wildcards.figure}'

rule train_and_evaluate:
    input:
         evaluation=expand(
             rules.report_all.output.performance,  # changed it to CSV as plots might not be generated without stat tests
             model=PATHWAYS_BY_FIGURE[FIGURE], data=DATASETS, figure=FIGURE
         )


# Reproducing the paper: data download -> ... -> evaluation -> figures.
# README.md describes the entry points; reproduce.sh wraps them.

# evaluate_all outputs of every figure configuration in PAPER_FIGURES
PAPER_EVALUATIONS = [
    y
    for figure in PAPER_FIGURES
    for y in expand(
        rules.evaluate_all.output.csv,
        model=PATHWAYS_BY_FIGURE[figure], data=DATASETS, figure=figure
    )
]

# MSigDB canonical pathways (C2:CP), read by figures_paper/figure_4.ipynb
MSIGDB_URL = (
    'https://data.broadinstitute.org/gsea-msigdb/msigdb/release/'
    '2025.1.Hs/c2.cp.v2025.1.Hs.symbols.gmt'
)

rule download_msigdb:
    output:
        data_dir / 'msigdb_c2cp_2025.1.Hs.json'
    run:
        # GMT rows are: name, description, genes...
        with urllib.request.urlopen(MSIGDB_URL) as response:
            rows = response.read().decode().splitlines()
        gene_sets = {
            name: genes for name, _, *genes in (row.split('\t') for row in rows)
        }
        Path(output[0]).write_text(json.dumps(gene_sets))

# Everything up to the evaluation; the expensive part, meant for a cluster.
rule paper_results:
    input:
        evaluation=PAPER_EVALUATIONS


def run_paper_notebooks(script):
    # Called from `run:` blocks, which never use the `singularity:` image: the
    # notebooks need R/limma and the notebook dependencies of the host.
    missing = [path for path in PAPER_EVALUATIONS if not Path(path).exists()]
    if missing:
        print(
            f'{len(missing)} evaluation output(s) missing, notebooks that '
            'read them will fail:\n  ' + '\n  '.join(missing)
        )
    shell(f'python3 {script} --paper-only')


# Single entry point, from data download to the paper figures.
rule paper:
    input:
        script='run_notebooks.py',
        evaluation=PAPER_EVALUATIONS,
        data=rules.load_data.output,
        msigdb=rules.download_msigdb.output,
    output:
        touch('notebook_output/paper.done')
    resources:
        mem="32GB",
        runtime="8h",
        nodes=1,
        threads=1
    run:
        run_paper_notebooks(input.script)

# Figures only, from precomputed evaluation outputs in eval/. Does not depend
# on the evaluation rules, so missing or outdated intermediate files (models,
# per-run evaluations) never trigger retraining. Still downloads the data the
# notebooks read.
rule figures:
    input:
        script='run_notebooks.py',
        data=rules.load_data.output,
        msigdb=rules.download_msigdb.output,
    output:
        touch('notebook_output/figures.done')
    resources:
        mem="32GB",
        runtime="8h",
        nodes=1,
        threads=1
    run:
        run_paper_notebooks(input.script)


localrules: download_msigdb, paper_results

ruleorder: pretrain_average_model > pretrain_per_sample
