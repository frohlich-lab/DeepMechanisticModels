import datetime
import os
import itertools as itt

from common import (
    PER_SAMPLE_OUTFILE_PARS,
    # TRAINING_OUTFILE_RESULTS,
    TRAINED_MODEL,
    # COLLECTED_TRAINING_RESULTS,
    per_sample_pretraining_train, per_sample_pretraining_test, tpl_petab_file,
    EVALUATION_TRAINING, EVALUATION_EMBEDDING, EVALUATION_PARAMETER_DEVIATIONS, EVALUATION_FULL_PARAMETERS,
    EVALUATION_REFERENCE, EVALUATION_REGRESSOR,
    MEASUREMENTS_FILE, FEATURES_OUTFILE, EVALUATE_ALL_CSVS,
    SafeDict,
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
# tmp_dir = basedir / 'tmp'
dmm_dir = basedir / 'dmm'
cytof_dir = basedir / 'cytof'

# Get config arguments from CLI
N_STARTS = int(config.get("num_starts", "5"))
STARTS = [str(i) for i in range(N_STARTS)]

DATE_TAG = str(datetime.date.today())

FIGURE = str(config.get("figure", "default"))

singularity: "docker://fabfroehlich/generic_parameter_estimation:main"

envvars:
    "SYNAPSE_AUTH_TOKEN",
    "WANDB_API_KEY"


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
        # result=TRAINING_OUTFILE_RESULTS,  # removed result files (hdf5)
        model=TRAINED_MODEL
    retries: 3
    resources:
        mem="4GB",
        # disk="2GB",
        # tmpdir=str(tmp_dir),
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
        # data=rules.process_data.output.datafiles,  # wait for download and processing
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


rule evaluate_all:
    input:
        script='evaluate_all.py',
        training = [
            y
            for x in rules.evaluate_training.output.csv
            for hyperparam_configuration in generate_run_configs(
                contexts_features=CONTEXTS_FEATURES_BY_FIGURE[FIGURE],
                n_starts=N_STARTS,
                select_central_values=SELECT_CENTRAL_VALUES_BY_FIGURE[FIGURE],
                params_to_scan=PARAMS_TO_SCAN[FIGURE],
                splits=SPLITS_BY_FIGURE[FIGURE],
            )
            for y in expand(
                x.format_map(SafeDict(**hyperparam_configuration)),
                model='{model}', data='{data}'  # dataset is defined in evaluate_training rule
            )
        ],
        reference=expand(
            rules.evaluate_references.output.csv,
            model='{model}',data='{data}', samples=SPLITS_BY_FIGURE[FIGURE],
        ) + [
            expand(
                rules.evaluate_regressors.output.csv,
                model='{model}', data='{data}', samples=SPLITS_BY_FIGURE[FIGURE],
                context=context, features=features
            )
            for context, features in CONTEXTS_FEATURES_BY_FIGURE[FIGURE]
        ]
    output:
        csv=[
            EVALUATE_ALL_CSVS.format_map(SafeDict(filename=filename))
            for filename in (
                f'evaluate_all_{FIGURE}',
            )
        ]
    resources:
        mem="96GB",
        runtime="8h",
        nodes=1,
        threads=4
    shell:
        'python3 {input.script} ' + ' '.join(
            f'--{arg}={{wildcards.{arg}}}'
            for arg in ('model', 'data')
        ) + ' --n_starts={N_STARTS} --figure={FIGURE}'



# Regular train_and_evaluate
rule report_all:
    input:
        script='report_all.py',
        evaluation=rules.evaluate_all.output.csv,
    output:
        performance=fig_dir / '{model}' / '{data}' / f'performance_{FIGURE}.pdf'
    resources:
        mem="8GB",
        runtime="4h",
        nodes=1,
        threads=1
    shell:
        'python3 {input.script} ' + ' '.join(
            f'--{arg}={{wildcards.{arg}}}'
            for arg in ('model', 'data')
        ) + ' --figure={FIGURE}'

rule train_and_evaluate:
    input:
         evaluation=expand(
             rules.report_all.output.performance,  # changed it to CSV as plots might not be generated without stat tests
             model=PATHWAYS_BY_FIGURE[FIGURE], data=DATASETS
         )


# # Only run references and regressors + whole data processing, feature selection, etc.
# rule evaluate_baselines:
#     input:
#          evaluation=expand(
#              rules.evaluate_references.output.csv,
#              model=PATHWAYS_BY_FIGURE[FIGURE], data=DATASETS, samples=SPLITS,
#          ) + expand(
#              rules.evaluate_regressors.output.csv,
#              model=PATHWAYS_BY_FIGURE[FIGURE],
#              data=DATASETS,
#              samples=SPLITS,
#              context=[c for c, _ in CONTEXTS_FEATURES_BY_FIGURE[FIGURE]],
#              features=[f for _, f in CONTEXTS_FEATURES_BY_FIGURE[FIGURE]],
#              zip_keys=["context", "features"]
#          )


ruleorder: pretrain_average_model > pretrain_per_sample
