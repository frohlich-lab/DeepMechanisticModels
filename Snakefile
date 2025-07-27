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
    CONTEXT_SET, FEATURES_SET, SafeDict,
    fig_dir
)
from generate_run_configs import generate_run_configs
from pathlib import Path
from training_configuration import (
    CONTEXTS_FEATURES, PATHWAYS, DATASETS, SPLITS, HP_RUN_MODE, REFINE_HPS
)

basedir = Path(os.getcwd())
# tmp_dir = basedir / 'tmp'
mencoder_dir = basedir / 'dmm'
cytof_dir = basedir / 'cytof'

# Get config arguments from CLI
N_STARTS = int(config.get("num_starts", "5"))
STARTS = [str(i) for i in range(N_STARTS)]

DATE_TAG = str(datetime.date.today())

singularity: "docker://fabfroehlich/generic_parameter_estimation:main"

envvars:
    "SYNAPSE_AUTH_TOKEN",
    "WANDB_API_KEY"


rule load_data:
    input:
        script='load_data.py'
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
        data_code=mencoder_dir / 'generate_data.py',
        model_code=mencoder_dir / 'mechanistic_model.py',
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
    wildcard_constraints:
        model='\w+',
        data='[\w\.]+'
    resources:
        mem="8GB",  # tried on cluster and process_data was OOM killed
        runtime="60m",
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
        data=rules.process_data.output.datafiles
    output:
        model= basedir / 'cytof' / 'amici_models' / '{model}_{data}_petab' / '{model}' / '{model}.py'
    wildcard_constraints:
        model='\w+',
        data='[\w\.]+'
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
        pretraining_code=mencoder_dir / 'pretraining.py',
        model=rules.compile_mechanistic_model.output.model,
        data=rules.process_data.output.datafiles
    output:
        pretraining=PER_SAMPLE_OUTFILE_PARS
    wildcard_constraints:
        model='\w+',
        data='[\w\.]+',
        sample='\w+'
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
        pretraining_code=mencoder_dir / 'pretraining.py',
        model=rules.compile_mechanistic_model.output.model,
        data=rules.process_data.output.datafiles
    output:
        pretraining=PER_SAMPLE_OUTFILE_PARS.format_map(SafeDict(sample='model_average_{samples}'))
    wildcard_constraints:
        model='\w+',
        data='[\w\.]+',
        samples='[0-9]+of[0-9]+'
    resources:
        mem="2GB",
        runtime="6h",
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
        data=MEASUREMENTS_FILE,
    output:
        data=[
            FEATURES_OUTFILE.format_map(SafeDict(dataset=dataset))
            for dataset in ['train', 'val']
        ]
    wildcard_constraints:
        model='\w+',
        data='[\w\.]+',
        context='\w+',
        features = '\w+',
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
    wildcard_constraints:
        model='\w+',
        data=r'[\w\.]+',
        samples='[0-9]+of[0-9]+'
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
        training = mencoder_dir / 'training.py',
        data=MEASUREMENTS_FILE,
        model=rules.compile_mechanistic_model.output.model,
        features=rules.select_features.output.data,
        pretrain=rules.pretrain_average_model.output.pretraining,
    output:
        # result=TRAINING_OUTFILE_RESULTS,  # removed result files (hdf5)
        model=TRAINED_MODEL
    wildcard_constraints:
        model = '\w+',
        data = r'[\w\.]+',
        samples = '[0-9]+of[0-9]+',
        pretrain = 'True|False',
        context = r"\w+(?:\+\w+)*",
        features = r"\w+(?:\+\w+)*",
        standardise_features= 'True|False',
        freeze_medians='True|False',
        n_hidden = '[0-9]+',
        nn_structure_multiplier = '[0-9]+',
        depth = '[0-9]+',
        use_layer_bias = 'True|False',
        last_layer_activation = 'True|False',
        nn_init_fn = '\w+',
        reconstruct = 'True|False',
        activation_fn_name = '\w+',
        optimiser = '\w+',
        orth_reg_strategy = '\w+',
        l1reg_inflate = '[0-9\.]+',
        l1reg_encode = '[0-9\.]+',
        oreg_inflate = '[0-9\.]+',
        oreg_encode = '[0-9\.]+',
        l1reg_inflater_output='[0-9\.]+',
        inflater_output_reg_epoch='[0-9\.]+',
        sparse_threshold_perc='\w+',
        recon_loss = '[0-9\.]+',
        symm_reg = '[0-9\.]+',
        median_reg='[0-9\.]+',
        max_lrate = '[0-9\.]+',
        lrate_span = '[0-9\.]+',
        lrate_decay = '[0-9\.]+',
        warmup_fct = '[0-9\.]+',
        opt_steps = '[0-9]+',
        opt_mult = '[0-9]+',
        weight_decay = '[0-9\.]+',
        momentum = '[0-9\.]+',
        use_simple_linear_schedule = 'True|False',
        use_early_stopping = 'True|False',
        job = '[0-9]+'
    retries: 3
    resources:
        mem="4GB",
        # disk="2GB",
        # tmpdir=str(tmp_dir),
        runtime="6h",
        nodes=1,
        threads=2
    shell:
        'python3 {input.script} ' + ' '.join(
            f'--{arg}={{wildcards.{arg}}}'
            for arg in (
                'model', 'data', 'samples', 'pretrain',
                'context', 'features', 'standardise_features',
                'freeze_medians',
                'n_hidden', 'nn_structure_multiplier', 'depth',
                'use_layer_bias', 'last_layer_activation', 'nn_init_fn',
                'reconstruct', 'activation_fn_name', 'optimiser',
                'orth_reg_strategy',
                'l1reg_inflate', 'oreg_inflate', 'l1reg_encode', 'oreg_encode', 'l1reg_inflater_output', 'l2reg_inflater_output',
                'recon_loss', 'symm_reg', 'median_reg', 'inflater_output_reg_epoch', 'sparse_threshold_perc',
                'max_lrate', 'lrate_span', 'lrate_decay', 'warmup_fct', 'opt_steps', 'opt_mult',
                'weight_decay', 'momentum',
                'use_simple_linear_schedule', 'use_early_stopping',
            )
        ) + ' --threads={resources.threads} --run_mode_tag={HP_RUN_MODE} --date_tag={DATE_TAG}'

rule evaluate_training:
    input:
        script='evaluate_training.py',
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
    wildcard_constraints:
        model='\w+',
        data=r'[\w\.]+',
        samples='[0-9]+of[0-9]+',
        pretrain='True|False',
        context=r"\w+(?:\+\w+)*",
        features=r"\w+(?:\+\w+)*",
        standardise_features= 'True|False',
        freeze_medians='True|False',
        n_hidden='[0-9]+',
        nn_structure_multiplier='[0-9]+',
        depth='[0-9]+',
        use_layer_bias='True|False',
        last_layer_activation='True|False',
        nn_init_fn='\w+',
        reconstruct='True|False',
        activation_fn_name='\w+',
        optimiser='\w+',
        orth_reg_strategy='\w+',
        l1reg_inflate='[0-9\.]+',
        l1reg_encode='[0-9\.]+',
        oreg_inflate='[0-9\.]+',
        oreg_encode='[0-9\.]+',
        l1reg_inflater_output='[0-9\.]+',
        l2reg_inflater_output='[0-9\.]+',
        inflater_output_reg_epoch='[0-9]+',
        sparse_threshold_perc='\w+',
        recon_loss='[0-9\.]+',
        symm_reg='[0-9\.]+',
        median_reg='[0-9\.]+',
        max_lrate='[0-9\.]+',
        lrate_span='[0-9\.]+',
        lrate_decay='[0-9\.]+',
        warmup_fct='[0-9\.]+',
        opt_steps='[0-9]+',
        opt_mult='[0-9]+',
        weight_decay='[0-9\.]+',
        momentum='[0-9\.]+',
        use_simple_linear_schedule='True|False',
        use_early_stopping='True|False',
        job='[0-9]+'
    retries: 1
    resources:
        mem="16GB",
        runtime="90min",
        nodes=1,
        threads=1
    shell:
        'python3 {input.script} ' + ' '.join(
            f'--{arg}={{wildcards.{arg}}}'
            for arg in (
                'model', 'data',
                'samples', 'pretrain',
                'context', 'features', 'standardise_features',
                'freeze_medians', 'n_hidden', 'nn_structure_multiplier', 'depth',
                'use_layer_bias', 'last_layer_activation', 'nn_init_fn',
                'reconstruct', 'activation_fn_name', 'optimiser',
                'orth_reg_strategy',
                'l1reg_inflate', 'oreg_inflate', 'l1reg_encode', 'oreg_encode', 'l1reg_inflater_output', 'l2reg_inflater_output',
                'recon_loss', 'symm_reg', 'median_reg', 'inflater_output_reg_epoch', 'sparse_threshold_perc',
                'max_lrate', 'lrate_span', 'lrate_decay', 'warmup_fct', 'opt_steps', 'opt_mult',
                'weight_decay', 'momentum',
                'use_simple_linear_schedule', 'use_early_stopping',
                'job',
            )
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
    wildcard_constraints:
        model='\w+',
        data=r'[\w\.]+',
        context='\w+',
        features='\w+',
        samples='[0-9]+of[0-9]+'
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
                n_starts=N_STARTS,
                hp_run_mode=HP_RUN_MODE,  # set in training_configuration.py
                refine_hps=REFINE_HPS,  # set in training_configuration.py
            )
            for y in expand(
                x.format_map(SafeDict(**hyperparam_configuration)),
                model='{model}', data='{data}'  # dataset is defined in evaluate_training rule
            )
        ],
        reference=expand(
            rules.evaluate_references.output.csv,
            model='{model}',data='{data}', samples=SPLITS,
        ) + [
            expand(
                rules.evaluate_regressors.output.csv,
                model='{model}', data='{data}', samples=SPLITS,
                context=context, features=features
            )
            for context, features in CONTEXTS_FEATURES
        ]
    output:  # TODO @GiacomoFabrini -- need to edit output plots and csvs
        # plot=[
            # EVALUATE_ALL.format_map(SafeDict(group=group))
            # for group in (
                # 'n_hidden',
                # 'reconstruct', 'activation_fn_name',
                # 'orth_reg_strategy',
                # 'l1reg_encode', 'l1reg_inflate', 'oreg_encode', 'oreg_inflate', 'recon_loss', 'symm_reg',
                # 'heatmaps_n_hidden_pairwise',
                # 'volcano_plot_stat_test',
            # )
        # ],
        csv=[
            EVALUATE_ALL_CSVS.format_map(SafeDict(filename=filename))
            for filename in (
                'evaluate_all',
                # 'stat_tests_all',
            )
        ]
    wildcard_constraints:
        model='\w+',
        data=r'[\w\.]+',
        samples='[0-9]+of[0-9]+'
    resources:
        mem="16GB",
        runtime="90m",
        nodes=1,
        threads=1
    shell:
        'python3 {input.script} ' + ' '.join(
            f'--{arg}={{wildcards.{arg}}}'
            for arg in ('model', 'data')
        ) + ' --n_starts={N_STARTS}'



# Regular train_and_evaluate
rule report_all:
    input:
        script='report_all.py',
        evaluation=rules.evaluate_all.output.csv,
    output:
        performance=fig_dir / '{model}' / '{data}' / 'performance.pdf'
    shell:
        'python3 {input.script}' + ' '.join(
            f'--{arg}={{wildcards.{arg}}}'
            for arg in ('model', 'data')
        )

rule train_and_evaluate:
    input:
         evaluation=expand(
             rules.report_all.output.performance,  # changed it to CSV as plots might not be generated without stat tests
             model=PATHWAYS, data=DATASETS
         )


# Only run references and regressors + whole data processing, feature selection, etc.
rule evaluate_baselines:
    input:
         evaluation=expand(
            rules.evaluate_references.output.csv,
            model=PATHWAYS, data=DATASETS, samples=SPLITS,
         ) + expand(
            rules.evaluate_regressors.output.csv,
            model=PATHWAYS, data=DATASETS, samples=SPLITS, context=CONTEXT_SET,
            features=FEATURES_SET,
         )


ruleorder: pretrain_average_model > pretrain_per_sample
