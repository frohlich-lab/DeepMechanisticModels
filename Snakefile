import os
import itertools as itt

from common import (
    PER_SAMPLE_OUTFILE_PARS, TRAINING_OUTFILE_RESULTS, TRAINED_BEST_MODELS,
    # COLLECTED_TRAINING_RESULTS,
    per_sample_pretraining_train, per_sample_pretraining_test, tpl_petab_file,
    EVALUATION_TRAINING, EVALUATE_ALL, EVALUATION_REFERENCE, EVALUATION_REGRESSOR,
    MEASUREMENTS_FILE_RW, FEATURES_OUTFILE, EVALUATE_ALL_CSVS,
    CONTEXT_SET
)
from pathlib import Path
from training_configuration import (
    PATHWAYS, DATASETS, CONTEXTS_FEATURES, SPLITS, PRETRAIN,
    LATENT_DIMS, NETWORK_LAYOUT, USE_BIAS, NN_INIT_FN,
    RECONSTRUCT, ACTIVATION_FNS, OPTIMISERS,
    ORTH_REG_STRATEGIES, ALPHAS, BETAS, GAMMAS, DELTAS, EPSILONS, ZETAS,
    MAX_LEARNING_RATES, LEARNING_RATE_SPANS, LEARNING_RATE_DECAYS, WARMUP_FCTS, OPT_STEPS, OPT_MULT, LINEAR_SCHEDULE,
    USE_EARLY_STOP, DROP_REG_POST_PRETRAIN, SPARSITY_THRESHOLD, FEATURES_TRANSFORM
)

basedir = Path(os.getcwd())
mencoder_dir = basedir / 'dmm'
cytof_dir = basedir / 'cytof'

class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'


N_STARTS = int(config.get("num_starts", "10"))
STARTS = [str(i) for i in range(N_STARTS)]

singularity: "docker://fabfroehlich/generic_parameter_estimation:main"

envvars:
    "SYNAPSE_AUTH_TOKEN",
    "WANDB_API_KEY"

rule process_data:
    input:
        script='process_data.py',
        data_code=mencoder_dir / 'generate_data.py',
        model_code=mencoder_dir / 'mechanistic_model.py',
        pathway=cytof_dir / 'pw_{model}.py',
        pathways=cytof_dir / 'pathways.py',
    output:
        datafiles=expand(
            tpl_petab_file,
            model='{model}',
            data='{data}',
            file=['measurements', 'conditions', 'observables']
        ),
    wildcard_constraints:
        model='\w+',
        data='[\w\.]+',
    resources:
        mem="2GB",
        runtime="15m",
        nodes=1,
        threads=1,
    shell:
        'python3 {input.script} ' + ' '.join(
            f'--{arg}={{wildcards.{arg}}}'
            for arg in ('model', 'data')
        )

rule compile_mechanistic_model:
    input:
        script='compile_model.py',
        model_code=rules.process_data.input.model_code,
        pathway=rules.process_data.input.pathway,
        pathways=rules.process_data.input.pathways,
        data=rules.process_data.output.datafiles
    output:
        model= basedir / 'cytof' / 'amici_models' / '{model}_{data}_petab' / '{model}' / '{model}.py',
    wildcard_constraints:
        model='\w+',
        data='[\w\.]+',
    resources:
        mem="8GB",
        runtime="1h",
        nodes=1,
        threads=1,
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
        sample='\w+',
    resources:
        mem="1GB",
        runtime="6h",
        nodes=1,
        threads=1,
    shell:
        'python3 {input.script} ' + ' '.join(
            f'--{arg}={{wildcards.{arg}}}'
            for arg in ('model', 'data', 'sample')
        )


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
        samples='[0-9]+_[0-9]+',
    resources:
        mem="1GB",
        runtime="6h",
        nodes=1,
        threads=1,
    shell:
        'python3 {input.script} ' + ' '.join(
            f'--{arg}={{wildcards.{arg}}}'
            for arg in ('model', 'data', 'samples')
        )


rule reweight_data:
    input:
        script='reweight_data.py',
        pretraining_code=mencoder_dir / 'pretraining.py',
        model=rules.compile_mechanistic_model.output.model,
        data=rules.process_data.output.datafiles,
        pretrain_per_sample=per_sample_pretraining_train,
    output:
        data=MEASUREMENTS_FILE_RW
    wildcard_constraints:
        model='\w+',
        data='[\w\.]+',
        samples='[0-9]+_[0-9]+',
    resources:
        mem="1GB",
        runtime="1h",
        nodes=1,
        threads=1,
    shell:
        'python3 {input.script} ' + ' '.join(
            f'--{arg}={{wildcards.{arg}}}'
            for arg in ('model', 'data', 'samples')
        )


rule select_features:
    input:
        script='select_features.py',
        data=rules.process_data.output.datafiles,
        data_rw=rules.reweight_data.output.data,
    output:
        data=[
            FEATURES_OUTFILE.format_map(SafeDict(dataset=dataset))
            for dataset in ['train', 'val']
        ]
    wildcard_constraints:
        model='\w+',
        data='[\w\.]+',
        samples='[0-9]+_[0-9]+',
    resources:
        mem="1GB",
        runtime="10h",
        nodes=1,
        threads=1,
    shell:
        'python3 {input.script} ' + ' '.join(
            f'--{arg}={{wildcards.{arg}}}'
            for arg in ('model', 'data', 'context', 'features', 'samples')
        )

# TODO @GiacomoFabrini - missing wildcard constraints for network structure parameters
rule estimate_parameters:
    input:
        script = 'train.py',
        # encoder = mencoder_dir / 'encoder.py',
        training = mencoder_dir / 'training.py',
        # autoencoder = mencoder_dir / 'autoencoder.py',
        data=rules.process_data.output.datafiles,
        data_rw=rules.reweight_data.output.data,
        model=rules.compile_mechanistic_model.output.model,
        features=rules.select_features.output.data,
        pretrain_per_sample=per_sample_pretraining_train,
    output:
        result=TRAINING_OUTFILE_RESULTS,
        model=TRAINED_BEST_MODELS,
    wildcard_constraints:
        model='\w+',
        data='[\w\.]+',
        samples='[0-9]+_[0-9]+',
        pretrain='True|False',
        context='\w+',
        n_hidden='[0-9]+',
        reconstruct='True|False',
        activation_fn_name='\w+',
        optimiser='\w+',
        orth_reg_strategy='\w+',
        l1reg_inflate='[0-9\.]+',
        l1reg_encode='[0-9\.]+',
        oreg_inflate='[0-9\.]+',
        oreg_encode='[0-9\.]+',
        recon_loss='[0-9\.]+',
        symm_reg='[0-9\.]+',
        max_lrate='[0-9\.]+',
        lrate_span='[0-9\.]+',
        lrate_decay='[0-9\.]+',
        warmup_fct='[0-9\.]+',
        opt_steps='[0-9]+',
        opt_mult='[0-9\.]+',
        use_simple_linear_schedule='True|False',
        use_early_stopping='True|False',
        drop_reg_after_pretrain='True|False',
        sparsity_threshold='[0-9\.]+',
        features_transform='\w+',
        job='[0-9]+',
    retries: 1
    resources:
        mem="1GB",
        runtime="24h",
        nodes=1,
        threads=2,
    shell:
        'python3 {input.script} ' + ' '.join(
            f'--{arg}={{wildcards.{arg}}}'
            for arg in (
                'model', 'data',
                'samples', 'pretrain',
                'context', 'features', 'features_transform', 'n_hidden',
                'encoder_layer_sizes', 'inflater_layer_sizes', 'linear_benchmark',
                'use_layer_bias', 'nn_init_fn',
                'reconstruct', 'activation_fn_name', 'optimiser',
                'orth_reg_strategy',
                'l1reg_inflate', 'oreg_inflate', 'l1reg_encode', 'oreg_encode',
                'recon_loss', 'symm_reg',
                'max_lrate', 'lrate_span', 'lrate_decay', 'warmup_fct', 'opt_steps', 'opt_mult',
                'use_simple_linear_schedule', 'use_early_stopping', 'drop_reg_after_pretrain', 'sparsity_threshold',
                'job',
            )
        ) + ' --threads={threads}'

# rule collect_estimation_results:
#     input:
#         script='collect_estimation.py',
#         trace=[
#             TRAINING_OUTFILE_RESULTS.format_map(SafeDict(job=job))
#             for job in STARTS
#         ]
#     output:
#         result=COLLECTED_TRAINING_RESULTS
#     wildcard_constraints:
#         model='\w+',
#         data='[\w\.]+',
#         context='\w+',
#         n_hidden='[0-9]+',
#         job='[0-9]+',
#         samples='[0-9]+_[0-9]+',
#         l1reg_inflate='[0-9\.]+',
#     resources:
#         mem="8GB",
#         runtime="1h",
#         nodes=1,
#         threads=1,
#     shell:
#         'python3 {input.script} ' + ' '.join(
#             f'--{arg}={{wildcards.{arg}}}'
#             for arg in ('model', 'data', 'context', 'samples', 'n_hidden', 'orth_reg_strategy',
#                         'l1reg_inflate', 'oreg_inflate', 'l1reg_encode', 'oreg_encode',
#                         'features')
#         ) + ' --n_starts={N_STARTS}'

rule evaluate_training:
    input:
        script='evaluate_training.py',
        training=rules.estimate_parameters.output.model
    output:
        csv=[
            EVALUATION_TRAINING.format_map(SafeDict(dataset=dataset))
            for dataset in ['train', 'test']
        ]
    wildcard_constraints:
        model='\w+',
        data=r'[\w\.]+',
        samples='[0-9]+_[0-9]+',
        pretrain='True|False',
        context='\w+',
        n_hidden='[0-9]+',
        reconstruct='True|False',
        activation_fn_name='\w+',
        optimiser='\w+',
        orth_reg_strategy='\w+',
        l1reg_inflate='[0-9\.]+',
        l1reg_encode='[0-9\.]+',
        oreg_inflate='[0-9\.]+',
        oreg_encode='[0-9\.]+',
        recon_loss='[0-9\.]+',
        symm_reg='[0-9\.]+',
        max_lrate='[0-9\.]+',
        lrate_span='[0-9\.]+',
        lrate_decay='[0-9\.]+',
        warmup_fct='[0-9\.]+',
        opt_steps='[0-9]+',
        opt_mult='[0-9\.]+',
        use_simple_linear_schedule='True|False',
        use_early_stopping='True|False',
        drop_reg_after_pretrain='True|False',
        sparsity_threshold='[0-9\.]+',
        features_transform='\w+',
        job='[0-9]+',
    resources:
        mem="16GB",
        runtime="90min",
        nodes=1,
        threads=1,
    shell:
        'python3 {input.script} ' + ' '.join(
            f'--{arg}={{wildcards.{arg}}}'
            for arg in (
                'model', 'data',
                'samples', 'pretrain',
                'context', 'features', 'features_transform', 'n_hidden',
                'encoder_layer_sizes', 'inflater_layer_sizes', 'linear_benchmark',
                'use_layer_bias', 'nn_init_fn',
                'reconstruct', 'activation_fn_name', 'optimiser',
                'orth_reg_strategy',
                'l1reg_inflate', 'oreg_inflate', 'l1reg_encode', 'oreg_encode',
                'recon_loss', 'symm_reg',
                'max_lrate', 'lrate_span', 'lrate_decay', 'warmup_fct', 'opt_steps', 'opt_mult',
                'use_simple_linear_schedule', 'use_early_stopping', 'drop_reg_after_pretrain', 'sparsity_threshold',
                'job',
            )
        )

rule evaluate_references:
    input:
        script='evaluate_reference.py',
        pretrain_per_sample=per_sample_pretraining_test,
        pretrain_average=rules.pretrain_average_model.output.pretraining
    output:
        csv=[
            EVALUATION_REFERENCE.format_map(SafeDict(dataset=dataset, mode=mode))
            for dataset, mode in itt.product(['train', 'test'], ['per_sample', 'avg_model'])
        ]
    wildcard_constraints:
        model='\w+',
        data=r'[\w\.]+',
        samples='[0-9]+_[0-9]+',
    retries: 1
    resources:
        mem="8GB",
        runtime="1h",
        nodes=1,
        threads=1,
    shell:
        'python3 {input.script} ' + ' '.join(
        f'--{arg}={{wildcards.{arg}}}'
        for arg in ('model','data','samples')
        ) + ' --n_starts={N_STARTS}'

rule evaluate_regressors:
    input:
        script='evaluate_regressors.py',
        data=rules.process_data.output.datafiles  # wait for download and processing
    output:
        csv=[
            EVALUATION_REGRESSOR.format_map(SafeDict(dataset=dataset,mode=mode, context=context))
            for dataset, mode, context in itt.product(
                ['train', 'test'],
                ['linreg', 'lasso', 'elasticnet'],
                [context for context in CONTEXT_SET]
            )
        ]
    wildcard_constraints:
        model='\w+',
        data=r'[\w\.]+',
        samples='[0-9]+_[0-9]+',
    retries: 1
    resources:
        mem="8GB",
        runtime="1h",
        nodes=1,
        threads=1,
    shell:
        'python3 {input.script} ' + ' '.join(
        f'--{arg}={{wildcards.{arg}}}'
        for arg in ('model','data','samples')
        ) + ' --n_starts={N_STARTS}'

rule evaluate_all:
    input:
        script='evaluate_all.py',
        training=[
            y
            for x in rules.evaluate_training.output.csv
            for context, features in CONTEXTS_FEATURES
            for encoder_layer_sizes, inflater_layer_sizes, linear_benchmark in NETWORK_LAYOUT
            for y in expand(
                x.format_map(
                    SafeDict(
                        context=context,
                        features=features,
                        encoder_layer_sizes=encoder_layer_sizes,
                        inflater_layer_sizes=inflater_layer_sizes,
                        linear_benchmark=linear_benchmark
                    )
                ),
                model='{model}',data='{data}',
                features_transform=FEATURES_TRANSFORM,
                samples=SPLITS,
                pretrain=PRETRAIN,
                n_hidden=LATENT_DIMS,
                use_layer_bias=USE_BIAS,
                nn_init_fn=NN_INIT_FN,
                reconstruct=RECONSTRUCT,
                activation_fn_name=ACTIVATION_FNS,
                optimiser=OPTIMISERS,
                orth_reg_strategy=ORTH_REG_STRATEGIES,
                l1reg_inflate=ALPHAS,
                oreg_inflate=BETAS,
                l1reg_encode=GAMMAS,
                oreg_encode=DELTAS,
                recon_loss=EPSILONS,
                symm_reg=ZETAS,
                max_lrate=MAX_LEARNING_RATES,
                lrate_span=LEARNING_RATE_SPANS,
                lrate_decay=LEARNING_RATE_DECAYS,
                warmup_fct=WARMUP_FCTS,
                opt_steps=OPT_STEPS,
                opt_mult=OPT_MULT,
                use_simple_linear_schedule=LINEAR_SCHEDULE,
                use_early_stopping=USE_EARLY_STOP,  # patience and min_improvement imported in `train.py`
                job=STARTS,
                drop_reg_after_pretrain=DROP_REG_POST_PRETRAIN,
                sparsity_threshold=SPARSITY_THRESHOLD,
            )
        ],
        reference=expand(
            rules.evaluate_references.output.csv,
            model='{model}',data='{data}',samples=SPLITS,
        )+expand(
            rules.evaluate_regressors.output.csv,
            model='{model}',data='{data}',samples=SPLITS,
        )
    output:
        plot=[
            EVALUATE_ALL.format_map(SafeDict(group=group))
            for group in (
                'n_hidden',
                'reconstruct', 'activation_fn_name',
                'orth_reg_strategy',
                'l1reg_encode', 'l1reg_inflate', 'oreg_encode', 'oreg_inflate', 'recon_loss', 'symm_reg',
                'heatmaps_n_hidden_pairwise',
                'volcano_plot_stat_test',
            )
        ],
        csv=[
            EVALUATE_ALL_CSVS.format_map(SafeDict(filename=filename))
            for filename in (
                'evaluate_all',
                'stat_tests_all',
            )
        ]
    wildcard_constraints:
        model='\w+',
        data=r'[\w\.]+',
        samples='[0-9]+_[0-9]+',
    resources:
        mem="16GB",
        runtime="90m",
        nodes=1,
        threads=1,
    shell:
        'python3 {input.script} ' + ' '.join(
            f'--{arg}={{wildcards.{arg}}}'
            for arg in ('model', 'data')
        )

rule train_and_evaluate:
    input:
         evaluation=expand(
             rules.evaluate_all.output.plot,
             model=PATHWAYS, data=DATASETS, samples=SPLITS
         ),

ruleorder: pretrain_average_model > pretrain_per_sample
