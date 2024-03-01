import os
import itertools as itt
from pathlib import Path

from common import (
    PER_SAMPLE_OUTFILE_PARS, TRAINING_OUTFILE_RESULTS,
    COLLECTED_TRAINING_RESULTS, per_sample_pretraining_train, per_sample_pretraining_test, tpl_petab_file,
    EVALUATION_TRAINING, EVALUATE_ALL, EVALUATION_REFERENCE, EVALUATION_REFERENCE_REG,
    MEASUREMENTS_FILE_RW, FEATURES_OUTFILE
)
from training_configuration import ORTH_REG_STRATEGIES, ALPHAS, BETAS, GAMMAS, DELTAS, LATENT_DIMS, PATHWAYS, DATASETS, SPLITS, PRETRAIN, CONTEXTS_FEATURES

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

rule estimate_parameters:
    input:
        script='train.py',
        encoder=mencoder_dir / 'encoder.py',
        training=mencoder_dir / 'training.py',
        autoencoder=mencoder_dir /'autoencoder.py',
        data=rules.process_data.output.datafiles,
        data_rw=rules.reweight_data.output.data,
        model=rules.compile_mechanistic_model.output.model,
        features=rules.select_features.output.data,
        pretrain_per_sample=per_sample_pretraining_train,
    output:
        result=TRAINING_OUTFILE_RESULTS
    wildcard_constraints:
        model='\w+',
        data='[\w\.]+',
        context='\w+',
        n_hidden='[0-9]+',
        job='[0-9]+',
        samples='[0-9]+_[0-9]+',
        l1reg_inflate='[0-9\.]+',
        oreg_inflate='[0-9\.]+',
        oreg_encode='[0-9\.]+',
    retries: 1
    resources:
        mem="1GB",
        runtime="24h",
        nodes=1,
        threads=2,
    shell:
        'python3 {input.script} ' + ' '.join(
            f'--{arg}={{wildcards.{arg}}}'
            for arg in ('model', 'data', 'context', 'samples', 'n_hidden', 'orth_reg_strategy',
                        'l1reg_inflate', 'oreg_inflate', 'l1reg_encode', 'oreg_encode',
                        'job', 'features')
        ) + ' --threads={threads}'

rule collect_estimation_results:
    input:
        script='collect_estimation.py',
        trace=[
            TRAINING_OUTFILE_RESULTS.format_map(SafeDict(job=job))
            for job in STARTS
        ]
    output:
        result=COLLECTED_TRAINING_RESULTS
    wildcard_constraints:
        model='\w+',
        data='[\w\.]+',
        context='\w+',
        n_hidden='[0-9]+',
        job='[0-9]+',
        samples='[0-9]+_[0-9]+',
        l1reg_inflate='[0-9\.]+',
    resources:
        mem="8GB",
        runtime="1h",
        nodes=1,
        threads=1,
    shell:
        'python3 {input.script} ' + ' '.join(
            f'--{arg}={{wildcards.{arg}}}'
            for arg in ('model', 'data', 'context', 'samples', 'n_hidden', 'orth_reg_strategy',
                        'l1reg_inflate', 'oreg_inflate', 'l1reg_encode', 'oreg_encode',
                        'features')
        ) + ' --n_starts={N_STARTS}'

rule evaluate_references:
    input:
        script='evaluate_reference.py',
        pretrain_per_sample=per_sample_pretraining_test,
        pretrain_average=rules.pretrain_average_model.output.pretraining
    output:
        csv=[
            EVALUATION_REFERENCE.format_map(SafeDict(dataset=dataset, mode=mode))
            for dataset, mode in itt.product(['train', 'test'], ['per_sample', 'average'])
        ] + [
            EVALUATION_REFERENCE_REG.format_map(SafeDict(dataset=dataset,mode=mode, context=context))
            for dataset, mode, context in itt.product(['train', 'test'],
                ['linreg', 'lasso', 'elasticnet'],
                [context for context, _ in CONTEXTS_FEATURES])
        ]
    wildcard_constraints:
        model='\w+',
        data=r'[\w\.]+',
        samples='[0-9]+_[0-9]+',
    retries: 1
    resources:
        mem="1GB",
        runtime="1h",
        nodes=1,
        threads=1,
    shell:
        'python3 {input.script} ' + ' '.join(
        f'--{arg}={{wildcards.{arg}}}'
        for arg in ('model','data','samples')
        ) + ' --n_starts={N_STARTS}'

rule evaluate_training:
    input:
        script='evaluate_training.py',
        training=rules.estimate_parameters.output.result
    output:
        csv=[
            EVALUATION_TRAINING.format_map(SafeDict(dataset=dataset))
            for dataset in ['train', 'test']
        ]
    wildcard_constraints:
        model='\w+',
        data=r'[\w\.]+',
        samples='[0-9]+_[0-9]+',
        context='\w+',
        n_hidden='[0-9]+',
        job='[0-9]+',
        orth_reg_strategy='\w+',
        l1reg_inflate='[0-9\.]+',
        oreg_inflate='[0-9\.]+',
        oreg_encode='[0-9\.]+',
        pretrain='True|False',
    resources:
        mem="1GB",
        runtime="1h",
        nodes=1,
        threads=1,
    shell:
        'python3 {input.script} ' + ' '.join(
            f'--{arg}={{wildcards.{arg}}}'
            for arg in ('model', 'data', 'context', 'samples', 'n_hidden', 'job', 'orth_reg_strategy',
                        'l1reg_inflate', 'oreg_inflate', 'l1reg_encode', 'oreg_encode',
                        'features')
        )

rule evaluate_all:
    input:
        script='evaluate_all.py',
        training=[
            y
            for x in rules.evaluate_training.output.csv
            for context, features in CONTEXTS_FEATURES
            for y in expand(
                x.format_map(SafeDict(context=context, features=features)),
                model='{model}',data='{data}',
                orth_reg_strategy=ORTH_REG_STRATEGIES,
                job=STARTS,
                l1reg_inflate=ALPHAS,
                oreg_inflate=BETAS,
                l1reg_encode=GAMMAS,
                oreg_encode=DELTAS,
                n_hidden=LATENT_DIMS,
                samples=SPLITS,
                pretrain=PRETRAIN
            )
        ],
        reference=expand(
            rules.evaluate_references.output.csv,
            model='{model}',data='{data}',samples=SPLITS,
        )
    output:
        plot=[
            EVALUATE_ALL.format_map(SafeDict(group=group))
            for group in ('orth_reg_strategy', 'l1reg_encode', 'l1reg_inflate', 'oreg_encode', 'oreg_inflate')
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
