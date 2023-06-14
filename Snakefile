import os
import itertools as itt
from pathlib import Path

from common import (
    PER_SAMPLE_OUTFILE_PARS, CROSS_SAMPLE_OUTFILE_PARS, CROSS_SAMPLE_OUTFILE_RESULTS, TRAINING_OUTFILE_RESULTS,
    COLLECTED_TRAINING_RESULTS, per_sample_pretraining_train, per_sample_pretraining_test, tpl_petab_file,
    EVALUATION_PRETRAINING, EVALUATION_TRAINING, EVALUATE_ALL, EVALUATION_REFERENCE,
    MEASUREMENTS_FILE_RW
)
from training_configuration import ALPHAS, LATENT_DIMS, CONTEXTS, PATHWAYS, DATASETS, SPLITS, PRETRAIN

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
        cpus_per_task=1,
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
        cpus_per_task=1,
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
        cpus_per_task=1,
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
        cpus_per_task=1,
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
        data=rules.process_data.output.datafiles
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
        cpus_per_task=1,
    shell:
        'python3 {input.script} ' + ' '.join(
            f'--{arg}={{wildcards.{arg}}}'
            for arg in ('model', 'data', 'samples')
        )


rule pretrain_cross_sample:
    input:
        script='pretrain_cross_samples.py',
        pretraining=mencoder_dir / 'pretraining.py',
        autoencoder=mencoder_dir / 'autoencoder.py',
        bounds=mencoder_dir / '__init__.py',
        pretrain_per_sample=per_sample_pretraining_train,
        model=rules.compile_mechanistic_model.output.model,
        data=rules.process_data.output.datafiles,
        data_rw=rules.reweight_data.output.data
    output:
        pars=CROSS_SAMPLE_OUTFILE_PARS,
        results=CROSS_SAMPLE_OUTFILE_RESULTS
    wildcard_constraints:
        model='\w+',
        data='[\w\.]+',
        context='\w+',
        n_hidden='[0-9]+',
        job='[0-9]+',
        samples='[0-9]+_[0-9]+',
        alpha='[0-9\.]+',
        pretrain='True|False',
    resources:
        mem="1.5GB",
        runtime="12h",
        nodes=1,
        cpus_per_task=1,
    retries: 1
    shell:
        'python3 {input.script} ' + ' '.join(
            f'--{arg}={{wildcards.{arg}}}'
            for arg in ('model', 'data', 'context', 'samples', 'n_hidden', 'alpha', 'job', 'pretrain')
        )


rule estimate_parameters:
    input:
        script='train.py',
        encoder=mencoder_dir / 'encoder.py',
        training=mencoder_dir / 'training.py',
        autoencoder=mencoder_dir /'autoencoder.py',
        data=rules.process_data.output.datafiles,
        data_rw=rules.reweight_data.output.data,
        pretrain_inflate=rules.pretrain_cross_sample.output.pars,
        model=rules.compile_mechanistic_model.output.model,
    output:
        result=TRAINING_OUTFILE_RESULTS
    wildcard_constraints:
        model='\w+',
        data='[\w\.]+',
        context='\w+',
        n_hidden='[0-9]+',
        job='[0-9]+',
        samples='[0-9]+_[0-9]+',
        alpha='[0-9\.]+',
        pretrain='True|False',
    retries: 1
    resources:
        mem="1GB",
        runtime="12h",
        nodes=1,
        cpus_per_task=1,
    shell:
        'python3 {input.script} ' + ' '.join(
            f'--{arg}={{wildcards.{arg}}}'
            for arg in ('model', 'data', 'context', 'samples', 'n_hidden', 'alpha', 'job', 'pretrain')
        )

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
        alpha='[0-9\.]+',
        pretrain='True|False',
    resources:
        mem="8GB",
        runtime="1h",
        nodes=1,
        cpus_per_task=1,
    shell:
        'python3 {input.script} ' + ' '.join(
            f'--{arg}={{wildcards.{arg}}}'
            for arg in ('model', 'data', 'context', 'samples', 'n_hidden', 'alpha', 'pretrain')
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
        cpus_per_task=1,
    shell:
        'python3 {input.script} ' + ' '.join(
        f'--{arg}={{wildcards.{arg}}}'
        for arg in ('model','data','samples',)
        ) + ' --n_starts={N_STARTS}'

rule evaluate_pretraining:
    input:
        script='evaluate_pretraining.py',
        cross_sample=expand(
            rules.pretrain_cross_sample.output.results,
            model='{model}', data='{data}', context='{context}', pretrain='{pretrain}',
            n_hidden='{n_hidden}', alpha='{alpha}', samples='{samples}', job=STARTS
        ),
    output:
        csv=[
            EVALUATION_PRETRAINING.format_map(SafeDict(dataset=dataset, mode='cross_sample'))
            for dataset in ['train', 'test']
        ]
    wildcard_constraints:
        model='\w+',
        data=r'[\w\.]+',
        samples='[0-9]+_[0-9]+',
        context='\w+',
        n_hidden='[0-9]+',
        alpha='[0-9\.]+',
        pretrain='True|False',
    resources:
        mem="1GB",
        runtime="1h",
        nodes=1,
        cpus_per_task=1,
    shell:
        'python3 {input.script} ' + ' '.join(
            f'--{arg}={{wildcards.{arg}}}'
            for arg in ('model', 'data', 'context', 'samples', 'n_hidden', 'alpha', 'pretrain')
        ) + ' --n_starts={N_STARTS}'

rule evaluate_training:
    input:
        script='evaluate_training.py',
        training=rules.collect_estimation_results.output.result
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
        alpha='[0-9\.]+',
        pretrain='True|False',
    resources:
        mem="1GB",
        runtime="1h",
        nodes=1,
        cpus_per_task=1,
    shell:
        'python3 {input.script} ' + ' '.join(
            f'--{arg}={{wildcards.{arg}}}'
            for arg in ('model', 'data', 'context', 'samples', 'n_hidden', 'alpha', 'pretrain')
        )

rule evaluate_all:
    input:
        script='evaluate_all.py',
        pretraining=expand(
            rules.evaluate_pretraining.output.csv,
            model='{model}',data='{data}',alpha=ALPHAS,n_hidden=LATENT_DIMS,context=CONTEXTS,samples=SPLITS,
            pretrain=PRETRAIN
        ),
        training=expand(
            rules.evaluate_training.output.csv,
            model='{model}',data='{data}',alpha=ALPHAS,n_hidden=LATENT_DIMS,context=CONTEXTS,samples=SPLITS,
            pretrain=PRETRAIN
        ),
        reference=expand(
            rules.evaluate_references.output.csv,
            model='{model}',data='{data}',samples=SPLITS,
        )
    output:
        plot=[
            EVALUATE_ALL.format_map(SafeDict(group=group))
            for group in ('observable', 'time', 'condition', 'sample', 'all')
        ]
    wildcard_constraints:
        model='\w+',
        data=r'[\w\.]+',
        samples='[0-9]+_[0-9]+',
    resources:
        mem="4GB",
        runtime="30m",
        nodes=1,
        cpus_per_task=1,
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
