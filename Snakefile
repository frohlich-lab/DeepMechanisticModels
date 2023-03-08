import os
from pathlib import Path

from common import (
    PER_SAMPLE_OUTFILE_PARS, CROSS_SAMPLE_OUTFILE_PARS, CROSS_SAMPLE_OUTFILE_RESULTS, TRAINING_OUTFILE_RESULTS,
    COLLECTED_TRAINING_RESULTS, per_sample_pretraining_train, per_sample_pretraining_test, tpl_petab_file,
    tpl_evaluation_file, EVALUATION_TRAINING, EVALUATE_ALL
)
from training_configuration import ALPHAS, LATENT_DIMS, CONTEXTS, PATHWAYS, DATASETS, SPLITS

basedir = Path(os.getcwd())
mencoder_dir = basedir / 'mEncoder'
cytof_dir = basedir / 'cytof'


N_STARTS = int(config.get("num_starts", "10"))
STARTS = [str(i) for i in range(N_STARTS)]

singularity: config.get("singularity", r"docker://fabfroehlich/generic_parameter_estimation:main")

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
        nodes="1",
        cpus_per_task="1",
    shell:
        'python3 {input.script} {wildcards.model} {wildcards.data}'

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
        nodes="1",
        cpus_per_task="1",
    shell:
        'python3 {input.script} {wildcards.model} {wildcards.data}'


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
        nodes="1",
        cpus_per_task="1",
    shell:
        'python3 {input.script} {wildcards.model} {wildcards.data} '
        '{wildcards.sample}'


rule pretrain_average_model:
    input:
        script='pretrain_average.py',
        pretraining_code=mencoder_dir / 'pretraining.py',
        model=rules.compile_mechanistic_model.output.model,
        data=rules.process_data.output.datafiles
    output:
        pretraining=PER_SAMPLE_OUTFILE_PARS.format(model='{model}', data='{data}', sample='model_average_{samples}')
    wildcard_constraints:
        model='\w+',
        data='[\w\.]+',
        samples='[0-9]+_[0-9]+',
    resources:
        mem="1GB",
        runtime="6h",
        nodes="1",
        cpus_per_task="1",
    shell:
        'python3 {input.script} {wildcards.model} {wildcards.data} {wildcards.samples}'


rule pretrain_cross_sample:
    input:
        script='pretrain_cross_samples.py',
        pretraining=os.path.join('mEncoder', 'pretraining.py'),
        autoencoder=os.path.join('mEncoder', 'autoencoder.py'),
        bounds=os.path.join('mEncoder', '__init__.py'),
        pretrain_per_sample=per_sample_pretraining_train,
        model=rules.compile_mechanistic_model.output.model,
        data=rules.process_data.output.datafiles
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
    resources:
        mem="1.5GB",
        runtime="6h",
        nodes="1",
        cpus_per_task="1",
    shell:
        'python3 {input.script} {wildcards.model} {wildcards.data} {wildcards.context} '
        '{wildcards.samples} {wildcards.n_hidden} {wildcards.alpha} {wildcards.job}'


rule estimate_parameters:
    input:
        script='train.py',
        encoder=mencoder_dir / 'encoder.py',
        training=mencoder_dir / 'training.py',
        autoencoder=mencoder_dir /'autoencoder.py',
        data=rules.process_data.output.datafiles,
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
        alpha='[0-9\.]+'
    resources:
        mem="1GB",
        runtime="6h",
        nodes="1",
        cpus_per_task="1",
    shell:
        'python3 {input.script} {wildcards.model} {wildcards.data} {wildcards.context} '
        '{wildcards.samples} {wildcards.n_hidden} {wildcards.alpha} {wildcards.job}'

rule collect_estimation_results:
    input:
        script='collect_estimation.py',
        #trace=expand(
        #    TRAINING_OUTFILE_RESULTS.format(
        #        context='{{context}}', samples='{{samples}}', model='{{model}}', data='{{data}}',
        #        n_hidden='{{n_hidden}}', alpha='{{alpha}}', job='{job}'
        #    ), job=STARTS
        #)
    output:
        result=COLLECTED_TRAINING_RESULTS
    wildcard_constraints:
        model='\w+',
        data='[\w\.]+',
        context='\w+',
        n_hidden='[0-9]+',
        job='[0-9]+',
        samples='[0-9]+_[0-9]+',
        alpha='[0-9\.]+'
    resources:
        mem="8GB",
        runtime="2h",
        nodes="1",
        cpus_per_task="1",
    shell:
        'python3 {input.script} {wildcards.model} {wildcards.data} {wildcards.context} '
        '{wildcards.samples} {wildcards.n_hidden} {wildcards.alpha} {N_STARTS}'

rule evaluate_pretraining:
    input:
        script='evaluate_pretraining.py',
        cross_sample=expand(
            rules.pretrain_cross_sample.output.results,
            model='{model}', data='{data}', context=CONTEXTS,
            n_hidden=LATENT_DIMS, alpha=ALPHAS, samples='{samples}', job=STARTS
        ),
        pretrain_per_sample=per_sample_pretraining_test,
        pretrain_average=rules.pretrain_average_model.output.pretraining
    output:
        csv=expand(
            tpl_evaluation_file,
            model='{model}', data='{data}', samples='{samples}',
            mode=['per_sample', 'cross_sample', 'average'],
            dataset=['train', 'test']
        )
    wildcard_constraints:
        model='\w+',
        data=r'[\w\.]+',
        samples='[0-9]+_[0-9]+',
    resources:
        mem="20GB",
        runtime="24h",
        nodes="1",
        cpus_per_task="1",
    shell:
        'python3 {input.script} {wildcards.model} {wildcards.data} {wildcards.samples} {N_STARTS}'

rule evaluate_training:
    input:
        script='evaluate_training.py',
        cross_sample=expand(
            rules.collect_estimation_results.output.result,
            model='{model}', data='{data}', context=CONTEXTS,
            n_hidden=LATENT_DIMS, alpha=ALPHAS, samples='{samples}',
        ),
    output:
        csv=expand(
            EVALUATION_TRAINING,
            model='{model}',data='{data}', samples='{samples}',
            dataset=['train', 'test']
        )
    wildcard_constraints:
        model='\w+',
        data=r'[\w\.]+',
        samples='[0-9]+_[0-9]+',
    resources:
        mem="8GB",
        runtime="4h",
        nodes="1",
        cpus_per_task="1",
    shell:
        'python3 {input.script} {wildcards.model} {wildcards.data} {wildcards.samples}'

rule evaluate_all:
    input:
        script='evaluate_all.py',
        pretraining=expand(
            rules.evaluate_pretraining.output.csv,
            model='{model}',data='{data}', samples=SPLITS
        )
        #training=rules.evaluate_training.output.csv,
    output:
        plot=expand(
            EVALUATE_ALL,
            model='{model}', data='{data}',
            group=('observable', 'time', 'condition', 'sample', 'all')
        )
    wildcard_constraints:
        model='\w+',
        data=r'[\w\.]+',
        samples='[0-9]+_[0-9]+',
    resources:
        mem="4GB",
        runtime="30m",
        nodes="1",
        cpus_per_task="1",
    shell:
        'python3 {input.script} {wildcards.model} {wildcards.data}'

rule train_and_evaluate:
    input:
         evaluation=expand(
             rules.evaluate_all.output.plot,
             model=PATHWAYS, data=DATASETS, samples=SPLITS
         ),

ruleorder: pretrain_average_model > pretrain_per_sample
