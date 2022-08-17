import os

from mEncoder import (
    basedir, data_dir, pretrain_dir, fig_dir, results_dir,
    PER_SAMPLE_OUTFILE_TEMP, COLLECTED_ESTIMATION_OUTFILE_TEMP,
    ESTIMATION_OUTFILE_TEMP
)
from process_data import pretraining_samples_fun
from training_configuration import ALPHAS, HIDDEN_LAYERS

mencoder_dir = basedir / 'mEncoder'

PATHWAYS = ['EGFR_MAPK']
DATASETS = ['dream_cytof']
SPLITS = ['0_5',]

STARTS = [str(i) for i in range(int(config.get("num_starts", "10")))]

singularity: config.get("singularity", r"docker://fabfroehlich/generic_parameter_estimation:main")

rule process_data:
    input:
        script='process_data.py',
        data_code=mencoder_dir / 'generate_data.py',
        model_code=mencoder_dir / 'mechanistic_model.py',
        pathway=basedir / 'pathways' / 'pw_{model}.py',
        pathways=mencoder_dir / 'pathways.py',
    output:
        datafiles=expand(
            str(data_dir / '{{data}}__{{model}}__{file}.tsv'),
            file=['conditions', 'measurements', 'observables']
        )
    wildcard_constraints:
        model='[\w_]+',
        data='[\w]+',
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
        model= basedir / 'amici_models' / '{model}_{data}__{model}_petab' /
            '{model}' / '{model}.py',
    wildcard_constraints:
        model='[\w_]+',
        data='[\w]+',
    shell:
        'python3 {input.script} {wildcards.model} {wildcards.data}'


rule pretrain_per_sample:
    input:
        script='pretrain_per_sample.py',
        pretraining_code=mencoder_dir / 'pretraining.py',
        model=rules.compile_mechanistic_model.output.model,
        data=rules.process_data.output.datafiles,
    output:
        pretraining=pretrain_dir / '{model}' / '{data}' / PER_SAMPLE_OUTFILE_TEMP
    wildcard_constraints:
        model='[\w_]+',
        data='[\w]+',
        sample='[\w_]+',
    shell:
        'python3 {input.script} {wildcards.model} {wildcards.data} '
        '{wildcards.sample}'


rule pretrain_cross_sample:
    input:
        script='pretrain_cross_samples.py',
        pretraining=os.path.join('mEncoder', 'pretraining.py'),
        autoencoder=os.path.join('mEncoder', 'autoencoder.py'),
        bounds=os.path.join('mEncoder', '__init__.py'),
        pretrain_per_sample=pretraining_samples_fun,
        model=rules.compile_mechanistic_model.output.model,
        data=rules.process_data.output.datafiles,
    output:
        pretraining=pretrain_dir / '{model}' / '{data}' / ESTIMATION_OUTFILE_TEMP
    wildcard_constraints:
        model='[\w_]+',
        data='[\w]+',
        n_hidden='[0-9]+',
        job='[0-9]+',
        samples='[0-9]+_[0-9]+',
        alpha='[0-9\.]+',
    shell:
        'AESARA_FLAGS="compiledir=./aesara/{wildcards.model}_{wildcards.data}_{wildcards.samples}_{wildcards.n_hidden}_{wildcards.job}" '
        'python3 {input.script} {wildcards.model} {wildcards.data} '
        '{wildcards.samples} {wildcards.n_hidden} {wildcards.alpha} '
        '{wildcards.job}'


rule estimate_parameters:
    input:
        script='run_estimation.py',
        encoder=mencoder_dir / 'encoder.py',
        training=mencoder_dir / 'training.py',
        autoencoder=mencoder_dir /'autoencoder.py',
        dataset=rules.process_data.output.datafiles,
        pretrain_inflate=rules.pretrain_cross_sample.output.pretraining,
        model=rules.compile_mechanistic_model.output.model,
    output:
        result=results_dir / '{model}' / '{data}' / ESTIMATION_OUTFILE_TEMP
    wildcard_constraints:
        model='[\w_]+',
        data='[\w]+',
        n_hidden='[0-9]+',
        job='[0-9]+',
        samples='[0-9]+_[0-9]+',
        alpha='[0-9\.]+'
    shell:
        'AESARA_FLAGS="compiledir=./aesara/{wildcards.model}_{wildcards.data}_{wildcards.samples}_{wildcards.n_hidden}__{wildcards.alpha}_{wildcards.job}" '
        'python3 {input.script} {wildcards.model} {wildcards.data} '
        '{wildcards.samples} {wildcards.n_hidden} {wildcards.alpha} '
        '{wildcards.job}'

rule collect_estimation_results:
    input:
        script='collect_estimation.py',
        trace=expand(os.path.join(
            results_dir, '{{model}}', '{{data}}',
            ESTIMATION_OUTFILE_TEMP.format(
                model='{{model}}', data='{{data}}', samples='{{samples}}',
                n_hidden='{{n_hidden}}', alpha='{{alpha}}', job='{job}'
            )
        ), job=STARTS)
    output:
        result=results_dir / '{model}' / '{data}' / COLLECTED_ESTIMATION_OUTFILE_TEMP
    wildcard_constraints:
        model='[\w_]+',
        data='[\w_]+',
        n_hidden='[0-9]+',
        job='[0-9]+',
        samples='[0-9]+_[0-9]+',
        alpha='[0-9\.]+'
    shell:
        'python3 {input.script} {wildcards.model} {wildcards.data} '
        '{wildcards.samples} {wildcards.n_hidden} {wildcards.alpha}'

rule visualize_estimation_results:
    input:
        script='visualize_results.py',
        estimation=rules.collect_estimation_results.output.result
    output:
        plots=expand(str(
            fig_dir / '{{model}}' / '{{data}}' /
            '__'.join(['{{samples}}', '{{n_hidden}}', '{{alpha}}'])
            + '__{plot}.pdf'
        ), plot=['waterfall', 'embedding'])
    wildcard_constraints:
        model='[\w_]+',
        data='[\w_]+',
        optimzer='[\w-]+',
        n_hidden='[0-9]+',
        job='[0-9]+',
        samples='[0-9]+_[0-9]+',
        alpha='[0-9\.]+'
    shell:
        'python3 {input.script} {wildcards.model} {wildcards.data} '
        '{wildcards.samples} {wildcards.n_hidden} {wildcards.alpha}'

rule evaluate_pretraining:
    input:
        script='evaluate_pretraining.py',
        cross_sample=expand(
            rules.pretrain_cross_sample.output.pretraining,
            model='{model}', data='{data}', n_hidden=HIDDEN_LAYERS,
            alpha=ALPHAS, samples=SPLITS
        )
    output:
        plot=fig_dir / '{{model}}' / '{{data}}' / '{samples}_evaluate_pretraining.pdf',
        csv=fig_dir / '{{model}}' / '{{data}}' / '{samples}_evaluate_pretraining.csv'
    wildcard_constraints:
        model='[\w_]+',
        data='[\w_]+',
        optimzer='[\w-]+',
        n_hidden='[0-9]+',
        job='[0-9]+',
        samples='[0-9]+_[0-9]+',
        alpha='[0-9\.]+'
    shell:
        'python3 {input.script} {wildcards.model} {wildcards.data} '
        '{wildcards.samples}'

rule collect_estimation:
    input:
         expand(
             rules.collect_estimation_results.output.result,
             model=PATHWAYS, data=DATASETS, n_hidden=HIDDEN_LAYERS,
             alpha=ALPHAS, samples=SPLITS
         )

rule visualize_estimation:
    input:
         training=expand(
             rules.visualize_estimation_results.output.plots,
             model=PATHWAYS, data=DATASETS, n_hidden=HIDDEN_LAYERS,
             alpha=ALPHAS, samples=SPLITS
         ),
         pretraining=expand(
             evaluate_pretraining,
             model=PATHWAYS, data=DATASETS, samples=SPLITS
         ),