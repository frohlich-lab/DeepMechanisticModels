# Deep Mechanistic Models

Deep Mechanistic Models combine a machine learning component, which learns
a latent embedding of the baseline, with a mechanistic model component,
which integrates prior knowledge and perturbation data. The machine
learning component is implemented using [jax](https://jax.readthedocs.io/en/latest/)
and the mechanistic component is implemented using [pysb](https://pysb.org).

Training is implemented as [snakemake](https://snakemake.readthedocs.io/en/stable) workflow consisting of three steps:

1) Training of the mechanistic model on individual samples (implemented in
   `pretrain_per_sample.py`).
2) Training of a single average mechanistic model shared across all samples
   of the training split (implemented in `pretrain_average.py`).
3) Training of the full model, initialised from the average model of step 2
   (implemented in `train.py`).

Hyperparameters can be specified via `training_configuration.py`.

Evaluation of the trained models is described in `evaluation_workflow.md`.

## Paper figures

The notebooks under `figures_paper/` produce the paper figures. To check they
all still run after a refactor:

```
./venv/bin/python run_notebooks.py --list   # show what would run
./venv/bin/python run_notebooks.py          # run all 12
```

Figures are redirected into `notebook_output/figures/`, prefixed with the
notebook name, so the repository root stays clean. This is local-only — the
notebooks need the pipeline's outputs (`eval/`, `res/`, `pretrain/`) and
external API access, so it cannot run in CI.
