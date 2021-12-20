# Mechanistic Autoencoders

Mechanistic Autoencoders combine a machine learning component, which learns 
a latent embedding of the baseline, with a mechanistic model component,
which integrates prior knowledge and perturbation data. The machine 
learning component is implemented using [aesara](https://aesara.readthedocs.io)
and the mechanistic component is implemented using [pysb](https://pysb.org).

Training is implemented as [snakemake](https://snakemake.readthedocs.io/en/stable) workflow consisting of three steps:

1) Training of the mechanistic model on individual samples (implemented in 
   `pretrain_per_sample.py`).
2) Training of the link between latent embedding and mechanistic model 
   based on a PCA initialization of the latent (implemented in 
   `pretrain_cross_samples.py`) 
3) Training of the full model (implemented in `run_estimation.py`).

Hyperparameters can be specified via `trainig_configuration.py`.