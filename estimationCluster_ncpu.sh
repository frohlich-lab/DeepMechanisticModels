#!/bin/bash
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -t 7-00:00
#SBATCH -p ncpu
#SBATCH --mem=8GB
#SBATCH -o snakelog.out
#SBATCH -e snakelog.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=giacomo.fabrini@crick.ac.uk

ml Python/3.10.8-GCCcore-12.2.0-bare
ml Singularity/3.6.4

source ./venv/bin/activate

export SLURM_MPI_TYPE=none
export WANDB_MODE=online

snakemake --unlock
snakemake train_and_evaluate --local-cores 1 -j 300 --config num_starts=5 \
    --use-singularity --slurm --default-resources slurm_account=u_froehlichf slurm_partition=ncpu \
    --rerun-incomplete