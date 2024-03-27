#!/bin/sh
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -t 3-00:00
#SBATCH -p cpu
#SBATCH --mem=8GB
#SBATCH -o snakelog.out
#SBATCH -e snakelog.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=giacomo.fabrini@crick.ac.uk

ml Python/3.10.8-GCCcore-12.2.0-bare
ml Singularity/3.6.4

source ./venv/bin/activate

snakemake train_and_evaluate --local-cores 1 -j 2000 --config num_starts=10 \
    --use-singularity --slurm --default-resources slurm_account=u_froehlichf slurm_partition=cpu \
    --rerun-incomplete --max-jobs-per-second=1 --max-status-checks-per-second=0.1