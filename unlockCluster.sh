#!/bin/bash
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -t 3-00:00
#SBATCH -p ncpu
#SBATCH --mem=8GB
#SBATCH -o snakelog.out
#SBATCH -e snakelog.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=giacomo.fabrini@crick.ac.uk

set -e

ml Python/3.10.8-GCCcore-12.2.0-bare
ml Singularity/3.6.4

source ./venv/bin/activate

snakemake --unlock