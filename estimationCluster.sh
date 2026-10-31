#!/bin/bash
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -t 2-00:00
#SBATCH -p ncpu
#SBATCH --mem=8GB
#SBATCH -o snakelog.out
#SBATCH -e snakelog.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fabian.frohlich@crick.ac.uk

ml Python/3.10.8-GCCcore-12.2.0-bare
ml Singularity/3.6.4

source ./venv/bin/activate

export SLURM_MPI_TYPE=none
export WANDB_MODE=online

# Target and extra snakemake options come from the command line, e.g.
#   sbatch estimationCluster.sh                                  # train_and_evaluate
#   sbatch estimationCluster.sh train_and_evaluate --config figure=figure2
#   sbatch estimationCluster.sh paper_results                    # all paper figures
TARGET=${1:-train_and_evaluate}
shift $(( $# > 0 ))

snakemake --unlock
snakemake "$TARGET" --local-cores 1 -j 400 --config num_starts=5 \
    --use-singularity --slurm --default-resources slurm_account=froehlichf slurm_partition=ncpu \
    --rerun-incomplete "$@"
