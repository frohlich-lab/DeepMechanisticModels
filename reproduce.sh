#!/usr/bin/env bash
# Reproduce the paper, from data download to figures (see README.md).
#
#   ./reproduce.sh all     [snakemake options]  download -> training -> figures
#   ./reproduce.sh results [snakemake options]  same, stopping before figures
#   ./reproduce.sh figures [snakemake options]  figures only, from precomputed
#                                               evaluation outputs in eval/
#
# Options are passed to snakemake, e.g. --cores 32 (default: --cores 1), or
# --config figures=figure2,figure3 to restrict to some figure configurations.
# On the Crick cluster, `sbatch estimationCluster.sh paper_results` replaces
# `results`.
set -euo pipefail
cd "$(dirname "$0")"

case "${1:-}" in
    all) target=paper ;;
    results) target=paper_results ;;
    figures) target=figures ;;
    *) sed -n '2,12p' "$0" | cut -c3-; exit 2 ;;
esac
shift

# snakemake needs a core count
[[ " ${*:-} " =~ \ (-c|--cores|-j|--jobs)[\ =0-9] ]] || set -- --cores 1 "$@"

# W&B logging is optional; WANDB_MODE=online (with a W&B account) enables it
export WANDB_MODE=${WANDB_MODE:-disabled}

if [[ -z ${SYNAPSE_AUTH_TOKEN:-} && ! -f ~/.synapseConfig ]]; then
    echo "No Synapse credentials: set SYNAPSE_AUTH_TOKEN or create" \
        "~/.synapseConfig (README.md, 'Data access')." >&2
    exit 1
fi
if [[ $target != figures && -z ${BNGPATH:-} ]]; then
    echo "Warning: BNGPATH is unset; pysb needs BioNetGen unless jobs run in" \
        "the Singularity image (README.md, 'Prerequisites')." >&2
fi

exec snakemake "$target" "$@"
