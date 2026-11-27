# Deep Mechanistic Models

Deep Mechanistic Models combine a machine learning component, which learns
a latent embedding of the baseline, with a mechanistic model component,
which integrates prior knowledge and perturbation data. The machine
learning component is implemented using [jax](https://jax.readthedocs.io/en/latest/)
and the mechanistic component is implemented using [pysb](https://pysb.org).

This repository contains the code for the paper *Deep Mechanistic Models
reveal pathway-extrinsic drivers of mammary MAPK signalling heterogeneity*
(Fabrini & Fröhlich; preprint
[doi:10.64898/2026.07.30.741759](https://doi.org/10.64898/2026.07.30.741759))
and a workflow that retrieves all input data and regenerates all figures.

## Reproducing the paper

The [snakemake](https://snakemake.readthedocs.io/en/stable) workflow in
`Snakefile` chains every step from data download to figures:

1. **Data download**: `load_data` (Synapse), MOSA embeddings (figshare, in
   `process_data`) and MSigDB gene sets (`download_msigdb`).
2. **Preprocessing**: `process_data` writes PEtab problems,
   `compile_mechanistic_model` compiles them with AMICI.
3. **Feature selection**: `select_features`.
4. **Pretraining**: the mechanistic model on individual cell lines
   (`pretrain_per_sample.py`) and one average model shared across the cell
   lines of a training split (`pretrain_average.py`).
5. **Training**: the full model, initialized from the average model
   (`train.py`, rule `estimate_parameters`).
6. **Evaluation**: `evaluate_training`, `evaluate_references`,
   `evaluate_regressors`, aggregated per figure configuration by
   `evaluate_all` (see `evaluation_workflow.md`).
7. **Figures**: the notebooks in `figures_paper/`, executed by
   `run_notebooks.py`.

Each figure configuration in `training_configuration.py` (`figure1a` ...
`figure6`) defines which models, input contexts, features, cross-validation
splits and hyperparameter scans are trained for one figure.

There are two entry points, both through `reproduce.sh`:

| Command | Snakemake target | Runs |
|---|---|---|
| `./reproduce.sh all` | `paper` | everything, steps 1-7 |
| `./reproduce.sh figures` | `figures` | figures only, from precomputed evaluation outputs in `eval/` |

`./reproduce.sh results` (target `paper_results`) runs steps 1-6 and stops
before the figures. That is the part to run on a cluster.

`reproduce.sh` checks for Synapse credentials, switches W&B logging off unless
`WANDB_MODE` is set, and passes any further options to snakemake:

```
./reproduce.sh all --cores 32
./reproduce.sh all --cores 32 --config figures=figure2,figure3  # only these configurations
./reproduce.sh figures
```

The figures land in `notebook_output/figures/`, prefixed with the notebook
name (see [Paper figures](#paper-figures)).

### 1. Data access

The data come from the Single Cell Signaling in Breast Cancer DREAM Challenge
(Gabor et al., 2021, [doi:10.15252/msb.202110402](https://doi.org/10.15252/msb.202110402)),
hosted on Synapse under [syn20366914](https://www.synapse.org/Synapse:syn20366914).
The Challenge's data use conditions do not permit redistribution, including of
processed data. This repository therefore contains none of it; the workflow
downloads it through the Synapse API instead. One-time setup:

1. **Create a free Synapse account** at <https://www.synapse.org/register>.
2. **Register for the Challenge** on its
   [How to Participate](https://www.synapse.org/Synapse:syn20366914/wiki/594732)
   page. Registering means accepting the Challenge
   [rules and data use conditions](https://www.synapse.org/Synapse:syn20366914/wiki/594736).
   Among them: publications using the data must acknowledge the data
   contributors and cite Gabor et al. (2021), and the data must not be
   redistributed.
3. **Create a personal access token** in Synapse under *Account Settings →
   Personal Access Tokens*, with *View* and *Download* permissions.
4. **Make the token available**, either as an environment variable

   ```
   export SYNAPSE_AUTH_TOKEN=<token>
   ```

   or in `~/.synapseConfig`:

   ```
   [authentication]
   authtoken = <token>
   ```

That is all. `load_data.py` then downloads the CyTOF, proteomics and RNA-seq
data automatically. The other inputs are public and also fetched
automatically: the MOSA latent embeddings (Cai et al., 2024; figshare
[doi:10.6084/m9.figshare.24562765.v1](https://doi.org/10.6084/m9.figshare.24562765.v1))
and the MSigDB C2 canonical pathways (v2025.1.Hs). Cell line annotations
(Cellosaurus, cBioPortal, Cell Model Passports, ClinVar/OncoKB) and the UniProt
ID mapping are cached in the repository (`features/.annotation_cache/`,
`figures_paper/.annotation_cache.pkl`, `up_ids.json`).

The five held-out test cell lines (CAL120, CAMA1, HCC1143, KPL1, ZR75B) come
from the mass cytometry data of Tognetti et al. (2021; Mendeley Data
[doi:10.17632/gvh2vtg86r.1](https://doi.org/10.17632/gvh2vtg86r.1),
CC BY 4.0), processed with `cytof/scan_fcs.py` and read from Synapse
`syn71996733`.
<!-- TODO(authors): syn71996733 is not readable by other Synapse users yet.
Open it to registered users, or change the workflow and this paragraph; see
revision_notes/S6_workflow.md. -->

Everything downloaded or derived ends up in gitignored directories (`data/`,
`features/`, `pretrain/`, `res/`, `eval/`, `figures/`, `notebook_output/`).
Do not commit or publish these.

### 2. Prerequisites

```
python3 -m venv venv
./venv/bin/python -m pip install -r requirements.txt
```

`pandas` is capped below 3.0 and that bound is load-bearing — on pandas 3 the
PEtab lint rejects the Arrow-backed string dtype and both pretraining steps
fail. See the notes at the top of `requirements.txt`.

The paper's results were computed with Snakemake 7 on Python 3.10; the
workflow also works with current Snakemake releases. Snakemake 7 does not
run on Python 3.12 or later, and needs `pulp<2.8`. Only
`estimationCluster.sh` depends on Snakemake 7, through its `--slurm` flag
(Snakemake 8 and later use `--executor slurm`).

Three prerequisites are not installable from `requirements.txt`:

- **BioNetGen**, needed by pysb for network generation. Download it, then point
  `BNGPATH` at the install directory (pysb only searches
  `/usr/local/share/BioNetGen` and `PATH` by default):

  ```
  export BNGPATH=/path/to/BioNetGen-2.9.1
  ```

- **R and the `limma` package**, used by `figures_paper/figure_3.ipynb`
  through rpy2:

  ```
  Rscript -e 'if (!requireNamespace("BiocManager", quietly=TRUE)) install.packages("BiocManager"); BiocManager::install("limma")'
  ```

- **Synapse credentials**, see [Data access](#1-data-access).

Training logs to [Weights & Biases](https://wandb.ai) when `WANDB_MODE=online`
(the default outside `reproduce.sh`), which needs a W&B account and
`WANDB_API_KEY`. Set `WANDB_MODE=disabled` to run without one; nothing
downstream reads the W&B logs.

On a cluster, jobs can run in the Singularity image
`docker://fabfroehlich/generic_parameter_estimation:main`
(`--use-singularity`), which provides Python 3.13, the Python dependencies
and BioNetGen 2.9.3; the host then only needs Snakemake. The figure
notebooks always run outside the container, so the machine that runs them
needs `requirements.txt` and R/limma.

### 3. Run it

**Full reproduction on a cluster (recommended).** On the Crick's SLURM cluster:

```
sbatch estimationCluster.sh paper_results
```

`estimationCluster.sh` takes a target and extra snakemake options, and runs
up to 400 jobs at a time in the Singularity image; adapt the SLURM account,
partition and e-mail for other clusters. Then generate the figures where R
and the notebook dependencies are installed, after copying `eval/` there if
needed:

```
./reproduce.sh figures
```

**Full reproduction on one machine.** `./reproduce.sh all --cores N` runs
everything locally. This is only practical on a large workstation, see below.

**Figures only.** Put the evaluation outputs into `eval/` (for example from a
cluster run) and run `./reproduce.sh figures`. This downloads the Synapse
data the notebooks read (about 0.4 GB) and executes the notebooks; it never
retrains models, even if per-run intermediate files are missing. Evaluation
outputs it cannot find are listed before the notebooks run.

### 4. Expected runtimes and resources

The full reproduction (`paper`, all ten figure configurations) is about
13,700 jobs. Per job, the Snakefile requests the following from SLURM; these
are limits, i.e. upper bounds on the actual runtimes:

| Step (rule) | Jobs | Per job: cores, memory, time limit |
|---|---:|---|
| Data download and preprocessing (`load_data`, `process_data`, `compile_mechanistic_model`) | 35 | 1, 8 GB, 1 h |
| Feature selection (`select_features`) | 586 | 1, 4 GB, 10 h |
| Pretraining per cell line (`pretrain_per_sample`) | 1,071 | 2, 2 GB, 6 h |
| Pretraining of average models (`pretrain_average_model`) | 130 | 2, 2 GB, 24 h |
| Training (`estimate_parameters`) | 5,570 | 2, 8 GB, 24 h |
| Evaluation (`evaluate_training`, `evaluate_references`, `evaluate_regressors`) | 6,286 | 1, 8-16 GB, 1-1.5 h |
| Aggregation (`evaluate_all`) | 29 | 4, 96 GB, 8 h |
| Figures (`run_notebooks.py`) | 1 | 1, 32 GB, 8 h |

Training dominates. The hyperparameter scans of figure1b (2,500 runs) and
figure1c (1,625 runs) account for most of it, and every run is repeated for
5 random starts. This is cluster work: `estimationCluster.sh` runs up to 400
jobs concurrently. A single workstation is impractical for the full set, and
`evaluate_all` alone needs 96 GB of memory. `--config figures=...` reproduces
a subset, e.g. one main figure.
<!-- TODO(authors): add the measured wall time and core-hours of a full
cluster run. -->

Snakemake first builds the graph of all jobs, which for the full set took
47 minutes in a dry run on a laptop at low priority. Restricting to the
figure configurations of interest with `--config figures=...` shortens this.

The figures-only path runs on a laptop: it downloads about 0.4 GB from
Synapse and executes the notebooks sequentially. It needs the aggregated
evaluation outputs in `eval/` (a few GB).

## Paper figures

The notebooks under `figures_paper/` produce the paper figures:

```
./venv/bin/python run_notebooks.py --list --paper-only   # show what would run
./venv/bin/python run_notebooks.py --paper-only          # the paper's notebooks
./venv/bin/python run_notebooks.py                       # also the exploratory ones
```

New notebooks under `figures_paper/` are picked up automatically, including
uncommitted ones; gitignored ones (such as `test_*.ipynb`) are not.
`--paper-only` skips the exploratory notebooks in the `... (not in paper)`
directories.

Everything a notebook writes is redirected under `notebook_output/` and prefixed
with the notebook name — figures into `figures/`, derived CSVs into `data/` — so
the repository never accumulates run artifacts. Failures write the offending
cell and its traceback to `notebook_output/logs/`.

The notebooks need the pipeline's outputs (`eval/`, `data/`, `features/`),
R's limma and external API access, so they cannot run in CI.

## Citation

See `CITATION.cff`. Publications using the Challenge data must also
acknowledge the data contributors and cite Gabor et al. (2021), as the
Challenge data use conditions require.
