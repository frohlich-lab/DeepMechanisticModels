# S6: Reproduction workflow, data and code availability

Branch `claude/revision-workflow-01ac68`. This note covers Reviewer 2's request
(a workflow that retrieves the data and reproduces the figures) and the
editor's requests (public deposition of code and data; STAR Methods with a key
resources table and a resource-availability section).

## What changed in the repository

- **Single entry point** (`Snakefile`, `reproduce.sh`).
  - The new Snakemake target `paper` runs everything: Synapse download →
    preprocessing → feature selection → pretraining → training → evaluation
    → `run_notebooks.py --paper-only`.
  - `paper_results` stops before the figures and is the part to run on the
    cluster.
  - `figures` is the light path: figures only, from precomputed `eval/`
    outputs. It doesn't depend on the evaluation rules, so missing per-run
    files never trigger retraining. It still downloads the Synapse data the
    notebooks read, and lists any missing evaluation outputs.
  - `reproduce.sh all|results|figures` wraps the three targets. It checks for
    Synapse credentials and defaults to `WANDB_MODE=disabled`.
  - To build all figure configurations in one DAG, `evaluate_all` and
    `report_all` now take `{figure}` as a wildcard instead of the global
    `FIGURE` config. `train_and_evaluate --config figure=...` behaves as
    before.
  - `PAPER_FIGURES` defaults to every configuration in
    `training_configuration.py` except `default`, i.e. figure1a, 1b, 1c, 2,
    3, 3b, 4, 5, 5b and 6. Restrict it with `--config figures=...`.
  - New rule `download_msigdb` regenerates
    `data/msigdb_c2cp_2025.1.Hs.json` from the public MSigDB GMT. Its output
    is byte-identical to the local copy (checked by running the rule).
- **Cluster.** `estimationCluster.sh` now takes a target and extra options,
  e.g. `sbatch estimationCluster.sh paper_results`. Without arguments it
  runs `train_and_evaluate` exactly as before.
- **`run_notebooks.py`.**
  - Discovery already covered all of `figures_paper/` recursively, but only
    git-tracked notebooks. It now also finds new, uncommitted notebooks,
    while still skipping gitignored ones.
  - New `--paper-only` flag skips the `... (not in paper)` directories.
    Tested with temporary notebooks.
- **Works without W&B and outside a git clone** (`dmm/wandb_init_log.py`).
  - `mode="online"` was hard-coded and overrode `WANDB_MODE` (checked with
    wandb 0.25.1), so training needed a W&B account. It now honors
    `WANDB_MODE`, still defaulting to `online`.
  - `repo.remotes.origin.url` crashed outside a git clone, e.g. in the
    Zenodo zip. Commit and remote are now optional.
  - The Snakefile no longer *requires* `SYNAPSE_AUTH_TOKEN` and
    `WANDB_API_KEY` to be set: `~/.synapseConfig` suffices.
- **`process_data.py`**: the MOSA citation comment pointed to the wrong DOI
  (Kunes et al. 2023, Nat Biotechnol). Corrected to Cai et al. 2024 and the
  figshare record.
- **`README.md`**: step-by-step data access, prerequisites, both entry
  points, expected runtimes and resources.
- **`CITATION.cff`** (validated against CFF 1.2.0): authors and ORCIDs come
  from the preprint's Crossref record. License, version, release date and
  Zenodo DOI are marked TODO.

**Verification.** Dry runs only (`snakemake -n` under `taskpolicy -c utility`).
- Snakemake 7.32.4 on Python 3.10: `train_and_evaluate` for figure1a, figure2,
  figure3 and figure4 gives job stats identical to the unmodified Snakefile.
- Snakemake 9.27.1 on Python 3.12: figure2 gives the same job counts, no
  warnings.
- `figures` schedules 3 jobs (`load_data`, `download_msigdb`, `figures`) and
  no training.
- `paper` (full DAG, all ten configurations, Snakemake 7.32.4): builds
  without errors. It schedules 13,708 jobs, matching the counts computed
  independently from the configuration (`download_msigdb` had already run).
  Building the DAG took 47 min at low priority on this laptop.
- Nothing was executed for real.

## Decisions needed from the authors

1. **The five test cell lines are not downloadable by others.** This blocks
   reproduction and affects the availability statement.
   - `cytof/data.py` reads CAL120, CAMA1, HCC1143, KPL1 and ZR75B (the
     validation set of the `all` split) from `syn71996744/45/46/34/47`.
   - These sit in the Synapse project `syn71996733` "DMM test data", owned
     by FF's account. They are not public and not shared with registered
     users; anonymous reads get "You lack READ access". They are not part
     of the Challenge project syn20366914.
   - The files were derived with `cytof/scan_fcs.py` from the FCS files of
     Tognetti et al. 2021 (Mendeley Data doi:10.17632/gvh2vtg86r.1,
     **CC BY 4.0**; about 650 MB, 313 FCS files for these five lines).

   Options:
   - (a) Open `syn71996733` to registered Synapse users, or make it public,
     crediting Tognetti et al. This is the smallest change.
   - (b) Deposit the five derived CSVs with the code on Zenodo (CC BY allows
     this).
   - (c) Add a workflow rule that downloads the FCS files from Mendeley and
     reruns `scan_fcs.py`. Most reproducible, most work.

   Either way the availability statement must cite the Mendeley dataset; the
   draft below does.
2. **No LICENSE.** The GitHub repository is public but has no license, which
   legally means "all rights reserved". Zenodo asks for a license and so does
   `CITATION.cff`. Pick one (e.g. MIT, BSD-3-Clause or Apache-2.0), add
   `LICENSE`, and fill `license:` in `CITATION.cff`.
3. **figure_S7.ipynb cannot be regenerated from the repository.**
   - It reads `figures_paper/egfr_spectronaut_rows.csv`,
     `egfr_peptides_mapped.csv`, `egfr_isoform_presence_matrix.csv` and
     `egfr_log2fc.csv`.
   - These come from `fetch_egfr_peptides_ftp.py` and
     `egfr_splicing_analysis.py` (PXD017199), neither of which is in git.
     The CSVs are also gitignored (`*.csv`).
   - Commit the scripts; they can then become workflow rules feeding the
     notebooks.
4. **Pin the container image.**
   - `docker://fabfroehlich/generic_parameter_estimation:main` is a moving
     tag. It was last pushed 2025-12-19, digest
     `sha256:8cef8a8d91924fb571915389a889217418d85c5d87f6a067ba05cfddb47a4428`
     (Python 3.13.11, its own pip install of a requirements file,
     BioNetGen 2.9.3).
   - Check which image the paper runs used (`.snakemake/singularity/` on the
     cluster), then pin it by digest in the `singularity:` line.
   - Optionally record `pip freeze` from inside it as the authoritative
     version list.
5. **Precomputed evaluation outputs for the "figures only" path.**
   - Deposit the aggregated `eval/{model}/dream_cytof/*_{figure}.csv` files
     only if they don't count as redistributing the Challenge data; they
     hold RMSEs, embeddings, parameter deviations, sensitivities and
     predicted trajectories, but no measured values.
   - The local copy is 3.5 GB including `trajectories_*.csv`, the predicted
     phospho time courses that come closest to the data. Check with the
     Challenge conditions or Sage first.
   - Without a deposit, the light path applies to anyone who has run
     `paper_results`.
   - The local `eval/` is also incomplete. figure4 outputs exist only for
     `EGFR_MAPK__logobs_tegfr_aggavg` (9 model variants have empty
     directories), and figure3 lacks tERBB2/tERBB3. The `figures` target
     lists such gaps before running.
6. **Challenge acknowledgement.**
   - The Challenge Terms of Use
     (https://www.synapse.org/Synapse:syn20366914/wiki/594736) require a
     fixed acknowledgement sentence that names Attila Gabor and Marco
     Tognetti as data contributors and gives the Synapse ID, plus a citation
     of the overview paper (Gabor et al., 2021).
   - The drafted availability sentence follows that template but omits
     "contributed by Attila Gabor and Marco Tognetti". Add the required
     wording to the Acknowledgments.
7. **Minor.**
   - `figures_paper/mmc2.xlsx` appears to be Table S1 of Marcotte et al.
     2016 (Cell). It is a publisher supplementary file committed to a
     public repo; consider downloading it in the workflow instead.
   - `requirements.txt` doesn't pin Snakemake: a fresh install gets
     Snakemake 9. That works for the workflow, but not for the `--slurm`
     flag in `estimationCluster.sh`.

## Expected runtimes

Job counts for `paper` (all ten figure configurations, `num_starts=5`) were
computed from `generate_run_configs` and the Snakefile's wildcards; see
*Verification* above for the `snakemake -n` cross-check. The README table
lists per-job resources.

- 5,570 training runs (`estimate_parameters`), each also evaluated. The
  figure1b ML scan (2,500 new runs) and the figure1c feature scan (1,625)
  account for about 74%.
- 1,071 per-cell-line and 130 average pretrainings, 586 feature selections
  and regressor evaluations, 29 `evaluate_all` aggregations (96 GB each).
- About 13,700 jobs in total.
- Upper bound from the SLURM limits in the Snakefile: about 302,000
  core-hours, of which 267,000 are training (5,570 × 24 h × 2 cores). Real
  usage is lower.

**TODO(authors):** fill in the measured wall time and core-hours of a full
cluster run (e.g. from `sacct` or the W&B run durations). The response
letter and README should quote the real figure, not the bound.

The figures-only path ran on this laptop before: `notebook_output/` in the
main checkout. It needs the aggregated `eval/` files (3.5 GB locally) plus a
0.4 GB Synapse download.

## STAR Methods inputs

### (a) Key resources table (draft)

Software versions are from the local environment that ran the figure
notebooks (`venv`, Python 3.13). Training ran in the container image (item 4
above), whose versions may differ; refresh the table from its `pip freeze`.
RRIDs were checked against the SciCrunch resolver. Where no RRID could be
found, the software paper or URL is given.

| REAGENT or RESOURCE | SOURCE | IDENTIFIER |
|---|---|---|
| **Deposited data** | | |
| Single Cell Signaling in Breast Cancer DREAM Challenge data (mass cytometry, RNA-seq, proteomics) | Gabor et al., 2021 | Synapse: syn20366914 |
| Mass cytometry data of breast cancer cell lines (FCS) | Tognetti et al., 2021 | Mendeley Data: https://doi.org/10.17632/gvh2vtg86r.1 |
| Proteomics of breast cancer cell lines | Tognetti et al., 2021 | ProteomeXchange: PXD017199 |
| MOSA multi-omic latent embeddings of cancer cell lines | Cai et al., 2024 | figshare: https://doi.org/10.6084/m9.figshare.24562765.v1 |
| Test cell lines CAL120, CAMA1, HCC1143, KPL1, ZR75B (processed from Tognetti et al., 2021) | This paper | Synapse: syn71996733 (see decision 1) |
| Code, workflow and figure notebooks | This paper | GitHub: https://github.com/frohlich-lab/DeepMechanisticModels; Zenodo: https://doi.org/10.5281/zenodo.TODO |
| Container image for model training | This paper | Docker Hub: fabfroehlich/generic_parameter_estimation@sha256:TODO |
| **Software and algorithms** | | |
| Python 3.10 (workflow), 3.13 (container, notebooks) | Python Software Foundation | RRID:SCR_008394 |
| Snakemake 7 (7.32.4) | Mölder et al., 2021 | RRID:SCR_003475; https://doi.org/10.12688/f1000research.29032.2 |
| JAX 0.6.2 | Bradbury et al., 2018 | https://github.com/jax-ml/jax |
| Equinox 0.13.0 | Kidger and Garcia, 2021 | https://arxiv.org/abs/2111.00254 |
| Optax 0.2.5 | DeepMind et al., 2020 | https://github.com/google-deepmind/optax |
| Flax 0.10.7 | Heek et al., 2024 | https://github.com/google/flax |
| AMICI 0.33.0 | Fröhlich et al., 2021 | https://doi.org/10.1093/bioinformatics/btab227 |
| PySB 1.17.0 | Lopez et al., 2013 | https://doi.org/10.1038/msb.2013.1 |
| BioNetGen 2.9.3 | Harris et al., 2016 | https://doi.org/10.1093/bioinformatics/btw469 |
| PEtab 0.5.0 | Schmiester et al., 2021 | https://doi.org/10.1371/journal.pcbi.1008646 |
| pyPESTO 0.5.6 | Schälte et al., 2023 | https://doi.org/10.1093/bioinformatics/btad711 |
| Fides 0.7.8 | Fröhlich and Sorger, 2022 | https://doi.org/10.1371/journal.pcbi.1010322 |
| scikit-learn 1.8.0 | Pedregosa et al., 2011 | RRID:SCR_002577 |
| NumPy 2.4.4 | Harris et al., 2020 | RRID:SCR_008633 |
| SciPy 1.17.1 | Virtanen et al., 2020 | RRID:SCR_008058 |
| pandas 2.3.3 | The pandas development team | RRID:SCR_018214; https://doi.org/10.5281/zenodo.3509134 |
| statsmodels 0.14.6 | Seabold and Perktold, 2010 | RRID:SCR_016074 |
| GSEApy 1.1.13 | Fang et al., 2023 | https://doi.org/10.1093/bioinformatics/btac757 |
| pytometry 0.1.6 | Büttner et al. | https://github.com/buettnerlab/pytometry |
| synapseclient 4.13.0 | Sage Bionetworks | https://github.com/Sage-Bionetworks/synapsePythonClient |
| R 4.6.1 | R Core Team | RRID:SCR_001905 |
| limma 3.68.4 | Ritchie et al., 2015 | RRID:SCR_010943 |
| rpy2 3.6.7 | rpy2 developers | https://rpy2.github.io |
| matplotlib 3.10.8 | Hunter, 2007 | RRID:SCR_008624 |
| seaborn 0.13.2 | Waskom, 2021 | RRID:SCR_018132 |
| plotnine 0.15.3 | plotnine developers | https://github.com/has2k1/plotnine |
| Jupyter Notebook (nbclient 0.10.4) | Project Jupyter | RRID:SCR_018315 |
| Weights & Biases 0.25.1 (optional logging) | Biewald, 2020 | https://wandb.ai |
| Singularity 3.6.4 | Kurtzer et al., 2017 | https://doi.org/10.1371/journal.pone.0177459 |
| **Other** | | |
| Synapse | Sage Bionetworks | RRID:SCR_006307 |
| ProteomeXchange / PRIDE | Deutsch et al., 2023; Perez-Riverol et al., 2022 | RRID:SCR_004055; RRID:SCR_003411 |
| Molecular Signatures Database, C2 canonical pathways v2025.1.Hs | Liberzon et al., 2011 | RRID:SCR_016863 |
| Cellosaurus | Bairoch, 2018 | RRID:SCR_013869 |
| cBioPortal | Cerami et al., 2012; Gao et al., 2013 | RRID:SCR_014555 |
| Cell Model Passports | van der Meer et al., 2019 | https://cellmodelpassports.sanger.ac.uk |
| UniProt (ID mapping, cached in `up_ids.json`) | UniProt Consortium, 2023 | RRID:SCR_002380 |
| OncoKB, ClinVar (mutation annotations, cached) | Chakravarty et al., 2017; Landrum et al., 2018 | https://www.oncokb.org; https://www.ncbi.nlm.nih.gov/clinvar |
| figshare / Zenodo / Mendeley Data | | RRID:SCR_004328; RRID:SCR_004129; RRID:SCR_015671 |

References whose DOIs are not given above (check against the manuscript's
reference list):
- Gabor et al., 2021, Mol Syst Biol 17:e10402, doi:10.15252/msb.202110402
- Tognetti et al., 2021, Cell Syst 12:401, doi:10.1016/j.cels.2021.04.002
- Cai et al., 2024, Nat Commun 15:10390, doi:10.1038/s41467-024-54771-4
- Marcotte et al., 2016, Cell 164:293, doi:10.1016/j.cell.2015.11.062
  (RNA-seq in the Challenge data, `mmc2.xlsx`)
- Ritchie et al., 2015, Nucleic Acids Res 43:e47, doi:10.1093/nar/gkv007
- Liberzon et al., 2011, Bioinformatics 27:1739, doi:10.1093/bioinformatics/btr260

### (b) Data and code availability (draft, Cell Press format)

- This paper analyzes existing data. The single-cell signaling,
  transcriptomic and proteomic data were obtained as part of the Single Cell
  Signaling in Breast Cancer Challenge through Synapse (Synapse ID
  syn20366914; Gabor et al., 2021). They are available to registered Synapse
  users who accept the Challenge data use conditions; these conditions do not
  permit redistribution of the data. Mass cytometry data for the five
  held-out test cell lines are publicly available from Mendeley Data
  (https://doi.org/10.17632/gvh2vtg86r.1; Tognetti et al., 2021)
  [TODO, depending on decision 1: "and, as processed for this study, from
  Synapse (syn71996733)" / "and, as processed for this study, with the
  code on Zenodo"]. The proteomics data are also available from
  ProteomeXchange (PXD017199). The MOSA multi-omic embeddings (Cai et al.,
  2024) are publicly available from figshare
  (https://doi.org/10.6084/m9.figshare.24562765.v1). Accession numbers are
  listed in the key resources table.
- Code to reproduce model training, hyperparameter tuning and figures is
  available at https://github.com/frohlich-lab/DeepMechanisticModels
  (archived at Zenodo, DOI: …), including a workflow that retrieves all input
  data and regenerates all figures. The DOI is listed in the key resources
  table.
- Any additional information required to reanalyze the data reported in this
  paper is available from the lead contact upon request.

The first two sentences of the code bullet and the Synapse sentences are the
drafted text, unchanged. The additions are the test cell lines, PXD017199
and MOSA. The workflow downloads all of these, so "retrieves all input data"
stays true once decision 1 is settled.

### (c) Response to Reviewer 2: text for the placeholder

Placeholder: "We have now [improved the documentation of the end-to-end
workflow and added a single entry point from data download to figures]."

Paste-ready (US spelling):

> We have now improved the documentation of the end-to-end workflow and added
> a single entry point that runs it from data download to the final figures
> (`reproduce.sh` in the repository). After a one-time setup (creating a
> free Synapse account, registering for the DREAM Challenge and thereby
> accepting its data use conditions, and providing a Synapse access token),
> the workflow downloads all input data through the Synapse API and from
> public repositories. It then preprocesses the data, performs feature
> selection, pretraining, training and the hyperparameter scans, evaluates
> all models, and regenerates all main and supplementary figures. Because
> the full training requires a computing cluster, we also provide a second,
> lightweight entry point that regenerates all figures from precomputed
> evaluation outputs on a laptop. The README now describes data access,
> software prerequisites, both entry points, and the expected runtime and
> resources. The code will be archived on Zenodo upon publication.

Before sending, check two claims. "all main and supplementary figures"
holds only once decisions 1 and 3 are resolved. "from precomputed
evaluation outputs" assumes the outputs are available to the reader
(decision 5); if they are not deposited, write "from the evaluation outputs
of a previous run" instead.
