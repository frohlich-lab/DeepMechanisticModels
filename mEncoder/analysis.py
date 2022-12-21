import petab
import numpy as np
import pypesto.objective
import seaborn as sns
import matplotlib.pyplot as plt

from mEncoder import data_dir, pretrain_dir, ESTIMATION_OUTFILE_TEMP
from mEncoder.autoencoder import MechanisticAutoEncoder
from mEncoder.plotting import plot_cross_samples

from pypesto.store import OptimizationResultHDF5Reader
from pypesto.C import MODE_RES
from pypesto import OptimizeResult
from amici.petab_objective import rdatas_to_simulation_df
from pathlib import Path

from process_data import training_samples, test_samples, Wildcards


def process_simulation(
    evaluations,
    measurement_df,
    simulation_df,
    context,
    sample,
    model_type,
    alpha,
    hidden_layers,
):
    idx = measurement_df[petab.PREEQUILIBRATION_CONDITION_ID] == sample
    mdf = measurement_df[idx]
    sdf = simulation_df[idx]

    res = (mdf[petab.MEASUREMENT] - sdf[petab.SIMULATION]) / mdf[petab.NOISE_PARAMETERS]

    evaluations.append(
        {
            "rmse": np.sqrt(np.power(res.values, 2).mean()),
            "sample": sample,
            "type": model_type,
            "context": context,
            "alpha": alpha,
            "layers": hidden_layers,
        }
    )


def load_mae(dataset, data, model, context, samples, latent_dim, l1reg):
    samples_train = training_samples(Wildcards(data, samples))

    datafiles = (
        data_dir / f"{data}__{model}__measurements.tsv",
        data_dir / f"{data}__{model}__conditions.tsv",
        data_dir / f"{data}__{model}__observables.tsv",
    )

    mae_train = MechanisticAutoEncoder(
        latent_dim,
        datafiles,
        contextualization=context,
        pathway_name=model,
        samples=samples_train,
        l1reg=l1reg,
    )

    if dataset == "train":
        return mae_train

    samples_test = test_samples(Wildcards(data, samples))

    return MechanisticAutoEncoder(
        latent_dim,
        datafiles,
        pathway_name=model,
        contextualization=context,
        samples=samples_test,
        l1reg=l1reg,
        features=mae_train.features,
        imputer=mae_train.imputer,
        scaler=mae_train.scaler,
        pca=mae_train.pca,
    )


def result_file_pretraining_cross_sample(
    job_id, model, context, data, samples, hidden_layers, alpha
) -> Path:
    indir = pretrain_dir / model / data
    output_prefix = Path(
        ESTIMATION_OUTFILE_TEMP.format(
            context=context,
            samples=samples,
            n_hidden=hidden_layers,
            alpha=alpha,
            job=job_id,
        )
    ).stem
    return indir / f"{output_prefix}.hdf5"


def load_optimize_result_pretraining_cross_samples(
    model, data, context, samples, hidden_layers, alpha
):
    result = OptimizeResult()
    for job in range(100):
        rfile = result_file_pretraining_cross_sample(
            job, model, context, data, samples, hidden_layers, alpha
        )
        if not rfile.exists():
            continue
        r = OptimizationResultHDF5Reader(str(rfile)).read().optimize_result.list[0]
        r["id"] = str(job)
        result.append(r)

    if result.list is not None:
        result.sort()

    return result


def evaluate_simulations(
    obj,
    x,
    samples,
    petab_problem,
    context,
    SAMPLES,
    dataset,
    l1reg,
    latent_dim,
    outdir,
    evaluations,
    model_type,
):

    res = obj(x, mode=MODE_RES, return_dict=True)

    if isinstance(obj, pypesto.objective.AggregatedObjective):
        amici_model = obj._objectives[0].amici_model
    else:
        amici_model = obj.amici_model

    simulation_df = rdatas_to_simulation_df(
        res["rdatas"],
        model=amici_model,
        measurement_df=petab_problem.measurement_df,
    )

    for sample in samples:
        process_simulation(
            evaluations,
            petab_problem.measurement_df,
            simulation_df,
            context,
            sample,
            model_type,
            l1reg,
            latent_dim,
        )

    plot_cross_samples(
        petab_problem.measurement_df,
        simulation_df,
        outdir / dataset,
        "__".join([SAMPLES, context, str(latent_dim), str(l1reg), dataset, model_type]),
    )


def plot_loss_vs_regularization(df):
    g = sns.FacetGrid(data=df, col="sample", col_wrap=5)
    g.map_dataframe(
        sns.lineplot,
        x="alpha",
        y="rmse",
        hue="layers",
        palette="Blues",
        style="context",
    )
    [ax.set(yscale="log", xscale="log") for ax in g.axes]
    plt.tight_layout()
