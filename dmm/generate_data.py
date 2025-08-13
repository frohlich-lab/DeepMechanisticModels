from pathlib import Path
from typing import Tuple

import amici
import jax
import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import pandas as pd
import petab.v1 as petab
from sklearn import decomposition

from . import MODEL_FEATURE_PREFIX, plot_and_save_fig
from .encoder import AutoEncoder
from .problem import Problem


def generate_synthetic_data(
    problem: Problem,
    data_dir: Path,
    data_name: str,
    latent_dimension: int = 2,
    n_samples: int = 45,
    n_features: int = 200,
    std_measurements: float = 0.1,
    std_features: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates sample data using the mechanistic model.
    """
    model, solver = problem.load_amici(
        problem.load_pysb(),
        amici_dir=data_dir / "amici_models",
        force_compile=True,
        add_observables=True,
        name_suffix=f"_{data_name}",
    )

    solver.setAbsoluteTolerance(1e-12)
    solver.setRelativeTolerance(1e-12)

    bounds = problem.bounds
    # setup model parameter scales
    model.setParameterScale(
        amici.parameterScalingFromIntVector(
            [
                amici.ParameterScaling.none
                if bounds[par_id.split("_")[-1]][2] == "lin"
                else amici.ParameterScaling.log10
                for par_id in model.getParameterIds()
            ]
        )
    )

    # run simulations to equilibrium
    model.setTimepoints([0, 2, 4, 8, 15, 30, 60])

    edata_base = amici.ExpData(model)
    edata_base.fixedParametersPreequilibration = edata_base.fixedParameters
    fp = list(edata_base.fixedParameters)
    fp[model.getFixedParameterIds().index("EGF_0")] = 1.0
    edata_base.fixedParameters = fp
    edata_base.reinitializeFixedParameterInitialStates = True

    edatas = [edata_base]

    for ipert, pert in enumerate(model.getFixedParameterIds()):
        if pert == "EGF_0":
            continue
        edata = amici.ExpData(edata_base)
        tmp = list(edata.fixedParameters)
        tmp[ipert] = 1.0
        edata.fixedParameters = tmp
        tmp[model.getFixedParameterIds().index("EGF_0")] = 0.0
        edata.fixedParametersPresimulation = tuple(tmp)
        edata.t_presim = 15
        edatas.append(edata)

    # set numpy random seed to ensure reproducibility
    np.random.seed(0)

    # set input parameters to zero to simulate the baseline
    deviations_names = [
        par_id
        for par_id in model.getParameterIds()
        if par_id.startswith(MODEL_FEATURE_PREFIX)
    ]

    for par_id in deviations_names:
        model.setParameterById(par_id, 0.0)

    # generate static parameters that are consistent across samples
    parameter_means = {}

    avg_model_vars = {
        "MED_EGFR__Y1173_p_sus_amp": -2.0,
        "MED_EGF_0_trans_tau": 0.0,
        "ERBB2_eq": 1.0,
        "MED_ERBB2_dephosphorylation_Y1248_base_kcat": 4.0,
        "MED_ERBB2_phosphorylation_Y1248_base_kr": 0.0,
        "MED_ERBB2_phosphorylation_Y1248_kr": 2.0,
        "MEK_eq": 1.0,
        "MED_MEK_dephosphorylation_S222_base_kcat": 0.0,
        "MED_MEK_phosphorylation_S222_base_kr": -4.0,
        "ERK_eq": 1.0,
        "MED_ERK_dephosphorylation_Y204_base_kcat": 0.0,
        "MED_MEK_phosphorylation_S222_kr": 4.0,
        "MED_MEK_dephosphorylation_S222_ERK__Y204_p_kw": 3.5,
        "MED_ERK_phosphorylation_Y204_EGFR__Y1173_p_kw": -4.0,
        "MED_iMEK_MEK__S222_p_obs_kd": -0.5,
        "MED_iEGFR_EGFR__Y1173_p_kd": 0.0,
        # following parameters have slightly different names due to petab import
        "pERBB2_Y1248_offset": 1.8,
        "pERBB2_Y1248_scale": 1.5,
        "pERK_Y204_offset": -2.0,
        "pERK_Y204_scale": 3.0,
        "pMEK_S222_offset": 1.0,
        "pMEK_S222_scale": 0.5,
    }
    for par_id in avg_model_vars:
        assert par_id in model.getParameterIds(), par_id

    while True:
        for par_id in model.getParameterIds():
            if par_id in deviations_names:
                continue

            if par_id in avg_model_vars:
                # use approximate values from average model
                parameter_means[par_id] = avg_model_vars[par_id]
            else:
                # sample from uniform distribution
                lb, ub, _ = bounds[par_id.split("_")[-1]]
                lb += 1
                ub -= 1
                parameter_means[par_id] = np.random.random() * (ub - lb) + lb
            model.setParameterById(par_id, parameter_means[par_id])

        rdatas = amici.runAmiciSimulations(model, solver, [edata_base])
        if rdatas[0].status == amici.AMICI_SUCCESS:
            break

    encoder = AutoEncoder(
        features=np.random.random((n_samples, n_features)),
        n_latent=latent_dimension,
        n_params=len(deviations_names),
    )

    # generate sparse encoder/decoder parameters
    tt_pars = np.random.random(encoder.n_encoder_pars)
    for ip, name in enumerate(encoder.x_names):
        # xavier glorot initialization
        if name.startswith("encoder"):
            n_inout = encoder.n_features + encoder.n_latent
        else:
            n_inout = encoder.n_latent + encoder.n_params
        ub = np.sqrt(6.0 / n_inout)
        lb = -ub
        tt_pars[ip] = tt_pars[ip] * (ub - lb) + lb

    encode_weights, inflate_weights = np.split(
        tt_pars, (encoder.n_encode_weights,)
    )
    pd.Series(dict(zip(encoder.x_names, tt_pars))).to_csv(
        data_dir / f"{problem.model_name}__{data_name}__reference_weights.csv"
    )

    samples = []
    embeddings = []

    encode_sample = jax.jit(encoder.encode_sample)
    decode = jax.jit(encoder.decode)
    inflate = jax.jit(encoder.inflate_params)
    while len(samples) < n_samples:
        # generate new fake data for sample
        embedding = np.random.random(latent_dimension) * 2 - 1
        sample_data = np.array(decode(embedding, encode_weights))
        mat = encode_weights.reshape((n_features, latent_dimension))
        assert np.allclose(
            sample_data, embedding.dot(np.linalg.pinv(mat)), atol=1e-6
        )
        assert np.allclose(sample_data.dot(mat), embedding, atol=1e-6)
        assert np.allclose(
            embedding, embedding.dot(np.linalg.pinv(mat)).dot(mat), atol=1e-6
        )
        assert np.allclose(
            np.asarray(encode_sample(sample_data, encode_weights)),
            embedding,
            atol=1e-6,
        )

        deviations = np.array(inflate(embedding, inflate_weights))
        assert len(deviations) == len(deviations_names)
        parameter_deviations = dict(zip(deviations_names, deviations))

        # set parameters in model
        for par_id, val in {**parameter_means, **parameter_deviations}.items():
            model.setParameterById(par_id, float(val))

        # run simulations, only add to samples if no integration error
        rdatas = amici.runAmiciSimulations(model, solver, edatas)
        if all(r.status == amici.AMICI_SUCCESS for r in rdatas):
            sample = amici.getSimulationObservablesAsDataFrame(
                model, edatas, rdatas
            )
            for obs in model.getObservableIds():
                sample[obs] = np.random.normal(sample[obs], std_measurements)
            sample["Sample"] = len(samples)
            for pid, val in parameter_deviations.items():
                sample[pid] = val
            for ifeature, value in enumerate(sample_data):
                sample[f"feature{ifeature}"] = value + numpy.random.normal(
                    0.0, std_features
                )
            samples.append(sample)
            embeddings.append(embedding)

    # prepare petab
    data_dir.mkdir(exist_ok=True, parents=True)

    df = pd.concat(samples)
    df.loc[
        (df.time == 0)
        & (
            df.loc[
                :,
                [
                    x
                    for x in list(model.getFixedParameterIds())
                    if x != "EGF_0"
                ],
            ]
            == 0
        ).all(axis=1),
        list(model.getObservableIds()),
    ].rename(
        columns={o: o.replace("_obs", "") for o in model.getObservableIds()}
    ).boxplot(rot=90)
    plot_and_save_fig(f"{problem.model_name}__{data_name}.pdf", data_dir)

    fig, ax = plt.subplots(1, 1)
    embeddings = np.vstack(embeddings)
    plot_embedding(embeddings, ax)

    pd.DataFrame(
        embeddings,
        index=[f"sample_{isample}" for isample in range(embeddings.shape[0])],
    ).to_csv(data_dir / f"{problem.model_name}__{data_name}__embeddings.csv")

    plot_and_save_fig(
        f"{problem.model_name}__{data_name}__embedding.pdf", data_dir
    )

    inputs = df.loc[
        (df.time == 0)
        & (
            df.loc[
                :,
                [
                    x
                    for x in list(model.getFixedParameterIds())
                    if x != "EGF_0"
                ],
            ]
            == 0
        ).all(axis=1),
        [
            col
            for col in df.columns
            if col.startswith(MODEL_FEATURE_PREFIX) or col == "Sample"
        ],
    ]
    inputs.Sample = inputs.Sample.apply(lambda x: f"sample_{x}")
    inputs.set_index("Sample", inplace=True)

    fig, ax = plt.subplots(1, 1)
    plot_pca_inputs(inputs.values, ax)

    plot_and_save_fig(
        f"{problem.model_name}__{data_name}__input_pca.pdf", data_dir
    )
    inputs.to_csv(
        data_dir / f"{problem.model_name}__{data_name}__reference_inputs.csv"
    )
    pd.Series(parameter_means).to_csv(
        data_dir / f"{problem.model_name}__{data_name}__reference_pars.csv"
    )

    fig, axes = plt.subplots(1, 2)
    plot_pca_inputs(
        df[list(model.getObservableIds())].values, axes[0], axes[1]
    )
    plot_and_save_fig(
        f"{problem.model_name}__{data_name}__data_pca.pdf", data_dir
    )

    # create petab & save to csv
    # MEASUREMENTS
    features = [x for x in df.columns if x.startswith("feature")]
    measurements = df[
        [
            "Sample",
            petab.TIME,
        ]
        + list(model.getObservableIds())
        + list(model.getFixedParameterIds())
        + features
    ]
    measurements = pd.melt(
        measurements,
        id_vars=[petab.TIME, "Sample"] + list(model.getFixedParameterIds()),
        value_name=petab.MEASUREMENT,
        var_name=petab.OBSERVABLE_ID,
    )

    measurements.loc[
        measurements[petab.OBSERVABLE_ID].apply(lambda x: x in features),
        "EGF_0",
    ] = 0

    measurements = measurements[
        # phospho
        measurements[petab.OBSERVABLE_ID].apply(lambda x: x.startswith("p"))
        | (
            # or total
            measurements[petab.OBSERVABLE_ID].apply(
                lambda x: x.startswith("t")
            )
            & (
                # and baseline
                measurements[[petab.TIME] + list(model.getFixedParameterIds())]
                == 0
            ).all(axis=1)
        )
        | (
            # or feature
            measurements[petab.OBSERVABLE_ID].apply(lambda x: x in features)
            & (
                # and baseline
                measurements[[petab.TIME] + list(model.getFixedParameterIds())]
                == 0
            ).all(axis=1)
        )
    ]

    # filter that only non-baseline conditions have dynamic measurements
    measurements = measurements[
        measurements.apply(
            lambda m: (m[list(model.getFixedParameterIds())] != 0).any()
            | (m[petab.TIME] == 0),
            axis=1,
        )
    ]

    # fix observable names so they are properly recognized in downstream
    # processing
    measurements[petab.OBSERVABLE_ID] = measurements[
        petab.OBSERVABLE_ID
    ].apply(lambda x: x.replace("_obs", ""))

    measurements[petab.SIMULATION_CONDITION_ID] = measurements.apply(
        lambda x: f'sample_{x["Sample"]}'
        + "".join(
            [
                f'__{fp.replace("_0", "")}' if x[fp] > 0 else ""
                for fp in model.getFixedParameterIds()
            ]
        ).replace("__EGF__", "__"),
        axis=1,
    )

    measurements[petab.PREEQUILIBRATION_CONDITION_ID] = measurements[
        "Sample"
    ].apply(lambda x: f"sample_{x}")

    measurements.drop(
        columns=["Sample"] + list(model.getFixedParameterIds()), inplace=True
    )

    measurements[petab.OBSERVABLE_PARAMETERS] = measurements[
        petab.OBSERVABLE_ID
    ].apply(lambda x: f"{x}_scale;{x}_offset")

    measurements[petab.NOISE_PARAMETERS] = "1.0"

    # CONDITIONS
    conditions = pd.DataFrame(
        {
            petab.CONDITION_ID: sorted(
                set(
                    measurements[petab.SIMULATION_CONDITION_ID].unique()
                ).union(
                    set(
                        measurements[
                            petab.PREEQUILIBRATION_CONDITION_ID
                        ].unique()
                    )
                )
            ),
        }
    )
    for fp in model.getFixedParameterIds():
        if fp == "EGF_0":
            conditions[fp] = conditions[petab.CONDITION_ID].apply(
                lambda x: float("__" in x)
            )
        else:
            conditions[fp] = conditions[
                petab.CONDITION_ID
            ].apply(
                lambda x: float(
                    fp.replace("_0", "")  # noqa: B023
                    in (cond for cond in x.split("__")[1:])
                )
            )

    return conditions, measurements


def plot_embedding(embedding: np.ndarray, ax: plt.Axes):
    ax.plot(embedding[:, 0], embedding[:, 1], "k*")


def plot_pca_inputs(
    x: np.ndarray, embed_ax: plt.Axes, vexpl_ax: plt.Axes = None
):
    pca = decomposition.PCA(n_components=min(x.shape[1], 10))
    pca.fit(x)
    x_pca = pca.transform(x)

    embed_ax.plot(x_pca[:, 0], x_pca[:, 1], "k*")

    if vexpl_ax is not None:
        vexpl_ax.plot(np.cumsum(pca.explained_variance_ratio_))
        vexpl_ax.set_xlabel("number of components")
        vexpl_ax.set_ylabel("cumulative explained variance")
