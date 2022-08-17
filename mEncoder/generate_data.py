from . import (
    load_model, parameter_boundaries_scales, MODEL_FEATURE_PREFIX,
    plot_and_save_fig, basedir
)

from .encoder import AutoEncoder

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import decomposition

import amici
import petab

from typing import Tuple
from pathlib import Path


def generate_synthetic_data(pathway_name: str,
                            latent_dimension: int = 2,
                            n_samples: int = 20) -> Tuple[Path, Path, Path]:
    """
    Generates sample data using the mechanistic model.

    :param pathway_name:
        name of pathway to use for model

    :param latent_dimension:
        number of latent dimensions that is used to generate the parameters
        that vary across samples

    :param n_samples:
        number of samples to generate

    :return:
        path to csv where generated data was saved
    """
    model, solver = load_model('pw_' + pathway_name, force_compile=True,
                               add_observables=True)

    solver.setAbsoluteTolerance(1e-12)
    solver.setRelativeTolerance(1e-12)

    # setup model parameter scales
    model.setParameterScale(amici.parameterScalingFromIntVector([
        amici.ParameterScaling.none
        if parameter_boundaries_scales[par_id.split('_')[-1]][2] == 'lin'
        else amici.ParameterScaling.log10
        for par_id in model.getParameterIds()
    ]))


    # run simulations to equilibrium
    model.setTimepoints([0, 2, 4, 8, 15, 30, 60])

    edata_base = amici.ExpData(model)

    edata_base.fixedParametersPreequilibration = edata_base.fixedParameters
    edatas = [edata_base]

    for ipert, pert in enumerate(model.getFixedParameterIds()):
        edata = amici.ExpData(edata_base)
        tmp = list(edata.fixedParameters)
        tmp[ipert] = 1
        edata.fixedParameters = tmp
        edatas.append(edata)

    # set numpy random seed to ensure reproducibility
    np.random.seed(0)

    sample_pars = [par_id for par_id in model.getParameterIds()
                   if par_id.startswith(MODEL_FEATURE_PREFIX)]

    for par_id in model.getParameterIds():
        if par_id.startswith(MODEL_FEATURE_PREFIX):
            model.setParameterById(par_id, 0.0)

    # generate static parameters that are consistent across samples
    static_pars = dict()

    while True:
        for par_id in model.getParameterIds():
            if par_id in sample_pars:
                continue
            lb, ub, _ = parameter_boundaries_scales[par_id.split('_')[-1]]
            static_pars[par_id] = np.random.random() * (ub - lb) + lb
            model.setParameterById(par_id, 0.0)
        rdatas = amici.runAmiciSimulations(model, solver, [edata_base])
        if rdatas[0].status == amici.AMICI_SUCCESS:
            break

    encoder = AutoEncoder(np.zeros((1, model.ny)),
                          n_hidden=latent_dimension, n_params=len(sample_pars))
    tt_pars = np.random.random(encoder.n_encoder_pars)
    for ip, name in enumerate(encoder.x_names):
        lb, ub, _ = parameter_boundaries_scales[name.split('_')[-1]]
        if np.random.binomial(1, 0.3):
            tt_pars[ip] = 0
        else:
            tt_pars[ip] = tt_pars[ip] * (ub - lb) + lb

    samples = []
    embeddings = []
    while len(samples) < n_samples:
        # generate new fake data
        encoder.data = np.random.random(encoder.data.shape) / 10

        if len(samples) < n_samples / 2:
            encoder.data += 0.2
        else:
            encoder.data -= 0.2

        # generate parameters from fake data
        embedding = encoder.compute_embedded_pars(tt_pars)
        sample_par_vals = encoder.compute_inflated_pars(tt_pars)
        sample_pars = dict(zip(sample_pars, sample_par_vals[0, :]))

        # set parameters in model
        for par_id, val in {**static_pars, **sample_pars}.items():
            model.setParameterById(par_id, val)

        # run simulations, only add to samples if no integration error
        rdatas = amici.runAmiciSimulations(model, solver, edatas)
        if all([r.status == amici.AMICI_SUCCESS for r in rdatas]):
            sample = amici.getSimulationObservablesAsDataFrame(
                model, edatas, rdatas
            )
            sample['Sample'] = len(samples)
            for pid, val in sample_pars.items():
                sample[pid] = val
            samples.append(sample)
            embeddings.append(embedding)

    # prepare petab
    datadir = basedir / 'data'
    datadir.mkdir(exist_ok=True, parents=True)

    df = pd.concat(samples)
    df.loc[(df.time == 0) &
           (df[list(model.getFixedParameterIds())] == 0).all(axis=1),
           list(model.getObservableIds())].rename(columns={
               o: o.replace('_obs', '') for o in model.getObservableIds()
           }).boxplot(rot=90)
    plot_and_save_fig(datadir / f'synthetic__{pathway_name}.pdf')

    fig, ax = plt.subplots(1, 1)
    plot_embedding(np.vstack(embeddings), ax)

    plot_and_save_fig(datadir / f'synthetic__{pathway_name}__embedding.pdf')

    inputs = df.loc[
        (df.time == 0) &
        (df[list(model.getFixedParameterIds())] == 0).all(axis=1),
        [col for col in df.columns if col.startswith(MODEL_FEATURE_PREFIX)]
    ]

    fig, ax = plt.subplots(1, 1)
    plot_pca_inputs(inputs.values, ax)

    plot_and_save_fig(datadir / f'synthetic__{pathway_name}__input_pca.pdf')

    inputs = df[[col for col in df.columns
                 if col.startswith(MODEL_FEATURE_PREFIX) or col == 'Sample']]

    inputs = pd.melt(inputs, id_vars=['Sample'])
    inputs.index = inputs['variable'] + \
        inputs['Sample'].apply(lambda x: f'_sample_{x}')
    ref = pd.concat([pd.Series(static_pars), inputs.value])
    ref.to_csv(datadir / f'synthetic__{pathway_name}__reference_inputs.csv')

    fig, axes = plt.subplots(1, 2)
    plot_pca_inputs(df[list(model.getObservableIds())].values, axes[0],
                    axes[1])
    plot_and_save_fig(datadir, f'synthetic__{pathway_name}__data_pca.pdf')

    # create petab & save to csv
    # MEASUREMENTS
    measurements = df[['Sample', petab.TIME, ] +
                      list(model.getObservableIds()) + list(model.getFixedParameterIds())]
    measurements = pd.melt(measurements,
                           id_vars=[petab.TIME, 'Sample'] + list(model.getFixedParameterIds()),
                           value_name=petab.MEASUREMENT,
                           var_name=petab.OBSERVABLE_ID)

    measurements = measurements[
        measurements[petab.OBSERVABLE_ID].apply(lambda x: x.startswith('p'))
        | (
            measurements[petab.OBSERVABLE_ID].apply(
                lambda x: x.startswith('t')
            ) & (measurements[[petab.TIME] +
                              list(model.getFixedParameterIds())] == 0
                 ).all(axis=1)
        )
    ]

    measurements = measurements[
        measurements.apply(
            lambda m: (m[list(model.getFixedParameterIds())] != 0).any() |
                      (m[petab.TIME] == 0),
            axis=1
        )
    ]

    measurements[petab.SIMULATION_CONDITION_ID] = \
        measurements.apply(
            lambda x: f'sample_{x["Sample"]}' + ''.join(
                [
                    f'___{fp}__{x[fp]}'.replace('.', '_') if x[fp] > 0 else ''
                    for fp in model.getFixedParameterIds()
                ]
            ),
            axis=1
        )

    measurements[petab.PREEQUILIBRATION_CONDITION_ID] = \
        measurements['Sample'].apply(
            lambda x: f'sample_{x}'
        )

    measurements.drop(columns=['Sample'] + list(model.getFixedParameterIds()),
                      inplace=True)

    measurements[petab.OBSERVABLE_PARAMETERS] = \
        measurements[petab.OBSERVABLE_ID].apply(
            lambda x: f'{x}_scale;{x}_offset'
        )

    measurement_file = datadir / f'synthetic__{pathway_name}__measurements.tsv'

    # CONDITIONS
    condition_file = datadir / f'synthetic__{pathway_name}__conditions.tsv'
    conditions = pd.DataFrame({
        petab.CONDITION_ID:
            measurements[petab.SIMULATION_CONDITION_ID].unique()
    })
    for fp in model.getFixedParameterIds():
        conditions[fp] = conditions[petab.CONDITION_ID].apply(
            lambda x: float(next(
                (cond for cond in x.split('___')
                 if cond.startswith(fp)),
                f'{fp}__0.0'
            ).split('__')[1].replace('_', '.'))
        )
    conditions[petab.CONDITION_ID] = conditions[petab.CONDITION_ID].apply(
        lambda x: x.replace('__', '_')
    )
    conditions.set_index(petab.CONDITION_ID, inplace=True)
    conditions.to_csv(condition_file, sep='\t')
    for col in [petab.SIMULATION_CONDITION_ID,
                petab.PREEQUILIBRATION_CONDITION_ID]:
        measurements[col] = measurements[col].apply(
            lambda x: x.replace('__', '_')
        )
    measurements.to_csv(measurement_file, sep='\t')

    # OBSERVABLES
    observables = pd.DataFrame([{
        petab.OBSERVABLE_ID: obs_id,
        petab.OBSERVABLE_NAME: obs_name,
        petab.OBSERVABLE_FORMULA:
            f'log({"_".join(obs_id.split("_")[:-1])} * '
            f'observableParameter1_{obs_id} + '
            f'observableParameter2_{obs_id})',
    } for obs_id, obs_name in zip(model.getObservableIds(),
                                  model.getObservableNames())])
    observables[petab.NOISE_DISTRIBUTION] = petab.NORMAL
    observables[petab.NOISE_FORMULA] = '1.0'

    observable_file = datadir / f'synthetic__{pathway_name}__observables.tsv'
    observables.set_index(petab.OBSERVABLE_ID, inplace=True)
    observables.to_csv(observable_file, sep='\t')

    return measurement_file, condition_file, observable_file


def plot_embedding(embedding: np.ndarray, ax: plt.Axes):
    middle = int(np.floor(len(embedding) / 2))
    ax.plot(embedding[:middle, 0], embedding[:middle, 1], 'k*')
    ax.plot(embedding[middle:, 0], embedding[middle:, 1], 'r*')


def plot_pca_inputs(x: np.ndarray, embed_ax: plt.Axes,
                    vexpl_ax: plt.Axes = None):
    pca = decomposition.PCA(n_components=10)
    pca.fit(x)
    x_pca = pca.transform(x)

    middle = int(np.floor(len(x) / 2))
    embed_ax.plot(x_pca[:middle, 0], x_pca[:middle, 1], 'k*')
    embed_ax.plot(x_pca[middle:, 0], x_pca[middle:, 1], 'r*')

    if vexpl_ax is not None:
        vexpl_ax.plot(np.cumsum(pca.explained_variance_ratio_))
        vexpl_ax.set_xlabel('number of components')
        vexpl_ax.set_ylabel('cumulative explained variance')
