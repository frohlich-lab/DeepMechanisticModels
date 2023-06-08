from typing import List, Optional, Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pandas as pd
import petab
import pypesto.petab
from jax.config import config
from sklearn.decomposition import PCA, SparsePCA
from sklearn.impute import KNNImputer

from . import MODEL_FEATURE_PREFIX
from .encoder import AutoEncoder
from .petab_subproblem import load_petab
from .problem import Problem

config.update("jax_enable_x64", True)


def contextualize_measurements(
    measurement_table: pd.DataFrame,
    observable_table: pd.DataFrame,
    contextualization: str,
) -> pd.DataFrame:
    input_measurements = measurement_table.copy()

    if contextualization.startswith("transcriptomics"):
        input_measurements = input_measurements[
            input_measurements["measurementType"] == "transcriptomics"
        ]
    elif contextualization.startswith("proteomics"):
        input_measurements = input_measurements[
            input_measurements["measurementType"] == "proteomics"
        ]
    elif contextualization.startswith("cytof"):
        input_measurements = input_measurements[
            input_measurements["measurementType"] == "cytof"
        ]

    if contextualization.startswith(
        ("transcriptomics", "proteomics", "cytof_init")
    ):
        input_measurements = input_measurements[
            input_measurements[petab.TIME] == 0
        ]

    if contextualization.startswith("cytof_dynamic"):
        input_measurements = input_measurements[
            input_measurements[petab.OBSERVABLE_ID].isin(
                list(observable_table.index)
            )
        ]
        input_measurements[petab.SIMULATION_CONDITION_ID] = input_measurements[
            petab.SIMULATION_CONDITION_ID
        ].apply(lambda x: x.split("__")[1])

        pivot_columns = (
            petab.OBSERVABLE_ID,
            petab.SIMULATION_CONDITION_ID,
            petab.TIME,
        )
    elif contextualization.startswith("cytof_init"):
        input_measurements = input_measurements[
            input_measurements[petab.SIMULATION_CONDITION_ID].apply(
                lambda x: x.endswith("__EGF")
            )
        ]
        pivot_columns = petab.OBSERVABLE_ID
    else:
        pivot_columns = petab.OBSERVABLE_ID

    input_data = input_measurements.pivot_table(
        index=petab.PREEQUILIBRATION_CONDITION_ID,
        columns=pivot_columns,
        values=petab.MEASUREMENT,
        aggfunc=np.nanmean,
    )
    return input_data


class DeepMechanisticModel(AutoEncoder):
    data_name: str = eqx.static_field()
    pathway_name: str = eqx.static_field()
    features: List[str] = eqx.static_field()
    imputer: KNNImputer = eqx.static_field()
    pca: PCA = eqx.static_field()
    data_pca: np.ndarray = eqx.static_field()
    n_model_inputs: int = eqx.static_field()
    n_kin_params: int = eqx.static_field()
    n_samples: int = eqx.static_field()
    sample_names: List[str] = eqx.static_field()
    x_names: List[str] = eqx.static_field()
    data_cols: List[str] = eqx.static_field()
    l2reg: float = eqx.static_field()
    petab_importer: pypesto.petab.PetabImporterPysb = eqx.static_field()
    pypesto_subproblem: pypesto.Problem = eqx.static_field()

    def __init__(
        self,
        problem: Problem,
        dataset: str,
        n_latent: int,
        measurement_table: pd.DataFrame,
        observable_table: pd.DataFrame,
        condition_table: pd.DataFrame,
        contextualization: str,
        samples: Sequence[str],
        l2reg: float = 0.0,
        features: Optional[Sequence[str]] = None,
        imputer: Optional[KNNImputer] = None,
        pca: Optional[PCA] = None,
        n_threads=1,
    ):
        """
        loads the mechanistic model as theano operator with loss as output and
        decoder output as input

        :param pathway_name:
            name of pathway to use for model

        :param n_latent:
            number of nodes in the hidden layer of the encoder

        :param l2reg:
            currently this parameter only influences the strength of l2
            regularization on the inflate layer (the respective gaussian
            prior has its standard deviation defined based on the value of
            this parameter). For bounded inflate functions, this parameter
            is also intended to rescale the inputs accordingly.

        """
        self.data_name = dataset
        self.pathway_name = problem.pathway_name

        input_data = contextualize_measurements(
            measurement_table, observable_table, contextualization
        )

        # subset samples
        input_data = input_data.loc[samples, :]

        if features:
            # for prediction, use feature set computed on training data
            input_data = input_data[features]
        else:
            # for training, compute feature set
            # filter too many nans
            input_data = input_data.loc[
                :, input_data.isna().sum() / input_data.shape[0] < 0.2
            ]

        self.features = list(input_data.columns)

        self.l2reg = l2reg
        self.petab_importer = load_petab(
            problem,
            dataset,
            l2reg,
            measurement_table,
            condition_table,
            observable_table,
            samples,
        )

        self.pypesto_subproblem = self.petab_importer.create_problem()

        # extract sample names, ordering of those is important since samples
        # must match when reshaping the inflated matrix
        petab_samples = []
        for name in self.pypesto_subproblem.x_names:
            if not name.startswith(MODEL_FEATURE_PREFIX):
                continue

            sample = name.split("__")[-1]
            if sample not in petab_samples and sample in input_data.index:
                petab_samples.append(sample)

        input_data = input_data.loc[petab_samples, :]

        # impute missing values
        if imputer is None:
            # training, fit imputer to training data
            self.imputer = KNNImputer()
            self.imputer.fit(input_data.values)
        else:
            # prediction, load imputer from training data
            self.imputer = imputer

        imputed = self.imputer.transform(input_data.values)

        # zero center input data, this is equivalent to estimating biases
        # for linear autoencoders
        # https://link.springer.com/article/10.1007/BF00332918
        # https://arxiv.org/pdf/1901.08168.pdf
        # note: transform also normalizes to unit standard deviation
        input_data = pd.DataFrame(
            imputed,
            index=input_data.index,
            columns=input_data.columns,
        )

        # generate PCA embedding for feature selection
        if pca is None:
            # use n_comps such that 90% of variance is explained

            var_expl = (
                PCA(n_components=input_data.shape[0])
                .fit(input_data)
                .explained_variance_ratio_
            )
            n_pca = np.nonzero(np.cumsum(var_expl) > 0.9)[0][0] + 1

            if contextualization.endswith("_pca"):
                pca = PCA(n_components=max(n_pca, n_latent), whiten=True).fit(
                    input_data
                )
            else:
                alpha = float(contextualization.split("_")[-1])
                spca = SparsePCA(n_components=n_pca, alpha=alpha)
                spca.fit(input_data)
                X = input_data.values
                P = spca.components_.T  # loadings
                T = spca.transform(X)  # score
                Xc = X - X.mean(axis=0)  # center data
                explained_variance = np.trace(P @ T.T @ T @ P.T)
                total_variance = np.trace(Xc.T @ Xc)
                frac_expl_var = explained_variance / total_variance
                print(f"Fraction of explained variance: {frac_expl_var}")

                pca = spca

        self.pca = pca
        self.data_pca = self.pca.transform(input_data)

        self.n_samples, self.n_features = self.data_pca.shape
        self.n_model_inputs = int(
            sum(
                name.startswith(MODEL_FEATURE_PREFIX)
                for name in self.pypesto_subproblem.x_names
            )
            / self.n_samples
        )
        self.n_kin_params = (
            self.pypesto_subproblem.dim - self.n_model_inputs * self.n_samples
        )

        self.sample_names = list(input_data.index)
        self.data_cols = [f"PC{i}" for i in range(self.data_pca.shape[1])]
        super().__init__(
            input_data=self.data_pca,
            n_latent=n_latent,
            n_params=self.n_model_inputs,
        )

        problem.apply_objective_settings(
            self.pypesto_subproblem.objective, n_threads=n_threads
        )

        self.x_names = self.x_names + [
            name
            for ix, name in enumerate(self.pypesto_subproblem.x_names)
            if not name.startswith(MODEL_FEATURE_PREFIX)
            and ix in self.pypesto_subproblem.x_free_indices
        ]

    def embedding(self, params: np.ndarray) -> np.ndarray:
        encode_weights, inflate_weights, kin_params = jnp.split(
            params,
            np.array(
                (
                    self.n_encode_weights,
                    self.n_inflate_weights + self.n_encode_weights,
                )
            ),
        )
        return jnp.concatenate(
            [
                kin_params,
                self.inflate_params(
                    self.encode(encode_weights), inflate_weights
                ).flatten(),
            ]
        )

    def inflate(self, params: jnp.ndarray) -> jnp.ndarray:
        inflate_weights, kin_params = jnp.split(
            params, np.array((self.n_inflate_weights,))
        )
        return jnp.concatenate(
            [
                kin_params,
                self.inflate_params(
                    self.data_pca[:, : self.n_latent], inflate_weights
                ).flatten(),
            ]
        )
