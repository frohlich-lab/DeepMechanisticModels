import os

import aesara
import aesara.tensor as aet
import numpy as np
import pandas as pd

import petab
import pypesto

from typing import Tuple, Sequence
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor

from . import MODEL_FEATURE_PREFIX, apply_objective_settings
from .encoder import AutoEncoder
from .petab_subproblem import load_petab, filter_observables

AFunction = aesara.compile.Function


class MechanisticAutoEncoder(AutoEncoder):
    def __init__(self,
                 n_hidden: int,
                 datafiles: Tuple[str, str, str],
                 pathway_name: str,
                 samples: Sequence[str],
                 par_modulation_scale: float = 1):
        """
        loads the mechanistic model as theano operator with loss as output and
        decoder output as input

        :param datafiles:
            tuple of paths to measurements, conditions and observables files

        :param pathway_name:
            name of pathway to use for model

        :param n_hidden:
            number of nodes in the hidden layer of the encoder

        :param par_modulation_scale:
            currently this parameter only influences the strength of l2
            regularization on the inflate layer (the respective gaussian
            prior has its standard deviation defined based on the value of
            this parameter). For bounded inflate functions, this parameter
            is also intended to rescale the inputs accordingly.

        """
        self.data_name = '__'.join(
            os.path.splitext(
                os.path.basename(datafiles[0])
            )[0].split('__')[:-1]
        )
        self.pathway_name = pathway_name

        full_measurements = pd.read_csv(datafiles[0], index_col=0, sep='\t')

        baseline_measurements = full_measurements[
            full_measurements[petab.TIME] == 0
        ]

        baseline_measurements = baseline_measurements[
            baseline_measurements[petab.SIMULATION_CONDITION_ID] ==
            baseline_measurements[petab.PREEQUILIBRATION_CONDITION_ID]
        ]

        input_data = baseline_measurements.pivot_table(
            index=petab.SIMULATION_CONDITION_ID,
            columns=petab.OBSERVABLE_ID,
            values=petab.MEASUREMENT,
            aggfunc=np.nanmean
        )

        input_data = input_data.loc[
            :,
            input_data.isna().sum() < input_data.shape[0] * 0.2
        ]

        # filter highly variable
        input_data = input_data.loc[
            :, input_data.var() > input_data.var().max() / 10
        ]

        # subset samples
        input_data = input_data.loc[samples, :]

        self.par_modulation_scale = par_modulation_scale
        self.petab_importer = load_petab(datafiles, 'pw_' + pathway_name,
                                         par_modulation_scale, samples)

        self.pypesto_subproblem = self.petab_importer.create_problem()

        # extract sample names, ordering of those is important since samples
        # must match when reshaping the inflated matrix
        petab_samples = []
        for name in self.pypesto_subproblem.x_names:
            if not name.startswith(MODEL_FEATURE_PREFIX):
                continue

            sample = name.split('__')[-1]
            if sample not in petab_samples and sample in input_data.index:
                petab_samples.append(sample)

        # pca_pert = PCA(n_components=1)
        # pert_data = pert_data.loc[pert_samples, :]
        # pert_decomp = pca_pert.fit_transform(pert_data.values)
        input_data = input_data.loc[petab_samples, :]

        pert_measurements = full_measurements[
            full_measurements[petab.TIME] > 0
        ]

        pert_measurements = pert_measurements[
            pert_measurements[petab.OBSERVABLE_ID].apply(
                lambda x: x in self.petab_importer.petab_problem.observable_df.index
            )
        ]

        pert_measurements['feature'] = pert_measurements.apply(
            lambda x: '__'.join([x[petab.OBSERVABLE_ID],
                                 x[petab.SIMULATION_CONDITION_ID].split('__')[
                                     1], str(x[petab.TIME])]), axis=1
        )

        pert_data = pert_measurements.pivot_table(
            index=petab.PREEQUILIBRATION_CONDITION_ID,
            columns='feature',
            values=petab.MEASUREMENT,
            aggfunc=np.nanmean
        ).dropna(axis=1)

        # pca_pert = PCA(n_components=1)
        # pert_decomp = pca_pert.fit_transform(pert_data.values)

        # impute missing values
        if len(samples) > 1:
            imputer = KNNImputer()
            input_data = pd.DataFrame(imputer.fit_transform(input_data.values),
                                      columns=input_data.columns,
                                      index=input_data.index)
        else:
            input_data = input_data.dropna(axis=1)

        # zero center input data, this is equivalent to estimating biases
        # for linear autoencoders
        # https://link.springer.com/article/10.1007/BF00332918
        # https://arxiv.org/pdf/1901.08168.pdf
        input_data -= input_data.mean()

        #regr = RandomForestRegressor()
        #pert_fit = regr.fit(input_data.values, pert_decomp)

        self.n_visible = input_data.shape[1]
        self.n_samples = input_data.shape[0]
        self.n_model_inputs = int(sum(name.startswith(MODEL_FEATURE_PREFIX)
                                      for name in
                                      self.pypesto_subproblem.x_names) /
                                  self.n_samples)
        self.n_kin_params = \
            self.pypesto_subproblem.dim - self.n_model_inputs * self.n_samples

        self.sample_names = list(input_data.index)
        self.data_cols = list(input_data.columns)
        super().__init__(input_data=input_data.values, n_hidden=n_hidden,
                         n_params=self.n_model_inputs)

        # generate PCA embedding for pretraining
        pca = PCA(n_components=np.min([self.n_hidden, self.n_samples]))
        self.data_pca = pca.fit_transform(self.data)

        apply_objective_settings(self.pypesto_subproblem, pathway_name)
        if isinstance(self.pypesto_subproblem.objective,
                      pypesto.objective.AmiciObjective):
            amici_objective = self.pypesto_subproblem.objective
        else:
            amici_objective = self.pypesto_subproblem.objective._objectives[0]
        amici_objective.n_threads = 4

        self.x_names = self.x_names + [
            name for ix, name in enumerate(self.pypesto_subproblem.x_names)
            if not name.startswith(MODEL_FEATURE_PREFIX)
            and ix in self.pypesto_subproblem.x_free_indices
        ]

        # assemble input to model theano op
        self.x = aet.specify_shape(
            aet.vector('x'),
            (self.n_kin_params + self.n_encoder_pars + self.n_inflate_weights,)
        )
        encoded_pars = self.encode_params(self.x[:-self.n_kin_params])
        self.model_pars = aet.concatenate([
            self.x[-self.n_kin_params:],
            aet.reshape(encoded_pars.T,
                        (self.n_model_inputs * self.n_samples,))],
            axis=0
        )

        # assemble embedding to model theano op for pretraining
        self.x_embedding = aet.specify_shape(
            aet.vector('x'),
            (self.n_kin_params + self.n_model_inputs * self.n_samples,)
        )
        inflated_pars = self.inflate_params_restricted(
            self.data_pca, self.x_embedding[:-self.n_kin_params]
        )
        self.embedding_model_pars = aet.concatenate([
            self.x_embedding[-self.n_kin_params:],
            aet.reshape(inflated_pars.T,
                        (self.n_model_inputs * self.n_samples,))],
            axis=0
        )

        self.embedding_fun = self.encode(self.x)
