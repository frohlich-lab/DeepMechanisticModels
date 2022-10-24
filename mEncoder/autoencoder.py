import aesara
import aesara.tensor as aet
import numpy as np
import pandas as pd

import petab
import pypesto

from typing import Tuple, Sequence, Optional
from pathlib import Path
from sklearn.decomposition import PCA, SparsePCA
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from . import MODEL_FEATURE_PREFIX, apply_objective_settings
from .encoder import AutoEncoder
from .petab_subproblem import load_petab

AFunction = aesara.compile.Function


class MechanisticAutoEncoder(AutoEncoder):
    def __init__(self,
                 n_hidden: int,
                 datafiles: Tuple[Path, Path, Path],
                 pathway_name: str,
                 samples: Sequence[str],
                 l2reg: float = 1,
                 features: Optional[Sequence[str]] = None,
                 imputer: Optional[KNNImputer] = None,
                 scaler: Optional[StandardScaler] = None,
                 pca: Optional[PCA] = None):
        """
        loads the mechanistic model as theano operator with loss as output and
        decoder output as input

        :param datafiles:
            tuple of paths to measurements, conditions and observables files

        :param pathway_name:
            name of pathway to use for model

        :param n_hidden:
            number of nodes in the hidden layer of the encoder

        :param l2reg:
            currently this parameter only influences the strength of l2
            regularization on the inflate layer (the respective gaussian
            prior has its standard deviation defined based on the value of
            this parameter). For bounded inflate functions, this parameter
            is also intended to rescale the inputs accordingly.

        """
        self.data_name = '__'.join(
            datafiles[0].stem.split('__')[:-1]
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

        if features:
            # for prediction, use feature set computed on training data
            input_data = input_data[features]
        else:
            # for training, compute feature set
            # filter too many nans
            input_data = input_data.loc[
                :,
                input_data.isna().sum() < input_data.shape[0] * 0.2
            ]

        self.features = list(input_data.columns)

        # subset samples
        input_data = input_data.loc[samples, :]

        self.l2reg = l2reg
        self.petab_importer = load_petab(datafiles, 'pw_' + pathway_name,
                                         l2reg, samples)

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

        if scaler is None:
            self.scaler = StandardScaler(with_std=False)
            self.scaler.fit(imputed)
        else:
            self.scaler = scaler

        # zero center input data, this is equivalent to estimating biases
        # for linear autoencoders
        # https://link.springer.com/article/10.1007/BF00332918
        # https://arxiv.org/pdf/1901.08168.pdf
        # note: transform also normalizes to unit standard deviation
        input_data = pd.DataFrame(
            self.scaler.transform(imputed),
            index=input_data.index,
            columns=input_data.columns
        )

        # generate PCA embedding for feature selection
        if pca is None:
            # use n_comps such that 90% of variance is explained
            n_pca = np.nonzero(np.cumsum(PCA(
                n_components=input_data.shape[0]
            ).fit(input_data).explained_variance_ratio_) > 0.9)[0][0] + 1
            pca = PCA(n_components=n_pca, whiten=True).fit(input_data)

        self.pca = pca

        self.data_pca = self.pca.transform(input_data)

        self.n_samples, self.n_features = self.data_pca.shape
        self.n_model_inputs = int(sum(name.startswith(MODEL_FEATURE_PREFIX)
                                      for name in
                                      self.pypesto_subproblem.x_names) /
                                  self.n_samples)
        self.n_kin_params = \
            self.pypesto_subproblem.dim - self.n_model_inputs * self.n_samples

        self.sample_names = list(input_data.index)
        self.data_cols = [f'PC{i}' for i in range(self.data_pca.shape[1])]
        super().__init__(input_data=self.data_pca, n_hidden=n_hidden,
                         n_params=self.n_model_inputs)

        apply_objective_settings(self.pypesto_subproblem, pathway_name)
        if isinstance(self.pypesto_subproblem.objective,
                      pypesto.objective.AmiciObjective):
            amici_objective = self.pypesto_subproblem.objective
        else:
            amici_objective = self.pypesto_subproblem.objective._objectives[0]
        amici_objective.n_threads = 1

        self.x_names = self.x_names + [
            name for ix, name in enumerate(self.pypesto_subproblem.x_names)
            if not name.startswith(MODEL_FEATURE_PREFIX)
            and ix in self.pypesto_subproblem.x_free_indices
        ]

        # assemble input to model aesara op
        self.x = aet.specify_shape(
            aet.vector('x'),
            (self.n_kin_params + self.n_encoder_pars,)
        )
        encoded_pars = self.encode_params(self.x[:-self.n_kin_params])
        self.model_pars = aet.concatenate([
            self.x[-self.n_kin_params:],
            aet.reshape(encoded_pars.T,
                        (self.n_model_inputs * self.n_samples,))
        ], axis=0)

        # assemble embedding to model aesara op for pretraining
        self.x_embedding = aet.specify_shape(
            aet.vector('x_embedding'),
            (self.n_kin_params + self.n_model_inputs * self.n_hidden,)
        )
        inflated_pars = self.inflate_params_restricted(
            self.data_pca[:, :self.n_hidden],
            self.x_embedding[:-self.n_kin_params]
        )
        self.embedding_model_pars = aet.concatenate([
            self.x_embedding[-self.n_kin_params:],
            aet.reshape(inflated_pars.T,
                        (self.n_model_inputs * self.n_samples,))],
            axis=0
        )

        self.embedding_fun = self.encode(self.x)
