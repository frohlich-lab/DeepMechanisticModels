import itertools as itt

import amici
import numpy as np
import petab

from dmm.initialisation import setup_models as  load_model  # TODO: still in use?!
from ..autoencoder import DeepMechanisticModel, load_petab
from ..generate_data import generate_synthetic_data
from training import train
from training_helper_funcs import generate_pypesto_objective

pathway_model = "FLT3_MAPK"


def test_model_compilation():
    """
    Test that we can load and simulate the mechanistic model in AMICI
    """
    model, solver = load_model("pw_" + pathway_model)
    amici.runAmiciSimulation(model, solver)


def test_petab_loading():
    """
    Test that we can load the mechanistic model plus data in PEtab
    """
    datafiles = generate_synthetic_data(pathway_model)
    petab_importer = load_petab(datafiles, "pw_" + pathway_model, 1.0)
    petab.lint.lint_problem(petab_importer.petab_problem)


def test_pypesto_objective():
    """
    Test that we can load and evaluate the theano loss function and gradient
    for the full autoencoder model, checks accuracy of gradient via finite
    differences
    """
    datafiles = generate_synthetic_data(pathway_model)
    n_hidden = 2

    mae = DeepMechanisticModel(n_hidden, datafiles, pathway_model)
    objective = generate_pypesto_objective(mae)
    x = np.random.random((mae.n_encoder_pars + mae.n_kin_params,))
    x[0 : mae.n_encoder_pars] /= 10
    assert np.isfinite(objective.get_fval(x))
    assert not any(np.isnan(objective.get_grad(x)))
    fd_df = objective.check_grad(
        x,
        eps=1e-3,
        x_indices=list(
            itt.chain(
                range(5), range(mae.n_encoder_pars, mae.n_encoder_pars + 5)
            )
        ),
    )
    assert (fd_df["abs_err"] < 1e-2).all()


def test_pypesto_optimization():
    """
    Test that we can minimize the loss using pypesto
    """
    datafile = generate_synthetic_data(pathway_model)
    n_hidden = 2

    mae = DeepMechanisticModel(n_hidden, datafile, pathway_model)
    train(mae, maxiter=5)
