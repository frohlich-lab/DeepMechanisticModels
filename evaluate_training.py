import fire
import jax
import jax.random as jr
import pandas as pd
import pypesto

from common import (
    Conf,
    TRAINING_OUTFILE_RESULTS,
    TRAINED_BEST_MODELS,
    EVALUATION_TRAINING,
    FEATURES_OUTFILE,
    Wildcards,
    fig_dir,
    results_dir,
    test_samples,
    training_samples,
)
from dmm.analysis import evaluate_simulations
from dmm.training import create_pypesto_problem
from dmm.initialisation import load_models
from pypesto.store import OptimizationResultHDF5Reader
from util import load_petab_base_files


conf = fire.Fire(Conf)

samples_training = training_samples(Wildcards(conf.data, conf.samples))
samples_test = test_samples(Wildcards(conf.data, conf.samples))

outdir = fig_dir / conf.model / conf.data
indir = results_dir / conf.model / conf.data

samples = {
    "train": training_samples(Wildcards(conf.data, conf.samples)),
    "test": test_samples(Wildcards(conf.data, conf.samples)),
}


def evaluate_training(dataset, conf):
    evaluations = []

    # Initialise model skeleton and get CytofProblem
    model, cytof_problem = load_models(conf, dataset)

    # Create pypesto problem
    pypesto_problem = create_pypesto_problem(model)
    # Extract base objective
    obj = pypesto_problem.objective.base_objective

    # Define filepaths for training results and serialized model
    # infile = TRAINING_OUTFILE_RESULTS.format(**conf.__dict__)
    trained_model_file = TRAINED_BEST_MODELS.format(**conf.__dict__)

    # Load training results - TODO @GiacomoFabrini - do we really need these?
    # reader = OptimizationResultHDF5Reader(infile)
    # result = pypesto.Result(pypesto_problem)
    # result.optimize_result = reader.read().optimize_result

    # Load serialised best model
    petab_base_files = load_petab_base_files(conf, reweight=True)
    model.load(
        trained_model_file,
        cytof_problem,
        petab_base_files['measurement_table'],
        petab_base_files['observable_table'],
        petab_base_files['condition_table'],
        jr.PRNGKey(conf.job)
    )

    # Load input features (train/val) to evaluate trained model
    input_features = pd.read_csv(
        FEATURES_OUTFILE.format_map(
            dict(**conf.__dict__, dataset=dataset)
        ),
        index_col=0
    ).values

    evaluate_simulations(
        model=model,
        input_features=input_features,
        obj=obj,
        samples=samples[dataset],
        petab_problem=model.petab_importer.petab_problem,
        context=conf.context,
        split=conf.samples,
        dataset=dataset,
        job=conf.job,  # adding job here to produce one plot per multistart
        orth_reg_strategy=conf.orth_reg_strategy,
        l1reg_inflate=conf.l1reg_inflate,
        oreg_inflate=conf.oreg_inflate,
        l1reg_encode=conf.l1reg_encode,
        oreg_encode=conf.oreg_encode,
        latent_dim=conf.n_hidden,
        features=conf.features,
        outdir=outdir / "simulation",
        evaluations=evaluations,
        model_type="full",
    )

    return pd.DataFrame(evaluations)


for dataset in ("train", "test"):
    # clear jax cache to avoid error where jitted function uses input with shape of train
    # which differs from test
    jax.clear_caches()
    df = evaluate_training(dataset, conf)
    df.to_csv(EVALUATION_TRAINING.format(dataset=dataset, **conf.__dict__))
