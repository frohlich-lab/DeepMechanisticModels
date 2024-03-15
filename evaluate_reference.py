import fire
import numpy as np
import pandas as pd
import petab
from amici.petab_objective import rdatas_to_simulation_df

from common import (
    EVALUATION_REFERENCE,
    EVALUATION_REFERENCE_REG,
    MEASUREMENTS_FILE,
    OBSERVABLES_FILE,
    Wildcards,
    fig_dir,
    pretrain_dir,
    test_samples,
    training_samples,
)
from cytof.problem import CytofProblem
from dmm.analysis import process_simulation
from dmm.petab_subproblem import load_petab
from dmm.feature_selection import load_data
from dmm.plotting import plot_cross_samples, plot_single_sample
from dmm.pretraining import (
    generate_average_pretraining_problem,
    generate_per_sample_pretraining_problems,
)
from training_configuration import CONTEXTS_FEATURES
from util import Conf, load_petab_base_files

from train_regressors_baseline import train_pipeline


conf = fire.Fire(Conf)

outdir = fig_dir / conf.model / conf.data
indir = pretrain_dir / conf.model / conf.data

cross_sample_dir = outdir / "pretrain_cross_sample"
cross_sample_dir.mkdir(exist_ok=True, parents=True)

# TODO @GiacomoFabrini: NEED TO CHANGE "train" to encompass "train" and "validation" (currently called
    # "test") from the splits. Change "test" to be the untouched "test" set. This is to ensure
    # that MultiTaskLassoCV and MultiTaskElasticNetCV have the same learning opportunities in
    # CV than the full DMM (i.e. their CV should be performed on train+val, not on train only)
samples = {
    "train": training_samples(Wildcards(conf.data, conf.samples)),
    "test": test_samples(Wildcards(conf.data, conf.samples)),
}


def evaluate_pretraining_per_sample(dataset, conf):
    evaluations = []
    problem = CytofProblem(conf.model)
    petab_base_files = load_petab_base_files(conf)
    for sample in samples[dataset]:
        rfile = indir / f"{sample}.csv"
        if not rfile.exists():
            continue

        petab_base_importer = load_petab(
            problem,
            conf.data,
            **petab_base_files,
        )

        importer = generate_per_sample_pretraining_problems(
            petab_base_importer,
            problem,
            conf.data,
            sample,
        )

        problem_sample = importer.create_problem()
        df = pd.read_csv(rfile, index_col=[0])
        problem.apply_objective_settings(problem_sample.objective)

        ress = []
        fvals = []
        for ipar in range(len(df)):
            x = problem_sample.get_reduced_vector(
                df.values[ipar, :], problem_sample.x_free_indices
            )
            res = problem_sample.objective(x, return_dict=True)
            ress.append(res)
            fvals.append(res["fval"])

        # Convert the simulation to PEtab format.
        simulation_df = rdatas_to_simulation_df(
            ress[np.argmin(fvals)]["rdatas"],
            model=problem_sample.objective.amici_model,
            measurement_df=importer.petab_problem.measurement_df,
        )
        process_simulation(
            evaluations=evaluations,
            measurement_df=importer.petab_problem.measurement_df,
            simulation_df=simulation_df,
            context="none",
            sample=sample,
            model_type="per_sample",
            orth_reg_strategy="None", # not needed for regression
            job=None,  # not needed here
            l1reg_inflate=0.0,
            oreg_encode=0.0,
            l1reg_encode=0.0,
            oreg_inflate=0.0,
            hidden_layers=0,
            features="none",
        )

        plot_single_sample(
            importer.petab_problem.measurement_df,
            simulation_df,
            outdir / "simulation" / dataset / sample,
            sample,
            "per_sample",
        )
    return pd.DataFrame(evaluations)


def evaluate_average(dataset, conf):
    df_meas = pd.read_csv(
        MEASUREMENTS_FILE.format(**conf.__dict__), sep="\t", index_col=0
    )
    df_obs = pd.read_csv(
        OBSERVABLES_FILE.format(**conf.__dict__), sep="\t", index_col=0
    )
    df_meas = df_meas[
        df_meas[petab.OBSERVABLE_ID].apply(lambda x: x in df_obs.index)
    ]

    df_train = df_meas[
        df_meas[petab.PREEQUILIBRATION_CONDITION_ID].isin(samples["train"])
    ]

    df_train["condition"] = df_train[petab.SIMULATION_CONDITION_ID].apply(
        lambda x: x.split("__")[1]
    )
    gb_cols = [petab.OBSERVABLE_ID, "condition", petab.TIME]
    avg_model = (
        df_train[gb_cols + [petab.MEASUREMENT]]
        .groupby(gb_cols)
        .agg(np.nanmean)
    )

    df_sim = df_meas.copy()
    df_sim = df_sim.loc[
        df_sim[petab.PREEQUILIBRATION_CONDITION_ID].isin(samples[dataset]), :
    ]

    for ir, r in df_meas.iterrows():
        # pick closest time point to avoid issues with non-canonical time points
        candidates = avg_model.loc[
            (r.observableId, r[petab.SIMULATION_CONDITION_ID].split("__")[1]),
            petab.MEASUREMENT,
        ]
        df_sim.loc[ir, petab.MEASUREMENT] = candidates.iloc[
            np.argmin(np.abs(candidates.index - r.time))
        ]

    df_sim[petab.SIMULATION] = df_sim[petab.MEASUREMENT]

    plot_cross_samples(df_meas, df_sim, outdir / "simulation" / dataset, "avg")

    evaluations = []

    for sample in samples[dataset]:
        process_simulation(
            evaluations=evaluations,
            measurement_df=df_meas,
            simulation_df=df_sim,
            context="none",
            sample=sample,
            model_type="avg",
            orth_reg_strategy="None", # not needed for regression
            job=None,  # not needed here
            l1reg_inflate=0.0,
            oreg_encode=0.0,
            l1reg_encode=0.0,
            oreg_inflate=0.0,
            hidden_layers=0,
            features="none",
        )

    return pd.DataFrame(evaluations)


def evaluate_average_model(dataset, conf):
    df_meas = pd.read_csv(
        MEASUREMENTS_FILE.format(**conf.__dict__), sep="\t", index_col=0
    )
    df_obs = pd.read_csv(
        OBSERVABLES_FILE.format(**conf.__dict__), sep="\t", index_col=0
    )
    df_meas = df_meas[
        df_meas[petab.OBSERVABLE_ID].apply(lambda x: x in df_obs.index)
    ]

    problem = CytofProblem(conf.model)
    petab_base_files = load_petab_base_files(conf)
    rfile = indir / f"model_average_{conf.samples}.csv"

    petab_base_importer = load_petab(
        problem,
        conf.data,
        **petab_base_files,
    )

    importer = generate_average_pretraining_problem(
        petab_base_importer,
        problem,
        conf.data,
        training_samples(Wildcards(conf.data, conf.samples))
        if dataset == "train"
        else test_samples(Wildcards(conf.data, conf.samples)),
    )
    problem_sample = importer.create_problem()
    df = pd.read_csv(rfile, index_col=[0])
    problem.apply_objective_settings(problem_sample.objective)

    ress = []
    fvals = []
    for ipar in range(len(df)):
        x = problem_sample.get_reduced_vector(
            df.values[0, :], problem_sample.x_free_indices
        )
        res = problem_sample.objective(x, return_dict=True)
        ress.append(res)
        fvals.append(res["fval"])

    # Convert the simulation to PEtab format.
    avg_model = rdatas_to_simulation_df(
        ress[np.argmin(fvals)]["rdatas"],
        model=problem_sample.objective.amici_model,
        measurement_df=importer.petab_problem.measurement_df,
    )

    avg_model[petab.SIMULATION_CONDITION_ID] = df_meas[
        petab.SIMULATION_CONDITION_ID
    ]
    avg_model[petab.PREEQUILIBRATION_CONDITION_ID] = df_meas[
        petab.PREEQUILIBRATION_CONDITION_ID
    ]

    df_meas = df_meas[
        df_meas[petab.PREEQUILIBRATION_CONDITION_ID].isin(samples[dataset])
    ]
    avg_model = avg_model[
        avg_model[petab.PREEQUILIBRATION_CONDITION_ID].isin(samples[dataset])
    ]

    plot_cross_samples(
        df_meas, avg_model, outdir / "simulation" / dataset, "avg_model"
    )

    evaluations = []

    for sample in samples[dataset]:
        process_simulation(
            evaluations=evaluations,
            measurement_df=df_meas,
            simulation_df=avg_model,
            context="none",
            sample=sample,
            model_type="avg_model",
            orth_reg_strategy="None", # not needed for regression
            job=None,  # not needed
            l1reg_inflate=0.0,
            oreg_encode=0.0,
            l1reg_encode=0.0,
            oreg_inflate=0.0,
            hidden_layers=0,
            features="none",
        )

    return pd.DataFrame(evaluations)


def evaluate_standard_regression(
        dataset,
        conf,
        samples,
        context,
        mode,  # 'linreg', 'lasso', 'elasticnet'
        trained_pipeline,
        features_train,
):

    # Check the regressors have been trained
    if trained_pipeline is None:
        raise ValueError("No trained_pipeline provided for this regressor!")
    elif (dataset == "test") and (features_train is None):
        raise ValueError(f"No features_train provided for {dataset} evaluation!")

    # Load petab files
    petab_base_files = load_petab_base_files(conf, reweight=False)
    # so far data has not been reweighed when evaluating references, only for training
    del petab_base_files["condition_table"]

    # Subset to "train"/"test"
    samples_eval = samples[dataset]

    # Load input and output data
    input_data, _ = load_data(
        contextualization=context,
        samples=samples_eval,
        features=features_train if dataset=="test" else None,
        **petab_base_files,
    )
    output_data, _ = load_data(
        contextualization="cytof_dynamic",
        samples=samples_eval,
        features=None,
        **petab_base_files,
    )

    # Process regression output/predictions (reg_pred) and output data before plotting and evaluating simulations
    # Convert into pandas dataframe with same index and column headers as output_test
    # Then process to use with plot_cross_samples() and process_simulation()
    # Finally drop index and rename column from 0 to 'simulation' to use in process_simulation()
    reg_pred = pd.DataFrame(
        trained_pipeline.predict(input_data),
        index=output_data.index,
        columns=output_data.columns
    ).T.stack().reset_index().sort_values(
        by=[
            'preequilibrationConditionId',
            'observableId',
            'simulationConditionId',
            'time'
        ]
    ).reset_index().drop(columns='index').rename(columns={0: "simulation"})

    # output_data
    # Column needs renaming from 0 to "measurement" for use in process_simulation()
    output_data = output_data.T.stack().reset_index().sort_values(
        by=[
            'preequilibrationConditionId',
            'observableId',
            'simulationConditionId',
            'time'
        ]
    ).reset_index().drop(columns='index').rename(columns={0: "measurement"})


    # Produce plots to analyse performance
    # import original output data as in avg/avg_model
    df_meas = pd.read_csv(
        MEASUREMENTS_FILE.format(**conf.__dict__), sep="\t", index_col=0
    )
    df_obs = pd.read_csv(
        OBSERVABLES_FILE.format(**conf.__dict__), sep="\t", index_col=0
    )
    df_meas = df_meas[
        df_meas[petab.OBSERVABLE_ID].apply(lambda x: x in df_obs.index)
    ]
    # Groupby to average replicates as done for regression output
    df_meas = df_meas.groupby(
        ['observableId', 'preequilibrationConditionId', 'time', 'simulationConditionId']).agg(
        {'measurement': 'mean', 'noiseParameters': 'mean'}).reset_index()
    # Sort to make comparable with output_train
    df_meas = df_meas.sort_values(
        by=['observableId', 'preequilibrationConditionId', 'simulationConditionId', 'time'])

    # Subset to cell lines that are in output_data (i.e. output_train/output_test)
    df_meas = df_meas[
        df_meas.preequilibrationConditionId.isin(output_data.preequilibrationConditionId)
    ].reset_index().drop(columns='index')

    # process simulation condition id
    df_meas[petab.SIMULATION_CONDITION_ID] = df_meas[
        petab.SIMULATION_CONDITION_ID
    ].apply(lambda x: x.split("__")[1])

    # reorder columns as in output_train
    df_meas = df_meas[['observableId', 'simulationConditionId',
                       'time', 'preequilibrationConditionId',
                       'measurement', 'noiseParameters']]

    # Plot -- reg_pred is either reg_pred_train or reg_pred_test
    plot_name = mode + "_" + context
    plot_cross_samples(
        df_meas, reg_pred, outdir / "simulation" / dataset, plot_name
    )

    # Process simulations/regressions, i.e. produce CSVs with residuals
    evaluations = []

    for sample in samples[dataset]:
        process_simulation(
            evaluations=evaluations,
            measurement_df=output_data,
            simulation_df=reg_pred,
            context=context,
            sample=sample,
            model_type=mode,
            orth_reg_strategy="None",  # not needed for regression
            job=None, # not needed here
            l1reg_inflate=0.0,
            oreg_encode=0.0,
            l1reg_encode=0.0,
            oreg_inflate=0.0,
            hidden_layers=0,
            features="none",
        )

    return pd.DataFrame(evaluations)

# Build and train regressors (linear regression, lasso, elasticnet)
print('Building pipelines and training estimators...')
trained_pipelines = {}
features_train = {}
for context, _ in CONTEXTS_FEATURES:
    trained_pipelines[context], features_train[context] = {}, {}
    for mode in ['linreg', 'lasso', 'elasticnet']:
        trained_pipelines[context][mode], features_train[context][mode] = train_pipeline(
            pipeline_steps= ["pca", mode],
            conf=conf,
            context=context,
            samples_train=samples["train"],
        )
        print(f'Estimator {mode} trained on context {context}')

# Evaluate references/baselines
for dataset in ["train", "test"]:

    # model average ("avg_model")
    df = evaluate_average_model(dataset, conf)
    df.to_csv(
        EVALUATION_REFERENCE.format(
            **conf.__dict__,
            dataset=dataset,
            mode="avg_model",
        )
    )

    # average -- this looks NOT to be in use at the moment (only avg_model)
    df = evaluate_average(dataset, conf)
    df.to_csv(
        EVALUATION_REFERENCE.format(
            **conf.__dict__,
            dataset=dataset,
            mode="average",
        )
    )

    # per sample ("sample")
    df = evaluate_pretraining_per_sample(dataset, conf)
    df.to_csv(
        EVALUATION_REFERENCE.format(
            **conf.__dict__,
            dataset=dataset,
            mode="per_sample",
        )
    )

    # Regressors ("linreg", "lasso", "elasticnet")
    for context, _ in CONTEXTS_FEATURES:
        # Regression baseline first
        for mode in ['linreg', 'lasso', 'elasticnet']:
            df = evaluate_standard_regression(
                dataset=dataset,
                conf=conf,
                samples=samples,
                context=context,
                mode=mode,
                trained_pipeline=trained_pipelines[context][mode],
                features_train=features_train[context][mode]
            )

            df.to_csv(
                EVALUATION_REFERENCE_REG.format(
                    model=conf.model,
                    data=conf.data,
                    samples=conf.samples,
                    dataset=dataset,
                    mode=mode,
                    context=context,
                )
            )
