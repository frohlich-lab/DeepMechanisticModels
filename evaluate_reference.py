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
from dmm.feature_selection import build_preprocesser, load_data
from dmm.plotting import plot_cross_samples, plot_single_sample
from dmm.pretraining import (
    generate_average_pretraining_problem,
    generate_per_sample_pretraining_problems,
)
from training_configuration import CONTEXTS_FEATURES
from util import Conf, load_petab_base_files

conf = fire.Fire(Conf)

outdir = fig_dir / conf.model / conf.data
indir = pretrain_dir / conf.model / conf.data

cross_sample_dir = outdir / "pretrain_cross_sample"
cross_sample_dir.mkdir(exist_ok=True, parents=True)

# REMEMBER: NEED TO CHANGE "train" to encompass "train" and "validation" (currently called
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
            job=None, # not needed here
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
            job=None, # not needed here
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
            job = None, # not needed
            l1reg_inflate=0.0,
            oreg_encode=0.0,
            l1reg_encode=0.0,
            oreg_inflate=0.0,
            hidden_layers=0,
            features="none",
        )

    return pd.DataFrame(evaluations)


def evaluate_standard_regression(dataset, conf, context,
                                 mode, # 'linreg' for LinearRegression/ 'lasso' for MultiTaskLassoCV/ 'elasticnet' for MultiTaskElasticNetCV
                                 trained_pipeline = None,
                                 features_train = None,
                                 sample_weighing = False):
    from sklearn.linear_model import (LinearRegression,
                                      MultiTaskLassoCV,
                                      MultiTaskElasticNetCV)
    #from sklearn.multioutput import MultiOutputRegressor # to implement multi-task linear regression

    def build_pipeline(
            steps_list: str,
            input_data: np.ndarray,
            sample_weighing: bool
    ):
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import KNNImputer
        from sklearn.decomposition import PCA
        from sklearn.pipeline import Pipeline
        from sklearn import set_config
        set_config(enable_metadata_routing=True)

        # standard steps: scaling, imputation via KNN
        steps = [
            ("scaler", StandardScaler()),
            ("imputer", KNNImputer()),
        ]
        # PCA + one among linear regression/lasso/elasticnet
        if (steps_list is not None) and (len(steps_list)>0):
            for step in steps_list:
                if step == "pca":
                    inputs = Pipeline(steps).fit_transform(input_data)
                    var_expl = (
                        PCA(n_components=input_data.shape[0])
                        .fit(inputs)
                        .explained_variance_ratio_
                    )
                    n_pca = np.nonzero(np.cumsum(var_expl) > 0.95)[0][0] + 1
                    steps.append(("pca", PCA(n_components=n_pca)))
                elif step == "linreg":
                    # steps.append(("linreg", MultiOutputRegressor(LinearRegression().set_fit_request(sample_weight=True))))
                    # seems like LinearRegression automatically supports MultiOutput/MultiTask
                    steps.append(("linreg", LinearRegression().set_fit_request(sample_weight=sample_weighing)))
                elif step == "lasso":
                    steps.append(("lasso", MultiTaskLassoCV(cv=5, n_alphas=20).set_fit_request(sample_weight=sample_weighing)))
                elif step == "elasticnet":
                    steps.append(("elasticnet", MultiTaskElasticNetCV(cv=5, n_alphas=20).set_fit_request(sample_weight=sample_weighing)))
                else:
                    raise ValueError(f"Unknown step {step}")
        else:
            if steps_list is None:
                raise TypeError("Expected type list for steps_list, got None type")
            elif len(steps_list) == 0:
                raise ValueError("List of pipeline steps is empty")

        return Pipeline(steps)


    # Load petab files
    petab_base_files = load_petab_base_files(conf, reweight=False)
    # so far data has not been reweighed when evaluating references, only for training
    del petab_base_files["condition_table"]

    if dataset == "train":
        # Fetch and load training samples -- always needed to train
        samples_train = samples["train"]

        # Load input and output training data
        input_train, features_train = load_data(
            contextualization=context,
            samples=samples_train,
            features=None,
            **petab_base_files,
            io_mode='input',
        )
        output_train, weights, targets_train = load_data(
            contextualization="cytof_dynamic",
            samples=samples_train,
            features=None,
            **petab_base_files,
            io_mode='output',
        )

        if trained_pipeline is None:
            # Build pipeline: add PCA and estimator to scaling and imputation steps
            pipeline = build_pipeline(steps_list= ["pca", mode],
                                      input_data = input_train,
                                      sample_weighing=sample_weighing)
            # Select whether to use sample weights - currently NOT using them
            if (mode in ['linreg']) and (sample_weighing == True): # others do not support sample_weight
                # Aggregate sample_weights - right now sample_weight only works if we have one weight per data row
                # i.e. it works well for single target regression, but not for MultiTask regression
                # might have to do one regression per target to use this info
                sample_weights = weights.to_numpy().sum(axis = 1)
                # Fetch last step name to produce argument for sample_weight
                kwargs = {pipeline.steps[-1][0] + '__sample_weight': sample_weights}
                # Perform weighted fit
                pipeline.fit(input_train, output_train, **kwargs)
            else: # either sample_weighing == False or mode != 'linreg'
                pipeline.fit(input_train, output_train)

            return pipeline, features_train

        elif trained_pipeline is not None:
            # Predict on train, using trained_pipeline
            reg_pred = trained_pipeline.predict(input_train)
            output_data = output_train

    # same for the test set
    elif dataset == "test":
        if trained_pipeline is None:
            raise ValueError("No pipeline provided as trained_pipeline!")
        elif features_train is None:
            raise ValueError("No features_train provided!")
        else:
            # Fetch test data
            samples_test = samples["test"]

            input_test, _ = load_data(
                contextualization=context,
                samples=samples_test,
                features=features_train,
                **petab_base_files,
                io_mode = 'input',
            )

            output_test, _, _ = load_data(
                contextualization="cytof_dynamic",
                samples=samples_test,
                features=None,
                **petab_base_files,
                io_mode='output',
            )

            # Transform test data with pipeline and predict (all in .predict())
            reg_pred = trained_pipeline.predict(input_test)
            # Some output_test turned out to be NaNs simply because of missing imputation at 13h for EGF (other conditions were imputed)
            # and at 40h (timepoint was not considered for imputation)
            # reg_pred = reg_pred[~output_test.isna()]
            output_data = output_test

    # Process regression output (reg_pred) and output data before plotting and evaluating simulations
    # Convert into pandas dataframe with same index and column headers as output_test
    reg_pred = pd.DataFrame(reg_pred, index=output_data.index, columns=output_data.columns)

    # Process dataframes to use with plot_cross_samples and process_simulation
    # reg_pred
    reg_pred = reg_pred.T.stack().reset_index().sort_values(by=['preequilibrationConditionId', 'observableId', 'simulationConditionId', 'time'])
    reg_pred = reg_pred.reset_index().drop(columns='index')
    # Rename value column from 0 to 'simulation' to use in process_simulation()
    reg_pred.rename(columns={0: "simulation"}, inplace=True)

    # output_data
    output_data = output_data.T.stack().reset_index().sort_values(by=['preequilibrationConditionId', 'observableId', 'simulationConditionId', 'time'])
    output_data = output_data.reset_index().drop(columns='index')
    # Rename value column from 0 to 'measurement' to use in process_simulation()
    output_data.rename(columns={0: "measurement"}, inplace=True)


    # Produce plots to analyse performance -- block of code is shared between train/test
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
            orth_reg_strategy="None", # not needed for regression
            job = None, # not needed here
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
        #print(f'Training estimator {mode} on context {context}...')
        trained_pipelines[context][mode], features_train[context][mode] = evaluate_standard_regression("train", conf, context, mode=mode)
        print(f'Estimator {mode} trained on context {context}')

# Evaluate regressions
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

    # Regression baseline ("linreg", "lasso", "elasticnet")
    for context, _ in CONTEXTS_FEATURES:
        # Regression baseline first
        for mode in ['linreg', 'lasso', 'elasticnet']:
            df = evaluate_standard_regression(dataset,
                                              conf,
                                              context,
                                              mode = mode,
                                              trained_pipeline=trained_pipelines[context][mode],
                                              features_train=features_train[context][mode]
                                              )
            df.to_csv(
                EVALUATION_REFERENCE_REG.format(
                    model = conf.model,
                    data = conf.data,
                    samples = conf.samples,
                    dataset=dataset,
                    mode=mode,
                    context=context,
                )
            )

