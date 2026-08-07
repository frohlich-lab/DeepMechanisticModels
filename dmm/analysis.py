import equinox as eqx
import jax.random as jr
import numpy as np
import pandas as pd
import petab.v1 as petab
from amici.petab import rdatas_to_simulation_df
from pypesto.C import MODE_RES, RDATAS

from .config_options import scan_attributes
from .plotting import plot_cross_samples
from .training_helper_funcs import (
    model_output_to_petab_input,
    model_output_to_petab_input_nojit,
)


def process_simulation(
    evaluations,
    measurement_df,
    simulation_df,
    conf,
    sample,
):
    # Set columns for multi-index
    cols_to_check = [
        petab.OBSERVABLE_ID,
        petab.PREEQUILIBRATION_CONDITION_ID,
        petab.TIME,
        petab.SIMULATION_CONDITION_ID,
    ]
    # Sort by same observables and subset to same cell-line/sample
    mdf = (
        measurement_df[
            measurement_df[petab.PREEQUILIBRATION_CONDITION_ID] == sample
        ]
        .sort_values(by=cols_to_check)
        .set_index(cols_to_check)
    )
    sdf = (
        simulation_df[
            simulation_df[petab.PREEQUILIBRATION_CONDITION_ID] == sample
        ]
        .sort_values(by=cols_to_check)
        .set_index(cols_to_check)
    )
    # Subset to common observables (needed for regressors) -- alignment step
    sdf = sdf[sdf.index.isin(mdf.index)]

    # Compute residual dataframe
    res = mdf.copy()
    res["res"] = res[petab.MEASUREMENT] - sdf[petab.SIMULATION]
    res["sim"] = sdf[petab.SIMULATION]
    # Unpack multi-index
    res.reset_index(inplace=True)

    for _, r in res.iterrows():
        # re-defining condition in such a way that fits both avg and avg_model references and regression standards
        if len(r[petab.SIMULATION_CONDITION_ID].split("__")) > 1:
            condition = r[petab.SIMULATION_CONDITION_ID].split("__")[1]
        else:
            condition = r[petab.SIMULATION_CONDITION_ID]

        subset_conf_dict = {
            k: v for k, v in conf.to_dict().items() if k in scan_attributes
        }
        evaluations.append(
            {
                "res": r["res"],
                "sim": r["sim"],
                "obs": r[petab.MEASUREMENT],
                "sample": sample,
                "observable": r[petab.OBSERVABLE_ID],
                "condition": condition,
                "time": r[petab.TIME],
                **subset_conf_dict,
            }
        )


def simulate_dmm(
    model, input_features, obj, petab_problem, jit_fn: bool = True
) -> pd.DataFrame:
    # Generally use the jitted model_output_to_petab_input function
    if jit_fn:
        fn = model_output_to_petab_input
    else:
        fn = model_output_to_petab_input_nojit

    res = obj(
        fn(eqx.nn.inference_mode(model), input_features, jr.PRNGKey(0)),
        mode=MODE_RES,
        return_dict=True,
    )

    amici_model = obj.amici_model

    try:
        simulation_df = rdatas_to_simulation_df(
            res[RDATAS],
            model=amici_model,
            measurement_df=petab_problem.measurement_df,
        )
    except Exception as e:
        print(f"Error occurred: {e}")
        # If there are NaNs in the simulation results (e.g. inf fval and zero grads during training),
        # simply return a dataframe with np.inf SIMULATION values
        if np.isnan(res["res"]).sum() > 0:
            # Create a DataFrame identical to petab_problem.measurement_df
            simulation_df = petab_problem.measurement_df.copy()
            # Rename the MEASUREMENT column to SIMULATION
            simulation_df.rename(
                columns={petab.MEASUREMENT: petab.SIMULATION}, inplace=True
            )
            # Set simulation values to np.inf to indicate that the simulation failed
            simulation_df[petab.SIMULATION] = np.inf
        else:
            # different error when trying to process simulation - raise and interrupt
            raise

    return simulation_df


def evaluate_simulations(
    model,
    input_features,
    obj,
    conf,
    samples,
    petab_problem,
    dataset,
    outdir,
    evaluations,
    plot_file_prefix: str,
):
    simulation_df = simulate_dmm(
        model, input_features, obj, petab_problem, jit_fn=False
    )

    for sample in samples:
        process_simulation(
            evaluations=evaluations,
            measurement_df=petab_problem.measurement_df,
            simulation_df=simulation_df,
            conf=conf,
            sample=sample,
        )

    plot_cross_samples(
        petab_problem.measurement_df,
        simulation_df,
        figdir=outdir / dataset,
        prefix=plot_file_prefix,
    )
