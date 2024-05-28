import pandas as pd
import petab

from common import Conf, MEASUREMENTS_FILE, OBSERVABLES_FILE


def get_measurements_and_obervables(conf: Conf):
    df_meas = pd.read_csv(
        MEASUREMENTS_FILE.format(**conf.__dict__), sep="\t", index_col=0
    )
    df_obs = pd.read_csv(
        OBSERVABLES_FILE.format(**conf.__dict__), sep="\t", index_col=0
    )
    df_meas = df_meas[
        df_meas[petab.OBSERVABLE_ID].apply(lambda x: x in df_obs.index)
    ]
    return df_meas, df_obs


def process_sim_df(df_sim: pd.DataFrame) -> pd.DataFrame:
    return df_sim.rename(
        columns={
            "sample": petab.PREEQUILIBRATION_CONDITION_ID,
            "observable": petab.OBSERVABLE_ID,
            "condition": petab.SIMULATION_CONDITION_ID,
        }
    )
