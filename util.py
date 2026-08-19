from typing import Dict

import pandas as pd
import petab.v1 as petab

from common import (
    CONDITIONS_FILE,
    MEASUREMENTS_FILE,
    OBSERVABLES_FILE,
)
from dmm.config_options import Conf

dtypes = {
    "measurement_table": {
        petab.OBSERVABLE_ID: str,
        petab.PREEQUILIBRATION_CONDITION_ID: str,
        petab.TIME: float,
        petab.MEASUREMENT: float,
        # petab.NOISE_PARAMETERS: float, // could be str/float, let pandas infer
        petab.SIMULATION_CONDITION_ID: str,
        petab.OBSERVABLE_PARAMETERS: str,
        "measurementType": str,
        "FEATURE_ID": str,
        "date": str,
        "time_course": str,
    },
    "condition_table": None,
    "observable_table": str,
}


def load_petab_base_files(conf: Conf) -> Dict[str, pd.DataFrame]:
    return {
        label: pd.read_csv(
            file.format(**conf.to_dict()),
            index_col=0,
            sep="\t",
            dtype=dtypes[label],
        )
        for label, file in (
            ("measurement_table", MEASUREMENTS_FILE),
            ("condition_table", CONDITIONS_FILE),
            ("observable_table", OBSERVABLES_FILE),
        )
    }
