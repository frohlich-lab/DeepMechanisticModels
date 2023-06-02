from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import petab
import pysb

from . import get_samples

SYNAPSE_FILES = [
    "syn20613594",  # 184A1
    "syn20613595",  # BT20
    "syn20613596",  # BT474
    "syn20613597",  # BT549
    "syn20613598",  # CAL148
    "syn20613599",  # CAL51
    "syn20613600",  # CAL851
    "syn20613601",  # DU4475
    "syn20613660",  # EFM192A
    "syn20613665",  # EVSAT
    "syn20613668",  # HBL100
    "syn20613674",  # HCC1187
    "syn20613687",  # HCC1395
    "syn20613696",  # HCC1419
    "syn20613702",  # HCC1500
    "syn20613708",  # HCC1569
    # "syn20613710",  # HCC1599  REMOVED AS OUTLIER, SEE `Cytof Data Analysis.ipynb`
    "syn20613719",  # HCC1937
    "syn20613739",  # HCC1954
    "syn20613793",  # HCC2157
    "syn20613802",  # HCC2185
    "syn20613814",  # HCC3153
    "syn20613821",  # HCC38
    "syn20613832",  # HCC70
    "syn20613849",  # HDQP1
    "syn20613865",  # JIMT1
    "syn20613880",  # MCF10A
    "syn20613911",  # MCF10F
    "syn20613920",  # MCF7
    "syn20613935",  # MDAMB134VI
    "syn20613939",  # MDAMB157
    "syn20613943",  # MDAMB175VII
    "syn20613962",  # MDAMB361
    "syn20613975",  # MDAMB415
    "syn20613988",  # MDAMB453
    "syn20613930",  # MDAkb2
    "syn20613995",  # MFM223
    "syn20614008",  # MPE600
    "syn20614033",  # MX1
    "syn20614045",  # OCUBM
    "syn20614052",  # T47D
    "syn20614063",  # UACC812
    "syn20614074",  # UACC893
    "syn20614085",  # ZR7530
]


def load_cytof_from_synapse() -> Tuple[pd.DataFrame, List[str]]:
    import synapseclient

    syn = synapseclient.Synapse()
    syn.login()
    files = SYNAPSE_FILES
    mean_data = []
    std_data = []
    group_ids = ["treatment", "cell_line", "time", "fileID"]
    for file in files:
        df = pd.read_csv(syn.get(file).path)
        for ids, data in df.groupby(group_ids):
            if f"c{ids[1]}" not in get_samples("dream_cytof"):
                continue
            markers = [
                c for c in data.columns if c not in group_ids + ["cellID"]
            ]
            m = data[markers].median()
            std = data[markers].std()
            for sdf in [m, std]:
                sdf["treatment"] = ids[0]
                sdf["cell_line"] = ids[1]
                sdf["time"] = ids[2]
                sdf["fileID"] = ids[3]
            mean_data.append(m)
            std[std.isna()] = 1.0
            std_data.append(std)

    d = {
        desc: pd.concat(data, axis=1).T
        for desc, data in (("mean", mean_data), ("std", std_data))
    }
    id_vars = ["cell_line", "treatment", "time", "fileID"]
    df_phospho_condition = d["mean"][id_vars]
    for sdf in d.values():
        sdf.drop(columns=id_vars, inplace=True)
    df_phospho = pd.concat(d.values(), axis=1, keys=d.keys()).swaplevel(
        0, 1, axis=1
    )
    df_phospho = pd.concat((df_phospho_condition, df_phospho), axis=1)

    measurement_table_phospho = pd.melt(
        df_phospho,
        id_vars=id_vars,
        var_name=petab.OBSERVABLE_ID,
    )

    return measurement_table_phospho, id_vars


def process_petab_cytof(
    measurement_table_phospho: pd.DataFrame, id_vars: List[str]
) -> pd.DataFrame:
    measurement_table_phospho[
        [petab.OBSERVABLE_ID, "type"]
    ] = measurement_table_phospho[petab.OBSERVABLE_ID].to_list()

    measurement_table_phospho = (
        measurement_table_phospho.set_index(
            ["type", petab.OBSERVABLE_ID] + id_vars
        )
        .unstack("type")
        .droplevel(axis=1, level=0)
        .reset_index()
    )

    measurement_table_phospho.rename(
        columns={
            "cell_line": petab.PREEQUILIBRATION_CONDITION_ID,
            "time": petab.TIME,
            "mean": petab.MEASUREMENT,
            "std": petab.NOISE_PARAMETERS,
        },
        inplace=True,
    )

    measurement_table_phospho[
        petab.PREEQUILIBRATION_CONDITION_ID
    ] = measurement_table_phospho[petab.PREEQUILIBRATION_CONDITION_ID].apply(
        lambda x: f"c{x}"
    )

    measurement_table_phospho[
        petab.SIMULATION_CONDITION_ID
    ] = measurement_table_phospho.apply(
        lambda x: f"{x[petab.PREEQUILIBRATION_CONDITION_ID]}__{x.treatment}",
        axis=1,
    )
    return measurement_table_phospho.drop(columns=["treatment", "fileID"])


def load_proteomics_from_synapse() -> pd.DataFrame:
    import synapseclient

    syn = synapseclient.Synapse()
    syn.login()
    df_proteomics = pd.read_csv(syn.get("syn20690775").path, index_col=[0])
    df_proteomics[petab.OBSERVABLE_ID] = df_proteomics.index

    df_proteomics = df_proteomics[
        df_proteomics[petab.OBSERVABLE_ID].apply(lambda x: ";" not in x)
    ]

    return pd.melt(
        df_proteomics,
        id_vars=[petab.OBSERVABLE_ID],
        var_name=petab.PREEQUILIBRATION_CONDITION_ID,
        value_name=petab.MEASUREMENT,
    )


def load_ids_from_uniprot(measurement_table_proteomics):
    import json
    import urllib.parse
    import urllib.request

    up_id_json = "up_ids.json"
    if Path(up_id_json).exists():
        with open(up_id_json, "r") as fp:
            up_ids = json.load(fp)
    else:
        url = "https://www.uniprot.org/uploadlists/"

        params = {
            "from": "ACC+ID",
            "to": "GENENAME",
            "format": "tab",
            "query": " ".join(
                measurement_table_proteomics[petab.OBSERVABLE_ID].unique()
            ),
        }

        data = urllib.parse.urlencode(params)
        data = data.encode("utf-8")
        req = urllib.request.Request(url, data)
        with urllib.request.urlopen(req) as f:
            response = f.read()
        up_ids = dict(
            [
                mapping.split("\t")
                for mapping in response.decode("utf-8").split("\n")
                if "\t" in mapping
            ]
        )
        with open(up_id_json, "w") as fp:
            json.dump(up_ids, fp)

    measurement_table_proteomics.loc[
        petab.OBSERVABLE_ID, :
    ] = measurement_table_proteomics[petab.OBSERVABLE_ID].apply(
        lambda x: up_ids.get(x, "")
    )
    return measurement_table_proteomics


def process_petab_proteomics(measurement_table_proteomics: pd.DataFrame):
    measurement_table_proteomics = measurement_table_proteomics.loc[
        measurement_table_proteomics[petab.OBSERVABLE_ID] != "", :
    ]

    measurement_table_proteomics.dropna(
        axis=0, subset=[petab.MEASUREMENT], inplace=True
    )

    measurement_table_proteomics[
        petab.PREEQUILIBRATION_CONDITION_ID
    ] = measurement_table_proteomics[petab.PREEQUILIBRATION_CONDITION_ID].apply(
        lambda x: f'c{x.split("_")[0]}'
    )

    measurement_table_proteomics[
        petab.SIMULATION_CONDITION_ID
    ] = measurement_table_proteomics[petab.PREEQUILIBRATION_CONDITION_ID]

    measurement_table_proteomics[petab.TIME] = 0.0
    return measurement_table_proteomics


def build_condition_table(
    measurement_table: pd.DataFrame, model: pysb.Model
) -> pd.DataFrame:
    condition_table = pd.DataFrame(
        {
            petab.CONDITION_ID: np.unique(
                np.concatenate(
                    [
                        measurement_table[petab.SIMULATION_CONDITION_ID],
                        measurement_table[petab.PREEQUILIBRATION_CONDITION_ID],
                    ]
                )
            )
        }
    )

    # ignore "full" for now
    condition_table = condition_table.loc[
        condition_table[petab.CONDITION_ID].apply(
            lambda x: "full" not in x.split("__")
        ),
        :,
    ]

    perturbations = np.unique(
        [
            p
            for c in condition_table[petab.CONDITION_ID]
            if len(c.split("__")) > 1
            for p in c.split("__")[1:]
            if p != "full"
        ]
    )
    for pert in perturbations:
        if model.parameters.get(f"{pert}_0") is None:
            # remove condition
            condition_table = condition_table.loc[
                condition_table[petab.CONDITION_ID].apply(
                    lambda x: pert not in x.split("__")
                ),
                :,
            ]
            continue
        condition_table[f"{pert}_0"] = condition_table[
            petab.CONDITION_ID
        ].apply(lambda x: float(int(pert in x.split("__"))))

    condition_table["EGF_0"] = condition_table[petab.CONDITION_ID].apply(
        lambda x: float("__" in x)
    )
    return condition_table


def load_dream_data(model: pysb.Model) -> Tuple[pd.DataFrame, pd.DataFrame]:
    measurement_table_cytof, id_vars = load_cytof_from_synapse()
    measurement_table_cytof = process_petab_cytof(
        measurement_table_cytof, id_vars
    )

    measurement_table_proteomics = load_proteomics_from_synapse()
    measurement_table_proteomics = load_ids_from_uniprot(
        measurement_table_proteomics
    )
    measurement_table_proteomics = process_petab_proteomics(
        measurement_table_proteomics
    )

    # ignore proteomics data for now
    measurement_table = pd.concat(
        [measurement_table_cytof, measurement_table_proteomics]
    )
    condition_table = build_condition_table(measurement_table, model)
    return measurement_table.copy(), condition_table.copy()
