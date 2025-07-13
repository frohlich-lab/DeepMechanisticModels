from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import petab.v1 as petab
import pysb

from . import get_samples

SYNAPSE_FILES = [
    # Subchallenge 4
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
    # Subchallenge 2 (added 11.04.2025)
    "syn20631041",  # 184B5
    "syn20631043",  # BT483
    "syn20631044",  # HCC1428
    "syn20631045",  # HCC1806
    "syn20631047",  # HCC202
    "syn20631048",  # Hs578T
    "syn20631049",  # MCF12A
    "syn20631050",  # MDAMB231
    "syn20631060",  # MDAMB468
    "syn20631061",  # SKBR3
    "syn20631062",  # UACC3199
    "syn20631063",  # ZR751
]


def load_cytof_from_synapse() -> Tuple[pd.DataFrame, List[str]]:
    import synapseclient

    syn = synapseclient.Synapse()
    syn.login()
    files = SYNAPSE_FILES
    mean_data = []
    std_data = []
    group_ids = ["treatment", "cell_line", "time", "fileID"]
    for file in files:  # syn20613939 for MDAMB157 -- has double the amount of fileIDs (biological replicates)
        df = pd.read_csv(syn.get(file).path)
        for ids, data in df.groupby(group_ids):
            if f"c{ids[1]}" not in get_samples("dream_cytof"):
                continue
            markers = [
                c for c in data.columns if c not in group_ids + ["cellID"]
            ]
            m = data[markers].mean()
            # std = data[markers].std()
            # Create a Series of ones for std with same index (i.e., same markers) -- same weight
            std = pd.Series(1.0, index=m.index)
            for sdf in [m, std]:
                sdf["treatment"] = ids[0]
                sdf["cell_line"] = ids[1]
                sdf["time"] = ids[2]
                sdf["fileID"] = ids[3]
            mean_data.append(m)
            # std[std.isna()] = 1.0
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
    measurement_table_phospho.drop(
        columns=["treatment", "fileID"], inplace=True
    )
    measurement_table_phospho["measurementType"] = "cytof"
    measurement_table_phospho["FEATURE_ID"] = measurement_table_phospho[
        petab.OBSERVABLE_ID
    ]
    return measurement_table_phospho


def load_proteomics_from_synapse() -> pd.DataFrame:
    import synapseclient

    syn = synapseclient.Synapse()
    syn.login()
    df_proteomics = pd.read_csv(syn.get("syn20690775").path, index_col=[0])
    df_proteomics["UPID"] = df_proteomics.index

    df_proteomics = pd.melt(
        df_proteomics,
        id_vars=["UPID"],
        var_name=petab.PREEQUILIBRATION_CONDITION_ID,
        value_name=petab.MEASUREMENT,
    )
    df_proteomics["measurementType"] = "proteomics"
    return df_proteomics


def load_transcriptomics_from_synapse() -> pd.DataFrame:
    import synapseclient

    syn = synapseclient.Synapse()
    syn.login()
    df_transcriptomics = pd.read_csv(
        syn.get("syn20631264").path, index_col=[0]
    )
    df_transcriptomics.drop(columns=["X"], inplace=True)
    df_transcriptomics.rename(
        columns={"cell_line": petab.PREEQUILIBRATION_CONDITION_ID},
        inplace=True,
    )

    df_transcriptomics = pd.melt(
        df_transcriptomics,
        id_vars=petab.PREEQUILIBRATION_CONDITION_ID,
        var_name="GENENAME",
        value_name=petab.MEASUREMENT,
    )
    df_transcriptomics["measurementType"] = "transcriptomics"
    return df_transcriptomics


def load_ids_from_uniprot(ids):
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
            "query": " ".join(ids),
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

    return up_ids


def process_petab_proteomics(df: pd.DataFrame):
    df.dropna(axis=0, subset=[petab.MEASUREMENT], inplace=True)

    df[petab.PREEQUILIBRATION_CONDITION_ID] = df[
        petab.PREEQUILIBRATION_CONDITION_ID
    ].apply(lambda x: f'c{x.split("_")[0]}')

    df[petab.SIMULATION_CONDITION_ID] = df[petab.PREEQUILIBRATION_CONDITION_ID]
    df[petab.OBSERVABLE_ID] = df["GENENAME"]

    df[petab.TIME] = 0.0
    df[petab.NOISE_PARAMETERS] = 1.0
    df["FEATURE_ID"] = df["GENENAME"]
    return df


def process_petab_transcriptomics(df: pd.DataFrame):
    df.dropna(axis=0, subset=[petab.MEASUREMENT], inplace=True)

    df[petab.PREEQUILIBRATION_CONDITION_ID] = df[
        petab.PREEQUILIBRATION_CONDITION_ID
    ].apply(lambda x: f'c{x.split("_")[0]}')

    df[petab.SIMULATION_CONDITION_ID] = df[petab.PREEQUILIBRATION_CONDITION_ID]
    df[petab.OBSERVABLE_ID] = df["GENENAME"]

    df[petab.TIME] = 0.0
    df[petab.NOISE_PARAMETERS] = 1.0
    df["FEATURE_ID"] = df["GENENAME"]
    return df


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
            lambda x: ("full" not in x)
            and ("iPI3K" not in x or "iPI3K_0" in model.parameters.keys())
            and ("iPKC" not in x or "iPKC_0" in model.parameters.keys())
        ),
        :,
    ]

    perturbations = np.unique(
        [
            p
            for c in condition_table[petab.CONDITION_ID]
            if len(c.split("__")) > 1
            for p in c.split("__")[1:]
        ]
    )
    for pert in perturbations:

        def not_part_of_condition(c: str, pert=pert) -> bool:
            return pert not in c.split("__")

        if model.parameters.get(f"{pert}_0") is None:
            # remove condition
            condition_table = condition_table.loc[
                condition_table[petab.CONDITION_ID].apply(
                    not_part_of_condition
                ),
                :,
            ]
            continue

        def part_of_condition(c: str, pert=pert) -> float:
            return float(int(pert in c.split("__")))

        condition_table[f"{pert}_0"] = condition_table[
            petab.CONDITION_ID
        ].apply(part_of_condition)

    condition_table["EGF_0"] = condition_table[petab.CONDITION_ID].apply(
        lambda x: float("__" in x)
    )
    for eq_par in model.parameters.keys():
        if eq_par == "EGFR_eq" and "egfra" in model.name.split("_"):
            EGFR_log2fc = (
                measurement_table[
                    measurement_table[petab.OBSERVABLE_ID] == "EGFR"
                ][[petab.PREEQUILIBRATION_CONDITION_ID, petab.MEASUREMENT]]
                .groupby(petab.PREEQUILIBRATION_CONDITION_ID)
                .agg("mean")[petab.MEASUREMENT]
            )
            condition_table[eq_par] = [
                2 ** EGFR_log2fc.get(c.split("__")[0], 0.0)
                for c in condition_table[petab.CONDITION_ID]
            ]
        elif eq_par.endswith("_eq") and not eq_par.startswith(
            ("DEV_", "MED_")
        ):
            condition_table[eq_par] = 1.0
    return condition_table


def load_dream_data(model: pysb.Model) -> Tuple[pd.DataFrame, pd.DataFrame]:
    measurement_table_cytof, id_vars = load_cytof_from_synapse()
    measurement_table_cytof = process_petab_cytof(
        measurement_table_cytof, id_vars
    )

    measurement_table_proteomics = load_proteomics_from_synapse()
    up_ids = load_ids_from_uniprot(
        measurement_table_proteomics["UPID"].unique()
    )
    # missing gene names: A2VCL2, A8MUA0, O00370, Q6ZSR9
    # A2VCL2: dropped from uniprot, CCDC162P https://varsome.com/gene/hg19/CCDC162P
    # A8MUA0: Putative UPF0607 protein, SPATA6: https://varsome.com/gene/hg19/SPATA6
    # O00370: ORF2p: https://www.uniprot.org/uniprotkb/O00370/entry
    # Q6ZSR9: Uncharacterized protein FLJ45252: https://www.uniprot.org/uniprotkb/Q6ZSR9/entry
    up_ids["A2VCL2"] = "CCDC162P"
    up_ids["A8MUA0"] = "SPATA6"
    up_ids["O00370"] = "ORF2P"
    up_ids["Q6ZSR9"] = "FLJ45252"

    measurement_table_proteomics.loc[
        :, "GENENAME"
    ] = measurement_table_proteomics["UPID"].apply(lambda x: up_ids.get(x))
    measurement_table_proteomics = process_petab_proteomics(
        measurement_table_proteomics
    )

    measurement_table_transcriptomics = load_transcriptomics_from_synapse()
    measurement_table_transcriptomics = process_petab_transcriptomics(
        measurement_table_transcriptomics
    )

    measurement_table = pd.concat(
        [
            measurement_table_cytof,
            measurement_table_proteomics,
            measurement_table_transcriptomics,
        ]
    )
    condition_table = build_condition_table(measurement_table, model)
    return measurement_table.copy(), condition_table.copy()
