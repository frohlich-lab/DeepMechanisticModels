from functools import partial
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import petab.v1 as petab
import pysb

from . import get_samples

figdir = Path(__file__).parent / "figures"

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
    "syn20613710",  # HCC1599  REMOVED AS OUTLIER, SEE `Cytof Data Analysis.ipynb`
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


def load_snp_from_synapse() -> pd.DataFrame:
    import synapseclient
    import xmltodict
    from Bio import Entrez

    syn = synapseclient.Synapse()
    syn.login()
    MUTATION_GENES = [
        "EGFR",
        "ERBB2",
        "ERBB3",
        "ERBB4",
        "MAP2K1",
        "MAP2K2",
        "MAPK1",
        "MAPK3",
        "RAF1",
        "BRAF",
        "KRAS",
        "NRAS",
        "HRAS",
        "GRB2",
        "SOS1",
        "PIK3CA",
        "NF1",
        "ALK",
        "EPHA3",
        "EPHA5",
        "KIT",
        "MAP2K4",
        "MET",
        "PDGFRA",
        "RET",
        "ROS1",
    ]

    df_snp_mapping = pd.read_csv(syn.get("syn20631265").path, index_col=0)
    df_snp_mapping.dropna(subset="GeneNames", axis=0, inplace=True)
    df_snp_mapping = df_snp_mapping[
        df_snp_mapping["GeneNames"].apply(
            lambda x: any(g in MUTATION_GENES for g in x.split(","))
        )
    ]
    df_snp_mapping = df_snp_mapping[
        df_snp_mapping["SNPid"].str.startswith("rs")
    ]

    Entrez.email = "froehlichfab@gmail.com"

    #########
    # dbSNP #
    #########
    stream = Entrez.efetch(
        db="snp", id=",".join(df_snp_mapping["SNPid"].tolist()), retmode="xml"
    )
    xml_data = stream.read()
    stream.close()

    data_snp = xmltodict.parse(xml_data)["ExchangeSet"]["DocumentSummary"]

    # filter out mapping
    df_snp_mapping = df_snp_mapping[[not ("error" in d) for d in data_snp]]
    # now trim data_snp
    data_snp = [d for d in data_snp if not ("error" in d)]
    df_snp_mapping["HGVS"] = [
        d["DOCSUM"].replace("HGVS=", "") for d in data_snp
    ]
    df_snp_mapping["clinvar"] = [
        ""
        if d["CLINICAL_SIGNIFICANCE"] is None
        else d["CLINICAL_SIGNIFICANCE"]
        for d in data_snp
    ]
    df_snp_mapping = df_snp_mapping[
        df_snp_mapping["clinvar"].apply(
            lambda x: (
                "uncertain-significance" in x
                or "likely-pathogenic" in x
                or "pathogenic" in x
                or "risk-factor" in x
                or "protective" in x
            )
        )
    ]

    snp_file_path = syn.get("syn20631266").path
    with open(snp_file_path, "r") as f:
        header_line = f.readline().strip()
    available_columns = header_line.split(",")
    available_columns = [c.replace('"', "") for c in available_columns]
    valid_snp_columns = [
        col
        for col in df_snp_mapping["SNPid"].tolist()
        if col in available_columns
    ]

    df_snp = pd.read_csv(
        snp_file_path,
        engine="c",
        index_col=0,
        low_memory=False,
        usecols=["Unnamed: 0"] + valid_snp_columns,
    )

    df_snp = df_snp.loc[
        :, df_snp.sum(axis=0) > 0
    ]  # filter out columns with all zeros
    df_snp_mapping = df_snp_mapping[
        df_snp_mapping["SNPid"].isin(df_snp.columns)
    ]

    return df_snp, df_snp_mapping


def load_cna_from_synapse() -> pd.DataFrame:
    import synapseclient

    syn = synapseclient.Synapse()
    syn.login()

    df_cna = pd.read_csv(
        syn.get("syn20631262").path,
        index_col=0,
    )

    return df_cna


def load_cytof_from_synapse() -> Tuple[pd.DataFrame, List[str]]:
    import synapseclient

    syn = synapseclient.Synapse()
    syn.login()
    files = SYNAPSE_FILES
    mean_data = []
    min_data = []
    std_data = []
    group_ids = [
        "treatment",
        "cell_line",
        "time",
        "fileID",
        "date",
        "time_course",
    ]

    # figdir.mkdir(parents=True, exist_ok=True)

    file_id_table = pd.read_csv(
        syn.get("syn20631269").path, index_col=0
    ).set_index("fileID")

    # file_id_table = pd.read_csv(syn.get("syn20631269").path, index_col=0)
    for file in files:  # syn20613939 for MDAMB157 -- has double the amount of fileIDs (biological replicates)
        df = pd.read_csv(syn.get(file).path)
        df["date"] = df["fileID"].apply(lambda x: file_id_table.loc[x, "date"])
        df["time_course"] = df["fileID"].apply(
            lambda x: file_id_table.loc[x, "time_course"]
        )

        # import seaborn as sns
        # import matplotlib.pyplot as plt
        #
        # markers = ['p.MEK', 'p.ERK', 'p.HER2', 'p.p90RSK', 'p.S6', 'p.p38', 'p.MAP2K3', 'p.MAPKAPK2', 'p.PDPK1', 'p.Akt.Ser473.', 'p.AKT.Thr308.', 'p.JNK', 'p.MKK3.MKK6', 'p.MKK4', 'p.S6K']
        # df_plot = df[df.treatment != 'full'].melt(
        #     id_vars=['treatment','cell_line', 'time', 'cellID', 'fileID', 'date', 'time_course'],
        #     value_vars=markers
        # )
        #
        # g = sns.FacetGrid(
        #     df_plot, col='treatment', row='variable'
        # )
        # g.map_dataframe(
        #     sns.boxenplot,
        #     data=df_plot,
        #     x='time',
        #     y='value',
        #     hue='time_course',
        # )
        # plt.savefig(str(figdir / df_plot.cell_line.values[0]) + '.pdf')

        for ids, data in df.groupby(group_ids):
            if f"c{ids[1]}" not in get_samples("dream_cytof"):
                continue
            markers = [
                c for c in data.columns if c not in group_ids + ["cellID"]
            ]
            m = data[markers].mean()
            n = data[markers].min()
            # std = data[markers].std()
            # Create a Series of ones for std with same index (i.e., same markers) -- same weight
            std = pd.Series(1.0, index=m.index)
            for sdf in [m, std, n]:
                sdf["treatment"] = ids[0]
                sdf["cell_line"] = ids[1]
                sdf["time"] = ids[2]
                sdf["fileID"] = ids[3]
                sdf["date"] = ids[4]
                sdf["time_course"] = ids[5]
            mean_data.append(m)
            min_data.append(n)
            # std[std.isna()] = 1.0
            std_data.append(std)

    d = {
        desc: pd.concat(data, axis=1).T
        for desc, data in (
            ("mean", mean_data),
            ("std", std_data),
            ("min", min_data),
        )
    }
    id_vars = [
        "cell_line",
        "treatment",
        "time",
        "fileID",
        "date",
        "time_course",
    ]
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
        condition_table[petab.CONDITION_ID].apply(lambda x: ("full" not in x)),
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
    if "__" in model.name:
        modifications = model.name.split("__")[1].split("_")
    else:
        modifications = ()

    for par_name in model.parameters.keys():
        # mutations
        if par_name.startswith("m_"):
            if par_name.replace("_", "").lower() in modifications:
                if par_name == "m_BRAF":
                    cell_lines = ["cDU4475"]
                elif par_name == "m_KRAS":
                    cell_lines = [
                        "cMDAMB134VI",
                        "cMDAMB231",
                        "cMDAMB453",
                        "cMPE600",
                    ]
                else:
                    cell_lines = []

                def filter_cl(x, cls):
                    return float(x.split("__")[0] in cls)

                fcl = partial(filter_cl, cls=cell_lines)

                condition_table[par_name] = condition_table[
                    petab.CONDITION_ID
                ].apply(fcl)

        # expression levels
        elif par_name.endswith("_eq"):
            gene = par_name.split("_")[-2]
            if gene in [
                "EGFR",
                "ERBB2",
                "TGFA",
                "BTC",
                "EREG",
                "NRG1",
                "NRG2",
            ]:
                if f"p{gene.lower()}" in modifications:
                    measurement_type = "proteomics"
                elif f"t{gene.lower()}" in modifications:
                    measurement_type = "transcriptomics"
                elif f"f{gene.lower()}" in modifications:
                    continue
                else:
                    condition_table[par_name] = float(
                        gene in ["EGFR", "ERBB2"]
                    )
                    continue

                prot_data = measurement_table[
                    (measurement_table[petab.OBSERVABLE_ID] == gene)
                    & (
                        measurement_table["measurementType"]
                        == measurement_type
                    )
                ]

                prot_log2fc = (
                    prot_data[
                        [
                            petab.PREEQUILIBRATION_CONDITION_ID,
                            petab.MEASUREMENT,
                        ]
                    ]
                    .groupby(petab.PREEQUILIBRATION_CONDITION_ID)
                    .agg("mean")[petab.MEASUREMENT]
                )
                prot_log2fc -= prot_log2fc.mean()
                condition_table[par_name] = [
                    2 ** prot_log2fc.get(c.split("__")[0], 0.0)
                    for c in condition_table[petab.CONDITION_ID]
                ]
                continue
            else:
                condition_table[par_name] = 1.0
        elif par_name.endswith("_eq") and not par_name.startswith(
            ("DEV_", "MED_")
        ):
            condition_table[par_name] = 1.0
    return condition_table


def load_dream_data(model: pysb.Model) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # table_SNP, SNP_mapping = load_snp_from_synapse()
    # table_cna = load_cna_from_synapse()

    measurement_table_cytof = pd.read_csv("./data/cytof.csv", index_col=0)

    measurement_table_proteomics = pd.read_csv(
        "./data/proteomics.csv", index_col=0
    )

    measurement_table_transcriptomics = pd.read_csv(
        "./data/transcriptomics.csv", index_col=0
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
