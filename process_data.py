import sys
import re
import petab
import pysb

import pandas as pd

from mEncoder.generate_data import generate_synthetic_data
from common import data_dir, CONDITIONS_FILE, MEASUREMENTS_FILE, OBSERVABLES_FILE


def observable_id_to_model_expr(obs_id: str, dataset: str, model: pysb.Model) -> str:
    """
    Maps site definitions from data to model observables

    :param obs_id:
        identifier of the phosphosite in the data table

    :param dataset:
        identifier of the dataset. Used to setup parse observable information

    :param model:
        model to which the observables are mapped

    :return:
        the name of the corresponding observable in the model
    """

    if dataset == "dream_cytof":
        obs_id = obs_id.replace("-", "_").upper()
        palias = {
            r"^P\.STAT5": "STAT5A_Y694",
            r"^P\.MEK": "pMEK_S222",
            r"^P\.S6K$": "RPS6KB1_S412",
            r"^P\.STAT1": "STAT1_Y727",
            r"^P\.AKT\.SER473\.": "pAKT_S473",
            r"^P\.ERK": "pERK_Y204",
            r"^P\.HER2": "ERBB2_Y1248",
            r"^P\.GSK3B": "GSK3B_S9",
            r"^P\.PDPK1": "PDPK1_S241",
            r"^P\.P90RSK": "RPS6KA1_S380",
            r"^P\.STAT3": "STAT3_Y705",
            r"^P\.S6$": "RPS6_S235_S236",
            r"^P\.AKT\.THR308\.": "pAKT_T308",
            r"^P\.4EBP1": "EIF4EBP1_T37_T46",
            r"^P\.SRC": "SRC_Y419",
            r"^P\.p.PLCG2": "PLCG2_Y759",
            r"^P\.BTK": "BTK_Y551",
            r"^P\.CREB": "CREB1_S133",
        }
    elif re.match(r'synthetic_[0-9]+_[0-9\.]+$', dataset):
        palias = {}
    else:
        raise ValueError("Dataset not supported!")

    for pname, prep in palias.items():
        obs_id = re.sub(pname, prep, obs_id)

    if model.observables.get(obs_id):
        return obs_id

    site_pattern = r"_([S|Y|T][0-9]+)"

    monomer = re.sub(site_pattern, "", obs_id)
    sites = sorted(list(re.findall(site_pattern, obs_id)))

    name = f'p{monomer}_{"_".join(sites)}' if sites else f"t{obs_id}"

    if model.observables.get(name, None):
        return name

    if model.monomers.get(monomer, None) and name.startswith("p"):
        print(f"could not map {obs_id} to {monomer}!")

    return ""


if __name__ == "__main__":
    MODEL = sys.argv[1]
    DATA = sys.argv[2]

    from cytof.problem import CytofProblem
    problem = CytofProblem(pathway_name=MODEL)
    model = problem.load_pysb()
    data_dir.mkdir(exist_ok=True, parents=True)

    if DATA == "dream_cytof":
        measurement_table, condition_table = problem.load_preprocess_petab_tables(model)
    elif DATA.startswith("synthetic"):
        N_HIDDEN = 6
        N_SAMPLES = int(DATA.split("_")[1])
        condition_table, measurement_table = generate_synthetic_data(
            problem, data_dir, DATA, latent_dimension=N_HIDDEN, n_samples=N_SAMPLES,
            std=float(DATA.split("_")[2]), n_features=200
        )
    else:
        raise RuntimeError("Unknown dataset!")

    # filter measurements for removed conditions
    condition_ids = condition_table[petab.CONDITION_ID].unique()
    measurement_table = measurement_table.loc[
        measurement_table.apply(
            lambda x: x[petab.SIMULATION_CONDITION_ID] in condition_ids
            and x[petab.PREEQUILIBRATION_CONDITION_ID] in condition_ids,
            axis=1,
        ), :
    ]

    observable_ids = [
        obs_id
        for obs_id in measurement_table.loc[:, petab.OBSERVABLE_ID].unique()
        if observable_id_to_model_expr(obs_id, DATA, model) != ""
    ]
    observable_table = pd.DataFrame(
        {
            petab.OBSERVABLE_NAME: observable_ids,
        }
    )
    observable_obs = [
        observable_id_to_model_expr(obs_id, DATA, model)
        for obs_id in observable_ids
    ]
    observable_table[petab.OBSERVABLE_ID] = [f"{obs}_obs" for obs in observable_obs]
    measurement_table[petab.OBSERVABLE_ID] = measurement_table[
        petab.OBSERVABLE_ID
    ].apply(
        lambda x: observable_id_to_model_expr(x, DATA, model) + "_obs"
        if observable_id_to_model_expr(x, DATA, model) != ""
        else x
    )

    observable_table[petab.OBSERVABLE_FORMULA] = [
        f"log(observableParameter1_{obs}_obs * {obs} "
        f"+ observableParameter2_{obs}_obs)"
        for obs in observable_obs
    ]
    observable_table[petab.NOISE_DISTRIBUTION] = petab.NORMAL
    observable_table[petab.NOISE_FORMULA] = [
        f"noiseParameter1_{obs}_obs" for obs in observable_obs
    ]

    def obs_pars(x):
        pars = f"{x[petab.OBSERVABLE_ID]}_scale;" f"{x[petab.OBSERVABLE_ID]}_offset"
        return pars

    measurement_table[petab.OBSERVABLE_PARAMETERS] = measurement_table.apply(
        obs_pars, axis=1
    )

    measurement_file = data_dir / MEASUREMENTS_FILE.format(data=DATA, model=MODEL)
    measurement_table.to_csv(measurement_file, sep="\t")

    condition_file = data_dir / CONDITIONS_FILE.format(data=DATA, model=MODEL)
    condition_table.set_index(petab.CONDITION_ID, inplace=True)
    condition_table.to_csv(condition_file, sep="\t")

    observable_file = data_dir / OBSERVABLES_FILE.format(data=DATA, model=MODEL)
    observable_table.set_index(petab.OBSERVABLE_ID, inplace=True)
    observable_table.to_csv(observable_file, sep="\t")
