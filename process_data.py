import sys
import re
import petab
import synapseclient
import pysb
import json

import pandas as pd
import numpy as np

import urllib.parse
import urllib.request

from collections import namedtuple
from pathlib import Path

from mEncoder.generate_data import generate_synthetic_data
from mEncoder import (
    load_pathway, data_dir, pretrain_dir,  PER_SAMPLE_OUTFILE_TEMP
)

SAMPLES = {
    'dream_cytof': [
        'c184A1', 'cBT20', 'cBT474', 'cBT549', 'cCAL148', 'cCAL851',
        'cCAL51', 'cDU4475', 'cEFM192A', 'cEVSAT', 'cHBL100', 'cHCC1187',
        'cHCC1395', 'cHCC1419', 'cHCC1500', 'cHCC1569', 'cHCC1599',
        'cHCC1937', 'cHCC1954', 'cHCC2157', 'cHCC2185', 'cHCC3153',
        'cHCC38', 'cHCC70', 'cHDQP1', 'cJIMT1', 'cMCF10A', 'cMCF10F',
        'cMCF7', 'cMDAMB134VI', 'cMDAMB157', 'cMDAMB175VII', 'cMDAMB361',
        'cMDAMB415', 'cMDAMB453', 'cMDAkb2', 'cMFM223', 'cMPE600', 'cMX1',
        'cOCUBM', 'cT47D', 'cUACC812', 'cUACC893', 'cZR7530'
    ],
    'synthetic_45': [f'sample_{isample}' for isample in range(45)],
    'synthetic_90': [f'sample_{isample}' for isample in range(90)],
}

Wildcards = namedtuple('Wildcards', ['data', 'samples'])


def training_samples(wildcards):
    samples = SAMPLES[wildcards.data]
    split, n_splits = wildcards.samples.split('_')
    split = int(split)
    n_splits = int(n_splits)
    n_samples = len(samples)
    return samples[:int(np.round(n_samples/n_splits*split))] + \
        samples[int(np.round(n_samples/n_splits*(split+1))):]


def test_samples(wildcards):
    samples = SAMPLES[wildcards.data]
    split, n_splits = wildcards.samples.split('_')
    split = int(split)
    n_splits = int(n_splits)
    n_samples = len(samples)
    return samples[int(np.round(n_samples/n_splits*split)):
                   int(np.round(n_samples/n_splits*(split+1))):]


def per_sample_pretraining_train(wildcards):
    return [
        pretrain_dir / '{model}' / '{data}' /
        PER_SAMPLE_OUTFILE_TEMP.format(sample=sample)
        for sample in training_samples(wildcards)
    ]


def per_sample_pretraining_test(wildcards):
    return [
        pretrain_dir / '{model}' / '{data}' /
        PER_SAMPLE_OUTFILE_TEMP.format(sample=sample)
        for sample in test_samples(wildcards)
    ]


def observable_id_to_model_expr(obs_id: str,
                                dataset: str,
                                model: pysb.Model) -> str:
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
    obs_id = obs_id.replace('-', '_').upper()
    if dataset == 'cytof':
        palias = {
            r'^P\.STAT5': 'STAT5A_Y694',
            r'^P\.MEK': 'pMEK_S222',
            r'^P\.S6K$': 'RPS6KB1_S412',
            r'^P\.STAT1': 'STAT1_Y727',
            r'^P\.AKT\.SER473\.': 'pAKT_S473',
            r'^P\.ERK': 'pERK_Y204',
            r'^P\.HER2': 'ERBB2_Y1248',
            r'^P\.GSK3B': 'GSK3B_S9',
            r'^P\.PDPK1': 'PDPK1_S241',
            r'^P\.P90RSK': 'RPS6KA1_S380',
            r'^P\.STAT3': 'STAT3_Y705',
            r'^P\.S6$': 'RPS6_S235_S236',
            r'^P\.AKT\.THR308\.': 'pAKT_T308',
            r'^P\.4EBP1': 'EIF4EBP1_T37_T46',
            r'^P\.SRC': 'SRC_Y419',
            r'^P\.p.PLCG2': 'PLCG2_Y759',
            r'^P\.BTK': 'BTK_Y551',
            r'^P\.CREB': 'CREB1_S133',
        }
    else:
        raise ValueError('Dataset not supported!')

    for pname, prep in palias.items():
        obs_id = re.sub(pname, prep, obs_id)

    if model.observables.get(obs_id, None):
        return obs_id

    site_pattern = r'_([S|Y|T][0-9]+)'

    monomer = re.sub(site_pattern, '', obs_id)
    sites = sorted(list(re.findall(site_pattern, obs_id)))

    name = f'p{monomer}_{"_".join(sites)}' if sites else f't{obs_id}'

    if model.observables.get(name, None):
        return name

    if model.monomers.get(monomer, None) and name.startswith('p'):
        print(f'could not map {obs_id} to {monomer}!')

    return ''


def convert_time_to_minutes(time_str):
    if not isinstance(time_str, str):
        return float(time_str)
    if time_str.endswith('min'):
        return float(time_str[:-4])
    if time_str.endswith('hr'):
        return float(time_str[:-3])*60


if __name__ == '__main__':
    MODEL = sys.argv[1]
    DATA = sys.argv[2]

    data_dir.mkdir(exist_ok=True, parents=True)

    if DATA.startswith('synthetic'):
        N_HIDDEN = 4
        N_SAMPLES = int(DATA.split('_')[1])
        generate_synthetic_data(MODEL, N_HIDDEN, N_SAMPLES)

    else:

        model = load_pathway('pw_' + MODEL)

        if DATA == 'dream_cytof':
            syn = synapseclient.Synapse()
            syn.login()
            files = [
                'syn20613594',  # 184A1
                'syn20613595',  # BT20
                'syn20613596',  # BT474
                'syn20613597',  # BT549
                'syn20613598',  # CAL148
                'syn20613599',  # CAL51
                'syn20613600',  # CAL851
                'syn20613601',  # DU4475
                'syn20613660',  # EFM192A
                'syn20613665',  # EVSAT
                'syn20613668',  # HBL100
                'syn20613674',  # HCC1187
                'syn20613687',  # HCC1395
                'syn20613696',  # HCC1419
                'syn20613702',  # HCC1500
                'syn20613708',  # HCC1569
                'syn20613710',  # HCC1599
                'syn20613719',  # HCC1937
                'syn20613739',  # HCC1954
                'syn20613793',  # HCC2157
                'syn20613802',  # HCC2185
                'syn20613814',  # HCC3153
                'syn20613821',  # HCC38
                'syn20613832',  # HCC70
                'syn20613849',  # HDQP1
                'syn20613865',  # JIMT1
                'syn20613880',  # MCF10A
                'syn20613911',  # MCF10F
                'syn20613920',  # MCF7
                'syn20613935',  # MDAMB134VI
                'syn20613939',  # MDAMB157
                'syn20613943',  # MDAMB175VII
                'syn20613962',  # MDAMB361
                'syn20613975',  # MDAMB415
                'syn20613988',  # MDAMB453
                'syn20613930',  # MDAkb2
                'syn20613995',  # MFM223
                'syn20614008',  # MPE600
                'syn20614033',  # MX1
                'syn20614045',  # OCUBM
                'syn20614052',  # T47D
                'syn20614063',  # UACC812
                'syn20614074',  # UACC893
                'syn20614085',  # ZR7530
            ]
            mean_data = []
            std_data = []
            group_ids = ['treatment', 'cell_line', 'time', 'fileID']
            for file in files:
                df = pd.read_csv(syn.get(file).path)
                for ids, data in df.groupby(group_ids):
                    if f'c{ids[1]}' not in SAMPLES[DATA]:
                        continue
                    markers = [c for c in data.columns
                               if c not in group_ids + ['cellID']]
                    m = data[markers].median()
                    std = data[markers].std()
                    for sdf in [m, std]:
                        sdf['treatment'] = ids[0]
                        sdf['cell_line'] = ids[1]
                        sdf['time'] = ids[2]
                        sdf['fileID'] = ids[3]
                    mean_data.append(m)
                    std[std.isna()] = 1.0
                    std_data.append(std)

            df_phospho_mean = pd.concat(mean_data, axis=1).T
            df_phospho_std = pd.concat(std_data, axis=1).T
            d = {
                desc: pd.concat(data, axis=1).T
                for desc, data in (
                    ('mean', mean_data),
                    ('std', std_data)
                )
            }
            id_vars = ['cell_line', 'treatment', 'time', 'fileID']
            df_phospho_condition = d['mean'][id_vars]
            for sdf in d.values():
                sdf.drop(columns=id_vars, inplace=True)
            df_phospho = pd.concat(
                d.values(),
                axis=1,
                keys=d.keys()
            ).swaplevel(0, 1, axis=1)
            df_phospho = pd.concat((df_phospho_condition, df_phospho), axis=1)

            measurement_table_phospho = pd.melt(
                df_phospho,
                id_vars=id_vars,
                var_name=petab.OBSERVABLE_ID,
            )

            measurement_table_phospho[[petab.OBSERVABLE_ID, 'type']] = \
                measurement_table_phospho[petab.OBSERVABLE_ID].to_list()

            measurement_table_phospho = measurement_table_phospho.set_index(
                ['type', petab.OBSERVABLE_ID] + id_vars
            ).unstack('type').droplevel(axis=1, level=0).reset_index()

            measurement_table_phospho.rename(columns={
                'cell_line': petab.PREEQUILIBRATION_CONDITION_ID,
                'time': petab.TIME,
                'mean': petab.MEASUREMENT,
                'std': petab.NOISE_PARAMETERS
            }, inplace=True)

            measurement_table_phospho[petab.PREEQUILIBRATION_CONDITION_ID] = \
                measurement_table_phospho[
                    petab.PREEQUILIBRATION_CONDITION_ID
                ].apply(lambda x: f'c{x}')

            measurement_table_phospho[petab.SIMULATION_CONDITION_ID] = \
                measurement_table_phospho.apply(
                    lambda x: f'{x[petab.PREEQUILIBRATION_CONDITION_ID]}__'
                              f'{x.treatment}', axis=1
                )
            measurement_table_phospho.drop(columns=['treatment', 'fileID'],
                                           inplace=True)

            df_proteomics = pd.read_csv(syn.get('syn20690775').path,
                                        index_col=[0])
            df_proteomics[petab.OBSERVABLE_ID] = df_proteomics.index

            df_proteomics = df_proteomics[
                df_proteomics[petab.OBSERVABLE_ID].apply(lambda x:
                                                         ';' not in x)
            ]

            measurement_table_proteomics = pd.melt(
                df_proteomics,
                id_vars=[petab.OBSERVABLE_ID],
                var_name=petab.PREEQUILIBRATION_CONDITION_ID,
                value_name=petab.MEASUREMENT,
            )

            UP_ID_JSON = 'up_ids.json'
            if Path(UP_ID_JSON).exists():
                with open(UP_ID_JSON, 'r') as fp:
                    up_ids = json.load(fp)
            else:
                url = 'https://www.uniprot.org/uploadlists/'

                params = {
                    'from': 'ACC+ID',
                    'to': 'GENENAME',
                    'format': 'tab',
                    'query':
                        ' '.join(df_proteomics[petab.OBSERVABLE_ID].unique())
                }

                data = urllib.parse.urlencode(params)
                data = data.encode('utf-8')
                req = urllib.request.Request(url, data)
                with urllib.request.urlopen(req) as f:
                    response = f.read()
                up_ids = dict([
                    mapping.split('\t')
                    for mapping in response.decode('utf-8').split('\n')
                    if '\t' in mapping
                ])
                with open(UP_ID_JSON, 'w') as fp:
                    json.dump(up_ids, fp)

            measurement_table_proteomics[petab.OBSERVABLE_ID] = \
                measurement_table_proteomics[petab.OBSERVABLE_ID].apply(
                    lambda x: up_ids.get(x, '')
                )

            measurement_table_proteomics = measurement_table_proteomics[
                measurement_table_proteomics[petab.OBSERVABLE_ID] != ''
            ]

            measurement_table_proteomics.dropna(axis=0,
                                                subset=[petab.MEASUREMENT],
                                                inplace=True)

            measurement_table_proteomics[
                petab.PREEQUILIBRATION_CONDITION_ID
            ] = measurement_table_proteomics[
                petab.PREEQUILIBRATION_CONDITION_ID
            ].apply(lambda x: f'c{x.split("_")[0]}')

            measurement_table_proteomics[petab.SIMULATION_CONDITION_ID] = \
                measurement_table_proteomics[
                    petab.PREEQUILIBRATION_CONDITION_ID
                ]

            measurement_table_proteomics[petab.TIME] = 0.0

            # ignore proteomics data for now
            measurement_table = pd.concat([
                measurement_table_phospho,
                measurement_table_proteomics
            ])

            condition_table = pd.DataFrame({
                petab.CONDITION_ID:
                    np.unique(np.concatenate([
                        measurement_table[petab.SIMULATION_CONDITION_ID],
                        measurement_table[petab.PREEQUILIBRATION_CONDITION_ID]
                    ]))
            })

            # ignore "full" for now
            condition_table = condition_table[
                condition_table[petab.CONDITION_ID].apply(
                    lambda x: 'full' not in x.split('__')
                )
            ]

            perturbations = np.unique([
                p
                for c in condition_table[petab.CONDITION_ID]
                if len(c.split('__')) > 1
                for p in c.split('__')[1:] if p != 'full'
            ])
            for pert in perturbations:
                if model.parameters.get(f'{pert}_0') is None:
                    # remove condition
                    condition_table = condition_table[
                        condition_table[petab.CONDITION_ID].apply(
                            lambda x: pert not in x.split('__')
                        )
                    ]
                    continue
                condition_table[f'{pert}_0'] = \
                    condition_table[petab.CONDITION_ID].apply(
                        lambda x: float(int(pert in x.split('__')))
                    )

            condition_table['EGF_0'] = \
                condition_table[petab.CONDITION_ID].apply(
                    lambda x: float('__' in x)
                )

            observable_mode = 'cytof'

        # filter measurements for removed conditions
        condition_ids = condition_table[petab.CONDITION_ID].unique()
        measurement_table = measurement_table[
            measurement_table.apply(
                lambda x: x[petab.SIMULATION_CONDITION_ID] in condition_ids and
                x[petab.PREEQUILIBRATION_CONDITION_ID] in condition_ids,
                axis=1
            )
        ]

        observable_ids = [
            obs_id for obs_id in
            measurement_table.loc[:, petab.OBSERVABLE_ID].unique()
            if observable_id_to_model_expr(obs_id, observable_mode,
                                           model) != ''
        ]
        observable_table = pd.DataFrame({
            petab.OBSERVABLE_NAME: observable_ids,
        })
        observable_obs = [
            observable_id_to_model_expr(obs_id, observable_mode, model)
            for obs_id in observable_ids
        ]
        observable_table[petab.OBSERVABLE_ID] = \
            [
                f'{obs}_obs'
                for obs in observable_obs
            ]
        measurement_table[petab.OBSERVABLE_ID] = \
            measurement_table[petab.OBSERVABLE_ID].apply(
                lambda x: observable_id_to_model_expr(x, observable_mode,
                                                      model)
                + '_obs'
                if observable_id_to_model_expr(x, observable_mode, model) != ''
                else x
            )

        observable_table[petab.OBSERVABLE_FORMULA] = [
            f'log(observableParameter1_{obs}_obs * {obs} '
            f'+ observableParameter2_{obs}_obs)'
            for obs in observable_obs
        ]
        observable_table[petab.NOISE_DISTRIBUTION] = 'normal'
        observable_table[petab.NOISE_FORMULA] = [
            f'noiseParameter1_{obs}_obs' for obs in observable_obs
        ]

        if DATA == 'dream_cytof':
            def obs_pars(x):
                pars = f'{x[petab.OBSERVABLE_ID]}_scale;' \
                       f'{x[petab.OBSERVABLE_ID]}_offset'
                return pars
        else:
            def obs_pars(x):
                pars = f'{x[petab.OBSERVABLE_ID]}_scale;' \
                       f'{x[petab.OBSERVABLE_ID]}_offset'
                return pars

        measurement_table[petab.OBSERVABLE_PARAMETERS] = \
            measurement_table.apply(obs_pars, axis=1)

        measurement_file = data_dir / f'{DATA}__{MODEL}__measurements.tsv'
        measurement_table.to_csv(measurement_file, sep='\t')

        condition_file = data_dir / f'{DATA}__{MODEL}__conditions.tsv'
        condition_table.set_index(petab.CONDITION_ID, inplace=True)
        condition_table.to_csv(condition_file, sep='\t')

        observable_file = data_dir / f'{DATA}__{MODEL}__observables.tsv'
        observable_table.set_index(petab.OBSERVABLE_ID, inplace=True)
        observable_table.to_csv(observable_file, sep='\t')
