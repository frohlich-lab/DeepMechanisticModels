import pandas as pd
import numpy as np
from scipy.stats import normaltest, shapiro, ttest_rel, wilcoxon, false_discovery_control
from training_configuration import (ORTH_REG_STRATEGIES,
                                    ALPHAS, BETAS, GAMMAS, DELTAS,
                                    LATENT_DIMS)
import itertools as itt
import seaborn as sns


def stat_test_hyperparameter(df, hyperparameter, hp_value1, hp_value2, alternative):
    return pd.Series([*shapiro_test(df, hyperparameter, hp_value1, hp_value2),
                      *ttest_rel_test(df, hyperparameter, hp_value1, hp_value2, alternative=alternative),
                      *wilcoxon_test(df, hyperparameter, hp_value1, hp_value2, alternative=alternative)],
                     index=['Shapiro_statistic', 'Shapiro_p-value',
                            't-test_statistic', 't-test_p-value',
                            'Wilcoxon_statistic', 'Wilcoxon_p-value'])


def wilcoxon_test(df, hyperparameter, hp_value1, hp_value2, alternative):
    return wilcoxon(np.concatenate(df[df[hyperparameter] == hp_value1]['rmse_list'].values),
                    np.concatenate(df[df[hyperparameter] == hp_value2]['rmse_list'].values), alternative=alternative,
                    axis=None)


def ttest_rel_test(df, hyperparameter, hp_value1, hp_value2, alternative):
    return ttest_rel(np.concatenate(df[df[hyperparameter] == hp_value1]['rmse_list'].values),
                     np.concatenate(df[df[hyperparameter] == hp_value2]['rmse_list'].values), alternative=alternative,
                     axis=None)


def shapiro_test(df, hyperparameter, hp_value1, hp_value2):
    return shapiro(np.concatenate(df[df[hyperparameter] == hp_value1]['rmse_list'].values) -
                   np.concatenate(df[df[hyperparameter] == hp_value2]['rmse_list'].values))


def statistical_significance_test(data_stat_tests):
    # only interested in carrying out statistical tests on DMM results
    dmm_stat_tests = data_stat_tests[data_stat_tests.ref == 'DMM']

    counter = 0
    counter_n_hidden = 0
    counter_hp = 0

    hyperparameter_values = {
        'n_hidden' : LATENT_DIMS,
        'l1reg_inflate': ALPHAS,
        'oreg_inflate': BETAS,
        'l1reg_encode': GAMMAS,
        'oreg_encode': DELTAS
    }

    for hyperparameter in ['orth_reg_strategy',
                           'l1reg_inflate',
                           'oreg_inflate',
                           'l1reg_encode',
                           'oreg_encode',
                           'n_hidden']:
        if hyperparameter == 'n_hidden':
            for n_hidden_pair in itt.combinations(hyperparameter_values[hyperparameter], 2):
                n_hidden1, n_hidden2 = n_hidden_pair

                res_df_partial = dmm_stat_tests.groupby(by='context').apply(
                    lambda df: stat_test_hyperparameter(df, hyperparameter='latent dim',
                                                        hp_value1=n_hidden1,
                                                        hp_value2=n_hidden2,
                                                        alternative='less')
                ).reset_index().rename(columns = {'latent dim' : 'n_hidden'})
                res_df_partial['hyperparameter'] = hyperparameter
                res_df_partial['hyperparameter_value'] = f"{n_hidden1} vs {n_hidden2}"
                res_df_partial['n_hidden1'] = n_hidden1
                res_df_partial['n_hidden2'] = n_hidden2
                res_df_partial['test_kind'] = f"RMSE_{n_hidden1} < RMSE_{n_hidden2}"
                res_df_partial['n_hidden'] = None

                if counter_n_hidden == 0:
                    res_df_n_hidden = res_df_partial
                    counter_n_hidden += 1
                else:
                    res_df_n_hidden = pd.concat([res_df_n_hidden, res_df_partial], axis=0)
            # after evaluating all pairs
            res_df_n_hidden['adj_t-test_p-value'] = res_df_n_hidden.groupby(by=['context'])['t-test_p-value'].transform(
                lambda x: false_discovery_control(x)
            )
            res_df_n_hidden['adj_Wilcoxon_p-value'] = res_df_n_hidden.groupby(by=['context'])['Wilcoxon_p-value'].transform(
                lambda x: false_discovery_control(x)
            )  # most conservative option - group all tests for the same context together
            # now concatenate this to the rest of the dataframe or start it if it does not exist yet
            if counter == 0:
                res_df = res_df_n_hidden
                counter += 1
            else:
                res_df = pd.concat([res_df, res_df_n_hidden])
                del res_df_n_hidden

        else:
            if hyperparameter == 'orth_reg_strategy':
                res_df_partial = dmm_stat_tests.groupby(by=['context', 'latent dim']).apply(
                    lambda df: stat_test_hyperparameter(df, hyperparameter=hyperparameter,
                                                        hp_value1='L1',
                                                        hp_value2='L2',
                                                        alternative='greater')
                ).reset_index().rename(columns = {'latent dim' : 'n_hidden'})
                res_df_partial['hyperparameter'] = hyperparameter
                res_df_partial['hyperparameter_value'] = 'L2 vs L1'
                res_df_partial['n_hidden1'] = None
                res_df_partial['n_hidden2'] = None
                res_df_partial['test_kind'] = f"RMSE_L1 > RMSE_L2"
                # p-values for orth_reg_strategy can be adjusted right away
                res_df_partial['adj_t-test_p-value'] = res_df_partial.groupby(
                    by=['n_hidden', 'context', 'hyperparameter'])['t-test_p-value'].transform(
                    lambda x: false_discovery_control(x)
                )
                res_df_partial['adj_Wilcoxon_p-value'] = res_df_partial.groupby(
                    by=['n_hidden', 'context', 'hyperparameter'])['Wilcoxon_p-value'].transform(
                    lambda x: false_discovery_control(x)
                )

                if counter == 0:
                    res_df = res_df_partial
                    counter += 1
                else:
                    res_df = pd.concat([res_df, res_df_partial], axis=0)

            else:
                pairs = [pair for pair in itt.combinations(hyperparameter_values[hyperparameter], 2) if pair[0] == 0]
                for pair in pairs:
                    hp_value1, hp_value2 = pair
                    res_df_partial = dmm_stat_tests.groupby(by=['context', 'latent dim']).apply(
                        lambda df: stat_test_hyperparameter(df, hyperparameter=hyperparameter,
                                                            hp_value1=hp_value1,
                                                            hp_value2=hp_value2,
                                                            alternative='greater')
                    ).reset_index().rename(columns = {'latent dim' : 'n_hidden'})
                    res_df_partial['hyperparameter'] = hyperparameter
                    res_df_partial['hyperparameter_value'] = hp_value2
                    res_df_partial['n_hidden1'] = None
                    res_df_partial['n_hidden2'] = None
                    res_df_partial['test_kind'] = f"RMSE_{hp_value1} > RMSE_{hp_value2}"

                    if counter_hp == 0:
                        res_df_hp = res_df_partial
                        counter_hp += 1
                    else:
                        res_df_hp = pd.concat([res_df_hp, res_df_partial], axis=0)
                # After evaluating all pairs
                res_df_hp['adj_t-test_p-value'] = res_df_hp.groupby(
                    by=['n_hidden', 'context', 'hyperparameter'])['t-test_p-value'].transform(
                    lambda x: false_discovery_control(x)
                )
                res_df_hp['adj_Wilcoxon_p-value'] = res_df_hp.groupby(
                    by=['n_hidden', 'context', 'hyperparameter'])['Wilcoxon_p-value'].transform(
                    lambda x: false_discovery_control(x)
                )
                # Restart counter
                counter_hp = 0
                # and concatenate results to growing res_df
                if counter == 0:
                    res_df = res_df_hp
                    counter += 1
                else:
                    res_df = pd.concat([res_df, res_df_hp], axis=0)
                    del res_df_hp

    res_df['-log10_adj_Wilcoxon_p-value'] = -np.log10(res_df['adj_Wilcoxon_p-value'])
    res_df['log10_Wilcoxon_statistic'] = np.log10(res_df['Wilcoxon_statistic'])
    res_df['stat-significant'] = res_df['adj_Wilcoxon_p-value'] < 0.05
    res_df['log10hp_value'] = res_df['hyperparameter_value'].apply(
        lambda x: np.log10(x) if np.issubdtype(type(x), np.number) else x)
    # res_df.to_csv('stat_tests_all.csv')
    return res_df