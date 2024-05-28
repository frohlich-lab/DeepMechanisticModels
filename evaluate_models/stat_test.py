import pandas as pd
import numpy as np
from scipy.stats import shapiro, ttest_rel, wilcoxon, false_discovery_control
from training_configuration import (ORTH_REG_STRATEGIES,
                                    ALPHAS, BETAS, GAMMAS, DELTAS,
                                    LATENT_DIMS)
import itertools as itt


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


def adjust_p_value(res_df, cols, group_attributes, method_func=false_discovery_control):
    for col in cols:
        res_df[f"adj_{col}"] = res_df.groupby(
            by=group_attributes)[col].transform(
            lambda x: method_func(x)
        )
    return res_df


def statistical_significance_test(data_stat_tests):
    # only interested in carrying out statistical tests on DMM results
    dmm_stat_tests = data_stat_tests[data_stat_tests.ref == 'DMM']

    hyperparameter_values = {
        'n_hidden': LATENT_DIMS,
        'l1reg_inflate': ALPHAS,
        'oreg_inflate': BETAS,
        'l1reg_encode': GAMMAS,
        'oreg_encode': DELTAS
    }

    hyperparam_list = [
        'l1reg_inflate',
        'oreg_inflate',
        'l1reg_encode',
        'oreg_encode',
        'n_hidden'
    ]

    # Initialise list of DataFrames for overall results
    res_dfs = []

    if len(ORTH_REG_STRATEGIES) > 1:  # if only one strategy, do not test
        hyperparam_list = ['orth_reg_strategy'] + hyperparam_list

    for hyperparameter in hyperparam_list:
        if hyperparameter == 'n_hidden':
            # Initialise list of DataFrames for n_hidden comparisons
            res_dfs_n_hidden = []
            for n_hidden_pair in itt.combinations(hyperparameter_values[hyperparameter], 2):
                n_hidden1, n_hidden2 = n_hidden_pair

                res_df_partial = dmm_stat_tests.groupby(by='context').apply(
                    lambda df: stat_test_hyperparameter(df, hyperparameter='n_hidden',
                                                        hp_value1=n_hidden1,
                                                        hp_value2=n_hidden2,
                                                        alternative='less')
                ).reset_index()
                res_df_partial['hyperparameter'] = hyperparameter
                res_df_partial['hyperparameter_value'] = f"{n_hidden1} vs {n_hidden2}"
                res_df_partial['n_hidden1'] = n_hidden1
                res_df_partial['n_hidden2'] = n_hidden2
                res_df_partial['test_kind'] = f"RMSE_{n_hidden1} < RMSE_{n_hidden2}"
                res_df_partial['n_hidden'] = None

                res_dfs_n_hidden.append(res_df_partial)

            # After evaluating all pairs
            # Concatenate into single DataFrame to adjust p-values
            res_df_n_hidden = pd.concat([*res_dfs_n_hidden])
            # Compute adjusted p-values for t-test and Wilcoxon test
            res_df_n_hidden['adj_t-test_p-value'] = res_df_n_hidden.groupby(
                by=['context']
            )['t-test_p-value'].transform(
                lambda x: false_discovery_control(x)
            )
            res_df_n_hidden['adj_Wilcoxon_p-value'] = res_df_n_hidden.groupby(
                by=['context']
            )['Wilcoxon_p-value'].transform(
                lambda x: false_discovery_control(x)
            )   # most conservative option - group all tests for the same context together
            # now concatenate this to the rest of the dataframe or start it if it does not exist yet

            res_dfs.append(res_df_n_hidden)

        elif hyperparameter == 'orth_reg_strategy':
            # Single comparison: L1 vs L2 - do not need list of DataFrames
            res_df_partial = dmm_stat_tests.groupby(by=['context', 'n_hidden']).apply(
                lambda df: stat_test_hyperparameter(df, hyperparameter=hyperparameter,
                                                    hp_value1='L1',
                                                    hp_value2='L2',
                                                    alternative='greater')
            ).reset_index()
            res_df_partial['hyperparameter'] = hyperparameter
            res_df_partial['hyperparameter_value'] = 'L2 vs L1'
            res_df_partial['n_hidden1'] = None
            res_df_partial['n_hidden2'] = None
            res_df_partial['test_kind'] = "RMSE_L1 > RMSE_L2"
            # p-values for orth_reg_strategy can be adjusted right away
            res_df_partial = adjust_p_value(
                res_df_partial,
                ['t-test_p-value', 'Wilcoxon_p-value'],
                ['n_hidden', 'context', 'hyperparameter'],
                method_func=false_discovery_control,
            )
            # Append to list of overall results
            res_dfs.append(res_df_partial)

        else:
            # Initialise list of DataFrames for pair-wise hyperparameter value vs 0 comparisons
            # (gets re-initialised for every newly examined hyperparameter)
            res_dfs_hp = []
            pairs = [pair for pair in itt.combinations(hyperparameter_values[hyperparameter], 2) if pair[0] == 0]
            for pair in pairs:
                hp_value1, hp_value2 = pair
                res_df_partial = dmm_stat_tests.groupby(by=['context', 'n_hidden']).apply(
                    lambda df: stat_test_hyperparameter(df, hyperparameter=hyperparameter,
                                                        hp_value1=hp_value1,
                                                        hp_value2=hp_value2,
                                                        alternative='greater')
                ).reset_index()
                res_df_partial['hyperparameter'] = hyperparameter
                res_df_partial['hyperparameter_value'] = hp_value2
                res_df_partial['n_hidden1'] = None
                res_df_partial['n_hidden2'] = None
                res_df_partial['test_kind'] = f"RMSE_{hp_value1} > RMSE_{hp_value2}"

                res_dfs_hp.append(res_df_partial)
            # After evaluating all pairs
            # Concatenate into single DataFrame to adjust p-values
            res_df_hp = pd.concat([*res_dfs_hp])
            # Compute adjusted p-values for t-test and Wilcoxon test
            res_df_hp = adjust_p_value(
                res_df_hp,
                ['t-test_p-value', 'Wilcoxon_p-value'],
                ['n_hidden', 'context', 'hyperparameter'],
                method_func=false_discovery_control,
            )
            # Append to list of overall results
            res_dfs.append(res_df_hp)

    # Concatenate all results DataFrames into one
    res_df = pd.concat([*res_dfs])
    # Compute log10-scaled Wilcoxon adjusted p-value and Wilcoxon statistic (for plots)
    res_df['-log10_adj_Wilcoxon_p-value'] = -np.log10(res_df['adj_Wilcoxon_p-value'])
    res_df['log10_Wilcoxon_statistic'] = np.log10(res_df['Wilcoxon_statistic'])
    # Determine which hyperparam combinations induce a statistically significant improvement
    # criterion: adjusted (false discovery rate) Wilcoxon test p-value < 0.05
    res_df['stat-significant'] = res_df['adj_Wilcoxon_p-value'] < 0.05
    # Compute log10-scaled hyperparameter value (apart from orth_reg_strategy)
    res_df['log10hp_value'] = res_df['hyperparameter_value'].apply(
        lambda x: np.log10(x) if np.issubdtype(type(x), np.number) else x)
    return res_df
