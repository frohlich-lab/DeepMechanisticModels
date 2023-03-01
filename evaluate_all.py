import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools as itt

from common import fig_dir, EVALUATE_ALL, tpl_evaluation_file
from training_configuration import ALPHAS, LATENT_DIMS, CONTEXTS, SPLITS

MODEL = sys.argv[1]
DATA = sys.argv[2]

outdir = fig_dir / MODEL / DATA

avgs = dict()
ps = dict()
dfs = []
for samples in SPLITS:
    for dataset in ["train", "test"]:
        # cross sample pretraining
        pretraining = pd.read_csv(
            tpl_evaluation_file.format(samples=samples, model=MODEL, data=DATA, dataset=dataset, mode='cross_sample'),
            index_col=0
        )

        # training
        # training = pd.read_csv(
        #     EVALUATION_TRAINING.format(dataset=dataset, model=MODEL, data=DATA, samples=SAMPLES), index_col=0
        # )

        # average
        avg = pd.read_csv(
            tpl_evaluation_file.format(samples=samples, model=MODEL, data=DATA, dataset=dataset, mode='average'),
            index_col=0
        )

        # model average
        avg_model = pd.read_csv(
            tpl_evaluation_file.format(samples=samples, model=MODEL, data=DATA, dataset=dataset, mode='avg_model'),
            index_col=0
        )

        # per sample
        ps = pd.read_csv(
            tpl_evaluation_file.format(samples=samples, model=MODEL, data=DATA, dataset=dataset, mode='per_sample'),
            index_col=0
        )

        avg_ps_dfs = []
        # copy average/per sample
        for alpha, ldim, ctxt in itt.product(ALPHAS, LATENT_DIMS, CONTEXTS):
            for rdf in [avg, avg_model, ps]:
                avg_ps_df = rdf.copy()
                avg_ps_df['alpha'] = alpha
                avg_ps_df['layers'] = ldim
                avg_ps_df['context'] = ctxt
                avg_ps_dfs.append(avg_ps_df)

        # dfd = pd.concat([training, pretraining])
        dfd = pd.concat([pretraining, *avg_ps_dfs])
        dfd["dataset"] = dataset
        dfs.append(dfd)

df = pd.concat(dfs).reset_index()
df.rename(columns={"alpha": "l1 regularization", "layers": "latent dim"}, inplace=True)
df.loc[df["type"] == "cross_sample", "type"] = "pca embedding"
df.loc[df["type"] == "full", "type"] = "end-to-end"

for gb in ('observable', 'time', 'condition', 'sample', 'all'):
    gbs = ["dataset", "type", "latent dim", "l1 regularization"]
    if gb != 'all':
        gbs = [gb, *gbs]
    df_gb = pd.DataFrame([
        dict(zip(gbs, group), rmse=np.sqrt(group_df["res"].apply(lambda x: np.power(x, 2)).mean()))
        for group, group_df in df.groupby(gbs)
    ])

    if gb == 'time':
        # filter non-canonical timepoints (not enough datapoints)
        data = df_gb[np.logical_not(df_gb.time.isin([12, 14, 15, 16, 25, 35]))]
    else:
        data = df_gb

    g = sns.FacetGrid(data=data, row=gb if gb != 'all' else "dataset", col="latent dim")
    g.map_dataframe(
        sns.lineplot, x="l1 regularization", y="rmse", style="dataset", hue="type"
    )
    g.set(xscale='log', ylim=(0, 1.5))
    g.add_legend()
    rfile = EVALUATE_ALL.format(model=MODEL, data=DATA, group=gb)
    plt.savefig(rfile)
    data.to_csv(rfile.replace('.pdf', '.csv'))
