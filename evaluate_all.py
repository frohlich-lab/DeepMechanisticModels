import sys
import pandas as pd
import numpy as np
import petab
import seaborn as sns
import matplotlib.pyplot as plt

from common import fig_dir, EVALUATE_ALL, tpl_evaluation_file, EVALUATION_TRAINING

MODEL = sys.argv[1]
DATA = sys.argv[2]
SAMPLES = sys.argv[3]

outdir = fig_dir / MODEL / DATA

avgs = dict()
ps = dict()
dfs = []
for dataset in ["train", "test"]:
    # cross sample pretraining
    pretraining = pd.read_csv(
        tpl_evaluation_file.format(samples=SAMPLES, model=MODEL, data=DATA, dataset=dataset, mode='cross_sample'),
        index_col=0
    )
    # training
    # training = pd.read_csv(
    #     EVALUATION_TRAINING.format(dataset=dataset, model=MODEL, data=DATA, samples=SAMPLES), index_col=0
    # )

    # average
    avg = pd.read_csv(
        tpl_evaluation_file.format(samples=SAMPLES, model=MODEL, data=DATA, dataset=dataset, mode='average'),
        index_col=0
    )

    # per sample
    df_ps = pd.read_csv(
        tpl_evaluation_file.format(samples=SAMPLES, model=MODEL, data=DATA, dataset=dataset, mode='per_sample'),
        index_col=0
    )

    # dfd = pd.concat([training, pretraining])
    dfd = pd.concat([pretraining, avg, df_ps])
    dfd["dataset"] = dataset
    dfs.append(dfd)

df = pd.concat(dfs).reset_index()
df.rename(columns={"alpha": "l1 regularization", "layers": "latent dim"}, inplace=True)
df.loc[df["type"] == "cross_sample", "type"] = "pca embedding"
df.loc[df["type"] == "full", "type"] = "end-to-end"

for gb in ('observable', 'time', 'condition', 'sample', 'all'):
    gbs = ("dataset", "type", "latent dim", "l1 regularization")
    if gb != 'all':
        gbs = (gb, *gbs)
    df_gb = pd.DataFrame([
        dict(zip(gbs, group), rmse=np.sqrt(group_df["res"].apply(lambda x: np.power(x, 2)).mean()))
        for group, group_df in df.groupby(gbs)
    ])

    if gb == 'all':
        data = df_gb
    else:
        data = df_gb[df_gb.type == 'pca embedding']

    g = sns.FacetGrid(data=data, row="dataset", col="latent dim")
    g.map_dataframe(
        sns.barplot, x=gb if gb != 'all' else "type", y="rmse", hue="l1 regularization"
    )
    g.set(yscale="log")
    g.add_legend()
    plt.savefig(EVALUATE_ALL.format(model=MODEL, data=DATA, samples=SAMPLES, group=gb))
