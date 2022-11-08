import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from mEncoder import (
    fig_dir
)

MODEL = sys.argv[1]
DATA = sys.argv[2]
SAMPLES = sys.argv[3]

outdir = fig_dir / MODEL / DATA


avgs = dict()
ps = dict()
dfs = []
for dataset in ['train', 'test']:
    # cross sample pretraining
    pretraining = pd.read_csv(
        outdir / f'{SAMPLES}_evaluate_pretrain_cross_sample_{dataset}.csv',
        index_col=0
    )
    # training
    training = pd.read_csv(
        outdir / f'{SAMPLES}_evaluate_training_{dataset}.csv', index_col=0
    )

    # average
    avg = pd.read_csv(
        outdir / f'{SAMPLES}_evaluate_average_{dataset}.csv', index_col=0
    )
    avgs[dataset] = np.power(10, avg.rmse.apply(np.log10).mean())

    # per sample
    df_ps = pd.read_csv(
        outdir / f'{SAMPLES}_evaluate_pretrain_per_sample_{dataset}.csv',
        index_col=0
    )
    ps[dataset] = np.power(10, df_ps.rmse.apply(np.log10).mean())

    dfd = pd.concat([training, pretraining])
    dfd['dataset'] = dataset
    dfs.append(dfd)

df = pd.concat(dfs).reset_index()
df.rename(columns={
    'alpha': 'l2 regularization',
    'layers': 'latent dim'
}, inplace=True)
df.loc[df['type'] == 'cross_sample', 'type'] = 'pca embedding'
df.loc[df['type'] == 'full', 'type'] = 'end-to-end'

g = sns.FacetGrid(data=df, row='dataset', col='latent dim')
g.map_dataframe(sns.lineplot, x='l2 regularization', y='rmse', style='type')
g.set(yscale='log', xscale='log')
g.add_legend()
for ids, dataset in enumerate(['train', 'test']):
    for ax in g.axes[ids, :]:
        ax.axhline(avgs[dataset], ls=':', c='r')
        ax.axhline(ps[dataset], ls=':', c='b')

plt.savefig(outdir / f'{SAMPLES}_evaluate_all.pdf')
