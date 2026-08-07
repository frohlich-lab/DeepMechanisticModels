from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

lb_labels = ["Luminal", "Basal", "Normal", "Other"]


def plot_parameter_heatmaps(
    param_df: pd.DataFrame,
    param_cols: List,
    figure_filepath: Union[str, Path],
):
    # Set colorbar range
    for samples in param_df.index.get_level_values("samples").unique():
        df = param_df.loc[
            param_df.index.get_level_values("samples") == samples
        ]
        df = df.droplevel("samples", axis="index")
        vlim = df[param_cols].abs().max().max()
        vmin, vmax = -vlim, vlim

        sns.clustermap(
            data=df[
                df[param_cols].abs().max()[lambda x: x > 0.0].index.tolist()
            ],
            col_cluster=True,
            vmin=vmin,
            vmax=vmax,
            cmap="vlag",
            xticklabels=True,
            yticklabels=True,
            figsize=(6, 12),
        )
        figure_filepath.parent.mkdir(exist_ok=True, parents=True)
        plt.tight_layout()
        plt.savefig(str(figure_filepath) + f"_{samples}.pdf")
        plt.close()
