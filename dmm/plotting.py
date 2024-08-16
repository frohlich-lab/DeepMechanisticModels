from pathlib import Path
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import petab
from plotnine import *

PLOTNINE_THEME = {
    "dpi": 300,
    "legend_background": element_blank(),
    "legend_key": element_blank(),
    "panel_background": element_blank(),
    "strip_background": element_blank(),
    "strip_text": element_text(size=6),
    "axis_line": element_line(size=1),
}


def set_errorbar(mdf: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    if petab.NOISE_PARAMETERS in mdf.columns:
        errorbar = True
        mdf["ymax"] = mdf[petab.MEASUREMENT] + mdf[petab.NOISE_PARAMETERS]
        mdf["ymin"] = mdf[petab.MEASUREMENT] - mdf[petab.NOISE_PARAMETERS]
    else:
        errorbar = False
    return mdf, errorbar


def process_dataframes(
    mdf: pd.DataFrame, sdfs: Union[pd.DataFrame, List[pd.DataFrame]]
):
    if isinstance(sdfs, list):
        # sdfs is already a list of dataframes
        single_dataframe_flag = False
        pass
    elif isinstance(sdfs, pd.DataFrame):
        # sdfs is a single dataframe -> wrap in a list, but remember we will need to unpack
        sdfs = [sdfs]
        single_dataframe_flag = True
    else:
        raise TypeError(
            "sdfs must be a pd.DataFrame or a list of pd.DataFrames"
        )
    for df in [*sdfs, mdf]:
        df[petab.OBSERVABLE_ID] = df[petab.OBSERVABLE_ID].apply(
            lambda x: x.replace("_obs", "")
        )

        if (
            df.shape[0] > 0
        ):  # for avg and avg_model, avoids error at evaluation of test samples during train (empty df)
            if len(df[petab.SIMULATION_CONDITION_ID].iloc[0].split("__")) > 1:
                df[petab.SIMULATION_CONDITION_ID] = df[
                    petab.SIMULATION_CONDITION_ID
                ].apply(
                    lambda x: (
                        "" if x.split("__")[1].startswith("EGF") else "EGF+"
                    )
                    + x.split("__")[1]
                )
            else:
                df[petab.SIMULATION_CONDITION_ID] = df[
                    petab.SIMULATION_CONDITION_ID
                ].apply(lambda x: (x if x == "EGF" else "EGF+" + x))
        df.rename(
            columns={petab.SIMULATION_CONDITION_ID: "treatment"}, inplace=True
        )
    if single_dataframe_flag:
        return mdf, sdfs[0]
    else:
        return mdf, sdfs


def plot_single_sample(
    measurement_df: pd.DataFrame,
    simulation_df: pd.DataFrame,
    figdir: Path,
    sample: str,
    prefix: str,
):
    mdf = measurement_df.copy()
    sdf = simulation_df.copy()

    # Process dataframes and handle errorbar
    mdf, sdf = process_dataframes(mdf, sdf)
    mdf, errorbar = set_errorbar(mdf)

    kwargs = {"x": "time", "color": "treatment", "group": "treatment"}

    plot = (
        ggplot()
        + geom_line(
            data=sdf,
            mapping=aes(y=petab.SIMULATION, **kwargs),
            size=1,
        )
        + geom_point(
            data=mdf,
            mapping=aes(y=petab.MEASUREMENT, **kwargs),
            size=1,
        )
        + facet_grid(f"{petab.OBSERVABLE_ID} ~ treatment")
        + xlab("time [min]")
        + ylab("measurement")
        + ggtitle(f"cell line: {sample[1:]}")
        + theme(**PLOTNINE_THEME)
    )

    if errorbar:
        plot += geom_errorbar(
            data=mdf, mapping=aes(ymax="ymax", ymin="ymin", **kwargs)
        )

    save_plot(plot, figdir, prefix)


def plot_cross_samples(measurement_df, simulation_df, figdir, prefix):
    for sample in measurement_df[petab.PREEQUILIBRATION_CONDITION_ID].unique():
        print(f"plotting {sample} for {prefix}")
        plot_single_sample(
            measurement_df[
                measurement_df[petab.PREEQUILIBRATION_CONDITION_ID] == sample
            ],
            simulation_df[
                simulation_df[petab.PREEQUILIBRATION_CONDITION_ID] == sample
            ],
            figdir / sample,
            sample,
            prefix,
        )


def plot_single_sample_multiple_simulations(
    measurement_df: pd.DataFrame,
    simulation_dfs: List[pd.DataFrame],
    linetypes: List[str],
    linesizes: List[float],
    figdir: Path,
    sample: str,
    prefix: str,
):
    mdf = measurement_df.copy()
    sdfs = [simulation_df.copy() for simulation_df in simulation_dfs]

    # Process dataframes and handle errorbar
    mdf, sdfs = process_dataframes(mdf, sdfs)
    mdf, errorbar = set_errorbar(mdf)

    kwargs = {"x": "time", "color": "treatment", "group": "treatment"}

    # Measurement data
    plot = (
        ggplot()
        + geom_point(
            data=mdf,
            mapping=aes(y=petab.MEASUREMENT, **kwargs),
            size=1,
        )
        + facet_grid(rows=petab.OBSERVABLE_ID, cols="treatment")
        + xlab("time [min]")
        + ylab("measurement")
        + ggtitle(f"cell line: {sample[1:]}")
        + theme(**PLOTNINE_THEME)
    )

    if errorbar:
        plot += geom_errorbar(
            data=mdf, mapping=aes(ymax="ymax", ymin="ymin", **kwargs)
        )

    for sdf, (lt, ls) in zip(
        sdfs, zip(linetypes, linesizes)
    ):  # discriminate between models via linetype+linesize
        plot += geom_line(
            data=sdf,
            mapping=aes(y=petab.SIMULATION, **kwargs),
            linetype=lt,
            size=ls,
        )

    save_plot(plot, figdir, prefix)


def plot_cross_samples_multiple_simulations(
    measurement_df,
    simulation_dfs,
    labels,  # TODO use to distinguish profiles/curves!
    linetypes: List[str],
    linesizes: List[float],
    figdir,
    prefix,
):
    for sample in measurement_df[petab.PREEQUILIBRATION_CONDITION_ID].unique():
        print(f"plotting {sample} for {prefix}")
        plot_single_sample_multiple_simulations(
            measurement_df[
                measurement_df[petab.PREEQUILIBRATION_CONDITION_ID] == sample
            ],
            [
                simulation_df[
                    simulation_df[petab.PREEQUILIBRATION_CONDITION_ID]
                    == sample
                ]
                for simulation_df in simulation_dfs
            ],
            linetypes,
            linesizes,
            figdir / sample,
            sample,
            prefix,
        )


def save_plot(plot, figdir: Path, name: str, fmt: str = "pdf"):
    plt.tight_layout()
    figdir.mkdir(exist_ok=True, parents=True)
    plot.save(figdir / f"{name}.{fmt}", verbose=False)
