from pathlib import Path

import matplotlib.pyplot as plt
import pandas
import petab
from plotnine import *
from typing import List

PLOTNINE_THEME = {
    "dpi": 300,
    "legend_background": element_blank(),
    "legend_key": element_blank(),
    "panel_background": element_blank(),
    "strip_background": element_blank(),
    "strip_text": element_text(size=6),
    "axis_line": element_line(size=1),
}


def plot_single_sample(
    measurement_df: pandas.DataFrame,
    simulation_df: pandas.DataFrame,
    figdir: Path,
    sample: str,
    prefix: str,
):
    mdf = measurement_df.copy()
    sdf = simulation_df.copy()

    for df in [sdf, mdf]:
        df[petab.OBSERVABLE_ID] = df[petab.OBSERVABLE_ID].apply(
            lambda x: x.replace("_obs", "")
        )

        if df.shape[0] > 0:  # for avg and avg_model, avoids error at evaluation of test samples during train (empty df)
            if len(df[petab.SIMULATION_CONDITION_ID].iloc[0].split("__")) > 1:
                df[petab.SIMULATION_CONDITION_ID] = df[
                    petab.SIMULATION_CONDITION_ID
                ].apply(
                    lambda x: ("" if x.split("__")[1].startswith("EGF") else "EGF+")
                    + x.split("__")[1]
                )
            else:
                df[petab.SIMULATION_CONDITION_ID] = df[
                    petab.SIMULATION_CONDITION_ID
                ].apply(
                    lambda x: (x if x == "EGF" else "EGF+"+x)
                )
        df.rename(
            columns={petab.SIMULATION_CONDITION_ID: "treatment"}, inplace=True
        )

    if petab.NOISE_PARAMETERS in mdf.columns:
        errorbar = True
        mdf["ymax"] = mdf[petab.MEASUREMENT] + mdf[petab.NOISE_PARAMETERS]
        mdf["ymin"] = mdf[petab.MEASUREMENT] - mdf[petab.NOISE_PARAMETERS]
    else:
        errorbar = False

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
            + facet_grid((petab.OBSERVABLE_ID, "treatment"))
            + xlab("time [min]")
            + ylab("measurement")
            + ggtitle(f"cell line: {sample[1:]}")
            + theme(**PLOTNINE_THEME)
    )

    if errorbar:
        plot += geom_errorbar(
            data=mdf,
            mapping=aes(ymax="ymax", ymin="ymin", **kwargs)
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
    measurement_df: pandas.DataFrame,
    simulation_dfs: List[pandas.DataFrame],
    linetypes: List[str],
    linesizes: List[float],
    figdir: Path,
    sample: str,
    prefix: str,
):
    mdf = measurement_df.copy()
    sdfs = [simulation_df.copy() for simulation_df in simulation_dfs]

    for df in [*sdfs, mdf]:
        df[petab.OBSERVABLE_ID] = df[petab.OBSERVABLE_ID].apply(
            lambda x: x.replace("_obs", "")
        )

        if df.shape[0] > 0:  # for avg and avg_model, avoids error at evaluation of test samples during train (empty df)
            if len(df[petab.SIMULATION_CONDITION_ID].iloc[0].split("__")) > 1:
                df[petab.SIMULATION_CONDITION_ID] = df[
                    petab.SIMULATION_CONDITION_ID
                ].apply(
                    lambda x: ("" if x.split("__")[1].startswith("EGF") else "EGF+")
                    + x.split("__")[1]
                )
            else:
                df[petab.SIMULATION_CONDITION_ID] = df[
                    petab.SIMULATION_CONDITION_ID
                ].apply(
                    lambda x: (x if x == "EGF" else "EGF+"+x)
                )
        df.rename(
            columns={petab.SIMULATION_CONDITION_ID: "treatment"}, inplace=True
        )

    if petab.NOISE_PARAMETERS in mdf.columns:
        errorbar = True
        mdf["ymax"] = mdf[petab.MEASUREMENT] + mdf[petab.NOISE_PARAMETERS]
        mdf["ymin"] = mdf[petab.MEASUREMENT] - mdf[petab.NOISE_PARAMETERS]
    else:
        errorbar = False

    kwargs = {"x": "time", "color": "treatment", "group": "treatment"}

    plot = (
            ggplot()
            + geom_point(
                data=mdf,
                mapping=aes(y=petab.MEASUREMENT, **kwargs),
                size=1,
            )
            + facet_grid((petab.OBSERVABLE_ID, "treatment"))
            + xlab("time [min]")
            + ylab("measurement")
            + ggtitle(f"cell line: {sample[1:]}")
            + theme(**PLOTNINE_THEME)
    )

    if errorbar:
        plot += geom_errorbar(
            data=mdf,
            mapping=aes(ymax="ymax", ymin="ymin", **kwargs)
        )

    for sdf, (lt, ls) in zip(sdfs, zip(linetypes, linesizes)):  # discriminate between models via linetype+linesize
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
        linetypes: List[str],
        linesizes: List[float],
        figdir,
        prefix
):
    for sample in measurement_df[petab.PREEQUILIBRATION_CONDITION_ID].unique():
        print(f"plotting {sample} for {prefix}")
        plot_single_sample_multiple_simulations(
            measurement_df[
                measurement_df[petab.PREEQUILIBRATION_CONDITION_ID] == sample
            ],
            [simulation_df[
                 simulation_df[petab.PREEQUILIBRATION_CONDITION_ID] == sample
                 ] for simulation_df in simulation_dfs],
            linetypes,
            linesizes,
            figdir / sample,
            sample,
            prefix,
        )


def save_plot(plot, figdir: Path, name: str, fmt: str = "pdf"):
    plt.tight_layout()
    figdir.mkdir(exist_ok=True, parents=True)
    plot.save(figdir / f"{name}.{fmt}")
