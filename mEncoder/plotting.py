import pandas
from plotnine import *
import matplotlib.pyplot as plt
import petab
from pathlib import Path

PLOTNINE_THEME = {
    'dpi': 300,
    'legend_background': element_blank(),
    'legend_key': element_blank(),
    'panel_background': element_blank(),
    'strip_background': element_blank(),
    'strip_text': element_text(size=6),
    'axis_line': element_line(size=1),
}


def plot_single_sample(
    measurement_df: pandas.DataFrame,
    simulation_df: pandas.DataFrame,
    figdir: Path,
    prefix: str
):
    mdf = measurement_df.copy()
    sdf = simulation_df.copy()

    for df in [sdf, mdf]:
        df[petab.OBSERVABLE_ID] = df[petab.OBSERVABLE_ID].apply(
            lambda x: x.replace('_obs', '')
        )
        df[petab.SIMULATION_CONDITION_ID] = \
            df[petab.SIMULATION_CONDITION_ID].apply(
                lambda x:
                    ('' if x.startswith('EGF_') else 'EGF+')
                    + x.split('__')[1]
            )
        df.rename(columns={
            petab.SIMULATION_CONDITION_ID: 'treatment'
        }, inplace=True)

    mdf['ymax'] = \
        mdf[petab.MEASUREMENT] + mdf[petab.NOISE_PARAMETERS]
    mdf['ymin'] = \
        mdf[petab.MEASUREMENT] - mdf[petab.NOISE_PARAMETERS]

    cell_line = df.loc[0, petab.PREEQUILIBRATION_CONDITION_ID][1:]

    kwargs = {
        'x': 'time',
        'color': 'treatment',
        'group': 'treatment'
    }
    plot = (
        ggplot() +
        geom_line(
            data=sdf,
            mapping=aes(y=petab.SIMULATION, **kwargs),
            size=1,
        )
        + geom_point(
            data=mdf,
            mapping=aes(y=petab.MEASUREMENT, **kwargs),
            size=1,
        )
        + geom_errorbar(
            data=mdf,
            mapping=aes(ymax='ymax', ymin='ymin', **kwargs)
        )
        + facet_grid((petab.OBSERVABLE_ID, 'treatment'))
        + xlab('time [min]')
        + ylab('measurement')
        + ggtitle(f'cell line: {cell_line}')
        + theme(**PLOTNINE_THEME)
    )

    save_plot(plot, figdir, prefix)


def plot_cross_samples(measurement_df, simulation_df, figdir, prefix):

    for sample in measurement_df[petab.PREEQUILIBRATION_CONDITION_ID].unique():
        plot_single_sample(
            measurement_df[
                measurement_df[petab.PREEQUILIBRATION_CONDITION_ID]
                == sample
            ],
            simulation_df[
                simulation_df[petab.PREEQUILIBRATION_CONDITION_ID] == sample
            ],
            figdir / sample,
            prefix
        )


def save_plot(plot, figdir: Path, name: str, fmt: str = 'pdf'):
    plt.tight_layout()
    figdir.mkdir(exist_ok=True, parents=True)
    plot.save(figdir / f'{name}.{fmt}')
