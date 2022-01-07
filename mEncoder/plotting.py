from plotnine import *
import matplotlib.pyplot as plt
import petab
import os

PLOTNINE_THEME = {
    'dpi': 300,
    'legend_background': element_blank(),
    'legend_key': element_blank(),
    'panel_background': element_blank(),
    'strip_background': element_blank(),
    'strip_text': element_text(size=6),
    'axis_line': element_line(size=1),
}


def plot_single_sample(measurement_df, simulation_df, figdir, prefix):
    measurement_df = measurement_df.copy()
    simulation_df = simulation_df.copy()

    dyn_sim = simulation_df[
        simulation_df[petab.PREEQUILIBRATION_CONDITION_ID] !=
        simulation_df[petab.SIMULATION_CONDITION_ID]
    ]
    dyn_mes = measurement_df[
        measurement_df[petab.PREEQUILIBRATION_CONDITION_ID] !=
        measurement_df[petab.SIMULATION_CONDITION_ID]
    ]

    kwargs = {
        'x': 'time',
        'color': petab.SIMULATION_CONDITION_ID,
        'group': petab.SIMULATION_CONDITION_ID
    }
    plot = (
        ggplot() +
        geom_line(
            data=dyn_sim,
            mapping=aes(y=petab.SIMULATION, **kwargs),
            size=1,
        )
        + geom_point(
            data=dyn_mes,
            mapping=aes(y=petab.MEASUREMENT, **kwargs),
            size=1,
        )
        + facet_wrap(petab.OBSERVABLE_ID, scales='free_y')
        + xlab('time [min]')
        + ylab('measurement')
        + theme(**PLOTNINE_THEME)
    )

    save_plot(plot, os.path.join(figdir, 'fit_dynamic'), prefix)

    simulation_df = simulation_df.sort_values([
        petab.PREEQUILIBRATION_CONDITION_ID, petab.TIME, petab.OBSERVABLE_ID,
        petab.SIMULATION_CONDITION_ID
    ]).reset_index()
    measurement_df = measurement_df.sort_values([
        petab.PREEQUILIBRATION_CONDITION_ID, petab.TIME, petab.OBSERVABLE_ID,
        petab.SIMULATION_CONDITION_ID
    ]).reset_index()
    simulation_df[petab.MEASUREMENT] = measurement_df[petab.MEASUREMENT]
    stat = simulation_df[
        simulation_df[petab.PREEQUILIBRATION_CONDITION_ID] ==
        simulation_df[petab.SIMULATION_CONDITION_ID]
    ]

    if len(stat):
        plot = (
                ggplot(data=stat,
                       mapping=aes(y=petab.MEASUREMENT, x=petab.SIMULATION,
                                   color=petab.PREEQUILIBRATION_CONDITION_ID,
                                   group=petab.PREEQUILIBRATION_CONDITION_ID))
                + geom_point(size=1)
                + facet_wrap(petab.OBSERVABLE_ID,
                             scales='free_y')
                + xlab('simulation')
                + ylab('measurement')
                + theme(**PLOTNINE_THEME)
        )

        save_plot(plot, os.path.join(figdir, 'fit_static'), prefix)


def plot_cross_samples(measurement_df, simulation_df, figdir, prefix):
    mdf = measurement_df.copy()
    sdf = simulation_df.copy()

    for df in [sdf, mdf]:
        df[petab.OBSERVABLE_ID] = df[petab.OBSERVABLE_ID].apply(
            lambda x: x.replace('_obs', '')
        )

    sdf = sdf.sort_values([
        petab.PREEQUILIBRATION_CONDITION_ID, petab.TIME, petab.OBSERVABLE_ID,
        petab.SIMULATION_CONDITION_ID
    ]).reset_index()
    mdf = mdf.sort_values([
        petab.PREEQUILIBRATION_CONDITION_ID, petab.TIME, petab.OBSERVABLE_ID,
        petab.SIMULATION_CONDITION_ID
    ]).reset_index()

    sdf[petab.MEASUREMENT] = mdf[petab.MEASUREMENT]

    dyn = sdf[
        sdf[petab.PREEQUILIBRATION_CONDITION_ID] !=
        sdf[petab.SIMULATION_CONDITION_ID]
        ].copy()

    dyn[petab.SIMULATION_CONDITION_ID] = \
        dyn[petab.SIMULATION_CONDITION_ID].apply(
            lambda x: x.split('__')[1]
        )

    plot = (
        ggplot(data=dyn,
               mapping=aes(y=petab.MEASUREMENT, x=petab.SIMULATION,
                           color=petab.TIME,
                           group=petab.TIME))
        + geom_point(size=1, alpha=0.2, shape='.')
        + facet_grid((petab.SIMULATION_CONDITION_ID, petab.OBSERVABLE_ID))
        + xlab('simulation')
        + ylab('measurement')
        + theme(**PLOTNINE_THEME)
    )

    save_plot(plot, os.path.join(figdir, 'fit_dynamic'), prefix)

    res = dyn.copy()
    res['error'] = res[petab.MEASUREMENT] - res[petab.SIMULATION]

    for var in [petab.TIME, petab.PREEQUILIBRATION_CONDITION_ID]:

        plot = (
            ggplot(data=res,
                   mapping=aes(x='error',
                               color=var,
                               group=var))
            + stat_density()
            + facet_grid((petab.SIMULATION_CONDITION_ID, petab.OBSERVABLE_ID),
                         scales='free_x')
            + xlab('simulation')
            + ylab('measurement')
            + theme(**PLOTNINE_THEME)
        )
        save_plot(plot, os.path.join(figdir, f'res_{var}'), prefix)

    stat = sdf[
        sdf[petab.PREEQUILIBRATION_CONDITION_ID] ==
        sdf[petab.SIMULATION_CONDITION_ID]
        ].copy()

    if len(stat):
        plot = (
                ggplot(data=stat,
                       mapping=aes(y=petab.MEASUREMENT, x=petab.SIMULATION,
                                   color=petab.PREEQUILIBRATION_CONDITION_ID,
                                   group=petab.PREEQUILIBRATION_CONDITION_ID))
                + geom_point(size=1)
                + facet_wrap(petab.OBSERVABLE_ID)
                + xlab('simulation')
                + ylab('measurement')
                + theme(**PLOTNINE_THEME)
        )

        save_plot(plot, os.path.join(figdir, 'fit_static'), prefix)

    for sample in sdf[petab.PREEQUILIBRATION_CONDITION_ID].unique():
        plot_single_sample(
            measurement_df[
                measurement_df[petab.PREEQUILIBRATION_CONDITION_ID]
                == sample
            ],
            simulation_df[
                simulation_df[petab.PREEQUILIBRATION_CONDITION_ID] == sample
            ],
            os.path.join(figdir, sample),
            prefix
        )


def save_plot(plot, dir, name, fmt='pdf'):
    plt.tight_layout()
    os.makedirs(dir, exist_ok=True)
    plot.save(os.path.join(dir, f'{name}.{fmt}'))
