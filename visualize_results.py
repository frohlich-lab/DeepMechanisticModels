import sys
import os
import pypesto
import amici.petab_objective
import numpy as np
import aesara
import matplotlib.pyplot as plt

from mEncoder.autoencoder import MechanisticAutoEncoder
from mEncoder.training import create_pypesto_problem
from mEncoder import (
    plot_and_save_fig, results_dir, basedir, COLLECTED_ESTIMATION_OUTFILE_TEMP
)
from mEncoder.plotting import plot_cross_samples

from process_data import training_samples, test_samples, Wildcards

from pypesto.visualize import waterfall, optimizer_convergence
from pypesto.store import OptimizationResultHDF5Reader
from pypesto.C import FVAL

MODEL = sys.argv[1]
DATA = sys.argv[2]
SAMPLES = sys.argv[3]
N_HIDDEN = int(sys.argv[4])
ALPHA = float(sys.argv[5])

N_STARTS = 5

result_path = os.path.join(results_dir, MODEL, DATA)
outfile = os.path.join(result_path, COLLECTED_ESTIMATION_OUTFILE_TEMP.format(
    samples=SAMPLES, n_hidden=N_HIDDEN, alpha=ALPHA
))

mae = MechanisticAutoEncoder(
    N_HIDDEN, (
        os.path.join('data', f'{DATA}__{MODEL}__measurements.tsv'),
        os.path.join('data', f'{DATA}__{MODEL}__conditions.tsv'),
        os.path.join('data', f'{DATA}__{MODEL}__observables.tsv'),
    ),
    pathway_name=MODEL, samples=training_samples(Wildcards(DATA, SAMPLES)),
    par_modulation_scale=ALPHA
)
problem = create_pypesto_problem(mae)

reader = OptimizationResultHDF5Reader(outfile)
result = pypesto.Result(problem)
result.optimize_result = reader.read().optimize_result

print(result.optimize_result.fval)

figdir = os.path.join(basedir, 'figures', MODEL, DATA)
os.makedirs(figdir, exist_ok=True)
output_prefix = '__'.join([SAMPLES, str(N_HIDDEN), str(ALPHA)])

waterfall(result)
plot_and_save_fig(os.path.join(figdir, output_prefix + '__waterfall.pdf'))

optimizer_convergence(result)
plot_and_save_fig(os.path.join(figdir,
                               output_prefix + '__optimizer_convergence.pdf'))

x = problem.get_reduced_vector(result.optimize_result.list[0]['x'],
                               problem.x_free_indices)
simulation = problem.objective(x, return_dict=True)


mae_prediction = MechanisticAutoEncoder(
    N_HIDDEN,
    (
        os.path.join('data', f'{DATA}__{MODEL}__measurements.tsv'),
        os.path.join('data', f'{DATA}__{MODEL}__conditions.tsv'),
        os.path.join('data', f'{DATA}__{MODEL}__observables.tsv'),
    ), MODEL, test_samples(Wildcards(DATA, SAMPLES)),  features=mae.features,
    imputer=mae.imputer, scaler=mae.scaler
)
prediction_problem = create_pypesto_problem(mae_prediction)
prediction = prediction_problem.objective(x, return_dict=True)


importer = mae.petab_importer
importer_prediction = mae_prediction.petab_importer
model = importer.create_model()
model_prediction = importer_prediction.create_model()

# Convert the simulation to PEtab format.
if np.isfinite(simulation[FVAL]):
    simulation_df = amici.petab_objective.rdatas_to_simulation_df(
        simulation['rdatas'],
        model=model,
        measurement_df=importer.petab_problem.measurement_df,
    )

    prediction_df = amici.petab_objective.rdatas_to_simulation_df(
        prediction['rdatas'],
        model=model_prediction,
        measurement_df=importer_prediction.petab_problem.measurement_df,
    )

    # Plot fit
    plot_cross_samples(importer.petab_problem.measurement_df,
                       simulation_df,
                       os.path.join(figdir, 'training'),
                       output_prefix)

    # Plot fit
    plot_cross_samples(importer_prediction.petab_problem.measurement_df,
                       prediction_df,
                       os.path.join(figdir, 'prediction'),
                       output_prefix)

embedding_fun = aesara.function(
    [mae.x], mae.embedding_fun
)

embedding = embedding_fun(result.optimize_result.list[0]['x'])


fig_embedding, axes_embedding = plt.subplots(1, 1,
                                             figsize=(18.5, 10.5))

axes_embedding.plot(embedding[:, 0], embedding[:, 1], 'bo')
axes_embedding.plot(mae.data_pca[:, 0], mae.data_pca[:, 1], 'bx')
#axes_embedding.arrow(x=embedding[:, 0], y=embedding[:, 1],
#                     dx=mae.data_pca[:, 0]-embedding[:, 0],
#                     dy=mae.data_pca[:, 1]-embedding[:, 1],
#                     color='r')

plot_and_save_fig(output_prefix + '__embedding.pdf',
                  figdir=figdir)
