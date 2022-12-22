from pysb import Model

from mEncoder.mechanistic_model import add_observables
from cytof.pathways import add_egfr, add_mapk, add_mtore_akt, add_inhibitors

model = Model("EGFR_MAPK_AKT")

add_egfr(model)
add_mapk(model)
add_mtore_akt(model)

add_observables(model)
add_inhibitors(model)
