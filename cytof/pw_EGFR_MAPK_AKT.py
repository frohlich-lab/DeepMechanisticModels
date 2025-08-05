from pysb import Model

from cytof.pathways import add_egfr, add_mapk, add_mtore_akt
from dmm.mechanistic_model import add_observables

model = Model("EGFR_MAPK_AKT")

add_egfr(model)
add_mapk(model)
add_mtore_akt(model)

add_observables(model)
