from pysb import Model

from cytof.pathways import add_egfr, add_inhibitors, add_mapk
from dmm.mechanistic_model import add_observables

model = Model("EGFR_MAPK_tegfr")

add_egfr(model)
add_mapk(model)

add_observables(model)
add_inhibitors(model)
