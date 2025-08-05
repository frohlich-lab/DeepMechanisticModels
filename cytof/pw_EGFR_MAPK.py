from pysb import Model

from cytof.pathways import add_egfr, add_mapk
from dmm.mechanistic_model import add_observables

model = Model("EGFR_MAPK")

add_egfr(model)
add_mapk(model)

add_observables(model)
