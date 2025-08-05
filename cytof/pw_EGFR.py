from pysb import Model

from cytof.pathways import add_egfr
from dmm.mechanistic_model import add_observables

model = Model("EGFR")

add_egfr(model)

add_observables(model)
