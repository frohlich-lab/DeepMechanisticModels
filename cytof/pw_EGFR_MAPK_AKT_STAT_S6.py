from pysb import Model

from cytof.pathways import (
    add_egfr,
    add_mapk,
    add_mtore_akt,
    add_s6,
    add_stat,
)
from dmm.mechanistic_model import add_observables

model = Model("EGFR_MAPK_AKT_STAT_S6")

add_egfr(model)
add_mapk(model)
add_mtore_akt(model)
add_s6(model)
add_stat(model)

add_observables(model)
