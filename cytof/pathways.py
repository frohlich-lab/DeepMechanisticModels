from pysb import Monomer, Observable, Parameter, Rule

from dmm.mechanistic_model import (
    add_activation,
    add_inhibitor,
    add_monomer_synth_deg,
    add_parameter,
    generate_pathway,
)

active_rtks = [
    "EGFR__Y1173_p",
    # 'ERBB2__Y1248_p' # TODO: reactivate this? also add turnover for HER2?
]
# active_erk = ['MAPK1__T185_p__Y187_p', 'MAPK3__T202_p__Y204_p']
active_erk = ["ERK__Y204_p"]
active_akt = ["AKT1__T308_p", "AKT2__T309_p", "AKT3__T305_p"]


def add_egfr(model):
    Parameter("EGF_0")
    EGFR = Monomer(
        "EGFR",
        ["Y1173", "compartment"],
        {"compartment": ["pm", "e"], "Y1173": ["u", "p"]},
    )

    erbb_cascade = [
        ("ERBB2", {"Y1248": ["EGFR__Y1173_p"]}),
        ("EGFR", {"Y1173": ["EGF_0"]}),
    ]
    generate_pathway(
        model,
        erbb_cascade,
        add_baseline_activation="first",
        species_with_synth="EGFR",
    )
    Rule(
        "EGFR_degradation",
        EGFR(compartment="e") >> None,
        add_parameter("EGFR_degradation_kcat", model),
    )
    Rule(
        "EGFR_endocytosis",
        EGFR(Y1173="p", compartment="pm") >> EGFR(Y1173="p", compartment="e"),
        add_parameter("EGFR_endocytosis_kcat", model),
    )


def add_mapk(model):
    if "her2" in model.name.split("_"):
        active_rtks.append("ERBB2__Y1248_p")

    mapk_cascade = [
        ("MEK", {"S222": (active_rtks, active_erk)}),
        ("ERK", {"Y204": ["MEK__S222_p", "EGFR__Y1173_p"]}),
        # ('RPS6KA1', {'S380': active_erk})  # p90RSK
    ]
    generate_pathway(model, mapk_cascade, add_baseline_activation="first")


def add_mtore_akt(model):
    add_monomer_synth_deg(
        "MTOR", asites=["C"], asite_states=["c0", "c1", "c2"]
    )

    # AKT
    akt_cascade = [
        ("PIK3CA", {"pip2": (active_rtks, active_erk)}),
        ("PDPK1", {"S241": ["PIK3CA__pip2_p"]}),
        (
            "AKT1",
            {
                "T308": ["PDPK1__S241_p"],
                "S473": ["MTOR__C_c2", "AKT1__T308_p"],
            },
        ),
        (
            "AKT2",
            {
                "T309": ["PDPK1__S241_p"],
                "S473": ["MTOR__C_c2", "AKT2__T309_p"],
            },
        ),
        (
            "AKT3",
            {
                "T305": ["PDPK1__S241_p"],
                "S473": ["MTOR__C_c2", "AKT3__T305_p"],
            },
        ),
    ]
    generate_pathway(model, akt_cascade)

    add_activation(
        model,
        "MTOR",
        "C",
        "activation",
        active_akt,
        [],
        site_states=["c0", "c2"],
    )

    add_activation(
        model,
        "MTOR",
        "C",
        "activation",
        ["RPS6KA1__S380_p"],
        [],
        site_states=["c0", "c1"],
    )

    Observable(
        "pAKT_S473",
        model.monomers["AKT1"](S473="p")
        + model.monomers["AKT2"](S473="p")
        + model.monomers["AKT3"](S473="p"),
    )

    Observable(
        "pAKT_T308",
        model.monomers["AKT1"](T308="p")
        + model.monomers["AKT2"](T309="p")
        + model.monomers["AKT3"](T305="p"),
    )


def add_stat(model):
    stat_cascade = [
        ("SRC", {"Y419": active_rtks}),
        ("STAT1", {"Y727": active_erk}),
        ("STAT3", {"Y705": ["SRC__Y419_p", *active_erk, "EGFR__Y1173_p"]}),
        ("STAT5A", {"Y694": ["SRC__Y419_p", "EGFR__Y1173_p"]}),
        ("BTK", {"Y551": ["SRC__Y419_p"]}),
        ("PLCG2", {"Y759": ["EGFR__Y1173_p", "SRC__Y419_p", "BTK__Y551_p"]}),
    ]
    generate_pathway(model, stat_cascade)


def add_s6(model):
    # S6
    s6_cascade = [
        (
            "RPS6KB1",
            {"S412": ["MTOR__C_c1"], "T252": ["PDPK1__S241_p"]},
        ),  # p70S6K
        (
            "RPS6",
            {"S235_S236": ["RPS6KA1__S380_p", "RPS6KB1__S412_p__T252_p"]},
        ),  # S6
    ]
    generate_pathway(model, s6_cascade)

    # GSK
    gsk_cascade = [
        ("GSK3B", {"S9": [*active_akt, "RPS6KA1__S380_p", "RPS6KB1__S412_p"]})
    ]
    generate_pathway(model, gsk_cascade)

    # TFs
    EIF4_cascade = [
        ("EIF4EBP1", {"T37_T46": ["MTOR__C_c1", "GSK3B__S9_p", *active_erk]}),
        (
            "CREB1",
            {"S133": ["AKT1__T308_p", "AKT2__T309_p", "RPS6KA1__S380_p"]},
        ),
    ]
    generate_pathway(model, EIF4_cascade)


def add_inhibitors(model):
    add_inhibitor(model, "iMEK", ["MEK__S222_p_obs"])
    add_inhibitor(model, "iEGFR", ["EGFR__Y1173_p_obs"])
    add_inhibitor(model, "iPI3K", ["PIK3CA"])
    add_inhibitor(model, "iPKC", ["PKC"])
