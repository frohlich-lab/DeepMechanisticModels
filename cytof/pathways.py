from pysb import Monomer, Parameter, Rule

from dmm.mechanistic_model import (
    add_inhibitor,
    add_parameter,
    generate_pathway,
)

active_rtks = [
    "EGFR__Y1173_p",
    "ERBB2__Y1248_p",
]
egfr_feedback = ["ERK__Y204_p"]
active_akt = ["AKT__T308_p"]


def add_egfr(model):
    Parameter("EGF_0")
    Parameter("TGFA_eq")
    Parameter("BTC_eq")
    Parameter("EREG_eq")
    Parameter("NRG1_eq")
    Parameter("NRG2_eq")

    EGFR = Monomer(
        "EGFR",
        ["Y1173", "compartment"],
        {"compartment": ["pm", "e"], "Y1173": ["u", "p"]},
    )

    erbb_cascade = [
        ("ERBB2", {"Y1248": ["EGFR__Y1173_p", "NRG1_eq", "NRG2_eq"]}),
        ("EGFR", {"Y1173": ["EGF_0", "TGFA_eq", "BTC_eq", "EREG_eq"]}),
    ]
    generate_pathway(
        model,
        erbb_cascade,
        add_baseline_activation=["EGFR", "ERBB2"],
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
    Parameter("m_KRAS")
    Parameter("m_BRAF")

    mapk_cascade = [
        ("MEK", {"S222": (active_rtks + ["m_KRAS", "m_BRAF"], egfr_feedback)}),
        ("ERK", {"Y204": ["MEK__S222_p", *active_rtks]}),
        # egfr can activate via p38
        ("RPS6KA1", {"S380": ["ERK__Y204_p", *active_rtks]}),  # p90RSK
    ]
    generate_pathway(
        model, mapk_cascade, add_baseline_activation=["MEK", "RPS6KA1"]
    )


def add_mtore_akt(model):
    # add_monomer_synth_deg(
    #     "MTOR", asites=["C"], asite_states=["c0", "c1", "c2"]
    # )

    # AKT
    akt_cascade = [
        ("PIK3CA", {"pip2": (active_rtks, egfr_feedback)}),
        ("PDPK1", {"S241": ["PIK3CA__pip2_p"]}),
        (
            "AKT",
            {
                "T308": ["PDPK1__S241_p"],
                # "S473": ["MTOR__C_c2", "AKT1__T308_p"],
            },
        ),
    ]
    generate_pathway(model, akt_cascade)

    # add_activation(
    #     model,
    #     "MTOR",
    #     "C",
    #     "activation",
    #     active_akt,
    #     [],
    #     site_states=["c0", "c2"],
    # )
    #
    # add_activation(
    #     model,
    #     "MTOR",
    #     "C",
    #     "activation",
    #     ["RPS6KA1__S380_p"],
    #     [],
    #     site_states=["c0", "c1"],
    # )


def add_stat(model):
    stat_cascade = [
        ("SRC", {"Y419": active_rtks}),
        ("STAT1", {"Y727": ["ERK__Y204_p"]}),
        ("STAT3", {"Y705": ["SRC__Y419_p", "ERK__Y204_p", "EGFR__Y1173_p"]}),
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
        (
            "EIF4EBP1",
            {"T37_T46": ["MTOR__C_c1", "GSK3B__S9_p", "ERK__Y204_p"]},
        ),
        (
            "CREB1",
            {"S133": ["AKT__T308_p", "RPS6KA1__S380_p"]},
        ),
    ]
    generate_pathway(model, EIF4_cascade)


def add_inhibitors(model):
    add_inhibitor(model, "iMEK", ["MEK__S222_p_obs"])
    add_inhibitor(model, "iEGFR", ["EGFR__Y1173_p_obs", "ERBB2__Y1248_p_obs"])
    add_inhibitor(model, "iPI3K", ["PIK3CA__pip2_p_obs"])
    add_inhibitor(model, "iPKC", ["PKC"])
