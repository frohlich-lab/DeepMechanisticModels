active_rtks = [
    "EGFR__Y1173_p",
    "ERBB2__Y1248_p",
]
rtk_feedback = ["ERK__Y204_p"]
active_akt = ["AKT__T308_p"]


def add_egfr(model):
    erbb2_activators = ["ERBB2", "EGFR__Y1173_p"]
    for gf in ["NRG1", "NRG2"]:
        if model.has_modification(f"t{gf.lower}"):
            par = f"{gf}_eq"
            erbb2_activators.append(f"{gf}_eq")
            model.parameters.add(par)
    model.pathway_elements["ERBB2"] = {
        "Y1248": (
            erbb2_activators,
            ["iEGFR_0"],
        ),
    }
    egfr_activators = ["EGF_0"]
    model.parameters.add("EGF_0")
    for gf in ["TGFA", "BTC", "EREG"]:
        if model.has_modification(f"t{gf.lower}"):
            par = f"{gf}_eq"
            egfr_activators.append(f"{gf}_eq")
            model.parameters.addt(par)
    model.pathway_elements["EGFR"] = {
        "Y1173": (
            egfr_activators,
            ["iEGFR_0"],
        ),
        "compartment": (
            ["EGF_0", "EGFR__Y1173_p", "ERK__Y204_p"],
            ["EGFR", "ERBB2"],
        ),
        "degradation": [],
    }
    model.species_with_synth.append("EGFR")
    model.delays["EGFR_compartment"] = 3
    model.require_phosphorylation["EGFR_compartment"] = "Y1173"
    model.require_compartment["EGFR_Y1173"] = "pm"
    model.require_compartment["EGFR_degradation"] = "e"
    if model.has_modification("fegfr"):
        model.species_with_free_levels.append("EGFR")
    if model.has_modification("ferbb2"):
        model.species_with_free_levels.append("ERBB2")


def add_mapk(model):
    mek_activators = [*active_rtks]
    for mut in ["KRAS", "BRAF"]:
        if model.has_modification(f"m{mut.lower()}"):
            par = f"m_{mut}"
            model.parameters.add(par)
            mek_activators.append(par)
    model.pathway_elements["MEK"] = {"S222": (mek_activators, rtk_feedback)}
    model.pathway_elements["ERK"] = {
        "Y204": (["MEK__S222_p", *active_rtks], ["iMEK_0"])
    }
    model.pathway_elements["RPS6KA1"] = {"S380": ["ERK__Y204_p", *active_rtks]}
    model.delays["MEK_S222"] = 3


def add_mtor_akt(model):
    # AKT
    model.pathway_elements["PDPK1"] = {
        "S241": (active_rtks, ["iPI3K", *rtk_feedback])
    }
    model.pathway_elements["AKT"] = {
        "T308": ["PDPK1__S241_p"],
    }

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
    pass
    # stat_cascade = [
    #     ("SRC", {"Y419": active_rtks}),
    #     ("STAT1", {"Y727": ["ERK__Y204_p"]}),
    #     ("STAT3", {"Y705": ["SRC__Y419_p", "ERK__Y204_p", "EGFR__Y1173_p"]}),
    #     ("STAT5A", {"Y694": ["SRC__Y419_p", "EGFR__Y1173_p"]}),
    #     ("BTK", {"Y551": ["SRC__Y419_p"]}),
    #     ("PLCG2", {"Y759": ["EGFR__Y1173_p", "SRC__Y419_p", "BTK__Y551_p"]}),
    # ]
    # generate_pathway(model, stat_cascade)


def add_s6(model):
    pass
    # # S6
    # s6_cascade = [
    #     (
    #         "RPS6KB1",
    #         {"S412": ["MTOR__C_c1"], "T252": ["PDPK1__S241_p"]},
    #     ),  # p70S6K
    #     (
    #         "RPS6",
    #         {"S235_S236": ["RPS6KA1__S380_p", "RPS6KB1__S412_p__T252_p"]},
    #     ),  # S6
    # ]
    # generate_pathway(model, s6_cascade)
    #
    # # GSK
    # gsk_cascade = [
    #     ("GSK3B", {"S9": [*active_akt, "RPS6KA1__S380_p", "RPS6KB1__S412_p"]})
    # ]
    # generate_pathway(model, gsk_cascade)
    #
    # # TFs
    # EIF4_cascade = [
    #     (
    #         "EIF4EBP1",
    #         {"T37_T46": ["MTOR__C_c1", "GSK3B__S9_p", "ERK__Y204_p"]},
    #     ),
    #     (
    #         "CREB1",
    #         {"S133": ["AKT__T308_p", "RPS6KA1__S380_p"]},
    #     ),
    # ]
    # generate_pathway(model, EIF4_cascade)
