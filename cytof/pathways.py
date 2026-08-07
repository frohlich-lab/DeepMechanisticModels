active_rtks = (
    "EGFR__Y1173_p",
    "ERBB2__Y1248_p",
    "ERBB3__Y1289_p",
)
rtk_feedback = ("ERK__Y204_p",)


def add_egfr(model):
    erbb3_activators = ["serum_0"]
    model.parameters.add("serum_0")
    for gf in ["NRG1", "NRG2"]:
        if model.has_modification(f"t{gf.lower()}"):
            par = f"{gf}_eq"
            erbb3_activators.append(f"{gf}_eq")
            model.parameters.add(par)
    model.pathway_elements["ERBB3"] = {
        "Y1289": (
            erbb3_activators,
            [],
        ),
    }
    model.pathway_elements["ERBB2"] = {
        "Y1248": (
            ["ERBB2", "EGFR__Y1173_p", "ERBB3__Y1289_p", "serum_0"],
            ["iEGFR_0"],
        ),
    }
    egfr_activators = ["EGF_0", "serum_0"]
    model.parameters.add("EGF_0")
    for gf in ["TGFA", "BTC", "EREG"]:
        if model.has_modification(f"t{gf.lower()}"):
            par = f"{gf}_eq"
            egfr_activators.append(f"{gf}_eq")
            model.parameters.add(par)
    model.pathway_elements["EGFR"] = {
        "Y1173": (
            egfr_activators,
            ["iEGFR_0"],
        ),
        "endocytosis": (
            ["ERK__Y204_p"],
            ["EGFR", "ERBB2"],
        ),
        "degradation": ["EGF_0"],
    }
    model.species_with_synth.append("EGFR")
    model.delays["EGFR_endocytosis"] = 3
    model.require_phosphorylation["EGFR_endocytosis"] = "Y1173"
    model.require_compartment["EGFR_Y1173"] = "pm"
    model.require_compartment["EGFR_degradation"] = "e"
    if model.has_modification("fegfr"):
        model.species_with_free_levels.append("EGFR")
    if model.has_modification("ferbb2"):
        model.species_with_free_levels.append("ERBB2")
    if model.has_modification("ferbb3"):
        model.species_with_free_levels.append("ERBB3")


def add_mapk(model):
    mek_activators = list(active_rtks)
    for mut in ["KRAS", "BRAF"]:
        if model.has_modification(f"m{mut.lower()}"):
            par = f"m_{mut}"
            model.parameters.add(par)
            mek_activators.append(par)

    # iMEK can inhibit activation by m_BRAF
    model.pathway_elements["MEK"] = {
        "S222": (mek_activators, list(rtk_feedback) + ["iMEK_0"])
    }
    # not clear whether this is really TOPK, see here: https://pmc.ncbi.nlm.nih.gov/articles/PMC2893265/
    # dream challenge pub mentions this happens in 5 cell lines
    # other cell lines are probably:
    # HCC2185, CAL148, HCC1187, MDAMB157, T47D
    # more complex: HCC38, MDAMB415

    # interesting tidbit: T47D has constitutively high levels of pH3,
    # which is putatively phosphorylated by TOPK (https://pubmed.ncbi.nlm.nih.gov/35115926/)
    # there also seems to be a link with CDH1, which is mutated in other cell lines
    # other cell lines that have strange pH3/pRB/IdU/cyclinB pattern show no
    # evidence of TOPK mediated ERK activation though
    model.pathway_elements["TOPK"] = {
        "Y74": list(active_rtks),
        # picked random site here, but not unrealistic as SRC/JAK are kinases
    }
    # for inhibition by pERK, there is evidence in literature. m_KRAS lines
    # MDAMB453, MDAMB134VI show hyperactivation of pMEK in iMEK condition
    # which cannot be explained by RTK negative feedback
    model.pathway_elements["ERK"] = {
        "Y204": (["MEK__S222_p", "TOPK__Y74_p"], ["iMEK_0", "ERK__Y204_p"])
    }
    model.pathway_elements["RPS6KA1"] = {"S380": ["ERK__Y204_p"]}


def add_mtor_akt(model):
    # https://doi.org/10.4161/trla.28174
    model.pathway_elements["PDPK1"] = {
        "S241": (list(active_rtks), ["iPI3K_0", *rtk_feedback])
    }
    model.pathway_elements["AKT"] = {
        "T308": ["PDPK1__S241_p"],
    }


def add_p38(model):
    mkk_activators = list(active_rtks)
    for mut in ["KRAS"]:
        if model.has_modification(f"m{mut.lower()}"):
            par = f"m_{mut}"
            model.parameters.add(par)
            mkk_activators.append(par)
    model.pathway_elements["MKK4"] = {
        "S257": (list(active_rtks), list(rtk_feedback))
    }
    model.pathway_elements["MKK36"] = {
        "S218": (mkk_activators, list(rtk_feedback))
    }
    model.pathway_elements["p38"] = {
        "T180": [
            "MKK4__S257_p",
            "MKK36__S218_p",
            "TOPK__Y74_p",
        ],
    }
    model.pathway_elements["MAPKAPK2"] = {
        "T334": ["p38__T180_p", "ERK__Y204_p"]
    }
    # add MAPKAPK2 as RPS6KA1 activator
    model.pathway_elements["RPS6KA1"]["S380"].append("MAPKAPK2__T334_p")
