# exported from PySB model 'EGFR_MAPK'

from pysb import (
    ANY,
    WILD,
    Annotation,
    Compartment,
    EnergyPattern,
    Expression,
    Initial,
    MatchOnce,
    Model,
    Monomer,
    MultiState,
    Observable,
    Parameter,
    Rule,
    Tag,
    as_complex_pattern,
)

Model()

Monomer("EGF", ["inh"])
Monomer("EGFR", ["Y1173", "inh"], {"Y1173": ["u", "p"]})
Monomer("MEK", ["S222", "inh"], {"S222": ["u", "p"]})
Monomer("ERK", ["Y204", "inh"], {"Y204": ["u", "p"]})

Parameter("EGF_0", 0.0)
Parameter("EGFR_eq", 100.0)
Parameter("INPUT_EGFR_eq", 0.0)
Parameter("EGFR_degradation_kdeg", 0.0)
Parameter("EGFR_dephosphorylation_Y1173_base_kcat", 0.0)
Parameter("INPUT_EGFR_dephosphorylation_Y1173_base_kcat", 0.0)
Parameter("EGFR_phosphorylation_Y1173_base_kr", 0.0)
Parameter("INPUT_EGFR_phosphorylation_Y1173_base_kr", 0.0)
Parameter("EGFR_phosphorylation_Y1173_kr", 1.0)
Parameter("INPUT_EGFR_phosphorylation_Y1173_kr", 0.0)
Parameter("degradation_EGFR__Y1173_p_kr", 0.0)
Parameter("MEK_eq", 100.0)
Parameter("INPUT_MEK_eq", 0.0)
Parameter("MEK_dephosphorylation_S222_base_kcat", 0.0)
Parameter("INPUT_MEK_dephosphorylation_S222_base_kcat", 0.0)
Parameter("MEK_phosphorylation_S222_base_kr", 0.0)
Parameter("INPUT_MEK_phosphorylation_S222_base_kr", 0.0)
Parameter("ERK_eq", 100.0)
Parameter("INPUT_ERK_eq", 0.0)
Parameter("ERK_dephosphorylation_Y204_base_kcat", 0.0)
Parameter("INPUT_ERK_dephosphorylation_Y204_base_kcat", 0.0)
Parameter("MEK_phosphorylation_S222_kr", 1.0)
Parameter("INPUT_MEK_phosphorylation_S222_kr", 0.0)
Parameter("MEK_deactivation_S222_ERK__Y204_p_kw", 0.0)
Parameter("INPUT_MEK_deactivation_S222_ERK__Y204_p_kw", 0.0)
Parameter("ERK_phosphorylation_Y204_kr", 1.0)
Parameter("INPUT_ERK_phosphorylation_Y204_kr", 0.0)
Parameter("iMEK_0", 0.0)
Parameter("iMEK_MEK__S222_p_obs_kd", 0.0)
Parameter("INPUT_iMEK_MEK__S222_p_obs_kd", 0.0)
Parameter("iEGFR_0", 0.0)
Parameter("iEGFR_EGFR__Y1173_p_obs_kd", 0.0)
Parameter("INPUT_iEGFR_EGFR__Y1173_p_obs_kd", 0.0)
Parameter("iPI3K_0", 0.0)
Parameter("iPKC_0", 0.0)

Expression("EGFR_init", EGFR_eq * INPUT_EGFR_eq)
Expression("EGFR_degradation_rate", EGFR_degradation_kdeg)
Expression("EGFR_synthesis_rate", EGFR_degradation_rate * EGFR_init)
Expression(
    "EGFR_dephosphorylation_Y1173_base_rate",
    EGFR_dephosphorylation_Y1173_base_kcat
    * INPUT_EGFR_dephosphorylation_Y1173_base_kcat,
)
Expression(
    "EGFR_phosphorylation_Y1173_base_rate",
    EGFR_dephosphorylation_Y1173_base_rate
    * EGFR_phosphorylation_Y1173_base_kr
    * INPUT_EGFR_phosphorylation_Y1173_base_kr,
)
Expression(
    "degradation_EGFR__Y1173_p_rate",
    EGFR_degradation_rate * degradation_EGFR__Y1173_p_kr,
)
Expression("MEK_init", INPUT_MEK_eq * MEK_eq)
Expression(
    "MEK_dephosphorylation_S222_base_rate",
    INPUT_MEK_dephosphorylation_S222_base_kcat
    * MEK_dephosphorylation_S222_base_kcat,
)
Expression(
    "MEK_phosphorylation_S222_base_rate",
    MEK_dephosphorylation_S222_base_rate
    * INPUT_MEK_phosphorylation_S222_base_kr
    * MEK_phosphorylation_S222_base_kr,
)
Expression("ERK_init", ERK_eq * INPUT_ERK_eq)
Expression(
    "ERK_dephosphorylation_Y204_base_rate",
    ERK_dephosphorylation_Y204_base_kcat
    * INPUT_ERK_dephosphorylation_Y204_base_kcat,
)

Observable("EGF_obs", EGF(inh=None))
Observable("EGFR__Y1173_p_obs", EGFR(Y1173="p", inh=None))
Observable("ERK__Y204_p_obs", ERK(Y204="p", inh=None))
Observable("MEK__S222_p_obs", MEK(S222="p", inh=None))
Observable("pEGFR_Y1173", EGFR(Y1173="p"))
Observable("pMEK_S222", MEK(S222="p"))
Observable("pERK_Y204", ERK(Y204="p"))

Expression(
    "free_EGFR__Y1173_p_obs",
    EGFR__Y1173_p_obs
    / (
        1
        + iEGFR_0
        / (INPUT_iEGFR_EGFR__Y1173_p_obs_kd * iEGFR_EGFR__Y1173_p_obs_kd)
    ),
)
Expression(
    "free_MEK__S222_p_obs",
    MEK__S222_p_obs
    / (1 + iMEK_0 / (INPUT_iMEK_MEK__S222_p_obs_kd * iMEK_MEK__S222_p_obs_kd)),
)
Expression(
    "EGFR_phosphorylation_Y1173_activation_rate",
    1.0
    * EGFR_dephosphorylation_Y1173_base_rate
    * EGF_obs
    * EGFR_phosphorylation_Y1173_kr
    * INPUT_EGFR_phosphorylation_Y1173_kr,
)
Expression(
    "MEK_phosphorylation_S222_activation_rate",
    MEK_dephosphorylation_S222_base_rate
    * free_EGFR__Y1173_p_obs
    * INPUT_MEK_phosphorylation_S222_kr
    * MEK_phosphorylation_S222_kr
    / (
        ERK__Y204_p_obs
        * INPUT_MEK_deactivation_S222_ERK__Y204_p_kw
        * MEK_deactivation_S222_ERK__Y204_p_kw
        + 1.0
    ),
)
Expression(
    "ERK_phosphorylation_Y204_activation_rate",
    1.0
    * ERK_dephosphorylation_Y204_base_rate
    * free_MEK__S222_p_obs
    * ERK_phosphorylation_Y204_kr
    * INPUT_ERK_phosphorylation_Y204_kr,
)

Rule("synthesis_EGFR", None >> EGFR(Y1173="u", inh=None), EGFR_synthesis_rate)
Rule("degradation_EGFR", EGFR() >> None, EGFR_degradation_rate)
Rule(
    "EGFR_base_regulation_Y1173_p",
    EGFR(Y1173="p") | EGFR(Y1173="u"),
    EGFR_dephosphorylation_Y1173_base_rate,
    EGFR_phosphorylation_Y1173_base_rate,
)
Rule(
    "EGFR_phosphorylation_Y1173_activation",
    EGFR(Y1173="u") >> EGFR(Y1173="p"),
    EGFR_phosphorylation_Y1173_activation_rate,
)
Rule(
    "degradation_EGFR__Y1173_p",
    EGFR(Y1173="p") >> None,
    degradation_EGFR__Y1173_p_rate,
)
Rule(
    "MEK_base_regulation_S222_p",
    MEK(S222="p") | MEK(S222="u"),
    MEK_dephosphorylation_S222_base_rate,
    MEK_phosphorylation_S222_base_rate,
)
Rule(
    "ERK_base_regulation_Y204_p",
    ERK(Y204="p") >> ERK(Y204="u"),
    ERK_dephosphorylation_Y204_base_rate,
)
Rule(
    "MEK_phosphorylation_S222_activation",
    MEK(S222="u") >> MEK(S222="p"),
    MEK_phosphorylation_S222_activation_rate,
)
Rule(
    "ERK_phosphorylation_Y204_activation",
    ERK(Y204="u") >> ERK(Y204="p"),
    ERK_phosphorylation_Y204_activation_rate,
)

Initial(EGF(inh=None), EGF_0, fixed=True)
Initial(EGFR(Y1173="u", inh=None), EGFR_init)
Initial(MEK(S222="u", inh=None), MEK_init)
Initial(ERK(Y204="u", inh=None), ERK_init)
