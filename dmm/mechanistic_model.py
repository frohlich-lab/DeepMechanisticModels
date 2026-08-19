import itertools as itt
import re
from typing import Iterable

from pysb import (
    Expression,
    Initial,
    Model,
    Monomer,
    Observable,
    Parameter,
    Rule,
)

from cytof.pathways import add_egfr, add_mapk, add_mtor_akt, add_p38


class MechanisticModel:
    def __init__(self, model_name: str):
        self.species_with_synth: list[str] = []
        self.species_with_free_levels: list[str] = []
        self.pathway_elements: dict[str, dict] = {}
        self.delays: dict[str, int] = {}
        self.require_phosphorylation: dict[str, str] = {}
        self.require_compartment: dict[str, str] = {}
        self.parameters: set[str] = set()
        self.components: tuple[str, ...] = ()
        self.modifications: tuple[str, ...] = ()

        self.components = tuple(model_name.split("__")[0].split("_"))
        if "__" in model_name:
            self.modifications = tuple(model_name.split("__")[1].split("_"))

        if "EGFR" in self.components:
            add_egfr(self)

        if "MAPK" in self.components:
            add_mapk(self)

        if "AKT" in self.components:
            add_mtor_akt(self)

        if "P38" in self.components:
            add_p38(self)

    def construct_pysb(self, model_name):
        pysb_model = Model(model_name)

        # sorted() since self.parameters is a set: without it the order of
        # Parameter declarations (and thus the exported pysb_flat file) varies
        # between runs due to string hash randomisation
        for par in sorted(self.parameters):
            Parameter(par)

        generate_pathway(
            model=pysb_model,
            proteins=self.pathway_elements,
            species_with_free_levels=self.species_with_free_levels,
            species_with_synth=self.species_with_synth,
            add_delay=self.delays,
            require_compartment=self.require_compartment,
            require_phosphorylation=self.require_phosphorylation,
        )

        add_observables(pysb_model)
        return pysb_model

    def has_modification(self, modification):
        return modification in self.modifications


from . import MODEL_FEATURE_PREFIX


def add_parameter(name: str, model: Model):
    """Adds a parameter to the model

    :param model:
        model to which the parameter will be added

    :param name:
        name of the parameter

    :param value:
        value of the parameter
    """
    if name in model.expressions.keys():
        return model.expressions[name]

    if not f"MED_{name}" in model.parameters.keys():
        kavg = Parameter(f"MED_{name}", 1.0)
    else:
        kavg = model.parameters[f"MED_{name}"]
    kmod = get_autoencoder_modulator(kavg, model)
    return Expression(name, kavg * kmod)


def generate_pathway(
    model: Model,
    proteins: dict[str, dict[str, Iterable[str]]],
    species_with_synth=None,
    species_with_free_levels=(),
    add_delay=None,
    require_compartment=None,
    require_phosphorylation=None,
):
    """Adds synthesis and phospho-signal transduction rules to the model
    based on the input specifications

    :param model:
        model to which rules will be added

    :param proteins:
        pathway specification
    """
    if add_delay is None:
        add_delay = {}
    if species_with_synth is None:
        species_with_synth = []
    if require_compartment is None:
        require_compartment = {}
    if require_phosphorylation is None:
        require_phosphorylation = {}

    for p_name, site_modulators in proteins.items():
        add_monomer_synth_deg(
            model,
            p_name,
            sites=[
                s if s != "endocytosis" else "compartment"
                for s in site_modulators.keys()
                if s != "degradation"
            ],
            with_synth=p_name in species_with_synth,
            compartments=["pm", "e"],
            species_with_free_levels=species_with_free_levels,
            add_delay=add_delay,
            require_compartment=require_compartment,
        )

    for p_name, site_modulators in proteins.items():
        for site, modulators in site_modulators.items():
            if isinstance(modulators, list):
                activators = modulators
                deactivators = []
            else:
                activators, deactivators = modulators

            add_activation(
                model=model,
                m_name=p_name,
                site=site,
                activators=activators,
                deactivators=deactivators,
                n_delays=add_delay.get(f"{p_name}_{site}", 0),
                require_compartment=require_compartment.get(
                    f"{p_name}_{site}"
                ),
                require_phosphorylation=require_phosphorylation.get(
                    f"{p_name}_{site}"
                ),
            )


def add_monomer_synth_deg(
    model: Model,
    m_name: str,
    add_delay: dict,
    require_compartment: dict,
    compartments: Iterable[str],
    sites: Iterable[str] = (),
    with_synth=False,
    species_with_free_levels=(),
):
    """Adds the respective monomer plus synthesis rules and basal
    activation/deactivation rules for all activateable sites

    :param model:
        model to which the monomer will be added

    :param m_name:
        monomer name

    :param psites:
        phospho sites
    """
    psites = sorted({site for site in sites if re.match(r"[YTS][0-9]+", site)})

    if m_name not in model.monomers.keys():
        # basic states
        site_states = {
            site: ["u", "p"] if site in psites else compartments
            for site in sites
        }
        # extend for delays
        site_states = {
            site: [
                states[0],
                *[
                    f"e{d}" if site == "compartment" else f"u{d}"
                    for d in range(
                        add_delay.get(
                            f"{m_name}_{site.replace('compartment', 'endocytosis')}",
                            0,
                        )
                    )
                ],
                states[1],
            ]
            for site, states in site_states.items()
        }
        m = Monomer(
            m_name,
            sites=sites,
            site_states={
                site: site_states[site]
                for site in sites
                if site in psites + ["compartment"]
            },
        )
    else:
        m = model.monomers[m_name]

    if m_name in species_with_free_levels:
        t = add_parameter(f"{m_name}_eq", model)
    else:
        t = Parameter(f"{m_name}_eq", 100.0)
    t0 = Expression(f"{m_name}_init", t)

    syn_prod = m(
        **{
            site: "u" if site in psites else compartments[0]
            for site in m.sites
        }
    )

    if with_synth:
        kdeg = add_parameter(f"{m_name}_deg_kdeg", model)
        syn_rate = Expression(f"{m_name}_syn_rate", t0 * kdeg)
        Rule(f"syn_{m_name}", None >> syn_prod, syn_rate)
        Rule(f"deg_{m_name}", m() >> None, kdeg)

    Initial(syn_prod, t0)

    # basal deactivation
    for site in psites:
        rstate = {site: "p"}
        fstate = {site: "u"}
        if comp := require_compartment.get(f"{m_name}_{site}"):
            rstate["compartment"] = comp
            fstate["compartment"] = comp

        Rule(
            f"{m_name}_dp_{site}_p",
            m(**rstate) >> m(**fstate),
            add_parameter(f"{m_name}_dp_{site}_kcat", model),
        )

    if "compartment" in sites:
        Rule(
            f"{m_name}_recycling",
            m(**{"compartment": compartments[1]})
            >> m(**{"compartment": compartments[0]}),
            add_parameter(f"{m_name}_recycle_kcat", model),
        )

    return m


def add_or_get_modulator_obs(model: Model, modulator: str):
    """Adds an observable to the model that tracks the specified modulator

    :param model:
        model to which the observable will be added

    :param modulator:
        string definition of an observable in format
        `{monomer_name}__{site}_{site_condition}`
    """
    if modulator in ["iEGFR_0", "iMEK_0", "iPKC_0", "iMTOR_0", "iPI3K_0"] and (
        modulator not in model.parameters.keys()
    ):
        Parameter(modulator)

    if modulator in model.expressions.keys():
        return model.expressions[modulator]

    if modulator in model.parameters.keys():
        return model.parameters[modulator]

    if (mod_name := f"{modulator}_obs") in model.observables.keys():
        modulator_obs = model.observables[f"{modulator}_obs"]
    else:
        mono_name, site_conditions = site_states_from_string(modulator)

        modulator_obs = Observable(
            mod_name, model.components[mono_name](**site_conditions)
        )

    return modulator_obs


def site_states_from_string(obs_string):
    desc = obs_string.split("__")
    mono_name = desc[0]
    if len(desc) > 1:
        site_conditions = desc[1:]
    else:
        site_conditions = []

    try:
        site_conditions = {
            cond.split("_")[0]: cond.split("_")[1] for cond in site_conditions
        }
    except IndexError as err:
        raise ValueError(
            f"Malformed site condition {site_conditions}"
        ) from err

    return mono_name, site_conditions


def add_activation(
    model: Model,
    m_name: str,
    site: str,
    n_delays: int,
    require_compartment: str | None,
    require_phosphorylation: str | None,
    activators: Iterable[str] = (),
    deactivators: Iterable[str] = (),
):
    """Adds activation/deactivation rules to a specific site

    :param model:
        model to which the rules will be added

    :param m_name:
        monomer name

    :param site:
        site name

    :param activators:
        molecular species that activate the respective site, format
        according to modulator format in :py:func:`add_or_get_modulator_obs`

    :param deactivators:
        molecular species that deactivate the respective site, format
        according to modulator format in :py:func:`add_or_get_modulator_obs`

    """
    if activators is None:
        activators = []

    if deactivators is None:
        deactivators = []

    if m_name not in model.monomers.keys():
        raise ValueError(f"{m_name} is not a monomer in the model.")

    mono = model.monomers[m_name]

    activation_types = {
        "endocytosis": "endo",
        "degradation": "deg",
    }

    activation_type = activation_types.get(site, "phospho")
    site = "compartment" if activation_type == "endo" else site

    if activation_type == "phospho":
        forward = "p"
        reverse = "dp"
        fstate = {site: "u"}
        rstate = {site: "p"}
        dstate = {}
        if require_compartment:
            fstate["compartment"] = require_compartment
            rstate["compartment"] = require_compartment
    elif activation_type == "endo":
        forward = "endo"
        reverse = "recycle"
        fstate = {site: "pm"}
        rstate = {site: "e"}
        dstate = {}
        if require_phosphorylation:
            fstate[require_phosphorylation] = "p"
            if n_delays == 0:
                fstate[require_phosphorylation] = "p"
            else:
                dstate[require_phosphorylation] = "p"
    elif activation_type == "deg":
        forward = "deg"
        reverse = "deg"
        fstate = {}
        rstate = {}
        dstate = {}
        if require_compartment:
            fstate["compartment"] = require_compartment
    else:
        raise RuntimeError(f"Unknown activation type {activation_type}")

    if site == "compartment":
        forward_name = f"{m_name}_{forward}"
        kref = model.expressions[f"{m_name}_{reverse}_kcat"]
    elif site == "degradation":
        forward_name = f"{m_name}_{forward}"
        kref = model.expressions[f"{m_name}_{reverse}_kdeg"]
    else:
        forward_name = f"{m_name}_{forward}_{site}"
        kref = model.expressions[f"{m_name}_{reverse}_{site}_kcat"]

    activations = [add_parameter(f"{forward_name}_bact", model)]

    for activator in activators:
        factor = add_or_get_modulator_obs(model, activator)
        weight = add_parameter(f"{forward_name}_{activator}_kw", model)
        factor *= weight
        activations.append(
            Expression(f"{forward_name}_{activator}_activating_factor", factor)
        )

    deactivations = []
    for deactivator in deactivators:
        factor = add_or_get_modulator_obs(model, deactivator)
        weight = add_parameter(f"{forward_name}_{deactivator}_kw", model)
        factor *= weight
        deactivations.append(
            Expression(
                f"{forward_name}_{deactivator}_inhibiting_factor", factor
            )
        )

    kr_name = f"{forward_name}_kr"
    kr = Expression(kr_name, sum(activations) / (1 + sum(deactivations)))

    rate = Expression(f"{forward_name}_activation_rate", kref * kr)

    if n_delays:
        prefix = "e" if activation_type == "endo" else "u"
        Rule(
            f"{forward_name}_activation",
            mono(**fstate) >> mono(**{site: f"{prefix}0", **dstate}),
            rate,
        )
        for idelay in range(n_delays - 1):
            Rule(
                f"{forward_name}_activation_d{idelay}",
                mono(**{site: f"{prefix}{idelay}"})
                >> mono(**{site: f"{prefix}{idelay+1}"}),
                kref,
            )
        Rule(
            f"{forward_name}_activation_d{n_delays-1}",
            mono(**{site: f"{prefix}{n_delays-1}"}) >> mono(**rstate),
            kref,
        )
    elif activation_type == "deg":
        Rule(
            forward_name,
            mono(**fstate) >> None,
            rate,
        )

    else:
        Rule(
            f"{forward_name}_activation",
            mono(**fstate) >> mono(**rstate),
            rate,
        )


def get_autoencoder_modulator(par: Parameter, model: Model):
    """Generate a new expression that allows modulation of a rate according to
    input parameter. Applies a sigmoid transformation.
    """
    name = par.name.replace("MED_", MODEL_FEATURE_PREFIX)
    if name in model.parameters.keys():
        return model.parameters[name]
    else:
        return Parameter(name, 1.0)


def add_observables(model: Model):
    """Adds a observable that tracks the normalized absolute abundance of all
    phosphorylated site combinations for all monomers
    """
    for monomer in model.monomers:
        if "pobs" in model.name.split("_"):
            Observable(f"t{monomer.name}", monomer())
        psites = [
            site
            for site in monomer.site_states.keys()
            if re.match(r"[YTS][0-9]+$", site)
        ]
        for nsites in range(1, len(psites) + 1):
            for sites in itt.combinations(psites, nsites):
                sites = sorted(sites)
                Observable(
                    f"p{monomer.name}_{'_'.join(sites)}",
                    monomer(**{site: "p" for site in sites}),
                )
