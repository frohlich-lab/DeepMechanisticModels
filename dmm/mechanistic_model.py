import itertools as itt
import re
from typing import Dict, Iterable, List, Tuple

import pysb.bng
import sympy as sp
from pysb import (
    Expression,
    Initial,
    Model,
    Monomer,
    Observable,
    Parameter,
    Rule,
)

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
    proteins: Iterable[Tuple[str, Dict[str, Iterable[str]]]],
    species_with_synth=None,
    species_with_free_levels=(),
    add_delay=None,
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

    for p_name, site_activators in proteins:
        add_monomer_synth_deg(
            model,
            p_name,
            sites=site_activators.keys(),
            with_synth=p_name in species_with_synth,
            compartments=["pm", "e"],
            species_with_free_levels=species_with_free_levels,
            add_delay=add_delay,
        )

    for p_name, site_activators in proteins:
        for site, modulators in site_activators.items():
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
                activation_type="phospho" if site != "compartment" else "endo",
                add_delay=add_delay,
            )


def add_monomer_synth_deg(
    model: Model,
    m_name: str,
    add_delay: dict,
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
                    f"u{d}"
                    for d in range(add_delay.get(f"{m_name}_{site}", 0))
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
        deg_rate = Expression(f"{m_name}_deg_rate", kdeg)
        syn_rate = Expression(f"{m_name}_syn_rate", t0 * deg_rate)
        Rule(f"syn_{m_name}", None >> syn_prod, syn_rate)
        Rule(f"deg_{m_name}", m() >> None, deg_rate)

    Initial(syn_prod, t0)

    # basal deactivation
    for site in psites:
        Rule(
            f"{m_name}_dp_{site}_p",
            m(**{site: "p"}) >> m(**{site: "u"}),
            add_parameter(f"{m_name}_dp_{site}_kcat", model),
        )

    if "compartment" in sites:
        site = "compartment"
        Rule(
            f"{m_name}_recycling",
            m(**{site: compartments[1]}) >> m(**{site: compartments[0]}),
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
    activation_type: str,
    add_delay: dict,
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

    if activation_type == "phospho":
        forward = "p"
        reverse = "dp"
        fstate = {site: "u"}
        rstate = {site: "p"}
    elif activation_type == "endo":
        forward = "endo"
        reverse = "recycle"
        fstate = {site: "pm"}
        rstate = {site: "e"}
    else:
        raise RuntimeError(f"Unknown activation type {activation_type}")

    if site == "compartment":
        reverse_name = f"{m_name}_{reverse}"
        forward_name = f"{m_name}_{forward}"
    else:
        reverse_name = f"{m_name}_{reverse}_{site}"
        forward_name = f"{m_name}_{forward}_{site}"
    koff = model.expressions[f"{reverse_name}_kcat"]

    activation = add_parameter(f"{reverse_name}_bact", model)
    for activator in activators:
        factor = add_or_get_modulator_obs(model, activator)
        factor *= add_parameter(f"{forward_name}_{activator}_kw", model)
        activation += Expression(
            f"{forward_name}_{activator}_activating_factor", factor
        )

    deactivation = 0
    for deactivator in deactivators:
        factor = add_or_get_modulator_obs(model, deactivator)
        factor *= add_parameter(f"{forward_name}_{deactivator}_kw", model)
        deactivation += Expression(
            f"{forward_name}_{deactivator}_inhibiting_factor", factor
        )

    # dynamic_range = sp.log(100)

    kr = Expression(f"{forward_name}_kr", activation / (1 + deactivation))

    rate = Expression(f"{forward_name}_activation_rate", koff * kr)

    if n_delays := add_delay.get(f"{m_name}_{site}", 0):
        Rule(
            f"{forward_name}_activation",
            mono(**fstate) >> mono(**{site: "u0"}),
            rate,
        )
        for idelay in range(n_delays - 1):
            Rule(
                f"{forward_name}_activation_d{idelay}",
                mono(**{site: f"u{idelay}"}) >> mono(**{site: f"u{idelay+1}"}),
                rate,
            )
        Rule(
            f"{forward_name}_activation_d{n_delays-1}",
            mono(**{site: f"u{n_delays-1}"}) >> mono(**rstate),
            rate,
        )

    else:
        Rule(
            f"{forward_name}_activation",
            mono(**fstate) >> mono(**rstate),
            rate,
        )


def add_degradation(model: Model, targets):
    for target in targets:
        mono_name, site_conditions = site_states_from_string(target)
        kr = add_parameter(f"degradation_{target}_kr", model)
        deg_rate = model.expressions[f"{mono_name}_degradation_rate"]
        rate = Expression(f"degradation_{target}_rate", kr * deg_rate)
        Rule(
            f"degradation_{target}",
            model.monomers[mono_name](**site_conditions) >> None,
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
        if "tobs" in model.name.split("_"):
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


def add_inhibitor(
    model: Model, name: str, targets: List[str], inhibits_activation=False
):
    inh = None

    free_targets = {}
    for target in targets:
        if target in model.observables.keys():
            target_sym = model.observables[target]
        elif (
            target_expr := target.replace("_obs", "")
        ) in model.expressions.keys():
            target_sym = model.expressions[target_expr]
            target = target_expr
        elif target in model.expressions.keys():
            target_sym = model.expressions[target]
        else:
            continue
        kd = add_parameter(f"{name}_{target}_kd", model)

        if inh is None:
            inh = Parameter(f"{name}_0", 0.0)

        # [A]*[I] = kD*[A:I]
        # [A]_0 = [A] + [A:I]
        # [A] = [A]_0 - [A:I]
        # [A] = [A]_0 - [A]*[I]/kD
        # [A](1 + [I]/kD) = [A]_0
        # [A] = [A]_0/(1 + [I]/kD)
        # free A: [A] = [A]_0 / (1 + kD*[I])

        free_targets[target] = Expression(
            f"free_{target}",
            target_sym / (1.0 + inh / kd),
            _export=False,
        )

        if target.endswith("_obs"):
            # just insert in the beginning, nothing to worry about
            model.expressions = pysb.ComponentSet(
                [kd, free_targets[target]] + list(model.expressions)
            )
        else:
            # we need to insert the new expression before all dependent
            # expression, but after the target expression
            expr_index = list(model.expressions.keys()).index(target)
            model.expressions = pysb.ComponentSet(
                list(model.expressions)[: expr_index + 1]
                + [kd, free_targets[target]]
                + list(model.expressions)[expr_index + 1 :]
            )

    if not free_targets:
        return

    for expr in model.expressions:
        if expr.name.startswith("free_"):
            continue

        for target, obs_or_expr in [
            (s.name, s)
            for s in expr.expr.free_symbols
            for name in (s.name, s.name + "_obs")
            if isinstance(s, (Observable, Expression)) and name in targets
        ]:
            expr.expr = expr.expr.subs(obs_or_expr, free_targets[target])


def add_gf_bolus(name: str):
    bolus = Monomer(f"{name}")
    Initial(bolus(), Parameter(f"{name}_0", 0.0), fixed=True)


def cleanup_unused(model):
    model.reset_equations()
    pysb.bng.generate_equations(model)

    observables = [
        obs.name for obs in model.expressions if obs.name.endswith("_obs")
    ]

    dynamic_eq = sp.Matrix(model.odes)

    expression_dynamic_symbols = set()
    for sym in dynamic_eq.free_symbols:
        if not isinstance(sym, Expression):
            continue
        if sym.name in model.expressions.keys():
            expression_dynamic_symbols |= (
                model.expressions[sym.name].expand_expr().free_symbols
            )

    initial_eq = sp.Matrix(
        [
            initial.value.expand_expr()
            if isinstance(initial.value, Expression)
            else initial.value
            for initial in model.initials
        ]
    )

    observable_eq = sp.Matrix(
        [
            expression.expand_expr()
            for expression in model.expressions
            if expression.name in observables
        ]
    )

    free_symbols = list(
        dynamic_eq.free_symbols
        | initial_eq.free_symbols
        | observable_eq.free_symbols
        | expression_dynamic_symbols
    )

    unused_pars = {
        par
        for par in model.parameters
        if par not in free_symbols and sp.Symbol(par.name) not in free_symbols
    }

    rule_reaction_count = {rule.name: 0 for rule in model.rules}

    for reaction in model.reactions:
        for rule in reaction["rule"]:
            rule_reaction_count[rule] += 1

    model.parameters = pysb.ComponentSet(
        [par for par in model.parameters if par not in unused_pars]
    )

    model.expressions = pysb.ComponentSet(
        [
            expr
            for expr in model.expressions
            if len(expr.expand_expr().free_symbols.intersection(unused_pars))
            == 0
            and not expr.name.startswith("_")
        ]
    )

    model.rules = pysb.ComponentSet(
        [rule for rule in model.rules if rule_reaction_count[rule.name] > 0]
    )

    model.reset_equations()
