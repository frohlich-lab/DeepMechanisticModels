"""Unified interface for fetching gene neighbourhood shells.

Supports three methods:
  - **STRING**: protein–protein interaction network shells via STRING API.
  - **MSigDB pathway**: co-occurrence in curated pathway gene sets (C2:CP).
  - **OmniPath**: interaction and protein-complex shells via OmniPath REST API.

Each method returns a :class:`ShellResult` dataclass with a consistent
interface (seeds, shell1, shell2, all gene lists).

Example
-------
>>> from gene_shells import get_shells, Methods
>>> result = get_shells(["ERBB2"], method="string", score=700)
>>> print(len(result.shell1), len(result.shell2))
>>> result.all_genes  # seed ∪ shell1 ∪ shell2
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import requests

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

Method = Literal[
    "string",
    "msigdb",
    "omnipath_interactions",
    "omnipath_complexes",
    "omnipath_combined",
]

METHODS: list[str] = [
    "string",
    "msigdb",
    "omnipath_interactions",
    "omnipath_complexes",
    "omnipath_combined",
]


@dataclass
class ShellResult:
    """Standardised output of any shell-fetching method.

    Attributes
    ----------
    method : str
        Which method produced this result.
    seeds : list[str]
        Input seed gene symbols.
    shell1 : list[str]
        First-shell genes (excluding seeds).
    shell2 : list[str]
        Second-shell genes (excluding seeds + shell1).
        Empty for methods that have no notion of a second shell.
    metadata : dict
        Method-specific extra information (e.g. number of pathways hit,
        number of complexes, number of edges, etc.).
    """

    method: str
    seeds: list[str]
    shell1: list[str]
    shell2: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    # -- derived helpers -----------------------------------------------------
    @property
    def all_genes(self) -> list[str]:
        """Sorted union of seeds + shell1 + shell2."""
        return sorted(set(self.seeds) | set(self.shell1) | set(self.shell2))

    @property
    def seed_and_shell1(self) -> list[str]:
        """Sorted union of seeds + shell1 (no 2nd shell)."""
        return sorted(set(self.seeds) | set(self.shell1))

    def summary(self) -> dict:
        return {
            "method": self.method,
            "n_seeds": len(self.seeds),
            "n_shell1": len(self.shell1),
            "n_shell2": len(self.shell2),
            "n_all": len(self.all_genes),
            "n_seed_and_shell1": len(self.seed_and_shell1),
        }


# ---------------------------------------------------------------------------
# Caching helper
# ---------------------------------------------------------------------------

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "gene_shells_cache"


def _cached_get_json(
    url: str,
    params: dict | None = None,
    data: dict | None = None,
    cache_path: Path | None = None,
    pause: float = 0.5,
    method: str = "GET",
) -> list | dict:
    """HTTP request with local JSON file cache."""
    if cache_path and cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    if method == "POST":
        r = requests.post(url, data=data or params)
    else:
        r = requests.get(url, params=params)
    r.raise_for_status()
    result = r.json()

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(result, f, indent=2)

    time.sleep(pause)
    return result


# ===================================================================
# STRING
# ===================================================================

_STRING_IDS_URL = "https://string-db.org/api/json/get_string_ids"
_STRING_NET_URL = "https://string-db.org/api/json/network"


def get_string_shells(
    seeds: list[str],
    *,
    score: int = 700,
    species: int = 9606,
    add_shell1: int = 1000,
    add_shell2: int = 1000,
    cache_dir: Path | None = None,
) -> ShellResult:
    """Fetch 1st + 2nd interaction shell from STRING.

    Parameters
    ----------
    seeds : list[str]
        Gene symbols.
    score : int
        STRING confidence score threshold (0–1000).
    species : int
        NCBI taxon ID (default 9606 = human).
    add_shell1, add_shell2 : int
        Max additional nodes per shell expansion.
    cache_dir : Path, optional
        Directory for caching API responses.
    """
    cd = _resolve_cache(cache_dir, "string")
    tag = "_".join(sorted(seeds))

    # Resolve STRING IDs
    seed_sids = [
        x["stringId"]
        for x in _cached_get_json(
            _STRING_IDS_URL,
            data={"identifiers": "\r".join(seeds), "species": species},
            cache_path=cd / f"ids_{tag}.json",
            method="POST",
        )
    ]

    # --- 1st shell ---
    edges_s1 = _cached_get_json(
        _STRING_NET_URL,
        data={
            "identifiers": "\r".join(seed_sids),
            "species": species,
            "required_score": score,
            "add_nodes": add_shell1,
        },
        cache_path=cd / f"net_s1_{tag}_sc{score}.json",
        method="POST",
    )
    shell1_all = set()
    for e in edges_s1:
        shell1_all.add(e["preferredName_A"])
        shell1_all.add(e["preferredName_B"])
    shell1_only = sorted(shell1_all - set(seeds))

    # --- 2nd shell ---
    sids2 = [
        x["stringId"]
        for x in _cached_get_json(
            _STRING_IDS_URL,
            data={
                "identifiers": "\r".join(sorted(shell1_all)),
                "species": species,
            },
            cache_path=cd / f"ids2_{tag}_sc{score}.json",
            method="POST",
        )
    ]
    edges_s2 = _cached_get_json(
        _STRING_NET_URL,
        data={
            "identifiers": "\r".join(sids2),
            "species": species,
            "required_score": score,
            "add_nodes": add_shell2,
        },
        cache_path=cd / f"net_s2_{tag}_sc{score}.json",
        method="POST",
    )
    all_genes = set()
    for e in edges_s2:
        all_genes.add(e["preferredName_A"])
        all_genes.add(e["preferredName_B"])
    shell2_only = sorted(all_genes - shell1_all)

    return ShellResult(
        method="string",
        seeds=sorted(seeds),
        shell1=shell1_only,
        shell2=shell2_only,
        metadata={
            "score": score,
            "n_edges_s1": len(edges_s1),
            "n_edges_s2": len(edges_s2),
        },
    )


# ===================================================================
# MSigDB pathway co-occurrence
# ===================================================================

# Default pathway database groups
PATHWAY_DB_GROUPS: dict[str, list[str]] = {
    "KEGG": ["KEGG_"],
    "Reactome": ["REACTOME_"],
    "PID": ["PID_"],
    "BioCarta": ["BIOCARTA_"],
    "WikiPath": ["WP_"],
    "Signaling": ["KEGG_", "REACTOME_", "PID_"],
    "All": [],  # empty → no prefix filter
}


def load_msigdb_gmt(path: str | Path) -> dict[str, list[str]]:
    """Load an MSigDB gene-set JSON file (name → gene list)."""
    with open(path) as f:
        return json.load(f)


def get_msigdb_shells(
    seeds: list[str],
    gmt: dict[str, list[str]],
    *,
    db_prefixes: list[str] | None = None,
) -> ShellResult:
    """Build a pathway co-occurrence shell from MSigDB C2:CP gene sets.

    Parameters
    ----------
    seeds : list[str]
        Gene symbols.
    gmt : dict[str, list[str]]
        Gene-set collection (pathway name → gene list).
    db_prefixes : list[str], optional
        If given, only consider pathways whose names start with one of these
        prefixes (e.g. ``["KEGG_", "REACTOME_"]``).  ``None`` → use all.
    """
    seed_set = set(seeds)
    pathways_hit: list[str] = []
    shell_genes: set[str] = set()

    for pw_name, pw_genes_list in gmt.items():
        if db_prefixes:
            if not any(pw_name.startswith(p) for p in db_prefixes):
                continue
        pw_genes = set(pw_genes_list)
        if pw_genes & seed_set:
            pathways_hit.append(pw_name)
            shell_genes |= pw_genes

    shell_only = sorted(shell_genes - seed_set)

    return ShellResult(
        method="msigdb",
        seeds=sorted(seeds),
        shell1=shell_only,
        shell2=[],  # pathway shells have no notion of 2nd shell
        metadata={
            "n_pathways_hit": len(pathways_hit),
            "pathways_hit": pathways_hit,
            "db_prefixes": db_prefixes,
        },
    )


# ===================================================================
# OmniPath
# ===================================================================

_OMNIPATH_BASE = "https://omnipathdb.org"
_OMNIPATH_DEFAULT_DATASETS = ["omnipath", "pathwayextra", "kinaseextra"]

# Gene-symbol → UniProt mapping for the /complexes endpoint.
# Extend as needed; only seeds used in complexes queries need to be here.
GENESYMBOL_TO_UNIPROT: dict[str, str] = {
    "ERBB2": "P04626",
    "MAPK1": "P28482",
    "MAPK3": "P27361",
    "MAP2K1": "Q02750",
    "MAP2K2": "P36507",
    "RPS6KA1": "Q15418",
    "RPS6KA2": "Q15349",
    "RPS6KA3": "P51812",
    "RPS6KA6": "Q9UK32",
}


def _omnipath_get(
    endpoint: str,
    params: dict,
    cache_path: Path | None = None,
    pause: float = 0.5,
) -> list | dict:
    params = {**params, "format": "json"}
    return _cached_get_json(
        f"{_OMNIPATH_BASE}/{endpoint}",
        params=params,
        cache_path=cache_path,
        pause=pause,
        method="GET",
    )


def get_omnipath_interaction_shells(
    seeds: list[str],
    *,
    datasets: list[str] | None = None,
    cache_dir: Path | None = None,
) -> ShellResult:
    """Fetch 1st + 2nd interaction shell via OmniPath REST API.

    Both shells are built by querying upstream regulators only: at each
    hop we ask "what regulates gene X?" (gene X = target in the
    interaction).  Shell-1 = upstream regulators of the seeds; Shell-2 =
    upstream regulators of the shell-1 genes.

    Parameters
    ----------
    seeds : list[str]
        Gene symbols.
    datasets : list[str], optional
        OmniPath interaction datasets (default: omnipath + pathwayextra +
        kinaseextra).
    cache_dir : Path, optional
        Cache directory for API responses.
    """
    datasets = datasets or _OMNIPATH_DEFAULT_DATASETS
    cd = _resolve_cache(cache_dir, "omnipath")
    seed_set = set(seeds)
    ds_str = ",".join(datasets)
    tag = "_".join(sorted(seeds))

    # --- 1st shell: upstream regulators of seeds (seed = target) ---
    data_s1 = _omnipath_get(
        "interactions",
        {
            "targets": ",".join(seeds),
            "genesymbols": 1,
            "datasets": ds_str,
        },
        cache_path=cd / f"ix_s1_tgt_{tag}.json",
    )

    shell1_genes: set[str] = set()
    for edge in data_s1:
        shell1_genes.add(edge["source_genesymbol"])
    # keep seeds in the combined set but not in shell1_only
    shell1_only = sorted(shell1_genes - seed_set)

    # --- 2nd shell: upstream regulators of shell-1 genes (batched) ---
    s1_query_genes = sorted(shell1_genes - seed_set)  # only new genes
    batch_size = 50
    data_s2: list[dict] = []
    for i in range(0, len(s1_query_genes), batch_size):
        batch = s1_query_genes[i : i + batch_size]
        batch_data = _omnipath_get(
            "interactions",
            {
                "targets": ",".join(batch),
                "genesymbols": 1,
                "datasets": ds_str,
            },
            cache_path=cd / f"ix_s2_up_{tag}_b{i}.json",
        )
        data_s2.extend(batch_data)

    shell2_genes: set[str] = set()
    for edge in data_s2:
        shell2_genes.add(edge["source_genesymbol"])
    shell2_only = sorted(shell2_genes - shell1_genes - seed_set)

    return ShellResult(
        method="omnipath_interactions",
        seeds=sorted(seeds),
        shell1=shell1_only,
        shell2=shell2_only,
        metadata={
            "datasets": datasets,
            "n_edges_s1": len(data_s1),
            "n_edges_s2": len(data_s2),
        },
    )


def _query_complexes(
    query_genes: list[str],
    gene_to_uniprot: dict[str, str],
    cd: Path,
    seen_ids: set[str],
    cache_prefix: str = "cx",
) -> tuple[set[str], list[dict], dict[str, str]]:
    """Query /complexes for a list of genes and return co-members.

    Returns
    -------
    new_genes : set[str]
        All gene symbols found as complex co-members.
    complexes_hit : list[dict]
        Metadata for each unique complex returned.
    gene_to_uniprot_updated : dict[str, str]
        Updated gene→UniProt mapping (extended from complex component lists).
    """
    new_genes: set[str] = set()
    complexes_hit: list[dict] = []

    for gene in query_genes:
        uid = gene_to_uniprot.get(gene)
        if uid is None:
            continue

        tag = uid.replace(":", "_")
        data_cx = _omnipath_get(
            "complexes",
            {
                "proteins": uid,
            },
            cache_path=cd / f"{cache_prefix}_{tag}.json",
        )

        for cx in data_cx:
            cx_id = cx.get("identifiers", "")
            if cx_id and cx_id in seen_ids:
                continue
            if cx_id:
                seen_ids.add(cx_id)

            members_sym = cx.get("components_genesymbols", [])
            members_uid = cx.get("components", [])
            if members_sym:
                complexes_hit.append(
                    {
                        "name": cx.get("name", ""),
                        "members": members_sym,
                        "sources": cx.get("sources", []),
                        "n_members": len(members_sym),
                    }
                )
                new_genes.update(members_sym)
                # extend gene→UniProt mapping from response
                for sym, uid_comp in zip(members_sym, members_uid):
                    if sym not in gene_to_uniprot:
                        gene_to_uniprot[sym] = uid_comp

    return new_genes, complexes_hit, gene_to_uniprot


def get_omnipath_complex_shells(
    seeds: list[str],
    *,
    uniprot_map: dict[str, str] | None = None,
    cache_dir: Path | None = None,
) -> ShellResult:
    """Fetch 1st + 2nd complex-membership shell via OmniPath /complexes.

    Shell-1: genes that share a complex with any seed.
    Shell-2: genes that share a complex with any shell-1 gene
             (excluding seeds and shell-1 genes).

    The complexes endpoint requires UniProt IDs.  A mapping from gene symbol
    to UniProt AC is used (``uniprot_map``; falls back to the built-in
    :data:`GENESYMBOL_TO_UNIPROT`).  The mapping is automatically extended
    with UniProt IDs discovered from complex component lists.

    Parameters
    ----------
    seeds : list[str]
        Gene symbols.
    uniprot_map : dict[str, str], optional
        Gene symbol → UniProt AC mapping.
    cache_dir : Path, optional
        Cache directory for API responses.
    """
    gene_to_uniprot = dict(uniprot_map or GENESYMBOL_TO_UNIPROT)
    cd = _resolve_cache(cache_dir, "omnipath")
    seed_set = set(seeds)
    seen_ids: set[str] = set()

    # --- Shell 1: complexes containing any seed ---
    s1_genes, s1_complexes, gene_to_uniprot = _query_complexes(
        seeds,
        gene_to_uniprot,
        cd,
        seen_ids,
        cache_prefix="cx_s1",
    )
    shell1 = sorted(s1_genes - seed_set)

    # --- Shell 2: complexes containing any shell-1 gene ---
    s2_genes, s2_complexes, gene_to_uniprot = _query_complexes(
        shell1,
        gene_to_uniprot,
        cd,
        seen_ids,
        cache_prefix="cx_s2",
    )
    shell1_set = set(shell1) | seed_set
    shell2 = sorted(s2_genes - shell1_set)

    return ShellResult(
        method="omnipath_complexes",
        seeds=sorted(seeds),
        shell1=shell1,
        shell2=shell2,
        metadata={
            "n_complexes_s1": len(s1_complexes),
            "n_complexes_s2": len(s2_complexes),
            "complexes_s1": s1_complexes,
            "complexes_s2": s2_complexes,
        },
    )


def get_omnipath_combined_shells(
    seeds: list[str],
    *,
    datasets: list[str] | None = None,
    uniprot_map: dict[str, str] | None = None,
    cache_dir: Path | None = None,
) -> ShellResult:
    """Union of OmniPath interactions and complexes shells.

    Returns a single :class:`ShellResult` whose *shell1* is the union of
    the interactions-shell1 and complex-shell, and *shell2* is the
    interactions-shell2 minus anything already in shell1.
    The ``metadata`` dict carries the individual sub-results.
    """
    ix = get_omnipath_interaction_shells(
        seeds, datasets=datasets, cache_dir=cache_dir
    )
    cx = get_omnipath_complex_shells(
        seeds, uniprot_map=uniprot_map, cache_dir=cache_dir
    )

    seed_set = set(seeds)
    combined_s1 = sorted((set(ix.shell1) | set(cx.shell1)) - seed_set)
    combined_s1_set = set(combined_s1) | seed_set
    combined_s2 = sorted(set(ix.shell2) - combined_s1_set)

    return ShellResult(
        method="omnipath_combined",
        seeds=sorted(seeds),
        shell1=combined_s1,
        shell2=combined_s2,
        metadata={
            "interactions": ix,
            "complexes": cx,
            "ix_only": sorted(set(ix.all_genes) - set(cx.all_genes)),
            "cx_only": sorted(set(cx.all_genes) - set(ix.all_genes)),
            "ix_cx_overlap": sorted(set(ix.all_genes) & set(cx.all_genes)),
        },
    )


# ===================================================================
# Unified dispatcher
# ===================================================================


def get_shells(
    seeds: list[str],
    method: Method = "string",
    **kwargs,
) -> ShellResult:
    """Unified entry point for fetching gene neighbourhood shells.

    Parameters
    ----------
    seeds : list[str]
        Gene symbols to expand.
    method : str
        One of ``"string"``, ``"msigdb"``, ``"omnipath_interactions"``,
        ``"omnipath_complexes"``, ``"omnipath_combined"``.
    **kwargs
        Forwarded to the method-specific function.  See individual docstrings
        for available parameters.

    Returns
    -------
    ShellResult
    """
    dispatch = {
        "string": get_string_shells,
        "msigdb": get_msigdb_shells,
        "omnipath_interactions": get_omnipath_interaction_shells,
        "omnipath_complexes": get_omnipath_complex_shells,
        "omnipath_combined": get_omnipath_combined_shells,
    }
    if method not in dispatch:
        raise ValueError(
            f"Unknown method {method!r}. Choose from {list(dispatch)}"
        )
    return dispatch[method](seeds, **kwargs)


# ===================================================================
# Helpers
# ===================================================================


def _resolve_cache(cache_dir: Path | None, subdir: str) -> Path:
    cd = (cache_dir or DEFAULT_CACHE_DIR) / subdir
    cd.mkdir(parents=True, exist_ok=True)
    return cd
