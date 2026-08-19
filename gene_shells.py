"""Loader for MSigDB gene-set files.

Example
-------
>>> from gene_shells import load_msigdb_gmt
>>> gene_sets = load_msigdb_gmt("c2.cp.v2023.2.Hs.json")
>>> len(gene_sets)
"""

from __future__ import annotations

import json
from pathlib import Path


def load_msigdb_gmt(path: str | Path) -> dict[str, list[str]]:
    """Load an MSigDB gene-set JSON file (name → gene list)."""
    with open(path) as f:
        return json.load(f)
