"""
Consolidated cell-line annotation module.

Merges functionality from ``cellosaurus_utils.py`` and inline notebook code
(cBioPortal / Cell Model Passports API calls) into a single, cached entry
point.

Usage
-----
>>> from cell_line_annotations import get_all_annotations
>>>
>>> # All annotation blocks at once (cached)
>>> ann = get_all_annotations(cell_lines, cache_dir=features_dir)
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

# ── project-local imports ────────────────────────────────────────────
from common import basedir, features_dir, subtypes_tognetti

# ═══════════════════════════════════════════════════════════════════════
# Cache helpers
# ═══════════════════════════════════════════════════════════════════════

_CACHE_VERSION = "v2"  # bump when data schema changes


def _cache_path(cache_dir: Path, name: str, ext: str = "parquet") -> Path:
    """Return the canonical cache file path for a given artefact."""
    return cache_dir / ".annotation_cache" / f"{name}_{_CACHE_VERSION}.{ext}"


def _load_cache(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        elif path.suffix == ".pkl":
            return pd.read_pickle(path)
        elif path.suffix == ".json":
            return json.loads(path.read_text())
    except Exception as exc:
        warnings.warn(f"Cache read failed for {path}: {exc}", stacklevel=2)
    return None


def _save_cache(obj: pd.DataFrame | dict | set, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if isinstance(obj, pd.DataFrame):
            if path.suffix == ".parquet":
                obj.to_parquet(path)
            else:
                obj.to_pickle(path)
        elif isinstance(obj, (dict, list, set)):
            path.write_text(
                json.dumps(
                    list(obj) if isinstance(obj, set) else obj,
                    default=str,
                )
            )
    except Exception as exc:
        warnings.warn(f"Cache write failed for {path}: {exc}", stacklevel=2)


# ═══════════════════════════════════════════════════════════════════════
# 1. Cellosaurus parsing  (from cellosaurus_utils.py)
# ═══════════════════════════════════════════════════════════════════════


def download_cellosaurus_file(
    file_dir: Path,
    url: str = "https://ftp.expasy.org/databases/cellosaurus/cellosaurus.txt",
    filename: str = "cellosaurus.txt",
) -> Path:
    """Download the Cellosaurus flat-file if not present and return its path."""
    file_path = file_dir / filename
    if not file_path.is_file():
        print(f"Downloading Cellosaurus from {url} …")
        file_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, file_path)
        print("Download complete.")
    return file_path


# ── field extractors ─────────────────────────────────────────────────


def _extract_site(comments: list[str]) -> str:
    for c in comments:
        m = re.search(r"Derived from site:\s*(.+?)(?:\s*UBERON=|$)", c)
        if m:
            return m.group(1).strip().rstrip(";")
    return "Unknown"


def _extract_msi(comments: list[str]) -> str:
    for c in comments:
        m = re.search(r"Microsatellite instability:\s*(.*?)(?:\s*\||$)", c)
        if m:
            sm = re.search(
                r"(Stable \(MSS\)|Instable \(MSI-low\)|Instable \(MSI-high\))",
                m.group(1),
            )
            if sm:
                return sm.group(0)
    return "Unknown"


def _extract_doubling_time(comments: list[str]) -> float:
    for c in comments:
        if not c.startswith("Doubling time:"):
            continue
        text = c.replace("Doubling time:", "").strip()
        m = re.search(r"~?(\d+\.?\d*)\s*(hours?|days?)", text, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            if m.group(2).lower().startswith("day"):
                val *= 24.0
            return val
    return float("nan")


def _extract_population(comments: list[str]) -> str:
    for c in comments:
        if c.startswith("Population:"):
            return c.replace("Population:", "").strip().rstrip(".")
    return "Unknown"


def _extract_category(ca_field: str | None) -> str:
    ca = (ca_field or "").strip()
    return ca if ca else "Unknown"


def _extract_sequence_variations(comments: list[str]) -> list[dict]:
    sv_data: list[dict] = []
    for c in comments:
        if not c.startswith("Sequence variation:"):
            continue
        entry = c.replace("Sequence variation:", "").strip()
        sv_type_m = re.match(
            r"^(Gene deletion|Mutation|Gene fusion|Gene amplification|Other):?",
            entry,
        )
        sv_type = sv_type_m.group(1) if sv_type_m else "Unknown"
        hgnc_m = re.search(r"HGNC:?;?\s*HGNC:(\d+);?\s*([A-Z0-9\-]+);?", entry)
        gene = hgnc_m.group(2) if hgnc_m else None
        hgnc_id = hgnc_m.group(1) if hgnc_m else None
        zyg_m = re.search(r"Zygosity=([A-Za-z]+)", entry)
        zygosity = zyg_m.group(1) if zyg_m else None
        pc_m = re.search(r"(p\.[A-Za-z0-9]+)", entry)
        protein_change = pc_m.group(1) if pc_m else None
        cdna_m = re.search(r"(c\.[0-9]+[A-Z]>[A-Z])", entry)
        cdna_change = cdna_m.group(1) if cdna_m else None
        cv_m = re.search(r"ClinVar=([A-Za-z0-9_]+)", entry)
        clinvar = cv_m.group(1) if cv_m else None
        sv_data.append(
            {
                "type": sv_type,
                "gene": gene,
                "hgnc_id": hgnc_id,
                "zygosity": zygosity,
                "protein_change": protein_change,
                "cdna_change": cdna_change,
                "clinvar_id": clinvar,
            }
        )
    return sv_data if sv_data else [{"type": "Not Found"}]


# ── Cellosaurus record processing ───────────────────────────────────


def _zygosity_score(z: str | None) -> float:
    if z == "Homozygous":
        return 1.0
    if z == "Heterozygous":
        return 0.5
    return 0.5  # impute unknown


def _gene_variation_label(var: dict) -> tuple[str | None, str | None]:
    vt = var.get("type")
    gene = var.get("gene")
    zygosity = var.get("zygosity") or "Unknown"
    protein = var.get("protein_change")
    if not gene or not vt:
        return None, None
    if vt == "Mutation":
        label = f"M ({protein})" if protein else "M"
    elif vt == "Gene deletion":
        label = "D"
    elif vt == "Gene amplification":
        label = "A"
    elif vt == "Gene fusion":
        label = "F"
    else:
        return None, None
    if vt != "Gene fusion":
        label = f"{label}, {zygosity}"
    return gene, label


def _get_mutation_modality(brca_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for variations in brca_df["Sequence_Variation"]:
        row: dict = {}
        if isinstance(variations, list):
            for var in variations:
                gene = var.get("gene")
                vt = var.get("type")
                z = var.get("zygosity", "Unknown")
                if not gene or not vt:
                    continue
                zs = _zygosity_score(z)
                if vt == "Mutation":
                    row[f"{gene}_M"] = max(row.get(f"{gene}_M", 0), zs)
                elif vt == "Gene deletion":
                    row[f"{gene}_D"] = max(row.get(f"{gene}_D", 0), zs)
                elif vt == "Gene fusion":
                    row[f"{gene}_F"] = 1
        rows.append(row)
    df = pd.DataFrame(rows).fillna(0.0)
    df.index = brca_df.index
    return df


def _one_hot_variations(annotation_df: pd.DataFrame) -> pd.DataFrame:
    gene_var_mat: list[dict] = []
    for variations in annotation_df["Sequence_Variation"]:
        labels: dict = {}
        if isinstance(variations, list):
            for var in variations:
                gene, label = _gene_variation_label(var)
                if gene and label:
                    labels[gene] = label
        gene_var_mat.append(labels)
    var_df = pd.DataFrame(gene_var_mat, index=annotation_df.index).fillna(
        np.nan
    )
    final = annotation_df.reset_index().merge(
        var_df.reset_index(), on="cell_line"
    )
    final.set_index("cell_line", inplace=True)
    return final


def _process_cell_lines(records: dict) -> dict:
    out: dict = {}
    for cl, info in records.items():
        comments = info.get("Comments", "")
        out[cl] = {
            "ID": info["ID"],
            "Site": _extract_site(comments),
            "MS_Status": _extract_msi(comments),
            "Disease": info["Disease"],
            "Sequence_Variation": _extract_sequence_variations(comments),
        }
    return out


def get_cellosaurus_annotations(
    cell_lines: list[str],
    file_dir: Path = features_dir,
    *,
    cache_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], dict[str, str]]:
    """Parse Cellosaurus and return annotation_df, modality_df, cl→SIDM, cl→DepMap maps.

    Parameters
    ----------
    cell_lines : list[str]
        Cell-line identifiers in ``cXXX`` format.
    file_dir : Path
        Directory containing (or to download) ``cellosaurus.txt``.
    cache_dir : Path or None
        If given, cache results as parquet.

    Returns
    -------
    annotation_df : DataFrame
        Cellosaurus annotations per cell line (index = cell_line).
    modality_df : DataFrame
        Mutation-modality features per cell line.
    cl_to_sidm : dict
        Mapping cell_line → CMP SIDM ID.
    cl_to_depmap : dict
        Mapping cell_line → DepMap ACH ID.
    """
    from Bio.ExPASy.cellosaurus import parse as cello_parse

    _annot_p = (
        _cache_path(cache_dir, "cellosaurus_annot") if cache_dir else None
    )
    _mod_p = (
        _cache_path(cache_dir, "cellosaurus_modality") if cache_dir else None
    )
    _ids_p = (
        _cache_path(cache_dir, "cellosaurus_ids", ext="json")
        if cache_dir
        else None
    )
    if _annot_p is not None and _mod_p is not None and _ids_p is not None:
        cached_annot = _load_cache(_annot_p)
        cached_mod = _load_cache(_mod_p)
        cached_ids = _load_cache(_ids_p)
        if (
            cached_annot is not None
            and cached_mod is not None
            and cached_ids is not None
        ):
            print("  [cache hit] Cellosaurus annotations loaded from cache")
            return (
                cached_annot,
                cached_mod,
                cached_ids.get("cl_to_sidm", {}),
                cached_ids.get("cl_to_depmap", {}),
            )

    cello_path = download_cellosaurus_file(file_dir)

    # Build case-insensitive lookup
    _upper_to_orig: dict[str, str] = {}
    for cl in cell_lines:
        _upper_to_orig[cl.lstrip("c").upper()] = cl

    cl_to_sidm: dict[str, str] = {}
    cl_to_depmap: dict[str, str] = {}
    brca_records: dict[str, dict] = {}

    with open(cello_path) as fh:
        for rec in cello_parse(fh):
            rid_norm = rec["ID"].replace("-", "").replace(" ", "").upper()
            matched = _upper_to_orig.get(rid_norm)

            # Hard-coded special cases
            if matched is None and rec["ID"] == "600MPE":
                matched = "cMPE600"
            elif matched is None and rec["ID"] == "Hs 578T":
                matched = "cHs578T"

            # Fallback: synonyms
            if matched is None:
                for syn in (rec.get("SY") or "").split(";"):
                    syn_norm = (
                        syn.strip().replace("-", "").replace(" ", "").upper()
                    )
                    if syn_norm and syn_norm in _upper_to_orig:
                        matched = _upper_to_orig[syn_norm]
                        break

            if matched and matched in cell_lines:
                comments = rec.get("CC", [])
                brca_records[matched] = {
                    "ID": rec["AC"],
                    "Comments": comments,
                    "Disease": (
                        rec["DI"][0].split(";")[2].lstrip(" ")
                        if rec["DI"]
                        else "Unknown"
                    ),
                    "Sex": rec.get("SX", "").strip(),
                    "Age": rec.get("AG", "").strip(),
                    "Category": _extract_category(rec.get("CA", "")),
                    "Doubling_Time_Hours": _extract_doubling_time(comments),
                    "Population": _extract_population(comments),
                    "N_CrossRefs": len(rec.get("DR", [])),
                }
                for db, acc in rec.get("DR", []):
                    if db == "Cell_Model_Passport":
                        cl_to_sidm[matched] = acc
                    elif db == "DepMap":
                        cl_to_depmap[matched] = acc

    # Build annotation and modality dataframes
    annotation_df = pd.DataFrame(_process_cell_lines(brca_records)).T
    annotation_df.index.name = "cell_line"

    extra_cols = [
        "Sex",
        "Age",
        "Category",
        "Doubling_Time_Hours",
        "Population",
        "N_CrossRefs",
    ]
    for col in extra_cols:
        annotation_df[col] = pd.Series(
            {cl: info[col] for cl, info in brca_records.items()},
            name=col,
        )

    modality_df = _get_mutation_modality(annotation_df)
    annotation_df = _one_hot_variations(annotation_df)

    missing = sorted(set(cell_lines) - set(annotation_df.index))
    if missing:
        print(f"  ⚠ Cellosaurus: missing {len(missing)} cell lines: {missing}")
        for cl in missing:
            modality_df.loc[cl] = 0.0

    annotation_df = annotation_df.sort_index()
    modality_df = modality_df.sort_index()

    if _annot_p is not None and _mod_p is not None and _ids_p is not None:
        _save_cache(annotation_df, _annot_p)
        _save_cache(modality_df, _mod_p)
        _save_cache(
            {"cl_to_sidm": cl_to_sidm, "cl_to_depmap": cl_to_depmap}, _ids_p
        )
        print("  [cache saved] Cellosaurus annotations")

    return annotation_df, modality_df, cl_to_sidm, cl_to_depmap


def filter_sequence_variation_modality(
    modality_df: pd.DataFrame,
) -> pd.DataFrame:
    """Drop columns with ≤ 2 non-zero values."""
    sparse = (modality_df > 0).sum(axis=0) <= 2
    return modality_df.loc[:, ~sparse]


# ═══════════════════════════════════════════════════════════════════════
# 2. cBioPortal (CCLE Broad 2019)
# ═══════════════════════════════════════════════════════════════════════


def fetch_cbioportal_mutations(
    cell_lines: list[str],
    *,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch breast-cancer mutation data from cBioPortal CCLE Broad 2019.

    Returns a DataFrame with columns:
        cell_line, gene, keyword, mutationType, proteinChange, variantType
    indexed numerically.  ``cell_line`` uses the ``cXXX`` naming convention.
    """
    p = _cache_path(cache_dir, "cbioportal_mutations") if cache_dir else None
    if p is not None:
        cached = _load_cache(p)
        if cached is not None:
            print("  [cache hit] cBioPortal mutations loaded from cache")
            return cached

    print("  Fetching mutations from cBioPortal (ccle_broad_2019) …")
    url = (
        "https://www.cbioportal.org/api/molecular-profiles/"
        "ccle_broad_2019_mutations/mutations/fetch"
    )
    resp = requests.post(
        url,
        params={"projection": "DETAILED"},
        json={"sampleListId": "ccle_broad_2019_all"},
        headers={
            "accept": "application/json",
            "content-type": "application/json",
        },
        timeout=120,
    )
    resp.raise_for_status()
    all_muts = resp.json()
    brca = [m for m in all_muts if "BREAST" in m.get("sampleId", "")]
    print(f"  Total BREAST mutations: {len(brca)}")

    df = pd.DataFrame(brca)
    df["sampleId"] = df["sampleId"].str.split("_").str[0]
    df["gene"] = df["gene"].apply(
        lambda g: g["hugoGeneSymbol"] if isinstance(g, dict) else g,
    )
    df = df[
        [
            "sampleId",
            "gene",
            "keyword",
            "mutationType",
            "proteinChange",
            "variantType",
        ]
    ].copy()

    # Map sample IDs to cXXX format
    cl_raw = [cl.lstrip("c") for cl in cell_lines]
    df = df[df["sampleId"].isin([c.upper() for c in cl_raw])]
    upper_to_c = {cl.lstrip("c").upper(): cl for cl in cell_lines}
    df["cell_line"] = df["sampleId"].map(upper_to_c)
    df = df.dropna(subset=["cell_line"])

    covered = set(df["cell_line"].unique())
    missing = sorted(set(cell_lines) - covered)
    print(
        f"  Matched {len(covered)}/{len(cell_lines)} cell lines in cBioPortal"
    )
    if missing:
        print(f"  Missing: {missing}")

    if p is not None:
        _save_cache(df, p)
        print("  [cache saved] cBioPortal mutations")
    return df


# ═══════════════════════════════════════════════════════════════════════
# 3. Cell Model Passports (Sanger CMP)
# ═══════════════════════════════════════════════════════════════════════

CMP_BASE = "https://api.cellmodelpassports.sanger.ac.uk"


def fetch_cmp_driver_genes(*, cache_dir: Path | None = None) -> set[str]:
    """Fetch the set of cancer-driver gene symbols from the CMP /genes API."""
    p = (
        _cache_path(cache_dir, "cmp_driver_genes", ext="json")
        if cache_dir
        else None
    )
    if p is not None:
        cached = _load_cache(p)
        if cached is not None:
            print(
                f"  [cache hit] {len(cached)} CMP driver genes loaded from cache"
            )
            return set(cached)

    print("  Fetching cancer driver genes from CMP …")
    genes: set[str] = set()
    page = 1
    while True:
        filt = json.dumps(
            [{"name": "cancer_driver", "op": "eq", "val": "true"}]
        )
        r = requests.get(
            f"{CMP_BASE}/genes",
            params={
                "filter": filt,
                "page[size]": 500,
                "page[number]": page,
                "fields[gene]": "symbol,cancer_driver",
            },
            timeout=30,
        )
        if r.status_code != 200:
            warnings.warn(
                f"CMP /genes failed ({r.status_code}). Using fallback list.",
                stacklevel=2,
            )
            break
        data = r.json().get("data", [])
        if not data:
            break
        for g in data:
            sym = g.get("attributes", {}).get("symbol")
            if sym:
                genes.add(sym)
        page += 1
        time.sleep(0.1)

    if len(genes) < 50:
        genes = _CURATED_DRIVER_GENES.copy()
        print(f"  Using curated fallback list ({len(genes)} genes)")
    else:
        print(f"  → {len(genes)} cancer driver genes from CMP API")

    if p is not None:
        _save_cache(genes, p)
        print("  [cache saved] CMP driver genes")
    return genes


def fetch_cmp_driver_mutations(
    cell_lines: list[str],
    cl_to_sidm: dict[str, str],
    *,
    cache_dir: Path | None = None,
) -> dict[str, list[str]]:
    """Fetch per-cell-line cancer-driver mutation gene lists from CMP.

    Returns dict: cell_line → list of mutated driver gene symbols.
    """
    p = (
        _cache_path(cache_dir, "cmp_driver_mutations", ext="json")
        if cache_dir
        else None
    )
    if p is not None:
        cached = _load_cache(p)
        if cached is not None:
            print("  [cache hit] CMP driver mutations loaded from cache")
            return cached

    print("  Fetching cancer driver mutations from CMP …")
    drivers: dict[str, list[str]] = {}
    matched = 0
    failed: list[str] = []

    for cl in cell_lines:
        sidm = cl_to_sidm.get(cl)
        if sidm is None:
            failed.append(cl)
            continue
        url = f"{CMP_BASE}/models/{sidm}/datasets/cancer_drivers"
        try:
            r = requests.get(
                url,
                params={"page[size]": 0, "include": "gene"},
                timeout=15,
            )
            if r.status_code == 200:
                resp = r.json()
                matched += 1
                gene_map: dict[str, str] = {}
                for inc in resp.get("included", []):
                    if inc.get("type") == "gene":
                        gene_map[inc["id"]] = inc.get("attributes", {}).get(
                            "symbol", ""
                        )
                syms: list[str] = []
                for entry in resp.get("data", []):
                    gid = (
                        entry.get("relationships", {})
                        .get("gene", {})
                        .get("data", {})
                        .get("id", "")
                    )
                    sym = gene_map.get(gid, "")
                    if sym:
                        syms.append(sym)
                if syms:
                    drivers[cl] = syms
            else:
                failed.append(cl)
        except Exception:
            failed.append(cl)
        time.sleep(0.15)

    print(f"  Matched {matched}/{len(cell_lines)} cell lines in CMP")
    if failed:
        print(f"  Not found / no SIDM: {sorted(failed)}")

    if p is not None:
        _save_cache(drivers, p)
        print("  [cache saved] CMP driver mutations")
    return drivers


def fetch_cmp_mutations(
    cell_lines: list[str],
    cl_to_sidm: dict[str, str],
    *,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch full mutation data from CMP for cell lines with SIDM IDs.

    Returns a DataFrame with columns:
        cell_line, gene, effect, proteinChange, coding, vaf,
        cancer_driver, source
    compatible with the cBioPortal DataFrame for comparison.
    """
    p = _cache_path(cache_dir, "cmp_mutations") if cache_dir else None
    if p is not None:
        cached = _load_cache(p)
        if cached is not None:
            print("  [cache hit] CMP mutations loaded from cache")
            return cached

    print("  Fetching mutations from CMP …")
    rows: list[dict[str, Any]] = []
    matched = 0
    failed: list[str] = []

    for cl in cell_lines:
        sidm = cl_to_sidm.get(cl)
        if sidm is None:
            continue

        # Paginate through all mutations for this model
        page = 1
        cl_count = 0
        while True:
            try:
                r = requests.get(
                    f"{CMP_BASE}/models/{sidm}/datasets/mutations",
                    params={
                        "page[size]": 500,
                        "page[number]": page,
                        "include": "gene",
                    },
                    timeout=30,
                )
            except Exception:
                failed.append(cl)
                break

            if r.status_code != 200:
                failed.append(cl)
                break

            resp = r.json()

            # Build gene-ID → symbol map from included records
            gene_map: dict[str, str] = {}
            for inc in resp.get("included", []):
                if inc.get("type") == "gene":
                    gene_map[inc["id"]] = inc.get("attributes", {}).get(
                        "symbol", ""
                    )

            data = resp.get("data", [])
            if not data:
                break

            for entry in data:
                attrs = entry.get("attributes", {})
                gid = attrs.get("gene_id", "")
                sym = gene_map.get(gid, "")
                rows.append(
                    {
                        "cell_line": cl,
                        "gene": sym,
                        "effect": attrs.get("effect", ""),
                        "proteinChange": attrs.get("protein", ""),
                        "coding": attrs.get("coding"),
                        "vaf": attrs.get("vaf"),
                        "cancer_driver": attrs.get("cancer_driver", False),
                        "source": attrs.get("source", ""),
                    }
                )
                cl_count += 1

            total = resp.get("meta", {}).get("count", 0)
            if page * 500 >= total:
                break
            page += 1
            time.sleep(0.05)

        if cl_count > 0:
            matched += 1
        time.sleep(0.15)

    df = pd.DataFrame(rows)
    print(f"  Matched {matched}/{len(cell_lines)} cell lines in CMP")
    print(f"  Total mutation records: {len(df)}")
    if failed:
        print(f"  Failed: {sorted(set(failed))}")

    if p is not None:
        _save_cache(df, p)
        print("  [cache saved] CMP mutations")
    return df


# ═══════════════════════════════════════════════════════════════════════
# 4. Subtype annotations  (from annotation_utils.py)
# ═══════════════════════════════════════════════════════════════════════


def get_subtype_annotations(
    samples: list[str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Return PAM50 and Luminal/Basal subtype dicts for given samples."""
    pam50 = {
        cl: subtypes_tognetti[cl]["PAM50"]
        for cl in samples
        if cl in subtypes_tognetti
    }
    lb = {
        cl: subtypes_tognetti[cl]["Luminal/Basal"]
        for cl in samples
        if cl in subtypes_tognetti
    }
    return pam50, lb


def load_marcotte_subtypes(samples: list[str]) -> pd.DataFrame:
    """Load and normalise Marcotte molecular subtypes, aligned to *samples*."""
    df = pd.read_csv(basedir / "cell_line_subtypes.txt", delimiter="\t")
    df["cell_line"] = df["cell_line"].apply(lambda x: "c" + str(x).upper())
    df.cell_line.replace("cHS578T", "cHs578T", inplace=True)
    df.cell_line.replace("c600MPE", "cMPE600", inplace=True)
    df.loc[df["cell_line"] == "cDU4475", "subtype_intrinsic"] = "Basal"
    df.sort_values("cell_line", inplace=True)
    df.set_index("cell_line", inplace=True)
    idx = [s for s in samples if s in df.index]
    return df.loc[idx]


def onehot_intrinsic(samples: list[str]) -> pd.DataFrame:
    """One-hot encode the Marcotte *subtype_intrinsic* column."""
    expected = [
        "intr_Basal",
        "intr_CL",
        "intr_HER2",
        "intr_LuminalA",
        "intr_LuminalB",
        "intr_Normal",
    ]
    if not samples:
        return pd.DataFrame(columns=expected)
    df = load_marcotte_subtypes(samples)
    ser = df["subtype_intrinsic"].astype(str).fillna("Unknown")
    X = pd.get_dummies(ser, prefix="intr", dtype=float)
    X = X.reindex(pd.Index([s for s in samples if s in X.index]))
    for col in expected:
        if col not in X.columns:
            X[col] = 0.0
    return X[expected]


def onehot_lb(samples: list[str]) -> pd.DataFrame:
    """Collapse intrinsic subtypes to Luminal / Basal and one-hot encode."""
    expected = ["lb_Basal", "lb_Luminal", "lb_Normal"]
    if not samples:
        return pd.DataFrame(columns=expected)
    df = load_marcotte_subtypes(samples).copy()
    lb = df["subtype_intrinsic"].astype(str)
    lb = lb.replace(["LuminalA", "LuminalB", "HER2"], "Luminal")
    lb = lb.replace(["CL"], "Basal")
    lb = lb.fillna("Unknown")
    X = pd.get_dummies(lb, prefix="lb", dtype=float)
    X = X.reindex(pd.Index([s for s in samples if s in X.index]))
    for col in expected:
        if col not in X.columns:
            X[col] = 0.0
    return X[expected]


# ═══════════════════════════════════════════════════════════════════════
# 5. Proteomics annotations  (from annotation_utils.py)
# ═══════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════
# 6. Curated fallback driver gene list
# ═══════════════════════════════════════════════════════════════════════

_CURATED_DRIVER_GENES: set[str] = {
    "EGFR",
    "ERBB2",
    "ERBB3",
    "KRAS",
    "NRAS",
    "HRAS",
    "BRAF",
    "RAF1",
    "MAP2K1",
    "MAP2K2",
    "MAP2K4",
    "MAP3K1",
    "MAPK1",
    "PIK3CA",
    "PIK3R1",
    "AKT1",
    "AKT2",
    "PTEN",
    "MTOR",
    "TSC1",
    "TSC2",
    "TP53",
    "RB1",
    "CDKN2A",
    "CDKN2B",
    "CDKN1B",
    "BRCA1",
    "BRCA2",
    "ATM",
    "ATR",
    "CHEK2",
    "PALB2",
    "BAP1",
    "CCND1",
    "CDK4",
    "CDK6",
    "MDM2",
    "MDM4",
    "BCL2",
    "MCL1",
    "ARID1A",
    "ARID1B",
    "KMT2C",
    "KMT2D",
    "SETD2",
    "NSD1",
    "SMAD4",
    "CREBBP",
    "EP300",
    "TET2",
    "APC",
    "CTNNB1",
    "NOTCH1",
    "NOTCH2",
    "FBXW7",
    "MYC",
    "MYCN",
    "GATA3",
    "FOXA1",
    "ESR1",
    "SF3B1",
    "RUNX1",
    "NF1",
    "NF2",
    "PTCH1",
    "SMARCA4",
    "SMARCB1",
    "FGFR1",
    "FGFR2",
    "FGFR3",
    "MET",
    "ALK",
    "RET",
    "ROS1",
    "KIT",
    "PDGFRA",
    "FLT3",
    "JAK2",
    "JAK1",
    "MLH1",
    "MSH2",
    "MSH6",
    "PMS2",
    "POLE",
    "IDH1",
    "IDH2",
    "VHL",
    "WT1",
    "STAG2",
    "SPOP",
    "PPP2R1A",
    "CDH1",
    "PBRM1",
    "KDM6A",
    "TERT",
    "RAD21",
}


# ═══════════════════════════════════════════════════════════════════════
# 7. High-level convenience:  get_all_annotations / build_feature_matrix
# ═══════════════════════════════════════════════════════════════════════


def get_all_annotations(
    cell_lines: list[str],
    *,
    cache_dir: Path | None = None,
    file_dir: Path = features_dir,
) -> dict[str, Any]:
    """Fetch and return all annotation blocks as a dict.

    Keys
    ----
    cello_annot, cello_modality : DataFrames from Cellosaurus
    cl_to_sidm, cl_to_depmap   : ID mappings from Cellosaurus DR cross-refs
    cbio_df                     : cBioPortal CCLE mutation table
    cmp_driver_genes            : set of CMP cancer driver gene symbols
    cmp_driver_mutations        : dict cell_line → list of mutated driver genes
    subtypes_pam50, subtypes_lb : subtype dicts
    """
    print("═" * 70)
    print("Fetching all cell-line annotations …")
    print("═" * 70)

    # 1. Cellosaurus
    print("\n[1/4] Cellosaurus …")
    (
        cello_annot,
        cello_mod,
        cl_to_sidm,
        cl_to_depmap,
    ) = get_cellosaurus_annotations(
        cell_lines,
        file_dir=file_dir,
        cache_dir=cache_dir,
    )

    # 2. cBioPortal
    print("\n[2/4] cBioPortal CCLE …")
    cbio_df = fetch_cbioportal_mutations(cell_lines, cache_dir=cache_dir)

    # 3. CMP
    print("\n[3/4] Cell Model Passports …")
    drv_genes = fetch_cmp_driver_genes(cache_dir=cache_dir)
    drv_muts = fetch_cmp_driver_mutations(
        cell_lines,
        cl_to_sidm,
        cache_dir=cache_dir,
    )

    # 4. Subtypes
    print("\n[4/4] Subtypes …")
    pam50, lb = get_subtype_annotations(cell_lines)
    print(f"  PAM50: {len(pam50)} cell lines, LB: {len(lb)} cell lines")

    print("\n" + "═" * 70)
    print("All annotations fetched.")
    print("═" * 70)

    return {
        "cello_annot": cello_annot,
        "cello_modality": cello_mod,
        "cl_to_sidm": cl_to_sidm,
        "cl_to_depmap": cl_to_depmap,
        "cbio_df": cbio_df,
        "cmp_driver_genes": drv_genes,
        "cmp_driver_mutations": drv_muts,
        "subtypes_pam50": pam50,
        "subtypes_lb": lb,
    }
