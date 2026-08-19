#!/usr/bin/env python3
"""
Scan FCS files in a directory, match them to single-cell data from Synapse,
and compute scale and offset parameters for each marker.

Outputs a CSV summary with scale/offset per file and average per cell line.

FCS files can be downloaded from: https://data.mendeley.com/datasets/gvh2vtg86r/1
Outputs have been uploaded to: https://www.synapse.org/Synapse:syn71996733
"""


import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytometry as pm
import synapseclient

# Mapping from cell line name to synapse ID
CELL_LINE_SYNAPSE = {
    "184A1": "syn20613594",
    "BT20": "syn20613595",
    "BT474": "syn20613596",
    "BT549": "syn20613597",
    "CAL148": "syn20613598",
    "CAL51": "syn20613599",
    "CAL851": "syn20613600",
    "DU4475": "syn20613601",
    "EFM192A": "syn20613660",
    "EVSAT": "syn20613665",
    "HBL100": "syn20613668",
    "HCC1187": "syn20613674",
    "HCC1395": "syn20613687",
    "HCC1419": "syn20613696",
    "HCC1500": "syn20613702",
    "HCC1569": "syn20613708",
    "HCC1937": "syn20613719",
    "HCC1954": "syn20613739",
    "HCC2157": "syn20613793",
    "HCC2185": "syn20613802",
    "HCC3153": "syn20613814",
    "HCC38": "syn20613821",
    "HCC70": "syn20613832",
    "HDQP1": "syn20613849",
    "JIMT1": "syn20613865",
    "MCF10A": "syn20613880",
    "MCF10F": "syn20613911",
    "MCF7": "syn20613920",
    "MDAMB134VI": "syn20613935",
    "MDAMB157": "syn20613939",
    "MDAMB175VII": "syn20613943",
    "MDAMB361": "syn20613962",
    "MDAMB415": "syn20613975",
    "MDAMB453": "syn20613988",
    "MDAkb2": "syn20613930",
    "MFM223": "syn20613995",
    "MPE600": "syn20614008",
    "MX1": "syn20614033",
    "OCUBM": "syn20614045",
    "T47D": "syn20614052",
    "UACC812": "syn20614063",
    "UACC893": "syn20614074",
    "ZR7530": "syn20614085",
    "184B5": "syn20631041",
    "BT483": "syn20631043",
    "HCC1428": "syn20631044",
    "HCC1806": "syn20631045",
    "HCC202": "syn20631047",
    "Hs578T": "syn20631048",
    "MCF12A": "syn20631049",
    "MDAMB231": "syn20631050",
    "MDAMB468": "syn20631060",
    "SKBR3": "syn20631061",
    "UACC3199": "syn20631062",
    "ZR751": "syn20631063",
    "AU565": "syn20631033",
    "EFM19": "syn20631035",
    "HCC2218": "syn20631036",
    "LY2": "syn20631037",
    "MACLS2": "syn20631038",
    "MDAMB436": "syn20631039",
}

# Mapping from FCS channel names to single-cell column names
CHANNEL_MAP = {
    "127I_IdU": "IdU",
    "139La_p-Creb": "p.CREB",
    "141Pr_p-Stat5": "p.STAT5",
    "142Nd_p-Src": "p.SRC",
    "143Nd_p-Fak": "p.FAK",
    "144Nd_p-Mek": "p.MEK",
    "145Nd_p-Mapkapk2": "p.MAPKAPK2",
    "146Nd_p-S6K": "p.S6K",
    "147Sm_p-MAP2K3": "p.MAP2K3",
    "148Sm_p-Stat1": "p.STAT1",
    "149Sm_p-p53": "p.p53",
    "150Sm_p-NFkB": "p.NFkB",
    "151Eu_p-p38": "p.p38",
    "152Gd_p-AMPK": "p.AMPK",
    "153Eu_p-Akt473": "p.Akt.Ser473.",
    "154Gd_p-Erk": "p.ERK",
    "155Gd_p-Her2": "p.HER2",
    "156Gd_CyclinB": "CyclinB",
    "158Gd_p-Gsk3b": "p.GSK3b",
    "159Tb_GAPDH": "GAPDH",
    "160Gd_p-MKK3-MKK6": "p.MKK3.MKK6",
    "161Dy_p-PDPK1": "p.PDPK1",
    "162Dy_p-BTK": "p.BTK",
    "163Dy_p-p90RSK": "p.p90RSK",
    "164Dy_p-Smad23": "p.SMAD23",
    "165Ho_b-Catenin": "b.CATENIN",
    "166Er_p-Stat3": "p.STAT3",
    "167Er_p-JNK": "p.JNK",
    "168Er_Ki-67": "Ki.67",
    "169Tm_p-PLCg2": "p.PLCg2",
    "170Yb_p-H3": "p.H3",
    "171Yb_p-S6": "p.S6",
    "172Yb_cleavedCas": "cleavedCas",
    "173Yb_p-MKK4": "p.MKK4",
    "174Yb_p-Akt308": "p.AKT.Thr308.",
    "175Lu_p-Rb": "p.RB",
    "176Yb_p-4EBP1": "p.4EBP1",
}


def scalar_affine_fit(A, B):
    """
    Scalar affine fit B = scale * A + offset using min/max.
    Returns (scale, offset).
    """
    if len(A) == 0 or len(B) == 0:
        return np.nan, np.nan
    a_min, a_max = float(np.min(A)), float(np.max(A))
    b_min, b_max = float(np.min(B)), float(np.max(B))
    if a_max == a_min:
        return np.nan, np.nan
    scale = (b_max - b_min) / (a_max - a_min)
    offset = b_min - scale * a_min
    return float(scale), float(offset)


def find_fcs_files(root_dir):
    root = Path(root_dir)
    fcs_files = []
    for subdir in root.iterdir():
        if subdir.is_dir():
            fcs_files.extend(subdir.glob("*.fcs"))
    return fcs_files


def read_fcs_basic(path):
    """
    Read FCS file into a pandas DataFrame using pytometry.
    Returns (df, channel_names) where df is DataFrame of events.
    """
    path = str(path)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Transforming to str index")
        adata = pm.io.read_fcs(path)
    # explicitly convert var_names to list of strings to avoid ImplicitModificationWarning
    channel_names = [str(v) for v in adata.var_names]
    df = pd.DataFrame(adata.X, columns=channel_names)
    return df, channel_names


def extract_metadata_from_path(fpath):
    """
    Heuristic extraction of metadata from filename.
    Returns dict with keys: cell_line, timepoint, condition, timecourse_id, file_prefix
    Returns None if file has ambiguous matches (e.g., third token is not A or B).

    Expected pattern: {fileID}_{cell_line}_{timecourse_id}_{timepoint}_{condition}_...
    - token 0: fileID (e.g., a313, c494 - saved as file_prefix)
    - token 1: cell_line
    - token 2: timecourse_id (A or B)
    - token 3: timepoint (in minutes, no unit)
    - token 4: condition (inhibitor: iPI3K/imTOR/iEGFR/iPKC/iMEK/egf)
    """
    p = Path(fpath)
    stem = p.stem
    # sometimes files end with .fcs.fcs because of double extension; remove trailing .fcs
    stem = re.sub(r"(\.fcs)+$", "", stem, flags=re.IGNORECASE)
    tokens = stem[1:].split("_")

    if len(tokens) < 5:
        return None

    # token 0 is fileID prefix (e.g., a313, c494)
    file_prefix = tokens[0]

    # token 1 is cell_line
    cell_line = tokens[1]

    # token 2 must be timecourse_id (A or B)
    if tokens[2].upper() not in ("A", "B"):
        return None
    timecourse = tokens[2].upper()

    # token 3 is timepoint (in minutes)
    timepoint = tokens[3]

    # token 4 is condition (inhibitor only or 'full')
    condition = tokens[4]

    # validate condition: must be valid inhibitor or 'full'/'egf', skipping 'imtor'
    valid_conditions = ("ipi3k", "iegfr", "ipkc", "imek", "egf", "full")
    if condition.lower() not in valid_conditions:
        return None

    # if condition is 'full', timepoint should be 0
    if condition.lower() == "full":
        timepoint = "0"

    # validate timepoint: must be numeric
    if not re.match(r"^\d+$", timepoint):
        return None

    return {
        "cell_line": cell_line or "",
        "timepoint": timepoint or "",
        "condition": condition or "",
        "timecourse_id": timecourse or "",
        "file_prefix": file_prefix or "",
    }


def load_single_cell_data_for_cell_line(
    cell_line, syn=None, file_id_table=None
):
    """
    Load single-cell data for a specific cell line using CELL_LINE_SYNAPSE mapping.
    Returns (counts, df_sc, file_meta) where:
      - counts: dict mapping fileID -> cell count
      - df_sc: DataFrame with single-cell data for this cell line
      - file_meta: dict mapping fileID -> dict with 'treatment', 'time', 'time_course'
    """
    if cell_line not in CELL_LINE_SYNAPSE:
        print(f"  Warning: no synapse ID for cell line {cell_line}")
        return {}, pd.DataFrame(), {}

    synapse_id = CELL_LINE_SYNAPSE[cell_line]

    if syn is None:
        syn = synapseclient.Synapse()
        syn.login()

    try:
        f = syn.get(synapse_id)
        df = pd.read_csv(f.path)
        counts = df.groupby("fileID").size().to_dict()

        # build file_meta from the single-cell data
        file_meta = {}
        for fid in df["fileID"].unique():
            df_fid = df[df["fileID"] == fid]
            # get treatment, time from the data (should be same for all cells in a fileID)
            treatment = (
                df_fid["treatment"].iloc[0]
                if "treatment" in df_fid.columns
                else None
            )
            time = df_fid["time"].iloc[0] if "time" in df_fid.columns else None
            # get time_course from file_id_table if available
            time_course = None
            if file_id_table is not None and fid in file_id_table.index:
                time_course = file_id_table.loc[fid, "time_course"]
            file_meta[fid] = {
                "treatment": treatment,
                "time": time,
                "time_course": time_course,
            }

        return counts, df, file_meta
    except Exception as e:
        print(f"  Warning: could not load {synapse_id} for {cell_line}: {e}")
        return {}, pd.DataFrame(), {}


# Gold standard synapse file to check for inexact matches
GOLD_STANDARD_FILES = ["syn20631273"]


def try_gold_standard_files(n_events, syn, file_id_table):
    """
    Try to find a matching fileID in gold standard files.
    Returns (fileid, df_sc) if found, (None, None) otherwise.
    """
    for gs_synapse_id in GOLD_STANDARD_FILES:
        try:
            f = syn.get(gs_synapse_id)
            df = pd.read_csv(f.path)
            counts = df.groupby("fileID").size().to_dict()
            exact_matches = [fid for fid, c in counts.items() if c == n_events]
            if len(exact_matches) == 1:
                return exact_matches[0], df
        except Exception as e:
            print(
                f"    Warning: could not load gold standard {gs_synapse_id}: {e}"
            )
    return None, None


def main():
    fcs_dir = "./cytof/fcs"
    out_csv = "fcs_scan_summary.csv"

    fcs_files = find_fcs_files(fcs_dir)
    print(f"Found {len(fcs_files)} .fcs files under {fcs_dir}")

    # group fcs files by cell line
    fcs_by_cell_line = {}
    for fpath in fcs_files:
        meta = extract_metadata_from_path(str(fpath))
        if meta is None:
            print(f"Skipping {fpath} (ambiguous or missing timecourse_id)")
            continue
        cell_line = meta["cell_line"]
        if cell_line not in fcs_by_cell_line:
            fcs_by_cell_line[cell_line] = []
        fcs_by_cell_line[cell_line].append((fpath, meta))

    # login to synapse once
    syn = synapseclient.Synapse()
    syn.login()

    # load file_id_table for time_course info
    file_id_table = pd.read_csv(
        syn.get("syn20631269").path, index_col=0
    ).set_index("fileID")

    rows = []
    skipped = []
    for cell_line, files_and_meta in fcs_by_cell_line.items():
        print(
            f"Processing cell line: {cell_line} ({len(files_and_meta)} files)"
        )

        # load single-cell data for this cell line
        counts, df_sc, file_meta = load_single_cell_data_for_cell_line(
            cell_line, syn=syn, file_id_table=file_id_table
        )

        for fpath, meta in files_and_meta:
            fpath = str(fpath)
            print("  Processing", fpath)
            try:
                df_fcs, channels = read_fcs_basic(fpath)
            except Exception as e:
                print("    Failed to read FCS:", e)
                skipped.append(
                    {"fcs_path": fpath, "reason": f"Failed to read FCS: {e}"}
                )
                continue
            n_events = len(df_fcs)

            # identify fileID by matching counts and metadata
            fileid = None
            if len(counts) > 0:
                # find all fileIDs with exact cell count match
                exact_matches = [
                    fid for fid, c in counts.items() if c == n_events
                ]

                # further filter by metadata if multiple matches
                if len(exact_matches) > 1:
                    # map FCS condition to treatment name in single-cell data
                    fcs_condition = meta["condition"].lower()
                    fcs_timepoint = int(meta["timepoint"])
                    fcs_timecourse = meta["timecourse_id"]  # 'A' or 'B'

                    filtered_matches = []
                    for fid in exact_matches:
                        fm = file_meta.get(fid, {})
                        sc_treatment = fm.get("treatment", "")
                        sc_time = fm.get("time")
                        sc_time_course = fm.get("time_course", "")

                        # check treatment match (condition)
                        if (
                            sc_treatment
                            and sc_treatment.lower() != fcs_condition
                        ):
                            continue
                        # check time match (timepoint)
                        if (
                            sc_time is not None
                            and int(sc_time) != fcs_timepoint
                        ):
                            continue
                        # check time_course match (A/B)
                        if (
                            sc_time_course
                            and sc_time_course.upper() != fcs_timecourse
                        ):
                            continue
                        filtered_matches.append(fid)
                    exact_matches = filtered_matches

                if len(exact_matches) == 1:
                    fileid = exact_matches[0]
                elif len(exact_matches) > 1:
                    print(
                        f"    Warning: multiple exact matches after metadata filtering (n_events={n_events}). fileIDs={exact_matches}. Skipping."
                    )
                    skipped.append(
                        {
                            "fcs_path": fpath,
                            "reason": f"Multiple exact matches after filtering: {exact_matches}",
                            "n_events": n_events,
                        }
                    )
                    continue
                elif (
                    len([fid for fid, c in counts.items() if c == n_events])
                    == 0
                ):
                    # no exact cell count match at all, try gold standard files
                    fileid, df_sc_fallback = try_gold_standard_files(
                        n_events, syn, file_id_table
                    )
                    if fileid is not None:
                        print(
                            f"    Found match in gold standard file: fileID={fileid}"
                        )
                        df_sc = df_sc_fallback
                        file_meta[fileid] = {}  # no metadata for gold standard
                    else:
                        diffs = {
                            fid: abs(n_events - c) for fid, c in counts.items()
                        }
                        closest = min(diffs, key=diffs.get)
                        print(
                            f"    Warning: no exact cell count match (n_events={n_events}). Closest fileID={closest} (diff={diffs[closest]}). Skipping."
                        )
                        skipped.append(
                            {
                                "fcs_path": fpath,
                                "reason": f"No exact match, closest={closest} (diff={diffs[closest]})",
                                "n_events": n_events,
                            }
                        )
                        continue
                else:
                    # had exact matches but metadata filtering removed all of them
                    print(
                        f"    Warning: exact matches filtered out by metadata (n_events={n_events}). Skipping."
                    )
                    skipped.append(
                        {
                            "fcs_path": fpath,
                            "reason": "Exact matches filtered out by metadata",
                            "n_events": n_events,
                        }
                    )
                    continue
            else:
                # no counts for cell line, skip
                print(
                    f"    Warning: no counts available for cell line {cell_line}. Skipping."
                )
                skipped.append(
                    {
                        "fcs_path": fpath,
                        "reason": f"No counts for cell line {cell_line}",
                        "n_events": n_events,
                    }
                )
                continue

            # compute scale and offset for each marker
            channel_scales = {}
            channel_offsets = {}
            if len(df_fcs) > 0 and fileid is not None and len(df_sc) > 0:
                # get single-cell data for this fileID
                df_sc_file = df_sc[df_sc["fileID"] == fileid]
                if len(df_sc_file) > 0:
                    # compute scale/offset for each channel in CHANNEL_MAP
                    sc_cols = set(df_sc_file.columns)
                    for ch, sc_col in CHANNEL_MAP.items():
                        if ch in channels and sc_col in sc_cols:
                            A = np.asinh(df_fcs[ch].values + 1)
                            B = df_sc_file[sc_col].values
                            scale, offset = scalar_affine_fit(A, B)
                            channel_scales[sc_col] = scale
                            channel_offsets[sc_col] = offset

            row = {
                "fcs_path": fpath,
                "cell_line": meta.get("cell_line", ""),
                "timepoint": meta.get("timepoint", ""),
                "condition": meta.get("condition", ""),
                "timecourse_id": meta.get("timecourse_id", ""),
                "fileID": int(fileid) if fileid is not None else None,
                "n_events": int(n_events),
            }
            # add per-channel scale and offset columns
            for sc_col in CHANNEL_MAP.values():
                row[f"scale_{sc_col}"] = channel_scales.get(sc_col, np.nan)
                row[f"offset_{sc_col}"] = channel_offsets.get(sc_col, np.nan)
            rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)
    print("Wrote", out_csv)

    # compute average scales/offsets per cell line
    scale_cols = [c for c in out_df.columns if c.startswith("scale_")]
    offset_cols = [c for c in out_df.columns if c.startswith("offset_")]

    if scale_cols:
        avg_df = out_df.groupby("cell_line")[scale_cols + offset_cols].agg(
            ["mean", "std", "count"]
        )
        avg_csv = "fcs_scan_avg_scale_offset.csv"
        avg_df.to_csv(avg_csv)
        print(f"Wrote average scales/offsets per cell line to {avg_csv}")

    # define two groups of cell lines with distinct scales/offsets
    GROUP2_CELL_LINES = [
        "BT483",
        "HCC1187",
        "HCC1419",
        "HCC1569",
        "HCC1806",
        "HCC1937",
        "HCC70",
        "MCF12A",
        "MDAMB231",
        "MDAMB361",
        "MDAMB453",
        "MDAMB468",
        "MDAkb2",
        "T47D",
        "UACC893",
    ]
    # group 1 is all other cell lines
    GROUP1_CELL_LINES = [
        cl
        for cl in out_df["cell_line"].unique()
        if cl not in GROUP2_CELL_LINES
    ]

    # cell lines without count data that need transformed CSVs
    MISSING_CELL_LINES = ["CAL120", "CAMA1", "HCC1143", "KPL1", "ZR75B"]

    # compute group averages
    group1_df = out_df[out_df["cell_line"].isin(GROUP1_CELL_LINES)]
    group2_df = out_df[out_df["cell_line"].isin(GROUP2_CELL_LINES)]

    group1_avg = group1_df[scale_cols + offset_cols].mean()
    group2_avg = group2_df[scale_cols + offset_cols].mean()

    # save group averages
    group_avg_df = pd.DataFrame(
        {
            "group1_mean": group1_avg,
            "group2_mean": group2_avg,
        }
    )
    group_avg_df.to_csv("fcs_scan_group_averages.csv")
    print("Wrote group averages to fcs_scan_group_averages.csv")

    # use group 1 averages to transform FCS data for missing cell lines
    # extract scale and offset per marker from group 2 averages
    group1_scales = {
        col.replace("scale_", ""): group1_avg[col] for col in scale_cols
    }
    group1_offsets = {
        col.replace("offset_", ""): group1_avg[col] for col in offset_cols
    }

    # process FCS files for missing cell lines
    for cell_line in MISSING_CELL_LINES:
        # find FCS files for this cell line
        cell_line_fcs = [
            (f, m)
            for f, m in [
                (str(f), extract_metadata_from_path(str(f))) for f in fcs_files
            ]
            if m is not None and m["cell_line"] == cell_line
        ]

        if not cell_line_fcs:
            print(f"No FCS files found for {cell_line}")
            continue

        print(
            f"Transforming FCS files for {cell_line} ({len(cell_line_fcs)} files)"
        )

        all_transformed = []
        for fpath, meta in cell_line_fcs:
            try:
                df_fcs, channels = read_fcs_basic(fpath)
            except Exception as e:
                print(f"  Failed to read {fpath}: {e}")
                continue

            # create transformed dataframe in df_sc style (vectorized)
            # normalize treatment name: replace 'egf' with 'EGF'
            treatment = meta["condition"]
            if treatment.lower() == "egf":
                treatment = "EGF"
            df_transformed = pd.DataFrame(
                {
                    "cellID": range(len(df_fcs)),
                    "cell_line": cell_line,
                    "treatment": treatment,
                    "time": int(meta["timepoint"]),
                    "time_course": meta[
                        "timecourse_id"
                    ],  # A or B from filename
                    "date": "",  # unknown for missing cell lines
                    "fileID": -1,  # placeholder, no real fileID
                }
            )

            # apply transformation: B = scale * asinh(fcs_value + 1) + offset
            for ch, sc_col in CHANNEL_MAP.items():
                if ch in channels:
                    A = np.asinh(df_fcs[ch].values + 1)
                    scale = group1_scales.get(sc_col, 1.0)
                    offset = group1_offsets.get(sc_col, 0.0)
                    if np.isnan(scale) or np.isnan(offset):
                        scale, offset = 1.0, 0.0
                    df_transformed[sc_col] = scale * A + offset

            all_transformed.append(df_transformed)

        if all_transformed:
            df_cell_line = pd.concat(all_transformed, ignore_index=True)
            out_path = f"fcs_transformed_{cell_line}.csv"
            df_cell_line.to_csv(out_path, index=False)
            print(f"  Wrote {len(df_cell_line)} rows to {out_path}")

    # write skipped files log
    if skipped:
        skipped_csv = "fcs_scan_skipped.csv"
        skipped_df = pd.DataFrame(skipped)
        skipped_df.to_csv(skipped_csv, index=False)
        print(f"Wrote {len(skipped)} skipped files to {skipped_csv}")


if __name__ == "__main__":
    main()
