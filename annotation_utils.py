import numpy as np
import pandas as pd
import petab
import synapseclient

from common import subtypes_tognetti, basedir
from dmm.config_options import Conf
from dmm.feature_selection import load_data
from util import load_petab_base_files


def generate_proteomics_annotations(
    protein_markers: list[str],
    uniprot_ids: list[str],
    samples_list: list[str],
    model: str = "EGFR_MAPK",
    data: str = "dream_cytof",
    impute: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """
    Generates subtype, HER2 CyTOF, and proteomics marker annotations for a given model and dataset.

    Parameters
    ----------
    protein_markers : list of str
        List of protein markers to include in the log2FC annotations.
    uniprot_ids : list of str
        List of UniProt IDs corresponding to the protein markers.
    samples_list : list
        List of all training and validation sample IDs.
    model : str
        Model name to use for the configuration.
    data : str
        Data name to use for the configuration.
    impute : bool
        Whether to impute missing values in the CyTOF data.

    Returns
    -------
    erbb2_cytof_init : pd.DataFrame
        Raw HER2 cytof measurements.
    erbb2_cytof_init_log2fc : pd.Series
        Log2 fold change HER2 levels (normalized to reference).
    proteomarkers_log2fc_annotations : dict
        Dictionary of log2FC DataFrames for relevant protein markers.
    """
    conf = Conf(model=model, data=data)
    petab_base_files = load_petab_base_files(conf)
    del petab_base_files["condition_table"]

    # Get pristine proteomics data from Synapse
    syn = synapseclient.Synapse()
    syn.login()
    df_proteomics = pd.read_csv(syn.get("syn20690774").path, index_col=[0])
    df_proteomics[petab.v1.OBSERVABLE_ID] = df_proteomics.index

    proteomarkers_log2fc_annotations = {}
    for marker, uniprot in zip(protein_markers, uniprot_ids):
        try:
            sub_df = (
                df_proteomics.loc[uniprot][["cell_line", "log2FC"]]
                .reset_index()
                .drop(columns="Protein")
            )
            sub_df["cell_line"] = "c" + sub_df["cell_line"].astype(str)
            sub_df.set_index("cell_line", inplace=True)

            available = [s for s in samples_list if s in sub_df.index]
            sub_df = sub_df.loc[available]

            missing = [s for s in samples_list if s not in sub_df.index]
            sub_df = pd.concat(
                [sub_df, pd.DataFrame({"log2FC": 0}, index=missing)], axis=0
            ).sort_index()
            proteomarkers_log2fc_annotations[marker] = sub_df
        except Exception as e:
            print(f"Marker {marker} not found with exception {e}.")

    # Get CyTOF data for ERBB2/HER2
    overall_cytof_init, _, _ = load_data(
        contextualization="cytof_init",
        samples=samples_list,
        features=None,
        **petab_base_files,
        impute=impute,
    )

    def extract_raw_marker(overall_df, marker, rename_to):
        df = overall_df[[marker]].rename(columns={marker: rename_to})
        df = df.rename_axis("cell_line").sort_index().reset_index()
        return df.set_index("cell_line")

    def compute_log2fc(overall_df, marker, normal_lines, rename_to):
        ref = overall_df.loc[normal_lines, marker].mean()
        overall_df[marker] = np.log2(overall_df[marker] / ref)
        overall_df.loc[normal_lines, marker] = 0
        log2fc = (
            overall_df[marker]
            .rename(rename_to)
            .rename_axis("cell_line")
            .sort_index()
        )
        return log2fc.reset_index().set_index("cell_line")

    # Define normal reference lines
    normal_cell_lines = ["c184A1", "c184B5", "cMCF10A", "cMCF12A"]

    # Extract raw marker levels
    phospho_annotations = {
        "pHER2_raw": extract_raw_marker(
            overall_cytof_init, "p.HER2", "pHER2_raw"
        ),
        "pHER2_log2FC": compute_log2fc(
            overall_cytof_init, "p.HER2", normal_cell_lines, "pHER2_log2FC"
        ),
        "p.p38_raw": extract_raw_marker(
            overall_cytof_init, "p.p38", "p.p38_raw"
        ),
        "p.p38_log2FC": compute_log2fc(
            overall_cytof_init, "p.p38", normal_cell_lines, "p.p38_log2FC"
        ),
        "pERK_raw": extract_raw_marker(
            overall_cytof_init, "p.ERK", "pERK_raw"
        ),
        "pERK_log2FC": compute_log2fc(
            overall_cytof_init, "p.ERK", normal_cell_lines, "pERK_log2FC"
        ),
    }

    return phospho_annotations, proteomarkers_log2fc_annotations


def generate_subtype_annotations(
    samples_list: list[str],
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Generate PAM50 and Luminal/Basal subtype annotations for a list of samples, i.e. cell-lines.

    Parameters
    ----------
    samples_list : list of str
        List of sample identifiers (cell-line names).

    Returns
    -------
    subtypes_pam50 : dict
        Mapping of cell_line to PAM50 subtype.
    subtypes_lb : dict
        Mapping of cell_line to Luminal/Basal subtype.
    """
    subtypes_pam50 = {
        cell_line: subtypes_tognetti[cell_line]["PAM50"]
        for cell_line in samples_list
        if cell_line in subtypes_tognetti
    }
    subtypes_lb = {
        cell_line: subtypes_tognetti[cell_line]["Luminal/Basal"]
        for cell_line in samples_list
        if cell_line in subtypes_tognetti
    }
    return subtypes_pam50, subtypes_lb


def load_marcotte_subtypes(samples: list[str]) -> pd.DataFrame:
    """
    Load and normalize Marcotte molecular subtypes to index by 'c<CELL>' IDs,
    aligned and subset to the provided samples list and order.
    """
    subtypes_marcotte = pd.read_csv(basedir / "cell_line_subtypes.txt", delimiter="\t")
    subtypes_marcotte["cell_line"] = subtypes_marcotte["cell_line"].apply(lambda x: "c" + str(x).upper())
    # Canonicalize a couple of known IDs to match our data
    subtypes_marcotte.cell_line.replace("cHS578T", "cHs578T", inplace=True)
    subtypes_marcotte.cell_line.replace("c600MPE", "cMPE600", inplace=True)
    # Explicit fix DU4475 - appears as NA in subtype_intrinsic, but is classified elsewhere as triple-negative-basal
    subtypes_marcotte.loc[subtypes_marcotte["cell_line"] == "cDU4475", "subtype_intrinsic"] = "Basal"
    subtypes_marcotte.sort_values(by="cell_line", inplace=True)
    subtypes_marcotte.set_index("cell_line", inplace=True)
    # Reindex to requested sample order (drop missing)
    idx = [s for s in samples if s in subtypes_marcotte.index]
    subtypes_marcotte = subtypes_marcotte.loc[idx]
    return subtypes_marcotte


def _onehot_intrinsic(samples: list[str]) -> pd.DataFrame:
    """
    One-hot encode the 'subtype_intrinsic' column for the given samples.
    Columns will be named like 'intr_LuminalA', 'intr_Basal', etc.
    """

    expected_cols = ["intr_Basal", "intr_CL", "intr_HER2",
                     "intr_LuminalA", "intr_LuminalB", "intr_Normal"]

    # Early exit: no samples → return empty with correct columns
    if len(samples) == 0:
        return pd.DataFrame(
            columns=expected_cols,
            index=pd.Index([], name=petab.v1.PREEQUILIBRATION_CONDITION_ID)
        )

    df = load_marcotte_subtypes(samples)
    ser = df["subtype_intrinsic"].astype(str).fillna("Unknown")
    X = pd.get_dummies(ser, prefix="intr", dtype=float)

    # Preserve order; drop any absent samples
    X = X.reindex(
        pd.Index(
            [s for s in samples if s in X.index],
            name=petab.v1.PREEQUILIBRATION_CONDITION_ID
        )
    )

    if len(X) == 1:
        for subtype in ["LuminalA", "LuminalB", "HER2", "CL", "Basal", "Normal"]:
            if f"intr_{subtype}" not in X.columns:
                X[f"intr_{subtype}"] = 0.0

    # Ensure consistent one-hot-encoded feature ordering
    X = X[["intr_Basal", "intr_CL", "intr_HER2", "intr_LuminalA", "intr_LuminalB", "intr_Normal"]]
    return X


def _onehot_lb(samples: list[str]) -> pd.DataFrame:
    """
    Collapse intrinsic subtypes to Luminal/Basal buckets, then one-hot encode.
      LuminalA/LuminalB/HER2 → Luminal
      CL → Basal
    """

    expected_cols = ["lb_Basal", "lb_Luminal", "lb_Normal"]

    # Early exit: no samples → return empty with correct columns
    if len(samples) == 0:
        return pd.DataFrame(
            columns=expected_cols,
            index=pd.Index([], name=petab.v1.PREEQUILIBRATION_CONDITION_ID)
        )

    df = load_marcotte_subtypes(samples).copy()
    lb = df["subtype_intrinsic"].astype(str)
    lb = lb.replace(["LuminalA", "LuminalB"], "Luminal")
    lb = lb.replace(["CL"], "Basal")
    lb = lb.replace(["HER2"], "Luminal")
    lb = lb.fillna("Unknown")
    X = pd.get_dummies(lb, prefix="lb", dtype=float)

    X = X.reindex(
        pd.Index(
            [s for s in samples if s in X.index],
            name=petab.v1.PREEQUILIBRATION_CONDITION_ID
        )
    )

    if len(X) == 1:
        for subtype in ["Luminal", "Basal", "Normal"]:
            if f"lb_{subtype}" not in X.columns:
                X[f"lb_{subtype}"] = 0.0

    # Ensure consistent one-hot-encoded feature ordering
    X = X[["lb_Basal", "lb_Luminal", "lb_Normal"]]
    return X


def annotate_pca_embeddings_with_metadata(
    pca_embedding_df: pd.DataFrame,
    subtypes_pam50: dict[str, str],
    subtypes_lb: dict[str, str],
    pher2_annotations: dict[str, pd.DataFrame],
    proteomarker_log2fc_annotations: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Annotates PCA-reduced embeddings with subtype, HER2 (CyTOF), and proteomics log2FC data.

    Parameters
    ----------
    pca_embedding_df : pd.DataFrame
        PCA-reduced embeddings with 'cell_line' as a column.
    subtypes_pam50 : dict
        Mapping from cell_line to PAM50 subtype.
    subtypes_lb : dict
        Mapping from cell_line to Luminal/Basal subtype.
    pher2_annotations : dict
        Dictionary containing HER2 CyTOF init annotations, including raw and log2FC values.
    proteomarker_log2fc_annotations : dict
        Dictionary of log2FC DataFrames for each protein marker.

    Returns
    -------
    pd.DataFrame
        Annotated PCA dataframe.
    """
    annotated_df = pca_embedding_df.copy()

    # Subtype annotations
    annotated_df["subtype_pam50"] = annotated_df["cell_line"].map(
        subtypes_pam50
    )
    annotated_df["subtype_lb"] = annotated_df["cell_line"].map(subtypes_lb)

    # HER2 CyTOF
    if (
        "pHER2_log2FC" in pher2_annotations
        and not pher2_annotations["pHER2_log2FC"].empty
    ):
        annotated_df = annotated_df.merge(
            pher2_annotations["pHER2_log2FC"].reset_index(),
            on="cell_line",
            how="left",
        )
    if (
        "pHER2_raw" in pher2_annotations
        and not pher2_annotations["pHER2_raw"].empty
    ):
        annotated_df = annotated_df.merge(
            pher2_annotations["pHER2_raw"].reset_index(),
            on="cell_line",
            how="left",
        )

    # Merge proteomics log2FC values
    for marker, marker_df in proteomarker_log2fc_annotations.items():
        df = marker_df.copy()
        # Ensure 'cell_line' is the index or a column
        if df.index.name != "cell_line":
            df.index.name = "cell_line"
        df_renamed = df.reset_index().rename(
            columns={"log2FC": f"{marker}_log2FC"}
        )
        annotated_df = annotated_df.merge(
            df_renamed, on="cell_line", how="left"
        )

    return annotated_df
