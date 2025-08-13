import numpy as np
import pandas as pd
import petab
import synapseclient

from common import subtypes_tognetti
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
