import numpy as np
import pandas as pd
import petab
import re
import synapseclient
import urllib.request

from Bio.ExPASy.cellosaurus import parse
from common import features_dir, subtypes_tognetti
from dmm.config_options import Conf
from dmm.feature_selection import load_data
from cytof import get_samples
from pathlib import Path
from util import load_petab_base_files


def download_cellosaurus_file(file_dir: Path, url="https://ftp.expasy.org/databases/cellosaurus/cellosaurus.txt", filename="cellosaurus.txt"):
    """Download the Cellosaurus annotation file if not present in the given directory (file_dir)."""
    file_path = file_dir / filename

    if not file_path.is_file():
        print(f"File '{filename}' not found in {features_dir}. Downloading from {url}...")
        try:
            features_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
            urllib.request.urlretrieve(url, file_path)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading file: {e}")
    else:
        print(f"File '{filename}' already exists in {features_dir}. Skipping download.")


def extract_site(comments_list):
    """Extract disease site preceded by 'Derived from site:' until the first occurrence of 'UBERON=' ontology reference or end of string."""
    for comment in comments_list:
        match = re.search(r'Derived from site:\s*(.+?)(?:\s*UBERON=|$)', comment)
        if match:
            return match.group(1).strip().rstrip(";")
    return 'Unknown'


def extract_msi(comments_list):
    """Extract microsatellite instability status from a list of comments or return 'Unknown'."""
    for comment in comments_list:
        match = re.search(r'Microsatellite instability:\s*(.*?)(?:\s*\||$)', comment)
        if match:
            status_match = re.search(r'(Stable \(MSS\)|Instable \(MSI-low\)|Instable \(MSI-high\))', match.group(1))
            if status_match:
                return status_match.group(0)
    return 'Unknown'


def process_cell_lines(cell_line_dict):
    """Process each cell line's comments to extract site and MSI status."""
    new_dict = {cell_line: {} for cell_line in cell_line_dict.keys()}
    for cell_line, info in cell_line_dict.items():
        comments = info.get('Comments', '')
        info['Site'] = extract_site(comments)
        info['MSI_Status'] = extract_msi(comments)
        new_dict[cell_line] = {
            "ID": info["ID"],
            "Site": info["Site"],
            "MS_Status": info["MSI_Status"],
            "Disease": info["Disease"]
        }
    return new_dict


def get_cell_line_cellosaurus_annotations(file_dir: Path):
    # Download annotations
    download_cellosaurus_file(file_dir)
    cellosaurus_filepath = file_dir / "cellosaurus.txt"
    cell_lines = [cell_line.lstrip("c") for cell_line in get_samples("dream_cytof")]
    brca_records = {}
    with open(cellosaurus_filepath) as handle:  # https://ftp.expasy.org/databases/cellosaurus/cellosaurus.txt
        records = parse(handle)
        for record in records:
            if (record["ID"].replace("-", "").upper() in cell_lines) or (record["ID"] == "600MPE"):
                # Save records with our cell-line notation (e.g., cMCF7, cMPE600)
                dict_id = "c" + record["ID"].replace("-", "").upper()
                if record["ID"] == "600MPE":
                    dict_id = "cMPE600"
                brca_records[dict_id] = {
                    "ID": record["AC"],
                    "Comments": record["CC"],
                    "Disease": record["DI"][0].split(";")[2].lstrip(" ") if record["DI"] else "Unknown"
                }
    if len(brca_records.keys()) != len(cell_lines):
        print(f"Warning: Number of cell lines in Cellosaurus annotations ({len(brca_records.keys())}) does not match the number of cell lines in the DREAM challenge ({len(cell_lines)}).")

    # Process annotations re. MS status and site and return dataframe
    annotation_df = pd.DataFrame(process_cell_lines(brca_records)).T
    annotation_df.index.name = "cell_line"
    return annotation_df


def generate_proteomics_annotations(
        protein_markers: list[str],
        uniprot_ids: list[str],
        samples_list: list[str],
        model: str = "EGFR_MAPK",
        data: str = "dream_cytof",
        impute: bool = False
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
            sub_df = df_proteomics.loc[uniprot][
                ["cell_line", "log2FC"]
            ].reset_index().drop(columns="Protein")
            sub_df["cell_line"] = "c" + sub_df["cell_line"].astype(str)
            sub_df.set_index("cell_line", inplace=True)

            available = [s for s in samples_list if s in sub_df.index]
            sub_df = sub_df.loc[available]

            missing = [s for s in samples_list if s not in sub_df.index]
            sub_df = pd.concat(
                [
                    sub_df,
                    pd.DataFrame({ "log2FC": 0 }, index=missing)
                ],
                axis=0
            ).sort_index()
            proteomarkers_log2fc_annotations[marker] = sub_df
        except Exception as e:
            print(f"Marker {marker} not found with exception {e}.")

    # Get CyTOF data for ERBB2/HER2
    overall_cytof_init, _ = load_data(
        contextualization="cytof_init",
        samples=samples_list,
        features=None,
        **petab_base_files,
        impute=impute
    )

    erbb2_cytof_init = overall_cytof_init["pERBB2_Y1248_obs"].rename_axis("cell_line").sort_index().reset_index()
    erbb2_cytof_init.set_index("cell_line", inplace=True)
    erbb2_cytof_init.rename(columns={"pERBB2_Y1248_obs": "pHER2_raw"}, inplace=True)

    # Compute log2 fold change for HER2 levels with respect to reference cell-lines
    normal_cell_lines = ["c184A1", "c184B5", "cMCF10A", "cMCF12A"]
    ref = overall_cytof_init.loc[normal_cell_lines, "pERBB2_Y1248_obs"].mean()
    overall_cytof_init["pERBB2_Y1248_obs"] = np.log2(overall_cytof_init["pERBB2_Y1248_obs"] / ref)
    overall_cytof_init.loc[normal_cell_lines, "pERBB2_Y1248_obs"] = 0
    erbb2_cytof_init_log2fc = overall_cytof_init["pERBB2_Y1248_obs"].rename_axis("cell_line").sort_index()
    erbb2_cytof_init_log2fc.rename("pHER2_log2FC", inplace=True).reset_index()

    pher2_annotations = {"pHER2_raw": erbb2_cytof_init, "pHER2_log2FC": erbb2_cytof_init_log2fc}

    return pher2_annotations, proteomarkers_log2fc_annotations


def generate_subtype_annotations(
        samples_list: list[str]
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
    proteomarker_log2fc_annotations: dict[str, pd.DataFrame]
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
    annotated_df["subtype_pam50"] = annotated_df["cell_line"].map(subtypes_pam50)
    annotated_df["subtype_lb"] = annotated_df["cell_line"].map(subtypes_lb)

    # HER2 CyTOF
    if "pHER2_log2FC" in pher2_annotations and not pher2_annotations["pHER2_log2FC"].empty:
        annotated_df = annotated_df.merge(
            pher2_annotations["pHER2_log2FC"].reset_index(),
            on="cell_line",
            how="left"
        )
    if "pHER2_raw" in pher2_annotations and not pher2_annotations["pHER2_raw"].empty:
        annotated_df = annotated_df.merge(
            pher2_annotations["pHER2_raw"].reset_index(),
            on="cell_line",
            how="left"
        )

    # Merge proteomics log2FC values
    for marker, marker_df in proteomarker_log2fc_annotations.items():
        df = marker_df.copy()
        # Ensure 'cell_line' is the index or a column
        if df.index.name != "cell_line":
            df.index.name = "cell_line"
        df_renamed = df.reset_index().rename(columns={"log2FC": f"{marker}_log2FC"})
        annotated_df = annotated_df.merge(df_renamed, on="cell_line", how="left")

    return annotated_df
