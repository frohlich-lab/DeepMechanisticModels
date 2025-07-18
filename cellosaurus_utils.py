import numpy as np
import pandas as pd
import re
import urllib.request

from Bio.ExPASy.cellosaurus import parse
from common import features_dir
from cytof import get_samples
from pathlib import Path


def download_cellosaurus_file(
        file_dir: Path,
        url="https://ftp.expasy.org/databases/cellosaurus/cellosaurus.txt",
        filename="cellosaurus.txt"
):
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


def extract_sequence_variations(comments_list):
    """Extract structured sequence variation annotations from Cellosaurus comments."""
    sv_data = []

    for comment in comments_list:
        if not comment.startswith("Sequence variation:"):
            continue

        entry = comment.replace("Sequence variation:", "").strip()

        # Basic fields
        sv_type_match = re.match(r"^(Gene deletion|Mutation|Gene fusion|Gene amplification|Other):?", entry)
        sv_type = sv_type_match.group(1) if sv_type_match else "Unknown"

        # Gene symbol and HGNC
        hgnc_match = re.search(r"HGNC:?;?\s*HGNC:(\d+);?\s*([A-Z0-9\-]+);?", entry)
        hgnc_id = hgnc_match.group(1) if hgnc_match else None
        gene = hgnc_match.group(2) if hgnc_match else None

        # Zygosity
        zygosity_match = re.search(r"Zygosity=([A-Za-z]+)", entry)
        zygosity = zygosity_match.group(1) if zygosity_match else None

        # Protein-level annotation (e.g. p.Val600Glu)
        protein_change_match = re.search(r"(p\.[A-Za-z0-9]+)", entry)
        protein_change = protein_change_match.group(1) if protein_change_match else None

        # cDNA-level annotation (e.g. c.1799T>A)
        cdna_change_match = re.search(r"(c\.[0-9]+[A-Z]>[A-Z])", entry)
        cdna_change = cdna_change_match.group(1) if cdna_change_match else None

        # ClinVar ID
        clinvar_match = re.search(r"ClinVar=([A-Za-z0-9_]+)", entry)
        clinvar = clinvar_match.group(1) if clinvar_match else None

        sv_data.append({
            "type": sv_type,
            "gene": gene,
            "hgnc_id": hgnc_id,
            "zygosity": zygosity,
            "protein_change": protein_change,
            "cdna_change": cdna_change,
            "clinvar_id": clinvar,
        })

    return sv_data if sv_data else [{"type": "Not Found"}]


def process_cell_lines(cell_line_dict):
    """Process each cell line's comments to extract site and MSI status."""
    new_dict = {cell_line: {} for cell_line in cell_line_dict.keys()}
    for cell_line, info in cell_line_dict.items():
        comments = info.get('Comments', '')
        info['Site'] = extract_site(comments)
        info['MSI_Status'] = extract_msi(comments)
        info["Sequence_Variation"] = extract_sequence_variations(comments)
        new_dict[cell_line] = {
            "ID": info["ID"],
            "Site": info["Site"],
            "MS_Status": info["MSI_Status"],
            "Disease": info["Disease"],
            "Sequence_Variation": info["Sequence_Variation"],
        }
    return new_dict


def gene_variation_label(var):
    var_type = var.get("type")
    gene = var.get("gene")
    zygosity = var.get("zygosity") or "Unknown"
    protein = var.get("protein_change")

    if not gene or not var_type:
        return None, None

    if var_type == "Mutation":
        label = f"M ({protein})" if protein else "M"
    elif var_type == "Gene deletion":
        label = "D"
    elif var_type == "Gene amplification":
        label = "A"
    elif var_type == "Gene fusion":
        label = "F"
    else:
        print(f"Found unaccounted-for sequence variation type: {var_type}")
        return None

    if var_type != "Gene fusion":
        final_label = f"{label}, {zygosity}"
    else:
        final_label = label
    return gene, final_label



def zygosity_score(zygosity):
    if zygosity == "Homozygous":
        return 1.0
    elif zygosity == "Heterozygous":
        return 0.5
    else:
        return 0.5  # impute unknown


def get_mutation_modality(brca_df):
    # Initialize rows
    expanded_rows = []

    for variations in brca_df["Sequence_Variation"]:
        row = {}
        if isinstance(variations, list):
            for var in variations:
                gene = var.get("gene")
                var_type = var.get("type")
                zygosity = var.get("zygosity", "Unknown")

                if not gene or not var_type:
                    continue

                z_score = zygosity_score(zygosity)

                # Assign values to specific columns
                if var_type == "Mutation":
                    row[f"{gene}_M"] = max(row.get(f"{gene}_M", 0), z_score)
                elif var_type == "Gene deletion":
                    row[f"{gene}_D"] = max(row.get(f"{gene}_D", 0), z_score)
                elif var_type == "Gene fusion":
                    row[f"{gene}_F"] = 1  # binary: 0/1
        expanded_rows.append(row)

    # Construct DataFrame
    mutation_modality_df = pd.DataFrame(expanded_rows).fillna(0.0)
    mutation_modality_df.index = brca_df.index  # Preserve original cell line indexing
    return mutation_modality_df


def one_hot_variations(annotation_df):
    # Process sequence variation per row
    gene_variation_matrix = []

    for variations in annotation_df["Sequence_Variation"]:
        gene_labels = {}
        if isinstance(variations, list):
            for var in variations:
                gene, label = gene_variation_label(var)
                if gene and label:
                    gene_labels[gene] = label  # If repeated, last one will stay
        gene_variation_matrix.append(gene_labels)
    # Convert to DataFrame
    variation_df = pd.DataFrame(gene_variation_matrix, index=annotation_df.index).fillna(np.nan)

    # Merge with original df
    final_annotation_df = annotation_df.reset_index().merge(
        variation_df.reset_index(), on="cell_line"
    )
    final_annotation_df.set_index("cell_line", inplace=True)
    return final_annotation_df


def get_cell_line_cellosaurus_annotations(file_dir: Path):
    # Download annotations
    download_cellosaurus_file(file_dir)
    cellosaurus_filepath = file_dir / "cellosaurus.txt"
    cell_lines = [cell_line.lstrip("c") for cell_line in get_samples("dream_cytof")]
    brca_records = {}
    with (open(cellosaurus_filepath) as handle):  # https://ftp.expasy.org/databases/cellosaurus/cellosaurus.txt
        records = parse(handle)
        for record in records:
            if (
                    (record["ID"].replace("-", "").upper() in cell_lines)
                    or (record["ID"] == "600MPE")
                    or (record["ID"] == "Hs 578T")
                ):
                # Save records with our cell-line notation (e.g., cMCF7, cMPE600)
                dict_id = "c" + record["ID"].replace("-", "").upper()
                if record["ID"] == "600MPE":
                    dict_id = "cMPE600"
                elif record["ID"] == "Hs 578T":
                    dict_id = "cHs578T"
                brca_records[dict_id] = {
                    "ID": record["AC"],
                    "Comments": record["CC"],
                    "Disease": record["DI"][0].split(";")[2].lstrip(" ") if record["DI"] else "Unknown"
                }

    # Process annotations re. MS status and site and return dataframe
    annotation_df = pd.DataFrame(process_cell_lines(brca_records)).T
    annotation_df.index.name = "cell_line"
    modality_df = get_mutation_modality(annotation_df)
    annotation_df = one_hot_variations(annotation_df)

    if len(brca_records.keys()) != len(cell_lines):
        print(
            f"Warning: Number of cell lines in Cellosaurus annotations ({len(brca_records.keys())}) "
            f"does not match the number of cell lines in the DREAM challenge ({len(cell_lines)})."
        )
        # Identify missing cell line
        missing_cell_lines = sorted(
            set(get_samples("dream_cytof")) - set(annotation_df.index)
        )
        if missing_cell_lines:
            print(f"Missing in Cellosaurus: {missing_cell_lines}")
            # Add zero rows to modality_df for missing cell lines
            for cl in missing_cell_lines:
                modality_df.loc[cl] = 0.0  # All zeros

    # Sort both DataFrames by index (cell-lines)
    annotation_df = annotation_df.sort_index()
    modality_df = modality_df.sort_index()

    return annotation_df, modality_df


def filter_sequence_variation_modality(modality_df):
    # Identify and drop columns with only one non-zero value
    sparse_columns = (modality_df > 0).sum(axis=0) <= 2
    modality_df_filtered = modality_df.loc[:, ~sparse_columns]
    return modality_df_filtered