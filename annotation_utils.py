import pandas as pd
import re
import urllib.request

from Bio.ExPASy.cellosaurus import parse
from common import features_dir
from cytof import get_samples
from pathlib import Path


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
