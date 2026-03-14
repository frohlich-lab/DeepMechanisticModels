"""
Backward-compatible shim – all functionality lives in cell_line_annotations.py.

Existing consumers can continue to do:
    from cellosaurus_utils import get_cell_line_cellosaurus_annotations
    from cellosaurus_utils import filter_sequence_variation_modality
"""
import warnings as _warnings

_warnings.warn(
    "cellosaurus_utils is deprecated; use cell_line_annotations instead.",
    DeprecationWarning,
    stacklevel=2,
)

from pathlib import Path

from cell_line_annotations import (  # noqa: F401, E402
    download_cellosaurus_file,
    filter_sequence_variation_modality,
)
from common import features_dir  # noqa: F401 – re-exported for consumers
from cytof import get_samples


def get_cell_line_cellosaurus_annotations(file_dir: Path = features_dir):
    """Backward-compatible wrapper around the new consolidated module."""
    from cell_line_annotations import get_cellosaurus_annotations

    cell_lines = list(get_samples("dream_cytof"))
    annotation_df, modality_df, _sidm, _depmap = get_cellosaurus_annotations(
        cell_lines,
        file_dir=file_dir,
    )
    return annotation_df, modality_df
