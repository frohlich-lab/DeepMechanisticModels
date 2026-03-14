"""
Backward-compatible shim – all functionality lives in cell_line_annotations.py.

Existing consumers can continue to do:
    from annotation_utils import load_marcotte_subtypes
    from annotation_utils import _onehot_intrinsic, _onehot_lb
    from annotation_utils import generate_subtype_annotations
    from annotation_utils import generate_proteomics_annotations
    from annotation_utils import annotate_pca_embeddings_with_metadata
"""
import warnings as _warnings

_warnings.warn(
    "annotation_utils is deprecated; use cell_line_annotations instead.",
    DeprecationWarning,
    stacklevel=2,
)

from cell_line_annotations import (  # noqa: F401, E402
    annotate_pca_embeddings_with_metadata,
    generate_proteomics_annotations,
    load_marcotte_subtypes,
)
