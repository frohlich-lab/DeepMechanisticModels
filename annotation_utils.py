"""
Backward-compatible shim – all functionality lives in cell_line_annotations.py.

No active consumers remain; this file is kept only for safety.
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
    onehot_intrinsic,
    onehot_lb,
)

# Backward-compatible aliases for the old private names
_onehot_intrinsic = onehot_intrinsic  # noqa: F401
_onehot_lb = onehot_lb  # noqa: F401
