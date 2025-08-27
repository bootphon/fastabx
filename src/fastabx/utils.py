"""Various utilities."""

import os

MIN_CELLS_FOR_TQDM = 50


def load_dtw_extension() -> None:
    """Load the DTW extension.

    We check that PyTorch has been installed with the correct version.
    """
    import torch  # noqa: F401, PLC0415

    from . import _C  # ty: ignore[unresolved-import] # noqa: F401, PLC0415


def with_librilight_bug() -> bool:
    """Whether to reproduce the results from LibriLight ABX or not."""
    return os.getenv("FASTABX_WITH_LIBRILIGHT_BUG", "0") == "1"
