"""Various utilities."""

import importlib.resources
import json
import os
import sys

import torch

MIN_CELLS_FOR_TQDM = 50


class PyTorchVersionError(ImportError):
    """PyTorch version mismatch."""

    def __init__(self, actual: str, expected: str) -> None:
        super().__init__(
            f"The DTW extension requires PyTorch {expected}, but you have {actual}. "
            "Please install the correct version of PyTorch, or install the wheel of fastabx "
            "from the GitHub release page that matches your PyTorch version."
        )


def load_dtw_extension() -> None:
    """Load the DTW extension.

    We check that PyTorch has been installed with the correct version.
    """
    expected = json.loads((importlib.resources.files("fastabx") / "torch_version.json").read_text())[sys.platform]
    if torch.__version__ != expected:
        raise PyTorchVersionError(torch.__version__, expected)
    from . import _C  # ty: ignore[unresolved-import] # noqa: F401, PLC0415


def with_librilight_bug() -> bool:
    """Whether to reproduce the results from LibriLight ABX or not."""
    return os.getenv("FASTABX_WITH_LIBRILIGHT_BUG", "0") == "1"
