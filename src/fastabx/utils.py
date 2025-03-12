"""Various utilities."""

import os
import sys
from dataclasses import dataclass, field
from typing import Literal

import polars as pl
import torch


def default_engine() -> Literal["cpu", "gpu"]:
    """Engine is 'gpu' if available, else 'cpu'."""
    os.environ["RMM_DEBUG_LOG_FILE"] = os.devnull  # Avoid creation of rmm_log.txt logs by RAPIDS Memory Manager
    try:
        pl.LazyFrame().collect(engine="gpu")
    except (ModuleNotFoundError, pl.exceptions.ComputeError):
        return "cpu"
    else:
        return "gpu"


@dataclass(frozen=True)
class Environment:
    """Store global environment variables."""

    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    engine: Literal["cpu", "gpu"] = field(default_factory=default_engine)


def load_dtw_extension() -> None:
    """Load the DTW extension.

    On Linux and Windows, we check that PyTorch has been installed with the correct CUDA version.
    """
    cuda_version = "12.4"
    if sys.platform in ["linux", "win32"] and torch.version.cuda != cuda_version:
        msg = (
            f"On Linux and Windows, the DTW extension requires PyTorch with CUDA {cuda_version}. "
            "It it not compatible with other CUDA versions, or with the CPU only version of PyTorch, "
            "even if you wanted to only use the CPU backend of the DTW. "
        )
        raise ImportError(msg)
    from . import _C  # type: ignore[attr-defined] # noqa: F401
