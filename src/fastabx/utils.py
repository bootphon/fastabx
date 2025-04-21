"""Various utilities."""

import os
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


def with_librilight_bug() -> bool:
    """Whether to reproduce the results from LibriLight ABX or not."""
    return os.getenv("FASTABX_WITH_LIBRILIGHT_BUG", "0") == "1"
