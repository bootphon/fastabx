"""Smoke-test an installed distribution; copy this script outside the checkout before running it."""

import math
import sys
from importlib.metadata import version
from pathlib import Path

import polars as pl
import torch

import fastabx
from fastabx import Dataset, InMemoryAccessor, Score, Task


def main() -> None:
    """Verify the installed package, typing marker, and a minimal DTW evaluation."""
    package = Path(fastabx.__file__).resolve().parent
    if not package.is_relative_to(Path(sys.prefix).resolve()):
        msg = f"Expected fastabx inside the active environment, imported {package}"
        raise RuntimeError(msg)
    if not (package / "py.typed").is_file():
        msg = "The installed distribution is missing py.typed"
        raise RuntimeError(msg)

    # Two identical two-frame tokens per category exercise both distance and DTW kernels.
    features = torch.tensor([[0.0], [0.1], [0.0], [0.1], [10.0], [10.1], [10.0], [10.1]])
    dataset = Dataset(
        labels=pl.DataFrame({"phone": ["a", "a", "b", "b"]}),
        accessor=InMemoryAccessor({i: (2 * i, 2 * i + 2) for i in range(4)}, features, torch.device("cpu")),
    )
    score = Score(Task(dataset, on="phone"), "euclidean", progress=False).collapse()
    if not math.isfinite(score) or score != 0:
        msg = f"Expected zero ABX error for separated categories, got {score}"
        raise RuntimeError(msg)
    print(f"fastabx {version('fastabx')}: installed DTW evaluation passed ({package})")


if __name__ == "__main__":
    main()
