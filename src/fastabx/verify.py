"""Safety checks for the public API."""

from collections.abc import Sequence
from itertools import chain

import polars as pl
from torch import Tensor

NDIM = 3
INVALID_COLUMN_SUFFIX = ("_a", "_b", "_x")
INVALID_COLUMN_NAMES = {"index", "score", "__group", "__lookup"}


class LabelReservedNameError(ValueError):
    """Invalid name for a condition."""

    def __init__(self, name: str) -> None:
        super().__init__(f"Invalid label: {name}. This name is reserved for internal computations.")


class LabelSuffixError(ValueError):
    """Invalid suffix for a condition."""

    def __init__(self, name: str) -> None:
        super().__init__(f"Invalid label: {name}. Cannot end by _a, _b, or _x.")


class EmptyDataPointsError(ValueError):
    """Empty data points in the dataset."""

    max_print_size: int = 10

    def __init__(self, empty: list[str]) -> None:
        super().__init__(
            f"{len(empty)} empty elements were found in the dataset (with indices ["
            + ", ".join(empty[: self.max_print_size])
            + (", ..." if len(empty) > self.max_print_size else "")
            + "])"
        )


def verify_empty_datapoints(indices: dict[int, tuple[int, int]]) -> None:
    """Check if some datapoints are empty by checking the start and end indices."""
    empty = []
    for key, (start, end) in indices.items():
        if end <= start:
            empty.append(str(key))
    if empty:
        raise EmptyDataPointsError(empty)


def verify_task_conditions(conditions: list[str]) -> None:
    """Conditions should be unique strings."""
    if len(conditions) != len(set(conditions)):
        raise ValueError("Conditions should not contain duplicates")
    for cond in conditions:
        if not isinstance(cond, str):
            raise TypeError("Conditions should be strings")


def verify_dataset_labels(df: pl.DataFrame) -> None:
    """Check the column labels."""
    for col in df.schema:
        if col in INVALID_COLUMN_NAMES:
            raise LabelReservedNameError(col)
        if col.endswith(INVALID_COLUMN_SUFFIX):
            raise LabelSuffixError(col)


def verify_subsampler_params(*sizes: int | None, seed: int) -> None:
    """All sizes must be positive integers."""
    if not all(isinstance(s, int) and s > 1 for s in sizes if s is not None):
        raise TypeError("sizes should be positive integers")
    if not isinstance(seed, int):
        raise TypeError("seed should be an integer")


def verify_cell(a_sa: tuple[Tensor, Tensor], b_sb: tuple[Tensor, Tensor], x_sx: tuple[Tensor, Tensor]) -> None:
    """Assert the integrity of a cell."""
    (a, sa), (b, sb), (x, sx) = a_sa, b_sb, x_sx
    if not a.ndim == b.ndim == x.ndim == NDIM:
        raise ValueError("A, B, and X should be tensors with 3 dimensions")
    if not a.size(2) == b.size(2) == x.size(2):
        raise ValueError("A, B, and X should have the same feature dimension")
    if not (a.size(0) == sa.size(0) and b.size(0) == sb.size(0) and x.size(0) == sx.size(0)):
        raise ValueError("Invalid size specification")


def format_score_levels(levels: Sequence[tuple[str, ...] | str]) -> list[tuple[str, ...]]:
    """Put all the levels in tuples."""
    formatted: list[tuple[str, ...]] = []
    for level in levels:
        if isinstance(level, str):
            formatted.append((level,))
        elif isinstance(level, tuple) and all(isinstance(x, str) for x in level):
            formatted.append(level)
        else:
            raise ValueError("`levels` should be list[tuple[str, ...] | str]")
    return formatted


def verify_score_levels(columns: list[str], levels: list[tuple[str, ...]]) -> None:
    """Levels should be unique columns of the DataFrame."""
    all_levels = list(chain.from_iterable(levels))
    unique_levels = set(all_levels)
    if len(all_levels) != len(unique_levels):
        raise ValueError("levels should not contain duplicates")
    if not unique_levels.issubset(set(columns)):
        raise ValueError("levels should be columns of the DataFrame")
