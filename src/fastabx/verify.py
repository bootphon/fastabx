"""Safety checks for the public API."""

import enum
from collections.abc import Sequence
from itertools import chain

import polars as pl
from torch import Tensor

NDIM = 3
INVALID_COLUMN_SUFFIX = ("_a", "_b", "_x")
INVALID_COLUMN_NAMES = {"index", "score", "size", "is_valid", "__group", "__lookup"}


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


class DuplicateConditionsError(ValueError):
    """Duplicate conditions found."""


class InputTypeError(TypeError):
    """All conditions should be strings."""

    def __init__(self, expected: type, received: type) -> None:
        super().__init__(f"Should be an instance of {expected}, not {received}")


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
        raise DuplicateConditionsError
    for cond in conditions:
        if not isinstance(cond, str):
            raise InputTypeError(str, type(cond))


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
        msg = "sizes should be positive integers"
        raise TypeError(msg)
    if not isinstance(seed, int):
        raise InputTypeError(int, type(seed))


class CellErrorType(enum.Enum):
    """All types of errors coming from a ``Cell``."""

    NDIM = enum.auto()
    FEATURE_DIM = enum.auto()
    SIZE = enum.auto()


class InvalidCellError(ValueError):
    """The cell is not built correctly."""

    def __init__(self, error_type: CellErrorType) -> None:
        match error_type:
            case CellErrorType.NDIM:
                msg = "A, B, and X should be tensors with 3 dimensions"
            case CellErrorType.FEATURE_DIM:
                msg = "A, B, and X should have the same feature dimension"
            case CellErrorType.SIZE:
                msg = "Invalid size specification"
            case _:
                msg = None
        super().__init__(msg)


def verify_cell(a_sa: tuple[Tensor, Tensor], b_sb: tuple[Tensor, Tensor], x_sx: tuple[Tensor, Tensor]) -> None:
    """Assert the integrity of a cell."""
    (a, sa), (b, sb), (x, sx) = a_sa, b_sb, x_sx
    if not a.ndim == b.ndim == x.ndim == NDIM:
        raise InvalidCellError(CellErrorType.NDIM)
    if not a.size(2) == b.size(2) == x.size(2):
        raise InvalidCellError(CellErrorType.FEATURE_DIM)
    if not (a.size(0) == sa.size(0) and b.size(0) == sb.size(0) and x.size(0) == sx.size(0)):
        raise InvalidCellError(CellErrorType.SIZE)


class LevelsErrorType(enum.Enum):
    """All types of errors coming that can arise from 'levels'."""

    FORMAT = enum.auto()
    DUPLICATES = enum.auto()
    COLUMNS = enum.auto()


class InvalidLevelsError(ValueError):
    """Levels are not well formatted."""

    def __init__(self, error_type: LevelsErrorType) -> None:
        match error_type:
            case LevelsErrorType.FORMAT:
                msg = "'levels' should be list[tuple[str, ...] | str]"
            case LevelsErrorType.DUPLICATES:
                msg = "levels should not contain duplicates"
            case LevelsErrorType.COLUMNS:
                msg = "levels should be columns of the DataFrame"
            case _:
                msg = None
        super().__init__(msg)


def format_score_levels(levels: Sequence[tuple[str, ...] | str]) -> list[tuple[str, ...]]:
    """Put all the levels in tuples."""
    formatted: list[tuple[str, ...]] = []
    for level in levels:
        if isinstance(level, str):
            formatted.append((level,))
        elif isinstance(level, tuple) and all(isinstance(x, str) for x in level):
            formatted.append(level)
        else:
            raise InvalidLevelsError(LevelsErrorType.FORMAT)
    return formatted


def verify_score_levels(columns: list[str], levels: list[tuple[str, ...]]) -> None:
    """Levels should be unique columns of the DataFrame."""
    all_levels = list(chain.from_iterable(levels))
    unique_levels = set(all_levels)
    if len(all_levels) != len(unique_levels):
        raise InvalidLevelsError(LevelsErrorType.DUPLICATES)
    if not unique_levels.issubset(set(columns)):
        raise InvalidLevelsError(LevelsErrorType.COLUMNS)
