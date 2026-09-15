"""Safety checks for the public API."""

import enum
from collections.abc import Sequence
from itertools import chain
from numbers import Integral

import polars as pl
import torch
from torch import Tensor

__all__ = [
    "DuplicateConditionsError",
    "EmptyDataPointsError",
    "EmptyDatasetError",
    "EmptyTaskError",
    "InputTypeError",
    "InvalidCellError",
    "InvalidDatasetError",
    "InvalidFeatureDtypeError",
    "InvalidFeaturesError",
    "InvalidLevelsError",
    "LabelReservedNameError",
    "LabelSuffixError",
    "NonContiguousIndicesError",
    "PrecomputedCellsError",
    "UnknownConditionError",
]


class InvalidDatasetError(ValueError):
    """Dataset labels and accessor rows do not agree."""


class InvalidFeaturesError(ValueError):
    """Feature dimensions or accessor slice boundaries are invalid."""


class InvalidFeatureDtypeError(TypeError):
    """A continuous operation received an unsupported feature dtype."""

    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__(
            f"Normalization and pooling require floating-point features, got {dtype}. "
            "Pass dtype=torch.float32 or dtype=torch.float64 to the Dataset constructor. "
            "Use 'identical' to compare discrete integer units."
        )


def verify_feature_shape(data: Tensor, dimension: int | None = None) -> None:
    """Features must be a matrix with a positive, consistent feature dimension."""
    if data.ndim != 2 or data.size(1) == 0 or (dimension is not None and data.size(1) != dimension):
        msg = f"Features must have shape (frames, dimension) with a positive consistent dimension, got {data.shape}."
        raise InvalidFeaturesError(msg)


def verify_continuous_dtype(*data: Tensor) -> None:
    """Require floating-point inputs for normalization and pooling; backend support may vary."""
    for tensor in data:
        if not tensor.is_floating_point():
            raise InvalidFeatureDtypeError(tensor.dtype)


NDIM = 3
MIN_A_LEN = 2  # Minimum length of A in the ABX task.
INVALID_COLUMN_SUFFIX = ("_a", "_b", "_x")
INVALID_COLUMN_NAMES = {"index", "score", "size", "is_valid", "__cell", "__group", "__lookup", "__pos", "__triplet"}


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


class EmptyDatasetError(ValueError):
    """The dataset holds no datapoint at all."""

    def __init__(self) -> None:
        super().__init__("The dataset is empty. Check the path to the features and the labels.")


class NonContiguousIndicesError(ValueError):
    """The accessor indices are not exactly ``range(len(indices))``."""

    def __init__(self, count: int, lowest: int, highest: int) -> None:
        super().__init__(
            f"The keys of `indices` must be exactly the row numbers 0 to {count - 1} of `Dataset.labels`, "
            f"but {count} keys were given, ranging from {lowest} to {highest}."
        )


def verify_empty_datapoints(indices: dict[int, tuple[int, int]]) -> None:
    """Check that there is at least one datapoint, that the indices cover every row exactly once, and none is empty."""
    if not indices:
        raise EmptyDatasetError
    if any(not isinstance(index, Integral) or isinstance(index, bool) for index in indices):
        msg = "Accessor indices must be integer row numbers."
        raise InvalidDatasetError(msg)
    lowest, highest = min(indices), max(indices)
    if lowest != 0 or highest != len(indices) - 1:
        raise NonContiguousIndicesError(len(indices), lowest, highest)
    empty = []
    for key, (start, end) in indices.items():
        if any(not isinstance(bound, Integral) or isinstance(bound, bool) for bound in (start, end)):
            msg = "Accessor slice boundaries must be integers."
            raise InvalidFeaturesError(msg)
        if end <= start:
            empty.append(str(key))
    if empty:
        raise EmptyDataPointsError(empty)


def verify_task_conditions(conditions: list[str]) -> None:
    """Conditions should be unique strings."""
    for cond in conditions:
        if not isinstance(cond, str):
            raise InputTypeError(str, type(cond))
    if len(conditions) != len(set(conditions)):
        raise DuplicateConditionsError


class UnknownConditionError(ValueError):
    """A condition is not a column of ``Dataset.labels``."""

    def __init__(self, missing: list[str], available: list[str]) -> None:
        names = ", ".join(repr(name) for name in missing)
        super().__init__(
            f"No column named {names} in `Dataset.labels`. Every ON, BY and ACROSS condition must be one "
            f"of: {available}. The usual cause is a typo, or labels built without the column at all."
        )


def verify_conditions_exist(columns: list[str], conditions: list[str]) -> None:
    """Every condition must be a column of the labels."""
    missing = [c for c in conditions if c not in columns]
    if missing:
        raise UnknownConditionError(missing, columns)


def verify_dataset_labels(df: pl.DataFrame) -> None:
    """Check the column labels."""
    for col in df.schema:
        if col in INVALID_COLUMN_NAMES:
            raise LabelReservedNameError(col)
        if col.endswith(INVALID_COLUMN_SUFFIX):
            raise LabelSuffixError(col)


class EmptyTaskError(ValueError):
    """No cell could be built for the given conditions."""

    def __init__(self, on: str, by: list[str], across: list[str]) -> None:
        conditions = f"ON({on})"
        if by:
            conditions += f", BY({', '.join(by)})"
        if across:
            conditions += f", ACROSS({', '.join(across)})"
        super().__init__(
            f"The task has no cell: no triplet satisfies {conditions}. A cell needs two different values "
            f"of the ON condition among datapoints that share the same BY values, with at least 2 "
            f"instances available for A. The usual causes are a BY condition that is redundant with the "
            f"ON condition (so each group holds a single ON value), or a corpus too small for the "
            f"conditions asked for."
        )


def verify_task_is_not_empty(num_cells: int, on: str, by: list[str], across: list[str]) -> None:
    """Reject a task without cells: it cannot be scored, and collapsing it would average nothing."""
    if num_cells == 0:
        raise EmptyTaskError(on, by, across)


REQUIRED_PRECOMPUTED_CELL_COLUMNS = ("header", "description", "index_a", "index_b", "index_x")


class PrecomputedCellsError(ValueError):
    """The precomputed cells DataFrame is not shaped like what ``Task`` expects."""


def verify_precomputed_cells(cells: pl.DataFrame, num_items: int, *, is_symmetric: bool) -> None:
    """Check that a user-supplied cells DataFrame is usable by ``Task.from_cells``.

    Verifies the required columns are present, that the index columns hold lists of integers, that
    no index list is empty, that every referenced row is in ``range(num_items)``, and, when
    ``is_symmetric``, that ``index_a`` equals ``index_x`` row by row (symmetric scoring drops the
    matrix diagonal and so requires X and A to be the same set in the same order) and that every
    ``index_a`` holds at least ``MIN_A_LEN`` rows.
    """
    missing = [c for c in REQUIRED_PRECOMPUTED_CELL_COLUMNS if c not in cells.columns]
    if missing:
        msg = f"Precomputed cells are missing required columns: {missing}"
        raise PrecomputedCellsError(msg)
    if cells.is_empty():
        msg = "Precomputed cells are empty: a task needs at least one cell to be scored."
        raise PrecomputedCellsError(msg)
    for col in ("index_a", "index_b", "index_x"):
        dtype = cells.schema[col]
        if not (isinstance(dtype, pl.List) and dtype.inner.is_integer()):
            msg = f"Column {col!r} must be a list of integers, got {dtype}"
            raise PrecomputedCellsError(msg)
        if cells.select(
            (pl.col(col).is_null() | pl.col(col).list.eval(pl.element().is_null()).list.any()).any()
        ).item():
            msg = f"Column {col!r} must not contain null lists or null indices"
            raise PrecomputedCellsError(msg)
    empty = cells.select(
        (pl.col("index_a").list.len() == 0).any().alias("a"),
        (pl.col("index_b").list.len() == 0).any().alias("b"),
        (pl.col("index_x").list.len() == 0).any().alias("x"),
    ).row(0, named=True)
    empty_cols = [f"index_{k}" for k, v in empty.items() if v]
    if empty_cols:
        msg = f"Precomputed cells contain empty index lists in column(s): {empty_cols}"
        raise PrecomputedCellsError(msg)
    if is_symmetric:
        if cells.select((pl.col("index_a") != pl.col("index_x")).any()).item():
            msg = "Symmetric precomputed cells require index_a == index_x for every row (X and A must be the same set)"
            raise PrecomputedCellsError(msg)
        if cells.select((pl.col("index_a").list.len() < MIN_A_LEN).any()).item():
            msg = (
                f"Symmetric precomputed cells require at least {MIN_A_LEN} rows in 'index_a' (scoring drops "
                f"the diagonal, so a cell with a single A has no triplet left to score)"
            )
            raise PrecomputedCellsError(msg)
    bounds = cells.select(
        pl.col("index_a").list.min().min().alias("lo_a"),
        pl.col("index_a").list.max().max().alias("hi_a"),
        pl.col("index_b").list.min().min().alias("lo_b"),
        pl.col("index_b").list.max().max().alias("hi_b"),
        pl.col("index_x").list.min().min().alias("lo_x"),
        pl.col("index_x").list.max().max().alias("hi_x"),
    ).row(0, named=True)
    lows = [v for v in (bounds["lo_a"], bounds["lo_b"], bounds["lo_x"]) if v is not None]
    highs = [v for v in (bounds["hi_a"], bounds["hi_b"], bounds["hi_x"]) if v is not None]
    if lows and min(lows) < 0:
        msg = f"Precomputed cells contain negative indices (min={min(lows)})"
        raise PrecomputedCellsError(msg)
    if highs and max(highs) >= num_items:
        msg = f"Precomputed cells reference index {max(highs)} but the dataset only has {num_items} items"
        raise PrecomputedCellsError(msg)


def verify_subsampler_params(*sizes: int | None, seed: int) -> None:
    """All sizes must be integers greater than or equal to 2."""
    if not all(isinstance(s, int) and s > 1 for s in sizes if s is not None):
        msg = "sizes should be integers >= 2"
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
        msg = None
        match error_type:
            case CellErrorType.NDIM:
                msg = "A, B, and X should be tensors with 3 dimensions"
            case CellErrorType.FEATURE_DIM:
                msg = "A, B, and X should have the same feature dimension"
            case CellErrorType.SIZE:
                msg = "Invalid size specification"
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
        msg = None
        match error_type:
            case LevelsErrorType.FORMAT:
                msg = "'levels' should be list[tuple[str, ...] | str]"
            case LevelsErrorType.DUPLICATES:
                msg = "levels should not contain duplicates"
            case LevelsErrorType.COLUMNS:
                msg = "levels should be columns of the DataFrame"
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
