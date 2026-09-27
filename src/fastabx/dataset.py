"""Data utilities."""

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import partial
from pathlib import Path
from typing import Any, Self

import numpy as np
import polars as pl
import polars.selectors as cs
import torch
from tqdm import tqdm

from fastabx.accessor import Accessor, ArrayLike, InMemoryAccessor
from fastabx.utils import hide_progress, resolve_device, with_librilight_bug
from fastabx.verify import EmptyDatasetError, InvalidDatasetError, InvalidFeaturesError, verify_feature_shape

__all__ = [
    "Dataset",
    "EmptyFeaturesError",
    "FeaturesSizeError",
    "FrequencyTypeError",
    "InvalidItemFileError",
    "InvalidTimesError",
    "NonFiniteError",
    "TimesArrayDimensionError",
    "TimesArrayFrontiersError",
]


def _is_pandas_dataframe(obj: object) -> bool:
    """Check if ``obj`` is a pandas DataFrame without importing pandas."""
    cls = type(obj)
    return cls.__name__ == "DataFrame" and cls.__module__.split(".", 1)[0] == "pandas"


def find_all_files(root: str | Path, extension: str) -> dict[str, Path]:
    """Recursively find all files whose name ends with ``extension`` in ``root``.

    Each one is keyed by its path relative to ``root``, in POSIX form, without ``extension``. The extension is a
    filename suffix, so it must contain a dot: ``".pt"``, or ``"_features.pt"``, but not ``"pt"``, which would also
    match ``script``.
    """
    if "." not in extension:
        msg = f"`extension` must contain a dot, such as '.{extension}', not {extension!r}: it is a filename suffix."
        raise ValueError(msg)
    r = Path(root)
    files = (p for p in r.rglob(f"*{extension}") if p.is_file())
    return dict(sorted((p.relative_to(r).as_posix().removesuffix(extension), p) for p in files))


class InvalidItemFileError(Exception):
    """The item file is invalid."""


def parse_times(column: pl.Series) -> pl.Series:
    """Parse a column of onsets or offsets, written as strings, into decimals.

    The precision and scale are inferred from the values, except when none of them has a nonzero digit: polars
    cannot infer a decimal type from zeros alone, and every such value is exactly representable with scale 0.
    Scientific notation is rejected, since the decimal parser would silently read ``1e-3`` as 0.
    """
    scientific = column.str.contains(r"^[+-]?(\d+\.?\d*|\.\d+)[eE][+-]?\d+$")
    if scientific.any():
        example = column.filter(scientific)[0]
        msg = (
            f"Column {column.name!r} holds times in scientific notation, such as {example!r}. "
            f"Write the times as plain decimal numbers instead, e.g. '0.00005' rather than '5e-05'."
        )
        raise InvalidItemFileError(msg)
    if not column.str.contains("[1-9]").any():
        return column.str.to_decimal(scale=0)
    return column.str.to_decimal(inference_length=len(column))


def read_labels(item: str | Path, file_col: str, onset_col: str, offset_col: str) -> pl.DataFrame:
    """Return the labels from the path to the item file.

    Every column of a text item file (``.item``, ``.csv``) is read as a string: inferring the types would
    merge distinct labels such as speakers ``01`` and ``1``, and fail on a label that only stops looking
    numeric after the first rows. The onsets and offsets are then parsed as decimals.
    """
    schema_overrides = {file_col: pl.String, onset_col: pl.String, offset_col: pl.String}
    match ext := Path(item).suffix:
        case ".item":
            df = pl.read_csv(item, separator=" ", infer_schema=False)
        case ".csv":
            df = pl.read_csv(item, infer_schema=False)
        case ".jsonl" | ".ndjson":
            df = pl.read_ndjson(item, schema_overrides=schema_overrides)
        case _:
            msg = f"File extension {ext} is not supported. Supported extensions are .item, .csv, .jsonl, .ndjson."
            raise InvalidItemFileError(msg)
    if df.is_empty():
        raise EmptyDatasetError
    if missing := {file_col, onset_col, offset_col} - set(df.columns):
        msg = f"Item metadata is missing required columns: {sorted(missing)}"
        raise InvalidItemFileError(msg)
    return df.with_columns(parse_times(df[onset_col]), parse_times(df[offset_col]))


def read_csv_labels_as_strings(
    source: str | Path, separator: str, feature_columns: str | Collection[str]
) -> pl.DataFrame:
    """Read a CSV file of labels and features, with every label column read as a string.

    Only the feature columns have their types inferred: as in :py:func:`read_labels`, inferring the types of the
    labels would merge distinct labels such as speakers ``01`` and ``1``.
    """
    features = {feature_columns} if isinstance(feature_columns, str) else set(feature_columns)
    columns = pl.scan_csv(source, separator=separator).collect_schema().names()
    labels = {name: pl.String for name in columns if name not in features}
    return pl.read_csv(source, separator=separator, schema_overrides=labels)


class FrequencyTypeError(TypeError):
    """If frequency is of a type that can lead to floating-point unexpected behavior."""

    def __init__(self) -> None:
        super().__init__(
            "`frequency` is getting converted to Decimal. To avoid floating point errors, it should be "
            "an int, str, or Decimal. In particular, we don't allow `frequency` to be a float but not an int."
        )


def decimal_frequency(frequency: int | str | Decimal) -> Decimal:
    """Convert frequency to a Decimal."""
    if isinstance(frequency, (int, str, Decimal)) and not isinstance(frequency, bool):
        try:
            value = Decimal(str(frequency))
        except InvalidOperation:
            msg = "frequency must be a positive finite decimal"
            raise ValueError(msg) from None
        if not value.is_finite() or value <= 0:
            msg = "frequency must be positive and finite"
            raise ValueError(msg)
        return value
    raise FrequencyTypeError


def item_frontiers(
    frequency: int | str | Decimal,
    onset_col: str,
    offset_col: str,
) -> tuple[pl.Expr, pl.Expr, pl.Expr, pl.Expr]:
    """Frontiers [start, end[ in the input features and in the concatenated ones."""
    frequency = decimal_frequency(frequency)
    start = (pl.col(onset_col) * frequency - Decimal("0.5")).ceil().cast(pl.Int64).alias("start")
    end = (pl.col(offset_col) * frequency - Decimal("0.5")).floor().cast(pl.Int64).alias("end")
    if not with_librilight_bug():
        end += 1
    length = (end - start).alias("length")
    right = length.cum_sum().alias("right")
    left = length.cum_sum().shift(1).fill_null(0).alias("left")
    return start, end, left, right


class FeaturesSizeError(ValueError):
    """To raise if the features size is not correct."""

    def __init__(self, fileid: str, start: int, end: int, actual: int) -> None:
        super().__init__(
            f"Input features length is not correct for file {fileid}. It has a length {actual}, "
            f"but we are slicing between [{start}, {end}[.\n"
            f"The most common reason for this is that there is one frame missing in the features, because "
            f"of how the convolutional layers are defined in your model and because the phoneme under consideration "
            f"is at the very end of the file. You can either add padding to the convolutions, or add a bit of silence "
            f"at the end of the audio file."
        )


class EmptyFeaturesError(ValueError):
    """Raised when empty features are found when building the dataset."""

    def __init__(self, df: pl.DataFrame) -> None:
        super().__init__(
            f"{len(df)} empty entries found. These entries are shorter than a single unit at the given frequency. "
            f"First, check that the given frequency is correct. Then, if you intend to compute ABX on units this large"
            f", you must first remove these entries from your item file. "
            f"Refer to https://docs.cognitive-ml.fr/fastabx/advanced/slicing.html for details on how features are "
            f"sliced. The empty entries are: \n{df}"
        )


def missing_files_error(found: set[str], to_find: set[str]) -> FileNotFoundError:
    """Error to raise when some files are missing."""
    return FileNotFoundError(
        f"{len(to_find - found)} files missing to build the Dataset. "
        f"Only {len(found & to_find)} out of {len(to_find)} have been found. "
        "Make sure to use the correct directory and file extension."
    )


class NonFiniteError(ValueError):
    """To raise if non-finite features have been found."""

    def __init__(self, fileid: str | None = None) -> None:
        source = "tabular input" if fileid is None else f"file '{fileid}'"
        super().__init__(f"Non-finite values detected in features for {source}")


def verify_intervals(labels: pl.DataFrame, onset_col: str, offset_col: str) -> None:
    """Reject empty metadata and invalid intervals; both boundaries are inclusive."""
    if labels.is_empty():
        raise EmptyDatasetError
    onset, offset = pl.col(onset_col), pl.col(offset_col)
    invalid = (
        onset.is_null()
        | offset.is_null()
        | ~onset.cast(pl.Float64).is_finite()
        | ~offset.cast(pl.Float64).is_finite()
        | (onset < 0)
        | (offset < onset)
    )
    if labels.select(invalid.any()).item():
        msg = "Item intervals must have finite non-null times with 0 <= onset <= offset."
        raise InvalidItemFileError(msg)


def prepare_features(
    features: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype | None,
    fileid: str | None = None,
    dimension: int | None = None,
) -> torch.Tensor:
    """Validate features before and after an explicit dtype conversion."""
    verify_feature_shape(features, dimension)
    if not torch.isfinite(features).all():
        raise NonFiniteError(fileid)
    converted = features.detach().to(device=device, dtype=dtype)
    if converted.dtype != features.dtype and not torch.isfinite(converted).all():
        raise NonFiniteError(fileid)
    return converted


class InvalidTimesError(ValueError):
    """Timestamps must be finite, sorted, and match the feature frame count."""


def load_data_from_item[T](
    mapping: Mapping[str, T],
    labels: pl.DataFrame,
    frequency: int | str | Decimal,
    feature_maker: Callable[[T], torch.Tensor],
    file_col: str,
    onset_col: str,
    offset_col: str,
    device: torch.device,
    *,
    dtype: torch.dtype | None = None,
    progress: bool = True,
) -> tuple[dict[int, tuple[int, int]], torch.Tensor]:
    """Load all data in memory on ``device``. Return a dictionary of indices and a tensor of data.

    The output tensor is allocated once, when the first file gives the feature dimension and dtype, and each
    item is copied into place: at most one file is held besides the output.
    """
    verify_intervals(labels, onset_col, offset_col)
    metadata = labels[[file_col, onset_col, offset_col]].with_row_index()
    frontiers = item_frontiers(frequency, onset_col, offset_col)
    lazy = metadata.lazy().sort(file_col, maintain_order=True).with_columns(*frontiers)
    empty = lazy.filter(pl.col("end") <= pl.col("start")).sort("index").select(file_col, onset_col, offset_col)
    indices_lazy = lazy.select("left", "right", "index").sort("index").select("left", "right")
    by_file_lazy = (
        lazy.select(file_col, "start", "end", "left")
        .group_by(file_col, maintain_order=True)
        .agg("start", "end", "left")
    )
    empty, indices, by_file = pl.collect_all([empty, indices_lazy, by_file_lazy])
    if not empty.is_empty():
        raise EmptyFeaturesError(empty)

    total = int(indices["right"].max())  # ty: ignore[invalid-argument-type]
    data = None
    for fileid, start_indices, end_indices, left_indices in tqdm(
        by_file.iter_rows(),
        desc="Building dataset",
        total=len(by_file),
        disable=hide_progress(progress=progress),
    ):
        try:
            source = mapping[fileid]
        except KeyError as error:
            raise missing_files_error(set(mapping), set(by_file[file_col].unique())) from error
        dim = None if data is None else data.size(1)
        features = prepare_features(feature_maker(source), device, dtype, fileid, dim)
        if data is None:
            data = features.new_empty((total, features.size(1)))
        elif features.dtype != data.dtype:
            msg = (
                f"The features of file {fileid!r} have dtype {features.dtype}, but the previous files have "
                f"{data.dtype}. Pass `dtype=` to convert every file to the same dtype."
            )
            raise InvalidFeaturesError(msg)
        for start, end, left in zip(start_indices, end_indices, left_indices, strict=True):
            if start < 0 or end > features.size(0):
                raise FeaturesSizeError(fileid, start, end, features.size(0))
            data[left : left + end - start] = features[start:end]
    return dict(enumerate(indices.rows())), data  # ty: ignore[invalid-return-type]


class TimesArrayDimensionError(ValueError):
    """To raise if the times array is not 1D."""

    def __init__(self) -> None:
        super().__init__("Only 1D times array are supported")


class TimesArrayFrontiersError(ValueError):
    """To raise if we select nothing."""

    def __init__(self, fileid: str, onset: float, offset: float) -> None:
        super().__init__(f"No times were found between onset={onset}, offset={offset} for file {fileid}")


def frames_between(times: torch.Tensor, onsets: list[Decimal], offsets: list[Decimal]) -> tuple[list[int], list[int]]:
    """Return the ``[start, end[`` frames of each item, whose times satisfy ``onset <= time <= offset``.

    Those frames are contiguous since ``times`` is sorted, so each item is located by binary search.
    """
    bounds = torch.tensor(
        [[float(t) for t in onsets], [float(t) for t in offsets]], dtype=times.dtype, device=times.device
    )
    starts = torch.searchsorted(times, bounds[0], side="left")
    ends = torch.searchsorted(times, bounds[1], side="right")
    return starts.tolist(), ends.tolist()


def load_data_from_item_with_times[T](
    paths_features: Mapping[str, T],
    paths_times: Mapping[str, T],
    labels: pl.DataFrame,
    feature_maker: Callable[[T], torch.Tensor],
    time_maker: Callable[[T], torch.Tensor],
    file_col: str,
    onset_col: str,
    offset_col: str,
    device: torch.device,
    *,
    dtype: torch.dtype | None = None,
    progress: bool = True,
) -> tuple[dict[int, tuple[int, int]], torch.Tensor]:
    """Load all data in memory on ``device``, using features and times array."""
    verify_intervals(labels, onset_col, offset_col)
    metadata = labels[[file_col, onset_col, offset_col]].with_row_index()
    by_file = (
        metadata.sort(file_col, maintain_order=True)
        .group_by(file_col, maintain_order=True)
        .agg("index", onset_col, offset_col)
    )
    data, all_indices, right = [], {}, 0
    scales = [
        column_dtype.scale
        for column_dtype in (labels.schema[onset_col], labels.schema[offset_col])
        if isinstance(column_dtype, pl.Decimal)
    ]
    decimals = max(scales) if len(scales) == 2 else None
    for fileid, indices, onsets, offsets in tqdm(
        by_file.iter_rows(),
        desc="Building dataset",
        total=len(by_file),
        disable=hide_progress(progress=progress),
    ):
        try:
            sources = paths_features[fileid], paths_times[fileid]
        except KeyError as error:
            raise missing_files_error(
                set(paths_features) & set(paths_times), set(by_file[file_col].unique())
            ) from error
        dimension = data[0].size(1) if data else None
        features = prepare_features(feature_maker(sources[0]), device, dtype, fileid, dimension)
        times = time_maker(sources[1]).detach().to(device=device, dtype=torch.float64)
        if times.ndim != 1:
            raise TimesArrayDimensionError
        if times.numel() != features.size(0) or not torch.isfinite(times).all():
            msg = f"Timestamps for {fileid!r} must be finite and have one entry per feature frame."
            raise InvalidTimesError(msg)
        if (times[1:] < times[:-1]).any():
            msg = f"Timestamps for {fileid!r} must be sorted in non-decreasing order."
            raise InvalidTimesError(msg)
        if decimals is not None:
            times = times.round(decimals=decimals)
        starts, ends = frames_between(times, onsets, offsets)
        pieces = []
        for index, onset, offset, start, end in zip(indices, onsets, offsets, starts, ends, strict=True):
            if end <= start:
                raise TimesArrayFrontiersError(fileid, float(onset), float(offset))
            pieces.append(features[start:end])
            all_indices[index] = (right, right + end - start)
            right += end - start
        data.append(torch.cat(pieces))  # One copy per file, so that the whole file is not kept alive.
    return all_indices, torch.cat(data, dim=0)


@dataclass
class Dataset:
    """Simple interface to a dataset.

    :param labels: ``pl.DataFrame`` containing the labels of the datapoints.
    :param accessor: :py:class:`.Accessor` to the data, usually an :py:class:`.InMemoryAccessor`.
    """

    labels: pl.DataFrame
    accessor: Accessor

    def __post_init__(self) -> None:
        if len(self.labels) != len(self.accessor):
            msg = f"Labels and accessor must have the same length, got {len(self.labels)} and {len(self.accessor)}."
            raise InvalidDatasetError(msg)

    def __repr__(self) -> str:
        return f"labels:\n{self.labels!r}\naccessor: {self.accessor!r}"

    def normalize_(self) -> Self:
        """L2 normalize nonzero frames and leave zero frames zero; append a compatibility border.

        Idempotent: a second call is a no-op. Angular distance assigns zero/zero distance 0 and
        zero/nonzero distance 1.
        """
        self.accessor.normalize_()
        return self

    @classmethod
    def from_item(
        cls,
        item: str | Path,
        root: str | Path,
        frequency: int | str | Decimal,
        *,
        feature_maker: Callable[[str | Path], torch.Tensor] | None = None,
        extension: str = ".pt",
        file_col: str = "#file",
        onset_col: str = "onset",
        offset_col: str = "offset",
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        progress: bool = True,
    ) -> Self:
        """Create a dataset from an item file.

        See :doc:`/items` for the format of the item file, and how ``#file`` is matched to the feature files.

        If you want to keep the Libri-Light bug to reproduce previous results,
        set the environment variable FASTABX_WITH_LIBRILIGHT_BUG=1.

        :param item: Path to the item file.
        :param root: Path to the root directory containing either the features or the audio files.
        :param frequency: The feature frequency of the features / the output of the feature maker, in Hz.
            If it is not an integer, pass it as a string to avoid floating-point errors.
        :param feature_maker: Function that takes a path and returns a torch.Tensor. Defaults to ``None``, which
            loads each file with ``torch.load``.
        :param extension: The filename extension of the files to process in ``root``, default is ".pt".
            It must contain a dot.
        :param file_col: Column in the item file that contains the file identifiers: the path of each feature
            file relative to ``root``, without the extension. Default is "#file".
        :param onset_col: Column in the item file that contains the onset times, default is "onset".
        :param offset_col: Column in the item file that contains the offset times, default is "offset".
        :param dtype: Optional torch dtype for feature conversion. None preserves the input dtype.
        :param device: Device on which to store the features, such as "cpu" or "cuda:1".
            Defaults to CUDA if available, and CPU otherwise.
        :param progress: Whether to display a progress bar while building the dataset.
        """
        labels = read_labels(item, file_col, onset_col, offset_col)
        paths = find_all_files(root, extension)
        resolved = resolve_device(device)
        indices, data = load_data_from_item(
            paths,
            labels,
            frequency,
            feature_maker or partial(torch.load, map_location="cpu"),
            file_col,
            onset_col,
            offset_col,
            resolved,
            dtype=dtype,
            progress=progress,
        )
        return cls(labels=labels, accessor=InMemoryAccessor(indices, data, resolved))

    @classmethod
    def from_item_with_times(
        cls,
        item: str | Path,
        root_features: str | Path,
        root_times: str | Path,
        *,
        feature_maker: Callable[[str | Path], torch.Tensor] | None = None,
        time_maker: Callable[[str | Path], torch.Tensor] | None = None,
        extension: str = ".pt",
        file_col: str = "#file",
        onset_col: str = "onset",
        offset_col: str = "offset",
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        progress: bool = True,
    ) -> Self:
        """Create a dataset from an item file.

        Use arrays containing the times associated to the features instead of a given frequency.
        See :doc:`/items` for the format of the item file.

        :param item: Path to the item file.
        :param root_features: Path to the root directory containing either the features or the audio files.
        :param root_times: Path to the root directory containing the times arrays.
        :param feature_maker: Function that takes a path and returns a torch.Tensor. Defaults to ``None``, which
            loads each file with ``torch.load``.
        :param time_maker: Function that takes a path and returns a 1D torch.Tensor. Defaults to ``None``, which
            loads each file with ``torch.load``.
        :param extension: The filename extension of the files to process in ``root_features`` and ``root_times``,
            default is ".pt".
        :param file_col: Column in the item file that contains the file identifiers: the path of each feature
            file relative to ``root_features``, and of each times file relative to ``root_times``, without the
            extension. Default is "#file".
        :param onset_col: Column in the item file that contains the onset times, default is "onset".
        :param offset_col: Column in the item file that contains the offset times, default is "offset".
        :param dtype: Optional torch dtype for feature conversion. None preserves the input dtype.
        :param device: Device on which to store the features, such as "cpu" or "cuda:1".
            Defaults to CUDA if available, and CPU otherwise.
        :param progress: Whether to display a progress bar while building the dataset.
        """
        labels = read_labels(item, file_col, onset_col, offset_col)
        paths_feat = find_all_files(root_features, extension)
        paths_time = find_all_files(root_times, extension)
        resolved = resolve_device(device)
        indices, data = load_data_from_item_with_times(
            paths_feat,
            paths_time,
            labels,
            feature_maker or partial(torch.load, map_location="cpu"),
            time_maker or partial(torch.load, map_location="cpu"),
            file_col,
            onset_col,
            offset_col,
            resolved,
            dtype=dtype,
            progress=progress,
        )
        return cls(labels=labels, accessor=InMemoryAccessor(indices, data, resolved))

    @classmethod
    def from_item_and_units(
        cls,
        item: str | Path,
        units: str | Path,
        frequency: int | str | Decimal,
        *,
        audio_key: str = "audio",
        units_key: str = "units",
        file_col: str = "#file",
        onset_col: str = "onset",
        offset_col: str = "offset",
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        progress: bool = True,
    ) -> Self:
        """Create a dataset from an item file with the units all described in a single JSONL file.

        See :doc:`/items` for the format of the item file. Unlike :py:meth:`from_item`, which matches the path of
        each feature file relative to a root directory, the units are matched on the **base name** of their
        ``audio_key`` path, without directories nor extension: ``/data/dev-clean/84/84-121123-0000.flac`` is the
        ``#file`` ``84-121123-0000``. Those base names must therefore be unique in the units file.

        :param item: Path to the item file.
        :param units: Path to the JSONL file containing the units.
        :param frequency: The feature frequency, in Hz.
            If it is not an integer, pass it as a string to avoid floating-point errors.
        :param audio_key: Key in the JSONL file that contains the audio file paths (str), default is "audio".
        :param units_key: Key in the JSONL file that contains the units (list[int]), default is "units".
        :param file_col: Column in the item file that contains the audio file names, default is "#file".
        :param onset_col: Column in the item file that contains the onset times, default is "onset".
        :param offset_col: Column in the item file that contains the offset times, default is "offset".
        :param dtype: Optional torch dtype for feature conversion. None preserves the input dtype.
        :param device: Device on which to store the features, such as "cpu" or "cuda:1".
            Defaults to CUDA if available, and CPU otherwise.
        :param progress: Whether to display a progress bar while building the dataset.
        """
        labels = read_labels(item, file_col, onset_col, offset_col)
        units_df = (
            pl.scan_ndjson(units)
            .with_columns(pl.col(audio_key).str.replace(r"^.*[/\\]", "").str.replace(r"\.[^.]+$", ""))
            .collect()
        )

        if duplicated := sorted(units_df.filter(pl.col(audio_key).is_duplicated())[audio_key].unique()):
            msg = (
                f"Units contain duplicate audio identifiers after removing directories and extensions: "
                f"{duplicated[:10]}{' ...' if len(duplicated) > 10 else ''}. Units are matched on base names only."
            )
            raise InvalidItemFileError(msg)

        def feature_maker(idx: int) -> torch.Tensor:
            return torch.tensor(units_df[idx, units_key]).unsqueeze(1)

        mapping: dict[str, int] = dict(zip(units_df[audio_key], range(len(units_df)), strict=True))
        resolved = resolve_device(device)
        indices, data = load_data_from_item(
            mapping,
            labels,
            frequency,
            feature_maker,
            file_col,
            onset_col,
            offset_col,
            resolved,
            dtype=dtype,
            progress=progress,
        )
        return cls(labels=labels, accessor=InMemoryAccessor(indices, data, resolved))

    @classmethod
    def from_dataframe(
        cls,
        source: str | Path | pl.DataFrame | Mapping[str, Sequence[object]] | Iterable[Mapping[str, Any]],
        feature_columns: str | Collection[str],
        *,
        separator: str = ",",
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Create a dataset from any tabular source containing both the labels and the features.

        Accepted inputs for ``source``:

        - ``str`` or ``Path``: path to a CSV file (uses ``separator``). As with :py:meth:`from_item`, every label
          column is read as a string, so that distinct labels such as speakers ``01`` and ``1`` are not merged.
        - A polars or pandas ``DataFrame``.
        - ``Mapping[str, Sequence]`` (column name → values).
        - ``Iterable[Mapping]`` (sequence of row dictionaries).

        :param source: The tabular source. See above for accepted types.
        :param feature_columns: Column name or list of column names containing the features.
        :param separator: Separator used in the CSV file. Only relevant when ``source`` is a path.
        :param dtype: Optional torch dtype for feature conversion. None preserves the input dtype.
        :param device: Device on which to store the features, such as "cpu" or "cuda:1".
            Defaults to CUDA if available, and CPU otherwise.
        """
        if isinstance(source, (str, Path)):
            df = read_csv_labels_as_strings(source, separator, feature_columns)
        elif isinstance(source, pl.DataFrame):
            df = source
        elif _is_pandas_dataframe(source):
            df: pl.DataFrame = pl.from_pandas(source)  # ty: ignore[invalid-assignment]
        elif isinstance(source, Mapping):
            df = pl.from_dict(source)  # ty: ignore[invalid-argument-type]
        elif isinstance(source, Iterable):
            df = pl.from_dicts(source)
        else:
            msg = f"Type of given `source` in Dataset.from_dataframe is not valid: {type(source).__name__}"
            raise TypeError(msg)
        labels = df.select(cs.exclude(feature_columns))
        indices = {i: (i, i + 1) for i in range(len(labels))}
        features = df.select(feature_columns)
        resolved = resolve_device(device)
        data = prepare_features(features.to_torch(), resolved, dtype)
        return cls(labels=labels, accessor=InMemoryAccessor(indices, data, resolved))

    @classmethod
    def from_numpy(
        cls,
        features: ArrayLike,
        labels: pl.DataFrame | Mapping[str, Sequence[object]],
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Create a dataset from the features and the labels.

        Despite the name, ``features`` is not restricted to a numpy array: any input accepted by ``np.asarray``
        works (Python lists, tuples, CPU torch tensors via the ``__array__`` protocol, ...).
        CUDA tensors must be moved to CPU first.

        :param features: 2D array-like containing the features.
        :param labels: Dictionary of sequences, or polars/pandas DataFrame containing the labels.
        :param dtype: Optional torch dtype for feature conversion. None preserves the input dtype.
        :param device: Device on which to store the features, such as "cpu" or "cuda:1".
            Defaults to CUDA if available, and CPU otherwise.
        """
        array = np.asarray(features)
        if array.ndim != 2:
            msg = "features must be a two-dimensional array (rows, dimension)"
            raise ValueError(msg)
        if isinstance(labels, pl.DataFrame):
            labels_df = labels
        elif _is_pandas_dataframe(labels):
            labels_df: pl.DataFrame = pl.from_pandas(labels)  # ty: ignore[invalid-assignment]
        else:
            labels_df = pl.from_dict(labels)
        if len(array) != len(labels_df):
            msg = f"`features` and `labels` must have the same length, got {len(array)} and {len(labels_df)}"
            raise ValueError(msg)
        resolved = resolve_device(device)
        data = prepare_features(torch.tensor(array), resolved, dtype)  # A copy: the dataset never aliases `features`.
        indices = {i: (i, i + 1) for i in range(len(labels_df))}
        return cls(labels=labels_df, accessor=InMemoryAccessor(indices, data, resolved))
