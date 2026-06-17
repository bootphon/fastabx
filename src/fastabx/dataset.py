"""Data utilities."""

import math
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Self

import numpy as np
import numpy.typing as npt
import polars as pl
import polars.selectors as cs
import torch
from tqdm import tqdm

from fastabx.utils import default_device, with_librilight_bug
from fastabx.verify import verify_empty_datapoints

__all__ = ["Batch", "Dataset", "InMemoryAccessor"]

type ArrayLike = npt.ArrayLike  # Better rendering in docs


def _is_pandas_dataframe(obj: object) -> bool:
    """Check if ``obj`` is a pandas DataFrame without importing pandas."""
    cls = type(obj)
    return cls.__name__ == "DataFrame" and cls.__module__.split(".", 1)[0] == "pandas"


@dataclass(frozen=True)
class Batch:
    """Batch of padded data."""

    data: torch.Tensor
    sizes: torch.Tensor

    def __repr__(self) -> str:
        return f"Batch(data=Tensor(shape={self.data.shape}, dtype={self.data.dtype}), sizes={self.sizes})"


class InMemoryAccessor:
    """Data accessor where everything is in memory."""

    def __init__(self, indices: dict[int, tuple[int, int]], data: torch.Tensor) -> None:
        self.device = default_device()
        self.indices = indices
        verify_empty_datapoints(self.indices)
        self.data = data.to(self.device)
        self.is_normalized = False
        size = max(self.indices) + 1
        starts, lengths = np.zeros(size, dtype=np.int64), np.zeros(size, dtype=np.int64)
        for i, (start, end) in self.indices.items():
            starts[i], lengths[i] = start, end - start
        self._lengths_np = lengths
        self._starts = torch.from_numpy(starts).to(self.device)
        self._lengths = torch.from_numpy(lengths).to(dtype=torch.int32, device=self.device)

    def __repr__(self) -> str:
        return f"InMemoryAccessor(data of shape {tuple(self.data.shape)}, with {len(self)} items)"

    def __getitem__(self, i: int) -> torch.Tensor:
        if i not in self.indices:
            msg = f"No item at index {i} (the accessor has {len(self.indices)} items)"
            raise IndexError(msg)
        start, end = self.indices[i]
        return self.data[start:end]

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self) -> Iterator[torch.Tensor]:
        for i in self.indices:
            yield self[i]

    def lengths(self, indices: list[int]) -> npt.NDArray[np.int64]:
        """Get the lengths of the data from a list of indices."""
        return self._lengths_np[indices]

    def batched(self, indices: ArrayLike) -> Batch:
        """Get the padded data and the original sizes of the data from a list of indices."""
        idx_np = np.asarray(indices, dtype=np.int64)
        smax = int(self._lengths_np[idx_np].max())
        idx = torch.from_numpy(idx_np).to(self.device)
        sizes = self._lengths.index_select(0, idx)
        starts = self._starts.index_select(0, idx)
        arange = torch.arange(smax, device=self.device)
        mask = arange < sizes.unsqueeze(1)
        src = torch.where(mask, starts.unsqueeze(1) + arange, 0)
        gathered = self.data.index_select(0, src.view(-1)).view(idx.size(0), smax, -1)
        gathered.mul_(mask.unsqueeze(-1))
        return Batch(gathered, sizes)


def find_all_files(root: str | Path, extension: str) -> dict[str, Path]:
    """Recursively find all files with the given `extension` in `root`."""
    root = Path(root)
    return dict(sorted((str(p.relative_to(root)).removesuffix(extension), p) for p in root.rglob(f"*{extension}")))


def normalize_with_singularity_(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize the given vector across the third dimension, in-place.

    Extend all vectors by eps to put the null vector at the maximal
    angular distance from any non-null vector.
    """
    norm = x.norm(dim=1, keepdim=True)
    zero_mask = norm.squeeze(1) == 0
    x[~zero_mask] /= norm[~zero_mask]
    x[zero_mask] = 1.0 / math.sqrt(x.size(1))
    border = x.new_full((x.size(0), 1), eps)
    border[zero_mask] = -2 * eps
    return torch.cat([x, border], dim=1)


class InvalidItemFileError(Exception):
    """The item file is invalid."""


def read_labels(item: str | Path, file_col: str, onset_col: str, offset_col: str) -> pl.DataFrame:
    """Return the labels from the path to the item file."""
    schema_overrides = {file_col: pl.String, onset_col: pl.String, offset_col: pl.String}
    match ext := Path(item).suffix:
        case ".item":
            df = pl.read_csv(item, separator=" ", schema_overrides=schema_overrides)
        case ".csv":
            df = pl.read_csv(item, schema_overrides=schema_overrides)
        case ".jsonl" | ".ndjson":
            df = pl.read_ndjson(item, schema_overrides=schema_overrides)
        case _:
            msg = f"File extension {ext} is not supported. Supported extensions are .item, .csv, .jsonl, .ndjson."
            raise InvalidItemFileError(msg)
    return df.with_columns(
        df[onset_col].str.to_decimal(inference_length=len(df)),
        df[offset_col].str.to_decimal(inference_length=len(df)),
    )


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
        return Decimal(str(frequency))
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
        f"Only {len(found)} out of {len(to_find)} have been found. "
        "Make sure to use the correct directory and file extension."
    )


class NonFiniteError(ValueError):
    """To raise if non-finite features have been found."""

    def __init__(self, fileid: str) -> None:
        super().__init__(f"Non-finite values detected in features for file '{fileid}'")


def load_data_from_item[T](
    mapping: Mapping[str, T],
    labels: pl.DataFrame,
    frequency: int | str | Decimal,
    feature_maker: Callable[[T], torch.Tensor],
    file_col: str,
    onset_col: str,
    offset_col: str,
) -> tuple[dict[int, tuple[int, int]], torch.Tensor]:
    """Load all data in memory. Return a dictionary of indices and a tensor of data."""
    metadata = labels[[file_col, onset_col, offset_col]].with_row_index()
    frontiers = item_frontiers(frequency, onset_col, offset_col)
    lazy = metadata.lazy().sort(file_col, maintain_order=True).with_columns(*frontiers)
    indices_lazy = lazy.select("left", "right", "index").sort("index").select("left", "right")
    by_file_lazy = lazy.select(file_col, "start", "end").group_by(file_col, maintain_order=True).agg("start", "end")
    indices, by_file = pl.collect_all([indices_lazy, by_file_lazy])

    data, device = [], default_device()
    for fileid, start_indices, end_indices in tqdm(by_file.iter_rows(), desc="Building dataset", total=len(by_file)):
        try:
            features = feature_maker(mapping[fileid]).detach().to(device)
            if not torch.isfinite(features).all():
                raise NonFiniteError(fileid)
        except KeyError as error:
            raise missing_files_error(set(mapping), set(by_file[file_col].unique())) from error
        for start, end in zip(start_indices, end_indices, strict=True):
            if start < 0 or end > features.size(0):
                raise FeaturesSizeError(fileid, start, end, features.size(0))
            if end <= start:
                raise EmptyFeaturesError(
                    lazy.filter(pl.col("end") <= pl.col("start"))
                    .sort("index")
                    .select(file_col, onset_col, offset_col)
                    .collect()
                )
            data.append(features[start:end])
    return dict(enumerate(indices.rows())), torch.cat(data, dim=0)


class TimesArrayDimensionError(ValueError):
    """To raise if the times array is not 1D."""

    def __init__(self) -> None:
        super().__init__("Only 1D times array are supported")


class TimesArrayFrontiersError(ValueError):
    """To raise if we select nothing."""

    def __init__(self, fileid: str, onset: float, offset: float) -> None:
        super().__init__(f"No times were found between onset={onset}, offset={offset} for file {fileid}")


def load_data_from_item_with_times[T](
    paths_features: Mapping[str, T],
    paths_times: Mapping[str, T],
    labels: pl.DataFrame,
    feature_maker: Callable[[T], torch.Tensor],
    time_maker: Callable[[T], torch.Tensor],
    file_col: str,
    onset_col: str,
    offset_col: str,
) -> tuple[dict[int, tuple[int, int]], torch.Tensor]:
    """Load all data in memory using features and times array. This is smaller than using a predefined frequency."""
    metadata = labels[[file_col, onset_col, offset_col]].with_row_index()
    by_file = (
        metadata.sort(file_col, maintain_order=True)
        .group_by(file_col, maintain_order=True)
        .agg("index", onset_col, offset_col)
    )
    data, device, all_indices, right = [], default_device(), {}, 0
    decimals = by_file["onset"].dtype.inner.scale  # ty: ignore[unresolved-attribute]
    for fileid, indices, onsets, offsets in tqdm(by_file.iter_rows(), desc="Building dataset", total=len(by_file)):
        try:
            features = feature_maker(paths_features[fileid]).detach().to(device)
            if not torch.isfinite(features).all():
                raise NonFiniteError(fileid)
            times = time_maker(paths_times[fileid]).detach().to(device).round(decimals=decimals)
        except KeyError as error:
            raise missing_files_error(
                set(paths_features) & set(paths_times), set(by_file[file_col].unique())
            ) from error
        if times.ndim > 1:
            raise TimesArrayDimensionError
        for index, onset, offset in zip(indices, onsets, offsets, strict=True):
            mask = torch.where(torch.logical_and(float(onset) <= times, times <= float(offset)))[0]
            if mask.numel() == 0:
                raise TimesArrayFrontiersError(fileid, float(onset), float(offset))
            data.append(features[mask])
            left = right
            right += len(mask)
            all_indices[index] = (left, right)
    return all_indices, torch.cat(data, dim=0)


@dataclass
class Dataset:
    """Simple interface to a dataset.

    :param labels: ``pl.DataFrame`` containing the labels of the datapoints.
    :param accessor: ``InMemoryAccessor`` to access the data.
    """

    labels: pl.DataFrame
    accessor: InMemoryAccessor

    def __repr__(self) -> str:
        return f"labels:\n{self.labels!r}\naccessor: {self.accessor!r}"

    def normalize_(self) -> Self:
        """L2 normalization of the data. Idempotent: a second call is a no-op."""
        if self.accessor.is_normalized:
            return self
        self.accessor.data = normalize_with_singularity_(self.accessor.data)
        self.accessor.is_normalized = True
        return self

    @classmethod
    def from_item(
        cls,
        item: str | Path,
        root: str | Path,
        frequency: int | str | Decimal,
        *,
        feature_maker: Callable[[str | Path], torch.Tensor] = torch.load,
        extension: str = ".pt",
        file_col: str = "#file",
        onset_col: str = "onset",
        offset_col: str = "offset",
    ) -> "Dataset":
        """Create a dataset from an item file.

        If you want to keep the Libri-Light bug to reproduce previous results,
        set the environment variable FASTABX_WITH_LIBRILIGHT_BUG=1.

        :param item: Path to the item file.
        :param root: Path to the root directory containing either the features or the audio files.
        :param frequency: The feature frequency of the features / the output of the feature maker, in Hz.
            If it is not an integer, pass it as a string to avoid floating-point errors.
        :param feature_maker: Function that takes a path and returns a torch.Tensor. Defaults to ``torch.load``.
        :param extension: The filename extension of the files to process in ``root``, default is ".pt".
        :param file_col: Column in the item file that contains the audio file names, default is "#file".
        :param onset_col: Column in the item file that contains the onset times, default is "onset".
        :param offset_col: Column in the item file that contains the offset times, default is "offset".
        """
        labels = read_labels(item, file_col, onset_col, offset_col)
        paths = find_all_files(root, extension)
        indices, data = load_data_from_item(paths, labels, frequency, feature_maker, file_col, onset_col, offset_col)
        return Dataset(labels=labels, accessor=InMemoryAccessor(indices, data))

    @classmethod
    def from_item_with_times(
        cls,
        item: str | Path,
        root_features: str | Path,
        root_times: str | Path,
        *,
        feature_maker: Callable[[str | Path], torch.Tensor] = torch.load,
        time_maker: Callable[[str | Path], torch.Tensor] = torch.load,
        extension: str = ".pt",
        file_col: str = "#file",
        onset_col: str = "onset",
        offset_col: str = "offset",
    ) -> "Dataset":
        """Create a dataset from an item file.

        Use arrays containing the times associated to the features instead of a given frequency.

        :param item: Path to the item file.
        :param root_features: Path to the root directory containing either the features or the audio files.
        :param root_times: Path to the root directory containing the times arrays.
        :param feature_maker: Function that takes a path and returns a torch.Tensor. Defaults to ``torch.load``.
        :param time_maker: Function that takes a path and returns a 1D torch.Tensor. Defaults to ``torch.load``.
        :param extension: The filename extension of the files to process in ``root_features`` and ``root_times``,
            default is ".pt".
        :param file_col: Column in the item file that contains the audio file names, default is "#file".
        :param onset_col: Column in the item file that contains the onset times, default is "onset".
        :param offset_col: Column in the item file that contains the offset times, default is "offset".
        """
        labels = read_labels(item, file_col, onset_col, offset_col)
        paths_feat = find_all_files(root_features, extension)
        paths_time = find_all_files(root_times, extension)
        indices, data = load_data_from_item_with_times(
            paths_feat, paths_time, labels, feature_maker, time_maker, file_col, onset_col, offset_col
        )
        return Dataset(labels=labels, accessor=InMemoryAccessor(indices, data))

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
    ) -> "Dataset":
        """Create a dataset from an item file with the units all described in a single JSONL file.

        :param item: Path to the item file.
        :param units: Path to the JSONL file containing the units.
        :param frequency: The feature frequency, in Hz.
            If it is not an integer, pass it as a string to avoid floating-point errors.
        :param audio_key: Key in the JSONL file that contains the audio file names (str), default is "audio".
        :param units_key: Key in the JSONL file that contains the units (list[int]), default is "units".
        :param file_col: Column in the item file that contains the audio file names, default is "#file".
        :param onset_col: Column in the item file that contains the onset times, default is "onset".
        :param offset_col: Column in the item file that contains the offset times, default is "offset".
        """
        labels = read_labels(item, file_col, onset_col, offset_col)
        units_df = (
            pl.scan_ndjson(units)
            .with_columns(pl.col(audio_key).str.split("/").list.last().str.replace(r"\.[^.]+$", ""))
            .collect()
        )

        def feature_maker(idx: int) -> torch.Tensor:
            return torch.tensor(units_df[idx, units_key]).unsqueeze(1)

        mapping: dict[str, int] = dict(zip(units_df[audio_key], range(len(units_df)), strict=True))
        indices, data = load_data_from_item(mapping, labels, frequency, feature_maker, file_col, onset_col, offset_col)
        return Dataset(labels=labels, accessor=InMemoryAccessor(indices, data))

    @classmethod
    def from_dataframe(
        cls,
        source: str | Path | pl.DataFrame | Mapping[str, Sequence[object]] | Iterable[Mapping[str, Any]],
        feature_columns: str | Collection[str],
        *,
        separator: str = ",",
    ) -> "Dataset":
        """Create a dataset from any tabular source containing both the labels and the features.

        Accepted inputs for ``source``:

        - ``str`` or ``Path``: path to a CSV file (uses ``separator``).
        - A polars or pandas ``DataFrame``.
        - ``Mapping[str, Sequence]`` (column name → values).
        - ``Iterable[Mapping]`` (sequence of row dictionaries).

        :param source: The tabular source. See above for accepted types.
        :param feature_columns: Column name or list of column names containing the features.
        :param separator: Separator used in the CSV file. Only relevant when ``source`` is a path.
        """
        if isinstance(source, (str, Path)):
            df = pl.read_csv(source, separator=separator)
        elif isinstance(source, pl.DataFrame):
            df = source
        elif _is_pandas_dataframe(source):
            df: pl.DataFrame = pl.from_pandas(source)  # ty: ignore[invalid-assignment]
        elif isinstance(source, Mapping):
            df = pl.from_dict(source)  # ty: ignore[invalid-argument-type]
        elif isinstance(source, Iterable):
            df = pl.from_dicts(source)
        else:
            msg = "Type of given `source` in Dataset.from_dataframe is not valid"
            raise ValueError(msg)
        labels = df.select(cs.exclude(feature_columns))
        indices = {i: (i, i + 1) for i in range(len(labels))}
        features = df.select(feature_columns)
        if any(dtype.is_float() for dtype in features.dtypes):
            features = features.cast(pl.Float32)
        data = features.to_torch()
        return Dataset(labels=labels, accessor=InMemoryAccessor(indices, data))

    @classmethod
    def from_numpy(
        cls,
        features: ArrayLike,
        labels: pl.DataFrame | Mapping[str, Sequence[object]],
    ) -> "Dataset":
        """Create a dataset from the features and the labels.

        Despite the name, ``features`` is not restricted to a numpy array: any input accepted by ``np.asarray``
        works (Python lists, tuples, CPU torch tensors via the ``__array__`` protocol, ...).
        CUDA tensors must be moved to CPU first.

        :param features: 2D array-like containing the features.
        :param labels: Dictionary of sequences, or polars/pandas DataFrame containing the labels.
        """
        features_df = pl.from_numpy(np.asarray(features))
        if isinstance(labels, pl.DataFrame):
            labels_df = labels
        elif _is_pandas_dataframe(labels):
            labels_df: pl.DataFrame = pl.from_pandas(labels)  # ty: ignore[invalid-assignment]
        else:
            labels_df = pl.from_dict(labels)
        if len(features_df) != len(labels_df):
            msg = f"`features` and `labels` must have the same length, got {len(features_df)} and {len(labels_df)}"
            raise ValueError(msg)
        collisions = sorted(set(features_df.columns) & set(labels_df.columns))
        if collisions:
            msg = (
                f"`labels` uses column name(s) {collisions} that collide with the auto-generated feature "
                f"column names ('column_0', 'column_1', ...). Rename the offending label column(s)."
            )
            raise ValueError(msg)
        return cls.from_dataframe(pl.concat((features_df, labels_df), how="horizontal"), features_df.columns)


def dummy_dataset_from_item(item: str | Path, frequency: int | str | Decimal | None) -> Dataset:
    """To debug."""
    labels = read_labels(item, "#file", "onset", "offset").with_columns(pl.lit(0).alias("dummy"))
    if frequency is not None:
        labels = labels.with_columns(*item_frontiers(frequency, "onset", "offset"))
    return Dataset.from_dataframe(labels, "dummy")
