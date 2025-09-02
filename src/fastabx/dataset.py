"""Data utilities."""

import abc
import math
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
import numpy.typing as npt
import polars as pl
import polars.selectors as cs
import torch
from polars.interchange.protocol import SupportsInterchange
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from fastabx.utils import with_librilight_bug
from fastabx.verify import verify_empty_datapoints

type ArrayLike = npt.ArrayLike


@dataclass(frozen=True)
class Batch:
    """Batch of padded data."""

    data: torch.Tensor
    sizes: torch.Tensor

    def __repr__(self) -> str:
        return f"Batch(data=Tensor(shape={self.data.shape}, dtype={self.data.dtype}), sizes={self.sizes})"


class DataAccessor(abc.ABC):
    """Abstract class for data accessors.

    A data accessor is a way to access a torch.Tensor given an index.
    """

    @abc.abstractmethod
    def __getitem__(self, i: int) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def __iter__(self) -> Iterator[torch.Tensor]:
        pass

    @abc.abstractmethod
    def batched(self, indices: Iterator[int]) -> Batch:
        """Get the padded data and the original sizes of the data from a list of indices."""


class InMemoryAccessor(DataAccessor):
    """Data accessor where everything is in memory."""

    def __init__(self, indices: dict[int, tuple[int, int]], data: torch.Tensor) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.indices = indices
        verify_empty_datapoints(self.indices)
        self.data = data.to(self.device)

    def __repr__(self) -> str:
        return f"InMemoryAccessor(data of shape {tuple(self.data.shape)}, with {len(self)} items)"

    def __getitem__(self, i: int) -> torch.Tensor:
        if i not in self.indices:
            raise IndexError
        start, end = self.indices[i]
        return self.data[start:end]

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self) -> Iterator[torch.Tensor]:
        for i in self.indices:
            yield self[i]

    def batched(self, indices: Iterator[int]) -> Batch:
        """Get the padded data and the original sizes of the data from a list of indices."""
        sizes, data = [], []
        for i in indices:
            this_data = self[i]
            sizes.append(this_data.size(0))
            data.append(this_data)
        return Batch(pad_sequence(data, batch_first=True), torch.tensor(sizes, dtype=torch.int64, device=self.device))


def find_all_files(root: str | Path, extension: str) -> dict[str, Path]:
    """Recursively find all files with the given `extension` in `root`."""
    return dict(sorted((p.stem, p) for p in Path(root).rglob(f"*{extension}")))


def normalize_with_singularity(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize the given vector across the third dimension.

    Extend all vectors by eps to put the null vector at the maximal
    angular distance from any non-null vector.
    """
    norm = torch.norm(x, dim=1, keepdim=True)
    zero_vals = norm == 0
    x = torch.where(zero_vals, 1 / math.sqrt(x.size(1)), x / norm)
    border = torch.full((x.size(0), 1), eps, dtype=x.dtype, device=x.device)
    border = torch.where(zero_vals, -2 * eps, border)
    return torch.cat([x, border], dim=1)


class InvalidItemFileError(Exception):
    """The item file does not have the correct columns."""


def read_item(item: str | Path) -> pl.DataFrame:
    """Read an item file."""
    schema = {
        "#file": pl.String,
        "onset": pl.String,
        "offset": pl.String,
        "#phone": pl.String,
        "prev-phone": pl.String,
        "next-phone": pl.String,
        "speaker": pl.String,
    }
    try:
        df = pl.read_csv(item, separator=" ", schema=schema)
        return df.with_columns(df["onset"].str.to_decimal(), df["offset"].str.to_decimal())
    except pl.exceptions.ComputeError as error:
        raise InvalidItemFileError from error


def read_labels(item: str | Path, file_col: str, onset_col: str, offset_col: str) -> pl.DataFrame:
    """Return the labels from the path to the item file."""
    schema_overrides = {file_col: pl.String, onset_col: pl.String, offset_col: pl.String}
    match ext := Path(item).suffix:
        case ".item":
            return read_item(item)
        case ".csv":
            df = pl.read_csv(item, schema_overrides=schema_overrides)
            return df.with_columns(df[onset_col].str.to_decimal(), df[offset_col].str.to_decimal())
        case ".jsonl" | ".ndjson":
            df = pl.read_ndjson(item, schema_overrides=schema_overrides)
            return df.with_columns(df[onset_col].str.to_decimal(), df[offset_col].str.to_decimal())
        case _:
            msg = f"File extension {ext} is not supported. Supported extensions are .item, .csv, .jsonl, .ndjson."
            raise InvalidItemFileError(msg)


def item_frontiers(frequency: float, onset_col: str, offset_col: str) -> tuple[pl.Expr, pl.Expr, pl.Expr, pl.Expr]:
    """Frontiers [start, end[ in the input features and in the concatenated ones."""
    # Remove the cast to float and -0.5 once RoundMode "half_ceil" and "half_floor" are added to polars round method
    # (or "half_away_from_zero" and "half_to_zero")
    # See: https://github.com/pola-rs/polars/issues/21800
    start = (pl.col(onset_col) * frequency - 0.5).cast(pl.Float64).ceil().cast(pl.Int64).alias("start")
    end = (pl.col(offset_col) * frequency - 0.5).cast(pl.Float64).floor().cast(pl.Int64).alias("end")
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


def load_data_from_item[T](
    mapping: dict[str, T],
    labels: pl.DataFrame,
    frequency: int,
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

    data, device = [], torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for fileid, start_indices, end_indices in tqdm(by_file.iter_rows(), desc="Building dataset", total=len(by_file)):
        features = feature_maker(mapping[fileid]).detach().to(device)
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


def load_data_from_item_with_times(
    paths_features: dict[str, Path],
    paths_times: dict[str, Path],
    labels: pl.DataFrame,
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
    data, device, all_indices, right = [], torch.device("cuda" if torch.cuda.is_available() else "cpu"), {}, 0
    decimals = by_file["onset"].dtype.inner.scale  # ty: ignore[unresolved-attribute]
    for fileid, indices, onsets, offsets in tqdm(by_file.iter_rows(), desc="Building dataset", total=len(by_file)):
        features = torch.load(paths_features[fileid], map_location=device).detach()
        times = torch.load(paths_times[fileid]).round(decimals=decimals)
        if times.ndim > 1:
            raise TimesArrayDimensionError
        for index, onset, offset in zip(indices, onsets, offsets, strict=True):
            mask = torch.where(torch.logical_and(float(onset) <= times, times <= float(offset)))[0]  # ty: ignore[invalid-argument-type]
            if not mask.any():
                raise TimesArrayFrontiersError(fileid, float(onset), float(offset))
            data.append(features[mask])
            left = right
            right += len(mask)
            all_indices[index] = (left, right)
    return all_indices, torch.cat(data, dim=0)


@dataclass(frozen=True)
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
        """L2 normalization of the data."""
        self.accessor.data = normalize_with_singularity(self.accessor.data.cpu()).to(self.accessor.device)
        return self

    @classmethod
    def from_item(
        cls,
        item: str | Path,
        root: str | Path,
        frequency: int,
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
        features: str | Path,
        times: str | Path,
        *,
        file_col: str = "#file",
        onset_col: str = "onset",
        offset_col: str = "offset",
    ) -> "Dataset":
        """Create a dataset from an item file.

        Use arrays containing the times associated to the features instead of a given frequency.

        :param item: Path to the item file.
        :param features: Path to the root directory containing either the features or the audio files.
        :param times: Path to the root directory containing the times arrays.
        :param file_col: Column in the item file that contains the audio file names, default is "#file".
        :param onset_col: Column in the item file that contains the onset times, default is "onset".
        :param offset_col: Column in the item file that contains the offset times, default is "offset".
        """
        labels = read_labels(item, file_col, onset_col, offset_col)
        paths_feat = find_all_files(features, ".pt")
        paths_time = find_all_files(times, ".pt")
        indices, data = load_data_from_item_with_times(paths_feat, paths_time, labels, file_col, onset_col, offset_col)
        return Dataset(labels=labels, accessor=InMemoryAccessor(indices, data))

    @classmethod
    def from_item_and_units(
        cls,
        item: str | Path,
        units: str | Path,
        frequency: int,
        *,
        audio_key: str = "audio",
        units_key: str = "units",
        separator: str = " ",
        file_col: str = "#file",
        onset_col: str = "onset",
        offset_col: str = "offset",
    ) -> "Dataset":
        """Create a dataset from an item file with the units all described in a single JSONL file.

        :param item: Path to the item file.
        :param units: Path to the JSONL file containing the units.
        :param frequency: The feature frequency, in Hz.
        :param audio_key: Key in the JSONL file that contains the audio file names, default is "audio".
        :param units_key: Key in the JSONL file that contains the units, default is "units".
        :param separator: Separator used in the units field, default is whitespace " ".
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
            return torch.tensor([int(unit) for unit in units_df[idx, units_key].split(separator)]).unsqueeze(1)

        mapping: dict[str, int] = dict(zip(units_df[audio_key], range(len(units_df)), strict=True))
        indices, data = load_data_from_item(mapping, labels, frequency, feature_maker, file_col, onset_col, offset_col)
        return Dataset(labels=labels, accessor=InMemoryAccessor(indices, data))

    @classmethod
    def from_dataframe(cls, df: SupportsInterchange, feature_columns: str | Collection[str]) -> "Dataset":
        """Create a dataset from a DataFrame (polars or pandas).

        :param df: DataFrame containing both the labels and the features.
        :param feature_columns: Column name or list of column names containing the features.
        """
        df = pl.from_dataframe(df.__dataframe__())
        labels = df.select(cs.exclude(feature_columns))
        indices = {i: (i, i + 1) for i in range(len(labels))}
        data = df.select(feature_columns).cast(pl.Float32).to_torch()
        return Dataset(labels=labels, accessor=InMemoryAccessor(indices, data))

    @classmethod
    def from_csv(cls, path: str | Path, feature_columns: str | Collection[str], *, separator: str = ",") -> "Dataset":
        """Create a dataset from a CSV file.

        :param path: Path to the CSV file containing both the labels and the features.
        :param feature_columns: Column name or list of column names containing the features.
        :param separator: Separator used in the CSV file.
        """
        return cls.from_dataframe(pl.read_csv(path, separator=separator), feature_columns)

    @classmethod
    def from_dict(cls, data: Mapping[str, Sequence[object]], feature_columns: str | Collection[str]) -> "Dataset":
        """Create a dataset from a dictionary of sequences.

        :param data: Dictionary of sequences containing both the labels and the features.
        :param feature_columns: Column name or list of column names containing the features.
        """
        return cls.from_dataframe(pl.from_dict(data), feature_columns)

    @classmethod
    def from_dicts(cls, data: Iterable[dict[str, Any]], feature_columns: str | Collection[str]) -> "Dataset":
        """Create a dataset from a sequence of dictionaries.

        :param data: Sequence of dictionaries containing both the labels and the features.
        :param feature_columns: Column name or list of column names containing the features.
        """
        return cls.from_dataframe(pl.from_dicts(data), feature_columns)

    @classmethod
    def from_numpy(
        cls,
        features: ArrayLike,
        labels: Mapping[str, Sequence[object]] | SupportsInterchange,
    ) -> "Dataset":
        """Create a dataset from the features (numpy array) and the labels (dictionary of sequences).

        :param features: 2D array-like containing the features.
        :param labels: Dictionary of sequences or DataFrame containing the labels.
        """
        features_df = pl.from_numpy(np.asarray(features))
        labels_df = (
            pl.from_dataframe(labels.__dataframe__()) if hasattr(labels, "__dataframe__") else pl.from_dict(labels)
        )
        if len(features_df) != len(labels_df):
            raise ValueError
        return cls.from_dataframe(pl.concat((features_df, labels_df), how="horizontal"), features_df.columns)


def dummy_dataset_from_item(item: str | Path, frequency: int | None) -> Dataset:
    """To debug."""
    labels = read_item(item).with_columns(pl.lit(0).alias("dummy"))
    if frequency is not None:
        labels = labels.with_columns(*item_frontiers(frequency, "onset", "offset"))
    return Dataset.from_dataframe(labels, "dummy")
