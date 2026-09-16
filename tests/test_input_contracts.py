"""Regression tests for constructor precision and public input contracts."""

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from fastabx import (
    Dataset,
    EmptyDatasetError,
    InMemoryAccessor,
    InvalidDatasetError,
    InvalidFeatureDtypeError,
    InvalidFeaturesError,
    InvalidItemFileError,
    InvalidTimesError,
    NonFiniteError,
    PoolingName,
    PrecomputedCellsError,
    Score,
    Task,
    TimesArrayDimensionError,
    pool_dataset,
)
from fastabx.alignment import alignment_function
from fastabx.dataset import decimal_frequency, load_data_from_item, load_data_from_item_with_times
from fastabx.distance import distance_function, euclidean_distance
from tests.conftest import DEVICE


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int64])
@pytest.mark.parametrize("convert", [None, torch.float32, torch.float64])
def test_tabular_dtype_policy(dtype: torch.dtype, convert: torch.dtype | None) -> None:
    features = torch.arange(8, dtype=dtype).reshape(4, 2)
    labels = {"phone": ["a", "a", "b", "b"]}
    datasets = [
        Dataset.from_numpy(features.numpy(), labels, dtype=convert, device=DEVICE),
        Dataset.from_dataframe(
            pl.from_numpy(features.numpy()).with_columns(pl.Series("phone", labels["phone"])),
            ["column_0", "column_1"],
            dtype=convert,
            device=DEVICE,
        ),
    ]
    for dataset in datasets:
        assert dataset.accessor[0].dtype == (convert or dtype)
        torch.testing.assert_close(dataset.accessor[0], features[:1].to(device=DEVICE, dtype=convert))
        if (convert or dtype).is_floating_point:
            score = Score(Task(dataset, on="phone"), "euclidean", progress=False).collapse()
            assert 0 <= score <= 1


@pytest.mark.parametrize("dtype", [None, torch.float32, torch.float64])
def test_item_constructor_dtypes(tmp_path: Path, dtype: torch.dtype | None) -> None:
    item = tmp_path / "data.item"
    item.write_text("#file onset offset phone\nf 0.0 1.0 a\n")
    features, times = tmp_path / "features", tmp_path / "times"
    features.mkdir()
    times.mkdir()
    torch.save(torch.tensor([[1.0], [2.0]], dtype=torch.float64), features / "f.pt")
    torch.save(torch.tensor([0.25, 0.75]), times / "f.pt")
    units = tmp_path / "units.jsonl"
    units.write_text('{"audio":"f.wav","units":[16777217,16777218]}\n')
    datasets = [
        Dataset.from_item(item, features, 2, dtype=dtype, device=DEVICE, progress=False),
        Dataset.from_item_with_times(item, features, times, dtype=dtype, device=DEVICE, progress=False),
        Dataset.from_item_and_units(item, units, 2, dtype=dtype, device=DEVICE, progress=False),
    ]
    for dataset, original in zip(datasets, [torch.float64, torch.float64, torch.int64], strict=True):
        assert dataset.accessor[0].dtype == (dtype or original)
    if dtype is None:
        assert datasets[2].accessor[0][0].item() == 16777217


def _timestamp_dataset(times: torch.Tensor, labels: pl.DataFrame | None = None) -> tuple[dict, torch.Tensor]:
    """Load two frames with configurable timestamps and metadata."""
    if labels is None:
        labels = pl.DataFrame({"file": ["f"], "start": [Decimal("0.0")], "stop": [Decimal("1.00")]})
    return load_data_from_item_with_times(
        {"f": torch.tensor([[1.0], [2.0]])},
        {"f": times},
        labels,
        lambda x: x,
        lambda x: x,
        "file",
        "start",
        "stop",
        DEVICE,
        progress=False,
    )


@pytest.mark.parametrize("times", [torch.tensor(0.5), torch.tensor([[0.0, 1.0]])])
def test_timestamp_requires_one_dimension(times: torch.Tensor) -> None:
    with pytest.raises(TimesArrayDimensionError):
        _timestamp_dataset(times)


@pytest.mark.parametrize(
    "times", [torch.tensor([0.0]), torch.tensor([0.0, float("nan")]), torch.tensor([0.0, float("inf")])]
)
def test_timestamp_length_and_finiteness(times: torch.Tensor) -> None:
    with pytest.raises(InvalidTimesError):
        _timestamp_dataset(times)


def test_timestamp_uses_both_boundary_precisions() -> None:
    labels = pl.DataFrame({"file": ["f"], "start": [Decimal("0.1")], "stop": [Decimal("0.15")]})
    indices, data = _timestamp_dataset(torch.tensor([0.1, 0.15]), labels)
    assert indices == {0: (0, 2)}
    assert data[:, 0].tolist() == [1.0, 2.0]


def test_timestamp_float_metadata_is_not_rounded_to_integer() -> None:
    labels = pl.DataFrame({"file": ["f"], "start": [0.1], "stop": [0.15]})
    _, data = _timestamp_dataset(torch.tensor([0.1, 0.15], dtype=torch.float64), labels)
    assert data.shape == (2, 1)


@pytest.mark.parametrize("frequency", [0, -1, "NaN", "Infinity", "-Infinity", "nope"])
def test_frequency_rejects_non_positive_or_non_finite(frequency: int | str) -> None:
    with pytest.raises(ValueError, match="frequency"):
        decimal_frequency(frequency)


@pytest.mark.parametrize(
    ("onset", "offset"), [(-1.0, 1.0), (2.0, 1.0), (None, 1.0), (0.0, float("inf")), (float("nan"), 1.0)]
)
def test_invalid_intervals(onset: float | None, offset: float) -> None:
    labels = pl.DataFrame({"file": ["f"], "start": [onset], "stop": [offset]})
    with pytest.raises(InvalidItemFileError, match="intervals"):
        _timestamp_dataset(torch.tensor([0.0, 1.0]), labels)


def test_empty_item_metadata() -> None:
    labels = pl.DataFrame(schema={"file": pl.String, "start": pl.Float64, "stop": pl.Float64})
    with pytest.raises(EmptyDatasetError):
        _timestamp_dataset(torch.tensor([0.0, 1.0]), labels)


def test_callable_dataclass_distance(tiny_dataset: Dataset) -> None:
    @dataclass
    class CustomDistance:
        def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return euclidean_distance(a, b)

    task = Task(tiny_dataset, on="phone")
    actual = Score(task, CustomDistance(), progress=False).collapse()
    expected = Score(task, "euclidean", progress=False).collapse()
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("resolver", [distance_function, alignment_function])
def test_non_callable_configuration_rejected(resolver: Callable) -> None:
    with pytest.raises(TypeError, match="callable"):
        resolver(42)


@pytest.mark.parametrize("shape", [(4,), (2, 0), (2, 2, 1)])
def test_invalid_feature_dimensions(shape: tuple[int, ...]) -> None:
    with pytest.raises(InvalidFeaturesError, match="shape"):
        InMemoryAccessor({0: (0, 1)}, torch.zeros(shape), DEVICE)


@pytest.mark.parametrize("interval", [(-1, 1), (0, 3)])
def test_accessor_slice_bounds(interval: tuple[int, int]) -> None:
    with pytest.raises(InvalidFeaturesError, match="slices"):
        InMemoryAccessor({0: interval}, torch.zeros(2, 1), DEVICE)


def test_dataset_label_accessor_lengths() -> None:
    accessor = InMemoryAccessor({0: (0, 1)}, torch.zeros(1, 1), DEVICE)
    with pytest.raises(InvalidDatasetError, match="same length"):
        Dataset(pl.DataFrame({"phone": ["a", "b"]}), accessor)


@pytest.mark.parametrize("indices", [None, [0, None]])
@pytest.mark.parametrize("column", ["index_a", "index_b", "index_x"])
def test_null_precomputed_indices(tiny_dataset: Dataset, indices: list[int | None] | None, column: str) -> None:
    cells = Task(tiny_dataset, on="phone").cells.head(1)
    cells = cells.with_columns(pl.Series(column, [indices], dtype=pl.List(pl.Int64)))
    with pytest.raises(PrecomputedCellsError, match="null"):
        Task.from_cells(tiny_dataset, cells, is_symmetric=True)


def test_duplicate_unit_identifiers(tmp_path: Path) -> None:
    item, units = tmp_path / "data.item", tmp_path / "units.jsonl"
    item.write_text("#file onset offset phone\nf 0.0 1.0 a\n")
    units.write_text('{"audio":"dir1/f.wav","units":[1,2]}\n{"audio":"dir2/f.wav","units":[3,4]}\n')
    with pytest.raises(InvalidItemFileError, match="duplicate"):
        Dataset.from_item_and_units(item, units, 2, progress=False)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("pooling", ["mean", "hamming"])
def test_pooling_float_dtypes(dtype: torch.dtype, pooling: PoolingName) -> None:
    accessor = InMemoryAccessor({0: (0, 2)}, torch.tensor([[1.0], [3.0]], dtype=dtype), DEVICE)
    pooled = pool_dataset(Dataset(pl.DataFrame({"phone": ["a"]}), accessor), pooling)
    assert pooled.accessor[0].dtype == dtype
    assert pooled.accessor[0].item() == pytest.approx(2.0)


def test_item_feature_dimension_consistency() -> None:
    labels = pl.DataFrame({"file": ["a", "b"], "start": [0.0, 0.0], "stop": [1.0, 1.0]})
    with pytest.raises(InvalidFeaturesError, match="consistent"):
        load_data_from_item(
            {"a": torch.zeros(2, 1), "b": torch.zeros(2, 2)},
            labels,
            2,
            lambda x: x,
            "file",
            "start",
            "stop",
            DEVICE,
            progress=False,
        )


def test_non_finite_input_rejected_before_integer_conversion() -> None:
    with pytest.raises(NonFiniteError):
        Dataset.from_numpy([[float("nan")]], {"phone": ["a"]}, dtype=torch.int64)


def test_numpy_requires_matrix() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        Dataset.from_numpy(np.zeros(2), {"phone": ["a", "b"]})


@pytest.mark.parametrize("index", [0.0, True])
def test_accessor_rejects_non_integer_row_keys(*, index: float | bool) -> None:
    with pytest.raises(InvalidDatasetError, match="integer row"):
        InMemoryAccessor({index: (0, 1)}, torch.zeros(1, 1), DEVICE)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("boundary", [0.5, True])
def test_accessor_rejects_non_integer_slice_bounds(*, boundary: float | bool) -> None:
    with pytest.raises(InvalidFeaturesError, match="integers"):
        InMemoryAccessor({0: (boundary, 2)}, torch.zeros(2, 1), DEVICE)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    "content",
    ["#file onset offset phone\n", "#file onset phone\nf 0.0 a\n", "#file onset offset phone\nf nope 1.0 a\n"],
)
def test_invalid_item_metadata_fails_before_loading(tmp_path: Path, content: str) -> None:
    item = tmp_path / "data.item"
    item.write_text(content)
    with pytest.raises((EmptyDatasetError, InvalidItemFileError)):
        Dataset.from_item(item, tmp_path, 2, progress=False)


def test_custom_timestamp_columns_end_to_end(tmp_path: Path) -> None:
    item = tmp_path / "data.csv"
    item.write_text("file,start,stop,phone\nf,0.1,0.15,a\n")
    features, times = tmp_path / "features", tmp_path / "times"
    features.mkdir()
    times.mkdir()
    torch.save(torch.tensor([[1.0], [2.0]]), features / "f.pt")
    torch.save(torch.tensor([0.1, 0.15]), times / "f.pt")
    dataset = Dataset.from_item_with_times(
        item,
        features,
        times,
        file_col="file",
        onset_col="start",
        offset_col="stop",
        progress=False,
    )
    assert dataset.accessor[0][:, 0].tolist() == [1.0, 2.0]


@pytest.mark.parametrize("pooling", ["mean", "hamming"])
def test_integer_pooling_rejected(pooling: PoolingName) -> None:
    dataset = Dataset.from_numpy([[1], [2]], {"phone": ["a", "b"]})
    with pytest.raises(RuntimeError, match="floating point"):
        pool_dataset(dataset, pooling)


def test_integer_angular_score_rejected_before_mutation() -> None:
    dataset = Dataset.from_numpy([[1], [2], [3], [4]], {"phone": ["a", "a", "b", "b"]})
    with pytest.raises(InvalidFeatureDtypeError):
        Score(Task(dataset, on="phone"), "angular", progress=False)
    assert not dataset.accessor.is_normalized


@pytest.mark.parametrize("symmetric", [False, True])
def test_precomputed_lists_count_positions_including_duplicates(*, symmetric: bool) -> None:
    dataset = Dataset.from_numpy([[0.0], [2.0]], {"phone": ["a", "b"]})
    cells = pl.DataFrame(
        {"header": ["h"], "description": ["d"], "index_a": [[0, 0]], "index_b": [[0, 1]], "index_x": [[0, 0]]}
    )
    task = Task.from_cells(dataset, cells, is_symmetric=symmetric)
    score = Score(task, "euclidean", progress=False)
    # d(X,A)=0; half the B positions tie at zero, the other half are farther away.
    assert score.collapse(weighted=True) == pytest.approx(0.25)
    assert score.cells["size"][0] == (4 if symmetric else 8)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("distance", ["euclidean", "angular", "kl_symmetric"])
def test_sequence_scoring_preserves_supported_feature_precision(
    seq_dataset: Dataset, dtype: torch.dtype, distance: str
) -> None:
    original = seq_dataset.accessor
    assert isinstance(original, InMemoryAccessor)
    features = original.data.to(dtype=dtype).softmax(dim=1)
    dataset = Dataset(seq_dataset.labels, InMemoryAccessor(original.indices, features, DEVICE))
    score = Score(Task(dataset, on="phone"), distance, progress=False)  # ty: ignore[invalid-argument-type]
    assert 0 <= score.collapse() <= 1
    assert dataset.accessor[0].dtype == dtype
