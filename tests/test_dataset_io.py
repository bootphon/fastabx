"""Tests for ``fastabx.dataset`` — constructors, accessors, and on-disk I/O paths."""

import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from fastabx import Dataset
from fastabx.dataset import (
    Batch,
    EmptyFeaturesError,
    FeaturesSizeError,
    FrequencyTypeError,
    InMemoryAccessor,
    InvalidItemFileError,
    NonFiniteError,
    TimesArrayDimensionError,
    TimesArrayFrontiersError,
    decimal_frequency,
    find_all_files,
    item_frontiers,
    load_data_from_item,
    load_data_from_item_with_times,
    normalize_with_singularity_,
    read_labels,
)


def test_batch_repr() -> None:
    b = Batch(torch.zeros(2, 3, 4), torch.tensor([3, 2], dtype=torch.int32))
    rep = repr(b)
    assert "Batch" in rep
    assert "shape" in rep


def test_from_dict_basic() -> None:
    ds = Dataset.from_dict(
        {"x0": [1.0, 2.0, 3.0], "x1": [0.5, 0.5, 0.5], "phone": ["a", "b", "c"]},
        feature_columns=["x0", "x1"],
    )
    assert ds.labels.columns == ["phone"]
    assert ds.accessor.data.shape == (3, 2)
    assert len(ds.accessor) == 3


def test_from_dicts_basic() -> None:
    ds = Dataset.from_dicts(
        [{"x0": 1.0, "x1": 2.0, "phone": "a"}, {"x0": 3.0, "x1": 4.0, "phone": "b"}],
        feature_columns=["x0", "x1"],
    )
    assert len(ds.accessor) == 2


def test_from_csv_basic(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("x0,x1,phone\n1,2,a\n3,4,b\n")
    ds = Dataset.from_csv(path, feature_columns=["x0", "x1"])
    assert len(ds.accessor) == 2
    assert ds.labels.columns == ["phone"]


def test_from_numpy_length_mismatch_raises() -> None:
    features = np.zeros((3, 2), dtype=np.float32)
    labels = {"phone": ["a", "b"]}
    with pytest.raises(ValueError):  # noqa: PT011 - raised without a message
        Dataset.from_numpy(features, labels)


def test_in_memory_accessor_iter_and_getitem() -> None:
    data = torch.arange(20, dtype=torch.float32).view(10, 2)
    indices = {i: (i, i + 1) for i in range(10)}
    acc = InMemoryAccessor(indices, data)
    assert len(acc) == 10
    items = list(acc)
    assert len(items) == 10
    assert torch.equal(items[0], data[0:1])
    with pytest.raises(IndexError):
        _ = acc[42]
    rep = repr(acc)
    assert "InMemoryAccessor" in rep


def test_in_memory_accessor_lengths_and_batched() -> None:
    # Build variable-length entries.
    lengths = [2, 3, 1, 4]
    cursor = 0
    indices = {}
    for i, length in enumerate(lengths):
        indices[i] = (cursor, cursor + length)
        cursor += length
    data = torch.arange(cursor * 2, dtype=torch.float32).view(cursor, 2)
    acc = InMemoryAccessor(indices, data)
    assert list(acc.lengths([0, 1, 2, 3])) == lengths
    batch = acc.batched([0, 1])
    assert batch.data.shape == (2, 3, 2)  # padded to max length
    # Padded entries beyond actual size are zero.
    assert torch.equal(batch.data[0, 2], torch.zeros(2))


def test_normalize_with_singularity_basic_case() -> None:
    x = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    out = normalize_with_singularity_(x.clone())
    # Width grows by 1; first row normalised to unit; second row is the singularity.
    assert out.shape == (2, 3)
    torch.testing.assert_close(out[0, 0].item(), 0.6)
    torch.testing.assert_close(out[0, 1].item(), 0.8)
    torch.testing.assert_close(out[0, 2].item(), 1e-12, atol=1e-13, rtol=0)
    # Zero row: 1/sqrt(2) for each feature, -2*eps in the border.
    import math

    torch.testing.assert_close(out[1, 0].item(), 1.0 / math.sqrt(2))
    assert out[1, 2].item() < 0


def test_read_labels_item(tmp_path: Path) -> None:
    path = tmp_path / "data.item"
    path.write_text("#file onset offset phone\nf1 0.1 0.2 a\nf1 0.3 0.4 b\n")
    df = read_labels(path, "#file", "onset", "offset")
    assert df.columns == ["#file", "onset", "offset", "phone"]
    assert df["onset"].dtype == pl.Decimal
    assert df["offset"].dtype == pl.Decimal


def test_read_labels_csv(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("#file,onset,offset,phone\nf1,0.1,0.2,a\n")
    df = read_labels(path, "#file", "onset", "offset")
    assert df.columns == ["#file", "onset", "offset", "phone"]


def test_read_labels_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps({"#file": "f1", "onset": "0.1", "offset": "0.2", "phone": "a"}) + "\n")
    df = read_labels(path, "#file", "onset", "offset")
    assert df.columns == ["#file", "onset", "offset", "phone"]


def test_read_labels_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "data.bogus"
    path.write_text("noop")
    with pytest.raises(InvalidItemFileError, match="not supported"):
        read_labels(path, "#file", "onset", "offset")


def test_decimal_frequency_accepts_int_str_decimal() -> None:
    assert decimal_frequency(50) == Decimal(50)
    assert decimal_frequency("50.5") == Decimal("50.5")
    assert decimal_frequency(Decimal(100)) == Decimal(100)
    with pytest.raises(FrequencyTypeError):
        decimal_frequency(50.0)  # ty: ignore[invalid-argument-type]


def test_item_frontiers_basic() -> None:
    # frequency=10 Hz: (onset*10 - 0.5).ceil() / (offset*10 - 0.5).floor()
    df = pl.DataFrame(
        {
            "onset": [Decimal("0.1"), Decimal("0.3")],
            "offset": [Decimal("0.3"), Decimal("0.5")],
        }
    )
    start, end, left, right = item_frontiers(10, "onset", "offset")
    out = df.select(start, end, left, right)
    assert out["start"].to_list() == [1, 3]
    assert out["end"].to_list() == [3, 5]  # +1 vs floor because librilight bug is off by default


def test_find_all_files(tmp_path: Path) -> None:
    (tmp_path / "a.pt").write_bytes(b"x")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.pt").write_bytes(b"x")
    (tmp_path / "skip.txt").write_bytes(b"x")
    found = find_all_files(tmp_path, ".pt")
    assert set(found.keys()) == {"a", "sub/b"}


def test_load_data_from_item_missing_file_raises() -> None:
    labels = pl.DataFrame(
        {
            "#file": ["only_file"],
            "onset": [Decimal("0.0")],
            "offset": [Decimal("0.1")],
        }
    )
    mapping: dict[str, Path] = {}  # missing the expected file

    def loader(_p: Path) -> torch.Tensor:
        return torch.zeros(10, 3)

    with pytest.raises(FileNotFoundError, match="missing"):
        load_data_from_item(mapping, labels, 50, loader, "#file", "onset", "offset")


def test_load_data_from_item_non_finite_raises() -> None:
    labels = pl.DataFrame(
        {
            "#file": ["f1"],
            "onset": [Decimal("0.0")],
            "offset": [Decimal("0.1")],
        }
    )
    bad = torch.full((10, 3), float("nan"))

    def loader(_p: str) -> torch.Tensor:
        return bad

    with pytest.raises(NonFiniteError, match="f1"):
        load_data_from_item({"f1": "anything"}, labels, 50, loader, "#file", "onset", "offset")


def test_load_data_from_item_features_size_error() -> None:
    labels = pl.DataFrame(
        {
            "#file": ["f1"],
            "onset": [Decimal("0.0")],
            "offset": [Decimal("10.0")],  # would slice into [0, 500[ at 50 Hz
        }
    )

    def loader(_p: str) -> torch.Tensor:
        return torch.zeros(5, 3)  # way too short

    with pytest.raises(FeaturesSizeError, match="f1"):
        load_data_from_item({"f1": "anything"}, labels, 50, loader, "#file", "onset", "offset")


def test_load_data_from_item_empty_features_error() -> None:
    labels = pl.DataFrame(
        {
            "#file": ["f1", "f1"],
            "onset": [Decimal("0.0"), Decimal("0.1")],
            "offset": [Decimal("0.0001"), Decimal("0.2")],  # first window too short → start >= end
        }
    )

    def loader(_p: str) -> torch.Tensor:
        return torch.zeros(100, 3)

    with pytest.raises(EmptyFeaturesError):
        load_data_from_item({"f1": "anything"}, labels, 50, loader, "#file", "onset", "offset")


def test_load_data_from_item_with_times_dimension_error(tmp_path: Path) -> None:
    features_path = tmp_path / "f1.pt"
    times_path = tmp_path / "f1_times.pt"
    torch.save(torch.zeros(10, 3), features_path)
    torch.save(torch.zeros(10, 2), times_path)  # 2D times — invalid
    labels = pl.DataFrame(
        {
            "#file": ["f1"],
            "onset": [Decimal("0.0")],
            "offset": [Decimal("0.1")],
        }
    )
    with pytest.raises(TimesArrayDimensionError):
        load_data_from_item_with_times(
            {"f1": features_path},
            {"f1": times_path},
            labels,
            torch.load,
            torch.load,
            "#file",
            "onset",
            "offset",
        )


def test_load_data_from_item_with_times_happy_path(tmp_path: Path) -> None:
    features_path = tmp_path / "f1.pt"
    times_path = tmp_path / "f1_times.pt"
    torch.save(torch.arange(30, dtype=torch.float32).view(10, 3), features_path)
    torch.save(torch.linspace(0.0, 1.0, 10), times_path)
    labels = pl.DataFrame(
        {
            "#file": ["f1", "f1"],
            "onset": [Decimal("0.10"), Decimal("0.50")],
            "offset": [Decimal("0.40"), Decimal("0.90")],
        }
    )
    indices, data = load_data_from_item_with_times(
        {"f1": features_path},
        {"f1": times_path},
        labels,
        torch.load,
        torch.load,
        "#file",
        "onset",
        "offset",
    )
    assert set(indices.keys()) == {0, 1}
    for start, end in indices.values():
        assert end > start
    assert data.shape[1] == 3


def test_from_item_with_times_end_to_end(tmp_path: Path) -> None:
    item = tmp_path / "data.item"
    item.write_text("#file onset offset phone\nf1 0.10 0.40 a\nf1 0.50 0.90 b\n")
    feats_dir = tmp_path / "feats"
    times_dir = tmp_path / "times"
    feats_dir.mkdir()
    times_dir.mkdir()
    torch.save(torch.arange(30, dtype=torch.float32).view(10, 3), feats_dir / "f1.pt")
    torch.save(torch.linspace(0.0, 1.0, 10), times_dir / "f1.pt")
    ds = Dataset.from_item_with_times(item, feats_dir, times_dir)
    assert ds.labels.height == 2
    assert ds.accessor.data.shape[1] == 3


def test_load_data_from_item_with_times_missing_file_raises() -> None:
    labels = pl.DataFrame(
        {
            "#file": ["only_file"],
            "onset": [Decimal("0.0")],
            "offset": [Decimal("0.1")],
        }
    )

    def loader(_p: str) -> torch.Tensor:
        return torch.zeros(10, 3)

    with pytest.raises(FileNotFoundError, match="missing"):
        load_data_from_item_with_times({}, {}, labels, loader, loader, "#file", "onset", "offset")


def test_load_data_from_item_with_times_non_finite_raises() -> None:
    labels = pl.DataFrame(
        {
            "#file": ["f1"],
            "onset": [Decimal("0.0")],
            "offset": [Decimal("0.1")],
        }
    )

    def feature_loader(_p: str) -> torch.Tensor:
        return torch.full((10, 3), float("nan"))

    def time_loader(_p: str) -> torch.Tensor:
        return torch.linspace(0.0, 1.0, 10)

    with pytest.raises(NonFiniteError, match="f1"):
        load_data_from_item_with_times(
            {"f1": "anything"},
            {"f1": "anything"},
            labels,
            feature_loader,
            time_loader,
            "#file",
            "onset",
            "offset",
        )


def test_load_data_from_item_with_times_frontiers_error(tmp_path: Path) -> None:
    features_path = tmp_path / "f1.pt"
    times_path = tmp_path / "f1_times.pt"
    torch.save(torch.zeros(10, 3), features_path)
    torch.save(torch.tensor([0.5, 0.6, 0.7]), times_path)  # no times in [0, 0.1]
    labels = pl.DataFrame(
        {
            "#file": ["f1"],
            "onset": [Decimal("0.0")],
            "offset": [Decimal("0.1")],
        }
    )
    with pytest.raises(TimesArrayFrontiersError):
        load_data_from_item_with_times(
            {"f1": features_path},
            {"f1": times_path},
            labels,
            torch.load,
            torch.load,
            "#file",
            "onset",
            "offset",
        )


def test_from_item_end_to_end(tmp_path: Path) -> None:
    item = tmp_path / "data.item"
    item.write_text("#file onset offset phone speaker\nf1 0.00 0.10 a s1\nf1 0.10 0.20 b s1\nf2 0.00 0.10 a s2\n")
    feats_dir = tmp_path / "feats"
    feats_dir.mkdir()
    torch.save(torch.arange(60, dtype=torch.float32).view(20, 3), feats_dir / "f1.pt")
    torch.save(torch.arange(60, 90, dtype=torch.float32).view(10, 3), feats_dir / "f2.pt")
    ds = Dataset.from_item(item, feats_dir, frequency=50)
    assert ds.labels.height == 3
    assert ds.accessor.data.shape[1] == 3


def test_dataset_repr_contains_labels_and_accessor() -> None:
    ds = Dataset.from_dict({"x0": [1.0, 2.0], "phone": ["a", "b"]}, feature_columns="x0")
    rep = repr(ds)
    assert "labels" in rep
    assert "accessor" in rep
    assert "InMemoryAccessor" in rep


def test_dummy_dataset_from_item_with_and_without_frequency(tmp_path: Path) -> None:
    from fastabx.dataset import dummy_dataset_from_item

    item = tmp_path / "data.item"
    item.write_text("#file onset offset phone\nf1 0.0 0.1 a\nf1 0.1 0.2 b\n")
    ds_no_freq = dummy_dataset_from_item(item, frequency=None)
    assert "#file" in ds_no_freq.labels.columns
    ds_with_freq = dummy_dataset_from_item(item, frequency=50)
    assert "start" in ds_with_freq.labels.columns
    assert "end" in ds_with_freq.labels.columns


def test_from_item_and_units(tmp_path: Path) -> None:
    item = tmp_path / "data.item"
    item.write_text("#file onset offset phone\nf1 0.00 0.10 a\nf1 0.10 0.20 b\n")
    units_path = tmp_path / "units.jsonl"
    units_path.write_text(json.dumps({"audio": "f1.wav", "units": list(range(20))}) + "\n")
    ds = Dataset.from_item_and_units(item, units_path, frequency=50)
    assert ds.labels.height == 2
    # The unit tensor is unsqueezed to add a feature dim of 1.
    assert ds.accessor.data.shape[1] == 1
