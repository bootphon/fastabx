"""Tests for ``fastabx.alignment``."""

from typing import get_args

import pytest
import torch
from torch import Tensor
from torch.testing import assert_close
from torchdtw import dtw_batch

from fastabx.alignment import AlignmentName, alignment_function
from fastabx.dataset import Dataset
from fastabx.distance import abx_on_cell, distance_matrix, euclidean_distance
from fastabx.pooling import pool_dataset
from fastabx.score import Score
from fastabx.task import Task

NAMES: list[AlignmentName] = list(get_args(AlignmentName.__value__))


@pytest.mark.parametrize("name", NAMES)
def test_alignment_function_returns_callable(name: AlignmentName) -> None:
    assert callable(alignment_function(name))


def test_alignment_function_unknown_raises() -> None:
    with pytest.raises(ValueError, match="bogus"):
        alignment_function("bogus")  # ty: ignore[invalid-argument-type]


def test_alignment_function_passes_custom_callable_through() -> None:
    def custom(distances: Tensor, _sx: Tensor, _sy: Tensor, *, symmetric: bool) -> Tensor:
        assert symmetric is False
        return distances.amin(dim=(2, 3))

    assert alignment_function(custom) is custom


def test_dtw_alignment_is_normalized_by_path_length() -> None:
    """The protocol requires a path-normalized cost, so an all-ones lattice averages to 1 whatever the shape."""
    for s1, s2 in ((3, 3), (4, 2), (1, 6)):
        cost = torch.ones(1, 1, s1, s2)
        sx, sy = torch.tensor([s1], dtype=torch.int32), torch.tensor([s2], dtype=torch.int32)
        assert_close(dtw_batch(cost, sx, sy, symmetric=False).item(), 1.0)


def test_dtw_on_single_frame_lattice_is_the_frame_cost() -> None:
    """The contract behind the bypass: one frame against one frame leaves a single path, of length one."""
    cost = torch.rand(3, 4, 1, 1)
    sx, sy = torch.ones(3, dtype=torch.int32), torch.ones(4, dtype=torch.int32)
    assert torch.equal(dtw_batch(cost, sx, sy, symmetric=False), cost.squeeze(2, 3))


def test_distance_matrix_bypasses_alignment_when_pooled() -> None:
    """A 1x1 lattice never reaches the alignment, so even a custom one is not called."""

    def explode(_distances: Tensor, _sx: Tensor, _sy: Tensor, *, symmetric: bool) -> Tensor:
        raise AssertionError(symmetric)

    a, b = torch.randn(3, 1, 5), torch.randn(4, 1, 5)
    sa, sb = torch.ones(3, dtype=torch.int32), torch.ones(4, dtype=torch.int32)
    out = distance_matrix(a, sa, b, sb, euclidean_distance, alignment=explode, symmetric=False)
    assert_close(out, euclidean_distance(a, b).squeeze(2, 3))


def test_distance_matrix_uses_alignment_on_sequences() -> None:
    calls = []

    def counting(distances: Tensor, sx: Tensor, sy: Tensor, *, symmetric: bool) -> Tensor:
        calls.append(distances.shape)
        return dtw_batch(distances, sx, sy, symmetric=symmetric)

    a, b = torch.randn(2, 3, 5), torch.randn(4, 3, 5)
    sa, sb = torch.full((2,), 3, dtype=torch.int32), torch.full((4,), 3, dtype=torch.int32)
    out = distance_matrix(a, sa, b, sb, euclidean_distance, alignment=counting, symmetric=False)
    assert calls == [(2, 4, 3, 3)]
    assert out.shape == (2, 4)


def _pooled_task() -> Task:
    dataset = Dataset.from_dataframe(
        {
            "phone": ["a", "a", "b", "b", "a", "b"],
            "speaker": ["s1", "s2", "s1", "s2", "s1", "s2"],
            "f0": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "f1": [1.0, 0.0, 3.0, 2.0, 5.0, 4.0],
        },
        ["f0", "f1"],
    )
    return Task(pool_dataset(dataset, "mean"), on="phone", by=["speaker"])


def test_score_default_alignment_is_dtw() -> None:
    score = Score(_pooled_task(), "euclidean")
    assert score.alignment == "dtw"
    assert "dtw alignment" in repr(score)
    assert "euclidean distance" in repr(score)


def test_score_with_custom_alignment_matches_dtw_on_sequences(seq_dataset: Dataset) -> None:
    """A custom callable is accepted and threaded all the way down to the grouped engine."""
    task = Task(seq_dataset, on="phone", by=["speaker"])
    custom = Score(task, "euclidean", alignment=dtw_batch)
    assert "dtw_batch alignment" in repr(custom)
    expected = Score(Task(seq_dataset, on="phone", by=["speaker"]), "euclidean")
    assert custom.collapse(weighted=True) == expected.collapse(weighted=True)


def test_score_repr_with_callable_object_alignment(seq_dataset: Dataset) -> None:
    """A stateful alignment object is really invoked, and the repr falls back to its class name."""

    class Warping:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, distances: Tensor, sx: Tensor, sy: Tensor, *, symmetric: bool) -> Tensor:
            self.calls += 1
            return dtw_batch(distances, sx, sy, symmetric=symmetric)

    alignment = Warping()
    score = Score(Task(seq_dataset, on="phone", by=["speaker"]), "euclidean", alignment=alignment)
    assert "Warping alignment" in repr(score)
    assert alignment.calls > 0


def test_abx_on_cell_accepts_alignment(seq_dataset: Dataset) -> None:
    task = Task(seq_dataset, on="phone", by=["speaker"])
    cell = task[0]
    assert cell.needs_alignment
    assert_close(abx_on_cell(cell, "euclidean", alignment=dtw_batch), abx_on_cell(cell, "euclidean"))
    assert_close(abx_on_cell(cell, "euclidean", alignment=alignment_function("dtw")), abx_on_cell(cell, "euclidean"))
