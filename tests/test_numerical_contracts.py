"""Mathematical scoring contracts independent of the implementation's intermediate representation."""

import math

import polars as pl
import pytest
import torch
from torch.testing import assert_close
from torchdtw import dtw_batch

from fastabx import Dataset, InMemoryAccessor, Score, Task, abx_on_cell
from fastabx.accessor import normalize_with_singularity
from fastabx.distance import DistanceName, angular_distance, euclidean_distance
from fastabx.group import grouped_distances
from tests.conftest import DEVICE

FLOAT_DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_angular_zero_semantics(dtype: torch.dtype) -> None:
    features = torch.tensor([[0, 0], [0, 0], [1, 0], [0, 1], [-1, 0], [1, 1]], dtype=dtype, device=DEVICE)
    normalized = normalize_with_singularity(features)[:, None, :]
    distances = angular_distance(normalized, normalized)[:, :, 0, 0]
    assert torch.equal(distances[:2, :2], torch.zeros_like(distances[:2, :2]))
    assert torch.equal(distances[:2, 2:], torch.ones_like(distances[:2, 2:]))
    assert torch.equal(distances[2:, :2], torch.ones_like(distances[2:, :2]))
    assert distances[2, 3].item() == pytest.approx(0.5, abs=torch.finfo(dtype).eps)  # orthogonal
    assert distances[2, 4].item() == pytest.approx(1.0, abs=torch.finfo(dtype).eps)  # antipodal
    assert distances[2, 2].item() == pytest.approx(0.0, abs=0, rel=0)  # identical basis vector


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("scale_kind", ["subnormal", "tiny", "large"])
def test_normalization_preserves_directions_at_extreme_scales(dtype: torch.dtype, scale_kind: str) -> None:
    zero = torch.tensor(0.0, dtype=dtype, device=DEVICE)
    scale = {
        "subnormal": torch.nextafter(zero, torch.ones_like(zero)).item(),
        "tiny": torch.finfo(dtype).tiny,
        "large": torch.finfo(dtype).max / 2,
    }[scale_kind]
    features = torch.tensor([[scale, scale], [-scale, scale], [0, 0]], dtype=dtype, device=DEVICE)
    original = features.clone()
    normalized = normalize_with_singularity(features)
    expected = torch.tensor(
        [[1 / math.sqrt(2), 1 / math.sqrt(2)], [-1 / math.sqrt(2), 1 / math.sqrt(2)], [0, 0]],
        dtype=dtype,
        device=DEVICE,
    )
    assert normalized.dtype == dtype
    assert torch.isfinite(normalized).all()
    assert_close(normalized[:, :-1], expected)
    assert torch.equal(features, original)
    distances = angular_distance(normalized[:, None, :], normalized[:, None, :])
    assert distances[0, 1].item() == pytest.approx(0.5, abs=torch.finfo(dtype).eps)


@pytest.mark.parametrize("chunk_rows", [1, 2, 3, 8192])
@pytest.mark.parametrize("sequence_length", [1, 2])
@pytest.mark.parametrize("feature_dtype", [torch.float16, torch.float32, torch.float64, torch.int64])
def test_chunking_preserves_custom_result_dtype(
    chunk_rows: int, sequence_length: int, feature_dtype: torch.dtype
) -> None:
    x = torch.tensor([0, 1], dtype=feature_dtype, device=DEVICE)[:, None, None].repeat(1, sequence_length, 1)
    targets = torch.tensor([0, 1, 2, 3, 4], dtype=feature_dtype, device=DEVICE)[:, None, None].repeat(
        1, sequence_length, 1
    )
    sx = torch.full((2,), sequence_length, dtype=torch.int32, device=DEVICE)
    sy = torch.full((5,), sequence_length, dtype=torch.int32, device=DEVICE)

    def precise_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return 1 + euclidean_distance(a.double(), b.double()) * 1e-9

    result = grouped_distances(x, sx, targets, sy, precise_distance, alignment=dtw_batch, max_rows=chunk_rows)
    expected = 1 + (x[:, 0, 0].double()[:, None] - targets[:, 0, 0].double()[None, :]).abs() * 1e-9
    assert result.dtype == torch.float64
    assert_close(result, expected, rtol=0, atol=1e-15)


@pytest.mark.parametrize("chunk_rows", [1, 3, 8192])
def test_chunking_preserves_custom_alignment_precision(chunk_rows: int) -> None:
    features = torch.arange(4, dtype=torch.float32, device=DEVICE)[:, None, None].repeat(1, 2, 1)
    sizes = torch.full((4,), 2, dtype=torch.int32, device=DEVICE)

    def precise_alignment(cost: torch.Tensor, sx: torch.Tensor, sy: torch.Tensor, *, symmetric: bool) -> torch.Tensor:
        return 1 + dtw_batch(cost, sx, sy, symmetric=symmetric).double() * 1e-9

    result = grouped_distances(
        features, sizes, features, sizes, euclidean_distance, alignment=precise_alignment, max_rows=chunk_rows
    )
    expected = (
        1 + (torch.arange(4, device=DEVICE)[:, None] - torch.arange(4, device=DEVICE)[None, :]).abs().double() * 1e-9
    )
    assert result.dtype == torch.float64
    assert_close(result, expected, rtol=0, atol=1e-15)


@pytest.mark.parametrize("chunk_rows", [1, 3, 8192])
def test_near_tie_abx_is_invariant_to_chunk_size(monkeypatch: pytest.MonkeyPatch, chunk_rows: int) -> None:
    monkeypatch.setenv("FASTABX_MAX_SCORE_CHUNK_ROWS", str(chunk_rows))
    dataset = Dataset.from_numpy([[0.0], [0.1], [1.0], [1.1]], {"phone": ["a", "a", "b", "b"]}, dtype=torch.float32)

    def precise_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return 1 + euclidean_distance(a.double(), b.double()) * 1e-9

    assert Score(Task(dataset, on="phone"), precise_distance, progress=False).collapse() == pytest.approx(
        0.0, abs=0, rel=0
    )


@pytest.mark.parametrize("scale", [1e-30, 1.0, 1e30])
def test_angular_score_is_invariant_to_positive_rescaling(scale: float) -> None:
    features = torch.tensor([[3.0, 4.0], [4.0, 3.0], [-3.0, -4.0], [-4.0, -3.0]]) * scale
    dataset = Dataset.from_numpy(features, {"phone": ["a", "a", "b", "b"]})
    assert Score(Task(dataset, on="phone"), "angular", progress=False).collapse() == pytest.approx(0.0, abs=0, rel=0)


def test_custom_float32_distance_retains_its_dtype_with_float64_features() -> None:
    features = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64, device=DEVICE)[:, None, None]
    sizes = torch.ones(3, dtype=torch.int32, device=DEVICE)

    def single_precision(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return euclidean_distance(a.float(), b.float())

    result = grouped_distances(features, sizes, features, sizes, single_precision, alignment=dtw_batch, max_rows=1)
    assert result.dtype == torch.float32
    assert_close(result, torch.tensor([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=torch.float32, device=DEVICE))


@pytest.mark.parametrize("distance", ["angular", "cosine"])
@pytest.mark.parametrize("chunk_rows", [1, 8192])
@pytest.mark.parametrize("lengths", [[1, 1, 1, 1], [1, 3, 2, 4]])
def test_zero_frames_score_as_a_separate_category(
    monkeypatch: pytest.MonkeyPatch, distance: DistanceName, chunk_rows: int, lengths: list[int]
) -> None:
    monkeypatch.setenv("FASTABX_MAX_SCORE_CHUNK_ROWS", str(chunk_rows))
    frames, indices, start = [], {}, 0
    for i, length in enumerate(lengths):
        frames.append(torch.full((length, 2), float(i >= 2), device=DEVICE))
        indices[i] = (start, start + length)
        start += length
    dataset = Dataset(
        pl.DataFrame({"phone": ["a", "a", "b", "b"]}),
        InMemoryAccessor(indices, torch.cat(frames), DEVICE),
    )
    task = Task(dataset, on="phone")
    assert Score(task, distance, progress=False).collapse() == pytest.approx(0.0, abs=0, rel=0)
    for cell in task:
        assert abx_on_cell(cell, distance).item() == pytest.approx(0.0, abs=0, rel=0)
