"""Per-cell ABX (``abx_on_cell``) and ``Cell`` correctness tests."""

import numpy as np
import pytest
import torch
from torch.testing import assert_close

from fastabx import Dataset
from fastabx.cell import MIN_A_LEN, Cell
from fastabx.dataset import Batch
from fastabx.distance import DistanceName, abx_on_cell
from fastabx.verify import CellErrorType, InvalidCellError
from tests.reference import reference_abx_pooled

POOLED_DISTANCES: list[DistanceName] = ["euclidean", "cosine", "angular", "kl_symmetric", "identical"]


def _prepare_features(
    distance: DistanceName,
    a: np.ndarray,
    b: np.ndarray,
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Make each array meet the assumptions of the given distance."""
    if distance == "kl_symmetric":
        a = np.abs(a) + 0.1
        b = np.abs(b) + 0.1
        x = np.abs(x) + 0.1
        return (
            (a / a.sum(1, keepdims=True)).astype(np.float32),
            (b / b.sum(1, keepdims=True)).astype(np.float32),
            (x / x.sum(1, keepdims=True)).astype(np.float32),
        )
    if distance in {"cosine", "angular"}:
        return (
            (a / np.linalg.norm(a, axis=1, keepdims=True)).astype(np.float32),
            (b / np.linalg.norm(b, axis=1, keepdims=True)).astype(np.float32),
            (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32),
        )
    if distance == "identical":
        return (
            (np.arange(a.shape[0]) % 3).reshape(-1, 1).astype(np.float32),
            (np.arange(b.shape[0]) % 3).reshape(-1, 1).astype(np.float32),
            (np.arange(x.shape[0]) % 3).reshape(-1, 1).astype(np.float32),
        )
    return a.astype(np.float32), b.astype(np.float32), x.astype(np.float32)


def _make_pooled_batch(arr: np.ndarray) -> Batch:
    """Build a Batch from a (n, d) array (time dim = 1)."""
    tensor = torch.from_numpy(np.asarray(arr, dtype=np.float32))[:, None, :]
    sizes = torch.ones(tensor.size(0), dtype=torch.int32)
    return Batch(tensor, sizes)


def _make_seq_batch(arrs: list[np.ndarray]) -> Batch:
    """Build a Batch from a list of (s_i, d) arrays — padded to common length."""
    lengths = [a.shape[0] for a in arrs]
    smax, d = max(lengths), arrs[0].shape[1]
    padded = np.zeros((len(arrs), smax, d), dtype=np.float32)
    for i, a in enumerate(arrs):
        padded[i, : a.shape[0]] = a
    sizes = torch.tensor(lengths, dtype=torch.int32)
    return Batch(torch.from_numpy(padded), sizes)


@pytest.fixture
def pooled_cell_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    a = rng.standard_normal((4, 5)).astype(np.float32)
    b = rng.standard_normal((3, 5)).astype(np.float32)
    x = rng.standard_normal((6, 5)).astype(np.float32)
    return a, b, x


@pytest.mark.parametrize("distance", POOLED_DISTANCES)
def test_abx_on_cell_pooled_asymmetric_matches_reference(
    distance: DistanceName,
    pooled_cell_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    a, b, x = _prepare_features(distance, *pooled_cell_data)
    cell = Cell(
        a=_make_pooled_batch(a),
        b=_make_pooled_batch(b),
        x=_make_pooled_batch(x),
        header="h",
        description="d",
        is_symmetric=False,
    )
    got = float(abx_on_cell(cell, distance))
    want = reference_abx_pooled(list(a), list(b), list(x), distance, is_symmetric=False)
    assert_close(got, want, atol=1e-6, rtol=0)


@pytest.mark.parametrize("distance", POOLED_DISTANCES)
def test_abx_on_cell_pooled_symmetric_matches_reference(distance: DistanceName) -> None:
    rng = np.random.default_rng(7)
    a = rng.standard_normal((5, 4)).astype(np.float32)
    b = rng.standard_normal((4, 4)).astype(np.float32)
    x = a
    a, b, x = _prepare_features(distance, a, b, x)
    x = a  # keep symmetric after prep
    cell = Cell(
        a=_make_pooled_batch(a),
        b=_make_pooled_batch(b),
        x=_make_pooled_batch(x),
        header="h",
        description="d",
        is_symmetric=True,
    )
    got = float(abx_on_cell(cell, distance))
    want = reference_abx_pooled(list(a), list(b), list(x), distance, is_symmetric=True)
    assert_close(got, want, atol=1e-6, rtol=0)


def test_abx_on_cell_grouped_matches_per_cell_for_sequential(seq_dataset: Dataset) -> None:
    """For DTW cells: score_task (grouped) must agree bit-for-bit with per-cell abx_on_cell."""
    from fastabx import Task
    from fastabx.distance import distance_function
    from fastabx.score import score_task

    task = Task(seq_dataset, on="phone", by=["context"], across=["speaker"])
    assert any(cell.use_dtw for cell in task)
    grouped_scores, grouped_sizes = score_task(task, distance_function("euclidean"))
    for i, cell in enumerate(task):
        per_cell = float(abx_on_cell(cell, "euclidean"))
        assert grouped_scores[i] == per_cell  # bit-identical
        assert grouped_sizes[i] == len(cell)


def test_abx_perfectly_separable_pooled_returns_zero() -> None:
    """X equals A exactly; B is far away — every triplet has dxa < dxb, score == 0."""
    a = np.eye(4, dtype=np.float32)
    b = np.full((3, 4), 10.0, dtype=np.float32)
    x = a.copy()
    cell = Cell(
        a=_make_pooled_batch(a),
        b=_make_pooled_batch(b),
        x=_make_pooled_batch(x),
        header="h",
        description="d",
        is_symmetric=False,
    )
    got = float(abx_on_cell(cell, "euclidean"))
    assert_close(got, 0.0, atol=1e-6, rtol=0)


def test_abx_fully_inverted_pooled_returns_one() -> None:
    """X equals B exactly; A is far away — every triplet has dxa > dxb, score == 1."""
    a = np.full((4, 3), 10.0, dtype=np.float32)
    b = np.eye(3, dtype=np.float32)
    x = b.copy()
    cell = Cell(
        a=_make_pooled_batch(a),
        b=_make_pooled_batch(b),
        x=_make_pooled_batch(x),
        header="h",
        description="d",
        is_symmetric=False,
    )
    got = float(abx_on_cell(cell, "euclidean"))
    assert_close(got, 1.0, atol=1e-6, rtol=0)


def test_cell_num_triplets_asymmetric_and_symmetric() -> None:
    a = _make_pooled_batch(np.zeros((4, 3), dtype=np.float32))
    b = _make_pooled_batch(np.zeros((2, 3), dtype=np.float32))
    x = _make_pooled_batch(np.zeros((5, 3), dtype=np.float32))
    cell_asym = Cell(a=a, b=b, x=x, header="h", description="d", is_symmetric=False)
    assert cell_asym.num_triplets == 4 * 2 * 5
    assert len(cell_asym) == 40
    cell_sym = Cell(a=a, b=b, x=a, header="h", description="d", is_symmetric=True)
    assert cell_sym.num_triplets == 4 * 2 * (4 - 1)


def test_cell_use_dtw_property() -> None:
    pooled_a = _make_pooled_batch(np.zeros((2, 3), dtype=np.float32))
    seq_a = _make_seq_batch([np.zeros((2, 3), dtype=np.float32), np.zeros((3, 3), dtype=np.float32)])
    pooled_cell = Cell(a=pooled_a, b=pooled_a, x=pooled_a, header="h", description="d", is_symmetric=False)
    seq_cell = Cell(a=seq_a, b=seq_a, x=seq_a, header="h", description="d", is_symmetric=False)
    assert pooled_cell.use_dtw is False
    assert seq_cell.use_dtw is True


def test_cell_min_a_len_constant() -> None:
    assert MIN_A_LEN == 2


def test_cell_post_init_ndim_error() -> None:
    bad = Batch(torch.zeros(2, 3), torch.tensor([1, 1], dtype=torch.int32))  # 2D, not 3D
    good = _make_pooled_batch(np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(InvalidCellError) as exc:
        Cell(a=bad, b=good, x=good, header="h", description="d", is_symmetric=False)
    assert "3 dimensions" in str(exc.value)
    _ = CellErrorType.NDIM  # touch enum


def test_cell_post_init_feature_dim_error() -> None:
    a = _make_pooled_batch(np.zeros((2, 3), dtype=np.float32))
    b = _make_pooled_batch(np.zeros((2, 4), dtype=np.float32))  # different feature dim
    with pytest.raises(InvalidCellError, match="feature dimension"):
        Cell(a=a, b=b, x=a, header="h", description="d", is_symmetric=False)


def test_cell_post_init_size_error() -> None:
    data = torch.zeros(3, 1, 4)
    bad = Batch(data, torch.tensor([1, 1], dtype=torch.int32))  # data has 3 rows, sizes has 2
    good = _make_pooled_batch(np.zeros((2, 4), dtype=np.float32))
    with pytest.raises(InvalidCellError, match="size"):
        Cell(a=bad, b=good, x=good, header="h", description="d", is_symmetric=False)


def test_cell_repr() -> None:
    a = _make_pooled_batch(np.zeros((2, 3), dtype=np.float32))
    cell = Cell(a=a, b=a, x=a, header="h", description="part1), part2", is_symmetric=False)
    rep = repr(cell)
    assert "Cell" in rep
    assert "part1" in rep
    assert "part2" in rep


def test_cell_min_a_len_minimal_cell() -> None:
    a = _make_pooled_batch(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    b = _make_pooled_batch(np.array([[10.0, 10.0]], dtype=np.float32))
    x = _make_pooled_batch(np.array([[1.0, 0.0]], dtype=np.float32))
    cell = Cell(a=a, b=b, x=x, header="h", description="d", is_symmetric=False)
    assert len(cell) == MIN_A_LEN * 1 * 1
    got = float(abx_on_cell(cell, "euclidean"))
    assert 0.0 <= got <= 1.0
