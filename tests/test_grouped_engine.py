"""Equivalence and invariance tests for the grouped scoring engine (``fastabx.group``)."""

import gc
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.testing import assert_close

from fastabx import Dataset, Task
from fastabx.constraints import constraints_all_different
from fastabx.distance import DistanceName, abx_on_cell, distance_function
from fastabx.group import GroupReducer, group_cells, grouped_contributions
from fastabx.score import score_task
from fastabx.utils import prefetch

POOLED_DISTANCES: list[DistanceName] = ["euclidean", "cosine", "kl_symmetric", "identical"]


def _normalize_for_distance(features: np.ndarray, distance: DistanceName) -> np.ndarray:
    if distance == "kl_symmetric":
        f = np.abs(features) + 0.1
        return (f / f.sum(1, keepdims=True)).astype(np.float32)
    if distance == "identical":
        return (np.arange(features.shape[0]) % 3).reshape(-1, 1).astype(np.float32)
    return features.astype(np.float32)


def _dataset_for_distance(distance: DistanceName, *, with_across: bool = True) -> Dataset:
    rng = np.random.default_rng(123)
    n, d = 36, 5
    features = rng.standard_normal((n, d)).astype(np.float32)
    features = _normalize_for_distance(features, distance)
    phones: list[object] = ["a", "b", "c"] * 12
    speakers: list[object] = (["s1"] * 12) + (["s2"] * 12) + (["s3"] * 12)
    contexts: list[object] = ["c1", "c2"] * 18
    labels: dict[str, list[object]] = {"phone": phones, "speaker": speakers, "context": contexts}
    if not with_across:
        labels.pop("speaker")
    return Dataset.from_numpy(features, labels)


@pytest.mark.parametrize("distance", POOLED_DISTANCES)
@pytest.mark.parametrize("with_across", [False, True])
def test_score_task_matches_abx_on_cell(distance: DistanceName, *, with_across: bool) -> None:
    dataset = _dataset_for_distance(distance, with_across=True)
    if distance in {"cosine", "angular"}:
        dataset.normalize_()
    by = ["context"]
    across = ["speaker"] if with_across else []
    task = Task(dataset, on="phone", by=by, across=across)
    grouped_scores, grouped_sizes = score_task(task, distance_function(distance))
    for i, cell in enumerate(task):
        per_cell = float(abx_on_cell(cell, distance))
        # Plan's central claim: "grouped == per-cell, bit-for-bit".
        assert grouped_scores[i] == per_cell
        assert grouped_sizes[i] == len(cell)


def test_grouped_contributions_unmasked_vs_all_true_mask() -> None:
    rng = torch.Generator().manual_seed(0)
    nx, na, nb = 4, 5, 6
    dxa = torch.randn(nx, na, generator=rng)
    dxb = torch.randn(nx, nb, generator=rng)
    mask = torch.ones(nx, na, nb)
    unmasked = grouped_contributions(dxa, dxb)
    masked = grouped_contributions(dxa, dxb, mask)
    assert_close(unmasked, masked)


def test_grouped_contributions_all_zero_mask_yields_zero() -> None:
    rng = torch.Generator().manual_seed(1)
    nx, na, nb = 3, 4, 5
    dxa = torch.randn(nx, na, generator=rng)
    dxb = torch.randn(nx, nb, generator=rng)
    mask = torch.zeros(nx, na, nb)
    masked = grouped_contributions(dxa, dxb, mask)
    assert torch.all(masked == 0)


def test_grouped_contributions_partial_mask_matches_hand_computed() -> None:
    """With a partial mask the per-B sum must equal a loop over the True triplets."""
    keep_prob = 0.4
    rng = torch.Generator().manual_seed(2)
    nx, na, nb = 4, 3, 5
    dxa = torch.randn(nx, na, generator=rng)
    dxb = torch.randn(nx, nb, generator=rng)
    mask = (torch.rand(nx, na, nb, generator=rng) > keep_prob).to(dxa.dtype)
    got = grouped_contributions(dxa, dxb, mask)
    # Hand-compute: per b column, sum 0.5*(1 - sign(dxa - dxb)) * mask over (nx, na).
    expected = torch.zeros(nb)
    for ib in range(nb):
        for ix in range(nx):
            for ia in range(na):
                if mask[ix, ia, ib] > 0:
                    diff = dxa[ix, ia] - dxb[ix, ib]
                    sign = 0.0 if diff == 0 else (1.0 if diff > 0 else -1.0)
                    expected[ib] += 0.5 * (1 - sign)
    torch.testing.assert_close(got, expected)


def test_max_score_chunk_rows_invariance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forcing tiny score-chunk rows must not change the output of score_task."""
    dataset = _dataset_for_distance("euclidean")
    task = Task(dataset, on="phone", by=["context"], across=["speaker"])
    baseline_scores, baseline_sizes = score_task(task, distance_function("euclidean"))
    monkeypatch.setattr("fastabx.group.MAX_SCORE_CHUNK_ROWS", 4)
    chunked_scores, chunked_sizes = score_task(task, distance_function("euclidean"))
    assert baseline_scores == chunked_scores
    assert baseline_sizes == chunked_sizes


def test_gather_chunk_rows_invariance(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _dataset_for_distance("euclidean")
    task = Task(dataset, on="phone", by=["context"], across=["speaker"])
    baseline_scores, _ = score_task(task, distance_function("euclidean"))
    monkeypatch.setattr("fastabx.group.GATHER_CHUNK_ROWS", 4)
    chunked_scores, _ = score_task(task, distance_function("euclidean"))
    assert baseline_scores == chunked_scores


def test_reduction_flush_cols_invariance(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _dataset_for_distance("euclidean")
    task = Task(dataset, on="phone", by=["context"], across=["speaker"])
    baseline_scores, _ = score_task(task, distance_function("euclidean"))
    monkeypatch.setattr("fastabx.group.REDUCTION_FLUSH_COLS", 2)
    flushed_scores, _ = score_task(task, distance_function("euclidean"))
    for a, b in zip(baseline_scores, flushed_scores, strict=True):
        assert_close(a, b, atol=1e-6, rtol=0)


def test_group_reducer_finalize_flushes_remainder() -> None:
    dataset = _dataset_for_distance("euclidean")
    task = Task(dataset, on="phone", by=["context"], across=["speaker"])
    reducer = GroupReducer(len(task))
    distance = distance_function("euclidean")
    for group in group_cells(task):
        reducer.add(group, distance, is_symmetric=task.is_symmetric)
    # Don't flush manually — finalize() must do it.
    scores, sizes = reducer.finalize()
    expected_scores, expected_sizes = score_task(task, distance)
    assert scores == expected_scores
    assert sizes == expected_sizes


def test_constrained_score_none_for_unsatisfiable_cells() -> None:
    """A constraint that filters out all triplets must produce ``None`` scores & sizes."""
    rng = np.random.default_rng(7)
    features = rng.standard_normal((12, 4)).astype(np.float32)
    labels = {
        "phone": ["a", "b"] * 6,
        "context": ["c1"] * 12,  # only one context — constraint context_a != context_b is impossible
    }
    dataset = Dataset.from_numpy(features, labels)
    task = Task(dataset, on="phone")
    scores, sizes = score_task(task, distance_function("euclidean"), constraints_all_different("context"))
    assert all(s is None for s in scores)
    assert all(sz is None for sz in sizes)


def test_constrained_score_sizes_equal_mask_sums() -> None:
    """Each constrained ``size`` must equal the number of True entries in the per-cell mask."""
    from fastabx.constraints import apply_constraints

    rng = np.random.default_rng(3)
    n = 18
    features = rng.standard_normal((n, 4)).astype(np.float32)
    labels = {
        "phone": ["a", "b", "c"] * 6,
        "speaker": (["s1"] * 6 + ["s2"] * 6 + ["s3"] * 6),
        "context": ["c1", "c2"] * 9,
    }
    dataset = Dataset.from_numpy(features, labels)
    task = Task(dataset, on="phone", across=["speaker"])
    cstrs = list(constraints_all_different("context"))
    _, sizes = score_task(task, distance_function("euclidean"), cstrs)

    # Build the expected sizes from the mask directly.
    masked = apply_constraints(task.cells, dataset.labels, cstrs, is_symmetric=False)
    expected_sizes = [sum(v) if any(v) else None for v in masked["is_valid"].to_list()]
    assert sizes == expected_sizes


def test_group_cells_empty_task_yields_nothing() -> None:
    """A task whose cells DataFrame is empty must produce no groups (hits the empty-loop branch)."""
    rng = np.random.default_rng(0)
    features = rng.standard_normal((3, 4)).astype(np.float32)
    # Each phone occurs only once → none meet MIN_A_LEN → cells DataFrame is empty.
    labels = {"phone": ["a", "b", "c"]}
    dataset = Dataset.from_numpy(features, labels)
    task = Task(dataset, on="phone")
    assert len(task) == 0
    assert list(group_cells(task)) == []


def test_group_reducer_constrained_without_mask_raises() -> None:
    """A constrained reducer receiving a group with no mask must raise NoConstraintsError."""
    from fastabx.constraints import NoConstraintsError
    from fastabx.dataset import Batch
    from fastabx.group import CellGroup

    reducer = GroupReducer(num_cells=1, constrained=True)
    # Build a trivial CellGroup with no mask.
    x = Batch(torch.zeros(2, 1, 3), torch.tensor([1, 1], dtype=torch.int32))
    targets = Batch(torch.zeros(3, 1, 3), torch.tensor([1, 1, 1], dtype=torch.int32))
    group = CellGroup(x=x, targets=targets, rows=[2, 1], positions=[0], mask=None)
    with pytest.raises(NoConstraintsError):
        reducer.add(group, distance_function("euclidean"), is_symmetric=False)


def test_prefetch_yields_same_items_as_generator() -> None:
    def gen() -> Iterable[int]:
        yield from range(10)

    assert list(prefetch(gen())) == list(range(10))


def test_prefetch_reraises_producer_exception() -> None:
    class BadError(RuntimeError):
        pass

    def gen() -> Iterable[int]:
        yield 1
        yield 2
        raise BadError("boom")

    seen: list[int] = []
    with pytest.raises(BadError, match="boom"):
        seen.extend(prefetch(gen()))
    # We should have at least consumed the items produced before the exception.
    assert 1 in seen


def test_prefetch_cleans_up_producer_thread_on_early_exit() -> None:
    """Abandoning the consumer must not leak the producer thread parked on a full queue."""
    closed = threading.Event()

    def gen() -> Iterable[int]:
        try:
            i = 0
            while True:
                yield i
                i += 1
        finally:
            closed.set()  # runs when the producer's iterator is closed

    before = threading.active_count()
    it = prefetch(gen())
    assert next(it) == 0
    it.close()  # triggers prefetch's finally: stop, drain, join

    # join() inside close() guarantees the producer is gone; the source generator is closed once
    # its last reference is dropped. Wait for both rather than rely on exact GC timing.
    deadline = time.time() + 5
    while (threading.active_count() > before or not closed.is_set()) and time.time() < deadline:
        gc.collect()
        time.sleep(0.01)
    assert threading.active_count() == before
    assert closed.is_set()


def test_prefetch_full_consumption_joins_thread() -> None:
    """The normal path also leaves no lingering producer thread."""
    before = threading.active_count()
    assert list(prefetch(iter(range(20)))) == list(range(20))
    deadline = time.time() + 5
    while threading.active_count() > before and time.time() < deadline:
        time.sleep(0.01)
    assert threading.active_count() == before


def test_env_var_chunking_invariance_via_subprocess(tmp_path: Path) -> None:
    """End-to-end check that the FASTABX_* chunk env vars don't change the final score."""
    script = tmp_path / "run.py"
    script.write_text(
        "import numpy as np, json, sys\n"
        "from fastabx import Dataset, Task, Score\n"
        "rng = np.random.default_rng(0)\n"
        "features = rng.standard_normal((24, 4)).astype('float32')\n"
        "labels = {'phone': ['a','b','c']*8, 'speaker': ['s1']*12 + ['s2']*12, 'context': ['c1','c2']*12}\n"
        "ds = Dataset.from_numpy(features, labels)\n"
        "task = Task(ds, on='phone', by=['context'], across=['speaker'])\n"
        "score = Score(task, 'euclidean')\n"
        "print(json.dumps(sorted(s for s in score.cells['score'].to_list() if s is not None)))\n"
    )
    env_base = {**os.environ, "TQDM_DISABLE": "1"}
    baseline = subprocess.check_output([sys.executable, str(script)], env=env_base, text=True)
    env_tiny = {
        **env_base,
        "FASTABX_MAX_SCORE_CHUNK_ROWS": "2",
        "FASTABX_GATHER_CHUNK_ROWS": "2",
        "FASTABX_REDUCTION_FLUSH_COLS": "2",
    }
    chunked = subprocess.check_output([sys.executable, str(script)], env=env_tiny, text=True)
    # The chunk env vars must be a no-op on the output, exactly.
    assert json.loads(baseline) == json.loads(chunked)
