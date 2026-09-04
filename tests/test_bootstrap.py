"""Tests for ``fastabx.bootstrap``: the weighted engine against brute-force references."""

import numpy as np
import polars as pl
import pytest
import torch

from fastabx import Bootstrap, Dataset, Score, Task
from fastabx.bootstrap import pool_key, weighted_counts
from fastabx.constraints import constraints_all_different
from fastabx.score import BootstrapConstraintsError, NoBootstrapError

# How far the bootstrap SD may drift from the true sampling SD before the calibration test fails.
SD_RATIO_BOUNDS = (0.5, 2.0)


def _dataset(n: int = 60, d: int = 8, seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    features = rng.standard_normal((n, d)).astype(np.float32)
    labels = {
        "phone": ["a", "b", "c"] * (n // 3),
        "speaker": (["s1"] * (n // 2)) + (["s2"] * (n // 2)),
        "context": ["c1", "c2"] * (n // 2),
    }
    return Dataset.from_numpy(features, labels)


def _brute_force_cell(
    xa: torch.Tensor,
    b: torch.Tensor,
    wx: np.ndarray,
    wa: np.ndarray,
    wb: np.ndarray,
) -> float:
    """Weighted ABX error rate of one symmetric cell, computed with explicit Python loops."""
    dxa = torch.cdist(xa, xa)
    dxb = torch.cdist(xa, b)
    num = 0.0
    for x in range(len(xa)):
        for a in range(len(xa)):
            if x == a:
                continue
            for bi in range(len(b)):
                delta = (dxa[x, a] - dxb[x, bi]).item()
                s = 1.0 if delta < 0 else (0.5 if delta == 0 else 0.0)
                num += wx[x] * wa[a] * wb[bi] * s
    pairs = sum(wx[x] * wa[a] for x in range(len(xa)) for a in range(len(xa)) if x != a)
    return 1 - num / (pairs * wb.sum())


def test_weighted_counts_matches_brute_force() -> None:
    """The chunked contraction must equal an explicit triple loop, ties included."""
    rng = np.random.default_rng(0)
    nx, na, nb, n_rep = 5, 5, 7, 4
    dxa = torch.from_numpy(rng.integers(0, 4, (nx, na)).astype(np.float32))  # small ints => real ties
    dxb = torch.from_numpy(rng.integers(0, 4, (nx, nb)).astype(np.float32))
    wx = torch.from_numpy(rng.integers(0, 3, (n_rep, nx)).astype(np.float32))
    wa = torch.from_numpy(rng.integers(0, 3, (n_rep, na)).astype(np.float32))

    got = weighted_counts(dxa, dxb, wx, wa)
    expected = np.zeros((n_rep, nb))
    for r in range(n_rep):
        for bi in range(nb):
            for x in range(nx):
                for a in range(na):
                    delta = (dxa[x, a] - dxb[x, bi]).item()
                    s = 1.0 if delta < 0 else (0.5 if delta == 0 else 0.0)
                    expected[r, bi] += wx[r, x].item() * wa[r, a].item() * s
    assert np.allclose(got.numpy(), expected, atol=1e-5)


def test_weighted_counts_chunking_invariance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forcing a tiny B chunk must not change the result."""
    rng = np.random.default_rng(1)
    dxa = torch.from_numpy(rng.standard_normal((4, 4)).astype(np.float32))
    dxb = torch.from_numpy(rng.standard_normal((4, 11)).astype(np.float32))
    wx = torch.from_numpy(rng.integers(0, 3, (6, 4)).astype(np.float32))
    wa = torch.from_numpy(rng.integers(0, 3, (6, 4)).astype(np.float32))
    baseline = weighted_counts(dxa, dxb, wx, wa)
    monkeypatch.setattr("fastabx.bootstrap.BOOTSTRAP_CHUNK_ELEMS", 1)
    assert torch.equal(baseline, weighted_counts(dxa, dxb, wx, wa))


def test_all_ones_weights_reproduce_the_point_estimate() -> None:
    """Weights of 1 everywhere are the identity resample: the replicate must equal the point estimate."""
    dataset = _dataset()
    task = Task(dataset, on="phone", by=["speaker"])
    bootstrap = Bootstrap(3, seed=0)
    # Force every pool's draw to be all-ones by pre-seeding the cache.
    for column in ("index_a", "index_b"):
        for pool in task.cells[column].to_list():
            bootstrap._weights[pool_key(pool)] = np.ones((3, len(pool)), dtype=np.int32)  # noqa: SLF001
    score = Score(task, "euclidean", bootstrap=bootstrap)

    point = score.collapse(levels=["phone", "speaker"])
    replicates = score.bootstrap_collapse(levels=["phone", "speaker"])
    assert np.allclose(replicates, point, atol=1e-6)
    # Sizes too: identity weights give exactly the point-estimate cell sizes.
    assert np.allclose(score._replicates[1][:, 0], score.cells["size"].to_numpy())  # noqa: SLF001


def test_bootstrap_cell_score_matches_brute_force() -> None:
    """End-to-end: a per-cell replicate score must equal an explicit loop over the drawn multiplicities."""
    dataset = _dataset(n=30, d=5, seed=3)
    task = Task(dataset, on="phone", by=["speaker"])
    bootstrap = Bootstrap(4, seed=7)
    score = Score(task, "euclidean", bootstrap=bootstrap)
    scores = score._replicates[0]  # noqa: SLF001

    data = dataset.accessor.data
    for cell in range(len(task)):
        index_a = task.cells[cell, "index_a"].to_list()
        index_b = task.cells[cell, "index_b"].to_list()
        wa = bootstrap.weights(tuple(index_a))
        wb = bootstrap.weights(tuple(index_b))
        xa = data[index_a].squeeze(1) if data.ndim == 3 else data[index_a]
        b = data[index_b].squeeze(1) if data.ndim == 3 else data[index_b]
        for replicate in range(4):
            expected = _brute_force_cell(xa, b, wa[replicate], wa[replicate], wb[replicate])
            assert scores[cell, replicate] == pytest.approx(expected, abs=1e-5)


def test_shared_pools_share_their_draw() -> None:
    """Two cells built on the same item pool must be resampled identically within a replicate."""
    dataset = _dataset()
    task = Task(dataset, on="phone", by=["speaker"])
    bootstrap = Bootstrap(5, seed=0)
    Score(task, "euclidean", bootstrap=bootstrap)
    # Cells (a,b) and (a,c) of a given speaker share their A pool.
    cells = task.cells.filter(pl.col("phone") == "a", pl.col("speaker") == "s1")
    assert len(cells) >= 2
    pools = {tuple(p) for p in cells["index_a"].to_list()}
    assert len(pools) == 1
    # And the A pool of (a, b) is the B pool of (b, a).
    a_pool = tuple(cells[0, "index_a"].to_list())
    mirror = task.cells.filter(pl.col("phone") == "b", pl.col("phone_b") == "a", pl.col("speaker") == "s1")
    assert tuple(mirror[0, "index_b"].to_list()) == a_pool
    assert np.array_equal(bootstrap.weights(a_pool), bootstrap.weights(tuple(mirror[0, "index_b"].to_list())))


def test_same_bootstrap_reused_gives_paired_resamples() -> None:
    """Reusing one Bootstrap across two Scores must resample both identically (paired comparison)."""
    dataset_a, dataset_b = _dataset(seed=0), _dataset(seed=1)
    bootstrap = Bootstrap(6, seed=0)
    task_a = Task(dataset_a, on="phone", by=["speaker"])
    task_b = Task(dataset_b, on="phone", by=["speaker"])
    Score(task_a, "euclidean", bootstrap=bootstrap)
    cached = {k: v.copy() for k, v in bootstrap._weights.items()}  # noqa: SLF001
    Score(task_b, "euclidean", bootstrap=bootstrap)
    for key, value in cached.items():
        assert np.array_equal(bootstrap._weights[key], value)  # noqa: SLF001


def test_bootstrap_is_centred_on_the_point_estimate() -> None:
    """The bootstrap distribution must be centred on the point estimate (no duplicate-item bias)."""
    dataset = _dataset(n=90, d=6, seed=11)
    task = Task(dataset, on="phone", by=["speaker"])
    score = Score(task, "euclidean", bootstrap=Bootstrap(200, seed=0))
    point = score.collapse(levels=["phone", "speaker"])
    replicates = score.bootstrap_collapse(levels=["phone", "speaker"])
    standard_error = np.nanstd(replicates) / np.sqrt(len(replicates))
    assert abs(np.nanmean(replicates) - point) < 5 * standard_error


def test_across_task_bootstrap_runs_and_is_centred() -> None:
    """The asymmetric (across) path: X and A are distinct pools, drawn independently."""
    dataset = _dataset(n=90, d=6, seed=5)
    task = Task(dataset, on="phone", by=["context"], across=["speaker"])
    score = Score(task, "euclidean", bootstrap=Bootstrap(200, seed=0))
    point = score.collapse(levels=["phone", "context"])
    replicates = score.bootstrap_collapse(levels=["phone", "context"])
    standard_error = np.nanstd(replicates) / np.sqrt(len(replicates))
    assert abs(np.nanmean(replicates) - point) < 5 * standard_error


def test_bootstrap_sd_recovers_the_true_sampling_sd() -> None:
    """Calibration: the bootstrap SD on one dataset must match the SD across independent datasets.

    Ground truth comes from redrawing the data many times from the same generative process. This is the
    property that actually matters — centring alone would still pass if the weights were mis-scaled.
    """
    n_phones, n_speakers, per_cell, dim = 5, 3, 10, 12
    means = np.random.default_rng(0).standard_normal((n_phones, dim))
    labels = {
        "phone": [f"p{p}" for p in range(n_phones) for _ in range(n_speakers * per_cell)],
        "speaker": [f"s{s}" for _ in range(n_phones) for s in range(n_speakers) for _ in range(per_cell)],
    }

    def draw(seed: int) -> Dataset:
        rng = np.random.default_rng(seed)
        features = np.stack(
            [means[int(phone[1:])] + 1.6 * rng.standard_normal(dim) for phone in labels["phone"]]
        ).astype(np.float32)
        return Dataset.from_numpy(features, labels)

    levels = ["phone", "speaker"]
    monte_carlo = np.array(
        [
            Score(Task(draw(1000 + seed), on="phone", by=levels[1:]), "euclidean").collapse(levels=levels)
            for seed in range(40)
        ]
    )
    score = Score(Task(draw(1000), on="phone", by=["speaker"]), "euclidean", bootstrap=Bootstrap(300, seed=0))
    ratio = np.nanstd(score.bootstrap_collapse(levels=levels), ddof=1) / monte_carlo.std(ddof=1)
    assert SD_RATIO_BOUNDS[0] < ratio < SD_RATIO_BOUNDS[1], f"bootstrap SD is {ratio:.2f}x the true sampling SD"


def test_confidence_interval_brackets_the_point_estimate() -> None:
    dataset = _dataset(n=90, d=6, seed=2)
    task = Task(dataset, on="phone", by=["speaker"])
    score = Score(task, "euclidean", bootstrap=Bootstrap(200, seed=0))
    point = score.collapse(levels=["phone", "speaker"])
    low, high = score.confidence_interval(levels=["phone", "speaker"])
    assert low < point < high


def test_weighted_collapse_of_replicates() -> None:
    dataset = _dataset(n=60, d=6, seed=4)
    task = Task(dataset, on="phone", by=["speaker"])
    score = Score(task, "euclidean", bootstrap=Bootstrap(50, seed=0))
    replicates = score.bootstrap_collapse(weighted=True)
    assert replicates.shape == (50,)
    assert np.isfinite(replicates).all()


def test_bootstrap_does_not_change_the_point_estimate() -> None:
    """Adding a Bootstrap must leave the point estimate bit-identical."""
    dataset = _dataset(seed=8)
    task = Task(dataset, on="phone", by=["speaker"])
    plain = Score(task, "euclidean").cells
    with_bootstrap = Score(task, "euclidean", bootstrap=Bootstrap(4, seed=0)).cells
    assert plain.equals(with_bootstrap)


def test_seed_reproducibility_and_sensitivity() -> None:
    dataset = _dataset(seed=9)
    task = Task(dataset, on="phone", by=["speaker"])
    levels = ["phone", "speaker"]
    first = Score(task, "euclidean", bootstrap=Bootstrap(20, seed=0)).bootstrap_collapse(levels=levels)
    same = Score(task, "euclidean", bootstrap=Bootstrap(20, seed=0)).bootstrap_collapse(levels=levels)
    other = Score(task, "euclidean", bootstrap=Bootstrap(20, seed=1)).bootstrap_collapse(levels=levels)
    assert np.array_equal(first, same)
    assert not np.array_equal(first, other)


def test_degenerate_replicates_are_null_not_biased() -> None:
    """A cell with na=2 whose two draws collide has no valid pair: it must be null, not scored."""
    rng = np.random.default_rng(0)
    features = rng.standard_normal((8, 4)).astype(np.float32)
    labels = {"phone": ["a", "a", "b", "b", "c", "c", "a", "b"], "speaker": ["s1"] * 8}
    dataset = Dataset.from_numpy(features, labels)
    task = Task(dataset, on="phone", by=["speaker"])
    score = Score(task, "euclidean", bootstrap=Bootstrap(100, seed=0))
    scores, sizes = score._replicates  # noqa: SLF001
    assert np.isnan(scores).any(), "expected some degenerate replicates on tiny cells"
    # Wherever the score is null the size must be null too, and vice versa.
    assert np.array_equal(np.isnan(scores), np.isnan(sizes))
    # Non-degenerate replicates stay in [0, 1].
    finite = scores[~np.isnan(scores)]
    assert ((finite >= 0) & (finite <= 1)).all()


def test_no_bootstrap_raises() -> None:
    dataset = _dataset()
    task = Task(dataset, on="phone", by=["speaker"])
    score = Score(task, "euclidean")
    assert score.n_replicates == 0
    with pytest.raises(NoBootstrapError):
        score.bootstrap_collapse(levels=["phone", "speaker"])


def test_bootstrap_with_constraints_raises() -> None:
    dataset = _dataset()
    task = Task(dataset, on="phone", by=["speaker"])
    with pytest.raises(BootstrapConstraintsError):
        Score(task, "euclidean", constraints=constraints_all_different("context"), bootstrap=Bootstrap(2))


@pytest.mark.parametrize("n_replicates", [0, -1, 2.5, True])
def test_invalid_n_replicates(n_replicates: object) -> None:
    with pytest.raises(TypeError):
        Bootstrap(n_replicates)  # ty: ignore[invalid-argument-type]


def test_invalid_seed() -> None:
    with pytest.raises(TypeError):
        Bootstrap(10, seed="0")  # ty: ignore[invalid-argument-type]


def test_repr() -> None:
    assert repr(Bootstrap(10, seed=3)) == "Bootstrap(n_replicates=10, seed=3)"
    dataset = _dataset()
    task = Task(dataset, on="phone", by=["speaker"])
    assert "5 bootstrap replicates" in repr(Score(task, "euclidean", bootstrap=Bootstrap(5)))
    assert "bootstrap" not in repr(Score(task, "euclidean"))


def test_dtw_path_bootstrap(seq_dataset: Dataset) -> None:
    """Variable-length items (DTW) must go through the same weighted reduction."""
    task = Task(seq_dataset, on="phone", by=["speaker"])
    score = Score(task, "euclidean", bootstrap=Bootstrap(30, seed=0))
    replicates = score.bootstrap_collapse(levels=["phone", "speaker"])
    assert replicates.shape == (30,)
    assert np.isfinite(replicates).any()
