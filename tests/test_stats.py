"""Tests for ``fastabx.stats``: the vectorized bootstrap collapse against a duplicated-categories reference."""

from collections.abc import Sequence

import numpy as np
import polars as pl
import pytest

from fastabx import Bootstrap, Dataset, Score, Task
from fastabx.score import (
    CollapseError,
    NoBootstrapError,
    SymmetricCollapseError,
    pl_weighted_mean,
    score_details,
    symmetrize_cells,
)
from fastabx.stats import (
    InvalidResampleError,
    NestedBootstrapWarning,
    bootstrap_collapse,
    bootstrap_details,
    factor_weights,
    resolve_factors,
)

LEVELS_BY = ["context", "speaker"]


def _dataset(n: int = 144, seed: int = 0, n_speakers: int = 4) -> Dataset:
    rng = np.random.default_rng(seed)
    features = rng.standard_normal((n, 6)).astype(np.float32)
    labels = {
        "phone": [f"p{i % 3}" for i in range(n)],
        "speaker": [f"s{(i // 3) % n_speakers}" for i in range(n)],
        "context": [f"c{(i // (3 * n_speakers)) % 2}" for i in range(n)],
    }
    return Dataset.from_numpy(features, labels)


def _by_score(bootstrap: Bootstrap | None = None) -> Score:
    task = Task(_dataset(), on="phone", by=["speaker", "context"])
    return Score(task, "euclidean", progress=False, bootstrap=bootstrap)


def _across_score() -> Score:
    return Score(Task(_dataset(), on="phone", by=["context"], across=["speaker"]), "euclidean", progress=False)


def _duplicate(cells: pl.DataFrame, factor: str, values: pl.Series, weights: np.ndarray) -> pl.DataFrame:
    """Turn a value drawn ``k`` times into ``k`` distinct copies, in every column of the condition."""
    copies = pl.DataFrame(
        {
            "__value": pl.Series([v for v, k in zip(values, weights, strict=True) for _ in range(k)], dtype=pl.String),
            "__copy": [c for k in weights for c in range(k)],
        }
    )
    for column in (factor, f"{factor}_b", f"{factor}_x"):
        if column in cells.columns:
            cells = (
                cells.join(copies.rename({"__value": column}), on=column, how="inner")
                .with_columns(pl.col(column) + "#" + pl.col("__copy").cast(pl.String))
                .drop("__copy")
            )
    return cells


def _replicate_cells(
    score: Score, *, resample: Sequence[str], n_replicates: int, seed: int, tokens: bool
) -> list[pl.DataFrame]:
    """Materialize every replicate: the token replicate scores if asked, then the drawn categories duplicated."""
    base = score.cells.select(~pl.selectors.starts_with("index_"))
    factors, _ = resolve_factors(base, resample, n_replicates=n_replicates, seed=seed)
    out = []
    for replicate in range(n_replicates):
        cells = base
        if tokens:
            cells = cells.with_columns(
                score=pl.Series(score.replicates[0][:, replicate], nan_to_null=True),
                size=pl.Series(score.replicates[1][:, replicate], nan_to_null=True),
            )
        for name, factor in factors.items():
            cells = _duplicate(cells, name, factor.values, factor.weights[replicate])
        out.append(cells)
    return out


def _reference(
    score: Score,
    *,
    levels: Sequence[tuple[str, ...] | str] | None,
    weighted: bool,
    resample: Sequence[str],
    n_replicates: int,
    seed: int,
    tokens: bool = False,
) -> np.ndarray:
    """Collapse each materialized replicate with the plain collapse."""
    out = []
    for cells in _replicate_cells(score, resample=resample, n_replicates=n_replicates, seed=seed, tokens=tokens):
        if weighted:
            out.append(cells.select(pl_weighted_mean("score", "size")).item())
        else:
            out.append(score_details(cells, levels=levels)["score"].mean())
    return np.array(out, dtype=np.float64)


def _reference_details(
    score: Score,
    *,
    levels: Sequence[tuple[str, ...] | str] | None,
    symmetric: bool,
    resample: Sequence[str],
    n_replicates: int,
    seed: int,
    tokens: bool = False,
) -> pl.DataFrame:
    """Details of each materialized replicate, with the plain (optionally symmetric) details, in long format."""
    frames = []
    replicates = _replicate_cells(score, resample=resample, n_replicates=n_replicates, seed=seed, tokens=tokens)
    for replicate, cells in enumerate(replicates):
        merged = symmetrize_cells(cells, "phone") if symmetric else cells
        details = score_details(merged, levels=levels).drop("size")
        frames.append(details.with_columns(replicate=pl.lit(replicate, dtype=pl.Int64)))
    return pl.concat(frames)


def _assert_same_details(got: pl.DataFrame, expected: pl.DataFrame) -> None:
    keys = [c for c in expected.columns if c != "score"]
    assert got.columns == [*keys, "score"]
    assert len(got) == len(expected)
    joined = got.join(expected, on=keys, how="full", suffix="_expected", nulls_equal=True)
    assert joined.filter(pl.col("score").is_null() != pl.col("score_expected").is_null()).is_empty()
    both = joined.drop_nulls(["score", "score_expected"])
    assert np.allclose(both["score"].to_numpy(), both["score_expected"].to_numpy(), atol=1e-12)


@pytest.mark.parametrize(
    ("resample", "levels"),
    [
        (["speaker"], LEVELS_BY),
        (["context"], LEVELS_BY),
        (["phone"], LEVELS_BY),
        (["speaker", "context"], LEVELS_BY),
        (["speaker", "phone"], [("context", "speaker")]),
        (["speaker"], ["speaker", "context"]),
    ],
)
def test_category_bootstrap_matches_duplication(resample: list[str], levels: list[str]) -> None:
    score = _by_score()
    got = bootstrap_collapse(score, levels=levels, resample=resample, n_replicates=8, seed=3)
    expected = _reference(score, levels=levels, weighted=False, resample=resample, n_replicates=8, seed=3)
    assert np.allclose(got, expected, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("resample", [["speaker"], ["phone", "context"]])
def test_across_bootstrap_weighs_both_sides(resample: list[str]) -> None:
    """In an across task, ``speaker`` and ``speaker_x`` share one draw, and the pair weighs w(s) * w(s_x)."""
    score = _across_score()
    levels = ["context", "speaker"]
    got = bootstrap_collapse(score, levels=levels, resample=resample, n_replicates=8, seed=1)
    expected = _reference(score, levels=levels, weighted=False, resample=resample, n_replicates=8, seed=1)
    assert np.allclose(got, expected, atol=1e-12, equal_nan=True)


def test_weighted_category_bootstrap_matches_duplication() -> None:
    score = _across_score()
    got = bootstrap_collapse(score, weighted=True, resample=["speaker"], n_replicates=8, seed=2)
    expected = _reference(score, levels=None, weighted=True, resample=["speaker"], n_replicates=8, seed=2)
    assert np.allclose(got, expected, atol=1e-12, equal_nan=True)


def test_default_levels_with_only_the_contrast() -> None:
    score = Score(Task(_dataset(), on="phone"), "euclidean", progress=False)
    got = bootstrap_collapse(score, resample=["phone"], n_replicates=6, seed=0)
    expected = _reference(score, levels=None, weighted=False, resample=["phone"], n_replicates=6, seed=0)
    assert np.allclose(got, expected, atol=1e-12, equal_nan=True)


def test_identity_draws_reproduce_the_point_estimate(monkeypatch: pytest.MonkeyPatch) -> None:
    score = _by_score()
    monkeypatch.setattr(
        "fastabx.stats.factor_weights", lambda values, *_, n_replicates, **__: np.ones((n_replicates, len(values)))
    )
    got = bootstrap_collapse(score, levels=LEVELS_BY, resample=["speaker"], n_replicates=3)
    assert np.allclose(got, score.collapse(levels=LEVELS_BY), atol=1e-12, equal_nan=True)
    got = bootstrap_collapse(score, weighted=True, resample=["speaker"], n_replicates=3)
    assert np.allclose(got, score.collapse(weighted=True), atol=1e-12, equal_nan=True)


def test_token_replicates_match_a_per_replicate_collapse() -> None:
    """The vectorized collapse of token replicates equals collapsing each replicate on its own."""
    score = _by_score(bootstrap=Bootstrap(12, seed=0))
    got = bootstrap_collapse(score, levels=LEVELS_BY)
    expected = _reference(score, levels=LEVELS_BY, weighted=False, resample=[], n_replicates=12, seed=0, tokens=True)
    assert np.allclose(got, expected, atol=1e-12, equal_nan=True)
    got = bootstrap_collapse(score, weighted=True)
    expected = _reference(score, levels=None, weighted=True, resample=[], n_replicates=12, seed=0, tokens=True)
    assert np.allclose(got, expected, atol=1e-12, equal_nan=True)


def test_tokens_and_categories_together_warn_and_combine() -> None:
    score = _by_score(bootstrap=Bootstrap(6, seed=0))
    with pytest.warns(NestedBootstrapWarning):
        got = bootstrap_collapse(score, levels=LEVELS_BY, resample=["speaker"], seed=4)
    expected = _reference(
        score, levels=LEVELS_BY, weighted=False, resample=["speaker"], n_replicates=6, seed=4, tokens=True
    )
    assert np.allclose(got, expected, atol=1e-12, equal_nan=True)


def test_chunking_invariance(monkeypatch: pytest.MonkeyPatch) -> None:
    score, plain = _by_score(bootstrap=Bootstrap(7, seed=0)), _by_score()
    baseline = bootstrap_collapse(score, levels=LEVELS_BY)
    categories = bootstrap_collapse(plain, levels=LEVELS_BY, resample=["speaker"], n_replicates=7)
    details = bootstrap_details(plain, levels=LEVELS_BY, symmetric=True, resample=["speaker"], n_replicates=7)
    monkeypatch.setenv("FASTABX_BOOTSTRAP_CHUNK_ELEMS", "1")
    assert np.array_equal(baseline, bootstrap_collapse(score, levels=LEVELS_BY))
    assert np.array_equal(
        categories, bootstrap_collapse(plain, levels=LEVELS_BY, resample=["speaker"], n_replicates=7)
    )
    chunked = bootstrap_details(plain, levels=LEVELS_BY, symmetric=True, resample=["speaker"], n_replicates=7)
    assert details.equals(chunked)


def test_draws_are_paired_and_reproducible() -> None:
    """The draws only depend on the seed and the condition name, not on the Score or the order of ``resample``."""
    values = pl.Series(["s0", "s1", "s2"])
    assert np.array_equal(
        factor_weights(values, "speaker", n_replicates=5, seed=0),
        factor_weights(values, "speaker", n_replicates=5, seed=0),
    )
    assert not np.array_equal(
        factor_weights(values, "speaker", n_replicates=50, seed=0),
        factor_weights(values, "speaker", n_replicates=50, seed=1),
    )
    score = _by_score()
    first = bootstrap_collapse(score, levels=LEVELS_BY, resample=["speaker", "context"], n_replicates=5)
    second = bootstrap_collapse(score, levels=LEVELS_BY, resample=["context", "speaker", "speaker"], n_replicates=5)
    assert np.array_equal(first, second)


def test_speaker_bootstrap_variance_matches_the_variance_of_the_mean() -> None:
    """Resampling speakers estimates (n - 1) / n * s^2 / n, s^2 being the variance between per-speaker scores."""
    n_speakers = 12
    dataset = _dataset(n=3 * n_speakers * 2 * 4, n_speakers=n_speakers, seed=5)
    score = Score(Task(dataset, on="phone", by=["speaker", "context"]), "euclidean", progress=False)
    levels = ["context", ("phone", "phone_b"), "speaker"]
    per_speaker = score.details(levels=levels[:2])["score"].to_numpy()
    expected = (n_speakers - 1) / n_speakers * per_speaker.var(ddof=1) / n_speakers
    replicates = bootstrap_collapse(score, levels=levels, resample=["speaker"], n_replicates=4000, seed=0)
    assert replicates.var() == pytest.approx(expected, rel=0.1)
    assert replicates.mean() == pytest.approx(score.collapse(levels=levels), abs=3 * np.sqrt(expected / 4000) + 1e-12)


def test_invalid_requests() -> None:
    score = _by_score()
    with pytest.raises(InvalidResampleError, match="not a condition"):
        bootstrap_collapse(score, levels=LEVELS_BY, resample=["unknown"], n_replicates=2)
    with pytest.raises(InvalidResampleError, match="n_replicates"):
        bootstrap_collapse(score, levels=LEVELS_BY, resample=["speaker"])
    with pytest.raises(NoBootstrapError):
        bootstrap_details(score, levels=LEVELS_BY)
    with pytest.raises(CollapseError):
        bootstrap_collapse(score, levels=LEVELS_BY, weighted=True, resample=["speaker"], n_replicates=2)
    with pytest.raises(TypeError):
        bootstrap_collapse(score, levels=LEVELS_BY, resample=["speaker"], n_replicates=0)
    with pytest.raises(TypeError, match="seed should be an integer >= 0"):
        bootstrap_collapse(score, levels=LEVELS_BY, resample=["speaker"], n_replicates=2, seed=-1)
    with_tokens = _by_score(bootstrap=Bootstrap(3))
    with pytest.raises(InvalidResampleError, match="set by the 3 token replicates"):
        bootstrap_collapse(with_tokens, levels=LEVELS_BY, n_replicates=3)
    with pytest.raises(InvalidResampleError, match="left in the details"):
        bootstrap_details(score, levels=["context"], resample=["speaker"], n_replicates=2)
    with pytest.raises(InvalidResampleError, match="must be averaged over"):
        bootstrap_details(_across_score(), levels=[], resample=["speaker"], n_replicates=2)


def test_category_bootstrap_with_constraints() -> None:
    """Constrained cells with no valid triplet are null, and skipped by the weighted collapse like by the plain one."""
    base = _dataset()
    labels = base.labels.with_columns(
        take=pl.when(pl.col("speaker") == "s0")
        .then(pl.lit("r0"))
        .otherwise("r" + (pl.int_range(pl.len()) // 24 % 2).cast(str))
    )
    dataset = Dataset.from_numpy(np.random.default_rng(0).standard_normal((len(labels), 6)), labels)
    task = Task(dataset, on="phone", by=["speaker", "context"])
    score = Score(task, "euclidean", constraints=[pl.col("take_a") != pl.col("take_x")], progress=False)
    assert 0 < score.cells["score"].null_count() < len(score.cells)
    got = bootstrap_collapse(score, levels=LEVELS_BY, resample=["speaker"], n_replicates=6, seed=0)
    expected = _reference(score, levels=LEVELS_BY, weighted=False, resample=["speaker"], n_replicates=6, seed=0)
    assert np.isfinite(got).any()
    assert np.allclose(got, expected, atol=1e-12, equal_nan=True)


def _one_direction_dataset() -> Dataset:
    """Phone p2 has a single item for speaker s0 in context c0: (p0, p2) exists there, but not (p2, p0)."""
    base = _dataset()
    lonely = (pl.col("phone") == "p2") & (pl.col("speaker") == "s0") & (pl.col("context") == "c0")
    labels = base.labels.with_row_index("row").filter(~lonely | (pl.col("row").cum_count().over(lonely) == 1))
    features = np.random.default_rng(0).standard_normal((len(labels), 6)).astype(np.float32)
    return Dataset.from_numpy(features, labels.drop("row"))


def test_symmetric_details_merge_both_directions() -> None:
    score = Score(Task(_one_direction_dataset(), on="phone", by=["speaker", "context"]), "euclidean", progress=False)
    cells = score.cells
    in_s0_c0 = (pl.col("speaker") == "s0") & (pl.col("context") == "c0")
    assert cells.filter(in_s0_c0, phone="p0", phone_b="p2").height == 1
    assert cells.filter(in_s0_c0, phone="p2", phone_b="p0").is_empty()

    merged = score.details(levels=[], symmetric=True)
    assert merged.filter(pl.col("phone") >= pl.col("phone_b")).is_empty()
    assert merged.filter(in_s0_c0, phone="p0", phone_b="p2").is_empty(), "one-direction pairs are dropped"
    assert len(merged) == (len(cells) - 1) // 2
    row = merged.filter(speaker="s1", context="c1", phone="p0", phone_b="p1")
    both = cells.filter(speaker="s1", context="c1").filter(
        ((pl.col("phone") == "p0") & (pl.col("phone_b") == "p1"))
        | ((pl.col("phone") == "p1") & (pl.col("phone_b") == "p0"))
    )
    assert row["score"].item() == pytest.approx(both["score"].mean())
    assert row["size"].item() == both["size"].sum()
    assert score.details(levels=LEVELS_BY).equals(score_details(score.cells, levels=LEVELS_BY))
    assert score.details(levels=LEVELS_BY, symmetric=True)["phone"].to_list() == ["p0", "p0", "p1"]


def test_symmetric_requires_the_on_condition() -> None:
    task = Task(_dataset(), on="phone", by=["speaker", "context"])
    precomputed = Task.from_cells(task.dataset, task.cells, is_symmetric=True)
    score = Score(precomputed, "euclidean", progress=False)
    with pytest.raises(SymmetricCollapseError):
        score.details(symmetric=True)
    with pytest.raises(SymmetricCollapseError):
        bootstrap_details(score, symmetric=True, resample=["phone"], n_replicates=2)


@pytest.mark.parametrize(
    ("resample", "levels", "symmetric"),
    [
        (["speaker"], LEVELS_BY, False),
        (["speaker"], LEVELS_BY, True),
        (["speaker", "context"], LEVELS_BY, True),
        (["phone", "speaker"], [("phone", "phone_b"), "speaker"], True),
    ],
)
def test_bootstrap_details_matches_duplication(
    resample: list[str],
    levels: list[str],
    symmetric: bool,  # ruff: ignore[boolean-type-hint-positional-argument]
) -> None:
    score = Score(Task(_one_direction_dataset(), on="phone", by=["speaker", "context"]), "euclidean", progress=False)
    got = bootstrap_details(score, levels=levels, symmetric=symmetric, resample=resample, n_replicates=6, seed=2)
    expected = _reference_details(score, levels=levels, symmetric=symmetric, resample=resample, n_replicates=6, seed=2)
    _assert_same_details(got, expected)


def test_bootstrap_details_across_and_tokens() -> None:
    across = _across_score()
    levels = ["context", "speaker"]
    got = bootstrap_details(across, levels=levels, symmetric=True, resample=["speaker"], n_replicates=5, seed=1)
    expected = _reference_details(across, levels=levels, symmetric=True, resample=["speaker"], n_replicates=5, seed=1)
    _assert_same_details(got, expected)

    tokens = _by_score(bootstrap=Bootstrap(5, seed=0))
    for symmetric in (False, True):
        got = bootstrap_details(tokens, levels=LEVELS_BY, symmetric=symmetric)
        expected = _reference_details(
            tokens, levels=LEVELS_BY, symmetric=symmetric, resample=[], n_replicates=5, seed=0, tokens=True
        )
        _assert_same_details(got, expected)
    got = bootstrap_details(Score(Task(_dataset(), on="phone"), "euclidean", bootstrap=Bootstrap(3), progress=False))
    assert got.columns == ["phone", "phone_b", "replicate", "score"]
    assert len(got) == 6 * 3


def test_bootstrap_details_identity_draws(monkeypatch: pytest.MonkeyPatch) -> None:
    score = _by_score()
    monkeypatch.setattr(
        "fastabx.stats.factor_weights", lambda values, *_, n_replicates, **__: np.ones((n_replicates, len(values)))
    )
    got = bootstrap_details(score, levels=LEVELS_BY, symmetric=True, resample=["speaker"], n_replicates=2)
    point = score.details(levels=LEVELS_BY, symmetric=True).drop("size")
    for replicate in range(2):
        rows = got.filter(replicate=replicate).drop("replicate")
        assert np.allclose(rows["score"].to_numpy(), point["score"].to_numpy(), atol=1e-12)
        assert rows.drop("score").equals(point.drop("score"))


def test_paired_comparison_recipe() -> None:
    """Two Scores collapsed with the same seed see the same speakers: their difference is paired."""
    first, second = _by_score(), _by_score()
    kwargs = {"levels": LEVELS_BY, "symmetric": True, "resample": ["speaker"], "n_replicates": 10, "seed": 0}
    joined = bootstrap_details(first, **kwargs).join(
        bootstrap_details(second, **kwargs), on=["phone", "phone_b", "replicate"], suffix="_2"
    )
    assert (joined["score"] - joined["score_2"]).abs().max() == 0
