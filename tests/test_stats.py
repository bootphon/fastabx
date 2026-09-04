"""Tests for ``fastabx.stats``: the vectorized bootstrap collapse against a duplicated-categories reference."""

from collections.abc import Sequence

import numpy as np
import polars as pl
import pytest

from fastabx import Bootstrap, Dataset, Score, Task
from fastabx.score import CollapseError, pl_weighted_mean, score_details
from fastabx.stats import (
    InvalidResampleError,
    NestedBootstrapWarning,
    bootstrap_collapse,
    factor_weights,
    percentile_interval,
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
    """Collapse each replicate by materializing it: duplicate the drawn categories, then use the plain collapse."""
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
        if weighted:
            out.append(cells.select(pl_weighted_mean("score", "size")).item())
        else:
            out.append(score_details(cells, levels=levels)["score"].mean())
    return np.array(out, dtype=np.float64)


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


def test_tokens_false_ignores_the_token_replicates() -> None:
    score = _by_score(bootstrap=Bootstrap(4, seed=0))
    got = bootstrap_collapse(score, levels=LEVELS_BY, resample=["speaker"], tokens=False, n_replicates=5, seed=0)
    plain = bootstrap_collapse(_by_score(), levels=LEVELS_BY, resample=["speaker"], n_replicates=5, seed=0)
    assert np.array_equal(got, plain)


def test_chunking_invariance(monkeypatch: pytest.MonkeyPatch) -> None:
    score = _by_score(bootstrap=Bootstrap(7, seed=0))
    baseline = bootstrap_collapse(score, levels=LEVELS_BY)
    categories = bootstrap_collapse(score, levels=LEVELS_BY, resample=["speaker"], tokens=False, n_replicates=7)
    monkeypatch.setenv("FASTABX_BOOTSTRAP_CHUNK_ELEMS", "1")
    assert np.array_equal(baseline, bootstrap_collapse(score, levels=LEVELS_BY))
    chunked = bootstrap_collapse(score, levels=LEVELS_BY, resample=["speaker"], tokens=False, n_replicates=7)
    assert np.array_equal(categories, chunked)


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


def test_percentile_interval() -> None:
    replicates = np.array([*np.linspace(0, 1, 101), np.nan])
    assert percentile_interval(replicates) == pytest.approx((0.025, 0.975))
    assert percentile_interval(replicates, alpha=0.5) == pytest.approx((0.25, 0.75))
    for alpha in (0, 1, -0.1):
        with pytest.raises(ValueError, match="alpha"):
            percentile_interval(replicates, alpha=alpha)


def test_invalid_requests() -> None:
    score = _by_score()
    with pytest.raises(InvalidResampleError, match="not a condition"):
        bootstrap_collapse(score, levels=LEVELS_BY, resample=["unknown"], n_replicates=2)
    with pytest.raises(InvalidResampleError, match="n_replicates"):
        bootstrap_collapse(score, levels=LEVELS_BY, resample=["speaker"])
    with pytest.raises(InvalidResampleError, match="Nothing to resample"):
        bootstrap_collapse(score, levels=LEVELS_BY, tokens=False)
    with pytest.raises(CollapseError):
        bootstrap_collapse(score, levels=LEVELS_BY, weighted=True, resample=["speaker"], n_replicates=2)
    with pytest.raises(TypeError):
        bootstrap_collapse(score, levels=LEVELS_BY, resample=["speaker"], n_replicates=0)
    with_tokens = _by_score(bootstrap=Bootstrap(3))
    with pytest.raises(InvalidResampleError, match="does not match"):
        bootstrap_collapse(with_tokens, levels=LEVELS_BY, n_replicates=4)


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
