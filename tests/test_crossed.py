"""Tests for the closed-form statistics of ``fastabx.stats``: ``crossed_mean`` and ``compare``."""

import itertools
from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from fastabx import Dataset, Score, Task
from fastabx.stats import InvalidDesignError, InvalidUnitError, MismatchedScoresError, compare, crossed_mean

from .reference import reference_crossed_variance, reference_schatz_variance, reference_unit_scores

PHONES, SPEAKERS, CONTEXTS, ITEMS = 3, 4, 2, 4
# Fraction of negative variance estimates tolerated on the random-effects model (about 0.5% in practice).
MAX_NEGATIVE_FRACTION = 0.01
# Coverage of the nominal 95% normal interval. Few models per group make it slightly anticonservative.
COVERAGE_BOUNDS = (0.90, 0.98)


def _table(values: np.ndarray, names: Sequence[str] | None = None) -> pl.DataFrame:
    """Long table of a tensor: one integer column per axis, and a ``score`` column."""
    names = list(names or [f"f{i}" for i in range(values.ndim)])
    index = np.array(list(itertools.product(*map(range, values.shape))), dtype=np.int64).reshape(-1, values.ndim)
    return pl.DataFrame({**{n: index[:, i] for i, n in enumerate(names)}, "score": values.reshape(-1)})


# ------------------------------------------------------------------------------------------------- crossed_mean


def test_single_factor_is_the_variance_of_the_mean() -> None:
    values = np.random.default_rng(0).normal(size=17)
    out = crossed_mean(_table(values), factors=["f0"])
    assert out.columns == ["estimate", "variance", "n_f0"]
    assert out["estimate"].item() == pytest.approx(values.mean())
    assert out["variance"].item() == pytest.approx(values.var(ddof=1) / len(values))
    assert out["n_f0"].item() == len(values)


@pytest.mark.parametrize("shape", [(2,), (3, 4), (2, 3, 4), (3, 2, 2), (2, 2, 2, 2)])
def test_matches_brute_force(shape: tuple[int, ...]) -> None:
    values = np.random.default_rng(sum(shape)).normal(size=shape) + 0.5
    out = crossed_mean(_table(values), factors=[f"f{i}" for i in range(len(shape))])
    assert out["estimate"].item() == pytest.approx(values.mean(), abs=1e-12)
    assert out["variance"].item() == pytest.approx(reference_crossed_variance(values), abs=1e-12)
    assert out.select(pl.selectors.starts_with("n_")).row(0) == shape


@pytest.mark.parametrize("shape", [(4, 3, 3), (5, 4, 3), (3, 2, 2), (2, 2, 2)])
def test_matches_the_paper_kernel(shape: tuple[int, int, int]) -> None:
    """The closed form equals the (2, 2, 2) kernel estimator of Schatz et al. on all three factors."""
    rng = np.random.default_rng(shape[0])
    values = rng.normal(size=shape) + rng.normal(size=(shape[0], 1, 1)) + rng.normal(size=(1, shape[1], 1))
    out = crossed_mean(_table(values, ["speaker", "model_1", "model_2"]), factors=["speaker", "model_1", "model_2"])
    assert out["variance"].item() == pytest.approx(reference_schatz_variance(values), abs=1e-12)


@settings(max_examples=150)
@given(
    arrays(
        np.float64,
        st.lists(st.integers(1, 3), min_size=1, max_size=3).map(tuple),
        elements=st.floats(-1, 1, allow_nan=False, width=32),
    )
)
def test_property_matches_brute_force(values: np.ndarray) -> None:
    """Any small design, single-level factors included."""
    out = crossed_mean(_table(values), factors=[f"f{i}" for i in range(values.ndim)])
    assert out["estimate"].item() == pytest.approx(values.mean(), abs=1e-9)
    expected = reference_crossed_variance(values)
    if expected is None:
        assert out["variance"].item() is None
    else:
        assert out["variance"].item() == pytest.approx(expected, abs=1e-9)


def test_unbiased_on_a_crossed_random_effects_model() -> None:
    """Over many simulated designs, vectorized with ``by``, the variance estimates average to the true variance."""
    rng = np.random.default_rng(0)
    n_sims, (ns, na, nj) = 3000, (5, 3, 3)
    sd = {"s": 1.0, "a": 0.5, "j": 0.7, "sa": 0.3, "sj": 0.2, "aj": 0.4, "saj": 0.6}
    true = (
        sd["s"] ** 2 / ns
        + sd["a"] ** 2 / na
        + sd["j"] ** 2 / nj
        + sd["sa"] ** 2 / (ns * na)
        + sd["sj"] ** 2 / (ns * nj)
        + sd["aj"] ** 2 / (na * nj)
        + sd["saj"] ** 2 / (ns * na * nj)
    )
    shape = (n_sims, ns, na, nj)
    values = (
        rng.normal(0, sd["s"], (n_sims, ns, 1, 1))
        + rng.normal(0, sd["a"], (n_sims, 1, na, 1))
        + rng.normal(0, sd["j"], (n_sims, 1, 1, nj))
        + rng.normal(0, sd["sa"], (n_sims, ns, na, 1))
        + rng.normal(0, sd["sj"], (n_sims, ns, 1, nj))
        + rng.normal(0, sd["aj"], (n_sims, 1, na, nj))
        + rng.normal(0, sd["saj"], shape)
    )
    out = crossed_mean(_table(values, ["sim", "s", "a", "j"]), factors=["s", "a", "j"], by=["sim"])
    assert out["sim"].to_list() == list(range(n_sims))
    variances = out["variance"].to_numpy()
    assert abs(variances.mean() - true) < 4 * variances.std() / np.sqrt(n_sims)
    assert out["estimate"].var() == pytest.approx(true, rel=0.1)
    assert (variances < 0).mean() < MAX_NEGATIVE_FRACTION


def test_normal_interval_has_its_nominal_coverage() -> None:
    """With enough units and models, ``estimate +- 1.96 sqrt(variance)`` covers the true mean about 95% of the time.

    This is the asymptotic z-test of the paper, under the null hypothesis.
    """
    rng = np.random.default_rng(1)
    n_sims, (ns, na, nj) = 2000, (20, 6, 6)
    values = (
        rng.normal(0, 1.0, (n_sims, ns, 1, 1))
        + rng.normal(0, 0.5, (n_sims, 1, na, 1))
        + rng.normal(0, 0.5, (n_sims, 1, 1, nj))
        + rng.normal(0, 0.5, (n_sims, ns, na, nj))
    )
    out = crossed_mean(_table(values, ["sim", "s", "a", "j"]), factors=["s", "a", "j"], by=["sim"])
    z = out["estimate"].to_numpy() / np.sqrt(out["variance"].to_numpy())
    coverage = (np.abs(z) < 1.96).mean()  # ruff: ignore[magic-value-comparison]
    assert COVERAGE_BOUNDS[0] < coverage < COVERAGE_BOUNDS[1]


def test_groups_are_independent() -> None:
    """Each ``by`` group, null keys included, gives the same result as on its own, in order of appearance."""
    rng = np.random.default_rng(1)
    tables = [
        _table(rng.normal(size=(3, 2)), ["speaker", "model"]).with_columns(contrast=pl.lit(name))
        for name in ["b", None, "a"]
    ]
    out = crossed_mean(pl.concat(tables), factors=["speaker", "model"], by=["contrast"])
    assert out["contrast"].to_list() == ["b", None, "a"]
    for row, table in zip(out.iter_rows(named=True), tables, strict=True):
        alone = crossed_mean(table, factors=["speaker", "model"]).row(0, named=True)
        assert row["estimate"] == pytest.approx(alone["estimate"])
        assert row["variance"] == pytest.approx(alone["variance"])


def test_groups_can_have_different_sizes() -> None:
    rng = np.random.default_rng(2)
    small, large = rng.normal(size=(2, 3)), rng.normal(size=(5, 3))
    table = pl.concat(
        [
            _table(small, ["speaker", "model"]).with_columns(contrast=pl.lit("x")),
            _table(large, ["speaker", "model"]).with_columns(contrast=pl.lit("y")),
        ]
    )
    out = crossed_mean(table, factors=["speaker", "model"], by=["contrast"])
    assert out["n_speaker"].to_list() == [2, 5]
    assert out["variance"].to_list() == pytest.approx(
        [reference_crossed_variance(small), reference_crossed_variance(large)]
    )


def test_invariant_to_row_order_shift_and_scaled_by_the_square() -> None:
    values = np.random.default_rng(3).normal(size=(4, 3, 2))
    table = _table(values)
    factors = ["f0", "f1", "f2"]
    base = crossed_mean(table, factors=factors).row(0, named=True)
    shuffled = crossed_mean(table.sample(fraction=1, shuffle=True, seed=0), factors=factors).row(0, named=True)
    assert shuffled["variance"] == pytest.approx(base["variance"], abs=1e-14)
    shifted = crossed_mean(table.with_columns(pl.col("score") + 1e6), factors=factors).row(0, named=True)
    assert shifted["variance"] == pytest.approx(base["variance"], rel=1e-6)
    assert shifted["estimate"] == pytest.approx(base["estimate"] + 1e6)
    scaled = crossed_mean(table.with_columns(pl.col("score") * 3), factors=factors).row(0, named=True)
    assert scaled["variance"] == pytest.approx(9 * base["variance"])


def test_single_level_factor_is_treated_as_fixed() -> None:
    values = np.random.default_rng(4).normal(size=(6, 1, 3))
    out = crossed_mean(_table(values), factors=["f0", "f1", "f2"]).row(0, named=True)
    without = crossed_mean(_table(values[:, 0, :], ["f0", "f2"]), factors=["f0", "f2"]).row(0, named=True)
    assert out["variance"] == pytest.approx(without["variance"])
    assert out["n_f1"] == 1


def test_all_single_level_factors_give_a_null_variance() -> None:
    out = crossed_mean(_table(np.array([[0.3]])), factors=["f0", "f1"])
    assert out["estimate"].item() == pytest.approx(0.3)
    assert out["variance"].item() is None


def test_constant_values_give_zero_variance() -> None:
    out = crossed_mean(_table(np.full((3, 4), 0.25)), factors=["f0", "f1"])
    assert out["variance"].item() == pytest.approx(0, abs=1e-15)


def test_negative_estimates_are_returned() -> None:
    """A pure interaction is anticorrelated across distinct cells: the unbiased estimate is negative."""
    out = crossed_mean(_table(np.array([[1.0, -1.0], [-1.0, 1.0]])), factors=["f0", "f1"])
    assert out["variance"].item() == pytest.approx(-1.0)
    assert reference_crossed_variance(np.array([[1.0, -1.0], [-1.0, 1.0]])) == pytest.approx(-1.0)


def test_value_and_factor_dtypes() -> None:
    table = pl.DataFrame(
        {
            "speaker": pl.Series(["s1", "s2", "s1", "s2"], dtype=pl.Categorical),
            "model": [1, 1, 2, 2],
            "errors": [1, 3, 2, 6],
        }
    )
    out = crossed_mean(table, factors=["speaker", "model"], value="errors").row(0, named=True)
    assert out["estimate"] == pytest.approx(3.0)
    assert out["variance"] == pytest.approx(reference_crossed_variance(np.array([[1, 2], [3, 6]])))


@pytest.mark.parametrize(
    ("table", "kwargs", "match"),
    [
        (_table(np.ones((2, 2)))[:3], {}, "not fully crossed"),
        (pl.concat([_table(np.ones((2, 2))), _table(np.ones((2, 2)))[:1]]), {}, "not fully crossed"),
        (_table(np.ones((2, 2))).with_columns(f1=pl.Series([0, 0, 0, 1])), {}, "not fully crossed"),
        (_table(np.ones((2, 2))).with_columns(score=pl.Series([1.0, None, 1.0, 1.0])), {}, "missing values"),
        (_table(np.array([[1.0, np.nan], [1.0, 1.0]])), {}, "missing values"),
        (_table(np.ones((2, 2))), {"factors": []}, "empty"),
        (_table(np.ones((2, 2))), {"by": ["f0"]}, "distinct"),
        (_table(np.ones((2, 2))), {"value": "f0"}, "distinct"),
        (_table(np.ones((2, 2))), {"factors": ["f0", "f0"]}, "distinct"),
        (_table(np.ones((2, 2))), {"by": ["unknown"]}, "No column"),
    ],
)
def test_invalid_designs(table: pl.DataFrame, kwargs: dict[str, Any], match: str) -> None:
    options: dict[str, Any] = {"factors": ["f0", "f1"], "value": "score", "by": []} | kwargs
    with pytest.raises(InvalidDesignError, match=match):
        crossed_mean(table, factors=options["factors"], value=options["value"], by=options["by"])


def test_large_design() -> None:
    """500 contrasts x 30 speakers x 10 x 10 models, as in the paper, in one call."""
    rng = np.random.default_rng(5)
    values = rng.normal(size=(500, 30, 10, 10))
    out = crossed_mean(_table(values, ["contrast", "s", "a", "j"]), factors=["s", "a", "j"], by=["contrast"])
    assert len(out) == 500
    for c in (0, 499):
        sub = values[c]
        assert out["estimate"][c] == pytest.approx(sub.mean())
        assert np.isfinite(out["variance"][c])


# ------------------------------------------------------------------------------------------------------ compare


def _labels() -> pl.DataFrame:
    n = PHONES * SPEAKERS * CONTEXTS * ITEMS
    return pl.DataFrame(
        {
            "phone": [f"p{i % PHONES}" for i in range(n)],
            "speaker": [f"s{(i // PHONES) % SPEAKERS}" for i in range(n)],
            "context": [f"c{(i // (PHONES * SPEAKERS)) % CONTEXTS}" for i in range(n)],
        }
    )


def _model(
    seed: int,
    *,
    separation: float = 1.0,
    labels: pl.DataFrame | None = None,
    by: Sequence[str] = ("speaker", "context"),
    across: Sequence[str] = (),
) -> Score:
    """Score of one "model": phone means scaled by ``separation``, plus speaker shifts and noise."""
    labels = _labels() if labels is None else labels
    rng = np.random.default_rng(seed)
    phone_means = np.random.default_rng(100).normal(size=(PHONES, 6))
    speaker_shift = np.random.default_rng(101).normal(size=(SPEAKERS, 6)) * 0.5
    phones = labels["phone"].str.slice(1).cast(int).to_numpy()
    speakers = labels["speaker"].str.slice(1).cast(int).to_numpy()
    features = separation * phone_means[phones] + speaker_shift[speakers] + rng.normal(size=(len(labels), 6))
    dataset = Dataset.from_numpy(features.astype(np.float32), labels)
    return Score(Task(dataset, on="phone", by=list(by), across=list(across)), "euclidean", progress=False)


def _reference_compare(
    first: list[Score], second: list[Score], *, symmetric: bool, average_contrasts: bool = False
) -> dict[tuple, tuple[float, float | None]]:
    """Compute the paper's estimator from plain cell rows with loops, for every contrast."""
    tables = [
        [
            reference_unit_scores(
                list(score.cells.iter_rows(named=True)),
                on="phone",
                unit="speaker",
                context="context",
                symmetric=symmetric,
                average_contrasts=average_contrasts,
            )
            for score in scores
        ]
        for scores in (first, second)
    ]
    groups: dict[tuple, list] = {}
    for key in tables[0][0]:
        groups.setdefault(key[:-1], []).append(key[-1])
    out = {}
    for group, units in groups.items():
        values = np.array(
            [[[e1[*group, u] - e2[*group, u] for e2 in tables[1]] for e1 in tables[0]] for u in sorted(units)]
        )
        out[group] = (float(values.mean()), reference_crossed_variance(values))
    return out


def _assert_compare(got: pl.DataFrame, expected: dict[tuple, tuple[float, float | None]], keys: list[str]) -> None:
    assert len(got) == len(expected)
    for row in got.iter_rows(named=True):
        estimate, variance = expected[tuple(row[k] for k in keys)]
        assert row["estimate"] == pytest.approx(estimate, abs=1e-12)
        if variance is None:
            assert row["variance"] is None
        else:
            assert row["variance"] == pytest.approx(variance, abs=1e-12)


@pytest.mark.parametrize("symmetric", [True, False])
def test_compare_matches_the_paper_estimator(symmetric: bool) -> None:  # ruff: ignore[boolean-type-hint-positional-argument]
    first = [_model(seed, separation=1.0) for seed in range(3)]
    second = [_model(seed, separation=0.6) for seed in range(10, 12)]
    got = compare(first, second, levels=["context"], symmetric=symmetric)
    assert got.columns == ["phone", "phone_b", "estimate", "variance", "n_speaker", "n_model_1", "n_model_2"]
    assert got.select("n_speaker", "n_model_1", "n_model_2").unique().rows() == [(SPEAKERS, 3, 2)]
    assert len(got) == (3 if symmetric else 6)
    _assert_compare(got, _reference_compare(first, second, symmetric=symmetric), ["phone", "phone_b"])
    assert (got["estimate"] < 0).all(), "better-separated phones give lower error rates"


def test_compare_matches_the_paper_kernel_exactly() -> None:
    """End to end against the literal (2, 2, 2) kernel of the paper, on one contrast."""
    first = [_model(seed, separation=1.0) for seed in range(3)]
    second = [_model(seed, separation=0.8) for seed in range(10, 13)]
    got = compare(first, second, levels=["context"], symmetric=True).filter(phone="p0", phone_b="p1")
    tables = [
        [
            reference_unit_scores(
                list(s.cells.iter_rows(named=True)),
                on="phone",
                unit="speaker",
                context="context",
                symmetric=True,
                average_contrasts=False,
            )
            for s in scores
        ]
        for scores in (first, second)
    ]
    values = np.array(
        [
            [[e1["p0", "p1", f"s{u}"] - e2["p0", "p1", f"s{u}"] for e2 in tables[1]] for e1 in tables[0]]
            for u in range(SPEAKERS)
        ]
    )
    assert got["variance"].item() == pytest.approx(reference_schatz_variance(values), abs=1e-12)


def test_one_model_per_side_is_a_paired_comparison_over_units() -> None:
    first, second = _model(0, separation=1.0), _model(1, separation=0.7)
    got = compare(first, second, levels=["context"], symmetric=True)
    assert got.select("n_model_1", "n_model_2").unique().rows() == [(1, 1)]
    per_unit = [
        first.details(levels=["context"], symmetric=True).sort("phone", "phone_b", "speaker"),
        second.details(levels=["context"], symmetric=True).sort("phone", "phone_b", "speaker"),
    ]
    paired = per_unit[0].with_columns(difference=pl.col("score") - per_unit[1]["score"])
    expected = paired.group_by("phone", "phone_b", maintain_order=True).agg(
        estimate=pl.col("difference").mean(), variance=pl.col("difference").var() / pl.len()
    )
    joined = got.join(expected, on=["phone", "phone_b"], suffix="_expected")
    assert np.allclose(joined["estimate"], joined["estimate_expected"], atol=1e-12)
    assert np.allclose(joined["variance"], joined["variance_expected"], atol=1e-12)


def test_single_score_or_sequence() -> None:
    first, second = _model(0), _model(1, separation=0.5)
    assert compare(first, second, levels=["context"]).equals(compare([first], (second,), levels=["context"]))


def test_default_levels_compare_every_cell() -> None:
    """Without ``levels``, every condition but the unit defines its own comparison."""
    first, second = _model(0), _model(1, separation=0.5)
    got = compare(first, second)
    assert set(got.columns[:3]) == {"phone", "phone_b", "context"}
    assert got["n_speaker"].unique().to_list() == [SPEAKERS]
    assert got.equals(compare(first, second, levels=[]))


def test_condition_named_model() -> None:
    labels = _labels().rename({"context": "model"})
    first = [_model(seed, labels=labels, by=["speaker", "model"]) for seed in range(2)]
    second = [_model(seed, labels=labels, by=["speaker", "model"], separation=0.5) for seed in range(10, 12)]
    got = compare(first, second, levels=["model"])
    assert got.columns == ["phone", "phone_b", "estimate", "variance", "n_speaker", "n_model_1", "n_model_2"]
    assert got.select("n_model_1", "n_model_2").unique().rows() == [(2, 2)]


def test_identical_groups() -> None:
    """A group compared to itself has a zero mean difference; a single model compared to itself has no variance.

    With several models, the differences between distinct models of the "two" groups are not zero, so neither is
    the variance (the groups are not independent samples either, so it is not meaningful).
    """
    scores = [_model(seed) for seed in range(2)]
    got = compare(scores, scores, levels=["context"], symmetric=True)
    assert np.allclose(got["estimate"], 0, atol=1e-15)
    single = compare(scores[0], scores[0], levels=["context"], symmetric=True)
    assert np.allclose(single["estimate"], 0, atol=1e-15)
    assert np.allclose(single["variance"], 0, atol=1e-15)


def test_swapping_groups_negates_the_estimate() -> None:
    first = [_model(seed) for seed in range(2)]
    second = [_model(seed, separation=0.5) for seed in range(10, 13)]
    forward = compare(first, second, levels=["context"])
    backward = compare(second, first, levels=["context"])
    assert np.allclose(forward["estimate"], -backward["estimate"], atol=1e-15)
    assert np.allclose(forward["variance"], backward["variance"], atol=1e-15)
    assert backward.select("n_model_1", "n_model_2").row(0) == (3, 2)


def test_averaging_the_contrasts_gives_a_single_comparison() -> None:
    first = [_model(seed) for seed in range(2)]
    second = [_model(seed, separation=0.6) for seed in range(10, 12)]
    got = compare(first, second, levels=["context", ("phone", "phone_b")], symmetric=True)
    assert got.columns == ["estimate", "variance", "n_speaker", "n_model_1", "n_model_2"]
    expected = _reference_compare(first, second, symmetric=True, average_contrasts=True)
    _assert_compare(got, expected, [])


def test_one_direction_contrasts_drop_the_unit() -> None:
    """With p2 having a single item for speaker s0, (p2, x) never exists there: s0 leaves the contrasts with p2."""
    labels = _labels().with_row_index("row")
    lonely = (pl.col("phone") == "p2") & (pl.col("speaker") == "s0")
    labels = labels.filter(~lonely | (pl.col("row").cum_count().over(lonely) == 1)).drop("row")
    first = [_model(seed, labels=labels) for seed in range(2)]
    second = [_model(seed, separation=0.6, labels=labels) for seed in range(10, 12)]
    got = compare(first, second, levels=["context"], symmetric=True)
    counts = dict(zip(zip(got["phone"], got["phone_b"], strict=True), got["n_speaker"], strict=True))
    assert counts == {("p0", "p1"): SPEAKERS, ("p0", "p2"): SPEAKERS - 1, ("p1", "p2"): SPEAKERS - 1}
    _assert_compare(got, _reference_compare(first, second, symmetric=True), ["phone", "phone_b"])
    asymmetric = compare(first, second, levels=["context"])
    assert asymmetric.filter(phone="p0", phone_b="p2")["n_speaker"].item() == SPEAKERS


def test_constrained_cells_are_left_out() -> None:
    """Cells with no valid triplet are null for every model, and are dropped before comparing."""
    labels = _labels().with_columns(
        take=pl.when(pl.col("speaker") == "s0")
        .then(pl.lit("r0"))
        .otherwise("r" + (pl.int_range(pl.len()) // 24 % 2).cast(str))
    )

    def constrained(seed: int, separation: float) -> Score:
        rng = np.random.default_rng(seed)
        dataset = Dataset.from_numpy(rng.normal(size=(len(labels), 6)) * separation, labels)
        task = Task(dataset, on="phone", by=["speaker", "context"])
        return Score(task, "euclidean", constraints=[pl.col("take_a") != pl.col("take_x")], progress=False)

    first, second = [constrained(s, 1.0) for s in range(2)], [constrained(s, 0.5) for s in range(10, 12)]
    assert first[0].cells.filter(speaker="s0")["score"].is_null().all()
    got = compare(first, second, levels=["context"], symmetric=True)
    assert got["n_speaker"].unique().to_list() == [SPEAKERS - 1]
    _assert_compare(got, _reference_compare(first, second, symmetric=True), ["phone", "phone_b"])


def test_invalid_units() -> None:
    score = _model(0)
    with pytest.raises(InvalidUnitError, match="the ON condition"):
        compare(score, score, unit="phone", levels=["context"])
    with pytest.raises(InvalidUnitError, match="not a condition"):
        compare(score, score, unit="unknown", levels=["context"])
    with pytest.raises(InvalidUnitError, match="averaged over"):
        compare(score, score, levels=["context", "speaker"])
    across = _model(0, by=["context"], across=["speaker"])
    with pytest.raises(InvalidUnitError, match="an ACROSS condition"):
        compare(across, across, levels=["context"])
    dataset = Dataset.from_numpy(np.random.default_rng(0).normal(size=(len(_labels()), 6)), _labels())
    task = Task(dataset, on="phone", by=["speaker", "context"])
    precomputed = Score(Task.from_cells(dataset, task.cells, is_symmetric=True), "euclidean", progress=False)
    with pytest.raises(InvalidUnitError, match="not a condition"):
        compare(precomputed, precomputed, levels=["context"])


def test_mismatched_scores() -> None:
    score = _model(0)
    with pytest.raises(MismatchedScoresError, match="at least one"):
        compare(score, [], levels=["context"])
    other_speakers = _model(1, labels=_labels().with_columns(pl.col("speaker").str.replace("s", "t")))
    with pytest.raises(MismatchedScoresError, match="same valid cells"):
        compare(score, other_speakers, levels=["context"])
    extra = _model(1, labels=_labels().with_columns(extra=pl.lit("x")), by=["speaker", "context", "extra"])
    with pytest.raises(MismatchedScoresError, match="same valid cells"):
        compare(score, extra, levels=["context"])
    # Same groups once the contexts are averaged, but s0 has a single context: the two sides would average
    # different cells within that speaker.
    missing = _labels().filter((pl.col("speaker") != "s0") | (pl.col("context") == "c0"))
    with pytest.raises(MismatchedScoresError, match="same valid cells"):
        compare(score, _model(1, labels=missing), levels=["context"])
