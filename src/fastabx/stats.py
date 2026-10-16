"""Statistics on scored ABX tasks: closed-form variances of crossed designs, and bootstrap distributions.

**Closed form.** :py:func:`crossed_mean` gives the mean of a fully crossed design (e.g. speakers x models) and the
unbiased estimator of its variance, the one of a generalized U-statistic. :py:func:`compare` uses it to compare two
groups of models, as in Schatz et al. (PNAS, Supplementary Materials and Methods 4). Tests, p-values and corrections
for multiple comparisons are left to SciPy or statsmodels.

**Bootstrap.** Two kinds of resampling are supported, and can be combined:

- **tokens**: the items inside each cell, drawn when scoring with ``Score(..., bootstrap=Bootstrap(n))``. They
  need the distances, so they are computed alongside the point estimate (see :py:mod:`fastabx.bootstrap`).
- **categories**: the values of a condition (speakers, contexts, phones...). Resampling a category never requires
  rescoring: a category drawn ``k`` times counts ``k`` times when the cells are averaged, and a category that was
  not drawn drops out. This is done entirely here, at collapse time.

Resampling categories is defined as duplicating them: a speaker drawn twice behaves as two distinct speakers with
identical cells. A cell involving two values of the same condition (the ``phone``/``phone_b`` contrast, or the
``speaker``/``speaker_x`` pair of an across task) therefore weighs ``w(a) * w(b)``, and no cell ever pairs a value
with one of its own copies.
"""

import functools
import hashlib
import itertools
import operator
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl
import polars.selectors as cs
import torch

from fastabx.cell import INDEX_COLUMNS
from fastabx.score import (
    CollapseError,
    NoBootstrapError,
    Score,
    canonical_contrast,
    collapse_levels,
    contrast_condition,
)
from fastabx.utils import bootstrap_chunk_elems
from fastabx.verify import verify_bootstrap_params

__all__ = [
    "InvalidDesignError",
    "InvalidResampleError",
    "InvalidUnitError",
    "MismatchedScoresError",
    "NestedBootstrapWarning",
    "bootstrap_collapse",
    "bootstrap_details",
    "compare",
    "crossed_mean",
]

CONDITION_SUFFIXES = ("", "_b", "_x")
DIRECTIONS = 2


class InvalidResampleError(ValueError):
    """The requested bootstrap cannot be built from this ``Score``."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)


class InvalidDesignError(ValueError):
    """The table given to :py:func:`.crossed_mean` is not a fully crossed design."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)


class InvalidUnitError(ValueError):
    """The unit of :py:func:`.compare` must be a BY condition of every ``Score``, left after ``levels``."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)


class MismatchedScoresError(ValueError):
    """The Scores given to :py:func:`.compare` do not have the same groups."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)


class NestedBootstrapWarning(UserWarning):
    """Tokens and categories are both resampled, which counts the token variability twice."""


def factor_weights(values: pl.Series, name: str, *, n_replicates: int, seed: int) -> npt.NDArray[np.int64]:
    """Draw the ``(n_replicates, len(values))`` multiplicities of each value of a resampled condition.

    Each row is a with-replacement resample of the values, of the same size. The draw only depends on ``seed`` and
    the condition ``name``, so the same values are drawn for every ``Score`` sharing that condition: comparing two
    models replicate by replicate is a paired comparison.
    """
    key = hashlib.blake2b(name.encode(), digest_size=8).digest()
    rng = np.random.default_rng([seed, int.from_bytes(key, "little")])
    n = len(values)
    return rng.multinomial(n, np.full(n, 1 / n), size=n_replicates)


@dataclass(frozen=True)
class Factor:
    """A resampled condition: its sorted values, and their multiplicities in every replicate."""

    values: pl.Series
    weights: npt.NDArray[np.int64]

    def codes(self, column: pl.Series) -> npt.NDArray[np.int64]:
        """Position of each label of ``column`` in ``values``."""
        return self.values.search_sorted(column).to_numpy().astype(np.int64)


@dataclass(frozen=True)
class Step:
    """One aggregation of the collapse: rows are averaged into ``n_groups`` groups given by ``ids``.

    ``weights`` lists, for every resampled column averaged over in this step, its condition name and the code of
    each row's label. The weight of a row is the product of the multiplicities of those labels.
    When ``rows`` is set, only those rows enter the step. When ``complete`` is set, a group is ``nan`` unless exactly
    that many of its rows are valid.
    """

    ids: torch.Tensor
    n_groups: int
    weights: list[tuple[str, npt.NDArray[np.int64]]]
    rows: torch.Tensor | None = None
    complete: int = 0


def resolve_factors(
    cells: pl.DataFrame, resample: Sequence[str], *, n_replicates: int, seed: int
) -> tuple[dict[str, Factor], dict[str, str]]:
    """Draw every resampled condition and map each of its columns (``c``, ``c_b``, ``c_x``) back to it."""
    factors, columns = {}, {}
    for name in dict.fromkeys(resample):
        present = [name + suffix for suffix in CONDITION_SUFFIXES if name + suffix in cells.columns]
        if not present:
            msg = f"Cannot resample {name!r}: it is not a condition of the scored task."
            raise InvalidResampleError(msg)
        values = pl.concat([cells[c].rename(name) for c in present]).unique().sort()
        factors[name] = Factor(values, factor_weights(values, name, n_replicates=n_replicates, seed=seed))
        columns |= dict.fromkeys(present, name)
    return factors, columns


def step_weights(
    frame: pl.DataFrame, averaged: Sequence[str], factors: dict[str, Factor], columns: dict[str, str]
) -> list[tuple[str, npt.NDArray[np.int64]]]:
    """Codes of the resampled columns among ``averaged``, for the rows of ``frame``."""
    return [(columns[c], factors[columns[c]].codes(frame[c])) for c in averaged if c in columns]


def group_rows(frame: pl.DataFrame, keys: list[str]) -> tuple[torch.Tensor, pl.DataFrame]:
    """Group id of every row of ``frame`` given ``keys``, and one row per group (in the same order as the ids)."""
    groups = frame.with_row_index("__row").group_by(keys, maintain_order=True).agg("__row").with_row_index("__group")
    ids = groups.select("__group", "__row").explode("__row", empty_as_null=False).sort("__row")["__group"]
    return torch.from_numpy(ids.to_numpy().astype(np.int64)), groups.select(keys)


def symmetrize_step(frame: pl.DataFrame, on: str) -> tuple[Step, pl.DataFrame]:
    """Merge the ``(a, b)`` and ``(b, a)`` rows, as :py:func:`fastabx.score.symmetrize_cells` does on the scores."""
    canonical = frame.with_columns(canonical_contrast(on)).with_row_index("__row")
    paired = canonical.filter(pl.len().over(frame.columns) == DIRECTIONS)
    ids, grouped = group_rows(paired.drop("__row"), frame.columns)
    rows = torch.from_numpy(paired["__row"].to_numpy().astype(np.int64))
    return Step(ids, len(grouped), [], rows=rows, complete=DIRECTIONS), grouped


def collapse_plan(
    cells: pl.DataFrame,
    levels: Sequence[tuple[str, ...] | str] | None,
    factors: dict[str, Factor],
    columns: dict[str, str],
    *,
    on: str | None,
    final: bool,
) -> tuple[list[Step], pl.DataFrame]:
    """Turn a collapse into a list of weighted aggregations, computed once for all replicates.

    Mirrors :py:meth:`.Score.details`: an optional merge of the two directions of each contrast, then each level
    averages over its columns, grouping by every other remaining column. With ``final``, a last step averages over
    everything left, as :py:meth:`.Score.collapse` does. The ``_x`` labels, dropped by the collapse, are averaged
    over in the first level.

    :returns: The steps, and the labels of the groups left after the last step.
    """
    steps = []
    if on is not None:
        step, cells = symmetrize_step(cells, on)
        steps.append(step)
    x_columns = cells.select(cs.ends_with("_x") - INDEX_COLUMNS).columns
    pending = step_weights(cells, x_columns, factors, columns)
    frame, levels_in_tuples = collapse_levels(cells, levels)
    if pending and not levels_in_tuples and not final:
        msg = f"The resampled columns {x_columns} must be averaged over: collapse at least one level."
        raise InvalidResampleError(msg)
    n_rows = len(frame)
    for level in [*levels_in_tuples, *([None] if final else [])]:
        averaged = frame.columns if level is None else list(level)
        keys = [c for c in frame.columns if c not in averaged]
        weights = pending + step_weights(frame, averaged, factors, columns)
        if keys:
            ids, frame = group_rows(frame, keys)
        else:
            ids, frame = torch.zeros(n_rows, dtype=torch.int64), pl.DataFrame()
        n_rows = len(frame) if keys else 1
        steps.append(Step(ids, n_rows, weights))
        pending = []
    return steps, frame


def row_weights(
    weights: list[tuple[str, npt.NDArray[np.int64]]], factors: dict[str, Factor], replicates: slice, n_rows: int
) -> torch.Tensor:
    """``(n_rows, len(replicates))`` product of the multiplicities of each row's labels."""
    out = torch.ones((n_rows, replicates.stop - replicates.start), dtype=torch.float64)
    for name, codes in weights:
        out *= torch.from_numpy(factors[name].weights[replicates].T[codes].astype(np.float64))
    return out


def weighted_mean(scores: torch.Tensor, weights: torch.Tensor, step: Step) -> torch.Tensor:
    """Per-group weighted mean of ``scores`` (``nan`` are skipped), ``nan`` for groups with no weight."""
    valid = ~torch.isnan(scores)
    weights = torch.where(valid, weights, 0)
    shape = (step.n_groups, scores.size(1))
    numerator = scores.new_zeros(shape).index_add_(0, step.ids, weights * scores.nan_to_num())
    denominator = scores.new_zeros(shape).index_add_(0, step.ids, weights)
    keep = denominator > 0
    if step.complete:
        keep &= scores.new_zeros(shape).index_add_(0, step.ids, valid.to(scores.dtype)) == step.complete
    return torch.where(keep, numerator / denominator, torch.nan)


def replicate_source(
    score: Score, *, resample: Sequence[str], n_replicates: int | None
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], int]:
    """Per-cell scores and sizes to collapse, either the token replicates or the point estimate, and their count."""
    if score.n_replicates > 0:
        if n_replicates is not None:
            msg = (
                f"`n_replicates` is set by the {score.n_replicates} token replicates of this Score. Omit it, or "
                f"resample categories on a Score built without a Bootstrap."
            )
            raise InvalidResampleError(msg)
        if resample:
            warnings.warn(
                "Resampling both tokens and categories counts the token variability twice: it is already part of "
                "the spread between categories. Resample the highest random level only (e.g. speakers), unless "
                "this is what you want.",
                NestedBootstrapWarning,
                stacklevel=3,
            )
        scores, sizes = score.replicates
        return scores, sizes, score.n_replicates
    if not resample:
        raise NoBootstrapError
    if n_replicates is None:
        msg = "`n_replicates` is required when resampling categories without token replicates."
        raise InvalidResampleError(msg)
    scores = score.cells["score"].to_numpy(allow_copy=True).astype(np.float64)[:, None]
    sizes = score.cells["size"].cast(pl.Float64).to_numpy().astype(np.float64)[:, None]
    return scores, sizes, n_replicates


@dataclass(frozen=True)
class Replicates:
    """The resampling scheme of a bootstrap: what to collapse, and the draws of the resampled categories."""

    scores: npt.NDArray[np.float64]
    sizes: npt.NDArray[np.float64]
    n_replicates: int
    cells: pl.DataFrame
    factors: dict[str, Factor]
    columns: dict[str, str]

    @classmethod
    def build(cls, score: Score, *, resample: Sequence[str], n_replicates: int | None, seed: int) -> "Replicates":
        """Gather the replicates to collapse and draw the resampled categories."""
        scores, sizes, n_replicates = replicate_source(score, resample=resample, n_replicates=n_replicates)
        verify_bootstrap_params(n_replicates, seed=seed)
        cells = score.cells.select(cs.exclude("score", "size") - INDEX_COLUMNS)
        factors, columns = resolve_factors(cells, resample, n_replicates=n_replicates, seed=seed)
        return cls(scores, sizes, n_replicates, cells, factors, columns)

    def run(self, steps: list[Step], *, weighted: bool = False) -> npt.NDArray[np.float64]:
        """Apply the steps to every replicate, in chunks, and return the ``(n_groups, n_replicates)`` results."""
        n_cells, n_rep = len(self.cells), self.n_replicates
        out = np.empty((steps[-1].n_groups if steps else n_cells, n_rep), dtype=np.float64)
        width = max(1, bootstrap_chunk_elems() // max(1, n_cells))
        for start in range(0, n_rep, width):
            chunk = slice(start, min(start + width, n_rep))
            count = chunk.stop - chunk.start
            current = self.select(self.scores, chunk, count)
            for i, step in enumerate(steps):
                if step.rows is not None:
                    current = current[step.rows]
                weights = row_weights(step.weights, self.factors, chunk, len(current))
                if weighted and i == 0:
                    weights *= self.select(self.sizes, chunk, count).nan_to_num()
                current = weighted_mean(current, weights, step)
            out[:, chunk] = current.numpy()
        return out

    @staticmethod
    def select(values: npt.NDArray[np.float64], chunk: slice, count: int) -> torch.Tensor:
        """Slice a ``chunk`` of replicates of ``values``, broadcasting a point estimate to every replicate."""
        return torch.from_numpy(values[:, chunk] if values.shape[1] > 1 else values).expand(-1, count)


def bootstrap_collapse(
    score: Score,
    *,
    levels: Sequence[tuple[str, ...] | str] | None = None,
    weighted: bool = False,
    resample: Sequence[str] = (),
    n_replicates: int | None = None,
    seed: int = 0,
) -> npt.NDArray[np.float64]:
    """Bootstrap distribution of :py:meth:`.Score.collapse`, resampling tokens, categories, or both.

    Every replicate is collapsed with the same ``levels``/``weighted`` logic as the point estimate, so the returned
    distribution is directly comparable to ``score.collapse(levels=..., weighted=...)``.

    - **tokens** are the items inside each cell. Their replicates come from ``Score(..., bootstrap=Bootstrap(n))``,
      and are used whenever the ``Score`` has them. They measure how the score would move with other recordings of
      the *same* speakers, contexts and phones.
    - **categories** are the values of the conditions listed in ``resample`` (e.g. ``["speaker"]``), drawn with
      replacement. They measure how the score would move with *other* speakers. They need no rescoring: use a
      ``Score`` built without a ``Bootstrap`` to resample categories only.

    Resampling both counts the token variability twice, since it is already part of the spread between categories,
    and emits a :py:class:`NestedBootstrapWarning`. Resampling two crossed conditions (e.g. speakers and contexts)
    overestimates the variance as well.

    :param score: The scored task.
    :param levels: List of levels to collapse, as in :py:meth:`.Score.collapse`.
    :param weighted: Whether to collapse with a mean weighted by the size of the cells, as in
        :py:meth:`.Score.collapse`.
    :param resample: Conditions whose values are resampled. The ``_b`` and ``_x`` columns of a condition share its
        draw.
    :param n_replicates: Number of replicates when only categories are resampled. With token replicates, it is
        their number and must be omitted.
    :param seed: Seed of the category draws. The token draws are seeded by the :py:class:`.Bootstrap`.
    :returns: A ``(n_replicates,)`` array of collapsed ABX error rates, ``nan`` for a replicate left with no valid
        cell.
    """
    if weighted and levels is not None:
        raise CollapseError(are_set=True)
    replicates = Replicates.build(score, resample=resample, n_replicates=n_replicates, seed=seed)
    if weighted:
        cells, columns = replicates.cells, replicates.columns
        ids = torch.zeros(len(cells), dtype=torch.int64)
        steps = [Step(ids, 1, step_weights(cells, list(columns), replicates.factors, columns))]
    else:
        steps, _ = collapse_plan(replicates.cells, levels, replicates.factors, replicates.columns, on=None, final=True)
    return replicates.run(steps, weighted=weighted)[0]


def bootstrap_details(
    score: Score,
    *,
    levels: Sequence[tuple[str, ...] | str] | None = None,
    symmetric: bool = False,
    resample: Sequence[str] = (),
    n_replicates: int | None = None,
    seed: int = 0,
) -> pl.DataFrame:
    """Bootstrap distribution of :py:meth:`.Score.details`: the score of every group, in every replicate.

    This is what statistics that are not a plain mean need: a median over contrasts, a correlation with
    per-contrast behavioural data, per-contrast intervals, or a paired comparison of two models (collapse both
    with the same ``seed``, join on the conditions and ``replicate``, and take the difference).

    The resampling is the same as in :py:func:`bootstrap_collapse`. Every resampled condition must be averaged
    over in ``levels``: a condition left in the output would lose the multiplicity of its values.

    :param score: The scored task.
    :param levels: List of levels to collapse, as in :py:meth:`.Score.details`.
    :param symmetric: Whether to merge the two directions of each contrast first, as in :py:meth:`.Score.details`.
        In a replicate, a pair is ``null`` when one of its directions is.
    :param resample: Conditions whose values are resampled. The ``_b`` and ``_x`` columns of a condition share its
        draw.
    :param n_replicates: Number of replicates when only categories are resampled. With token replicates, it is
        their number and must be omitted.
    :param seed: Seed of the category draws. The token draws are seeded by the :py:class:`.Bootstrap`.
    :returns: A long DataFrame with the conditions left after ``levels``, a ``replicate`` column and a ``score``
        column, ``null`` for a group left with no valid cell in that replicate.
    """
    replicates = Replicates.build(score, resample=resample, n_replicates=n_replicates, seed=seed)
    on = contrast_condition(score.on) if symmetric else None
    steps, groups = collapse_plan(replicates.cells, levels, replicates.factors, replicates.columns, on=on, final=False)
    if kept := [c for c in groups.columns if c in replicates.columns]:
        msg = f"The resampled columns {kept} are left in the details: average them over in `levels`."
        raise InvalidResampleError(msg)
    values = replicates.run(steps)
    long = groups.with_row_index("__group").join(
        pl.DataFrame({"replicate": np.arange(replicates.n_replicates)}), how="cross"
    )
    flat = values[long["__group"].to_numpy(), long["replicate"].to_numpy()]
    return long.drop("__group").with_columns(score=pl.Series(flat, nan_to_null=True))


def verify_design(df: pl.DataFrame, *, factors: list[str], value: str, by: list[str]) -> None:
    """Check that the factors, value and groups are distinct columns of ``df``."""
    if not factors:
        msg = "`factors` is empty: give at least one factor, e.g. `factors=['speaker']`."
        raise InvalidDesignError(msg)
    names = [*factors, *by, value]
    if len(set(names)) != len(names):
        msg = f"`factors` ({factors}), `by` ({by}) and `value` ({value!r}) must be distinct columns."
        raise InvalidDesignError(msg)
    if missing := [c for c in names if c not in df.columns]:
        msg = f"No column named {missing} in the table. Its columns are: {df.columns}."
        raise InvalidDesignError(msg)
    values = df[value]
    if values.null_count() or (values.dtype.is_float() and values.is_nan().any()):
        msg = f"The column {value!r} has missing values: drop them, or the cells they belong to, first."
        raise InvalidDesignError(msg)


def verify_crossed(groups: pl.DataFrame, *, factors: list[str], by: list[str]) -> None:
    """Check that every group holds exactly one value per combination of the factors."""
    counts = [pl.col(f"n_{f}") for f in factors]
    full = functools.reduce(operator.mul, counts)
    incomplete = groups.filter((pl.col("__rows") != pl.col("__combinations")) | (pl.col("__rows") != full))
    if not incomplete.is_empty():
        first = incomplete.select(*by, pl.col("__rows").alias("rows"), *counts).row(0, named=True)
        msg = (
            f"{len(incomplete)} group(s) are not fully crossed: each must hold exactly one value for every "
            f"combination of {factors}. First one: {first}."
        )
        raise InvalidDesignError(msg)


def crossed_mean(
    df: pl.DataFrame, *, factors: Sequence[str], value: str = "score", by: Sequence[str] = ()
) -> pl.DataFrame:
    """Mean of a fully crossed design, and the unbiased estimator of its variance, for every group of ``by``.

    In each group, ``df`` must hold exactly one ``value`` for every combination of the ``factors`` (e.g. every
    speaker x model), and the levels of every factor are treated as independent samples of a population (speakers,
    model seeds...). The mean is then a generalized U-statistic, and its variance is estimated without bias by
    ``mean^2 - Q``, ``Q`` being the average product of two values that differ on every factor. It is computed in
    ``O(rows x 2^len(factors))`` by inclusion-exclusion over the factors.

    A factor with a single level in a group (e.g. one model) is treated as fixed: the variance is then conditional
    on that level. The variance is ``null`` when every factor has a single level. Being unbiased, the estimate can be
    slightly negative when the true variance is close to 0.

    With a single factor, this is the usual ``var(ddof=1) / n``.

    :param df: The table, in long format.
    :param factors: Columns of the random factors, crossed with each other.
    :param value: Column of the values to average.
    :param by: Columns defining independent groups, each with its own mean and variance.
    :returns: One row per group: the ``by`` columns, ``estimate``, ``variance``, and ``n_<factor>`` the number of
        levels of each factor.
    """
    factors, by = list(factors), list(by)
    verify_design(df, factors=factors, value=value, by=by)
    key = by or ["__group"]
    data = df.select(*by, *factors, pl.col(value).cast(pl.Float64).alias("__raw")).with_columns(__group=pl.lit(0))
    # The estimator is invariant to a shift of the values: centering each group avoids cancellations.
    data = data.with_columns(__centered=pl.col("__raw") - pl.col("__raw").mean().over(key))
    subsets = [subset for r in range(len(factors) + 1) for subset in itertools.combinations(factors, r)]
    # Sum over the groups of ``subset`` of the squared sum of their values, spread over their rows.
    terms = [
        (pl.col("__centered").sum().over([*key, *subset]).pow(2) / pl.len().over([*key, *subset])).alias(f"__term{i}")
        for i, subset in enumerate(subsets)
    ]
    groups = (
        data.with_columns(terms)
        .group_by(key, maintain_order=True)
        .agg(
            pl.col("__raw").mean().alias("estimate"),
            pl.len().alias("__rows"),
            pl.struct(factors).n_unique().alias("__combinations"),
            *[pl.col(f).n_unique().alias(f"n_{f}") for f in factors],
            *[pl.col(f"__term{i}").sum() for i in range(len(subsets))],
        )
    )
    verify_crossed(groups, factors=factors, by=by)
    # Factors with a single level are fixed: they leave the inclusion-exclusion and the degrees of freedom.
    is_random = {f: pl.col(f"n_{f}") > 1 for f in factors}
    distinct_pairs = pl.sum_horizontal(
        pl.when(pl.all_horizontal(pl.lit(value=True), *[is_random[f] for f in subset]))
        .then((-1) ** len(subset) * pl.col(f"__term{i}"))
        .otherwise(0)
        for i, subset in enumerate(subsets)
    )
    freedom = functools.reduce(
        operator.mul, [pl.when(is_random[f]).then(pl.col(f"n_{f}") - 1).otherwise(1) for f in factors]
    )
    variance = pl.when(pl.any_horizontal(*is_random.values())).then(-distinct_pairs / (pl.col("__rows") * freedom))
    return groups.select(*by, "estimate", variance.alias("variance"), *[f"n_{f}" for f in factors])


def verify_unit(score: Score, unit: str) -> None:
    """Check that the unit is a BY condition of the Score."""
    if unit in score.by:
        return
    if unit == score.on or unit in score.across:
        role = "the ON" if unit == score.on else "an ACROSS"
        msg = (
            f"{unit!r} is {role} condition of the task, but the unit must be a BY condition: with ACROSS, every cell "
            f"involves two {unit}s, so they are not independent units. Use `fastabx.stats.bootstrap_details` instead."
        )
        raise InvalidUnitError(msg)
    msg = f"{unit!r} is not a condition of the task. Its BY conditions are: {score.by}."
    raise InvalidUnitError(msg)


def compare(
    scores_1: Score | Sequence[Score],
    scores_2: Score | Sequence[Score],
    *,
    unit: str = "speaker",
    levels: Sequence[tuple[str, ...] | str] | None = None,
    symmetric: bool = False,
) -> pl.DataFrame:
    """Compare two groups of models: mean difference of their ABX error rates, and its unbiased variance.

    Each Score is first collapsed with ``score.details(levels=levels, symmetric=symmetric)``, which averages the
    ``levels`` within each unit (e.g. the contexts within each speaker). With ``E1[u, m]`` and ``E2[u, m]`` the
    resulting error rates of the models of each group, the estimate is the mean of ``E1[u, a] - E2[u, j]`` over every
    unit ``u``, model ``a`` of the first group and model ``j`` of the second, and the variance is the one of
    :py:func:`crossed_mean` with the factors ``unit``, ``model_1`` and ``model_2``. This is the estimator of Schatz
    et al. (PNAS, Supplementary Materials and Methods 4), with ``symmetric=True`` and the contexts as ``levels``.

    The conditions left after ``levels``, besides the unit, define one comparison per group: for instance, one per
    contrast. Averaging the contrast as well in ``levels`` gives a single comparison.

    With a single model in a group, that factor is treated as fixed: the variance is then conditional on that
    model. For instance, comparing one model to one other gives the variance over units only. Cells with no valid
    triplet (constraints) are left out.

    The test itself is left to SciPy: ``z = estimate / sqrt(variance)`` is asymptotically normal, so
    ``p = 2 * scipy.stats.norm.sf(abs(z))``, and ``scipy.stats.false_discovery_control`` corrects for multiple
    comparisons.

    :param scores_1: The Scores of the first group of models, or a single Score.
    :param scores_2: The Scores of the second group of models, or a single Score. Every Score must share the same
        groups, e.g. by being scored on the same Task.
    :param unit: The BY condition whose values are the independent units, e.g. the speakers.
    :param levels: The conditions to average within each unit, as in :py:meth:`.Score.details`.
    :param symmetric: Whether to merge the two directions of each contrast first, as in :py:meth:`.Score.details`.
    :returns: One row per group: the conditions left after ``levels`` besides the unit, ``estimate`` (group 1 minus
        group 2), ``variance``, and the numbers of units and of models, ``n_<unit>``, ``n_model_1`` and ``n_model_2``.
    """
    sides = [[scores] if isinstance(scores, Score) else list(scores) for scores in (scores_1, scores_2)]
    if not all(sides):
        msg = "Each group needs at least one Score."
        raise MismatchedScoresError(msg)
    tables: list[list[pl.DataFrame]] = [[], []]
    reference = None
    for side, scores in enumerate(sides):
        for model, score in enumerate(scores):
            verify_unit(score, unit)
            details = score.details(levels=levels, symmetric=symmetric)
            if unit not in details.columns:
                msg = f"The unit {unit!r} is averaged over in `levels`: keep it to compare the models unit by unit."
                raise InvalidUnitError(msg)
            details = details.filter(pl.col("score").is_not_null()).drop("size")
            keys = details.drop("score")
            if reference is None:
                reference = keys.sort(keys.columns)
            elif keys.columns != reference.columns or not keys.sort(keys.columns).equals(reference):
                msg = (
                    "The Scores do not have the same groups. They must come from Tasks with the same conditions and "
                    "labels, and be collapsed with the same `levels`."
                )
                raise MismatchedScoresError(msg)
            tables[side].append(details.with_columns(model=pl.lit(model, dtype=pl.Int64)))
    assert reference is not None
    keys = reference.columns
    first = pl.concat(tables[0]).rename({"score": "__score_1", "model": "model_1"})
    second = pl.concat(tables[1]).rename({"score": "__score_2", "model": "model_2"})
    paired = first.join(second, on=keys, nulls_equal=True).with_columns(
        __difference=pl.col("__score_1") - pl.col("__score_2")
    )
    return crossed_mean(
        paired, factors=[unit, "model_1", "model_2"], value="__difference", by=[k for k in keys if k != unit]
    )
