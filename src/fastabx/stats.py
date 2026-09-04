"""Statistics on scored ABX tasks: bootstrap distributions over tokens and categories.

Two kinds of resampling are supported, and can be combined:

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

import hashlib
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl
import polars.selectors as cs
import torch

from fastabx.cell import INDEX_COLUMNS
from fastabx.score import CollapseError, NoBootstrapError, Score, collapse_levels
from fastabx.utils import bootstrap_chunk_elems
from fastabx.verify import verify_bootstrap_params

__all__ = [
    "InvalidResampleError",
    "NestedBootstrapWarning",
    "bootstrap_collapse",
    "percentile_interval",
]

CONDITION_SUFFIXES = ("", "_b", "_x")


class InvalidResampleError(ValueError):
    """The requested bootstrap cannot be built from this ``Score``."""

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
    """

    ids: torch.Tensor
    n_groups: int
    weights: list[tuple[str, npt.NDArray[np.int64]]]


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


def collapse_plan(
    cells: pl.DataFrame,
    levels: Sequence[tuple[str, ...] | str] | None,
    factors: dict[str, Factor],
    columns: dict[str, str],
) -> list[Step]:
    """Turn the ``levels`` of a collapse into a list of weighted aggregations, computed once for all replicates.

    Mirrors :py:func:`fastabx.score.score_details` followed by a mean: each level averages over its columns,
    grouping by every other remaining column, and the final mean averages over everything left. The ``_x`` labels,
    dropped by the collapse, are averaged over in the first aggregation.
    """
    frame, levels_in_tuples = collapse_levels(cells, levels)
    x_columns = cells.select(cs.ends_with("_x") - INDEX_COLUMNS).columns
    pending = step_weights(cells, x_columns, factors, columns)
    steps, n_rows = [], len(frame)
    for level in [*levels_in_tuples, None]:
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
    return steps


def row_weights(
    weights: list[tuple[str, npt.NDArray[np.int64]]], factors: dict[str, Factor], replicates: slice, n_rows: int
) -> torch.Tensor:
    """``(n_rows, len(replicates))`` product of the multiplicities of each row's labels."""
    out = torch.ones((n_rows, replicates.stop - replicates.start), dtype=torch.float64)
    for name, codes in weights:
        out *= torch.from_numpy(factors[name].weights[replicates].T[codes].astype(np.float64))
    return out


def weighted_mean(scores: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Per-group weighted mean of ``scores`` (``nan`` are skipped), ``nan`` for groups with no weight."""
    valid = ~torch.isnan(scores)
    weights = torch.where(valid, weights, 0)
    numerator = scores.new_zeros((n_groups, scores.size(1))).index_add_(0, ids, weights * scores.nan_to_num())
    denominator = scores.new_zeros((n_groups, scores.size(1))).index_add_(0, ids, weights)
    return torch.where(denominator > 0, numerator / denominator, torch.nan)


def replicate_source(
    score: Score, *, resample: Sequence[str], tokens: bool | None, n_replicates: int | None
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], int]:
    """Per-cell scores and sizes to collapse, either the token replicates or the point estimate, and their count."""
    use_tokens = score.n_replicates > 0 if tokens is None else tokens
    if use_tokens:
        scores, sizes = score.replicates
        if n_replicates is not None and n_replicates != score.n_replicates:
            msg = (
                f"`n_replicates={n_replicates}` does not match the {score.n_replicates} token replicates of this "
                f"Score. Omit it, or pass `tokens=False` to resample categories only."
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
        return scores, sizes, score.n_replicates
    if tokens is None and not resample:
        raise NoBootstrapError
    if not resample:
        msg = "Nothing to resample: `tokens=False` and `resample` is empty."
        raise InvalidResampleError(msg)
    if n_replicates is None:
        msg = "`n_replicates` is required when resampling categories without token replicates."
        raise InvalidResampleError(msg)
    scores = score.cells["score"].to_numpy(allow_copy=True).astype(np.float64)[:, None]
    sizes = score.cells["size"].cast(pl.Float64).to_numpy().astype(np.float64)[:, None]
    return scores, sizes, n_replicates


def bootstrap_collapse(
    score: Score,
    *,
    levels: Sequence[tuple[str, ...] | str] | None = None,
    weighted: bool = False,
    resample: Sequence[str] = (),
    tokens: bool | None = None,
    n_replicates: int | None = None,
    seed: int = 0,
) -> npt.NDArray[np.float64]:
    """Bootstrap distribution of :py:meth:`.Score.collapse`, resampling tokens, categories, or both.

    Every replicate is collapsed with the same ``levels``/``weighted`` logic as the point estimate, so the returned
    distribution is directly comparable to ``score.collapse(levels=..., weighted=...)``.

    - **tokens** are the items inside each cell. Their replicates come from ``Score(..., bootstrap=Bootstrap(n))``
      and measure how the score would move with other recordings of the *same* speakers, contexts and phones.
    - **categories** are the values of the conditions listed in ``resample`` (e.g. ``["speaker"]``), drawn with
      replacement. They measure how the score would move with *other* speakers. They need no rescoring.

    Resampling both counts the token variability twice, since it is already part of the spread between categories,
    and emits a :py:class:`NestedBootstrapWarning`. Resampling two crossed conditions (e.g. speakers and contexts)
    overestimates the variance as well. For a mean over units, prefer an analytical variance.

    :param score: The scored task.
    :param levels: List of levels to collapse, as in :py:meth:`.Score.collapse`.
    :param weighted: Whether to collapse with a mean weighted by the size of the cells, as in
        :py:meth:`.Score.collapse`.
    :param resample: Conditions whose values are resampled. The ``_b`` and ``_x`` columns of a condition share its
        draw.
    :param tokens: Whether to use the token replicates of ``score``. By default, they are used if the ``Score`` has
        some.
    :param n_replicates: Number of replicates. Defaults to the number of token replicates, and is required when only
        categories are resampled.
    :param seed: Seed of the category draws. The token draws are seeded by the :py:class:`.Bootstrap`.
    :returns: A ``(n_replicates,)`` array of collapsed ABX error rates, ``nan`` for a replicate left with no valid
        cell.
    """
    if weighted and levels is not None:
        raise CollapseError(are_set=True)
    scores, sizes, n_replicates = replicate_source(score, resample=resample, tokens=tokens, n_replicates=n_replicates)
    verify_bootstrap_params(n_replicates, seed=seed)
    cells = score.cells.select(cs.exclude("score", "size"))
    factors, columns = resolve_factors(cells, resample, n_replicates=n_replicates, seed=seed)
    if weighted:
        ids = torch.zeros(len(cells), dtype=torch.int64)
        steps = [Step(ids, 1, step_weights(cells, list(columns), factors, columns))]
    else:
        steps = collapse_plan(cells, levels, factors, columns)

    out = np.empty(n_replicates, dtype=np.float64)
    width = max(1, bootstrap_chunk_elems() // max(1, len(cells)))
    for start in range(0, n_replicates, width):
        chunk = slice(start, min(start + width, n_replicates))
        count = chunk.stop - chunk.start
        current = torch.from_numpy(scores[:, chunk] if scores.shape[1] > 1 else scores).expand(-1, count)
        for i, step in enumerate(steps):
            weights = row_weights(step.weights, factors, chunk, len(current))
            if weighted and i == 0:
                cell_sizes = torch.from_numpy(sizes[:, chunk] if sizes.shape[1] > 1 else sizes).expand(-1, count)
                weights *= cell_sizes.nan_to_num()
            current = weighted_mean(current, weights, step.ids, step.n_groups)
        out[chunk] = current[0].numpy()
    return out


def percentile_interval(replicates: npt.ArrayLike, *, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap confidence interval, ignoring ``nan`` replicates.

    :param replicates: Bootstrap distribution, e.g. from :py:func:`bootstrap_collapse`.
    :param alpha: Two-sided miscoverage, default 0.05 for a 95% interval.
    :returns: The ``(lower, upper)`` bounds.
    """
    if not 0 < alpha < 1:
        msg = f"`alpha` must be in (0, 1), got {alpha}."
        raise ValueError(msg)
    low, high = np.nanpercentile(np.asarray(replicates, dtype=np.float64), [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)
