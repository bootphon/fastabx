"""Score the ABX task for each cell and collapse the scores into a final score."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl
import polars.selectors as cs
from tqdm import tqdm

from fastabx.alignment import Alignment, AlignmentName, alignment_function
from fastabx.cell import INDEX_COLUMNS
from fastabx.bootstrap import Bootstrap, BootstrapReducer
from fastabx.constraints import Constraints
from fastabx.distance import Distance, DistanceName, distance_function
from fastabx.group import GroupReducer, group_cells, group_distance_matrix
from fastabx.task import Task
from fastabx.utils import display_name, hide_progress, prefetch
from fastabx.verify import format_score_levels, verify_score_levels

__all__ = ["CollapseError", "EmptyScoreError", "IncompatibleNormalizationError", "Score"]


def pl_weighted_mean(value_col: str, weight_col: str) -> pl.Expr:
    """Generate a Polars aggregation expression to take a weighted mean.

    https://github.com/pola-rs/polars/issues/7499#issuecomment-2569748864
    """
    values = pl.col(value_col)
    weights = pl.when(values.is_not_null()).then(weight_col)
    return weights.dot(values).truediv(weights.sum()).fill_nan(None)


class CollapseError(Exception):
    """Something wrong happened when collapsing the ``Score``."""

    def __init__(self, *, are_set: bool, conditions: Sequence[str] = ()) -> None:
        if are_set:
            msg = "Cannot set `weighted=True` and `levels` at the same time."
        else:
            msg = (
                f"Either set `levels` or `weighted=True`. Collapsing without `levels` only works when the cells "
                f"have exactly two condition columns left (the ON condition of A and of B), but these have "
                f"{len(conditions)}: {list(conditions)}. Use `levels` to choose in which order to average those "
                f"conditions, or `weighted=True` to average every cell weighted by its size."
            )
        super().__init__(msg)


class EmptyScoreError(Exception):
    """Every cell has a null score, so there is nothing left to average."""

    def __init__(self) -> None:
        super().__init__(
            "Every cell has a null score, so collapsing them would average nothing. This only happens with "
            "`constraints`: not a single cell kept a valid triplet. Loosen the constraints, or check that "
            "they name the labels you meant (with the `_a`, `_b` and `_x` suffixes)."
        )


class IncompatibleNormalizationError(Exception):
    """The dataset was already L2-normalized for a previous angular score and cannot be reused for this distance."""

    def __init__(self, distance_name: str) -> None:
        super().__init__(
            f"The dataset has been L2-normalized (with a singularity border) by a previous cosine/angular "
            f"Score and its features are no longer in their original space. Computing {distance_name!r} on it "
            f"would be silently wrong. Build a fresh Dataset/Task for this distance."
        )


class BootstrapConstraintsError(Exception):
    """Bootstrap and constraints cannot be combined yet."""

    def __init__(self) -> None:
        super().__init__(
            "`bootstrap` and `constraints` cannot be used together: the per-triplet constraint mask is indexed "
            "by position in the cell, and the bootstrap reweights the items behind those positions."
        )


class NoBootstrapError(Exception):
    """The ``Score`` was built without a ``Bootstrap``, so there are no replicates to read."""

    def __init__(self) -> None:
        super().__init__("This Score has no bootstrap replicates. Build it with `Score(..., bootstrap=Bootstrap(n))`.")


def score_details(cells: pl.DataFrame, *, levels: Sequence[tuple[str, ...] | str] | None) -> pl.DataFrame:
    """Collapse the scored cells and return the final scores and sizes for each (A, B) pairs."""
    if levels is None:
        conditions = [c for c in cells.columns if c not in {"index_a", "index_b", "index_x", "score", "size"}]
        if len(conditions) != 2:
            raise CollapseError(are_set=False, conditions=conditions)
        levels = []
    cells = cells.select(~(INDEX_COLUMNS | cs.ends_with("_x")))
    levels_in_tuples = format_score_levels(levels)
    verify_score_levels(cells.columns, levels_in_tuples)
    for level in levels_in_tuples:
        group_key = cs.exclude("score", "size", *level)
        cells = cells.group_by(group_key, maintain_order=True).agg(pl.col("score").mean(), pl.col("size").sum())
    return cells


def score_task(
    task: Task,
    distance: Distance,
    *,
    alignment: Alignment,
    constraints: Constraints | None = None,
    progress: bool = True,
    bootstrap: BootstrapReducer | None = None,
) -> tuple[list[float | None], list[int | None]]:
    """Score each cell of a :py:class:`.Task` using a given distance and alignment, and return scores and sizes.

    With ``constraints``, the per-triplet mask is carried through the same grouped engine; cells left with no valid
    triplet get a ``None`` score and size.

    :param bootstrap: An optional :py:class:`.BootstrapReducer`, filled **in place** with the per-replicate
        scores. It reuses each group's distance matrix, so the replicates cost no extra distance computation.
    """
    reducer = GroupReducer(len(task), constrained=constraints is not None)
    pbar = tqdm(total=len(task), desc="Scoring each cell", disable=hide_progress(progress=progress))
    for group in prefetch(group_cells(task, constraints=constraints)):
        distances = group_distance_matrix(group, distance)
        reducer.add(group, distance, alignment=alignment, is_symmetric=task.is_symmetric, distances=distances)
        if bootstrap is not None:
            bootstrap.add(group, distances, is_symmetric=task.is_symmetric)
        pbar.update(len(group.positions))
    pbar.close()
    return reducer.finalize()


class Score:
    """Compute the score of a :py:class:`.Task` using a given distance specified by ``distance_name``.

    All the scores reported by this class are ABX error rates (1 - discriminability).
    Lower is better, and chance level is 0.5.

    Additional :py:class:`.Constraints` can be provided to restrict the possible triplets in each cell.

    The full scoring runs eagerly in ``__init__``: constructing a ``Score`` is the expensive step,
    and ``collapse``/``details`` afterwards are cheap.

    .. warning::
        Constructing a ``Score`` with the ``"cosine"``/``"angular"`` distance **mutates the
        shared** ``task.dataset`` **in place**: it L2-normalizes the features and appends the
        singularity-border column, so the dataset's feature dimension grows by one and
        ``task.dataset.accessor.is_normalized`` becomes ``True``.
        If you need the original features back, keep a separate, un-normalized ``Dataset``.

    :param task: The :py:class:`.Task` to score.
    :param distance_name: The distance to use, either the name of a built-in one ("euclidean", "cosine",
        "angular", "kl_symmetric", "identical") or a custom :py:class:`.Distance` callable.
    :param alignment: How to reduce the frame-level cost lattice to one distance per pair of sequences,
        either the name of a built-in alignment ("dtw") or a custom :py:class:`.Alignment`.
        Defaults to "dtw". Bypassed entirely when the dataset is pooled, since there is nothing to align.
    :param constraints: Optional constraints to restrict the possible triplets.
    :param progress: Whether to display a progress bar while scoring the cells.
    :param bootstrap: Optional :py:class:`.Bootstrap` scheme. When given, every cell is additionally scored on
        ``bootstrap.n_replicates`` with-replacement resamples of its items, reusing the same distance
        computations as the point estimate. Read the replicates back with :py:meth:`bootstrap_collapse` or
        :py:meth:`confidence_interval`.
    """

    def __init__(
        self,
        task: Task,
        distance_name: DistanceName | Distance,
        *,
        alignment: AlignmentName | Alignment = "dtw",
        constraints: Constraints | None = None,
        progress: bool = True,
        bootstrap: Bootstrap | None = None,
    ) -> None:
        self.distance_name = distance_name
        self.alignment = alignment
        distance = distance_function(distance_name)
        align = alignment_function(alignment)
        if distance_name in {"cosine", "angular"}:
            task.dataset.normalize_()
        elif task.dataset.accessor.is_normalized:
            raise IncompatibleNormalizationError(display_name(distance_name))
        if bootstrap is not None and constraints is not None:
            raise BootstrapConstraintsError
        reducer = BootstrapReducer(len(task), bootstrap) if bootstrap is not None else None
        scores, sizes = score_task(
            task, distance, alignment=align, constraints=constraints, progress=progress, reducer=reducer
        )
        self._cells = task.cells.select(cs.exclude("description", "header")).with_columns(
            score=pl.Series(scores, dtype=pl.Float32), size=pl.Series(sizes, dtype=pl.Int32)
        )
        self._replicates = reducer.finalize() if reducer is not None else None

    @property
    def cells(self) -> pl.DataFrame:
        """Scored cells.

        The ``score`` column is the ABX error rate of each cell, and ``size`` its number of triplets.
        """
        return self._cells

    @property
    def n_replicates(self) -> int:
        """Number of bootstrap replicates, 0 when the ``Score`` was built without a :py:class:`.Bootstrap`."""
        return 0 if self._replicates is None else self._replicates[0].shape[1]

    def __repr__(self) -> str:
        distance, align = display_name(self.distance_name), display_name(self.alignment)
        bootstrap = f", {self.n_replicates} bootstrap replicates" if self._replicates is not None else ""
        return f"Score({len(self.cells)} cells, {distance} distance{bootstrap}, {align} alignment)"

    def write_csv(self, file: str | Path) -> None:
        """Write the results of all the cells to a CSV file.

        Nested list columns (the per-cell ``index_a``/``index_b``/``index_x``) are dropped, since
        CSV cannot represent them. Use ``self.cells`` directly to keep them.

        :param file: Path to the output CSV file.
        """
        nested = [name for name, dtype in self.cells.schema.items() if dtype == pl.List]
        (self.cells.select(cs.exclude(nested)) if nested else self.cells).write_csv(file)

    def details(self, *, levels: Sequence[tuple[str, ...] | str] | None = None) -> pl.DataFrame:
        """Collapse the scored cells and return the final ABX error rates and sizes for each (A, B) pairs.

        :param levels: List of levels to collapse. The order matters a lot.
        """
        return score_details(self.cells, levels=levels)

    def collapse(self, *, levels: Sequence[tuple[str, ...] | str] | None = None, weighted: bool = False) -> float:
        """Collapse the scored cells into the final ABX error rate.

        Use either `levels` or `weighted=True` to collapse the scores.

        :param levels: List of levels to collapse. The order matters a lot.
        :param weighted: Whether to collapse the scores using a mean weighted by the size of the cells.
        :returns: The overall ABX error rate, between 0 and 1.
        """
        if weighted:
            if levels is not None:
                raise CollapseError(are_set=True)
            collapsed = self.cells.select(pl_weighted_mean("score", "size")).item()
        else:
            collapsed = self.details(levels=levels)["score"].mean()
        if collapsed is None:
            raise EmptyScoreError
        return float(collapsed)  # ty: ignore[invalid-argument-type]

    def bootstrap_collapse(
        self,
        *,
        levels: Sequence[tuple[str, ...] | str] | None = None,
        weighted: bool = False,
    ) -> npt.NDArray[np.float64]:
        """Collapse each bootstrap replicate into a score, giving the sampling distribution of :py:meth:`collapse`.

        Each replicate is collapsed with exactly the same logic as the point estimate, so the returned
        distribution is directly comparable to ``self.collapse(levels=..., weighted=...)``. Cells whose
        replicate is degenerate (no valid ``x != a`` pair left after resampling) are ``null`` and skipped
        by the collapse, in the same way constrained cells with no valid triplet are.

        :param levels: List of levels to collapse. The order matters a lot.
        :param weighted: Whether to collapse using a mean weighted by the size of the cells.
        :returns: A ``(n_replicates,)`` array of collapsed scores.
        """
        if self._replicates is None:
            raise NoBootstrapError
        scores, sizes = self._replicates
        base = self.cells.select(cs.exclude("score", "size"))
        collapsed = np.empty(scores.shape[1], dtype=np.float64)
        for replicate in range(scores.shape[1]):
            cells = base.with_columns(
                score=pl.Series(scores[:, replicate], dtype=pl.Float64, nan_to_null=True),
                size=pl.Series(sizes[:, replicate], dtype=pl.Float64, nan_to_null=True),
            )
            collapsed[replicate] = collapse_cells(cells, levels=levels, weighted=weighted)
        return collapsed

    def confidence_interval(
        self,
        *,
        levels: Sequence[tuple[str, ...] | str] | None = None,
        weighted: bool = False,
        alpha: float = 0.05,
    ) -> tuple[float, float]:
        """Percentile bootstrap confidence interval of the collapsed score.

        :param levels: List of levels to collapse. The order matters a lot.
        :param weighted: Whether to collapse using a mean weighted by the size of the cells.
        :param alpha: Two-sided miscoverage, default 0.05 for a 95% interval.
        :returns: The ``(lower, upper)`` bounds.
        """
        collapsed = self.bootstrap_collapse(levels=levels, weighted=weighted)
        low, high = np.nanpercentile(collapsed, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return float(low), float(high)
