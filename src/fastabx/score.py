"""Score the ABX task for each cell and collapse the scores into a final score."""

import os
from collections.abc import Sequence
from pathlib import Path

import polars as pl
import polars.selectors as cs
from tqdm import tqdm

from fastabx.constraints import Constraints
from fastabx.distance import Distance, DistanceName, distance_function
from fastabx.group import GroupReducer, group_cells
from fastabx.task import Task
from fastabx.utils import MIN_CELLS_FOR_TQDM, prefetch
from fastabx.verify import format_score_levels, verify_score_levels

__all__ = ["Score"]


def pl_weighted_mean(value_col: str, weight_col: str) -> pl.Expr:
    """Generate a Polars aggregation expression to take a weighted mean.

    https://github.com/pola-rs/polars/issues/7499#issuecomment-2569748864
    """
    values = pl.col(value_col)
    weights = pl.when(values.is_not_null()).then(weight_col)
    return weights.dot(values).truediv(weights.sum()).fill_nan(None)


class CollapseError(Exception):
    """Something wrong happened when collapsing the ``Score``."""

    def __init__(self, *, are_set: bool) -> None:
        if are_set:
            msg = "Cannot set `weighted=True` and `levels` at the same time."
        else:
            msg = "Either set `levels` or `weighted=True`."
        super().__init__(msg)


class IncompatibleNormalizationError(Exception):
    """The dataset was already L2-normalized for a previous angular score and cannot be reused for this distance."""

    def __init__(self, distance_name: str) -> None:
        super().__init__(
            f"The dataset has been L2-normalized (with a singularity border) by a previous cosine/angular "
            f"Score and its features are no longer in their original space. Computing {distance_name!r} on it "
            f"would be silently wrong. Build a fresh Dataset/Task for this distance."
        )


def score_details(cells: pl.DataFrame, *, levels: Sequence[tuple[str, ...] | str] | None) -> pl.DataFrame:
    """Collapse the scored cells and return the final scores and sizes for each (A, B) pairs."""
    if levels is None:
        if len(set(cells.columns) - {"index_a", "index_b", "index_x", "score", "size"}) != 2:
            raise CollapseError(are_set=False)
        levels = []
    cells = cells.select(~(cs.starts_with("index") | cs.ends_with("_x")))
    levels_in_tuples = format_score_levels(levels)
    verify_score_levels(cells.columns, levels_in_tuples)
    for level in levels_in_tuples:
        group_key = cs.exclude("score", "size", *level)
        cells = cells.group_by(group_key, maintain_order=True).agg(pl.col("score").mean(), pl.col("size").sum())
    return cells


def score_task(
    task: Task,
    distance: Distance,
    constraints: Constraints | None = None,
) -> tuple[list[float | None], list[int | None]]:
    """Score each cell of a :py:class:`.Task` using a given distance, and return scores and sizes.

    With ``constraints``, the per-triplet mask is carried through the same grouped engine; cells left with no valid
    triplet get a ``None`` score and size.
    """
    reducer = GroupReducer(len(task), constrained=constraints is not None)
    disable_tqdm = len(task) < MIN_CELLS_FOR_TQDM or os.getenv("TQDM_DISABLE")
    pbar = tqdm(total=len(task), desc="Scoring each cell", disable=bool(disable_tqdm))
    for group in prefetch(group_cells(task, constraints)):
        reducer.add(group, distance, is_symmetric=task.is_symmetric)
        pbar.update(len(group.positions))
    pbar.close()
    return reducer.finalize()


class Score:
    """Compute the score of a :py:class:`.Task` using a given distance specified by ``distance_name``.

    Additional :py:type:`.Constraints` can be provided to restrict the possible triplets in each cell.

    The full scoring runs eagerly in ``__init__``: constructing a ``Score`` is the expensive step,
    and ``collapse``/``details`` afterwards are cheap.

    :param task: The :py:class:`.Task` to score.
    :param distance_name: Name of the distance, "angular" (same as "cosine"), "euclidean", "kl_symmetric"
        or "identical". Defaults to "angular".
    :param constraints: Optional constraints to restrict the possible triplets.
    """

    def __init__(self, task: Task, distance_name: DistanceName, *, constraints: Constraints | None = None) -> None:
        self.distance_name = distance_name
        distance = distance_function(distance_name)
        if distance_name in {"cosine", "angular"}:
            task.dataset.normalize_()
        elif task.dataset.accessor.is_normalized:
            raise IncompatibleNormalizationError(distance_name)
        scores, sizes = score_task(task, distance, constraints)
        self._cells = task.cells.select(cs.exclude("description", "header")).with_columns(
            score=pl.Series(scores, dtype=pl.Float32), size=pl.Series(sizes, dtype=pl.Int32)
        )

    @property
    def cells(self) -> pl.DataFrame:
        """Return the scored cells."""
        return self._cells

    def __repr__(self) -> str:
        return f"Score({len(self.cells)} cells, {self.distance_name} distance)"

    def write_csv(self, file: str | Path) -> None:
        """Write the results of all the cells to a CSV file.

        Nested list columns (the per-cell ``index_a``/``index_b``/``index_x``) are dropped, since
        CSV cannot represent them. Use ``self.cells`` directly to keep them.

        :param file: Path to the output CSV file.
        """
        nested = [name for name, dtype in self.cells.schema.items() if dtype == pl.List]
        (self.cells.select(cs.exclude(nested)) if nested else self.cells).write_csv(file)

    def details(self, *, levels: Sequence[tuple[str, ...] | str] | None = None) -> pl.DataFrame:
        """Collapse the scored cells and return the final scores and sizes for each (A, B) pairs.

        :param levels: List of levels to collapse. The order matters a lot.
        """
        return score_details(self.cells, levels=levels)

    def collapse(self, *, levels: Sequence[tuple[str, ...] | str] | None = None, weighted: bool = False) -> float:
        """Collapse the scored cells into the final score.

        Use either `levels` or `weighted=True` to collapse the scores.

        :param levels: List of levels to collapse. The order matters a lot.
        :param weighted: Whether to collapse the scores using a mean weighted by the size of the cells.
        """
        if weighted:
            if levels is not None:
                raise CollapseError(are_set=True)
            return self.cells.select(pl_weighted_mean("score", "size")).item()
        return self.details(levels=levels)["score"].mean()  # ty:ignore[invalid-return-type]
