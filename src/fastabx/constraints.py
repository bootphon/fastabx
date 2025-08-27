"""Additional constraints for building cells."""

import functools
import operator
from collections.abc import Generator, Iterable

import polars as pl
import polars.selectors as cs
import torch
from polars._typing import IntoExprColumn
from tqdm import tqdm

from fastabx.cell import Cell
from fastabx.distance import Distance, abx_on_cell
from fastabx.task import Task
from fastabx.utils import MIN_CELLS_FOR_TQDM

type Constraints = Iterable[IntoExprColumn]


def constraints_all_different(*columns: str) -> Constraints:
    """Return :py:type:`.Constraints` that ensure that each specified column has different values for A, B and X.

    :param columns: The columns to apply the constraints on.
    """
    return [
        pl.col(f"{c}_a").ne(pl.col(f"{c}_x"))
        & pl.col(f"{c}_a").ne(pl.col(f"{c}_b"))
        & pl.col(f"{c}_x").ne(pl.col(f"{c}_b"))
        for c in columns
    ]


class NoConstraintsError(ValueError):
    """Invalid constraints."""

    def __init__(self) -> None:
        super().__init__("No valid column provided in the constraints")


def apply_constraints(
    cells: pl.DataFrame,
    labels: pl.DataFrame,
    constraints: Constraints,
    *,
    is_symmetric: bool,
) -> pl.DataFrame:
    """Apply constraints to the cells DataFrame."""
    columns_to_retrieve = {
        name.removesuffix("_x").removesuffix("_a").removesuffix("_b")
        for constraint in constraints
        for name in constraint.meta.root_names()
    }
    if not columns_to_retrieve or not columns_to_retrieve.issubset(labels.columns):
        raise NoConstraintsError
    if is_symmetric:
        constraints = [*constraints, pl.col("index_a") != pl.col("index_x")]
    labels_lazy = labels.lazy().select(*columns_to_retrieve).with_row_index()
    cells_lazy = cells.lazy()
    is_valid = (
        cells_lazy.explode("index_x")
        .explode("index_a")
        .explode("index_b")
        .join(labels_lazy.rename({c: f"{c}_x" for c in (columns_to_retrieve | {"index"})}), on="index_x")
        .join(labels_lazy.rename({c: f"{c}_a" for c in (columns_to_retrieve | {"index"})}), on="index_a")
        .join(labels_lazy.rename({c: f"{c}_b" for c in (columns_to_retrieve | {"index"})}), on="index_b")
        .with_columns(is_valid=functools.reduce(operator.and_, constraints))
        .select(cs.exclude([f"{c}_{s}" for c in columns_to_retrieve for s in ("a", "b", "x")]))
        .group_by(cs.exclude(cs.starts_with("index_") | pl.col("is_valid")), maintain_order=True)
        .agg("is_valid")
        .select("is_valid")
    )
    return pl.concat((cells_lazy, is_valid), how="horizontal").collect()


def constrained_cell_generator(
    task: Task, constraints: Constraints
) -> Generator[tuple[Cell, torch.Tensor], None, None]:
    """Generate cells with constraints applied, yielding (Cell, mask) tuples."""
    is_symmetric, device = not bool(task.across), task.dataset.accessor.device
    cells = apply_constraints(task.cells, task.dataset.labels, constraints, is_symmetric=is_symmetric)
    columns = ["header", "description", "index_a", "index_b", "index_x", "is_valid"]
    for header, description, index_a, index_b, index_x, is_valid in cells[columns].iter_rows():
        a = task.dataset.accessor.batched(index_a)
        b = task.dataset.accessor.batched(index_b)
        x = task.dataset.accessor.batched(index_x)
        mask = torch.tensor(is_valid, device=device).view((len(x.sizes), len(a.sizes), len(b.sizes)))
        yield Cell(a=a, b=b, x=x, header=header, description=description, is_symmetric=is_symmetric), mask


def score_task_with_constraints(
    task: Task, distance: Distance, constraints: Constraints
) -> tuple[list[float], list[int]]:
    """Score each cell of a :py:class:`.Task` with additional constraints."""
    scores, sizes = [], []
    for cell, mask in tqdm(
        constrained_cell_generator(task, constraints),
        "Scoring each cell with constraints",
        total=len(task),
        disable=len(task) < MIN_CELLS_FOR_TQDM,
    ):
        if not mask.any():
            scores.append(None)
            sizes.append(None)
        else:
            scores.append(abx_on_cell(cell, distance, mask=mask).item())
            sizes.append(mask.sum())
    return scores, sizes
