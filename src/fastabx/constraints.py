"""Additional constraints for building cells."""

import functools
import operator
from collections.abc import Iterable

import polars as pl

__all__ = ["Constraints", "constraints_all_different"]

type Constraints = Iterable[pl.Expr]


def constraints_all_different(*columns: str) -> Constraints:
    """Return :py:class:`.Constraints` that ensure that each specified column has different values for A, B and X.

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
        super().__init__("No valid column provided in the constraints or a mask is missing for constrained scoring.")


def apply_constraints(
    cells: pl.DataFrame,
    labels: pl.DataFrame,
    constraints: Constraints,
    *,
    is_symmetric: bool,
) -> pl.DataFrame:
    """Apply constraints to the cells DataFrame.

    .. note::
        The per-cell ``is_valid`` lists are rebuilt by exploding to one row per triplet, evaluating the
        constraints there, and regrouping. Two things keep that affordable on a large task. Only the three
        index columns are exploded, each carrying a synthetic ``__cell`` row number, so the condition
        columns are not replicated once per triplet; and the query runs on the streaming engine, which
        chunks the group-by instead of materialising every triplet at once.

        Regrouping on ``__cell`` rather than on the condition columns also means the ``is_valid`` lists
        line up with ``cells`` by construction, whatever those columns contain. Constraints therefore work
        with a :py:class:`.Task` built by :py:meth:`.Task.from_cells` too.

        The explode order is ``index_x``, then ``index_a``, then ``index_b``, so that each cell's flat list
        reshapes to the ``(nx, na, nb)`` mask that :py:func:`fastabx.group.group_cells` expects. That order
        has to be pinned explicitly: the joins and the group-by do **not** preserve row order on the
        streaming engine, and an unordered mask is silently wrong rather than an error. ``__triplet`` is
        stamped right after the explodes and the aggregation sorts by it, which restores the order the
        reshape depends on.
    """
    constraints = list(constraints)
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
    is_valid = (
        cells.lazy()
        .select("index_a", "index_b", "index_x")
        .with_row_index("__cell")
        .explode("index_x")
        .explode("index_a")
        .explode("index_b")
        .with_row_index("__triplet")
        .join(labels_lazy.rename({c: f"{c}_x" for c in (columns_to_retrieve | {"index"})}), on="index_x")
        .join(labels_lazy.rename({c: f"{c}_a" for c in (columns_to_retrieve | {"index"})}), on="index_a")
        .join(labels_lazy.rename({c: f"{c}_b" for c in (columns_to_retrieve | {"index"})}), on="index_b")
        .select("__cell", "__triplet", is_valid=functools.reduce(operator.and_, constraints))
        .group_by("__cell", maintain_order=True)
        .agg(pl.col("is_valid").sort_by("__triplet"))
        .sort("__cell")
        .select("is_valid")
        .collect(engine="streaming")
    )
    return pl.concat((cells, is_valid), how="horizontal")
