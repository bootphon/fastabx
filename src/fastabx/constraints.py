"""Additional constraints for building cells."""

import functools
import operator
from collections.abc import Iterable

import polars as pl
import polars.selectors as cs

__all__ = ["Constraints", "constraints_all_different"]

type Constraints = Iterable[pl.Expr]


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
        super().__init__("No valid column provided in the constraints or a mask is missing for constrained scoring.")


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
