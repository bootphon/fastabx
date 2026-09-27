"""Additional constraints for building cells."""

import functools
import operator
from collections.abc import Iterable

import numpy as np
import polars as pl
import torch

__all__ = ["Constraints", "NoConstraintsError", "constraints_all_different"]

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

    def __init__(self, msg: str | None = None) -> None:
        super().__init__(
            msg or "No valid column provided in the constraints or a mask is missing for constrained scoring."
        )


CONSTRAINT_SUFFIXES = ("_a", "_b", "_x")


def constrained_columns(constraints: list[pl.Expr], columns: list[str]) -> set[str]:
    """Label columns referenced by the constraints, each named with exactly one ``_a``, ``_b`` or ``_x`` suffix.

    Only that one suffix is stripped, so a label that itself ends like a suffix (``mic_b``, referenced as
    ``mic_b_a``) keeps its name.
    """
    retrieved = set()
    for name in {name for constraint in constraints for name in constraint.meta.root_names()}:
        if not name.endswith(CONSTRAINT_SUFFIXES):
            msg = (
                f"Constraint column {name!r} has no '_a', '_b' or '_x' suffix: constraints must say to which of "
                f"A, B or X each label belongs, e.g. pl.col('{name}_a') != pl.col('{name}_x')."
            )
            raise NoConstraintsError(msg)
        retrieved.add(name[:-2])
    if unknown := sorted(retrieved - set(columns)):
        msg = f"Constraints reference label(s) {unknown}, which are not columns of `Dataset.labels`."
        raise NoConstraintsError(msg)
    return retrieved


# Largest number of distinct (x, a, b) combinations of the constrained labels whose validity is tabulated once for
# the whole task. Beyond that, each block of triplets evaluates the constraints on the combinations it contains.
MAX_CONSTRAINT_TABLE = 2**22


class ConstraintMask:
    """Evaluate :py:class:`.Constraints` on blocks of triplets, without materialising the labels of each triplet.

    A constraint only depends on the values of the labels it names. Every datapoint gets the code of its
    combination of those values, and the constraints are evaluated once per distinct ``(x, a, b)`` combination of
    codes, in a table. The validity of a block of triplets is then a lookup in that table, on the scoring device.
    When there are too many combinations for one table, each block evaluates the combinations it contains.

    Constraints must therefore be row-wise expressions, relating the labels of A, B and X of a single triplet.
    A triplet whose constraints evaluate to null (a null label in a constrained column) is not valid. In a
    symmetric task, X and A must also be different datapoints.

    :param labels: The ``Dataset.labels``.
    :param constraints: The constraints, whose columns are labels suffixed by ``_a``, ``_b`` or ``_x``.
    :param is_symmetric: Whether X and A are the same set, in which case a datapoint is never both X and A.
    :param device: Where the triplet masks are built.
    """

    def __init__(
        self, labels: pl.DataFrame, constraints: Constraints, *, is_symmetric: bool, device: torch.device
    ) -> None:
        constraints = list(constraints)
        columns = sorted(constrained_columns(constraints, labels.columns))
        if not columns:
            raise NoConstraintsError
        self.columns = columns
        self.expr = functools.reduce(operator.and_, constraints).fill_null(value=False)
        self.is_symmetric = is_symmetric
        self.device = device
        selected = labels.select(columns)
        self.unique = selected.unique(maintain_order=True)
        codes = selected.join(
            self.unique.with_row_index("__code"), on=columns, how="left", nulls_equal=True, maintain_order="left"
        )["__code"].to_numpy()
        self.codes_host = codes.astype(np.int64)
        self.codes = torch.from_numpy(self.codes_host).to(device)
        num_unique = len(self.unique)
        self.table = None
        if num_unique**3 <= MAX_CONSTRAINT_TABLE:
            every = np.arange(num_unique)
            self.table = torch.from_numpy(self.evaluate(every, every, every)).to(device)

    def evaluate(self, codes_x: np.ndarray, codes_a: np.ndarray, codes_b: np.ndarray) -> np.ndarray:
        """Evaluate the constraints on every combination of the given codes, as a ``(x, a, b)`` boolean array."""
        nx, na, nb = len(codes_x), len(codes_a), len(codes_b)
        gathered = {
            "x": np.repeat(codes_x, na * nb),
            "a": np.tile(np.repeat(codes_a, nb), nx),
            "b": np.tile(codes_b, nx * na),
        }
        frame = pl.DataFrame(
            {f"{c}_{suffix}": self.unique[c].gather(g) for c in self.columns for suffix, g in gathered.items()}
        )
        valid = frame.select(self.expr).to_series().to_numpy()
        return np.asarray(valid, dtype=bool).reshape(nx, na, nb)

    def block(self, index_x: torch.Tensor, index_a: torch.Tensor, index_b: torch.Tensor) -> torch.Tensor:
        """Return the ``(nx, na, nb)`` validity of every triplet between the given datapoints, on ``device``."""
        if self.table is not None:
            valid = self.table[
                self.codes[index_x][:, None, None],
                self.codes[index_a][None, :, None],
                self.codes[index_b][None, None, :],
            ]
        else:
            host = [self.codes_host[index.cpu().numpy()] for index in (index_x, index_a, index_b)]
            (ux, ix), (ua, ia), (ub, ib) = (np.unique(codes, return_inverse=True) for codes in host)
            local = self.evaluate(ux, ua, ub)[ix[:, None, None], ia[None, :, None], ib[None, None, :]]
            valid = torch.from_numpy(np.ascontiguousarray(local)).to(self.device)
        if self.is_symmetric:
            valid &= index_x[:, None, None] != index_a[None, :, None]
        return valid


def apply_constraints(
    cells: pl.DataFrame,
    labels: pl.DataFrame,
    constraints: Constraints,
    *,
    is_symmetric: bool,
) -> pl.DataFrame:
    """Add to the cells their ``is_valid`` column: the flat mask of each cell's triplets, in x, a, b order.

    This materialises one boolean per triplet, for inspection. Scoring does not use it: it evaluates the
    constraints block by block with a :py:class:`ConstraintMask`.
    """
    mask = ConstraintMask(labels, constraints, is_symmetric=is_symmetric, device=torch.device("cpu"))
    valid = [
        mask.block(torch.tensor(index_x), torch.tensor(index_a), torch.tensor(index_b)).flatten().tolist()
        for index_a, index_b, index_x in cells[["index_a", "index_b", "index_x"]].iter_rows()
    ]
    return cells.with_columns(is_valid=pl.Series(valid, dtype=pl.List(pl.Boolean)))
