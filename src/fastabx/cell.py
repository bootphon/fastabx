"""Task module. The Task class builds all the cells for the 'by', 'on' and 'across' conditions."""

from dataclasses import dataclass

import polars as pl
import polars.selectors as cs

from fastabx.dataset import Batch
from fastabx.verify import verify_cell

MIN_A_LEN = 2  # Minimum length of A in the ABX task.


@dataclass(frozen=True)
class Cell:
    """Individual cell of the ABX task.

    Cells are the unit of work for the ABX :py:class:`.Task` and :py:class:`.Score`.
    They are collections of triplets (A, B, X) that share the same values for the ``on``, ``by`` and ``across``
    conditions.

    :param a: Batch of A samples.
    :param b: Batch of B samples.
    :param x: Batch of X samples.
    :param header: Short string identifying the cell.
    :param description: Long string describing the cell.
    :param is_symmetric: Whether or not the cell is symmetric (i.e., A and X are the same set).
    """

    a: Batch
    b: Batch
    x: Batch
    header: str
    description: str
    is_symmetric: bool

    def __post_init__(self) -> None:
        verify_cell((self.a.data, self.a.sizes), (self.b.data, self.b.sizes), (self.x.data, self.x.sizes))

    @property
    def num_triplets(self) -> int:
        """Get the number of triplets in the cell."""
        nx = self.a.data.size(0) - 1 if self.is_symmetric else self.x.data.size(0)
        return self.a.data.size(0) * self.b.data.size(0) * nx

    @property
    def use_dtw(self) -> bool:
        """Whether or not to use the DTW when computing the distances for this cell.

        We don't need DTW if all samples in the cell have a time dimension of 1.
        """
        return not (1 == self.a.data.size(1) == self.b.data.size(1) == self.x.data.size(1))

    def __len__(self) -> int:
        return self.num_triplets

    def __repr__(self) -> str:
        return "Cell(\n\t" + ")\n\t".join(self.description.split("), ")) + "\n)"


def cells_on_by(df: pl.LazyFrame, on: str, by: list[str]) -> pl.LazyFrame:
    """Generate the cells for the ABX task given 'on' and 'by' conditions.

    The 'by' condition is optional, the function still works if 'by' is an empty list.
    """
    df = (
        df.select([on, *by])
        .with_row_index()
        .group_by([on, *by], maintain_order=True)
        .agg(pl.col("index"))
        .with_columns(len=pl.col("index").list.len())
        .with_row_index(name="__lookup")
    )
    lookup = df.select("__lookup", "index")
    cells = df.select(cs.exclude("index"))
    cells = (
        cells.join(cells, on=by or None, suffix="_b", how="inner" if by else "cross")  # A_by == B_by
        .filter(pl.col(on) != pl.col(f"{on}_b"), pl.col("len") >= MIN_A_LEN)  # A_on != B_on,  A_len >= MIN_A_LEN
        .select(cs.exclude(f"{on}_x", "len", "len_b"))
    )
    return (
        cells.join(lookup.rename({"index": "index_a"}), on="__lookup", how="left")
        .join(lookup.rename({"__lookup": "__lookup_b", "index": "index_b"}), on="__lookup_b", how="left")
        .select(cs.exclude("__lookup", "__lookup_b"))
        .with_columns(index_x=pl.col("index_a"))
    )


class NoAcrossError(ValueError):
    """To raise if you expect 'across' conditions but received none."""

    def __init__(self) -> None:
        super().__init__("across must be non-empty")


def cells_on_by_across(df: pl.LazyFrame, on: str, by: list[str], across: list[str]) -> pl.LazyFrame:
    """Generate the cells for the ABX task given 'on', 'by' and 'across' conditions."""
    if not across:
        raise NoAcrossError
    df = (
        df.select([on, *by, *across])
        .with_row_index()
        .group_by([on, *by, *across], maintain_order=True)
        .agg(pl.col("index"))
        .with_row_index(name="__lookup")
    )
    lookup = df.select("__lookup", "index")
    cells = df.select(cs.exclude("index"))
    cells = (
        cells.join(cells, on=by + across, suffix="_b", how="inner")  # A_{by, across} == B_{by, across}
        .filter(pl.col(on) != pl.col(f"{on}_b"))  # A_on != B_on
        .join(cells, on=by or None, suffix="_x", how="inner" if by else "cross")  # A_by, B_by == X_by
        .filter(
            pl.col(on) == pl.col(f"{on}_x"),  # A_on == X_on
            pl.col(f"{on}_b") != pl.col(f"{on}_x"),  # B_on != X_on
            *[pl.col(c) != pl.col(f"{c}_x") for c in across],  # A_across, B_across != X_across
        )
        .select(cs.exclude(f"{on}_x"))
    )
    return (
        cells.join(lookup.rename({"index": "index_a"}), on="__lookup", how="left")
        .join(lookup.rename({"__lookup": "__lookup_b", "index": "index_b"}), on="__lookup_b", how="left")
        .join(lookup.rename({"__lookup": "__lookup_x", "index": "index_x"}), on="__lookup_x", how="left")
        .select(cs.exclude("__lookup", "__lookup_b", "__lookup_x"))
    )


def cell_description(on: str, by: list[str], across: list[str]) -> pl.Expr:
    """Long description of a cell."""
    cell = f"ON({on}_ax = " + pl.col(on).cast(pl.String) + f", {on}_b = " + pl.col(f"{on}_b").cast(pl.String) + ")"
    if by:
        cell += ", " + pl.concat_str([f"BY({c}_abx = " + pl.col(c).cast(pl.String) + ")" for c in by], separator=", ")
    if across:
        exprs = [
            f"{c}_ab = " + pl.col(c).cast(pl.String) + f", {c}_x = " + pl.col(f"{c}_x").cast(pl.String) for c in across
        ]
        cell += ", " + pl.format("ACROSS({})", pl.concat_str(exprs, separator=", "))
    return cell


def cell_header(on: str, by: list[str], across: list[str]) -> pl.Expr:
    """Header of a cell."""
    return pl.concat_str([on, f"{on}_b"] + by + across + [f"{c}_x" for c in across], separator="-")
