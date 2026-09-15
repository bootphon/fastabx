"""Subsampling functions."""

import polars as pl
import polars.selectors as cs

from fastabx.cell import INDEX_COLUMNS
from fastabx.verify import verify_subsampler_params

__all__ = ["Subsampler"]


def subsample_each_cell(df: pl.LazyFrame, size: int, seed: int) -> pl.LazyFrame:
    """Subsample each cell by taking at most ``size`` instances of A, B, and X independently.

    The shuffle uses a single fixed ``seed`` under ``.over("__group")``. Because ``shuffle(seed)``
    applies the *same* permutation to every equal-length group, this is what keeps A and X in step
    for symmetric cells (where ``index_a == index_x``): both columns are shuffled identically, so
    ``.head(size)`` selects the same subset and the diagonal exclusion in scoring stays valid.

    The flip side is that the choice of which items survive is correlated across cells of the same
    size rather than drawn independently per cell. This is intentional (it preserves the A/X
    invariant and keeps the subsampling reproducible), but it is *not* an i.i.d. sample; do not rely
    on per-cell independence of the retained items.
    """
    return (
        df.with_row_index("__group")
        .with_columns(
            INDEX_COLUMNS.explode(empty_as_null=False).shuffle(seed=seed).implode().over("__group").list.head(size)
        )
        .select(cs.exclude("__group"))
    )


def subsample_across_group(df: pl.LazyFrame, size: int, seed: int) -> pl.LazyFrame:
    """Retain up to ``size`` observed tuples of X conditions for each A/B condition group."""
    x_cols = [c for c in df.collect_schema() if c.endswith("_x") and c != "index_x"]
    group_cols = df.select(~(INDEX_COLUMNS | cs.ends_with("_x"))).collect_schema().names()
    return (
        df.group_by(group_cols, maintain_order=True)
        .agg(pl.struct(x_cols).unique(maintain_order=True).shuffle(seed).head(size).alias("__group"))
        .explode("__group", empty_as_null=False)
        .unnest("__group")
        .join(df, on=[*group_cols, *x_cols], how="left", maintain_order="left")
    )


class Subsampler:
    """Subsample the ABX :py:class:`.Task`.

    Each cell is limited to ``max_size_group`` items for A, B and X independently.
    When using "across" conditions, each group of (A, B) is limited to ``max_x_across`` observed
    combinations of X condition values.
    Subsampling for one or more conditions can be disabled by setting the corresponding argument to ``None``.

    .. note::
        The subsampling is reproducible given ``seed``, but it is not an i.i.d. sample. A single fixed seed
        shuffles every cell, and the same permutation is applied to all the groups of the same length. This is
        what keeps A and X in step in symmetric cells (where they are the same set, and where scoring relies on
        it to drop the diagonal), and the flip side is that the items retained in cells of equal size are
        correlated rather than drawn independently.

    :param max_size_group: Maximum number of instances of A, B, or X in each :py:class:`.Cell`.
        Set to 10 in the original ZeroSpeech ABX code. Disabled if set to ``None``.
    :param max_x_across: In the "across" speaker mode, maximum number of X considered for given values of A and B.
        Set to 5 in the original ZeroSpeech ABX code. Disabled if set to ``None``.
    :param seed: The random seed for the subsampling, default is 0.
    """

    def __init__(self, max_size_group: int | None, max_x_across: int | None, seed: int = 0) -> None:
        verify_subsampler_params(max_size_group, max_x_across, seed=seed)
        self.max_size_group = max_size_group
        self.max_x_across = max_x_across
        self.seed = seed

    def __call__(self, lazy_cells: pl.LazyFrame, *, with_across: bool) -> pl.LazyFrame:
        """Subsample the cells."""
        if with_across and self.max_x_across is not None:
            lazy_cells = subsample_across_group(lazy_cells, self.max_x_across, self.seed)
        if self.max_size_group is not None:
            lazy_cells = subsample_each_cell(lazy_cells, self.max_size_group, self.seed)
        return lazy_cells

    def description(self, *, with_across: bool) -> str:
        """Return a description of the subsampling."""
        desc = []
        if self.max_size_group is not None:
            desc.append(f"maximal number of A, B, or X in a cell: {self.max_size_group}")
        if with_across and self.max_x_across is not None:
            desc.append(f"maximal number of X for (A, B): {self.max_x_across}")
        return ",".join(desc)
