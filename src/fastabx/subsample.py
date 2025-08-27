"""Subsampling functions."""

import polars as pl
import polars.selectors as cs

from fastabx.verify import verify_subsampler_params


def subsample_each_cell(df: pl.LazyFrame, size: int, seed: int) -> pl.LazyFrame:
    """Subsample each cell by taking at most ``size`` instances of A, B, and X independently."""
    return (
        df.with_columns(pl.concat_str(~cs.starts_with("index"), separator="-").alias("__group"))
        .with_columns(cs.starts_with("index").explode().shuffle(seed=seed).implode().over("__group").list.head(size))
        .select(cs.exclude("__group"))
    )


def subsample_across_group(df: pl.LazyFrame, size: int, seed: int) -> pl.LazyFrame:
    """Subsample each group of 'across' condition by taking ``size`` possible values for X in each group."""
    x_cols = [c for c in df.collect_schema() if c.endswith("_x") and c != "index_x"]
    to_ignore = cs.starts_with("index") | cs.ends_with("_x")
    df = df.with_columns(pl.concat_str(~to_ignore, separator="-").alias("__group"))
    return (
        df.group_by("__group", maintain_order=True)
        .agg((cs.ends_with("_x") & (~cs.starts_with("index"))).unique(maintain_order=True).shuffle(seed).head(size))
        .explode(x_cols)
        .join(df, on=["__group", *x_cols], how="left")
        .select(cs.exclude("__group"))
    )


class Subsampler:
    """Subsample the ABX :py:class:`.Task`.

    Each cell is limited to ``max_size_group`` items for A, B and X independently.
    When using "across" conditions, each group of (A, B) is limited to ``max_x_across`` possible values for X.
    Subsampling for one or more conditions can be disabled by setting the corresponding argument to ``None``.

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
        if self.max_size_group is not None:
            lazy_cells = subsample_each_cell(lazy_cells, self.max_size_group, self.seed)
        if with_across and self.max_x_across is not None:
            lazy_cells = subsample_across_group(lazy_cells, self.max_x_across, self.seed)
        return lazy_cells

    def description(self, *, with_across: bool) -> str:
        """Return a description of the subsampling."""
        desc = []
        if self.max_size_group is not None:
            desc.append(f"maximal number of A, B, or X in a cell: {self.max_size_group}")
        if with_across and self.max_x_across is not None:
            desc.append(f"maximal number of X for (A, B): {self.max_x_across}")
        return ",".join(desc)
