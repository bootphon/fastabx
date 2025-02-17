"""Subsampling functions."""

import polars as pl
import polars.selectors as cs

from fastabx.verify import verify_subsampler_params


def subsample_each_cell(df: pl.LazyFrame, size: int, seed: int = 0) -> pl.LazyFrame:
    """Subsample each cell by taking at most ``size`` instances of A, B, and X independently."""
    return (
        df.with_columns(group=pl.concat_str(~cs.starts_with("index"), separator="-"))
        .with_columns(cs.starts_with("index").explode().shuffle(seed=seed).implode().over("group").list.head(size))
        .select(cs.exclude("group"))
    )


def subsample_across_group(df: pl.LazyFrame, size: int, seed: int = 0) -> pl.LazyFrame:
    """Subsample each group of 'across' condition by taking ``size`` possible values for X in each group."""
    x_cols = [c for c in df.collect_schema() if c.endswith("_x") and c != "index_x"]
    df = df.with_columns(group=pl.concat_str(~(cs.starts_with("index") | cs.ends_with("_x")), separator="-"))
    return (
        df.group_by("group", maintain_order=True)
        .agg((cs.ends_with("_x") & (~cs.starts_with("index"))).unique().shuffle(seed).head(size))
        .explode(x_cols)
        .join(df, on=["group", *x_cols], how="left")
        .select(cs.exclude("group"))
    )


class Subsampler:
    """Subsample the ABX :py:class:`.Task`.

    Each cell is limited to ``max_size_group`` items for A, B and X independently.
    When using "across" conditions, each group of (A, B) is limited to ``max_x_across`` possible values for X.
    """

    def __init__(self, max_size_group: int = 10, max_x_across: int = 5, seed: int = 0) -> None:
        verify_subsampler_params(max_size_group, max_x_across, seed=seed)
        self.max_size_group = max_size_group
        self.max_x_across = max_x_across
        self.seed = seed

    def __call__(self, lazy_cells: pl.LazyFrame, *, with_across: bool) -> pl.LazyFrame:
        """Subsample the cells."""
        lazy_cells = subsample_each_cell(lazy_cells, self.max_size_group)
        if with_across:
            lazy_cells = subsample_across_group(lazy_cells, self.max_x_across, self.seed)
        return lazy_cells

    def description(self, *, with_across: bool) -> str:
        """Return a description of the subsampling."""
        desc = f"maximal cell size: {self.max_size_group}"
        if with_across:
            desc += f"maximal number of X for (A, B): {self.max_x_across}"
        return desc
