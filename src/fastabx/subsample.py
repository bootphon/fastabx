"""Subsampling functions."""

import abc
from collections.abc import Iterable

import polars as pl
import polars.selectors as cs

from fastabx.verify import verify_subsampler_params


class Subsampler(abc.ABC):
    """Subsample the ABX :py:class:`.Task`."""

    _description: str

    def description(self) -> str:
        """Return a description of the subsampling."""
        return self._description

    @abc.abstractmethod
    def __call__(self, cells: pl.LazyFrame) -> pl.LazyFrame:
        """Subsample the cells."""

    def __add__(self, other: "Subsampler") -> "ConcatSubsampler":
        """Chain multiple :py:class:`.Subsampler`."""
        return ConcatSubsampler([self, other])

    def __repr__(self) -> str:
        return f"Subsampler({self.description()})"


class ConcatSubsampler(Subsampler):
    """Concatenation of multiple :py:class:`.Subsampler`. Apply them one after the other."""

    def __init__(self, subsamplers: Iterable[Subsampler]) -> None:
        self._subsamplers = list(subsamplers)
        self._description = ", ".join([subsampler.description() for subsampler in self._subsamplers])

    def __call__(self, cells: pl.LazyFrame) -> pl.LazyFrame:
        """Subsample the cells."""
        for subsampler in self._subsamplers:
            cells = subsampler(cells)
        return cells


class SizeSubsampler(Subsampler):
    """Subsample each cell by taking at most ``size`` instances of A, B, and X independently."""

    def __init__(self, seed: int, size: int) -> None:
        self._seed = seed
        self._size = size
        self._description = f"maximal number of A, B, or X in a cell: {self._size}"
        verify_subsampler_params(size, seed=seed)

    def __call__(self, cells: pl.LazyFrame) -> pl.LazyFrame:
        """Subsample the cells."""
        return (
            cells.with_columns(pl.concat_str(~cs.starts_with("index"), separator="-").alias("__group"))
            .with_columns(
                cs.starts_with("index")
                .explode()
                .shuffle(seed=self._seed)
                .implode()
                .over("__group")
                .list.head(self._size)
            )
            .select(cs.exclude("__group"))
        )


class AcrossGroupSubsampler(Subsampler):
    """Subsample each group of 'across' condition by taking ``size`` possible values for X in each group."""

    def __init__(self, seed: int, size: int) -> None:
        self._seed = seed
        self._size = size
        self._description = f"maximal number of X for (A, B): {self._size}"
        verify_subsampler_params(size, seed=seed)

    def __call__(self, cells: pl.LazyFrame) -> pl.LazyFrame:
        """Subsample the cells."""
        x_cols = [c for c in cells.collect_schema() if c.endswith("_x") and c != "index_x"]
        to_ignore = cs.starts_with("index") | cs.ends_with("_x")
        cells = cells.with_columns(pl.concat_str(~to_ignore, separator="-").alias("__group"))
        return (
            cells.group_by("__group", maintain_order=True)
            .agg((cs.ends_with("_x") & (~cs.starts_with("index"))).unique().shuffle(self._seed).head(self._size))
            .explode(x_cols)
            .join(cells, on=["__group", *x_cols], how="left")
            .select(cs.exclude("__group"))
        )


def librilight_subsampler(seed: int, max_size_group: int | None, max_x_across: int | None) -> Subsampler | None:
    """Replicates Libri-Light subsampling."""
    match max_size_group, max_x_across:
        case None, None:
            return None
        case int(), None:
            return SizeSubsampler(seed, max_size_group)
        case None, int():
            return AcrossGroupSubsampler(seed, max_x_across)
        case int(), int():
            return SizeSubsampler(seed, max_size_group) + AcrossGroupSubsampler(seed, max_x_across)
        case _:
            raise ValueError(max_size_group, max_x_across)
