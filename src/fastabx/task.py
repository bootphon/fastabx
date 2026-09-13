"""Task module. The Task class builds all the cells for the 'by', 'on' and 'across' conditions."""

from collections.abc import Generator

import polars as pl

from fastabx.cell import Cell, cell_description, cell_header, cells_on_by, cells_on_by_across
from fastabx.dataset import Dataset
from fastabx.subsample import Subsampler
from fastabx.verify import (
    verify_conditions_exist,
    verify_dataset_labels,
    verify_precomputed_cells,
    verify_task_conditions,
    verify_task_is_not_empty,
)

__all__ = ["Task"]


def task_cells(
    dataset: Dataset,
    on: str,
    by: list[str],
    across: list[str],
    subsampler: Subsampler | None,
) -> pl.DataFrame:
    """Build cells from the dataset and the task conditions."""
    if across:
        cells = cells_on_by_across(dataset.labels.lazy(), on, by, across)
    else:
        cells = cells_on_by(dataset.labels.lazy(), on, by)
    if subsampler:
        cells = subsampler(cells, with_across=bool(across))
    return cells.with_columns(
        description=cell_description(on, by, across),
        header=cell_header(on, by, across),
    ).collect()


class Task:
    """The ABX task class.

    A Task builds all the :py:class:`.Cell` given ``on``, ``by`` and ``across`` conditions.
    It can be subsampled to limit the number of cells.

    To bypass the standard construction with a precomputed cells DataFrame, use
    :py:meth:`Task.from_cells` instead.

    Every condition must be a column of ``dataset.labels``.

    :param dataset: The dataset containing the features and the labels.
    :param on: The ``on`` condition.
    :param by: The list of ``by`` conditions.
    :param across: The list of ``across`` conditions.
    :param subsampler: An optional subsampler to limit the number of cells and their sizes.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        on: str,
        by: list[str] | None = None,
        across: list[str] | None = None,
        subsampler: Subsampler | None = None,
    ) -> None:
        by, across = by or [], across or []
        conditions = [on, *by, *across]
        verify_task_conditions(conditions)
        verify_conditions_exist(dataset.labels.columns, conditions)
        verify_dataset_labels(dataset.labels.select(conditions))
        cells = task_cells(dataset, on, by, across, subsampler)
        verify_task_is_not_empty(len(cells), on, by, across)
        self._set_parts(
            dataset,
            cells,
            on=on,
            by=by,
            across=across,
            is_symmetric=not bool(across),
            subsampler_description=subsampler.description(with_across=bool(across)) if subsampler else "",
        )

    def _set_parts(
        self,
        dataset: Dataset,
        cells: pl.DataFrame,
        *,
        on: str,
        by: list[str],
        across: list[str],
        is_symmetric: bool,
        subsampler_description: str,
    ) -> None:
        self.dataset = dataset
        self.on = on
        self.by = by
        self.across = across
        self.is_symmetric = is_symmetric
        self._subsampler_description = subsampler_description
        self._cells = cells

    @classmethod
    def from_cells(cls, dataset: Dataset, cells: pl.DataFrame, *, is_symmetric: bool) -> "Task":
        """Build a Task from a precomputed cells DataFrame.

        Use this when you have hardcoded your own triplets and want to skip the
        standard ``on``/``by``/``across`` construction. The DataFrame must carry the
        following columns: ``header``, ``description``, ``index_a``, ``index_b``, ``index_x``.

        :param dataset: The dataset containing the features and the labels.
        :param cells: The precomputed cells DataFrame.
        :param is_symmetric: Whether each cell's A and X share the same rows (no across condition).
        """
        verify_precomputed_cells(cells, num_items=len(dataset.accessor), is_symmetric=is_symmetric)
        task = cls.__new__(cls)
        task._set_parts(dataset, cells, on="", by=[], across=[], is_symmetric=is_symmetric, subsampler_description="")  # ruff: ignore[private-member-access]
        return task

    @property
    def cells(self) -> pl.DataFrame:
        """Read-only view of the task's cells."""
        return self._cells

    def __len__(self) -> int:
        return len(self.cells)

    def __getitem__(self, i: int) -> Cell:
        num_cells = len(self)
        index = i + num_cells if i < 0 else i
        if index < 0 or index >= num_cells:
            msg = f"Cell index {i} out of range for a task with {num_cells} cells"
            raise IndexError(msg)
        a = self.dataset.accessor.batched(self.cells[index, "index_a"])
        b = self.dataset.accessor.batched(self.cells[index, "index_b"])
        x = self.dataset.accessor.batched(self.cells[index, "index_x"])
        header, description = self.cells[index, "header"], self.cells[index, "description"]
        return Cell(a=a, b=b, x=x, header=header, description=description, is_symmetric=self.is_symmetric)

    def __iter__(self) -> Generator[Cell, None, None]:
        columns = ["header", "description", "index_a", "index_b", "index_x"]
        for header, description, index_a, index_b, index_x in self.cells[columns].iter_rows():
            a = self.dataset.accessor.batched(index_a)
            b = self.dataset.accessor.batched(index_b)
            x = self.dataset.accessor.batched(index_x)
            yield Cell(a=a, b=b, x=x, header=header, description=description, is_symmetric=self.is_symmetric)

    def __repr__(self) -> str:
        return (
            f"Task(\n\tON({self.on})"
            + (f"\n\tBY({', '.join(self.by)})" if self.by else "")
            + (f"\n\tACROSS({', '.join(self.across)})" if self.across else "")
            + (f"\n\t{self._subsampler_description}" if self._subsampler_description else "")
            + "\n)"
        )
