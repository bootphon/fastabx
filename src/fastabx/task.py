"""Task module. The Task class builds all the cells for the 'by', 'on' and 'across' conditions."""

from collections.abc import Generator

from fastabx.cell import Cell, cell_description, cell_header, cells_on_by, cells_on_by_across
from fastabx.dataset import Dataset
from fastabx.subsample import Subsampler
from fastabx.verify import verify_dataset_labels, verify_task_conditions


class Task:
    """The ABX task class.

    A Task builds all the :py:class:`.Cell` given ``on``, ``by`` and ``across`` conditions.
    It can be subsampled to limit the number of cells.

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
        self.dataset = dataset
        self.on = on
        self.by = by or []
        self.across = across or []
        verify_task_conditions([self.on, *self.by, *self.across])
        verify_dataset_labels(dataset.labels.select([self.on, *self.by, *self.across]))
        self._subsampler_description = subsampler.description(with_across=bool(self.across)) if subsampler else ""

        if self.across:
            cells = cells_on_by_across(self.dataset.labels.lazy(), self.on, self.by, self.across)
        else:
            cells = cells_on_by(self.dataset.labels.lazy(), self.on, self.by)
        if subsampler:
            cells = subsampler(cells, with_across=bool(self.across))
        self.cells = cells.with_columns(
            description=cell_description(self.on, self.by, self.across),
            header=cell_header(self.on, self.by, self.across),
        ).collect()

    def __len__(self) -> int:
        return len(self.cells)

    def __getitem__(self, i: int) -> Cell:
        if i < 0 or i >= len(self):
            raise IndexError
        a = self.dataset.accessor.batched(self.cells[i, "index_a"])
        b = self.dataset.accessor.batched(self.cells[i, "index_b"])
        x = self.dataset.accessor.batched(self.cells[i, "index_x"])
        header, description = self.cells[i, "header"], self.cells[i, "description"]
        is_symmetric = not bool(self.across)
        return Cell(a=a, b=b, x=x, header=header, description=description, is_symmetric=is_symmetric)

    def __iter__(self) -> Generator[Cell, None, None]:
        is_symmetric = not bool(self.across)
        columns = ["header", "description", "index_a", "index_b", "index_x"]
        for header, description, index_a, index_b, index_x in self.cells[columns].iter_rows():
            a = self.dataset.accessor.batched(index_a)
            b = self.dataset.accessor.batched(index_b)
            x = self.dataset.accessor.batched(index_x)
            yield Cell(a=a, b=b, x=x, header=header, description=description, is_symmetric=is_symmetric)

    def __repr__(self) -> str:
        return (
            f"Task(\n\tON({self.on})"
            + (f"\n\tBY({', '.join(self.by)})" if self.by else "")
            + (f"\n\tACROSS({', '.join(self.across)})" if self.across else "")
            + f"\n\t{self._subsampler_description}\n)"
        )
