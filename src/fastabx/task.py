"""Task module. The Task class builds all the cells for the 'by', 'on' and 'across' conditions."""

from collections.abc import Generator, Iterable

import polars as pl
import torch
from torch.nn.utils.rnn import pad_sequence

from fastabx.cell import Cell, cell_description, cell_header, cells_on_by, cells_on_by_across
from fastabx.dataset import Batch, Dataset, InMemoryAccessor
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


def batched(accessor: InMemoryAccessor, indices: Iterable[int], padding_tensor: torch.Tensor) -> Batch:
    """Get the padded data and the original sizes of the data from a list of indices."""
    sizes, data = [], []
    for i in indices:
        this_data = accessor[i]
        sizes.append(this_data.size(0))
        data.append(this_data)
    num_padding = padding_tensor.size(0) - len(data)
    sizes += [1] * num_padding
    data += [padding_tensor[0] for _ in range(num_padding)]
    return Batch(pad_sequence(data, batch_first=True), torch.tensor(sizes, dtype=torch.int64, device=accessor.device))


def padded_cell_generator(task: Task) -> Generator[tuple[Cell, torch.Tensor], None, None]:
    sizes = (
        task.cells.lazy()
        .with_columns(pl.col(index).list.len() for index in ["index_a", "index_b", "index_x"])
        .select(pl.max(index) for index in ["index_a", "index_b", "index_x"])
        .collect()
        .to_dicts()[0]
    )
    max_length = max(end - start for start, end in task.dataset.accessor.indices.values())
    device = task.dataset.accessor.device
    max_a, max_b, max_x = sizes["index_a"] + 1, sizes["index_b"] + 1, sizes["index_x"] + 1
    padding_a = torch.zeros((max_a, max_length, 769), dtype=torch.float32, device=device)
    padding_b = torch.zeros((max_b, max_length, 769), dtype=torch.float32, device=device)
    padding_x = torch.zeros((max_x, max_length, 769), dtype=torch.float32, device=device)
    max_x_t = torch.arange(max_x, device=device)[:, None, None]
    max_a_t = torch.arange(max_a, device=device)[None, :, None]
    max_b_t = torch.arange(max_b, device=device)[None, None, :]

    is_symmetric = not bool(task.across)
    columns = ["header", "description", "index_a", "index_b", "index_x"]
    for header, description, index_a, index_b, index_x in task.cells[columns].iter_rows():
        a = batched(task.dataset.accessor, index_a, padding_a)
        b = batched(task.dataset.accessor, index_b, padding_b)
        x = batched(task.dataset.accessor, index_x, padding_x)
        mask = (max_x_t < len(index_x)) & (max_a_t < len(index_a)) & (max_b_t < len(index_b))
        yield Cell(a=a, b=b, x=x, header=header, description=description, is_symmetric=is_symmetric), mask
