"""Pooling utilities."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Literal

import torch

from fastabx.dataset import Dataset, InMemoryAccessor

type PoolingName = Literal["mean", "hamming"]


def hamming_window(x: torch.Tensor) -> torch.Tensor:
    """Apply the hamming window on the input Tensor."""
    window = torch.hamming_window(x.size(0), device=x.device)
    return (window @ x) / window.sum()


def pooling_function(name: PoolingName) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the corresponding pooling function."""
    match name:
        case "mean":
            return partial(torch.mean, dim=0)
        case "hamming":
            return hamming_window
        case _:
            raise ValueError(name)


@dataclass(frozen=True)
class PooledDataset(Dataset):
    """Pooled dataset."""

    pooling: PoolingName

    def __repr__(self) -> str:
        return f"labels:\n{self.labels!r}\naccessor: {self.accessor!r}\npooling: {self.pooling}"


def pooling(dataset: Dataset, pooling_name: PoolingName) -> PooledDataset:
    """Pool the :py:class:`.Dataset` using the pooling method given by ``pooling_name``.

    The pooled dataset is a new one, with data stored in memory. For simplicity, we iterate through the original
    dataset and apply pooling on each element.

    :param dataset: The dataset to pool.
    :param pooling_name: The pooling method, either "mean" or "hamming".
    """
    labels = dataset.labels
    indices = {i: (i, i + 1) for i in range(len(labels))}
    pooling_fn = pooling_function(pooling_name)
    data = torch.stack([pooling_fn(x) for x in dataset.accessor], dim=0)
    return PooledDataset(pooling=pooling_name, labels=labels, accessor=InMemoryAccessor(indices, data))
