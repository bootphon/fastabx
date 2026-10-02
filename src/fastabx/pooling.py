"""Pooling utilities."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Literal

import torch

from fastabx.accessor import InMemoryAccessor
from fastabx.dataset import Dataset

__all__ = ["PooledDataset", "PoolingName", "PoolingNormalizedError", "pool_dataset"]

type PoolingName = Literal["mean", "hamming"]


class PoolingNormalizedError(ValueError):
    """The dataset has already been L2-normalized and cannot be pooled."""

    def __init__(self) -> None:
        super().__init__(
            "The dataset has been L2-normalized (with a singularity border) by a previous cosine/angular "
            "Score, so its features are no longer in their original space and carry an extra column. "
            "Pooling them would average that border in and silently produce a different measure, and the "
            "pooled dataset would no longer be flagged as normalized. Pool a fresh Dataset instead, and "
            "score it afterwards."
        )


def hamming_pooling(x: torch.Tensor) -> torch.Tensor:
    """Apply the symmetric hamming window on the input Tensor."""
    window = torch.hamming_window(x.size(0), periodic=False, device=x.device, dtype=x.dtype)
    return (window @ x) / window.sum()


def pooling_function(name: PoolingName) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the corresponding pooling function."""
    match name:
        case "mean":
            return partial(torch.mean, dim=0)
        case "hamming":
            return hamming_pooling
        case _:
            raise ValueError(name)


@dataclass
class PooledDataset(Dataset):
    """Pooled dataset."""

    pooling: PoolingName

    def __repr__(self) -> str:
        return f"labels:\n{self.labels!r}\naccessor: {self.accessor!r}\npooling: {self.pooling}"


def pool_dataset(dataset: Dataset, pooling_name: PoolingName) -> PooledDataset:
    """Pool the :py:class:`.Dataset` using the pooling method given by ``pooling_name``.

    The pooled dataset is a new one, with data stored in memory on the same device as ``dataset``. For simplicity,
    we iterate through the original dataset and apply pooling on each element.

    :param dataset: The dataset to pool.
    :param pooling_name: The pooling method, either "mean" or "hamming".
    """
    if dataset.accessor.is_normalized:
        raise PoolingNormalizedError
    labels = dataset.labels
    indices = {i: (i, i + 1) for i in range(len(labels))}
    pooling_fn = pooling_function(pooling_name)
    data = torch.stack([pooling_fn(x) for x in dataset.accessor], dim=0)
    accessor = InMemoryAccessor(indices, data, dataset.accessor.device)
    return PooledDataset(pooling=pooling_name, labels=labels, accessor=accessor)
