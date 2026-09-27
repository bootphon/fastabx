"""Pooling utilities."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Literal

import numpy as np
import torch

from fastabx.accessor import InMemoryAccessor
from fastabx.dataset import Dataset
from fastabx.utils import gather_chunk_rows
from fastabx.verify import verify_continuous_dtype

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


def pool_batch(data: torch.Tensor, name: PoolingName) -> torch.Tensor:
    """Pool a ``(n, length, dim)`` batch of sequences that all have the same length, without padding.

    Gives exactly the same result as :py:func:`pooling_function` applied to each sequence.
    """
    if name == "mean":
        return data.mean(dim=1)
    if data.size(2) == 1:  # With one dimension, the batched product rounds differently from the per-item one.
        return torch.stack([hamming_pooling(x) for x in data])
    window = torch.hamming_window(data.size(1), periodic=False, device=data.device, dtype=data.dtype)
    return torch.matmul(window, data) / window.sum()


@dataclass
class PooledDataset(Dataset):
    """Pooled dataset."""

    pooling: PoolingName

    def __repr__(self) -> str:
        return f"labels:\n{self.labels!r}\naccessor: {self.accessor!r}\npooling: {self.pooling}"


def pool_dataset(dataset: Dataset, pooling_name: PoolingName) -> PooledDataset:
    """Pool the :py:class:`.Dataset` using the pooling method given by ``pooling_name``.

    The pooled dataset is a new one, with data stored in memory on the same device as ``dataset``. The items are
    pooled by batches of equal length, read with ``accessor.batched`` in chunks of at most
    ``FASTABX_GATHER_CHUNK_ROWS`` items (see :ref:`perf-env`).

    :param dataset: The dataset to pool.
    :param pooling_name: The pooling method, either "mean" or "hamming".
    """
    if dataset.accessor.is_normalized:
        raise PoolingNormalizedError
    pooling_function(pooling_name)  # Reject an unknown name before reading anything.
    source = dataset.accessor
    num_items = len(source)
    lengths = source.lengths(list(range(num_items)))
    order = np.argsort(lengths, kind="stable")
    same_length = np.split(order, np.flatnonzero(np.diff(lengths[order])) + 1)
    max_rows = gather_chunk_rows()
    data = None
    for bucket in same_length:
        for start in range(0, len(bucket), max_rows):
            rows = bucket[start : start + max_rows]
            batch = source.batched(rows)
            verify_continuous_dtype(batch.data)
            pooled = pool_batch(batch.data, pooling_name)
            if data is None:
                data = pooled.new_empty((num_items, pooled.size(1)))
            data[torch.from_numpy(rows).to(data.device)] = pooled
    indices = {i: (i, i + 1) for i in range(num_items)}
    accessor = InMemoryAccessor(indices, data, source.device)  # ty: ignore[invalid-argument-type]
    return PooledDataset(pooling=pooling_name, labels=dataset.labels, accessor=accessor)
