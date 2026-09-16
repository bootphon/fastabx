"""Access to the features of a dataset: the ``Accessor`` protocol and the in-memory implementation."""

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt
import torch

from fastabx.verify import InvalidFeaturesError, verify_continuous_dtype, verify_empty_datapoints, verify_feature_shape

__all__ = ["Accessor", "Batch", "InMemoryAccessor"]


type ArrayLike = npt.ArrayLike  # Better rendering in docs


@dataclass(frozen=True)
class Batch:
    """Batch of padded data."""

    data: torch.Tensor
    sizes: torch.Tensor

    def __repr__(self) -> str:
        return f"Batch(data=Tensor(shape={self.data.shape}, dtype={self.data.dtype}), sizes={self.sizes})"


class Accessor(Protocol):
    """How the ABX pipeline reads the features of a :py:class:`.Dataset`.

    :py:class:`.InMemoryAccessor` is the only implementation shipped by fastabx, and everything it needs
    fits in memory. Anything satisfying this protocol works in its place: a memory-mapped store, a lazy
    reader, an accessor keeping its data on a device of its own. The pipeline only ever reads through
    :py:meth:`lengths` and :py:meth:`batched`, so those two are the ones that must be fast.

    Indices are the row numbers of ``Dataset.labels``: item ``i`` of the accessor describes row ``i``.
    """

    device: torch.device
    is_normalized: bool

    def __len__(self) -> int:
        """Return the number of datapoints."""
        ...

    def __getitem__(self, i: int) -> torch.Tensor:
        """Return the features of the datapoint ``i``, of shape ``(length, dim)``."""
        ...

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Iterate over the features of every datapoint, in index order."""
        ...

    def lengths(self, indices: list[int]) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
        """Return the number of frames of each given datapoint, without reading the features themselves.

        :param indices: The indices of the datapoints.
        """
        ...

    def batched(self, indices: ArrayLike) -> Batch:
        """Gather the given datapoints into a single padded :py:class:`.Batch`.

        :param indices: The indices of the datapoints. The order of the batch follows the order given here.
        """
        ...

    def normalize_(self) -> None:
        """L2 normalize the features in place, and extend them with a singularity border.

        Must be idempotent, and must set ``is_normalized``: a second call, and any :py:class:`.Score`
        using the angular distance on an already normalized accessor, are no-ops.
        """
        ...


class InMemoryAccessor:
    """Data accessor where everything is in memory.

    :param indices: Mapping from the index of a datapoint to its ``[start, end[`` frontiers in ``data``.
    :param data: The features of all the datapoints, concatenated along the time dimension.
    :param device: Device on which the data is stored.
    """

    def __init__(self, indices: dict[int, tuple[int, int]], data: torch.Tensor, device: torch.device) -> None:
        self.device = device
        self.indices = indices
        verify_empty_datapoints(self.indices)
        verify_feature_shape(data)
        if any(start < 0 or end > data.size(0) for start, end in indices.values()):
            msg = "Accessor slices must lie within the feature tensor's frame dimension."
            raise InvalidFeaturesError(msg)
        self.data = data.to(self.device)
        self.is_normalized = False
        size = max(self.indices) + 1
        starts, lengths = np.zeros(size, dtype=np.int64), np.zeros(size, dtype=np.int64)
        for i, (start, end) in self.indices.items():
            starts[i], lengths[i] = start, end - start
        self._lengths_np = lengths
        self._starts = torch.from_numpy(starts).to(self.device)
        self._lengths = torch.from_numpy(lengths).to(dtype=torch.int32, device=self.device)

    def __repr__(self) -> str:
        return f"InMemoryAccessor(data of shape {tuple(self.data.shape)}, with {len(self)} items)"

    def __getitem__(self, i: int) -> torch.Tensor:
        if i not in self.indices:
            msg = f"No item at index {i} (the accessor has {len(self.indices)} items)"
            raise IndexError(msg)
        start, end = self.indices[i]
        return self.data[start:end]

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self) -> Iterator[torch.Tensor]:
        for i in range(len(self)):
            yield self[i]

    def lengths(self, indices: list[int]) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
        """Get the lengths of the data from a list of indices."""
        return self._lengths_np[indices]

    def normalize_(self) -> None:
        """L2 normalize the data in place, and extend it with a singularity border. Idempotent."""
        if self.is_normalized:
            return
        self.data = normalize_with_singularity(self.data)
        self.is_normalized = True

    def batched(self, indices: ArrayLike) -> Batch:
        """Get the padded data and the original sizes of the data from a list of indices."""
        idx_np = np.asarray(indices, dtype=np.int64)
        smax = int(self._lengths_np[idx_np].max())
        idx = torch.from_numpy(idx_np).to(self.device)
        sizes = self._lengths.index_select(0, idx)
        starts = self._starts.index_select(0, idx)
        arange = torch.arange(smax, device=self.device)
        mask = arange < sizes.unsqueeze(1)
        src = starts.unsqueeze(1) + arange
        src.mul_(mask)
        gathered = self.data.index_select(0, src.view(-1)).view(idx.size(0), smax, -1)
        gathered.mul_(mask.unsqueeze(-1))
        return Batch(gathered, sizes)


def normalize_with_singularity(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize the given vector across the third dimension.

    Extend all vectors by eps to put the null vector at the maximal angular distance from any non-null vector.
    """
    verify_continuous_dtype(x)
    dim = x.size(1)
    norm = x.norm(dim=1, keepdim=True)
    zero_mask = norm.squeeze(1) == 0
    out = x.new_empty((x.size(0), dim + 1))
    head = out[:, :dim]
    head.copy_(x)
    head.div_(norm.masked_fill(zero_mask.unsqueeze(1), 1.0))  # zero rows are overwritten just below
    head[zero_mask] = 1.0 / math.sqrt(dim)
    out[:, dim] = eps
    out[zero_mask, dim] = -2 * eps
    return out
