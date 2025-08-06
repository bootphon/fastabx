"""DTW implementation using PyTorch C++ extensions, with CPU and CUDA backends."""

import torch


def dtw(distances: torch.Tensor) -> torch.Tensor:
    """Compute the DTW of the given ``distances`` 2D tensor."""
    return torch.ops.fastabx.dtw.default(distances)


def dtw_batch(distances: torch.Tensor, sx: torch.Tensor, sy: torch.Tensor, *, symmetric: bool) -> torch.Tensor:
    """Compute the batched DTW on the ``distances`` 4D tensor."""
    return torch.ops.fastabx.dtw_batch.default(distances, sx, sy, symmetric)
