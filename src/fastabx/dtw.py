"""DTW implementation using PyTorch C++ extensions, with CPU and CUDA backends."""

import torch


def dtw(distances: torch.Tensor) -> torch.Tensor:
    """Compute the DTW of the given ``distances`` 2D tensor.

    :param distances: A 2D tensor of shape (n, m) representing the pairwise distances between two sequences.
    """
    return torch.ops.fastabx.dtw.default(distances)


def dtw_batch(distances: torch.Tensor, sx: torch.Tensor, sy: torch.Tensor, *, symmetric: bool) -> torch.Tensor:
    """Compute the batched DTW on the ``distances`` 4D tensor.

    :param distances: A 4D tensor of shape (n1, n2, s1, s2) representing the pairwise distances between two
        batches of sequences.
    :param sx: A 1D tensor of shape (n1,) representing the lengths of the sequences in the first batch.
    :param sy: A 1D tensor of shape (n2,) representing the lengths of the sequences in the second batch.
    :param symmetric: Whether or not the DTW is symmetric (i.e., the two batches are the same).
    """
    return torch.ops.fastabx.dtw_batch.default(distances, sx, sy, symmetric)


@torch.library.register_fake("fastabx::dtw")
def _(distances: torch.Tensor) -> torch.Tensor:
    """Register the FakeTensor kernel for dtw, for compatibility with torch.compile."""
    torch._check(distances.ndim == 2)
    torch._check(distances.dtype == torch.float32)
    return torch.empty((), dtype=torch.float32, layout=distances.layout, device=distances.device)


@torch.library.register_fake("fastabx::dtw_batch")
def _(distances: torch.Tensor, sx: torch.Tensor, sy: torch.Tensor, symmetric: bool) -> torch.Tensor:  # noqa: FBT001
    """Register the FakeTensor kernel for dtw_batch, for compatibility with torch.compile."""
    torch._check(distances.ndim == 4)
    torch._check(sx.ndim == 1)
    torch._check(sy.ndim == 1)
    torch._check(distances.dtype == torch.float32)
    torch._check(sx.dtype == torch.long)
    torch._check(sy.dtype == torch.long)
    torch._check(isinstance(symmetric, bool))
    nx, ny, _, _ = distances.shape
    return torch.empty((nx, ny), dtype=torch.float32, layout=distances.layout, device=distances.device)
