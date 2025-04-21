"""DTW implementation using PyTorch C++ extensions, with CPU and CUDA backends."""

import sys

import torch

TORCH_VERSION = "2.6.0"
CUDA_VERSION = "12.4"

if torch.__version__ != TORCH_VERSION:
    msg = (
        f"The DTW PyTorch backend has been built with {TORCH_VERSION}."
        "Since there is no ABI/API compatibility between releases of PyTorch, "
        f"you must install torch=={TORCH_VERSION} to use this backend."
    )
    raise ImportError(msg)

if sys.platform in ["linux", "win32"] and torch.version.cuda != CUDA_VERSION:
    msg = (
        f"On Linux and Windows, the DTW PyTorch backend requires PyTorch with CUDA {CUDA_VERSION}. "
        "It it not compatible with other CUDA versions, or with the CPU only version of PyTorch, "
        "even if you wanted to only use the CPU backend of the DTW. "
    )
    raise ImportError(msg)

from . import _C  # type: ignore[attr-defined] # noqa: E402, F401


def dtw(distances: torch.Tensor) -> torch.Tensor:
    """Compute the DTW of the given ``distances`` 2D tensor."""
    return torch.ops.fastabx.dtw.default(distances)  # type: ignore[no-any-return]


def dtw_batch(distances: torch.Tensor, sx: torch.Tensor, sy: torch.Tensor, *, symmetric: bool) -> torch.Tensor:
    """Compute the batched DTW on the ``distances`` 4D tensor."""
    return torch.ops.fastabx.dtw_batch.default(distances, sx, sy, symmetric)  # type: ignore[no-any-return]


@torch.library.register_fake("fastabx::dtw")
def _(distances: torch.Tensor) -> torch.Tensor:
    """Register the FakeTensor kernel for dtw, for compatibility with torch.compile."""
    torch._check(distances.ndim == 2)  # noqa: PLR2004
    torch._check(distances.dtype == torch.float32)
    return torch.empty((), dtype=torch.float32, layout=distances.layout, device=distances.device)


@torch.library.register_fake("fastabx::dtw_batch")
def _(distances: torch.Tensor, sx: torch.Tensor, sy: torch.Tensor, symmetric: bool) -> torch.Tensor:  # noqa: FBT001, ARG001
    """Register the FakeTensor kernel for dtw_batch, for compatibility with torch.compile."""
    torch._check(distances.ndim == 4)  # noqa: PLR2004
    torch._check(sx.ndim == 1)
    torch._check(sy.ndim == 1)
    torch._check(distances.dtype == torch.float32)
    torch._check(sx.dtype == torch.long)
    torch._check(sy.dtype == torch.long)
    nx, ny, _, _ = distances.shape
    return torch.empty((nx, ny), dtype=torch.float32, layout=distances.layout, device=distances.device)
