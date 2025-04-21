"""DTW implementation using either PyTorch or Cython backend."""

import enum
import os
from importlib.util import find_spec

import torch


class ExtensionBackend(enum.Enum):
    TORCH = enum.auto()
    CYTHON = enum.auto()


def get_dtw_backend() -> ExtensionBackend:
    if os.getenv("FASTABX_CYTHON_DTW", "0") == "1":
        return ExtensionBackend.CYTHON
    try:
        find_spec("fastabx.dtw._dtw_torch")
    except ImportError:
        return ExtensionBackend.CYTHON
    return ExtensionBackend.TORCH


if get_dtw_backend() == ExtensionBackend.TORCH:
    from fastabx.dtw._dtw_torch import dtw, dtw_batch
else:
    from fastabx.dtw._dtw_numpy import dtw as dtw_numpy
    from fastabx.dtw._dtw_numpy import dtw_batch as dtw_batch_numpy

    def dtw(distances: torch.Tensor) -> torch.Tensor:
        """Compute the DTW of the given ``distances`` 2D tensor."""
        return torch.asarray(dtw_numpy(distances.cpu().numpy()), device=distances.device)

    def dtw_batch(distances: torch.Tensor, sx: torch.Tensor, sy: torch.Tensor, *, symmetric: bool) -> torch.Tensor:
        """Compute the batched DTW on the ``distances`` 4D tensor."""
        return torch.asarray(
            dtw_batch_numpy(distances.cpu().numpy(), sx.cpu().numpy(), sy.cpu().numpy(), symmetric),
            device=distances.device,
        )


__all__ = ["dtw", "dtw_batch"]
