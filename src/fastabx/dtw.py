"""DTW implementation using PyTorch C++ extensions, with CPU and CUDA backends."""

import torch


def dtw(distances: torch.Tensor) -> float:
    """Compute the DTW of the given ``distances`` 2D tensor."""
    return torch.ops.fastabx.dtw(distances)  # type: ignore[no-any-return]


def dtw_batch(distances: torch.Tensor, sx: torch.Tensor, sy: torch.Tensor, *, symmetric: bool) -> torch.Tensor:
    """Compute the batched DTW on the ``distances`` 4D tensor."""
    return torch.ops.fastabx.dtw_batch(distances, sx, sy, symmetric)  # type: ignore[no-any-return]


def monkeypatch_dtw() -> None:
    """Monkey patch the dtw and dtw_batch functions to the Cython backend if something goes wrong."""
    import fastabx
    from fastabx import _dtw_cython

    def patched(distances: torch.Tensor) -> float:
        return _dtw_cython.dtw(distances.cpu().numpy())

    def batch_patched(distances: torch.Tensor, sx: torch.Tensor, sy: torch.Tensor, *, symmetric: bool) -> torch.Tensor:
        return torch.from_numpy(
            _dtw_cython.dtw_batched(distances.cpu().numpy(), sx.cpu().numpy(), sy.cpu().numpy(), symmetric)
        )

    fastabx.dtw.dtw = patched
    fastabx.dtw.dtw_batch = batch_patched
