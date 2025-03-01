"""DTW implementation using PyTorch C++ extensions, with CPU and CUDA backends."""

import torch


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
