"""Check for compatibility with torch.compile."""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from torch.library import opcheck

import fastabx  # noqa: F401 # Need to import it to register dtw operation

DIM, BATCH = st.integers(1, 1280), st.integers(1, 3)
LOW, HIGH_MINUS_LOW = st.floats(-100, 100), st.floats(0.1, 100)
CUDA_AVAILABLE = torch.cuda.is_available()


def make_tensor(shape: tuple[int, ...], *, dtype: torch.dtype, low: float, high: float) -> torch.Tensor:
    """Build a tensor for testing."""
    if low == high and dtype == torch.long:
        return torch.ones(shape, dtype=torch.long, device="cpu")
    return torch.testing.make_tensor(shape, dtype=dtype, device="cpu", low=low, high=high)


@given(x=DIM, y=DIM, low=LOW, high_minus_low=HIGH_MINUS_LOW)
@settings(deadline=None)
def test_opcheck_dtw(x: int, y: int, low: float, high_minus_low: float) -> None:
    """Verify that dtw can be torch compiled."""
    sample = make_tensor((x, y), dtype=torch.float32, low=low, high=high_minus_low + low)
    opcheck(torch.ops.fastabx.dtw.default, (sample,))
    if CUDA_AVAILABLE:
        opcheck(torch.ops.fastabx.dtw.default, (sample.cuda(),))


@given(n=BATCH, x=DIM, low=LOW, high_minus_low=HIGH_MINUS_LOW)
@settings(deadline=None)
def test_opcheck_dtw_batch_symmetric(n: int, x: int, low: float, high_minus_low: float) -> None:
    """Verify that dtw_batch can be torch compiled, with symmetric input."""
    sample = make_tensor((n, n, x, x), dtype=torch.float32, low=low, high=high_minus_low + low)
    sx = make_tensor((n,), dtype=torch.long, low=1, high=x)
    i, j = torch.triu_indices(n, n)
    sample[i, j] = sample[j, i]
    opcheck(torch.ops.fastabx.dtw_batch.default, (sample, sx, sx), {"symmetric": True})
    if CUDA_AVAILABLE:
        opcheck(torch.ops.fastabx.dtw_batch.default, (sample.cuda(), sx.cuda(), sx.cuda()), {"symmetric": True})


@given(n=BATCH, m=BATCH, x=DIM, y=DIM, low=LOW, high_minus_low=HIGH_MINUS_LOW)
@settings(deadline=None)
def test_opcheck_dtw_batch_not_symmetric(n: int, m: int, x: int, y: int, low: float, high_minus_low: float) -> None:
    """Verify that dtw_batch can be torch compiled, with symmetric input."""
    sample = make_tensor((n, m, x, y), dtype=torch.float32, low=low, high=high_minus_low + low)
    sx = make_tensor((n,), dtype=torch.long, low=1, high=x)
    sy = make_tensor((m,), dtype=torch.long, low=1, high=y)
    opcheck(torch.ops.fastabx.dtw_batch.default, (sample, sx, sy), {"symmetric": False})
    if CUDA_AVAILABLE:
        opcheck(torch.ops.fastabx.dtw_batch.default, (sample.cuda(), sx.cuda(), sy.cuda()), {"symmetric": False})
