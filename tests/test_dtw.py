"""Compare CPU and CUDA dtw implementations."""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from fastabx.dtw import dtw, dtw_batch

rtol, atol = 0, 1e-9
skipifnogpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")

DIM, BATCH = st.integers(1, 1280), st.integers(1, 3)
LOW, HIGH_MINUS_LOW = st.floats(-100, 100), st.floats(0.1, 100)


def make_tensor(shape: tuple[int, ...], *, dtype: torch.dtype, low: float, high: float) -> torch.Tensor:
    """Build a tensor for testing."""
    if low == high and dtype == torch.long:
        return torch.ones(shape, dtype=torch.long, device="cpu")
    return torch.testing.make_tensor(shape, dtype=dtype, device="cpu", low=low, high=high)


@skipifnogpu
@given(x=DIM, y=DIM, low=LOW, high_minus_low=HIGH_MINUS_LOW)
@settings(deadline=None)
def test_dtw(x: int, y: int, low: float, high_minus_low: float) -> None:
    """Compare the output of dtw between CPU and GPU implementations."""
    d = make_tensor((x, y), dtype=torch.float32, low=low, high=high_minus_low + low)
    torch.testing.assert_close(dtw(d), dtw(d.cuda()).cpu(), rtol=rtol, atol=atol)


@skipifnogpu
@given(n=BATCH, x=DIM, low=LOW, high_minus_low=HIGH_MINUS_LOW)
@settings(deadline=None)
def test_dtw_batch_symmetric(n: int, x: int, low: float, high_minus_low: float) -> None:
    """Compare the output of dtw_batch between CPU and GPU implementations, symmetric case."""
    d = make_tensor((n, n, x, x), dtype=torch.float32, low=low, high=high_minus_low + low)
    sx = make_tensor((n,), dtype=torch.long, low=1, high=x)
    i, j = torch.triu_indices(n, n)
    d[i, j] = d[j, i]
    torch.testing.assert_close(
        dtw_batch(d, sx, sx, symmetric=True),
        dtw_batch(d.cuda(), sx.cuda(), sx.cuda(), symmetric=True).cpu(),
        rtol=rtol,
        atol=atol,
    )


@skipifnogpu
@given(n=BATCH, m=BATCH, x=DIM, y=DIM, low=LOW, high_minus_low=HIGH_MINUS_LOW)
@settings(deadline=None)
def test_dtw_batch_not_symmetric(n: int, m: int, x: int, y: int, low: float, high_minus_low: float) -> None:  # noqa: PLR0913
    """Compare the output of dtw_batch between CPU and GPU implementations, non symmetric case."""
    d = make_tensor((n, m, x, y), dtype=torch.float32, low=low, high=high_minus_low + low)
    sx = make_tensor((n,), dtype=torch.long, low=1, high=x)
    sy = make_tensor((m,), dtype=torch.long, low=1, high=y)
    torch.testing.assert_close(
        dtw_batch(d, sx, sy, symmetric=False),
        dtw_batch(d.cuda(), sx.cuda(), sy.cuda(), symmetric=False).cpu(),
        rtol=rtol,
        atol=atol,
    )
