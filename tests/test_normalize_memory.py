"""Compare peak CPU memory between old and new normalize_with_singularity_ implementations."""

import math
from collections.abc import Callable

import torch
from torch.profiler import ProfilerActivity, profile
from torch.testing import assert_close

from fastabx.dataset import normalize_with_singularity_


def normalize_with_singularity_old(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Old normalize implementation."""
    norm = torch.norm(x, dim=1, keepdim=True)
    zero_vals = norm == 0
    x = torch.where(zero_vals, 1 / math.sqrt(x.size(1)), x / norm)
    border = torch.full((x.size(0), 1), eps, dtype=x.dtype, device=x.device)
    border = torch.where(zero_vals, -2 * eps, border)
    return torch.cat([x, border], dim=1)


def peak_cpu_memory_bytes[**P, R](fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> tuple[R, float]:
    """Use PyTorch CPU profiler."""
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, acc_events=True) as prof:
        result = fn(*args, **kwargs)
    events = sorted(prof.events(), key=lambda e: e.time_range.start)
    cumulative, peak = 0, 0
    for event in events:
        cumulative += event.cpu_memory_usage
        peak = max(peak, cumulative)
    return result, peak


def test_normalize_memory_cpu() -> None:
    """Test the memory usage and correctness and the new normalization function."""
    torch.manual_seed(0)
    n, d = 100_000, 512
    data = torch.randn(n, d)
    data[::10] = 0  # inject some zero vectors
    result_old, peak_old = peak_cpu_memory_bytes(normalize_with_singularity_old, data.clone())
    result_new, peak_new = peak_cpu_memory_bytes(normalize_with_singularity_, data.clone())
    assert_close(result_old, result_new)
    print(f"\nPeak memory old: {peak_old / 1e6:.1f} MB")
    print(f"Peak memory new: {peak_new / 1e6:.1f} MB")
    print(f"Reduction:       {(1 - peak_new / peak_old) * 100:.1f}%")
    assert peak_new < peak_old
