"""Compare peak CPU memory between old and new normalize_with_singularity implementations."""

import math
from collections.abc import Callable

import torch
from torch.profiler import ProfilerActivity, profile
from torch.testing import assert_close

from fastabx.accessor import normalize_with_singularity


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
    result_new, peak_new = peak_cpu_memory_bytes(normalize_with_singularity, data.clone())
    assert_close(result_old, result_new)
    print(f"\nPeak memory old: {peak_old / 1e6:.1f} MB")
    print(f"Peak memory new: {peak_new / 1e6:.1f} MB")
    print(f"Reduction:       {(1 - peak_new / peak_old) * 100:.1f}%")
    assert peak_new < peak_old


def test_normalize_does_not_modify_its_input() -> None:
    """The border forces a new tensor anyway, so the input is left alone rather than normalized in place."""
    torch.manual_seed(0)
    data = torch.randn(64, 8)
    data[::7] = 0  # zero vectors take the singularity branch
    before = data.clone()
    out = normalize_with_singularity(data)
    assert out.shape == (64, 9)
    assert torch.equal(data, before)


def test_score_does_not_modify_a_caller_owned_tensor() -> None:
    """A Dataset built around a tensor the caller still holds must not have it rewritten by Score.

    ``InMemoryAccessor`` deliberately does not copy when the data is already on the target device, so
    before this guarantee the angular distance normalized the caller's own tensor in place.
    """
    import polars as pl

    from fastabx import Dataset, Score, Task
    from fastabx.accessor import InMemoryAccessor

    torch.manual_seed(0)
    data = torch.randn(12, 5)
    before = data.clone()
    dataset = Dataset(
        labels=pl.DataFrame({"phone": ["a", "b"] * 6, "speaker": ["s0", "s0", "s1", "s1"] * 3}),
        accessor=InMemoryAccessor({i: (i, i + 1) for i in range(12)}, data, torch.device("cpu")),
    )
    Score(Task(dataset, on="phone", by=["speaker"]), "angular", progress=False)
    assert torch.equal(data, before)
    # The dataset itself is still normalized in place: its accessor swaps in the wider tensor.
    assert dataset.accessor.is_normalized
