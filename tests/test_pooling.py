"""Tests for ``fastabx.pooling``."""

import numpy as np
import pytest
import torch

from fastabx import Dataset
from fastabx.dataset import InMemoryAccessor
from fastabx.pooling import PooledDataset, hamming_pooling, pool_dataset, pooling_function


def test_pooling_function_mean_and_hamming() -> None:
    x = torch.arange(12, dtype=torch.float32).view(4, 3)
    mean_fn = pooling_function("mean")
    out_mean = mean_fn(x)
    assert out_mean.shape == (3,)
    torch.testing.assert_close(out_mean, x.mean(dim=0))

    hamming_fn = pooling_function("hamming")
    out_hamming = hamming_fn(x)
    assert out_hamming.shape == (3,)


def test_pooling_function_unknown_raises() -> None:
    with pytest.raises(ValueError, match="bogus"):
        pooling_function("bogus")  # ty: ignore[invalid-argument-type]


def test_hamming_window_matches_manual() -> None:
    x = torch.arange(8, dtype=torch.float32).view(4, 2)
    window = torch.hamming_window(x.size(0))
    expected = (window @ x) / window.sum()
    torch.testing.assert_close(hamming_pooling(x), expected)


def test_pooling_returns_pooled_dataset() -> None:
    rng = np.random.default_rng(0)
    n, length, d = 4, 3, 5
    data = torch.from_numpy(rng.standard_normal((n * length, d)).astype(np.float32))
    indices = {i: (i * length, (i + 1) * length) for i in range(n)}
    import polars as pl

    dataset = Dataset(
        labels=pl.DataFrame({"phone": ["a", "b", "c", "d"]}),
        accessor=InMemoryAccessor(indices, data),
    )
    pooled = pool_dataset(dataset, "mean")
    assert isinstance(pooled, PooledDataset)
    assert pooled.pooling == "mean"
    # Each item now has time dim 1.
    for item in pooled.accessor:
        assert item.shape[0] == 1
    assert "mean" in repr(pooled)


def test_pooling_mean_of_constant_sequence() -> None:
    n, length, d = 2, 4, 3
    data = torch.ones(n * length, d)
    indices = {i: (i * length, (i + 1) * length) for i in range(n)}
    import polars as pl

    dataset = Dataset(
        labels=pl.DataFrame({"phone": ["a", "b"]}),
        accessor=InMemoryAccessor(indices, data),
    )
    pooled = pool_dataset(dataset, "mean")
    for item in pooled.accessor:
        torch.testing.assert_close(item.squeeze(0), torch.ones(d))
