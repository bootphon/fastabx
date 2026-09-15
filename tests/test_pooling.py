"""Tests for ``fastabx.pooling``."""

from pathlib import Path

import numpy as np
import pytest
import torch

from fastabx import Dataset, Score, Task
from fastabx.accessor import InMemoryAccessor
from fastabx.pooling import (
    PooledDataset,
    PoolingName,
    PoolingNormalizedError,
    hamming_pooling,
    pool_dataset,
    pooling_function,
)
from tests.conftest import DEVICE


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
    window = torch.hamming_window(x.size(0), periodic=False)
    expected = (window @ x) / window.sum()
    torch.testing.assert_close(hamming_pooling(x), expected)


def test_hamming_pooling_is_symmetric() -> None:
    """The window must weight the first and the last frame equally.

    The periodic window (the default of ``torch.hamming_window``) does not: it attenuates the leading
    boundary but leaves the trailing one nearly untouched, which would make the pooled vector depend on
    the direction of the sequence.
    """
    for length in (2, 3, 4, 5, 8):
        x = torch.zeros(length, 1)
        x[0, 0] = 1.0
        torch.testing.assert_close(hamming_pooling(x), hamming_pooling(x.flip(0)))


def test_hamming_pooling_downweights_both_boundaries() -> None:
    """A frame at either end must count for less than one in the middle."""
    length = 5
    pooled = []
    for position in range(length):
        x = torch.zeros(length, 1)
        x[position, 0] = 1.0
        pooled.append(hamming_pooling(x).item())
    assert pooled[0] < pooled[1] < pooled[2]
    assert pooled[-1] < pooled[-2] < pooled[2]


def test_pooling_returns_pooled_dataset() -> None:
    rng = np.random.default_rng(0)
    n, length, d = 4, 3, 5
    data = torch.from_numpy(rng.standard_normal((n * length, d)).astype(np.float32))
    indices = {i: (i * length, (i + 1) * length) for i in range(n)}
    import polars as pl

    dataset = Dataset(
        labels=pl.DataFrame({"phone": ["a", "b", "c", "d"]}),
        accessor=InMemoryAccessor(indices, data, DEVICE),
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
        accessor=InMemoryAccessor(indices, data, DEVICE),
    )
    pooled = pool_dataset(dataset, "mean")
    for item in pooled.accessor:
        torch.testing.assert_close(item.squeeze(0), torch.ones(d))


def test_pool_dataset_rejects_normalized_dataset(tiny_dataset: Dataset) -> None:
    """Pooling after an angular Score is refused: the new accessor would lose ``is_normalized``."""
    Score(Task(tiny_dataset, on="phone", by=["speaker"]), "angular")
    assert tiny_dataset.accessor.is_normalized
    with pytest.raises(PoolingNormalizedError):
        pool_dataset(tiny_dataset, "mean")


@pytest.mark.parametrize("files", [("a", "a", "z"), ("z", "z", "a"), ("z", "a", "z")])
@pytest.mark.parametrize("pooling", ["mean", "hamming"])
def test_timestamp_loading_pooling_preserves_label_order(
    tmp_path: Path, files: tuple[str, ...], pooling: PoolingName
) -> None:
    """File sorting during loading must not change which features belong to a label row."""
    item = tmp_path / "data.item"
    item.write_text("#file onset offset phone\n" + "".join(f"{file} 0.0 1.0 p{i}\n" for i, file in enumerate(files)))
    features, times = tmp_path / "features", tmp_path / "times"
    features.mkdir()
    times.mkdir()
    for file, value in (("a", 1.0), ("z", 9.0)):
        torch.save(torch.full((2, 1), value), features / f"{file}.pt")
        torch.save(torch.tensor([0.0, 1.0]), times / f"{file}.pt")
    dataset = Dataset.from_item_with_times(item, features, times, device=DEVICE, progress=False)
    pooled = pool_dataset(dataset, pooling)
    assert pooled.labels.equals(dataset.labels)
    for i, file in enumerate(files):
        expected = torch.tensor([[1.0 if file == "a" else 9.0]], device=DEVICE)
        torch.testing.assert_close(pooled.accessor[i], expected)
        torch.testing.assert_close(list(dataset.accessor)[i], dataset.accessor[i])
