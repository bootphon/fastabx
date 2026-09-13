"""Pytest configuration and shared fixtures."""

import numpy as np
import polars as pl
import pytest
import torch
from hypothesis import settings

from fastabx import Dataset
from fastabx.accessor import InMemoryAccessor

# Run more examples than hypothesis's default (100) so adversarial cases (e.g. the cosine
# antipodal boundary in test_distances.py) get explored harder by default. Individual tests
# can still override via their own ``@settings(...)`` decorator.
settings.register_profile("fastabx", max_examples=500, deadline=None)
settings.load_profile("fastabx")


def pytest_addoption(parser: pytest.Parser) -> None:
    """CLI arguments."""
    parser.addoption("--item", action="store", default=None, help="Path to the item file")
    parser.addoption("--features", action="store", default=None, help="Path to the features directory")


def accessor_data(dataset: Dataset) -> torch.Tensor:
    """Concrete feature tensor of a dataset, for the tests that check the in-memory layout.

    ``Dataset.accessor`` is typed as the ``Accessor`` protocol, which has no ``data``: this narrows it back
    to the implementation the constructors actually build.
    """
    assert isinstance(dataset.accessor, InMemoryAccessor)
    return dataset.accessor.data


@pytest.fixture
def tiny_dataset() -> Dataset:
    """Small pooled (time-dim=1) dataset with phone / speaker / context labels.

    Designed so that on=phone has multiple values, by=context groups it, and
    across=speaker has at least two values per (phone, context) group.
    """
    rng = np.random.default_rng(0)
    n, d = 24, 4
    features = rng.standard_normal((n, d)).astype(np.float32)
    phones = ["a", "b", "c"] * 8
    speakers = (["s1"] * 12) + (["s2"] * 12)
    contexts = (["c1", "c2"] * 6) + (["c1", "c2"] * 6)
    labels = {"phone": phones, "speaker": speakers, "context": contexts}
    return Dataset.from_numpy(features, labels)


@pytest.fixture
def seq_dataset() -> Dataset:
    """Dataset with variable time lengths per item (forces needs_alignment=True)."""
    rng = np.random.default_rng(1)
    d = 3
    phones = ["a", "b", "c"] * 6
    speakers = (["s1"] * 9) + (["s2"] * 9)
    lengths = [1 + (i % 3) + 1 for i in range(len(phones))]  # 2..4
    pieces, indices, cursor = [], {}, 0
    for i, length in enumerate(lengths):
        piece = rng.standard_normal((length, d)).astype(np.float32)
        pieces.append(piece)
        indices[i] = (cursor, cursor + length)
        cursor += length
    data = torch.from_numpy(np.concatenate(pieces, axis=0))
    labels = pl.DataFrame({"phone": phones, "speaker": speakers, "context": ["c1", "c2", "c1"] * 6})
    return Dataset(labels=labels, accessor=InMemoryAccessor(indices, data))
