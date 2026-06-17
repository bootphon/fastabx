"""Pytest configuration and shared fixtures."""

import numpy as np
import polars as pl
import pytest
import torch

from fastabx import Dataset
from fastabx.dataset import InMemoryAccessor


def pytest_addoption(parser: pytest.Parser) -> None:
    """CLI arguments."""
    parser.addoption("--item", action="store", default=None, help="Path to the item file")
    parser.addoption("--features", action="store", default=None, help="Path to the features directory")


def pytest_configure(config: pytest.Config) -> None:
    """Global configuration."""
    config.reference_scores = {  # HuBERT base L11
        "triphone-dev-clean.item": {
            ("within", "within"): 0.03074,
            ("across", "within"): 0.03777,
        },
        "phoneme-dev-clean.item": {
            ("within", "within"): 0.01579,
            ("across", "within"): 0.02216,
            ("within", "any"): 0.07738,
            ("across", "any"): 0.08357,
        },
    }
    config.distance = "cosine"
    config.max_size_group = 50
    config.max_x_across = 10
    config.seed = 0
    config.frequency = 50


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
    """Dataset with variable time lengths per item (forces use_dtw=True)."""
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
