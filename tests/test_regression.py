"""End-to-end regression test on a small, fully synthetic dataset.

Hardcodes the expected ``Score.collapse()`` value for each distance so that
silent numerical/behavioural changes in the scoring pipeline are caught even
without the large HuBERT features needed by ``test_zerospeech.py``.
"""

import numpy as np
import pytest

from fastabx import Dataset, Score, Task
from fastabx.distance import DistanceName


def _build_dataset(distance: DistanceName) -> Dataset:
    rng = np.random.default_rng(2026)
    n, d = 36, 6
    features = rng.standard_normal((n, d)).astype(np.float32)
    if distance == "kl_symmetric":
        features = np.abs(features) + 0.1
        features = features / features.sum(1, keepdims=True)
    if distance == "identical":
        features = (np.arange(n) % 4).reshape(-1, 1).astype(np.float32)
    phones = ["a", "b", "c"] * 12
    speakers = (["s1"] * 12) + (["s2"] * 12) + (["s3"] * 12)
    contexts = ["c1", "c2"] * 18
    return Dataset.from_numpy(features, {"phone": phones, "speaker": speakers, "context": contexts})


# Computed once on this machine; pinned to detect regressions.
EXPECTED: dict[DistanceName, float] = {
    "euclidean": 0.4913194353381793,
    "cosine": 0.4809027786056201,
    "kl_symmetric": 0.4340277736385663,
    "identical": 0.5,
}


@pytest.mark.parametrize("distance", list(EXPECTED))
def test_regression_collapse(distance: DistanceName) -> None:
    dataset = _build_dataset(distance)
    task = Task(dataset, on="phone", by=["context"], across=["speaker"])
    score = Score(task, distance)
    out = score.collapse(levels=["speaker"])
    # Scoring is fully deterministic on a fixed input; pin the value exactly.
    assert out == EXPECTED[distance]
