"""Verify that we reproduce the ABX error rate from the Zerospeech library using features from HuBERT base L11."""

from pathlib import Path
from typing import Literal

import pytest
import torch

from fastabx import zerospeech_abx
from fastabx.distance import DistanceName

DISTANCE: DistanceName = "cosine"
MAX_SIZE_GROUP = 50
MAX_X_ACROSS = 10
SEED = 0
FREQUENCY = 50
REFERENCE_SCORES = {
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


@pytest.fixture
def item(request: pytest.FixtureRequest) -> Path:
    """Item file."""
    path = Path(request.config.getoption("--item"))
    if not path.is_file():
        pytest.fail(f"Item file not found: {path}")
    if path.name not in REFERENCE_SCORES:
        pytest.fail(f"Invalid item, must be one of {set(REFERENCE_SCORES)})")
    return path


@pytest.fixture
def features(request: pytest.FixtureRequest) -> Path:
    """Features directory."""
    path = Path(request.config.getoption("--features"))
    if not path.is_dir():
        pytest.fail(f"Features directory not found: {path}")
    return path


@pytest.mark.skipif("not config.getoption('item') or not config.getoption('features')")
@pytest.mark.parametrize("speaker", ["within", "across"])
@pytest.mark.parametrize("context", ["within", "any"])
def test_zerospeech(
    item: Path,
    features: Path,
    speaker: Literal["within", "across"],
    context: Literal["within", "any"],
) -> None:
    """Test reproducibility."""
    if (speaker, context) not in REFERENCE_SCORES[item.name]:
        pytest.skip(f"Configuration not supported for {item.stem}: {speaker} speaker, {context} context")
    reference = REFERENCE_SCORES[item.name][speaker, context]
    score = zerospeech_abx(
        item,
        features,
        max_size_group=MAX_SIZE_GROUP,
        max_x_across=MAX_X_ACROSS,
        speaker=speaker,
        context=context,
        distance=DISTANCE,
        frequency=FREQUENCY,
        seed=SEED,
    )
    torch.testing.assert_close(score, reference, rtol=0, atol=1e-5)
