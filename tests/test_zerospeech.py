"""Verify that we reproduce the ABX error rate from the Zerospeech library using features from HuBERT base L11."""

from pathlib import Path
from typing import Literal

import pytest
import torch

from fastabx import zerospeech_abx


@pytest.fixture
def item(request: pytest.FixtureRequest) -> Path:
    """Item file."""
    path = Path(request.config.getoption("--item"))
    if not path.is_file():
        pytest.fail(f"Item file not found: {path}")
    if path.name not in request.config.reference_scores:
        pytest.fail(f"Invalid item, must be one of {set(request.config.reference_scores)})")
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
    pytestconfig: pytest.Config,
    item: Path,
    features: Path,
    speaker: Literal["within", "across"],
    context: Literal["within", "any"],
) -> None:
    """Test reproducibility."""
    if (speaker, context) not in pytestconfig.reference_scores[item.name]:
        pytest.skip(f"Configuration not supported for {item.stem}: {speaker} speaker, {context} context")
    reference = pytestconfig.reference_scores[item.name][(speaker, context)]
    score = zerospeech_abx(
        item,
        features,
        speaker=speaker,
        context=context,
        distance=pytestconfig.distance,
        frequency=pytestconfig.frequency,
        max_size_group=pytestconfig.max_size_group,
        max_x_across=pytestconfig.max_x_across,
        seed=pytestconfig.seed,
    )
    torch.testing.assert_close(score, reference, rtol=0, atol=1e-5)
