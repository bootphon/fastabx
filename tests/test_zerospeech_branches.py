"""Cover the speaker/context branches of ``zerospeech_abx`` without needing real features."""

from pathlib import Path
from typing import Literal

import pytest
import torch

from fastabx import zerospeech_abx
from fastabx.zerospeech import InvalidSpeakerOrContextError


def _build_tiny_corpus(tmp_path: Path) -> tuple[Path, Path]:
    """Tiny synthetic item file + features dir, large enough to span all (speaker, context) modes."""
    item = tmp_path / "data.item"
    rows = ["#file onset offset #phone prev-phone next-phone speaker"]
    # Two files (speakers), each repeating phones a/b/c twice in two prev/next contexts.
    times = [(i * 0.10, (i + 1) * 0.10) for i in range(12)]
    phones = ["a", "b", "c", "a", "b", "c"] * 2
    prev_phones = (["p1"] * 6) + (["p2"] * 6)
    next_phones = (["n1"] * 6) + (["n2"] * 6)
    for spk, fname in [("s1", "f1"), ("s2", "f2")]:
        for (onset, offset), ph, pp, npph in zip(times, phones, prev_phones, next_phones, strict=True):
            rows.append(f"{fname} {onset:.2f} {offset:.2f} {ph} {pp} {npph} {spk}")
    item.write_text("\n".join(rows) + "\n")

    feats = tmp_path / "feats"
    feats.mkdir()
    torch.manual_seed(0)
    for fname in ("f1", "f2"):
        torch.save(torch.randn(80, 4), feats / f"{fname}.pt")
    return item, feats


@pytest.mark.parametrize(
    ("speaker", "context"),
    [("within", "within"), ("within", "any"), ("across", "within"), ("across", "any")],
)
def test_zerospeech_abx_all_modes(
    tmp_path: Path, speaker: Literal["within", "across"], context: Literal["within", "any"]
) -> None:
    item, feats = _build_tiny_corpus(tmp_path)
    score = zerospeech_abx(
        item,
        feats,
        max_size_group=None,
        max_x_across=5 if speaker == "across" else None,
        speaker=speaker,
        context=context,
        distance="euclidean",
        frequency=50,
        seed=0,
    )
    assert 0.0 <= score <= 1.0


def test_zerospeech_abx_invalid_speaker_context(tmp_path: Path) -> None:
    item, feats = _build_tiny_corpus(tmp_path)
    with pytest.raises(InvalidSpeakerOrContextError):
        zerospeech_abx(
            item,
            feats,
            max_size_group=None,
            max_x_across=None,
            speaker="bogus",  # ty: ignore[invalid-argument-type]
            context="within",
            distance="euclidean",
            frequency=50,
        )
