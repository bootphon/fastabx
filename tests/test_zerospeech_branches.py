"""Cover the speaker/context branches of ``zerospeech_abx`` without needing real features."""

from pathlib import Path
from typing import Literal

import pytest
import torch

from fastabx import zerospeech_abx
from fastabx.zerospeech import InvalidSpeakerOrContextError, MissingMaxXAcrossError


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


def test_zerospeech_abx_max_x_across_required_across(tmp_path: Path) -> None:
    """In "across" mode, omitting ``max_x_across`` is an error; ``None`` explicitly disables it."""
    item, feats = _build_tiny_corpus(tmp_path)
    with pytest.raises(MissingMaxXAcrossError):
        # The overloads make this a type error; the runtime guard is for callers passing `speaker` dynamically.
        zerospeech_abx(item, feats, max_size_group=None, speaker="across", distance="euclidean")  # ty: ignore[invalid-argument-type]
    score = zerospeech_abx(item, feats, max_size_group=None, max_x_across=None, speaker="across", distance="euclidean")
    assert 0.0 <= score <= 1.0


def test_zerospeech_abx_max_x_across_optional_within(tmp_path: Path) -> None:
    """In "within" mode, ``max_x_across`` can be omitted entirely (it is ignored)."""
    item, feats = _build_tiny_corpus(tmp_path)
    score = zerospeech_abx(item, feats, max_size_group=None, speaker="within", distance="euclidean")
    assert 0.0 <= score <= 1.0


def test_zerospeech_abx_invalid_speaker_context(tmp_path: Path) -> None:
    item, feats = _build_tiny_corpus(tmp_path)
    with pytest.raises(InvalidSpeakerOrContextError):
        # No overload accepts an unknown speaker mode: this checks the runtime guard behind them.
        zerospeech_abx(  # ty: ignore[no-matching-overload]
            item,
            feats,
            max_size_group=None,
            max_x_across=None,
            speaker="bogus",
            context="within",
            distance="euclidean",
            frequency=50,
        )


def test_zerospeech_abx_validates_subsampler_before_loading(tmp_path: Path) -> None:
    """A bad ``max_size_group`` must fail before the features are read, not after.

    Loading a real corpus takes minutes; pointing at a directory with no features at all means the
    only way this raises ``TypeError`` rather than ``FileNotFoundError`` is if the subsampler is
    validated first.
    """
    item, _ = _build_tiny_corpus(tmp_path)
    empty = tmp_path / "no-features"
    empty.mkdir()
    with pytest.raises(TypeError, match="sizes should be integers >= 2"):
        zerospeech_abx(item, empty, max_size_group=1, max_x_across=None, distance="euclidean")


def test_zerospeech_abx_progress_can_be_disabled(tmp_path: Path) -> None:
    item, feats = _build_tiny_corpus(tmp_path)
    score = zerospeech_abx(item, feats, max_size_group=None, max_x_across=None, distance="euclidean", progress=False)
    assert 0.0 <= score <= 1.0


def test_zerospeech_abx_accepts_decimal_frequency(tmp_path: Path) -> None:
    """A non-integer frequency has to be reachable through the ZeroSpeech helper too."""
    item, feats = _build_tiny_corpus(tmp_path)
    score = zerospeech_abx(item, feats, max_size_group=None, max_x_across=None, distance="euclidean", frequency="50.0")
    assert 0.0 <= score <= 1.0
