"""ZeroSpeech ABX evaluation. Reproduces ZeroSpeech 2021."""

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import torch

from fastabx.dataset import Dataset
from fastabx.distance import DistanceName
from fastabx.score import Score
from fastabx.subsample import Subsampler
from fastabx.task import Task


class InvalidSpeakerOrContextError(ValueError):
    """The speaker or context conditions are not set correctly."""


def zerospeech_abx(
    item: str | Path,
    root: str | Path,
    *,
    max_size_group: int | None,
    max_x_across: int | None,
    speaker: Literal["within", "across"] = "within",
    context: Literal["within", "any"] = "within",
    distance: DistanceName = "angular",
    frequency: int = 50,
    feature_maker: Callable[[str | Path], torch.Tensor] = torch.load,
    extension: str = ".pt",
    seed: int = 0,
) -> float:
    """Compute the ABX similarly to the ZeroSpeech 2021 challenge.

    On triphone or phoneme, described by an item file.
    Within or across speaker, and within context or ignoring context.

    :param item: Path to the item file.
    :param root: Path to the root directory containing either the features or the audio files.
    :param max_size_group: Maximum number of instances of A, B, or X in each :py:class:`.Cell`.
        Passed to the :py:class:`.Subsampler` of the :py:class:`.Task`. Set to 10 in the original ZeroSpeech ABX code.
        Disabled if set to ``None``.
    :param max_x_across: In the "across" speaker mode, maximum number of X considered for given values of A and B.
        Passed to the :py:class:`.Subsampler` of the :py:class:`.Task`.
        Set to 5 in the original ZeroSpeech ABX code. Disabled if set to ``None``.
    :param speaker: The speaker mode, either "within" or "across". Defaults to "within".
    :param context: The context mode, either "within" or "any". Always use "within" with representations of triphones.
        Defaults to "within".
    :param distance: The distance metric, "angular" (same as "cosine"), "euclidean", "kl_symmetric" or "identical".
        Defaults to "angular".
    :param frequency: The feature frequency of the features / the output of the feature maker, in Hz.
        Defaults to 50 Hz.
    :param feature_maker: Function that takes a path and returns a torch.Tensor. Defaults to ``torch.load``.
    :param extension: The filename extension of the files to process in ``root``, default is ".pt".
    :param seed: The random seed for the subsampling, default is 0.
    """
    dataset = Dataset.from_item(item, root, frequency, feature_maker=feature_maker, extension=extension)
    by: list[str] | None
    across: list[str] | None
    match (speaker, context):
        case ("within", "within"):
            by, across = ["prev-phone", "next-phone", "speaker"], None
        case ("within", "any"):
            by, across = ["speaker"], None
        case ("across", "within"):
            by, across = ["prev-phone", "next-phone"], ["speaker"]
        case ("across", "any"):
            by, across = None, ["speaker"]
        case _:
            raise InvalidSpeakerOrContextError
    subsampler = Subsampler(max_size_group, max_x_across, seed)
    task = Task(dataset, on="#phone", by=by, across=across, subsampler=subsampler)
    levels = ([("next-phone", "prev-phone")] if context == "within" else []) + ["speaker"]
    return Score(task, distance).collapse(levels=levels)
