"""ZeroSpeech ABX evaluation. Reproduces ZeroSpeech 2021."""

from pathlib import Path
from typing import Literal

import torch

from fastabx.dataset import Dataset, FeatureMaker
from fastabx.distance import DistanceName
from fastabx.score import Score
from fastabx.subsample import Subsampler
from fastabx.task import Task


class InvalidSpeakerOrContextError(ValueError):
    """The speaker or context conditions are not set correctly."""


def zerospeech_abx(  # noqa: PLR0913
    item: str | Path,
    root: str | Path,
    *,
    speaker: Literal["within", "across"] = "within",
    context: Literal["within", "any"] = "within",
    distance: DistanceName = "cosine",
    frequency: int = 50,
    feature_maker: FeatureMaker = torch.load,
    max_size_group: int | None = 10,
    max_x_across: int | None = 5,
    extension: str = ".pt",
    seed: int = 0,
) -> float:
    """Compute the ABX similarly to the ZeroSpeech 2021 challenge.

    On triphone or phoneme, described by an item file.
    Within or across speaker, and within context or ignoring context.

    :param item: the item file
    :param root: the root directory containing either the features or the audio files
    :param speaker: the speaker mode, either "within" or "across"
    :param context: the context mode, either "within" or "any"
    :param distance: the distance metric, either "cosine", "euclidean", "kl_symmetric" or "identical"
    :param frequency: the feature frequency of the features / the output of the feature maker, in Hz. Default is 50 Hz
    :param feature_maker: the feature maker. Defaults to just loading the file with ``torch.load``
    :param max_size_group: maximum number of instances of A, B, or X in each :py:class:`.Cell`. Default is 10.
        Passed to the :py:class:`.Subsampler` of the :py:class:`.Task`. Disabled if set to ``None``
    :param max_x_across: in the "across" speaker mode, maximum number of X considered for given values of A and B.
        Default is 5. Passed to the :py:class:`.Subsampler` of the :py:class:`.Task`. Disabled if set to ``None``
    :param extension: the filename extension of the files to process in ``root``, default is ".pt"
    :param seed: the random seed for the subsampling, default is 0
    """
    dataset = Dataset.from_item(item, root, frequency, feature_maker, extension=extension)
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
