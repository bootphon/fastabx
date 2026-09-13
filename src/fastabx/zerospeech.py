"""ZeroSpeech ABX evaluation. Reproduces ZeroSpeech 2021."""

from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Literal, overload

import torch

from fastabx.dataset import Dataset
from fastabx.distance import DistanceName
from fastabx.score import Score
from fastabx.subsample import Subsampler
from fastabx.task import Task

__all__ = ["InvalidSpeakerOrContextError", "MissingMaxXAcrossError", "zerospeech_abx"]


class InvalidSpeakerOrContextError(ValueError):
    """The speaker or context conditions are not set correctly."""


class MissingMaxXAcrossError(ValueError):
    """``max_x_across`` must be set in the "across" speaker mode."""

    def __init__(self) -> None:
        super().__init__("max_x_across is required when speaker='across' (pass None explicitly to disable it).")


_UNSET = object()


@overload
def zerospeech_abx(
    item: str | Path,
    root: str | Path,
    *,
    max_size_group: int | None,
    max_x_across: int | None = ...,
    speaker: Literal["within"] = ...,
    context: Literal["within", "any"] = ...,
    distance: DistanceName = ...,
    frequency: int | str | Decimal = ...,
    feature_maker: Callable[[str | Path], torch.Tensor] = ...,
    extension: str = ...,
    seed: int = ...,
    device: str | torch.device | None = ...,
    write_csv: str | Path | None = ...,
    progress: bool = ...,
) -> float: ...


@overload
def zerospeech_abx(
    item: str | Path,
    root: str | Path,
    *,
    max_size_group: int | None,
    max_x_across: int | None,
    speaker: Literal["across"],
    context: Literal["within", "any"] = ...,
    distance: DistanceName = ...,
    frequency: int | str | Decimal = ...,
    feature_maker: Callable[[str | Path], torch.Tensor] = ...,
    extension: str = ...,
    seed: int = ...,
    device: str | torch.device | None = ...,
    write_csv: str | Path | None = ...,
    progress: bool = ...,
) -> float: ...


def zerospeech_abx(
    item: str | Path,
    root: str | Path,
    *,
    max_size_group: int | None,
    max_x_across: int | None = _UNSET,  # ty: ignore[invalid-parameter-default]
    speaker: Literal["within", "across"] = "within",
    context: Literal["within", "any"] = "within",
    distance: DistanceName = "angular",
    frequency: int | str | Decimal = 50,
    feature_maker: Callable[[str | Path], torch.Tensor] = torch.load,
    extension: str = ".pt",
    seed: int = 0,
    device: str | torch.device | None = None,
    write_csv: str | Path | None = None,
    progress: bool = True,
) -> float:
    """Compute the ABX similarly to the ZeroSpeech 2021 challenge.

    On triphone or phoneme, described by an item file.
    Within or across speaker, and within context or ignoring context.

    The item file must have the ZeroSpeech columns: ``#phone``, ``prev-phone``, ``next-phone`` and ``speaker``.
    See :doc:`/items`, and :ref:`item-downloads` to get the ZeroSpeech item files.

    Returns the **ABX error rate** (1 - discriminability), between 0 and 1: lower is better, and chance level is 0.5.

    :param item: Path to the item file.
    :param root: Path to the root directory containing either the features or the audio files.
    :param max_size_group: Maximum number of instances of A, B, or X in each :py:class:`.Cell`.
        Passed to the :py:class:`.Subsampler` of the :py:class:`.Task`. Set to 10 in the original ZeroSpeech ABX code.
        Required; disabled if set to ``None``.
    :param max_x_across: In the "across" speaker mode, maximum number of X considered for given values of A and B.
        Passed to the :py:class:`.Subsampler` of the :py:class:`.Task`.
        Set to 5 in the original ZeroSpeech ABX code. Required when ``speaker="across"`` (pass ``None`` explicitly to
        disable it); ignored otherwise.
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
    :param device: Device on which to store the features, such as "cpu" or "cuda:1".
        Defaults to CUDA if available, and CPU otherwise.
    :param write_csv: Optional path to a CSV file where the score of every :py:class:`.Cell` is written,
        as done by :py:meth:`.Score.write_csv`.
    :param progress: Whether to display the progress bars while building the dataset and scoring the cells.
    """
    if speaker == "across" and max_x_across is _UNSET:
        raise MissingMaxXAcrossError
    if max_x_across is _UNSET:
        max_x_across = None
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
    dataset = Dataset.from_item(
        item,
        root,
        frequency,
        feature_maker=feature_maker,
        extension=extension,
        device=device,
        progress=progress,
    )
    task = Task(dataset, on="#phone", by=by, across=across, subsampler=subsampler)
    levels = ([("next-phone", "prev-phone")] if context == "within" else []) + ["speaker"]
    score = Score(task, distance, progress=progress)
    if write_csv is not None:
        score.write_csv(write_csv)
    return score.collapse(levels=levels)
