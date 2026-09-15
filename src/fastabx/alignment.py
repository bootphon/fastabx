"""Sequence alignment: collapse a frame-level distance lattice into one distance per pair of sequences.

A :py:class:`.Distance` compares individual frames and produces a ``(n1, n2, s1, s2)`` distance lattice.
An :py:class:`.Alignment` turns that lattice into the ``(n1, n2)`` distance between the sequences themselves.
Dynamic time warping is the only alignment fastabx ships, but other alignments following the protocol can be used.
"""

from typing import Literal, Protocol

from torch import Tensor
from torchdtw import dtw_batch

__all__ = ["Alignment", "AlignmentName"]

type AlignmentName = Literal["dtw"]


class Alignment(Protocol):
    """Reduce a frame-level distance lattice to one distance per pair of sequences.

    Implementations must return the distance **normalized by the length of the alignment path**, so that pairs of
    different lengths stay comparable: the ABX decision compares a X-to-A distance against a X-to-B distance,
    and an unnormalized distance would bias it towards the shorter pair.

    Only the ``(sx[i], sy[j])`` sub-block of each pair may be read; anything beyond those lengths is padding.
    """

    def __call__(self, distances: Tensor, sx: Tensor, sy: Tensor, /, *, symmetric: bool) -> Tensor:
        """Align every pair of sequences.

        The three tensors are positional-only: an implementation is free to name them whatever suits it,
        but ``symmetric`` is passed by keyword and must keep its name.

        :param distances: The ``(n1, n2, s1, s2)`` frame-level distance lattice.
        :param sx: The ``(n1,)`` real lengths of the first batch.
        :param sy: The ``(n2,)`` real lengths of the second batch.
        :param symmetric: Whether the two batches are the same set.
        :returns: A ``(n1, n2)`` tensor of sequence distances.
        """
        ...


def alignment_function(alignment: AlignmentName | Alignment) -> Alignment:
    """Return the corresponding alignment function, or pass a custom one through unchanged.

    :param alignment: Either the name of a built-in alignment, or any callable satisfying
        the :py:class:`.Alignment` protocol.
    """
    if not isinstance(alignment, str):
        if not callable(alignment):
            msg = "alignment must be a built-in name or a callable"
            raise TypeError(msg)
        return alignment
    match alignment:
        case "dtw":
            return dtw_batch
        case _:
            msg = f"Unknown alignment: {alignment!r}. Choose dtw, or pass a callable."
            raise ValueError(msg)
