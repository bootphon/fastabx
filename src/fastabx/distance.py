"""Distance computation."""

import math
from collections.abc import Callable
from typing import Literal

import torch
from torch import Tensor
from torchdtw import dtw_batch

from fastabx.alignment import Alignment
from fastabx.cell import Cell

__all__ = ["Distance", "DistanceName", "IdenticalDistanceDimensionError", "abx_on_cell"]

type Distance = Callable[[Tensor, Tensor], Tensor]
type DistanceName = Literal["euclidean", "cosine", "angular", "kl_symmetric", "identical"]


def distance_function(distance: DistanceName | Distance) -> Distance:
    """Return the corresponding distance function, or pass a custom one through unchanged.

    :param distance: Either the name of a built-in distance, or any callable satisfying
        the :py:class:`.Distance` type alias.
    """
    if not isinstance(distance, str):
        if not callable(distance):
            msg = "distance must be a built-in name or a callable"
            raise TypeError(msg)
        return distance
    match distance:
        case "euclidean":
            return euclidean_distance
        case "cosine" | "angular":
            return angular_distance
        case "kl_symmetric":
            return kl_symmetric_distance
        case "identical":
            return identical_distance
        case _:
            raise ValueError(distance)


def kl_symmetric_distance(a1: Tensor, a2: Tensor, epsilon: float = 1e-6) -> Tensor:
    """KL symmetric distance. The two tensors must correspond to probability distributions.

    Each row's log is centred to mean zero before the matmuls: equivalent for probability
    distributions and better precision near 0 in float32.
    """
    n1, s1, d = a1.size()
    n2, s2, d = a2.size()
    p, q = a1.view(n1 * s1, d), a2.view(n2 * s2, d)
    log_p = (p + epsilon).log()
    log_q = (q + epsilon).log()
    log_p -= log_p.mean(1, keepdim=True)
    log_q -= log_q.mean(1, keepdim=True)
    self_p = (p * log_p).sum(1).unsqueeze(1)
    self_q = (q * log_q).sum(1).unsqueeze(0)
    cross = p @ log_q.T + log_p @ q.T
    return (0.5 * (self_p + self_q - cross)).view(n1, s1, n2, s2).transpose(1, 2)


def angular_distance(a1: Tensor, a2: Tensor) -> Tensor:
    """Angular distance (default). WARNING: a1 and a2 must be normalized."""
    n1, s1, d = a1.size()
    n2, s2, d = a2.size()
    dot_prods = torch.mm(a1.view(n1 * s1, d), a2.view(n2 * s2, d).T).view(n1, s1, n2, s2).transpose(1, 2)
    return dot_prods.clamp_(-1, 1).acos_().div_(math.pi)


def euclidean_distance(a1: Tensor, a2: Tensor) -> Tensor:
    """Euclidean distance."""
    n1, s1, d = a1.size()
    n2, s2, d = a2.size()
    dist = torch.cdist(a1.view(n1 * s1, d), a2.view(n2 * s2, d), compute_mode="donot_use_mm_for_euclid_dist")
    return dist.view(n1, s1, n2, s2).transpose(1, 2)


class IdenticalDistanceDimensionError(ValueError):
    """The "identical" distance got features with more than one dimension."""

    def __init__(self, dim: int) -> None:
        super().__init__(
            f"The 'identical' distance compares discrete units, so the features must have a single "
            f"dimension of shape (length, 1), but they have {dim}. Either encode each unit as one "
            f"integer, or use a distance defined on vectors ('euclidean', 'angular', 'kl_symmetric')."
        )


def identical_distance(a1: Tensor, a2: Tensor) -> Tensor:
    """0/1 distance. Useful for computing the ABX on discrete speech units."""
    n1, s1, d = a1.size()
    n2, s2, _ = a2.size()
    if d != 1:
        raise IdenticalDistanceDimensionError(d)
    return (a1.view(n1, 1, s1, 1) != a2.view(1, n2, 1, s2)).float()


def distance_matrix(
    x: Tensor,
    sx: Tensor,
    y: Tensor,
    sy: Tensor,
    distance: Distance,
    *,
    alignment: Alignment,
    symmetric: bool,
) -> torch.Tensor:
    """Compute the ``(nx, ny)`` distance matrix between all X and all Y.

    ``distance`` builds the frame-level cost lattice and ``alignment`` reduces it to one distance per pair.
    """
    cost = distance(x, y)
    if cost.size(2) == 1 and cost.size(3) == 1:
        return cost.squeeze(2, 3)
    return alignment(cost, sx, sy, symmetric=symmetric)


def abx_on_cell(
    cell: Cell,
    distance_name: DistanceName | Distance = "angular",
    *,
    alignment: Alignment = dtw_batch,
) -> torch.Tensor:
    """Compute the ABX of a ``cell`` using the given ``distance``.

    Returns the ABX error rate (1 - discriminability) of the cell, as a scalar tensor.

    .. warning::
        Unlike :py:class:`.Score`, this low-level helper does **not** normalize the features.
        For the default ``"angular"`` (and ``"cosine"``) distance the cell's features must already
        be L2-normalized (e.g. via :py:meth:`.Dataset.normalize_`); otherwise the dot products are
        only clamped to ``[-1, 1]`` and the score is silently wrong. Likewise ``"kl_symmetric"``
        expects the features to be probability distributions.

    :param cell: The cell to compute the ABX on.
    :param distance_name: The distance to use, either the name of a built-in one ("euclidean", "cosine",
        "angular", "kl_symmetric", "identical") or a custom :py:class:`.Distance` callable.
        Defaults to "angular".
    :param alignment: How to align sequences that span several frames, as an :py:class:`.Alignment` callable.
        Defaults to ``torchdtw.dtw_batch``. Never called on the distance matrices whose lattice is ``1x1``.
    """
    distance = distance_function(distance_name)
    symmetric = cell.is_symmetric
    x, a, b = cell.x, cell.a, cell.b
    dxa = distance_matrix(x.data, x.sizes, a.data, a.sizes, distance, alignment=alignment, symmetric=symmetric)
    if symmetric:
        dxa.fill_diagonal_(float("inf"))
    dxb = distance_matrix(x.data, x.sizes, b.data, b.sizes, distance, alignment=alignment, symmetric=False)
    nx, na = dxa.size()
    nx, nb = dxb.size()
    sc = 0.5 * (1 - torch.sign(dxa.view(nx, na, 1) - dxb.view(nx, 1, nb)))
    return (1 - sc.sum(dtype=torch.float64) / len(cell)).to(sc.dtype)
