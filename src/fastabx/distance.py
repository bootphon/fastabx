"""Distance computation."""

import math
from collections.abc import Callable
from typing import Literal

import torch
from torch import Tensor
from torchdtw import dtw_batch

from fastabx.cell import Cell

__all__ = ["Distance", "DistanceName", "abx_on_cell"]

type Distance = Callable[[Tensor, Tensor], Tensor]
type DistanceName = Literal["euclidean", "cosine", "angular", "kl_symmetric", "identical"]


def distance_function(name: DistanceName) -> Distance:
    """Return the corresponding distance function."""
    match name:
        case "euclidean":
            return euclidean_distance
        case "cosine" | "angular":
            return angular_distance
        case "kl_symmetric":
            return kl_symmetric_distance
        case "identical":
            return identical_distance
        case _:
            raise ValueError(name)


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
    log_p = log_p - log_p.mean(1, keepdim=True)
    log_q = log_q - log_q.mean(1, keepdim=True)
    self_p = (p * log_p).sum(1).unsqueeze(1)
    self_q = (q * log_q).sum(1).unsqueeze(0)
    cross = p @ log_q.T + log_p @ q.T
    return (0.5 * (self_p + self_q - cross)).view(n1, s1, n2, s2).transpose(1, 2)


def angular_distance(a1: Tensor, a2: Tensor) -> Tensor:
    """Angular distance (default). WARNING: a1 and a2 must be normalized."""
    n1, s1, d = a1.size()
    n2, s2, d = a2.size()
    dot_prods = torch.mm(a1.view(n1 * s1, d), a2.view(n2 * s2, d).T).view(n1, s1, n2, s2).transpose(1, 2)
    return torch.clamp(dot_prods, -1, 1).acos() / math.pi


def euclidean_distance(a1: Tensor, a2: Tensor) -> Tensor:
    """Euclidean distance."""
    n1, s1, d = a1.size()
    n2, s2, d = a2.size()
    dist = torch.cdist(a1.view(n1 * s1, d), a2.view(n2 * s2, d), compute_mode="donot_use_mm_for_euclid_dist")
    return dist.view(n1, s1, n2, s2).transpose(1, 2)


def identical_distance(a1: Tensor, a2: Tensor) -> Tensor:
    """0/1 distance. Useful for computing the ABX on discrete speech units."""
    n1, s1, _ = a1.size()
    n2, s2, _ = a2.size()
    return (a1.view(n1, 1, s1, 1) != a2.view(1, n2, 1, s2)).float()


def distance_matrix(
    x: Tensor,
    sx: Tensor,
    y: Tensor,
    sy: Tensor,
    distance: Distance,
    *,
    use_dtw: bool,
    symmetric: bool,
) -> torch.Tensor:
    """Compute the ``(nx, ny)`` distance matrix between all X and all Y, with or without DTW."""
    if use_dtw:
        return dtw_batch(distance(x, y), sx, sy, symmetric=symmetric)
    return distance(x, y).squeeze(2, 3)


def abx_on_cell(cell: Cell, distance_name: DistanceName = "angular") -> torch.Tensor:
    """Compute the ABX of a ``cell`` using the given ``distance``.

    .. warning::
        Unlike :py:class:`.Score`, this low-level helper does **not** normalize the features.
        For the default ``"angular"`` (and ``"cosine"``) distance the cell's features must already
        be L2-normalized (e.g. via :py:meth:`.Dataset.normalize_`); otherwise the dot products are
        only clamped to ``[-1, 1]`` and the score is silently wrong. Likewise ``"kl_symmetric"``
        expects the features to be probability distributions.

    :param cell: The cell to compute the ABX on.
    :param distance_name: The name of the distance to use. Defaults to "angular".
        Must be one of "euclidean", "cosine", "angular", "kl_symmetric", "identical".
    """
    distance = distance_function(distance_name)
    use_dtw, symmetric = cell.use_dtw, cell.is_symmetric
    x, a, b = cell.x, cell.a, cell.b
    dxa = distance_matrix(x.data, x.sizes, a.data, a.sizes, distance, use_dtw=use_dtw, symmetric=symmetric)
    if symmetric:
        dxa.fill_diagonal_(float("inf"))
    dxb = distance_matrix(x.data, x.sizes, b.data, b.sizes, distance, use_dtw=use_dtw, symmetric=False)
    nx, na = dxa.size()
    nx, nb = dxb.size()
    sc = 0.5 * (1 - torch.sign(dxa.view(nx, na, 1) - dxb.view(nx, 1, nb)))
    return 1 - sc.sum() / len(cell)
