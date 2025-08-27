"""Distance computation."""

import math
from collections.abc import Callable
from typing import Literal, get_args

import torch
from torch import Tensor

from fastabx.cell import Cell
from fastabx.dtw import dtw_batch

type Distance = Callable[[Tensor, Tensor], Tensor]
type DistanceName = Literal["euclidean", "cosine", "angular", "kl", "kl_symmetric", "identical", "null"]


def available_distances() -> tuple[str, ...]:
    """Names of the available distances."""
    return get_args(DistanceName.__value__)


def distance_function(name: DistanceName) -> Distance:
    """Return the corresponding distance function."""
    match name:
        case "euclidean":
            return euclidean_distance
        case "cosine" | "angular":
            return angular_distance
        case "kl":
            return kl_distance
        case "kl_symmetric":
            return kl_symmetric_distance
        case "identical":
            return identical_distance
        case "null":
            return null_distance
        case _:
            raise ValueError(name)


def null_distance(a1: Tensor, a2: Tensor) -> Tensor:
    """Null distance, useful for debugging."""
    n1, s1, _ = a1.size()
    n2, s2, _ = a2.size()
    return torch.zeros(n1, n2, s1, s2, device=a1.device)


def kl_distance(a1: Tensor, a2: Tensor, epsilon: float = 1e-6) -> Tensor:
    """KL distance. You might want to use kl_symmetric_distance in most cases."""
    n1, s1, d = a1.size()
    n2, s2, d = a2.size()
    a1_view = a1.view(n1 * s1, 1, d)
    a2_view = a2.view(1, n2 * s2, d)
    div = (a1_view + epsilon) / (a2_view + epsilon)
    return torch.sum(a1_view * div.log(), dim=2).view(n1, s1, n2, s2).transpose(1, 2)


def kl_symmetric_distance(a1: Tensor, a2: Tensor, epsilon: float = 1e-6) -> Tensor:
    """KL symmetric distance. The two tensors must correspond to probability distributions."""
    n1, s1, d = a1.size()
    n2, s2, d = a2.size()
    a1_view = a1.view(n1 * s1, 1, d)
    a2_view = a2.view(1, n2 * s2, d)
    div1 = (a1_view + epsilon) / (a2_view + epsilon)
    div2 = (a2_view + epsilon) / (a1_view + epsilon)
    out1 = torch.sum(a1_view * div1.log(), dim=2)
    out2 = torch.sum(a2_view * div2.log(), dim=2)
    return (0.5 * out1 + 0.5 * out2).view(n1, s1, n2, s2).transpose(1, 2)


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


def distance_on_cell(cell: Cell, distance: Distance) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the distance matrices between all A and X, and all B and X in the ``cell``, for a given ``distance``.

    :param cell: The cell to compute the distances on.
    :param distance: The distance function to use. It takes two tensors of shape
        (n1, s1, d) and (n2, s2, d) and returns a tensor of shape (n1, n2, s1, s2).
    """
    (a, sa), (b, sb), (x, sx) = (cell.a.data, cell.a.sizes), (cell.b.data, cell.b.sizes), (cell.x.data, cell.x.sizes)
    if cell.use_dtw:
        dxa = dtw_batch(distance(x, a), sx, sa, symmetric=cell.is_symmetric)
        dxb = dtw_batch(distance(x, b), sx, sb, symmetric=False)
    else:
        dxa, dxb = distance(x, a).squeeze(2, 3), distance(x, b).squeeze(2, 3)
    return dxa, dxb


def abx_on_cell(cell: Cell, distance: Distance, *, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Compute the ABX of a ``cell`` using the given ``distance``.

    :param cell: The cell to compute the ABX on.
    :param distance: The distance function to use. It takes two tensors of shape
        (n1, s1, d) and (n2, s2, d) and returns a tensor of shape (n1, n2, s1, s2).
    :param mask: Optional boolean mask of shape (nx, na, nb) to select which triplets to include in the score.
    """
    dxa, dxb = distance_on_cell(cell, distance)
    if cell.is_symmetric:
        dxa.fill_diagonal_(float("inf"))
    nx, na = dxa.size()
    nx, nb = dxb.size()
    dxb = dxb.view(nx, 1, nb).expand(nx, na, nb)
    dxa = dxa.view(nx, na, 1).expand(nx, na, nb)
    if mask is None:
        sc = (dxa < dxb).sum() + 0.5 * (dxa == dxb).sum()
        sc /= len(cell)
    else:
        sc = (dxa < dxb)[mask].sum() + 0.5 * (dxa == dxb)[mask].sum()
        sc /= mask.sum()
    return 1 - sc
