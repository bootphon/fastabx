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
            return cosine_distance
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
    div = (a1.view(n1, 1, s1, 1, d) + epsilon) / (a2.view(1, n2, 1, s2, d) + epsilon)
    prod = (a1.view(n1, 1, s1, 1, d)) * div.log()
    return prod.sum(dim=4)


def kl_symmetric_distance(a1: Tensor, a2: Tensor, epsilon: float = 1e-6) -> Tensor:
    """KL symmetric distance. The two tensors must correspond to probability distributions."""
    n1, s1, d = a1.size()
    n2, s2, d = a2.size()
    div1 = (a1.view(n1, 1, s1, 1, d) + epsilon) / (a2.view(1, n2, 1, s2, d) + epsilon)
    div2 = (a2.view(1, n2, 1, s2, d) + epsilon) / (a1.view(n1, 1, s1, 1, d) + epsilon)
    prod1 = (a1.view(n1, 1, s1, 1, d)) * div1.log()
    prod2 = (a2.view(1, n2, 1, s2, d)) * div2.log()
    return (0.5 * prod1 + 0.5 * prod2).sum(dim=4)


def cosine_distance(a1: Tensor, a2: Tensor) -> Tensor:
    """Angular distance (default). WARNING: a1 and a2 must be normalized."""
    n1, s1, d = a1.size()
    n2, s2, d = a2.size()
    prod = (a1.view(n1, 1, s1, 1, d)) * (a2.view(1, n2, 1, s2, d))
    return torch.clamp(prod.sum(dim=4), -1, 1).acos() / math.pi


def euclidean_distance(a1: Tensor, a2: Tensor) -> Tensor:
    """Euclidean distance."""
    n1, s1, d = a1.size()
    n2, s2, d = a2.size()
    diff = a1.view(n1, 1, s1, 1, d) - a2.view(1, n2, 1, s2, d)
    return torch.sqrt((diff**2).sum(dim=4))


def identical_distance(a1: Tensor, a2: Tensor) -> Tensor:
    """0/1 distance. Useful for computing the ABX on discrete speech units."""
    n1, s1, _ = a1.size()
    n2, s2, _ = a2.size()
    return (a1.view(n1, 1, s1, 1) != a2.view(1, n2, 1, s2)).float()


def distance_on_cell(cell: Cell, distance: Distance) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the distance matrices between all A and X, and all B and X in the ``cell``, for a given ``distance``."""
    (a, sa), (b, sb), (x, sx) = (cell.a.data, cell.a.sizes), (cell.b.data, cell.b.sizes), (cell.x.data, cell.x.sizes)
    if cell.use_dtw:
        dxa = dtw_batch(distance(x, a), sx, sa, symmetric=cell.is_symmetric)
        dxb = dtw_batch(distance(x, b), sx, sb, symmetric=False)
    else:
        dxa, dxb = distance(x, a).squeeze(2, 3), distance(x, b).squeeze(2, 3)
    return dxa, dxb


def abx_on_cell(cell: Cell, distance: Distance) -> torch.Tensor:
    """Compute the ABX of a ``cell`` using the given ``distance``."""
    dxa, dxb = distance_on_cell(cell, distance)
    if cell.is_symmetric:
        dxa.fill_diagonal_(dxb.max() + 1)
    nx, na = dxa.size()
    nx, nb = dxb.size()
    dxb = dxb.view(nx, 1, nb).expand(nx, na, nb)
    dxa = dxa.view(nx, na, 1).expand(nx, na, nb)
    sc = (dxa < dxb).sum() + 0.5 * (dxa == dxb).sum()
    sc /= len(cell)
    return 1 - sc
