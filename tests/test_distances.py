"""Verify that the new distances functions match the original ones."""

import math
from collections.abc import Callable

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from torch import Tensor
from torch.testing import assert_close, make_tensor

from fastabx.distance import DistanceName, distance_function

BATCH, SEQ, DIM = st.integers(1, 20), st.integers(1, 10), st.integers(1, 1024)
LOW, HIGH_MINUS_LOW = st.floats(-100, 100), st.floats(0.1, 100)
MIN_POSITIVE = 0.1


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


OLD_DISTANCE_FN: dict[str, Callable[[Tensor, Tensor], Tensor]] = {
    "kl": kl_distance,
    "kl_symmetric": kl_symmetric_distance,
    "euclidean": euclidean_distance,
    "cosine": cosine_distance,
}


@pytest.mark.parametrize("name", list(OLD_DISTANCE_FN.keys()))
@given(n1=BATCH, n2=BATCH, s1=SEQ, s2=SEQ, d=DIM, low=LOW, high_minus_low=HIGH_MINUS_LOW)
def test_distance_new_implementation(  # noqa: PLR0913
    name: DistanceName,
    n1: int,
    n2: int,
    s1: int,
    s2: int,
    d: int,
    low: float,
    high_minus_low: float,
) -> None:
    """Compare the new distance implementation to the old one."""
    a = make_tensor((n1, s1, d), dtype=torch.float32, low=low, high=high_minus_low + low, device="cpu")
    b = make_tensor((n2, s2, d), dtype=torch.float32, low=low, high=high_minus_low + low, device="cpu")
    if name.startswith("kl"):
        a, b = torch.clamp(a, min=MIN_POSITIVE), torch.clamp(b, min=MIN_POSITIVE)
    distance_old = OLD_DISTANCE_FN[name]
    distance_new = distance_function(name)
    assert_close(distance_new(a, b), distance_old(a, b))
