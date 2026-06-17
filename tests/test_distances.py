"""Verify that the new distance implementations match the original ones."""

import math
from collections.abc import Callable

import pytest
import torch
from hypothesis import assume, given
from hypothesis import strategies as st
from torch import Tensor
from torch.testing import assert_close, make_tensor

from fastabx.distance import DistanceName, distance_function

BATCH, SEQ, DIM = st.integers(1, 20), st.integers(1, 10), st.integers(1, 1024)
LOW, HIGH_MINUS_LOW = st.floats(-100, 100), st.floats(0.1, 100)


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
    "kl_symmetric": kl_symmetric_distance,
    "euclidean": euclidean_distance,
}


@pytest.mark.parametrize("name", list(OLD_DISTANCE_FN.keys()))
@given(n1=BATCH, n2=BATCH, s1=SEQ, s2=SEQ, d=DIM, low=LOW, high_minus_low=HIGH_MINUS_LOW)
def test_distance_new_implementation(
    name: DistanceName,
    n1: int,
    n2: int,
    s1: int,
    s2: int,
    d: int,
    low: float,
    high_minus_low: float,
) -> None:
    """Bit-close agreement of the new and old implementations for non-cosine distances."""
    a = make_tensor((n1, s1, d), dtype=torch.float32, low=low, high=high_minus_low + low, device="cpu")
    b = make_tensor((n2, s2, d), dtype=torch.float32, low=low, high=high_minus_low + low, device="cpu")
    if name.startswith("kl"):
        a, b = torch.clamp(a, min=0.1), torch.clamp(b, min=0.1)
        a, b = a / a.sum(dim=-1, keepdim=True), b / b.sum(dim=-1, keepdim=True)
    assert_close(distance_function(name)(a, b), OLD_DISTANCE_FN[name](a, b))


COSINE_WELL_CONDITIONED_DOT_BOUND = 0.95


def _normalize(t: Tensor) -> Tensor:
    return t / t.norm(dim=-1, keepdim=True).clamp(min=1e-12)


@given(n1=BATCH, n2=BATCH, s1=SEQ, s2=SEQ, d=DIM, low=LOW, high_minus_low=HIGH_MINUS_LOW)
def test_cosine_new_implementation_boundary(
    n1: int, n2: int, s1: int, s2: int, d: int, low: float, high_minus_low: float
) -> None:
    """Cosine agreement in the boundary regime, including parallel/antipodal unit vectors.

    Near ±1 ``acos`` has unbounded derivative, so the matmul-vs-pointwise summation orders
    diverge by up to ~sqrt(d) * ulp amplified by acos's steepness. ``1e-3`` is the float32
    ceiling we can guarantee. The well-conditioned regime is exercised separately below
    with a much tighter check.
    """
    a = _normalize(make_tensor((n1, s1, d), dtype=torch.float32, low=low, high=high_minus_low + low, device="cpu"))
    b = _normalize(make_tensor((n2, s2, d), dtype=torch.float32, low=low, high=high_minus_low + low, device="cpu"))
    assert_close(distance_function("cosine")(a, b), cosine_distance(a, b), atol=1e-3, rtol=1.3e-6)


@given(n1=BATCH, n2=BATCH, s1=SEQ, s2=SEQ, d=DIM, low=LOW, high_minus_low=HIGH_MINUS_LOW)
def test_cosine_new_implementation_well_conditioned(
    n1: int, n2: int, s1: int, s2: int, d: int, low: float, high_minus_low: float
) -> None:
    """Cosine agreement when all dot products are bounded away from ±1.

    Restricting ``|<a, b>| <= 0.95`` keeps ``acos`` Lipschitz with constant
    ``1/sqrt(1 - 0.95**2) ≈ 3.2``. With float32 dot-product noise of ~``sqrt(d) * ulp ≈ 4e-6``
    for ``d`` up to 1024, the output noise is bounded by ``3.2 * 4e-6 / pi ≈ 4e-6`` — well
    under the default ``assert_close`` atol of ``1e-5``.
    """
    a = _normalize(make_tensor((n1, s1, d), dtype=torch.float32, low=low, high=high_minus_low + low, device="cpu"))
    b = _normalize(make_tensor((n2, s2, d), dtype=torch.float32, low=low, high=high_minus_low + low, device="cpu"))
    # Compute the dot-product matrix and let hypothesis re-sample if it lands too close to ±1.
    dots = torch.einsum("nsd,mtd->nmst", a, b)
    assume(dots.abs().max().item() <= COSINE_WELL_CONDITIONED_DOT_BOUND)
    assert_close(distance_function("cosine")(a, b), cosine_distance(a, b))
