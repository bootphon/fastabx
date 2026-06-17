"""Additional correctness tests for ``fastabx.distance``."""

import math
from collections.abc import Callable
from typing import get_args

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from torch.testing import assert_close, make_tensor
from torchdtw import dtw_batch

from fastabx.distance import (
    DistanceName,
    angular_distance,
    distance_function,
    distance_matrix,
    euclidean_distance,
    identical_distance,
    kl_symmetric_distance,
)

NAMES: list[DistanceName] = list(get_args(DistanceName.__value__))
KL_FLOAT32_TOL = 1e-5  # observed slack of mean-centered KL in float32 is well under this


@pytest.mark.parametrize("name", NAMES)
def test_distance_function_returns_callable(name: DistanceName) -> None:
    fn = distance_function(name)
    assert callable(fn)


def test_distance_function_cosine_and_angular_alias() -> None:
    assert distance_function("cosine") is distance_function("angular")


def test_distance_function_unknown_raises() -> None:
    with pytest.raises(ValueError, match="bogus"):
        distance_function("bogus")  # ty: ignore[invalid-argument-type]


def test_euclidean_known_values() -> None:
    a = torch.tensor([[[0.0, 0.0]], [[1.0, 0.0]]])  # (2, 1, 2)
    b = torch.tensor([[[0.0, 0.0]], [[0.0, 1.0]]])  # (2, 1, 2)
    out = euclidean_distance(a, b)
    assert out.shape == (2, 2, 1, 1)
    assert_close(out[0, 0, 0, 0].item(), 0.0)
    assert_close(out[0, 1, 0, 0].item(), 1.0)
    assert_close(out[1, 0, 0, 0].item(), 1.0)
    assert_close(out[1, 1, 0, 0].item(), math.sqrt(2.0))


def test_angular_known_values() -> None:
    # orthogonal, identical, opposite — pre-normalised inputs.
    a = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]], [[1.0, 0.0]]])  # (3, 1, 2)
    b = torch.tensor([[[0.0, 1.0]], [[1.0, 0.0]], [[-1.0, 0.0]]])
    out = angular_distance(a, b)
    assert out.shape == (3, 3, 1, 1)
    assert_close(out[0, 0, 0, 0].item(), 0.5, atol=1e-6, rtol=0)  # orthogonal -> 0.5
    assert_close(out[1, 1, 0, 0].item(), 0.0, atol=1e-6, rtol=0)  # identical -> 0
    assert_close(out[2, 2, 0, 0].item(), 1.0, atol=1e-6, rtol=0)  # opposite -> 1


def test_identical_known_values() -> None:
    a = torch.tensor([[[1.0]], [[2.0]]])  # (2, 1, 1)
    b = torch.tensor([[[1.0]], [[3.0]]])
    out = identical_distance(a, b)
    assert out.shape == (2, 2, 1, 1)
    assert out[0, 0, 0, 0].item() == 0.0
    assert out[0, 1, 0, 0].item() == 1.0
    assert out[1, 0, 0, 0].item() == 1.0
    assert out[1, 1, 0, 0].item() == 1.0


def test_kl_symmetric_zero_distance_for_same_distribution() -> None:
    p = torch.tensor([[[0.25, 0.25, 0.5]]])
    out = kl_symmetric_distance(p, p)
    assert_close(out.item(), 0.0, atol=1e-6, rtol=0)


def test_kl_symmetric_non_negative_and_finite_near_zero() -> None:
    # Near-degenerate distributions: one mass concentrated, the other uniform.
    eps = 1e-7
    n, d = 4, 5
    concentrated = torch.full((n, 1, d), eps)
    concentrated[:, 0, 0] = 1.0 - (d - 1) * eps
    uniform = torch.full((n, 1, d), 1.0 / d)
    out = kl_symmetric_distance(concentrated, uniform)
    assert torch.isfinite(out).all()
    assert (out >= -KL_FLOAT32_TOL).all()


@pytest.mark.parametrize(
    ("name", "fn"),
    [
        ("euclidean", euclidean_distance),
        ("kl_symmetric", kl_symmetric_distance),
        ("identical", identical_distance),
    ],
)
def test_output_shape_contract(name: str, fn: Callable[..., torch.Tensor]) -> None:
    n1, n2, s1, s2, d = 3, 2, 4, 5, 1 if name == "identical" else 6
    a = make_tensor((n1, s1, d), dtype=torch.float32, low=0.1, high=1.0, device="cpu")
    b = make_tensor((n2, s2, d), dtype=torch.float32, low=0.1, high=1.0, device="cpu")
    if name == "kl_symmetric":
        a, b = a / a.sum(-1, keepdim=True), b / b.sum(-1, keepdim=True)
    out = fn(a, b)
    assert out.shape == (n1, n2, s1, s2)


def test_angular_output_in_unit_interval() -> None:
    rng = torch.Generator().manual_seed(0)
    a = torch.randn(5, 1, 8, generator=rng)
    b = torch.randn(4, 1, 8, generator=rng)
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    out = angular_distance(a, b)
    assert (out >= 0).all()
    assert (out <= 1).all()


@given(
    n=st.integers(1, 6),
    s=st.integers(1, 4),
    d=st.integers(2, 8),
    seed=st.integers(0, 10_000),
)
@settings(deadline=None, max_examples=20)
def test_euclidean_symmetry_and_diagonal(n: int, s: int, d: int, seed: int) -> None:
    rng = torch.Generator().manual_seed(seed)
    a = torch.randn(n, s, d, generator=rng)
    dij = euclidean_distance(a, a)  # (n, n, s, s)
    # symmetry: d(a, a)[i, j] == d(a, a)[j, i].T (swap s1<->s2)
    assert_close(dij, dij.transpose(0, 1).transpose(2, 3))
    # non-negative
    assert (dij >= 0).all()
    # diagonal: same vector at same time has distance 0
    for i in range(n):
        for t in range(s):
            assert_close(dij[i, i, t, t].item(), 0.0, atol=1e-5, rtol=0)


@given(
    n=st.integers(1, 5),
    s=st.integers(1, 3),
    d=st.integers(2, 6),
    seed=st.integers(0, 10_000),
)
@settings(deadline=None, max_examples=20)
def test_kl_symmetric_non_negative(n: int, s: int, d: int, seed: int) -> None:
    rng = torch.Generator().manual_seed(seed)
    a = torch.rand(n, s, d, generator=rng).clamp(min=0.05)
    b = torch.rand(n, s, d, generator=rng).clamp(min=0.05)
    a, b = a / a.sum(-1, keepdim=True), b / b.sum(-1, keepdim=True)
    out = kl_symmetric_distance(a, b)
    assert (out >= -KL_FLOAT32_TOL).all()


def test_distance_matrix_no_dtw_squeezes() -> None:
    a = torch.randn(3, 1, 5)
    b = torch.randn(4, 1, 5)
    sa, sb = torch.tensor([1, 1, 1], dtype=torch.int32), torch.tensor([1, 1, 1, 1], dtype=torch.int32)
    out = distance_matrix(a, sa, b, sb, euclidean_distance, use_dtw=False, symmetric=False)
    assert out.shape == (3, 4)
    expected = euclidean_distance(a, b).squeeze(2, 3)
    assert_close(out, expected)


def test_distance_matrix_dtw_matches_torchdtw() -> None:
    rng = torch.Generator().manual_seed(0)
    a = torch.randn(2, 3, 4, generator=rng)
    b = torch.randn(2, 4, 4, generator=rng)
    sa = torch.tensor([3, 2], dtype=torch.int32)
    sb = torch.tensor([4, 3], dtype=torch.int32)
    out = distance_matrix(a, sa, b, sb, euclidean_distance, use_dtw=True, symmetric=False)
    expected = dtw_batch(euclidean_distance(a, b), sa, sb, symmetric=False)
    assert_close(out, expected)


def test_distance_matrix_symmetric_flag_passed_through() -> None:
    rng = torch.Generator().manual_seed(0)
    a = torch.randn(3, 3, 4, generator=rng)
    sa = torch.tensor([3, 2, 3], dtype=torch.int32)
    sym = distance_matrix(a, sa, a, sa, euclidean_distance, use_dtw=True, symmetric=True)
    asym = distance_matrix(a, sa, a, sa, euclidean_distance, use_dtw=True, symmetric=False)
    # Same numerical contract; symmetric=True only changes how torchdtw fills the upper triangle.
    assert sym.shape == asym.shape == (3, 3)
    # Diagonal is the self-DTW cost, which equals the sum of pointwise self-distances on the path
    # — for euclidean self-distance that's 0.
    for i in range(3):
        assert_close(sym[i, i].item(), 0.0, atol=1e-5, rtol=0)
        assert_close(asym[i, i].item(), 0.0, atol=1e-5, rtol=0)
