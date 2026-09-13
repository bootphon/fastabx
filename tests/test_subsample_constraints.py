"""Tests for ``fastabx.subsample`` and ``fastabx.constraints``."""

import numpy as np
import polars as pl
import pytest

from fastabx import Dataset, Task
from fastabx.constraints import (
    NoConstraintsError,
    apply_constraints,
    constraints_all_different,
)
from fastabx.subsample import Subsampler, subsample_across_group, subsample_each_cell
from fastabx.verify import InputTypeError


def _cells_frame() -> pl.LazyFrame:
    """Larger synthetic cells DataFrame (so subsample seeds are statistically distinguishable)."""
    return pl.DataFrame(
        {
            "phone": ["a", "a"],
            "phone_b": ["b", "b"],
            "index_a": [list(range(20)), list(range(40, 60))],
            "index_b": [list(range(20, 40)), list(range(60, 80))],
            "index_x": [list(range(20)), list(range(40, 60))],
        }
    ).lazy()


def test_subsample_each_cell_caps_sizes() -> None:
    out = subsample_each_cell(_cells_frame(), size=5, seed=0).collect()
    for col in ("index_a", "index_b", "index_x"):
        for value in out[col].to_list():
            assert len(value) == 5


def test_subsample_each_cell_deterministic_for_seed() -> None:
    a = subsample_each_cell(_cells_frame(), size=5, seed=42).collect()
    b = subsample_each_cell(_cells_frame(), size=5, seed=42).collect()
    assert a.equals(b)


def test_subsample_each_cell_different_seed_different_selection() -> None:
    # With 20-item lists and size=5, choosing the same 5-subset under two seeds has probability
    # C(20,5) ** -1 ≈ 6e-5 per column. The combined event across A, B, X is astronomically rare.
    a = subsample_each_cell(_cells_frame(), size=5, seed=0).collect()
    b = subsample_each_cell(_cells_frame(), size=5, seed=1).collect()
    assert (
        a["index_a"].to_list() != b["index_a"].to_list()
        or a["index_b"].to_list() != b["index_b"].to_list()
        or a["index_x"].to_list() != b["index_x"].to_list()
    )


def test_subsampler_description() -> None:
    sub = Subsampler(max_size_group=10, max_x_across=5)
    desc = sub.description(with_across=True)
    assert "10" in desc
    assert "5" in desc
    assert "X" in desc

    desc_no_across = sub.description(with_across=False)
    assert "10" in desc_no_across
    assert "5" not in desc_no_across


def test_subsampler_disabled_passes_through() -> None:
    sub = Subsampler(max_size_group=None, max_x_across=None)
    df = _cells_frame()
    out = sub(df, with_across=False).collect()
    expected = df.collect()
    assert out.equals(expected)


def test_subsampler_rejects_invalid_sizes() -> None:
    with pytest.raises(TypeError):
        Subsampler(max_size_group=1, max_x_across=None)  # must be > 1
    with pytest.raises(TypeError):
        Subsampler(max_size_group="oops", max_x_across=None)  # ty: ignore[invalid-argument-type]
    with pytest.raises(InputTypeError):
        Subsampler(max_size_group=10, max_x_across=5, seed=0.5)  # ty: ignore[invalid-argument-type]


def test_constraints_all_different_builds_expressions() -> None:
    cstrs = list(constraints_all_different("speaker"))
    assert len(cstrs) == 1
    # The expression references three columns: speaker_a, speaker_x, speaker_b.
    roots = set(cstrs[0].meta.root_names())
    assert roots == {"speaker_a", "speaker_x", "speaker_b"}


def test_apply_constraints_adds_is_valid_matches_hand_computed_mask() -> None:
    rng = np.random.default_rng(0)
    features = rng.standard_normal((9, 3)).astype(np.float32)
    contexts = ["c1", "c2", "c3"] * 3
    labels = {
        "phone": ["a", "b", "c"] * 3,
        "speaker": (["s1"] * 3 + ["s2"] * 3 + ["s3"] * 3),
        "context": contexts,
    }
    dataset = Dataset.from_numpy(features, labels)
    task = Task(dataset, on="phone", across=["speaker"])
    out = apply_constraints(task.cells, dataset.labels, constraints_all_different("context"), is_symmetric=False)
    assert "is_valid" in out.columns

    # Hand-compute the expected mask per cell and compare bit-for-bit.
    for row in out.iter_rows(named=True):
        nx = len(row["index_x"])
        na = len(row["index_a"])
        nb = len(row["index_b"])
        expected = []
        for ix in row["index_x"]:
            for ia in row["index_a"]:
                for ib in row["index_b"]:
                    cx, ca, cb = contexts[ix], contexts[ia], contexts[ib]
                    expected.append(len({ca, cb, cx}) == 3)
        assert row["is_valid"] == expected
        assert len(row["is_valid"]) == nx * na * nb


def test_apply_constraints_no_matching_columns_raises() -> None:
    rng = np.random.default_rng(0)
    features = rng.standard_normal((6, 3)).astype(np.float32)
    labels = {"phone": ["a", "b", "c"] * 2, "speaker": ["s1", "s1", "s1", "s2", "s2", "s2"]}
    dataset = Dataset.from_numpy(features, labels)
    task = Task(dataset, on="phone", across=["speaker"])
    with pytest.raises(NoConstraintsError):
        apply_constraints(task.cells, dataset.labels, constraints_all_different("not_a_column"), is_symmetric=False)


def test_apply_constraints_handles_cells_with_identical_conditions() -> None:
    """Two distinct cells sharing the same condition columns must each get their own mask.

    Regrouping used to happen on the condition columns, which collapsed these two cells into one and
    raised rather than misaligning. They are now matched back by row number, so this works and each
    cell keeps the mask of its own triplets.
    """
    rng = np.random.default_rng(0)
    features = rng.standard_normal((6, 3)).astype(np.float32)
    labels = pl.DataFrame({"phone": ["a", "b"] * 3, "context": ["c1", "c2", "c3"] * 2})
    dataset = Dataset.from_numpy(features, labels)
    cells = pl.DataFrame(
        {
            "phone": ["a", "a"],
            "phone_b": ["b", "b"],
            "index_a": [[0, 2], [0, 4]],
            "index_b": [[1, 3], [1, 5]],
            "index_x": [[0, 2], [0, 4]],
        }
    )
    out = apply_constraints(cells, dataset.labels, constraints_all_different("context"), is_symmetric=False)
    assert len(out) == 2
    contexts = labels["context"].to_list()
    for row, (index_a, index_b, index_x) in enumerate(
        zip(cells["index_a"], cells["index_b"], cells["index_x"], strict=True)
    ):
        expected = [
            contexts[a] != contexts[x] and contexts[a] != contexts[b] and contexts[x] != contexts[b]
            for x in index_x
            for a in index_a
            for b in index_b
        ]
        assert out["is_valid"].to_list()[row] == expected


def test_apply_constraints_symmetric_adds_index_inequality() -> None:
    rng = np.random.default_rng(0)
    features = rng.standard_normal((9, 3)).astype(np.float32)
    labels = {
        "phone": ["a", "b", "c"] * 3,
        "context": ["c1", "c2", "c3"] * 3,
    }
    dataset = Dataset.from_numpy(features, labels)
    task = Task(dataset, on="phone")
    out = apply_constraints(task.cells, dataset.labels, constraints_all_different("context"), is_symmetric=True)
    assert "is_valid" in out.columns
    # In symmetric mode the diagonal (X==A) is excluded: at least one False in the flat mask.
    any_false = any(False in vals for vals in out["is_valid"].to_list())
    assert any_false


def test_subsample_across_group_caps_x_values() -> None:
    df = pl.DataFrame(
        {
            "phone": ["a"] * 5,
            "phone_b": ["b"] * 5,
            "speaker": ["s1"] * 5,
            "speaker_x": ["s2", "s3", "s4", "s5", "s6"],
            "index_a": [[0], [0], [0], [0], [0]],
            "index_b": [[1], [1], [1], [1], [1]],
            "index_x": [[2], [3], [4], [5], [6]],
        }
    ).lazy()
    out = subsample_across_group(df, size=2, seed=0).collect()
    distinct_x_speakers = set(out["speaker_x"].to_list())
    assert len(distinct_x_speakers) == 2


@pytest.mark.parametrize("is_symmetric", [True, False])
def test_apply_constraints_mask_order_matches_brute_force(*, is_symmetric: bool) -> None:
    """The flat mask must be in x-major, then a, then b order, element for element.

    ``group_cells`` reshapes each cell's ``is_valid`` list to ``(nx, na, nb)``, so any permutation
    inside a cell silently scrambles which triplets are kept instead of raising. The joins and the
    group-by do not preserve row order on the streaming engine, so this pins the order against an
    independent brute-force oracle rather than against a checksum, which a permutation would pass.
    """
    rng = np.random.default_rng(3)
    n = 240
    dataset = Dataset.from_numpy(
        rng.standard_normal((n, 3)).astype(np.float32),
        {
            "phone": [f"p{i % 5}" for i in range(n)],
            "ctx": [f"c{(i // 5) % 3}" for i in range(n)],
            "speaker": [f"s{(i // 2) % 4}" for i in range(n)],
        },
    )
    across = None if is_symmetric else ["speaker"]
    task = Task(dataset, on="phone", by=["ctx"], across=across)
    assert task.is_symmetric is is_symmetric
    out = apply_constraints(
        task.cells, dataset.labels, constraints_all_different("speaker"), is_symmetric=is_symmetric
    )
    speakers = dataset.labels["speaker"].to_list()
    got = out["is_valid"].to_list()
    assert len(got) == len(task)
    for flat, (index_a, index_b, index_x) in zip(
        got, task.cells[["index_a", "index_b", "index_x"]].iter_rows(), strict=True
    ):
        expected = [
            speakers[a] != speakers[x]
            and speakers[a] != speakers[b]
            and speakers[x] != speakers[b]
            and (a != x or not is_symmetric)
            for x in index_x
            for a in index_a
            for b in index_b
        ]
        assert list(flat) == expected


def _dataset_with_index_prefixed_label() -> Dataset:
    """Build a dataset whose BY condition is named ``indexer``: legal, but it starts with "index"."""
    rng = np.random.default_rng(0)
    features = rng.standard_normal((12, 4)).astype(np.float32)
    labels = {"phone": list("aabbcc") * 2, "indexer": ["g1"] * 6 + ["g2"] * 6}
    return Dataset.from_numpy(features, labels)


def test_condition_named_like_an_index_column_is_not_treated_as_one() -> None:
    """Only ``index_a``/``index_b``/``index_x`` are index columns, not everything starting with "index".

    A condition the user happened to name ``indexer`` used to be swept up by the ``starts_with("index")``
    selectors: the subsampler tried to explode it, and the collapse dropped it before averaging.
    """
    from fastabx import Score

    dataset = _dataset_with_index_prefixed_label()
    task = Task(dataset, on="phone", by=["indexer"], subsampler=Subsampler(2, None))
    assert "indexer" in task.cells.columns
    score = Score(task, "euclidean")
    assert "indexer" in score.details(levels=[]).columns
    assert 0 <= score.collapse(levels=["indexer"]) <= 1
