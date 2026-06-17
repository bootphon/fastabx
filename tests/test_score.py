"""Tests for ``fastabx.score``: end-to-end Score, collapsing, write_csv."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from fastabx import Dataset, Score, Task
from fastabx.constraints import constraints_all_different
from fastabx.score import CollapseError, pl_weighted_mean, score_details


@pytest.fixture
def small_score(tiny_dataset: Dataset) -> Score:
    task = Task(tiny_dataset, on="phone", by=["context"], across=["speaker"])
    return Score(task, "euclidean")


def test_score_repr_and_cells(small_score: Score) -> None:
    assert "Score" in repr(small_score)
    assert "euclidean" in repr(small_score)
    assert "score" in small_score.cells.columns
    assert "size" in small_score.cells.columns


def test_score_cells_setter_is_read_only(small_score: Score) -> None:
    with pytest.raises(AttributeError, match="read-only"):
        small_score.cells = small_score.cells  # type: ignore[misc]


def test_score_auto_normalizes_for_cosine(tiny_dataset: Dataset) -> None:
    import torch

    original = tiny_dataset.accessor.data.clone()
    task = Task(tiny_dataset, on="phone", by=["context"])
    Score(task, "cosine")
    # `normalize_with_singularity_` appends a border column → data width grows by 1.
    assert tiny_dataset.accessor.data.shape[1] == original.shape[1] + 1
    # Each row (excluding the appended border) must now have unit L2 norm.
    body = tiny_dataset.accessor.data[:, :-1]
    torch.testing.assert_close(body.norm(dim=1), torch.ones(body.size(0)), atol=1e-5, rtol=0)


def test_score_does_not_normalize_for_euclidean(tiny_dataset: Dataset) -> None:
    import torch

    original = tiny_dataset.accessor.data.clone()
    task = Task(tiny_dataset, on="phone", by=["context"])
    Score(task, "euclidean")
    # Shape AND values unchanged.
    assert tiny_dataset.accessor.data.shape == original.shape
    torch.testing.assert_close(tiny_dataset.accessor.data, original)


def test_collapse_weighted_with_levels_raises(small_score: Score) -> None:
    with pytest.raises(CollapseError, match="Cannot set"):
        small_score.collapse(weighted=True, levels=["speaker"])


def test_collapse_no_args_succeeds_with_exactly_two_non_index_columns() -> None:
    """Hits the ``levels = []`` default branch in ``score_details``."""
    rng = np.random.default_rng(0)
    features = rng.standard_normal((12, 3)).astype(np.float32)
    labels = {"phone": ["a", "b", "c"] * 4}
    dataset = Dataset.from_numpy(features, labels)
    task = Task(dataset, on="phone")  # symmetric, no by, no across → phone + phone_b only
    score = Score(task, "euclidean")
    out = score.collapse()  # no levels, no weighted — default path
    assert isinstance(out, float)
    assert 0.0 <= out <= 1.0


def test_collapse_no_args_raises_when_columns_unclear() -> None:
    # Two non-index columns aside from score/size: 'speaker' + 'speaker_x' isn't exactly 2.
    # Use a tighter task where the column count would fail.
    rng = np.random.default_rng(0)
    features = rng.standard_normal((12, 3)).astype(np.float32)
    labels = {"phone": ["a", "b"] * 6, "speaker": (["s1"] * 6 + ["s2"] * 6), "extra": ["e"] * 12}
    dataset = Dataset.from_numpy(features, labels)
    task = Task(dataset, on="phone", by=["speaker"])
    score = Score(task, "euclidean")
    with pytest.raises(CollapseError, match="Either"):
        score.collapse()  # ambiguous: too many non-index columns


def test_collapse_weighted_value_close_to_mean(small_score: Score) -> None:
    weighted = small_score.collapse(weighted=True)
    plain = float(small_score.cells["score"].drop_nulls().mean())  # ty: ignore[invalid-argument-type]
    # In our tiny setup, weights are equal, so the two reductions are equal.
    assert weighted == pytest.approx(plain, abs=1e-6)


def test_collapse_with_levels(small_score: Score) -> None:
    out = small_score.collapse(levels=["speaker"])
    assert isinstance(out, float)
    assert 0.0 <= out <= 1.0


def test_score_details_returns_dataframe(small_score: Score) -> None:
    details = small_score.details(levels=["speaker"])
    assert isinstance(details, pl.DataFrame)
    assert "score" in details.columns
    assert "size" in details.columns


def test_pl_weighted_mean_handles_nulls() -> None:
    df = pl.DataFrame({"score": [0.2, None, 0.4], "size": [10, 5, 20]})
    weighted = df.select(pl_weighted_mean("score", "size")).item()
    expected = (0.2 * 10 + 0.4 * 20) / (10 + 20)
    assert weighted == pytest.approx(expected)


def test_pl_weighted_mean_all_null_returns_none() -> None:
    df = pl.DataFrame({"score": [None, None], "size": [3, 4]}, schema={"score": pl.Float32, "size": pl.Int32})
    assert df.select(pl_weighted_mean("score", "size")).item() is None


def test_score_write_csv_drops_list_columns(small_score: Score, tmp_path: Path) -> None:
    out = tmp_path / "scores.csv"
    small_score.write_csv(out)
    assert out.exists()
    readback = pl.read_csv(out)
    # None of the list columns should round-trip into the CSV.
    for name, dtype in small_score.cells.schema.items():
        if dtype == pl.List:
            assert name not in readback.columns
    assert "score" in readback.columns
    assert "size" in readback.columns


def test_score_details_with_unknown_level_raises(small_score: Score) -> None:
    from fastabx.verify import InvalidLevelsError

    with pytest.raises(InvalidLevelsError):
        small_score.details(levels=["nope"])


def test_score_constrained_propagates_none() -> None:
    # Build a task where the constraint kills every triplet.
    rng = np.random.default_rng(0)
    features = rng.standard_normal((8, 3)).astype(np.float32)
    labels = {
        "phone": ["a", "b"] * 4,
        "context": ["c1"] * 8,  # only one context
    }
    dataset = Dataset.from_numpy(features, labels)
    task = Task(dataset, on="phone")
    score = Score(task, "euclidean", constraints=constraints_all_different("context"))
    assert score.cells["score"].null_count() == len(score.cells)


def test_score_details_collapse_order_matters_with_unequal_subgroups() -> None:
    """Different collapse orderings give different unweighted means when the subgroup counts differ.

    When subgroup counts along the path are not uniform, the order of collapsing changes the
    final value — this is what ``Score.details`` warns about.
    """
    # Unequal contexts per speaker: s1 has c1,c2; s2 has c1,c2,c3.
    cells = pl.DataFrame(
        {
            "phone": ["a"] * 5,
            "phone_b": ["b"] * 5,
            "speaker": ["s1", "s1", "s2", "s2", "s2"],
            "context": ["c1", "c2", "c1", "c2", "c3"],
            "score": pl.Series([0.0, 0.0, 1.0, 1.0, 1.0], dtype=pl.Float32),
            "size": pl.Series([10] * 5, dtype=pl.Int32),
        }
    )
    # context first: collapse over context → s1: mean(0,0)=0, s2: mean(1,1,1)=1; then speaker → 0.5
    by_context_first = score_details(cells, levels=["context", "speaker"])
    # speaker first: each (context,) group then collapsed:
    #   c1: mean(0, 1)=0.5; c2: mean(0, 1)=0.5; c3: mean(1)=1; then collapse context → mean = ~0.667
    by_speaker_first = score_details(cells, levels=["speaker", "context"])
    assert by_context_first["score"][0] == pytest.approx(0.5, abs=1e-6)
    assert by_speaker_first["score"][0] == pytest.approx(2 / 3, abs=1e-6)
    assert by_context_first["score"][0] != by_speaker_first["score"][0]


def test_score_details_single_level_aggregates_correctly() -> None:
    cells = pl.DataFrame(
        {
            "phone": ["a", "a", "b", "b"],
            "phone_b": ["b", "b", "a", "a"],
            "speaker": ["s1", "s2", "s1", "s2"],
            "score": pl.Series([0.1, 0.3, 0.4, 0.2], dtype=pl.Float32),
            "size": pl.Series([10, 10, 10, 10], dtype=pl.Int32),
        }
    )
    out = score_details(cells, levels=["speaker"])
    rows = dict(
        zip(
            zip(out["phone"].to_list(), out["phone_b"].to_list(), strict=True),
            out["score"].to_list(),
            strict=True,
        )
    )
    assert rows[("a", "b")] == pytest.approx(0.2, abs=1e-6)
    assert rows[("b", "a")] == pytest.approx(0.3, abs=1e-6)
