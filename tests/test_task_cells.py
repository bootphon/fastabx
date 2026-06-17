"""Tests for ``fastabx.task`` and the cell-construction helpers in ``fastabx.cell``."""

import polars as pl
import pytest

from fastabx import Dataset, Task
from fastabx.cell import (
    MIN_A_LEN,
    Cell,
    NoAcrossError,
    cell_description,
    cell_header,
    cells_on_by,
    cells_on_by_across,
)
from fastabx.subsample import Subsampler


def _df(d: dict[str, list[object]]) -> pl.LazyFrame:
    return pl.DataFrame(d).lazy()


def test_cells_on_by_no_by_filters_min_a_len() -> None:
    # phone 'a' has length 1 — should be excluded because it's below MIN_A_LEN.
    df = _df({"phone": ["a", "b", "b", "c", "c"]})
    cells = cells_on_by(df, on="phone", by=[]).collect()
    assert "index_a" in cells.columns
    assert "index_b" in cells.columns
    assert "index_x" in cells.columns
    # Only b/c qualify; the (b, c) and (c, b) pairs survive.
    pairs = set(zip(cells["phone"].to_list(), cells["phone_b"].to_list(), strict=True))
    assert ("a", "b") not in pairs
    assert ("b", "c") in pairs
    assert ("c", "b") in pairs
    # Symmetric: index_x == index_a.
    for ia, ix in zip(cells["index_a"].to_list(), cells["index_x"].to_list(), strict=True):
        assert ia == ix


def test_cells_on_by_with_by_groups_correctly() -> None:
    df = _df(
        {
            "phone": ["a", "a", "b", "b", "c", "c"],
            "ctx": ["c1", "c1", "c1", "c1", "c2", "c2"],
        }
    )
    cells = cells_on_by(df, on="phone", by=["ctx"]).collect()
    # The shared 'ctx' restricts cells to within-context pairs (a vs b in c1; nothing in c2 because only c).
    pairs = set(zip(cells["phone"].to_list(), cells["phone_b"].to_list(), strict=True))
    assert pairs == {("a", "b"), ("b", "a")}


def test_cells_on_by_across_empty_across_raises() -> None:
    df = _df({"phone": ["a", "b"]})
    with pytest.raises(NoAcrossError):
        cells_on_by_across(df, on="phone", by=[], across=[]).collect()


def test_cells_on_by_across_basic_filter() -> None:
    df = _df(
        {
            "phone": ["a", "a", "b", "b", "a"],
            "speaker": ["s1", "s2", "s1", "s2", "s2"],  # across speaker
        }
    )
    cells = cells_on_by_across(df, on="phone", by=[], across=["speaker"]).collect()
    # A_on == X_on, B_on != X_on, A_across != X_across, B_across != X_across.
    for row in cells.iter_rows(named=True):
        assert row["phone"] != row["phone_b"]
        assert row["speaker"] != row["speaker_x"]


def test_cell_description_and_header_format() -> None:
    desc_expr = cell_description("phone", ["context"], ["speaker"])
    head_expr = cell_header("phone", ["context"], ["speaker"])
    df = pl.DataFrame(
        {
            "phone": ["a"],
            "phone_b": ["b"],
            "context": ["c1"],
            "speaker": ["s1"],
            "speaker_x": ["s2"],
        }
    )
    desc = df.select(desc_expr.alias("d"))["d"][0]
    head = df.select(head_expr.alias("h"))["h"][0]
    assert "ON(phone_ax = a, phone_b = b)" in desc
    assert "BY(context_abx = c1)" in desc
    assert "ACROSS(speaker_ab = s1, speaker_x = s2)" in desc
    assert head == "a-b-c1-s1-s2"


def test_task_len_and_getitem(tiny_dataset: Dataset) -> None:
    task = Task(tiny_dataset, on="phone", by=["context"])
    assert len(task) == len(task.cells)
    first = task[0]
    assert isinstance(first, Cell)
    with pytest.raises(IndexError):
        _ = task[len(task)]
    with pytest.raises(IndexError):
        _ = task[-1]


def test_task_iter_matches_indexing(tiny_dataset: Dataset) -> None:
    task = Task(tiny_dataset, on="phone", by=["context"])
    iter_cells = list(task)
    assert len(iter_cells) == len(task)
    for i, cell in enumerate(iter_cells):
        assert cell.header == task[i].header
        assert cell.description == task[i].description


def test_task_repr_reflects_conditions(tiny_dataset: Dataset) -> None:
    sub = Subsampler(max_size_group=3, max_x_across=2, seed=0)
    task = Task(tiny_dataset, on="phone", by=["context"], across=["speaker"], subsampler=sub)
    rep = repr(task)
    assert "ON(phone)" in rep
    assert "BY(context)" in rep
    assert "ACROSS(speaker)" in rep
    assert "maximal" in rep  # subsampler description


def test_task_is_symmetric_iff_no_across(tiny_dataset: Dataset) -> None:
    sym = Task(tiny_dataset, on="phone", by=["context"])
    asym = Task(tiny_dataset, on="phone", across=["speaker"])
    assert sym.is_symmetric is True
    assert asym.is_symmetric is False


def test_task_from_cells_bypasses_generation(tiny_dataset: Dataset) -> None:
    real = Task(tiny_dataset, on="phone", by=["context"])
    precomputed = Task.from_cells(tiny_dataset, real.cells, is_symmetric=True)
    assert len(precomputed) == len(real)
    assert precomputed.cells.equals(real.cells)
    assert precomputed.is_symmetric is True
    # Iteration works without on/by/across being validated against the dataset.
    assert next(iter(precomputed)).is_symmetric is True


def test_task_from_cells_respects_is_symmetric(tiny_dataset: Dataset) -> None:
    real = Task(tiny_dataset, on="phone", across=["speaker"])
    precomputed = Task.from_cells(tiny_dataset, real.cells, is_symmetric=False)
    assert precomputed.is_symmetric is False
    assert next(iter(precomputed)).is_symmetric is False


def test_task_from_cells_rejects_missing_columns(tiny_dataset: Dataset) -> None:
    from fastabx.verify import PrecomputedCellsError

    real = Task(tiny_dataset, on="phone", by=["context"])
    bad = real.cells.drop("header")
    with pytest.raises(PrecomputedCellsError, match="header"):
        Task.from_cells(tiny_dataset, bad, is_symmetric=True)


def test_task_from_cells_rejects_out_of_range_indices(tiny_dataset: Dataset) -> None:
    from fastabx.verify import PrecomputedCellsError

    real = Task(tiny_dataset, on="phone", by=["context"])
    too_big = len(tiny_dataset.accessor)
    bad = real.cells.with_columns(pl.col("index_a").list.eval(pl.element() + too_big))
    with pytest.raises(PrecomputedCellsError, match="only has"):
        Task.from_cells(tiny_dataset, bad, is_symmetric=True)


def test_task_cells_setter_is_read_only(tiny_dataset: Dataset) -> None:
    task = Task(tiny_dataset, on="phone", by=["context"])
    with pytest.raises(AttributeError, match="no setter"):
        task.cells = task.cells  # type: ignore[misc]


def test_min_a_len_module_constant() -> None:
    assert MIN_A_LEN == 2
