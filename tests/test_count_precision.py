"""Large-count regressions without materializing billions of triplets."""

from fractions import Fraction
from pathlib import Path

import polars as pl
import pytest
import torch

from fastabx import Batch, Cell, Dataset, Score, Task, abx_on_cell
from fastabx.group import GroupReducer, grouped_contributions


@pytest.mark.parametrize("size", [2_000_000_000, 3_000_000_000])
def test_large_sizes_survive_scoring_collapse_and_csv(
    tiny_dataset: Dataset, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, size: int
) -> None:
    cells = pl.DataFrame(
        {
            "header": ["h", "h"],
            "description": ["d", "d"],
            "phone": ["a", "a"],
            "phone_b": ["b", "b"],
            "speaker": ["s1", "s2"],
            "index_a": [[0, 1], [0, 1]],
            "index_x": [[0, 1], [0, 1]],
            "index_b": [[2], [2]],
        }
    )
    task = Task.from_cells(tiny_dataset, cells, is_symmetric=True)
    # Supply already-reduced counts to exercise storage/export without allocating the corresponding triplets.
    monkeypatch.setattr("fastabx.score.score_task", lambda *_args, **_kwargs: ([0.25, 0.75], [size, size]))
    score = Score(task, "euclidean", progress=False)
    assert score.cells.schema["size"] == pl.Int64
    assert score.details(levels=["speaker"])["size"].item() == 2 * size
    assert score.collapse(weighted=True) == pytest.approx(0.5)
    output = tmp_path / "scores.csv"
    score.write_csv(output)
    assert pl.read_csv(output)["size"].to_list() == [size, size]


@pytest.mark.parametrize("constrained", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_reducer_preserves_large_sizes_and_half_counts(dtype: torch.dtype, *, constrained: bool) -> None:
    reducer = GroupReducer(1, constrained=constrained)
    size = 2**31 + 1
    # Simulate two B columns: all successes in the first, one tie in the second.
    reducer._per_b = [torch.tensor([size - 1, 0.5], dtype=dtype)]  # ruff: ignore[private-member-access]
    reducer._positions = [0]  # ruff: ignore[private-member-access]
    reducer._nb = [2]  # ruff: ignore[private-member-access]
    if constrained:
        reducer._per_b_valid = [torch.tensor([size - 1, 1], dtype=torch.int64)]  # ruff: ignore[private-member-access]
    else:
        reducer.sizes = [size]
    scores, sizes = reducer.finalize()
    assert sizes == [size]
    expected = float(Fraction(1, 2 * size))
    assert scores[0] == pytest.approx(expected, rel=1e-7, abs=0)


@pytest.mark.parametrize("constrained", [False, True])
@pytest.mark.parametrize("na", [2**23, 2**23 + 1, 2**24 + 1])
def test_grouped_reduction_preserves_half_count_above_float32_precision(na: int, *, constrained: bool) -> None:
    dxa = torch.zeros(1, na)
    dxa[0, -1] = 1.0
    mask = torch.ones(1, na, 1, dtype=torch.bool) if constrained else None
    result = grouped_contributions(dxa, torch.ones(1, 1), mask)
    assert result.dtype == (torch.float32 if na <= 2**23 else torch.float64)
    assert result.item() == pytest.approx(na - 0.5, rel=0, abs=0)


def test_float16_counts_do_not_overflow_in_grouped_or_single_cell_scoring() -> None:
    size = 257  # size**2 exceeds float16's largest finite value.
    dxa = torch.zeros(size, size, dtype=torch.float16)
    counts = grouped_contributions(dxa, torch.ones(size, 1, dtype=torch.float16))
    assert counts.item() == size**2

    a = Batch(torch.zeros(size, 1, 1, dtype=torch.float16), torch.ones(size, dtype=torch.int32))
    b = Batch(torch.ones(1, 1, 1, dtype=torch.float16), torch.ones(1, dtype=torch.int32))
    cell = Cell(a=a, b=b, x=a, header="h", description="d", is_symmetric=False)

    def absolute(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (x[:, None, :, None, 0] - y[None, :, None, :, 0]).abs()

    assert float(abx_on_cell(cell, absolute)) == pytest.approx(0, abs=0)


def test_flush_preserves_mixed_precision_groups() -> None:
    reducer = GroupReducer(2)
    size = 2**31
    # A large group's half-count must survive concatenation with a small float32 group.
    reducer._per_b = [  # ruff: ignore[private-member-access]
        torch.tensor([size - 0.5], dtype=torch.float64),
        torch.tensor([0.5], dtype=torch.float32),
    ]
    reducer._positions = [0, 1]  # ruff: ignore[private-member-access]
    reducer._nb = [1, 1]  # ruff: ignore[private-member-access]
    reducer.sizes = [size, 1]
    scores, sizes = reducer.finalize()
    assert sizes == [size, 1]
    assert scores == pytest.approx([float(Fraction(1, 2 * size)), 0.5], rel=1e-7, abs=0)
