"""Grouped scoring engine: gather cells sharing the same (X, A) and reduce their distances to per-cell scores."""

import math
from collections.abc import Generator
from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
from torch import Tensor

from fastabx.constraints import Constraints, NoConstraintsError, apply_constraints
from fastabx.dataset import Batch, InMemoryAccessor
from fastabx.distance import Distance, distance_matrix
from fastabx.task import Task
from fastabx.utils import GATHER_CHUNK_ROWS, MAX_SCORE_CHUNK_ROWS, REDUCTION_FLUSH_COLS

__all__ = []


@dataclass(frozen=True, slots=True)
class CellGroup:
    """A group of cells that share the same X and A sample sets, with all targets built in one gather.

    :param x: The shared X samples, gathered and padded to a common length.
    :param targets: The concatenation of the A samples (the first ``rows[0]`` rows) and then every cell's B samples,
        gathered and padded together in a single ``accessor.batched`` call instead of one call per A/B.
    :param rows: The per-target row counts (``rows[0]`` = na, ``rows[1:]`` = nb per cell).
    :param positions: The position of each cell in the task's ``cells`` DataFrame, so the A block, the per-cell B
        blocks, and the scores can be sliced/written back in the original order without rebuilding anything.
    :param mask: The optional ``(nx, na, sum(b_rows))`` per-triplet constraint mask (every cell's ``(nx, na, nb)``
        mask concatenated along the B axis, in the same order as the B blocks of ``targets``);
        ``None`` when scoring without constraints.
    """

    x: Batch
    targets: Batch
    rows: list[int]
    positions: list[int]
    mask: Tensor | None = None


@dataclass(frozen=True, slots=True)
class GroupSpec:
    """A group's host-side description, before its data is gathered.

    :param smax: The max sequence length among its samples (used to length-sort groups so batched gathers pad tightly).
    :param is_symmetric: Whether the group is symmetric (X == A) or not.
    :param nx: The X count.
    :param indices: The flat gather list (X (only when X != A), then A, then every B).
    :param rows: The per-target row counts.
    :param positions: The cells' DataFrame positions.
    :param mask: The optional ``(nx, na, sum(b_rows))`` constraint mask.
    """

    smax: int
    is_symmetric: bool
    nx: int
    indices: list[int]
    rows: list[int]
    positions: list[int]
    mask: np.ndarray | None


def gather_chunk(accessor: InMemoryAccessor, chunk: list[GroupSpec]) -> Generator[CellGroup, None, None]:
    """Gather a chunk of groups in one ``accessor.batched`` call, then yield each group as a slice of the result.

    :param accessor: The dataset's :py:class:`.InMemoryAccessor`.
    :param chunk: A list of :py:class:`.GroupSpec` objects, each describing a group to gather.
    :returns: A generator of :py:class:`.CellGroup` objects, one per group in ``chunk``.
    """
    flat = []
    for spec in chunk:
        flat.extend(spec.indices)
    batch = accessor.batched(flat)
    offset = 0
    for spec in chunk:
        n = len(spec.indices)
        data, sizes = batch.data[offset : offset + n], batch.sizes[offset : offset + n]
        offset += n
        if spec.is_symmetric:
            na = spec.rows[0]
            x, targets = Batch(data[:na], sizes[:na]), Batch(data, sizes)
        else:
            x, targets = Batch(data[: spec.nx], sizes[: spec.nx]), Batch(data[spec.nx :], sizes[spec.nx :])
        mask = None if spec.mask is None else torch.from_numpy(spec.mask).to(accessor.device)
        yield CellGroup(x=x, targets=targets, rows=spec.rows, positions=spec.positions, mask=mask)


def group_cells(task: Task, constraints: Constraints | None = None) -> Generator[CellGroup, None, None]:
    """Yield groups of cells sharing the same X and A sample sets, gathering many groups per ``batched`` call.

    Each group's targets (A first, then every cell's B, with X prepended when X != A) are concatenated into a single
    index list. Groups are length-sorted and gathered in chunks of up to ``GATHER_CHUNK_ROWS``
    rows, one ``batched`` call per chunk, then sliced back into individual groups.

    With ``constraints``, :py:func:`fastabx.constraints.apply_constraints` adds an ``is_valid`` column to the cells.
    Each cell's mask is reshaped to ``(nx, na, nb)`` and the group's cells are concatenated along the B axis
    so the mask lines up column-for-column with the group's B targets.
    """
    accessor = task.dataset.accessor
    constrained = constraints is not None
    grouped = (
        (
            apply_constraints(task.cells, task.dataset.labels, constraints, is_symmetric=task.is_symmetric)
            if constrained
            else task.cells
        )
        .lazy()
        .with_row_index("__pos")
        .group_by(["index_x", "index_a"], maintain_order=True)
        .agg(pl.col("__pos"), pl.col("index_b"), *([pl.col("is_valid")] if constrained else []))
        .collect()
    )

    specs = []
    for index_x, index_a, positions, b_lists, *valids in grouped.iter_rows():
        nx, na = len(index_x), len(index_a)
        indices = list(index_a) if task.is_symmetric else [*index_x, *index_a]
        rows = [na]
        for index_b in b_lists:
            indices.extend(index_b)
            rows.append(len(index_b))
        if constrained:
            parts = [np.asarray(v, dtype=bool).reshape(nx, na, nb) for v, nb in zip(valids[0], rows[1:], strict=True)]
            mask = np.concatenate(parts, axis=2)
        else:
            mask = None
        smax = int(accessor.lengths(indices).max())
        specs.append(GroupSpec(smax, task.is_symmetric, nx, indices, rows, list(positions), mask))
    specs.sort(key=lambda spec: spec.smax)

    chunk, chunk_rows = [], 0
    for spec in specs:
        if chunk and chunk_rows + len(spec.indices) > GATHER_CHUNK_ROWS:
            yield from gather_chunk(accessor, chunk)
            chunk, chunk_rows = [], 0
        chunk.append(spec)
        chunk_rows += len(spec.indices)
    if chunk:
        yield from gather_chunk(accessor, chunk)


def grouped_distances(
    x: Tensor,
    sx: Tensor,
    targets: Tensor,
    target_sizes: Tensor,
    distance: Distance,
    *,
    use_dtw: bool,
) -> Tensor:
    """Distance matrix between the shared ``x`` and every target of a group, in as few launches as possible.

    Groups with more than ``MAX_SCORE_CHUNK_ROWS`` rows are scored in row-chunks to bound the peak memory cost.

    :param x: The group's X samples already concatenated and padded to a common length.
    :param sx: The real lengths of the X samples.
    :param targets: The group's targets already concatenated and padded to a common length (the A samples first,
        then every cell's B samples; built in one gather by :py:func:`fastabx.group.group_cells`).
    :param target_sizes: The real lengths of the targets.
    :param distance: The distance function to use.
    :param use_dtw: Whether to use DTW or not. DTW is needed unless every sample is pooled (time dimension of 1).
    :returns: A ``(x.size(0), targets.size(0))`` tensor of distances in the target order.
    """
    total = targets.size(0)
    if total <= MAX_SCORE_CHUNK_ROWS:
        return distance_matrix(x, sx, targets, target_sizes, distance, use_dtw=use_dtw, symmetric=False)
    out = x.new_empty(x.size(0), total)
    for start in range(0, total, MAX_SCORE_CHUNK_ROWS):
        end = min(start + MAX_SCORE_CHUNK_ROWS, total)
        chunk, chunk_sizes = targets[start:end], target_sizes[start:end]
        out[:, start:end] = distance_matrix(x, sx, chunk, chunk_sizes, distance, use_dtw=use_dtw, symmetric=False)
    return out


def grouped_contributions(dxa: Tensor, dxb_all: Tensor, mask: Tensor | None = None) -> Tensor:
    """Per-B-column ABX contribution of a group: ``0.5 * (1 - sign(dxa - dxb))`` summed over the X and A axes.

    :param dxa: The shared ``(nx, na)`` X-to-A distance (diagonal already set to infinity for symmetric cells).
    :param dxb_all: The concatenation of every cell's B columns, ``(nx, sum(b_rows))``.
    :param mask: The optional ``(nx, na, sum(b_rows))`` per-triplet constraint mask (every cell's ``(nx, na, nb)``
        mask concatenated along the B axis, in the same order as the B blocks of ``dxb_all``);
        ``None`` when scoring without constraints.
    :returns: A 1D ``(dxb_all.size(1),)`` tensor of half-integer counts.
    """
    nx, na = dxa.size()
    diff = dxa.unsqueeze(2) - dxb_all.unsqueeze(1)
    if mask is not None:
        return (0.5 * (1 - torch.sign(diff)) * mask).sum(dim=(0, 1))
    return 0.5 * (nx * na - torch.sign(diff).sum(dim=(0, 1)))


class GroupReducer:
    """Accumulate per-group ABX contributions and reduce them to per-cell scores in batched passes.

    For each group the cheap, unavoidable part — the half-integer count per B column
    (:py:func:`grouped_contributions`) — is computed eagerly. The per-group segment machinery that turns those
    counts into per-cell scores (a host→device index build plus an ``index_add_`` and a division) is instead
    amortised over many groups: it runs once per ``REDUCTION_FLUSH_COLS`` columns rather than once per
    group. This is bit-identical to the per-group reduction (the counts are exact half-integers), but removes the
    per-group overhead that dominates when groups are tiny (``nx ≈ na ≈ 2``), as in the across-speaker task.
    """

    def __init__(self, num_cells: int, *, constrained: bool = False) -> None:
        self.constrained = constrained
        self.scores = torch.full((num_cells,), float("nan"))  # per-cell score, written back by position
        self.sizes: list[int | None] = [0] * num_cells
        self._per_b: list[torch.Tensor] = []  # per-group (sum(b_rows),) half-integer counts
        self._per_b_valid: list[torch.Tensor] = []  # per-group (sum(b_rows),) valid-triplet counts (constrained)
        self._positions: list[int] = []  # cell position in the DataFrame, one per cell
        self._nb: list[int] = []  # number of B per cell
        self._cols = 0

    def add(self, group: CellGroup, distance: Distance, *, is_symmetric: bool) -> None:
        """Register a group's distance matrix: keep its per-B counts, record per-cell metadata, flush if full.

        :param group: The group's gathered data and metadata.
        :param distance: The distance function to use.
        :param is_symmetric: Whether the group is symmetric (X == A) or not.
        """
        distances = grouped_distances(
            group.x.data,
            group.x.sizes,
            group.targets.data,
            group.targets.sizes,
            distance,
            use_dtw=group.targets.data.size(1) > 1 or group.x.data.size(1) > 1,
        )
        na, nx = group.rows[0], group.x.data.size(0)
        dxa = distances[:, :na]
        if is_symmetric:
            dxa.fill_diagonal_(float("inf"))

        self._per_b.append(grouped_contributions(dxa, distances[:, na:], group.mask))
        if self.constrained:
            if group.mask is None:
                raise NoConstraintsError
            self._per_b_valid.append(group.mask.sum(dim=(0, 1)))

        factor = na * ((na - 1) if is_symmetric else nx)
        for position, nb in zip(group.positions, group.rows[1:], strict=True):
            self._positions.append(position)
            self._nb.append(nb)
            if not self.constrained:
                self.sizes[position] = nb * factor

        self._cols += distances.size(1) - na
        if self._cols >= REDUCTION_FLUSH_COLS:
            self.flush()

    def flush(self) -> None:
        """Reduce all buffered groups in one pass: one ``index_add_`` over the concatenated per-B counts."""
        if not self._per_b:
            return
        per_b_all = torch.cat(self._per_b)
        device = per_b_all.device
        positions = self._positions
        n_cells = len(positions)
        cell_ids = torch.from_numpy(np.repeat(np.arange(n_cells), self._nb)).to(device)
        counts = per_b_all.new_zeros(n_cells).index_add_(0, cell_ids, per_b_all)

        if self.constrained:
            valid_all = torch.cat(self._per_b_valid).to(counts.dtype)
            denom = counts.new_zeros(n_cells).index_add_(0, cell_ids, valid_all)
            for size, position in zip(denom.tolist(), positions, strict=True):
                self.sizes[position] = int(size) if size > 0 else None
        else:
            denom = torch.tensor([self.sizes[p] for p in positions], device=device, dtype=counts.dtype)

        cell_scores = 1 - counts / denom
        self.scores[torch.tensor(positions)] = cell_scores.cpu()
        self._per_b, self._per_b_valid, self._positions, self._nb, self._cols = [], [], [], [], 0

    def finalize(self) -> tuple[list[float | None], list[int | None]]:
        """Return the final scores and sizes for each cell, flushing any remaining groups."""
        self.flush()
        values = self.scores.tolist()
        scores = [None if math.isnan(v) else v for v in values] if self.constrained else list(values)
        return scores, self.sizes
