"""Grouped scoring engine: gather cells sharing the same (X, A) and reduce their distances to per-cell scores."""

import math
from collections.abc import Generator
from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
from torch import Tensor

from fastabx.accessor import Batch
from fastabx.alignment import Alignment
from fastabx.constraints import Constraints, NoConstraintsError, apply_constraints
from fastabx.distance import Distance, distance_matrix
from fastabx.task import Task
from fastabx.utils import gather_chunk_rows, max_score_chunk_rows, reduction_flush_cols

__all__ = []

MAX_FLOAT32_SIZE = 2**23
# Keep the largest temporary X-by-A-by-B comparison at a size that is efficient for
# vectorized kernels while avoiding the pathological multi-gigabyte broadcast.  This
# is deliberately an implementation detail, not a user-facing memory-budget API.
MAX_TRIPLET_CHUNK_ELEMENTS = 2**24


def contribution_dtype(size: int, input_dtype: torch.dtype) -> torch.dtype:
    """Choose an accumulator that represents every possible half-integer contribution exactly."""
    return torch.float64 if size > MAX_FLOAT32_SIZE or input_dtype == torch.float64 else torch.float32


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
    mask: np.ndarray | Tensor | None = None
    completed: int | None = None


@dataclass(frozen=True, slots=True)
class GroupSpec:
    """A group's host-side description, before its data is gathered.

    :param smax: The max sequence length among its samples (used to length-sort groups so batched gathers pad tightly).
    :param is_symmetric: Whether the group is symmetric (X == A) or not.
    :param nx: The X count.
    :param indices: The flat gather list (X (only when X != A), then A, then every B).
    :param rows: The per-target row counts.
    :param positions: The cells' DataFrame positions.
    :param b_lists: The B indices for each cell fragment. Kept on the host so constraint masks can be built lazily.
    :param completed: Number of complete cells represented by this fragment, for progress reporting.
    """

    smax: int
    is_symmetric: bool
    nx: int
    indices: list[int]
    rows: list[int]
    positions: list[int]
    b_lists: list[list[int]]
    completed: int | None


def _constraint_masks(task: Task, chunk: list[GroupSpec], constraints: Constraints | None) -> list[np.ndarray | None]:
    """Evaluate one bounded gather chunk and split its host-side masks back into group fragments."""
    if constraints is None:
        return [None] * len(chunk)
    index_x: list[list[int]] = []
    index_a: list[list[int]] = []
    index_b: list[list[int]] = []
    for spec in chunk:
        xs = spec.indices[: spec.nx] if not spec.is_symmetric else spec.indices[: spec.rows[0]]
        a_start = spec.nx if not spec.is_symmetric else 0
        a = spec.indices[a_start : a_start + spec.rows[0]]
        index_x.extend([list(xs) for _ in spec.b_lists])
        index_a.extend([list(a) for _ in spec.b_lists])
        index_b.extend(spec.b_lists)
    fragment_cells = pl.DataFrame({"index_x": index_x, "index_a": index_a, "index_b": index_b})
    valid = iter(
        apply_constraints(fragment_cells, task.dataset.labels, constraints, is_symmetric=task.is_symmetric)["is_valid"]
    )
    masks = []
    for spec in chunk:
        parts = [np.asarray(next(valid), dtype=bool).reshape(spec.nx, spec.rows[0], nb) for nb in spec.rows[1:]]
        masks.append(np.concatenate(parts, axis=2))
    return masks


def gather_chunk(
    task: Task, chunk: list[GroupSpec], constraints: Constraints | None
) -> Generator[CellGroup, None, None]:
    """Gather a chunk of groups in one ``accessor.batched`` call, then yield each group as a slice of the result.

    :param task: The task whose features and labels are being scored.
    :param chunk: A list of :py:class:`.GroupSpec` objects, each describing a group to gather.
    :returns: A generator of :py:class:`.CellGroup` objects, one per group in ``chunk``.
    """
    accessor = task.dataset.accessor
    flat = []
    for spec in chunk:
        flat.extend(spec.indices)
    batch = accessor.batched(flat)
    masks = _constraint_masks(task, chunk, constraints)
    offset = 0
    for spec, mask in zip(chunk, masks, strict=True):
        n = len(spec.indices)
        data, sizes = batch.data[offset : offset + n], batch.sizes[offset : offset + n]
        offset += n
        if spec.is_symmetric:
            na = spec.rows[0]
            x, targets = Batch(data[:na], sizes[:na]), Batch(data, sizes)
        else:
            x, targets = Batch(data[: spec.nx], sizes[: spec.nx]), Batch(data[spec.nx :], sizes[spec.nx :])
        yield CellGroup(
            x=x,
            targets=targets,
            rows=spec.rows,
            positions=spec.positions,
            mask=mask,
            completed=spec.completed,
        )


def _group_specs(
    index_x: list[int],
    index_a: list[int],
    positions: list[int],
    b_lists: list[list[int]],
    *,
    is_symmetric: bool,
    max_rows: int,
) -> Generator[GroupSpec, None, None]:
    """Split a logical group into bounded gather/comparison fragments."""
    nx, na = len(index_x), len(index_a)
    prefix = list(index_a) if is_symmetric else [*index_x, *index_a]
    # A very large X/A set cannot be divided here without duplicating pairwise work,
    # but B is always bounded. The scoring path separately chunks its comparison tensor.
    row_capacity = max(1, max_rows - len(prefix))
    triplet_capacity = max(1, MAX_TRIPLET_CHUNK_ELEMENTS // max(1, nx * na))
    capacity = min(row_capacity, triplet_capacity)
    fragment_positions: list[int] = []
    fragment_b: list[list[int]] = []
    fragment_rows = 0
    completed = 0

    def make_spec() -> GroupSpec:
        indices = [*prefix, *(index for values in fragment_b for index in values)]
        smax = 0  # Filled by group_cells after all indices are known.
        return GroupSpec(
            smax,
            is_symmetric,
            nx,
            indices,
            [na, *(len(values) for values in fragment_b)],
            list(fragment_positions),
            [list(values) for values in fragment_b],
            completed,
        )

    for position, values in zip(positions, b_lists, strict=True):
        for start in range(0, len(values), capacity):
            part = values[start : start + capacity]
            if fragment_b and fragment_rows + len(part) > capacity:
                yield make_spec()
                fragment_positions, fragment_b, fragment_rows, completed = [], [], 0, 0
            fragment_positions.append(position)
            fragment_b.append(part)
            fragment_rows += len(part)
            completed += int(start + len(part) == len(values))
    if fragment_b:
        yield make_spec()


def group_cells(task: Task, *, constraints: Constraints | None = None) -> Generator[CellGroup, None, None]:
    """Yield groups of cells sharing the same X and A sample sets, gathering many groups per ``batched`` call.

    Each group's targets (A first, then every cell's B, with X prepended when X != A) are concatenated into a single
    index list. Groups are length-sorted and gathered in chunks of up to :py:func:`.gather_chunk_rows`
    rows, one ``batched`` call per chunk, then sliced back into individual groups.

    With ``constraints``, each bounded fragment is evaluated immediately before it is yielded. Its mask stays on
    the host and is copied to the scoring device in smaller slices, so neither a full-task mask nor a full-group
    device mask is retained.
    """
    accessor = task.dataset.accessor
    constraints = None if constraints is None else list(constraints)
    grouped = (
        task.cells.lazy()
        .with_row_index("__pos")
        .group_by(["index_x", "index_a"], maintain_order=True)
        .agg(pl.col("__pos"), pl.col("index_b"))
        .collect()
    )

    specs = []
    max_chunk_rows = gather_chunk_rows()
    for index_x, index_a, positions, b_lists in grouped.iter_rows():
        nx, na = len(index_x), len(index_a)
        prefix = list(index_a) if task.is_symmetric else [*index_x, *index_a]
        total_b = sum(map(len, b_lists))
        if len(prefix) + total_b <= max_chunk_rows and nx * na * total_b <= MAX_TRIPLET_CHUNK_ELEMENTS:
            indices = [*prefix, *(index for values in b_lists for index in values)]
            specs.append(
                GroupSpec(
                    int(accessor.lengths(indices).max()),
                    task.is_symmetric,
                    nx,
                    indices,
                    [na, *(len(values) for values in b_lists)],
                    positions,
                    b_lists,
                    None,
                )
            )
            continue
        for spec in _group_specs(
            index_x,
            index_a,
            positions,
            b_lists,
            is_symmetric=task.is_symmetric,
            max_rows=max_chunk_rows,
        ):
            smax = int(accessor.lengths(spec.indices).max())
            specs.append(
                GroupSpec(
                    smax,
                    spec.is_symmetric,
                    spec.nx,
                    spec.indices,
                    spec.rows,
                    spec.positions,
                    spec.b_lists,
                    spec.completed,
                )
            )
    specs.sort(key=lambda spec: spec.smax)

    chunk, chunk_rows, chunk_triplets = [], 0, 0
    for spec in specs:
        triplets = spec.nx * spec.rows[0] * sum(spec.rows[1:])
        if chunk and (
            chunk_rows + len(spec.indices) > max_chunk_rows or chunk_triplets + triplets > MAX_TRIPLET_CHUNK_ELEMENTS
        ):
            yield from gather_chunk(task, chunk, constraints)
            chunk, chunk_rows, chunk_triplets = [], 0, 0
        chunk.append(spec)
        chunk_rows += len(spec.indices)
        chunk_triplets += triplets
    if chunk:
        yield from gather_chunk(task, chunk, constraints)


def grouped_distances(
    x: Tensor,
    sx: Tensor,
    targets: Tensor,
    target_sizes: Tensor,
    distance: Distance,
    *,
    alignment: Alignment,
    max_rows: int,
) -> Tensor:
    """Distance matrix between the shared ``x`` and every target of a group, in as few launches as possible.

    Groups with more than ``max_rows`` rows are scored in row-chunks to bound the peak memory cost.

    :param x: The group's X samples already concatenated and padded to a common length.
    :param sx: The real lengths of the X samples.
    :param targets: The group's targets already concatenated and padded to a common length (the A samples first,
        then every cell's B samples; built in one gather by :py:func:`fastabx.group.group_cells`).
    :param target_sizes: The real lengths of the targets.
    :param distance: The distance function to use.
    :param alignment: The alignment used to reduce the frame-level cost lattice to one distance per pair.
        Bypassed when every sample is pooled (time dimension of 1).
    :param max_rows: Maximum number of target rows compared in one go, from :py:func:`.max_score_chunk_rows`.
    :returns: A ``(x.size(0), targets.size(0))`` tensor of distances in the target order.
    """
    total = targets.size(0)
    if total <= max_rows:
        return distance_matrix(x, sx, targets, target_sizes, distance, alignment=alignment, symmetric=False)
    out = x.new_empty(x.size(0), total, dtype=x.dtype if x.is_floating_point() else torch.float32)
    for start in range(0, total, max_rows):
        end = min(start + max_rows, total)
        chunk, chunk_sizes = targets[start:end], target_sizes[start:end]
        out[:, start:end] = distance_matrix(x, sx, chunk, chunk_sizes, distance, alignment=alignment, symmetric=False)
    return out


def _contribution_chunk(dxa: Tensor, dxb: Tensor, mask: np.ndarray | Tensor | None) -> Tensor:
    """Reduce one bounded block of B columns, reusing its broadcast temporary in place."""
    nx, na = dxa.size()
    diff = dxa.unsqueeze(2) - dxb.unsqueeze(1)
    dtype = contribution_dtype(nx * na, diff.dtype)
    if mask is None:
        return 0.5 * (nx * na - diff.sign_().sum(dim=(0, 1), dtype=dtype))
    mask_tensor = torch.as_tensor(mask, device=diff.device, dtype=torch.bool)
    diff.sign_().neg_().add_(1).mul_(0.5).mul_(mask_tensor)
    return diff.sum(dim=(0, 1), dtype=dtype)


def grouped_contributions(dxa: Tensor, dxb_all: Tensor, mask: np.ndarray | Tensor | None = None) -> Tensor:
    """Per-B-column ABX contribution of a group: ``0.5 * (1 - sign(dxa - dxb))`` summed over the X and A axes.

    :param dxa: The shared ``(nx, na)`` X-to-A distance (diagonal already set to infinity for symmetric cells).
    :param dxb_all: The concatenation of every cell's B columns, ``(nx, sum(b_rows))``.
    :param mask: The optional ``(nx, na, sum(b_rows))`` per-triplet constraint mask (every cell's ``(nx, na, nb)``
        mask concatenated along the B axis, in the same order as the B blocks of ``dxb_all``);
        ``None`` when scoring without constraints.
    :returns: A 1D ``(dxb_all.size(1),)`` tensor of half-integer counts.
    """
    nx, na = dxa.size()
    max_cols = max(1, MAX_TRIPLET_CHUNK_ELEMENTS // max(1, nx * na))
    if dxb_all.size(1) <= max_cols:
        return _contribution_chunk(dxa, dxb_all, mask)
    parts = []
    for start in range(0, dxb_all.size(1), max_cols):
        end = min(start + max_cols, dxb_all.size(1))
        mask_chunk = None if mask is None else mask[:, :, start:end]
        parts.append(_contribution_chunk(dxa, dxb_all[:, start:end], mask_chunk))
    return torch.cat(parts)


class GroupReducer:
    """Accumulate per-group ABX contributions and reduce them to per-cell scores in batched passes.

    For each group the cheap, unavoidable part — the half-integer count per B column
    (:py:func:`grouped_contributions`) — is computed eagerly. The per-group segment machinery that turns those
    counts into per-cell scores (a host→device index build plus an ``index_add_`` and a division) is instead
    amortised over many groups: it runs once per :py:func:`.reduction_flush_cols` columns rather than once per
    group. This is bit-identical to the per-group reduction (the counts are exact half-integers), but removes the
    per-group overhead that dominates when groups are tiny (``nx ≈ na ≈ 2``), as in the across-speaker task.
    """

    def __init__(self, num_cells: int, *, constrained: bool = False) -> None:
        self.constrained = constrained
        self.scores = torch.full((num_cells,), float("nan"))  # per-cell score, written back by position
        self.sizes: list[int | None] = [0] * num_cells
        self._counts: torch.Tensor | None = None
        self._has_fragments = False
        self._per_b: list[torch.Tensor] = []  # per-group (sum(b_rows),) half-integer counts
        self._per_b_valid: list[torch.Tensor] = []  # per-group (sum(b_rows),) valid-triplet counts (constrained)
        self._positions: list[int] = []  # cell position in the DataFrame, one per cell
        self._nb: list[int] = []  # number of B per cell
        self._cols = 0
        self._flush_cols = reduction_flush_cols()
        self._max_score_rows = max_score_chunk_rows()

    def add(self, group: CellGroup, distance: Distance, *, alignment: Alignment, is_symmetric: bool) -> None:
        """Register a group's distance matrix: keep its per-B counts, record per-cell metadata, flush if full.

        :param group: The group's gathered data and metadata.
        :param distance: The distance function to use.
        :param alignment: The alignment used to reduce the frame-level cost lattice to one distance per pair.
        :param is_symmetric: Whether the group is symmetric (X == A) or not.
        """
        na, nx = group.rows[0], group.x.data.size(0)
        self._has_fragments |= group.completed is not None and group.completed != len(group.positions)
        targets, target_sizes = group.targets.data, group.targets.sizes
        if targets.size(0) <= self._max_score_rows:
            distances = distance_matrix(
                group.x.data,
                group.x.sizes,
                targets,
                target_sizes,
                distance,
                alignment=alignment,
                symmetric=False,
            )
            dxa = distances[:, :na]
            if is_symmetric:
                dxa.fill_diagonal_(float("inf"))
            per_b = grouped_contributions(dxa, distances[:, na:], group.mask)
        else:
            dxa = grouped_distances(
                group.x.data,
                group.x.sizes,
                targets[:na],
                target_sizes[:na],
                distance,
                alignment=alignment,
                max_rows=self._max_score_rows,
            )
            if is_symmetric:
                dxa.fill_diagonal_(float("inf"))
            parts = []
            for start in range(na, targets.size(0), self._max_score_rows):
                end = min(start + self._max_score_rows, targets.size(0))
                dxb = distance_matrix(
                    group.x.data,
                    group.x.sizes,
                    targets[start:end],
                    target_sizes[start:end],
                    distance,
                    alignment=alignment,
                    symmetric=False,
                )
                mask = None if group.mask is None else group.mask[:, :, start - na : end - na]
                parts.append(grouped_contributions(dxa, dxb, mask))
            per_b = torch.cat(parts)
        self._per_b.append(per_b)
        if self.constrained:
            if group.mask is None:
                raise NoConstraintsError
            if isinstance(group.mask, np.ndarray):
                valid = torch.from_numpy(group.mask.sum(axis=(0, 1), dtype=np.int64))
            else:
                valid = group.mask.sum(dim=(0, 1), dtype=torch.int64).to("cpu")
            self._per_b_valid.append(valid)

        factor = na * ((na - 1) if is_symmetric else nx)
        for position, nb in zip(group.positions, group.rows[1:], strict=True):
            self._positions.append(position)
            self._nb.append(nb)
            if not self.constrained:
                size = self.sizes[position]
                self.sizes[position] = (0 if size is None else size) + nb * factor

        self._cols += targets.size(0) - na
        if self._cols >= self._flush_cols:
            self.flush()

    def flush(self) -> None:
        """Reduce all buffered groups in one pass: one ``index_add_`` over the concatenated per-B counts."""
        if not self._per_b:
            return
        per_b_all = torch.cat(self._per_b).to(torch.float64)
        device = per_b_all.device
        positions = self._positions
        n_cells = len(positions)
        cell_ids = torch.from_numpy(np.repeat(np.arange(n_cells), self._nb)).to(device)
        counts = per_b_all.new_zeros(n_cells).index_add_(0, cell_ids, per_b_all)

        if self.constrained:
            valid_all = torch.cat(self._per_b_valid).to(device)
            denom = valid_all.new_zeros(n_cells).index_add_(0, cell_ids, valid_all)
        else:
            denom = torch.tensor([self.sizes[p] for p in positions], device=device, dtype=torch.int64)

        if not self._has_fragments:
            if self.constrained:
                for size, position in zip(denom.tolist(), positions, strict=True):
                    self.sizes[position] = int(size) if size > 0 else None
            cell_scores = 1 - counts / denom
            self.scores[torch.tensor(positions)] = cell_scores.to(device="cpu", dtype=self.scores.dtype)
        else:
            if self._counts is None:
                self._counts = torch.zeros_like(self.scores, dtype=torch.float64)
            cpu_positions = torch.tensor(positions)
            self._counts.index_add_(0, cpu_positions, counts.to(device="cpu", dtype=torch.float64))
            if self.constrained:
                for size, position in zip(denom.tolist(), positions, strict=True):
                    previous = self.sizes[position]
                    self.sizes[position] = (0 if previous is None else previous) + int(size)
        self._per_b, self._per_b_valid, self._positions, self._nb, self._cols = [], [], [], [], 0

    def finalize(self) -> tuple[list[float | None], list[int | None]]:
        """Return the final scores and sizes for each cell, flushing any remaining groups."""
        self.flush()
        if self._counts is not None:
            denom = torch.tensor([0 if size is None else size for size in self.sizes], dtype=torch.float64)
            pending = torch.isnan(self.scores) & (denom > 0)
            self.scores[pending] = (1 - self._counts[pending] / denom[pending]).to(self.scores.dtype)
        values = self.scores.tolist()
        if self.constrained:
            scores = [None if math.isnan(v) else v for v in values]
            sizes = [size or None for size in self.sizes]
        else:
            scores, sizes = list(values), self.sizes
        return scores, sizes
