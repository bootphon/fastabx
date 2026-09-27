"""Grouped scoring engine: gather cells sharing the same (X, A) and reduce their distances to per-cell scores."""

import math
from collections.abc import Generator
from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
from torch import Tensor

from fastabx.accessor import Accessor, Batch
from fastabx.alignment import Alignment
from fastabx.constraints import ConstraintMask, Constraints, NoConstraintsError
from fastabx.distance import Distance, NaNDistanceError, distance_matrix
from fastabx.task import Task
from fastabx.utils import gather_chunk_rows, max_lattice_elements, max_score_chunk_rows, reduction_flush_cols

__all__ = []

# Number of (x, a, b) triplets compared at once when counting under constraints. Each one costs a few bytes of
# boolean temporaries (its validity and two comparisons), so this bounds that step whatever the size of the group.
CONSTRAINED_COUNT_CHUNK = 2**24


@dataclass(frozen=True, slots=True)
class GroupMask:
    """The constraints of a group, evaluated block by block of B columns when its triplets are counted.

    :param constraints: The task's :py:class:`.ConstraintMask`.
    :param x: The X datapoint indices, on the device.
    :param a: The A datapoint indices, on the device.
    :param b: Every cell's B datapoint indices, concatenated, on the device.
    """

    constraints: ConstraintMask
    x: Tensor
    a: Tensor
    b: Tensor

    def block(self, start: int, end: int) -> Tensor:
        """Return the ``(nx, na, end - start)`` validity of the triplets with the B columns ``[start, end[``."""
        return self.constraints.block(self.x, self.a, self.b[start:end])


def mask_block(mask: Tensor | GroupMask, start: int, end: int, device: torch.device) -> Tensor:
    """Columns ``[start, end[`` of a constraint mask, as a boolean tensor on ``device``."""
    if isinstance(mask, GroupMask):
        return mask.block(start, end)
    return mask[:, :, start:end].to(device=device, dtype=torch.bool)


@dataclass(frozen=True, slots=True)
class LazyTargets:
    """The targets of a group too large to gather at once, read from the accessor chunk by chunk when compared.

    :param accessor: The dataset's :py:class:`.Accessor`.
    :param indices: The targets' datapoint indices (the A samples first, then every cell's B samples).
    :param smax: The max sequence length among the targets, bounding the padded length of any chunk.
    """

    accessor: Accessor
    indices: list[int]
    smax: int


def target_rows(targets: Batch | LazyTargets, start: int, end: int) -> Batch:
    """Rows ``[start, end[`` of the targets, gathering them if they are lazy."""
    if isinstance(targets, LazyTargets):
        return targets.accessor.batched(targets.indices[start:end])
    return Batch(targets.data[start:end], targets.sizes[start:end])


def num_targets(targets: Batch | LazyTargets) -> int:
    """Count the targets, gathered or not."""
    return len(targets.indices) if isinstance(targets, LazyTargets) else targets.data.size(0)


def max_target_length(targets: Batch | LazyTargets) -> int:
    """Return the padded length of the targets, or an upper bound on it for lazy ones."""
    return targets.smax if isinstance(targets, LazyTargets) else targets.data.size(1)


@dataclass(frozen=True, slots=True)
class CellGroup:
    """A group of cells that share the same X and A sample sets.

    :param x: The shared X samples, gathered and padded to a common length.
    :param targets: The A samples (the first ``rows[0]`` rows) and then every cell's B samples. Usually gathered
        and padded together in a single ``accessor.batched`` call instead of one call per A/B; a group with more
        rows than :py:func:`.gather_chunk_rows` keeps them as :py:class:`LazyTargets` instead.
    :param rows: The per-target row counts (``rows[0]`` = na, ``rows[1:]`` = nb per cell).
    :param positions: The position of each cell in the task's ``cells`` DataFrame, so the A block, the per-cell B
        blocks, and the scores can be sliced/written back in the original order without rebuilding anything.
    :param mask: The optional constraints of the group, evaluated on its ``(nx, na, sum(b_rows))`` triplets (every
        cell's B concatenated, in the same order as the B blocks of ``targets``) one block of B columns at a time;
        ``None`` when scoring without constraints.
    """

    x: Batch
    targets: Batch | LazyTargets
    rows: list[int]
    positions: list[int]
    mask: GroupMask | None = None


@dataclass(frozen=True, slots=True)
class GroupSpec:
    """A group's host-side description, before its data is gathered.

    :param smax: The max sequence length among its samples (used to length-sort groups so batched gathers pad tightly).
    :param is_symmetric: Whether the group is symmetric (X == A) or not.
    :param nx: The X count.
    :param indices: The flat gather list (X (only when X != A), then A, then every B).
    :param rows: The per-target row counts.
    :param positions: The cells' DataFrame positions.
    """

    smax: int
    is_symmetric: bool
    nx: int
    indices: list[int]
    rows: list[int]
    positions: list[int]


def group_mask(spec: GroupSpec, constraints: ConstraintMask | None) -> GroupMask | None:
    """Bind the constraints to the X, A and B datapoints of a group."""
    if constraints is None:
        return None
    x, targets = split_x(spec)
    na = spec.rows[0]
    x_index, a_index, b_index = (
        torch.tensor(indices, dtype=torch.int64, device=constraints.device)
        for indices in (x, targets[:na], targets[na:])
    )
    return GroupMask(constraints, x_index, a_index, b_index)


def split_x(spec: GroupSpec) -> tuple[list[int], list[int]]:
    """Split a group into its X indices and its target indices. X is the A block itself in a symmetric group."""
    if spec.is_symmetric:
        return spec.indices[: spec.rows[0]], spec.indices
    return spec.indices[: spec.nx], spec.indices[spec.nx :]


def gather_chunk(
    accessor: Accessor, chunk: list[GroupSpec], constraints: ConstraintMask | None
) -> Generator[CellGroup, None, None]:
    """Gather a chunk of groups in one ``accessor.batched`` call, then yield each group as a slice of the result.

    :param accessor: The dataset's :py:class:`.Accessor`.
    :param chunk: A list of :py:class:`.GroupSpec` objects, each describing a group to gather.
    :param constraints: The task's constraints, if any.
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
        if spec.smax < data.size(1):
            data = data[:, : spec.smax].contiguous()
        if spec.is_symmetric:
            na = spec.rows[0]
            x, targets = Batch(data[:na], sizes[:na]), Batch(data, sizes)
        else:
            x, targets = Batch(data[: spec.nx], sizes[: spec.nx]), Batch(data[spec.nx :], sizes[spec.nx :])
        mask = group_mask(spec, constraints)
        yield CellGroup(x=x, targets=targets, rows=spec.rows, positions=spec.positions, mask=mask)


def gather_oversized(accessor: Accessor, spec: GroupSpec, constraints: ConstraintMask | None) -> CellGroup:
    """Gather only the X of a group too large for one gather; its targets are read chunk by chunk when compared."""
    x_indices, target_indices = split_x(spec)
    targets = LazyTargets(accessor, target_indices, spec.smax)
    return CellGroup(
        x=accessor.batched(x_indices),
        targets=targets,
        rows=spec.rows,
        positions=spec.positions,
        mask=group_mask(spec, constraints),
    )


def group_cells(task: Task, *, constraints: Constraints | None = None) -> Generator[CellGroup, None, None]:
    """Yield groups of cells sharing the same X and A sample sets, gathering many groups per ``batched`` call.

    Each group's targets (A first, then every cell's B, with X prepended when X != A) are concatenated into a single
    index list. Groups are length-sorted and gathered in chunks of up to :py:func:`.gather_chunk_rows`
    rows, one ``batched`` call per chunk, then sliced back into individual groups. A single group larger than that
    only gathers its X up front, and its targets chunk by chunk as they are compared.

    With ``constraints``, each group carries a :py:class:`GroupMask`, evaluated one block of B columns at a time
    when its triplets are counted, so no per-triplet mask is ever stored.
    """
    accessor = task.dataset.accessor
    mask = None
    if constraints is not None:
        mask = ConstraintMask(task.dataset.labels, constraints, is_symmetric=task.is_symmetric, device=accessor.device)
    grouped = (
        task.cells.lazy()
        .with_row_index("__pos")
        .group_by(["index_x", "index_a"], maintain_order=True)
        .agg(pl.col("__pos"), pl.col("index_b"))
        .collect()
    )

    specs = []
    for index_x, index_a, positions, b_lists in grouped.iter_rows():
        nx, na = len(index_x), len(index_a)
        indices = list(index_a) if task.is_symmetric else [*index_x, *index_a]
        rows = [na]
        for index_b in b_lists:
            indices.extend(index_b)
            rows.append(len(index_b))
        smax = int(accessor.lengths(indices).max())
        specs.append(GroupSpec(smax, task.is_symmetric, nx, indices, rows, list(positions)))
    specs.sort(key=lambda spec: spec.smax)

    max_chunk_rows = gather_chunk_rows()
    chunk, chunk_rows = [], 0
    for spec in specs:
        if len(spec.indices) > max_chunk_rows:
            yield gather_oversized(accessor, spec, mask)
            continue
        if chunk and chunk_rows + len(spec.indices) > max_chunk_rows:
            yield from gather_chunk(accessor, chunk, mask)
            chunk, chunk_rows = [], 0
        chunk.append(spec)
        chunk_rows += len(spec.indices)
    if chunk:
        yield from gather_chunk(accessor, chunk, mask)


def chunk_shape(
    nx: int, total: int, frames_per_pair: int, *, max_rows: int, max_elements: int | None
) -> tuple[int, int]:
    """Choose how many X rows and target rows are compared at once.

    At most ``max_rows`` targets, and at most ``max_elements`` elements in the ``(x rows, target rows, frames,
    frames)`` lattice. The X side is only split when a single target row per call would still exceed the budget.
    """
    targets = min(total, max_rows)
    if max_elements is None:
        return nx, targets
    pairs = max(1, max_elements // frames_per_pair)
    targets = max(1, min(targets, pairs // nx))
    return min(nx, max(1, pairs // targets)), targets


def grouped_distances(
    x: Batch,
    targets: Batch | LazyTargets,
    distance: Distance,
    *,
    alignment: Alignment,
    max_rows: int,
    max_elements: int | None = None,
) -> Tensor:
    """Distance matrix between the shared ``x`` and every target of a group, in as few launches as possible.

    The comparisons are split into blocks of at most ``max_rows`` targets and at most ``max_elements`` lattice
    elements, to bound the peak memory cost. Lazy targets are gathered one block at a time.
    The output preserves the distance/alignment result dtype, which may differ from the feature dtype.

    :param x: The group's X samples, padded to a common length.
    :param targets: The group's targets (the A samples first, then every cell's B samples), either gathered and
        padded to a common length or :py:class:`LazyTargets`.
    :param distance: The distance function to use.
    :param alignment: The alignment used to reduce the frame-level cost lattice to one distance per pair.
        Bypassed when every sample is pooled (time dimension of 1).
    :param max_rows: Maximum number of target rows compared in one go, from :py:func:`.max_score_chunk_rows`.
    :param max_elements: Maximum number of elements of the frame-level lattice built in one go, from
        :py:func:`.max_lattice_elements`. ``None`` for no limit.
    :returns: A ``(x.size(0), number of targets)`` tensor of distances in the target order.
    """
    nx, total = x.data.size(0), num_targets(targets)
    frames_per_pair = x.data.size(1) * max_target_length(targets)
    x_rows, t_rows = chunk_shape(nx, total, frames_per_pair, max_rows=max_rows, max_elements=max_elements)
    if x_rows == nx and t_rows == total and not isinstance(targets, LazyTargets):
        return distance_matrix(
            x.data, x.sizes, targets.data, targets.sizes, distance, alignment=alignment, symmetric=False
        )

    def blocks() -> Generator[tuple[slice, slice, Tensor], None, None]:
        for t_start in range(0, total, t_rows):
            t_block = slice(t_start, min(t_start + t_rows, total))
            block = target_rows(targets, t_block.start, t_block.stop)
            for x_start in range(0, nx, x_rows):
                x_block = slice(x_start, min(x_start + x_rows, nx))
                yield (
                    x_block,
                    t_block,
                    distance_matrix(
                        x.data[x_block],
                        x.sizes[x_block],
                        block.data,
                        block.sizes,
                        distance,
                        alignment=alignment,
                        symmetric=False,
                    ),
                )

    results = blocks()
    x_block, t_block, first = next(results)
    out = first.new_empty(nx, total)
    out[x_block, t_block] = first
    for x_block, t_block, result in results:
        out[x_block, t_block] = result
    return out


def grouped_contributions(dxa: Tensor, dxb_all: Tensor, mask: Tensor | GroupMask | None = None) -> Tensor:
    """Per-B-column ABX count of a group, **doubled** so that it is an exact integer.

    Each ``(x, a, b)`` triplet counts 2 when X is closer to A than to B, 1 on a tie and 0 otherwise, which is twice
    ``0.5 * (1 - sign(dxa - dxb))``. Without a mask each X row of ``dxa`` is sorted once and every B is located in
    it by binary search, in ``O(nx (na + nb) log na)`` time and ``O(nx nb)`` memory, instead of materialising every
    triplet. With a mask, see :py:func:`masked_contributions`.

    :param dxa: The shared ``(nx, na)`` X-to-A distance (diagonal already set to infinity for symmetric cells).
    :param dxb_all: The concatenation of every cell's B columns, ``(nx, sum(b_rows))``.
    :param mask: The optional per-triplet constraints, see :py:func:`masked_contributions`.
    :returns: A 1D ``(dxb_all.size(1),)`` int64 tensor of doubled counts.
    """
    if mask is not None:
        return masked_contributions(dxa, dxb_all, mask)[0]
    if dxa.dtype in {torch.float16, torch.bfloat16}:  # Widening is exact and keeps every comparison unchanged.
        dxa, dxb_all = dxa.float(), dxb_all.float()
    sorted_dxa = dxa.sort(dim=1).values
    dxb_all = dxb_all.contiguous()
    closer = torch.searchsorted(sorted_dxa, dxb_all, side="left")  # number of A with dxa < dxb
    closer_or_tied = torch.searchsorted(sorted_dxa, dxb_all, side="right")  # number of A with dxa <= dxb
    return (closer + closer_or_tied).sum(dim=0)


def masked_contributions(dxa: Tensor, dxb_all: Tensor, mask: Tensor | GroupMask) -> tuple[Tensor, Tensor]:
    """Doubled per-B-column ABX counts over the valid triplets only, and the number of valid triplets per column.

    The triplets are compared in chunks of B columns of at most :py:data:`CONSTRAINED_COUNT_CHUNK` triplets, and
    the validity of each chunk is only built when it is compared.

    :param dxa: The shared ``(nx, na)`` X-to-A distance (diagonal already set to infinity for symmetric cells).
    :param dxb_all: The concatenation of every cell's B columns, ``(nx, sum(b_rows))``.
    :param mask: Either a :py:class:`GroupMask`, or a ``(nx, na, sum(b_rows))`` tensor that is nonzero for the valid
        triplets (every cell's ``(nx, na, nb)`` mask concatenated along the B axis, in the same order as the B
        blocks of ``dxb_all``), possibly on another device.
    :returns: Two 1D ``(dxb_all.size(1),)`` int64 tensors: the doubled counts and the valid triplet counts.
    """
    nx, na = dxa.size()
    total = dxb_all.size(1)
    doubled = torch.empty(total, dtype=torch.int64, device=dxa.device)
    valid_counts = torch.empty(total, dtype=torch.int64, device=dxa.device)
    step = max(1, CONSTRAINED_COUNT_CHUNK // (nx * na))
    dxa = dxa.unsqueeze(2)
    for start in range(0, total, step):
        end = min(start + step, total)
        dxb = dxb_all[:, None, start:end]
        valid = mask_block(mask, start, end, dxa.device)
        closer = ((dxa < dxb) & valid).sum(dim=(0, 1))
        tied = ((dxa == dxb) & valid).sum(dim=(0, 1))
        doubled[start:end] = 2 * closer + tied
        valid_counts[start:end] = valid.sum(dim=(0, 1))
    return doubled, valid_counts


class GroupReducer:
    """Accumulate per-group ABX counts and reduce them to per-cell scores in batched passes.

    For each group the cheap, unavoidable part — the doubled count per B column (:py:func:`grouped_contributions`)
    — is computed eagerly. The per-group segment machinery that turns those counts into per-cell scores (a
    host→device index build plus an ``index_add_`` and a division) is instead amortised over many groups: it runs
    once per :py:func:`.reduction_flush_cols` columns rather than once per group. The counts are exact integers,
    so this is bit-identical to a per-group reduction, but removes the per-group overhead that dominates when
    groups are tiny (``nx ≈ na ≈ 2``), as in the across-speaker task.
    """

    def __init__(self, num_cells: int, *, constrained: bool = False) -> None:
        self.constrained = constrained
        self.scores = torch.full((num_cells,), float("nan"))  # per-cell score, written back by position
        self.sizes: list[int | None] = [0] * num_cells
        self._per_b: list[torch.Tensor] = []  # per-group (sum(b_rows),) doubled counts
        self._per_b_valid: list[torch.Tensor] = []  # per-group (sum(b_rows),) valid-triplet counts (constrained)
        self._positions: list[int] = []  # cell position in the DataFrame, one per cell
        self._nb: list[int] = []  # number of B per cell
        self._cols = 0
        self._any_nan: Tensor | None = None  # device-side flag, only synchronised once per flush
        self._flush_cols = reduction_flush_cols()
        self._max_score_rows = max_score_chunk_rows()
        self._max_lattice_elements = max_lattice_elements()

    def add(self, group: CellGroup, distance: Distance, *, alignment: Alignment, is_symmetric: bool) -> None:
        """Register a group's distance matrix: keep its per-B counts, record per-cell metadata, flush if full.

        :param group: The group's gathered data and metadata.
        :param distance: The distance function to use.
        :param alignment: The alignment used to reduce the frame-level cost lattice to one distance per pair.
        :param is_symmetric: Whether the group is symmetric (X == A) or not.
        """
        if self.constrained and group.mask is None:
            raise NoConstraintsError
        distances = grouped_distances(
            group.x,
            group.targets,
            distance,
            alignment=alignment,
            max_rows=self._max_score_rows,
            max_elements=self._max_lattice_elements,
        )
        has_nan = torch.isnan(distances).any()
        self._any_nan = has_nan if self._any_nan is None else self._any_nan | has_nan
        na, nx = group.rows[0], group.x.data.size(0)
        dxa = distances[:, :na]
        if is_symmetric:
            dxa.fill_diagonal_(float("inf"))

        if group.mask is None:
            self._per_b.append(grouped_contributions(dxa, distances[:, na:]))
        else:
            doubled, valid = masked_contributions(dxa, distances[:, na:], group.mask)
            self._per_b.append(doubled)
            self._per_b_valid.append(valid)

        factor = na * ((na - 1) if is_symmetric else nx)
        for position, nb in zip(group.positions, group.rows[1:], strict=True):
            self._positions.append(position)
            self._nb.append(nb)
            if not self.constrained:
                self.sizes[position] = nb * factor

        self._cols += distances.size(1) - na
        if self._cols >= self._flush_cols:
            self.flush()

    def flush(self) -> None:
        """Reduce all buffered groups in one pass: one ``index_add_`` over the concatenated per-B counts."""
        if not self._per_b:
            return
        if self._any_nan is not None and self._any_nan.item():
            raise NaNDistanceError
        self._any_nan = None
        per_b_all = torch.cat(self._per_b)
        device = per_b_all.device
        positions = self._positions
        n_cells = len(positions)
        cell_ids = torch.from_numpy(np.repeat(np.arange(n_cells), self._nb)).to(device)
        doubled = per_b_all.new_zeros(n_cells).index_add_(0, cell_ids, per_b_all)

        if self.constrained:
            valid_all = torch.cat(self._per_b_valid)
            denom = valid_all.new_zeros(n_cells).index_add_(0, cell_ids, valid_all)
            for size, position in zip(denom.tolist(), positions, strict=True):
                self.sizes[position] = int(size) if size > 0 else None
        else:
            denom = torch.tensor([self.sizes[p] for p in positions], device=device, dtype=torch.int64)

        cell_scores = 1 - doubled.to(torch.float64) * 0.5 / denom
        self.scores[torch.tensor(positions)] = cell_scores.to(device="cpu", dtype=self.scores.dtype)
        self._per_b, self._per_b_valid, self._positions, self._nb, self._cols = [], [], [], [], 0

    def finalize(self) -> tuple[list[float | None], list[int | None]]:
        """Return the final scores and sizes for each cell, flushing any remaining groups."""
        self.flush()
        values = self.scores.tolist()
        scores = [None if math.isnan(v) else v for v in values] if self.constrained else list(values)
        return scores, self.sizes
