"""Bootstrap resampling of the items behind the ABX cells.

The resampling is expressed as *integer weights over the unique item pools* rather than as duplicated
row indices. That representation is what makes the whole thing tractable:

- the gathers and the distance matrices stay at unique-pool size, so all replicates reuse **one** distance
  computation per group instead of recomputing DTW once per replicate;
- excluding the degenerate ``x is a`` comparisons is automatic. When indices are duplicated instead, an item
  drawn twice lands at two positions and the ``d(x, a) = 0`` off-diagonal pairs between them survive the
  positional diagonal exclusion and are always scored correct, which biases the score. Over unique pools the
  diagonal *is* the identity exclusion, so the estimator stays the U-statistic it is meant to be.

Weights are drawn per *pool*, not per cell: cells that share an item pool (the same ``on``/``by``/``across``
label group appears as the A pool of one cell and the B pool of another) get the same draw within a replicate.
That is the coherent stratified bootstrap — resample the dataset once per replicate, then recompute every
statistic on it — rather than an independent draw per cell, which would understate the correlation between
cells that share tokens.
"""

import hashlib
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from fastabx.group import CellGroup, group_pools
from fastabx.utils import BOOTSTRAP_CHUNK_ELEMS
from fastabx.verify import verify_bootstrap_params

__all__ = ["Bootstrap"]


def pool_key(pool: Sequence[int]) -> bytes:
    """Stable content hash of an item pool, used both as a cache key and as the pool's seed material.

    Hashed rather than keyed on the tuple itself so the cost is linear once instead of on every lookup,
    and stable across runs (unlike :py:func:`hash`) so a given ``seed`` always reproduces the same draws.
    """
    return hashlib.blake2b(np.asarray(pool, dtype=np.int64).tobytes(), digest_size=16).digest()


class Bootstrap:
    """Bootstrap resampling scheme for an ABX :py:class:`.Task`.

    Pass it to :py:class:`.Score` to get, alongside the point estimate, ``n_replicates`` resampled
    scores per cell. Each replicate redraws every item pool with replacement, keeping its size
    (a stratified bootstrap over the ``on``/``by``/``across`` label groups).

    The draws are cached per pool and keyed on the pool's contents, so the *same* ``Bootstrap`` instance
    reused across two ``Score`` calls resamples both identically. That is what you want for a paired
    comparison of two models or two distances: the difference of their bootstrap distributions is then
    computed on matching resamples.

    .. warning::
        Bootstrap over a subsampled :py:class:`.Task` measures the variability *of the subsample*, not of
        the data. Build the ``Task`` without a :py:class:`.Subsampler` when bootstrapping.

    .. note::
        Cells with very few items produce degenerate replicates: a cell with ``na = 2`` draws the same item
        twice half the time, leaving no valid ``x != a`` pair. Those replicates score ``None`` and are skipped
        by the collapse, so tasks made of tiny cells (typically the "across" condition) need more replicates
        for a given precision.

    Memory: the cached draws take ``4 * n_replicates * (total pool size)`` bytes, and the per-replicate cell
    scores take ``8 * num_cells * n_replicates`` bytes.

    :param n_replicates: Number of bootstrap replicates to draw.
    :param seed: The random seed, default is 0.
    """

    def __init__(self, n_replicates: int, *, seed: int = 0) -> None:
        verify_bootstrap_params(n_replicates, seed=seed)
        self.n_replicates = n_replicates
        self.seed = seed
        self._weights: dict[bytes, npt.NDArray[np.int32]] = {}

    def __repr__(self) -> str:
        return f"Bootstrap(n_replicates={self.n_replicates}, seed={self.seed})"

    def weights(self, pool: Sequence[int]) -> npt.NDArray[np.int32]:
        """Draw the ``(n_replicates, len(pool))`` multiplicities of each item of ``pool`` in every replicate.

        Each row is a multinomial draw of ``len(pool)`` items over the pool, i.e. a with-replacement resample
        of the same size. Cached on the pool's contents.
        """
        key = pool_key(pool)
        if (cached := self._weights.get(key)) is not None:
            return cached
        n = len(pool)
        rng = np.random.default_rng([self.seed, int.from_bytes(key[:8], "little")])
        weights = rng.multinomial(n, np.full(n, 1 / n), size=self.n_replicates).astype(np.int32)
        self._weights[key] = weights
        return weights


def weighted_counts(dxa: Tensor, dxb_all: Tensor, wx: Tensor, wa: Tensor) -> Tensor:
    """Per-replicate, per-B-column ABX contribution, weighted by the bootstrap multiplicities of X and A.

    The unweighted :py:func:`fastabx.group.grouped_contributions` sums ``0.5 * (1 - sign(dxa - dxb))`` over the
    X and A axes; here that sum is instead contracted against the replicates' X and A weight vectors, which is
    the same thing with every (x, a) pair counted ``wx[r, x] * wa[r, a]`` times.

    The B axis is processed in chunks so the ``(n_replicates, na, chunk)`` intermediate stays bounded.

    :param dxa: The ``(nx, na)`` X-to-A distance (diagonal already set to infinity for symmetric groups, which
        makes those terms contribute 0 and is exactly the ``x is a`` exclusion).
    :param dxb_all: The concatenation of every cell's B columns, ``(nx, sum(b_rows))``.
    :param wx: The ``(n_replicates, nx)`` X multiplicities.
    :param wa: The ``(n_replicates, na)`` A multiplicities.
    :returns: A ``(n_replicates, sum(b_rows))`` tensor of weighted half-integer counts.
    """
    nx, na = dxa.size()
    nb, n_rep = dxb_all.size(1), wx.size(0)
    out = dxb_all.new_zeros((n_rep, nb))
    width = max(1, BOOTSTRAP_CHUNK_ELEMS // max(1, n_rep * na))
    for start in range(0, nb, width):
        end = min(start + width, nb)
        contribution = 0.5 * (1 - torch.sign(dxa.unsqueeze(2) - dxb_all[:, start:end].unsqueeze(1)))
        # (n_rep, nx) @ (nx, na * chunk) -> (n_rep, na, chunk), then contract the A axis against wa.
        over_x = (wx @ contribution.reshape(nx, -1)).view(n_rep, na, end - start)
        out[:, start:end] = torch.bmm(wa.unsqueeze(1), over_x).squeeze(1)
    return out


class BootstrapReducer:
    """Accumulate per-cell, per-replicate bootstrap scores from the same groups as :py:class:`.GroupReducer`.

    Kept separate from ``GroupReducer`` rather than folded into it: it consumes the group's already-computed
    distance matrix, so adding it to a scoring run costs no extra distance or DTW work.
    """

    def __init__(self, num_cells: int, bootstrap: Bootstrap) -> None:
        self.bootstrap = bootstrap
        shape = (num_cells, bootstrap.n_replicates)
        self.scores = np.full(shape, np.nan, dtype=np.float64)
        self.sizes = np.full(shape, np.nan, dtype=np.float64)

    def add(self, group: CellGroup, distances: Tensor, *, is_symmetric: bool) -> None:
        """Register a group: contract its distance matrix against every replicate's weights.

        :param group: The group's gathered data and metadata.
        :param distances: The group's ``(nx, na + sum(b_rows))`` distance matrix from
            :py:func:`fastabx.group.group_distance_matrix`.
        :param is_symmetric: Whether the group is symmetric (X == A) or not.
        """
        na = group.rows[0]
        dxa = distances[:, :na]
        if is_symmetric:
            # Idempotent: GroupReducer.add may have done it already on the same tensor.
            dxa.fill_diagonal_(float("inf"))
        index_x, index_a, index_b_blocks = group_pools(group, is_symmetric=is_symmetric)

        device, dtype = distances.device, distances.dtype
        wa = torch.from_numpy(self.bootstrap.weights(index_a)).to(device=device, dtype=dtype)
        wx = wa if is_symmetric else torch.from_numpy(self.bootstrap.weights(index_x)).to(device=device, dtype=dtype)
        wb_all = torch.from_numpy(
            np.concatenate([self.bootstrap.weights(index_b) for index_b in index_b_blocks], axis=1)
        ).to(device=device, dtype=dtype)

        counts = weighted_counts(dxa, distances[:, na:], wx, wa)
        # Number of (x, a) pairs per replicate, excluding x is a when X and A are the same pool.
        pairs = wx.sum(1).square() - wx.square().sum(1) if is_symmetric else wx.sum(1) * wa.sum(1)

        n_cells = len(group.positions)
        cell_ids = torch.from_numpy(np.repeat(np.arange(n_cells), group.rows[1:])).to(device)
        numerator = counts.new_zeros((counts.size(0), n_cells)).index_add_(1, cell_ids, counts * wb_all)
        b_drawn = wb_all.new_zeros((wb_all.size(0), n_cells)).index_add_(1, cell_ids, wb_all)

        denominator = pairs.unsqueeze(1) * b_drawn
        valid = denominator > 0
        scores = torch.where(valid, 1 - numerator / denominator, torch.nan)
        sizes = torch.where(valid, denominator, torch.nan)
        positions = list(group.positions)
        self.scores[positions] = scores.T.double().cpu().numpy()
        self.sizes[positions] = sizes.T.double().cpu().numpy()

    def finalize(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return the ``(num_cells, n_replicates)`` scores and sizes, with ``nan`` for degenerate replicates."""
        return self.scores, self.sizes
