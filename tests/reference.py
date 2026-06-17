"""Reference ABX oracle: slow, pure-loop, intentionally independent of the production code.

Used as the ground truth for cell-level scoring tests. Only supports pooled cells
(sequence length 1), which is enough to validate the production distance + scoring math.
DTW equivalence is covered by comparing the grouped engine against ``abx_on_cell``.
"""

import math
from collections.abc import Sequence

import numpy as np


def _normalize(features: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    out = features / np.where(norm == 0, 1.0, norm)
    out[norm.squeeze(1) == 0] = 1.0 / math.sqrt(features.shape[1])
    border = np.full((features.shape[0], 1), 1e-12)
    border[norm.squeeze(1) == 0] = -2e-12
    return np.concatenate([out, border], axis=1)


def reference_pointwise_distance(name: str, x: np.ndarray, y: np.ndarray) -> float:
    """Pointwise distance between two feature vectors ``x`` and ``y``.

    For ``cosine``/``angular`` the inputs are assumed to be already normalised.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if name == "euclidean":
        return float(np.sqrt(((x - y) ** 2).sum()))
    if name in {"cosine", "angular"}:
        dot = float(np.clip(np.dot(x, y), -1.0, 1.0))
        return math.acos(dot) / math.pi
    if name == "kl_symmetric":
        eps = 1e-6
        lp = np.log(x + eps)
        lq = np.log(y + eps)
        return float(0.5 * ((x * (lp - lq)).sum() + (y * (lq - lp)).sum()))
    if name == "identical":
        return 0.0 if x[0] == y[0] else 1.0
    msg = f"Unknown distance: {name}"
    raise ValueError(msg)


def reference_abx_pooled(
    a_features: Sequence[np.ndarray],
    b_features: Sequence[np.ndarray],
    x_features: Sequence[np.ndarray],
    distance: str,
    *,
    is_symmetric: bool,
) -> float:
    """Triplet-by-triplet ABX score for a cell with pooled features (one vector per item)."""
    a_arr = np.stack([np.asarray(v).reshape(-1) for v in a_features])
    b_arr = np.stack([np.asarray(v).reshape(-1) for v in b_features])
    x_arr = np.stack([np.asarray(v).reshape(-1) for v in x_features])
    if distance in {"cosine", "angular"}:
        a_arr = _normalize(a_arr)
        b_arr = _normalize(b_arr)
        x_arr = _normalize(x_arr)
    na, nb, nx = len(a_arr), len(b_arr), len(x_arr)
    total, count = 0.0, 0
    for ix in range(nx):
        for ia in range(na):
            if is_symmetric and ix == ia:
                continue
            dxa = reference_pointwise_distance(distance, x_arr[ix], a_arr[ia])
            for ib in range(nb):
                dxb = reference_pointwise_distance(distance, x_arr[ix], b_arr[ib])
                if dxa < dxb:
                    sc = 1.0
                elif dxa > dxb:
                    sc = 0.0
                else:
                    sc = 0.5
                total += sc
                count += 1
    return 1.0 - (total / count)
