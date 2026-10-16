"""Reference ABX oracle: slow, pure-loop, intentionally independent of the production code.

Used as the ground truth for cell-level scoring tests. Only supports pooled cells
(sequence length 1), which is enough to validate the production distance + scoring math.
DTW equivalence is covered by comparing the grouped engine against ``abx_on_cell``.

Also holds brute-force references for the statistics of ``fastabx.stats``.
"""

import itertools
import math
from collections.abc import Sequence

import numpy as np


def _normalize(features: np.ndarray) -> np.ndarray:
    # math.hypot provides an independent, scale-safe norm; no production border representation is needed.
    return np.stack([row / (math.hypot(*row) or 1.0) for row in features])


def reference_pointwise_distance(name: str, x: np.ndarray, y: np.ndarray) -> float:
    """Pointwise distance between two feature vectors ``x`` and ``y``.

    For ``cosine``/``angular`` the inputs are assumed to be already normalised.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if name == "euclidean":
        return float(np.sqrt(((x - y) ** 2).sum()))
    if name in {"cosine", "angular"}:
        zero_x, zero_y = not np.any(x), not np.any(y)
        if zero_x or zero_y:
            return 0.5 * (zero_x != zero_y)
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


def reference_crossed_variance(values: np.ndarray) -> float | None:
    """Unbiased variance of the mean of a fully crossed design, by brute-force enumeration.

    ``values`` has one axis per factor. Axes of size 1 are fixed factors and are dropped. The estimator is
    ``mean^2 - Q``, ``Q`` being the average product of two values that differ on every remaining axis.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values.reshape([n for n in values.shape if n > 1])
    if values.ndim == 0:
        return None
    indices = list(itertools.product(*map(range, values.shape)))
    total, count = 0.0, 0
    for i in indices:
        for j in indices:
            if all(a != b for a, b in zip(i, j, strict=True)):
                total += values[i] * values[j]
                count += 1
    mean = values.mean()
    return float(mean * mean - total / count)


def reference_schatz_variance(values: np.ndarray) -> float:
    """Variance estimator of Schatz et al. (PNAS, SI section 4), with its order 3, degree (2, 2, 2) kernel.

    A literal transcription of the symmetrized kernel ``psi`` averaged over all pairs of speakers, of models of the
    first language and of models of the second language. ``values[s, a, j]`` is the D-hat of the paper.
    """
    n1, n2, n3 = values.shape

    def psi(s: tuple[int, ...], a: tuple[int, ...], j: tuple[int, ...]) -> float:
        (s1, s2), (a1, a2), (j1, j2) = s, a, j
        d = values
        base = d[s1, a1, j1] * d[s2, a2, j2]
        out = (n1 - 1) * (n2 - 1) * (d[s1, a1, j1] * d[s2, a2, j1] - base)
        out += (n1 - 1) * (n3 - 1) * (d[s1, a1, j1] * d[s2, a1, j2] - base)
        out += (n2 - 1) * (n3 - 1) * (d[s1, a1, j1] * d[s1, a2, j2] - base)
        out += (n1 - 1) * (d[s1, a1, j1] * d[s2, a1, j1] - base)
        out += (n2 - 1) * (d[s1, a1, j1] * d[s1, a2, j1] - base)
        out += (n3 - 1) * (d[s1, a1, j1] * d[s1, a1, j2] - base)
        out += d[s1, a1, j1] ** 2 - base
        return out / (n1 * n2 * n3)

    total = 0.0
    for s in itertools.combinations(range(n1), 2):
        for a in itertools.combinations(range(n2), 2):
            for j in itertools.combinations(range(n3), 2):
                total += np.mean(
                    [psi(s[::fs], a[::fa], j[::fj]) for fs, fa, fj in itertools.product([1, -1], repeat=3)]
                )
    return float(total / (math.comb(n1, 2) * math.comb(n2, 2) * math.comb(n3, 2)))


def reference_unit_scores(
    cells: Sequence[dict], *, on: str, unit: str, context: str, symmetric: bool, average_contrasts: bool
) -> dict[tuple, float]:
    """Error rate of every (contrast, unit), averaged over the contexts available, from plain cell rows.

    ``cells`` are rows of ``Score.cells`` with the ON, unit and context labels and the score. With ``symmetric``, a
    contrast (a, b) is the mean of the (a, b) and (b, a) cells, and is skipped when one of them is missing or null.
    With ``average_contrasts``, the contrasts are then averaged within each unit, and the key is ``(unit,)``.
    """
    scores = {(row[on], row[f"{on}_b"], row[unit], row[context]): row["score"] for row in cells}
    per_unit: dict[tuple, list[float]] = {}
    for (a, b, u, c), value in scores.items():
        if value is None:
            continue
        if symmetric:
            other = scores.get((b, a, u, c))
            if a > b or other is None:
                continue
            value = (value + other) / 2  # ruff: ignore[redefined-loop-name]
        per_unit.setdefault((a, b, u), []).append(value)
    by_contrast = {key: float(np.mean(values)) for key, values in per_unit.items()}
    if not average_contrasts:
        return by_contrast
    per_speaker: dict[tuple, list[float]] = {}
    for (_, _, u), value in by_contrast.items():
        per_speaker.setdefault((u,), []).append(value)
    return {key: float(np.mean(values)) for key, values in per_speaker.items()}
