"""Benchmark scoring and loading on synthetic data: wall time, peak memory, and the resulting score.

Each scenario runs in its own subprocess so that its peak resident memory is measured in isolation.
The collapsed score is reported too, to check that an optimisation leaves the results unchanged.

    uv run python scripts/benchmark.py                   # every scenario, as a table
    uv run python scripts/benchmark.py --json out.json   # also save the results
    uv run python scripts/benchmark.py --only pooled-full dtw-full
"""

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl
import torch

from fastabx import Dataset, InMemoryAccessor, Score, Subsampler, Task, constraints_all_different, pool_dataset

SEED = 0


def synthetic(
    *, speakers: int, phones: int, items_per_speaker: int, dim: int, max_len: int
) -> tuple[pl.DataFrame, list[np.ndarray]]:
    """Labels and variable-length features, where each phone has its own mean so the ABX is not at chance."""
    rng = np.random.default_rng(SEED)
    centers = rng.standard_normal((phones, dim)).astype(np.float32)
    rows, features = [], []
    for speaker in range(speakers):
        offset = 0.5 * rng.standard_normal(dim).astype(np.float32)
        for _ in range(items_per_speaker):
            phone = int(rng.integers(phones))
            length = int(rng.integers(max(2, max_len // 4), max_len + 1))
            noise = rng.standard_normal((length, dim)).astype(np.float32)
            features.append(centers[phone] + offset + 6.0 * noise)
            rows.append(
                {
                    "phone": f"p{phone}",
                    "prev": f"p{rng.integers(phones)}",
                    "next": f"p{rng.integers(phones)}",
                    "speaker": f"s{speaker}",
                    "rec": f"r{rng.integers(4)}",
                }
            )
    return pl.DataFrame(rows), features


def in_memory(labels: pl.DataFrame, features: list[np.ndarray]) -> Dataset:
    """Build a dataset directly from the per-item arrays."""
    lengths = np.array([len(f) for f in features])
    ends = np.cumsum(lengths)
    indices = {i: (int(end - length), int(end)) for i, (end, length) in enumerate(zip(ends, lengths, strict=True))}
    data = torch.from_numpy(np.concatenate(features))
    return Dataset(labels, InMemoryAccessor(indices, data, torch.device("cpu")))


def score(dataset: Dataset, **task_kwargs: object) -> Callable[[], float]:
    """Return a closure scoring a task on ``dataset`` with the angular distance."""
    subsampler = task_kwargs.pop("subsampler", None)
    constraints = task_kwargs.pop("constraints", None)
    levels = task_kwargs.pop("levels", None)

    def run() -> float:
        task = Task(dataset, subsampler=subsampler, **task_kwargs)  # ty: ignore[invalid-argument-type]
        result = Score(task, "angular", constraints=constraints, progress=False)  # ty: ignore[invalid-argument-type]
        return result.collapse(levels=levels) if levels else result.collapse(weighted=True)

    return run


def scenario_within_subsampled() -> Callable[[], float]:
    """ZeroSpeech-like within-speaker, within-context task, subsampled: many small groups with DTW."""
    labels, features = synthetic(speakers=8, phones=12, items_per_speaker=1500, dim=64, max_len=16)
    return score(
        in_memory(labels, features),
        on="phone",
        by=["prev", "next", "speaker"],
        subsampler=Subsampler(10, None),
        levels=[("next", "prev"), "speaker"],
    )


def scenario_across_subsampled() -> Callable[[], float]:
    """ZeroSpeech-like across-speaker, any-context task, subsampled: tiny groups with DTW."""
    labels, features = synthetic(speakers=6, phones=10, items_per_speaker=300, dim=64, max_len=16)
    return score(
        in_memory(labels, features),
        on="phone",
        across=["speaker"],
        subsampler=Subsampler(10, 5),
        levels=["speaker"],
    )


def scenario_dtw_full() -> Callable[[], float]:
    """No subsampling and long sequences: few, large groups whose frame lattice is large."""
    labels, features = synthetic(speakers=2, phones=4, items_per_speaker=1200, dim=32, max_len=30)
    return score(in_memory(labels, features), on="phone", by=["speaker"])


def scenario_pooled_full() -> Callable[[], float]:
    """Pooled features and no subsampling: large groups, dominated by the win/tie counting."""
    labels, features = synthetic(speakers=2, phones=4, items_per_speaker=2500, dim=32, max_len=8)
    pooled = pool_dataset(in_memory(labels, features), "mean")
    return score(pooled, on="phone", by=["speaker"])


def scenario_constrained() -> Callable[[], float]:
    """Pooled features, no subsampling, and a constraint on another label."""
    labels, features = synthetic(speakers=2, phones=4, items_per_speaker=1200, dim=32, max_len=8)
    pooled = pool_dataset(in_memory(labels, features), "mean")
    return score(pooled, on="phone", by=["speaker"], constraints=constraints_all_different("rec"))


def scenario_pool() -> Callable[[], float]:
    """Mean and hamming pooling of many short items."""
    labels, features = synthetic(speakers=8, phones=12, items_per_speaker=5000, dim=64, max_len=16)
    dataset = in_memory(labels, features)

    def run() -> float:
        mean, hamming = pool_dataset(dataset, "mean"), pool_dataset(dataset, "hamming")
        return float(sum(x.sum() for x in mean.accessor) + sum(x.sum() for x in hamming.accessor))

    return run


def scenario_from_numpy() -> Callable[[], float]:
    """Tabular construction from a large 2D array."""
    rng = np.random.default_rng(SEED)
    features = rng.standard_normal((200_000, 256)).astype(np.float32)
    labels = {"phone": [f"p{i % 10}" for i in range(len(features))]}

    def run() -> float:
        dataset = Dataset.from_numpy(features, labels, device="cpu")
        return float(dataset.accessor[len(features) - 1].sum())

    return run


def _write_item(root: Path, labels: pl.DataFrame, features: list[np.ndarray], *, frequency: int) -> Path:
    """Write the items into files of 40 tokens each, one after the other, plus an item file."""
    rows, per_file = [], 40
    for start in range(0, len(features), per_file):
        chunk = features[start : start + per_file]
        torch.save(torch.from_numpy(np.concatenate(chunk)), root / f"f{start}.pt")
        times = (torch.arange(sum(len(f) for f in chunk), dtype=torch.float64) + 0.5) / frequency
        torch.save(times, root / "times" / f"f{start}.pt")
        cursor = 0
        for i, feats in enumerate(chunk):
            onset, offset = cursor / frequency, (cursor + len(feats) - 1) / frequency
            rows.append(
                {"#file": f"f{start}", "onset": f"{onset:.4f}", "offset": f"{offset:.4f}"}
                | labels.row(start + i, named=True)
            )
            cursor += len(feats)
    item = root / "data.csv"
    pl.DataFrame(rows).write_csv(item)
    return item


def scenario_from_item() -> Callable[[], float]:
    """Load from feature files and an item file, with a frequency."""
    labels, features = synthetic(speakers=8, phones=12, items_per_speaker=4000, dim=128, max_len=16)
    root = Path(tempfile.mkdtemp())
    (root / "times").mkdir()
    item = _write_item(root, labels, features, frequency=100)

    def run() -> float:
        dataset = Dataset.from_item(item, root, 100, device="cpu", progress=False)
        return float(dataset.accessor[0].sum())

    return run


def scenario_from_item_with_times() -> Callable[[], float]:
    """Load from feature files, timestamp files and an item file."""
    labels, features = synthetic(speakers=8, phones=12, items_per_speaker=4000, dim=128, max_len=16)
    root = Path(tempfile.mkdtemp())
    (root / "times").mkdir()
    item = _write_item(root, labels, features, frequency=100)

    def run() -> float:
        dataset = Dataset.from_item_with_times(item, root, root / "times", device="cpu", progress=False)
        return float(dataset.accessor[0].sum())

    return run


SCENARIOS: dict[str, Callable[[], Callable[[], float]]] = {
    "within-subsampled": scenario_within_subsampled,
    "across-subsampled": scenario_across_subsampled,
    "dtw-full": scenario_dtw_full,
    "pooled-full": scenario_pooled_full,
    "constrained": scenario_constrained,
    "pool": scenario_pool,
    "from-numpy": scenario_from_numpy,
    "from-item": scenario_from_item,
    "from-item-with-times": scenario_from_item_with_times,
}


def peak_rss_mb() -> float:
    """Peak resident memory of this process, in MiB (``ru_maxrss`` is in bytes on macOS, KiB on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def run_one(name: str) -> dict[str, float | str]:
    """Prepare a scenario, then time it and measure the peak memory the run adds on top of the preparation."""
    torch.manual_seed(SEED)
    run = SCENARIOS[name]()
    before = peak_rss_mb()
    start = time.perf_counter()
    value = run()
    elapsed = time.perf_counter() - start
    return {
        "scenario": name,
        "seconds": elapsed,
        "peak_mb": peak_rss_mb(),
        "added_mb": peak_rss_mb() - before,
        "value": value,
    }


def main() -> None:
    """Run every scenario in a subprocess and print a table."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=list(SCENARIOS), help="Run only these scenarios")
    parser.add_argument("--json", type=Path, help="Save the results to this JSON file")
    parser.add_argument("--scenario", choices=list(SCENARIOS), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.scenario:
        print(json.dumps(run_one(args.scenario)))
        return

    results = []
    env = os.environ | {"TQDM_DISABLE": "1"}
    for name in args.only or SCENARIOS:
        out = subprocess.run(
            [sys.executable, __file__, "--scenario", name], capture_output=True, text=True, check=True, env=env
        )
        result = json.loads(out.stdout.strip().splitlines()[-1])
        results.append(result)
        print(
            f"{name:<22} {result['seconds']:8.2f} s  peak {result['peak_mb']:8.0f} MiB  "
            f"(+{result['added_mb']:6.0f} MiB)  value {result['value']:.9g}",
            flush=True,
        )
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
