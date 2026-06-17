# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "rich>=15.0.0",
# ]
# ///

# ruff: noqa: S603
"""Compare the currently installed fastabx against a reference version from PyPI.

Calls :func:`fastabx.zerospeech.zerospeech_abx` with the exact same arguments and options against:
  - the version currently installed in this project (``uv run python ...``)
  - a reference version fetched from PyPI with uv (``uv run --no-project --with fastabx==X.Y.Z python ...``)

The ``zerospeech_abx`` function is used directly (rather than the ``fastabx`` CLI) because the
CLI JSON output only exists since 0.7.1, whereas the ``zerospeech_abx`` signature has been stable
for a while. The score returned by both versions should be identical, and the current version
should be at least as fast as the reference. The reference version reuses the same arguments and
options as the CLI entry point.

Everything is passed as normal options; the same arguments are forwarded to both versions. Example:

    uv run scripts/compare_versions.py path/to/file.item path/to/features \
        --max-size-group 10 --speaker within --runs 3
"""

import argparse
import json
import math
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

from rich.box import HORIZONTALS
from rich.console import Console
from rich.table import Table

console = Console()

RUNNER = """
import json, sys, time
from fastabx.zerospeech import zerospeech_abx

params = json.loads(sys.argv[1])
start = time.perf_counter()
score = zerospeech_abx(**params)
elapsed = time.perf_counter() - start
print(json.dumps({"score": score, "elapsed": elapsed}))
"""


def fmt_meanstd(times: list[float]) -> str:
    """Format a list of times as mean ± std, or just mean if only one value."""
    if not times:
        return "N/A"
    if len(times) > 1:
        return f"{statistics.mean(times):.3f} ± {statistics.stdev(times):.3f}"
    return f"{statistics.mean(times):.3f}"


def latest_version() -> str:
    """Query PyPI to get the latest fastabx version."""
    cmd = ["uv", "run", "--with", "pip", "pip", "index", "versions", "fastabx", "--json"]
    process = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(process.stdout)["latest"]


def installed_version() -> str:
    """Version of fastabx currently installed in the project."""
    cmd = ["uv", "run", "fastabx", "--version"]
    process = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return process.stdout.strip().removeprefix("fastabx ")


def run(cmd: list[str], params: dict, runs: int) -> tuple[float, list[float]]:
    """Run the command ``runs`` times, returning the score and the per-run elapsed times."""
    cmd = [*cmd, json.dumps(params)]
    score, times = float("inf"), []
    for _ in range(runs):
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        result = json.loads(process.stdout.strip().splitlines()[-1])
        score, elapsed = result["score"], result["elapsed"]
        times.append(elapsed)
    return score, times


def run_both_versions(
    console: Console,
    params: dict,
    ref_version: str,
    runs: int,
) -> tuple[float, list[float], float, list[float]]:
    """Run both the reference and current versions, returning their scores and elapsed times."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(RUNNER)
        runner = f.name
    try:
        current_cmd = ["uv", "run", "python", runner]
        ref_cmd = ["uv", "run", "--no-project", "--with", f"fastabx=={ref_version}", "python", runner]

        console.print("[dim]Running reference version...[/dim]")
        try:
            ref_score, ref_times = run(ref_cmd, params, runs)
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as error:
            console.print(f"[yellow]! reference version {ref_version} failed to run, skipping comparison.[/yellow]")
            if isinstance(error, subprocess.CalledProcessError) and error.stderr:
                for line in error.stderr.strip().splitlines()[-10:]:
                    console.print(f"  [dim]{line}[/dim]")
            sys.exit(0)

        console.print("[dim]Running current version...[/dim]")
        current_score, current_times = run(current_cmd, params, runs)
    finally:
        Path(runner).unlink(missing_ok=True)
    console.print()
    return ref_score, ref_times, current_score, current_times


def main() -> int:
    """Entry point to compare the installed fastabx against a reference version from PyPI."""
    parser = argparse.ArgumentParser(
        description="Compare installed fastabx against a reference version from PyPI.",
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ref-version", help="Reference fastabx version from PyPI. Defaults to the latest published.")
    parser.add_argument("--runs", type=int, default=1, help="Number of times to run each version.")
    parser.add_argument("--tolerance", type=float, default=1e-8, help="Absolute tolerance for score comparison.")

    parser.add_argument("item", help="Path to the item file")
    parser.add_argument("features", help="Path to the features directory")
    parser.add_argument("--max-size-group", type=int, required=True, help="Maximum number of A, B, or X in a cell.")
    parser.add_argument("--max-x-across", type=int, help="With 'across', maximum number of X given (A, B).")
    parser.add_argument("--frequency", type=int, default=50, help="Feature frequency (in Hz)")
    parser.add_argument("--speaker", choices=["within", "across"], default="within", help="Speaker mode")
    parser.add_argument("--context", choices=["within", "any"], default="within", help="Context mode")
    parser.add_argument(
        "--distance",
        choices=["angular", "euclidean", "kl_symmetric", "identical"],
        default="angular",
        help="Distance",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()
    if args.max_x_across is None and args.speaker == "across":
        parser.error("--max-x-across is required when using 'across' speaker mode")
    params = {
        "item": args.item,
        "root": args.features,
        "max_size_group": args.max_size_group if args.max_size_group >= 0 else None,
        "max_x_across": args.max_x_across if (args.max_x_across is not None and args.max_x_across >= 0) else None,
        "speaker": args.speaker,
        "context": args.context,
        "distance": args.distance,
        "frequency": args.frequency,
        "seed": args.seed,
    }
    ref_version = args.ref_version or latest_version()
    current_version = installed_version()

    info = Table.grid(padding=(0, 2))
    info.add_row("current (installed):", f"[cyan]{current_version}[/cyan]")
    info.add_row("reference (PyPI):", f"[cyan]{ref_version}[/cyan]")
    info.add_row("parameters:", f"[dim]{json.dumps(params)}[/dim]")
    console.print(info)
    console.print()

    ref_score, ref_times, curr_score, curr_times = run_both_versions(console, params, ref_version, args.runs)
    table = Table(box=HORIZONTALS)
    table.add_column("Version", no_wrap=True)
    table.add_column("Mean ± std (s)", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Runs", justify="right")
    table.add_row(f"reference {ref_version}", fmt_meanstd(ref_times), str(ref_score), str(len(ref_times)))
    table.add_row(f"current {current_version}", fmt_meanstd(curr_times), str(curr_score), str(len(curr_times)))
    console.print(table)
    console.print()

    scores_match = math.isclose(ref_score, curr_score, abs_tol=args.tolerance)
    diff = f"abs. difference: {abs(ref_score - curr_score):.2e}"
    ref_mean, curr_mean = statistics.mean(ref_times), statistics.mean(curr_times)
    speedup, delta = ref_mean / curr_mean if curr_mean else float("inf"), curr_mean - ref_mean
    msg = f"{speedup:.2f}x  ({delta:+.3f}s)"
    console.print(
        f"[bold green]Scores match ({diff}) ✓[/bold green]"
        if scores_match
        else f"[bold red]Scores do not match ({diff})✗[/bold red]"
    )
    console.print(
        f"[bold green]{msg}: faster ✓[/bold green]" if speedup >= 1.0 else f"[bold red]{msg}: slower ✗[/bold red]"
    )
    return int(scores_match)


if __name__ == "__main__":
    main()
