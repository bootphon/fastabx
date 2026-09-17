"""Regression tests for the comparison command's validation and process exit status."""

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import compare_versions


@pytest.mark.parametrize(
    "arguments",
    [
        ["--runs", "0"],
        ["--runs", "-1"],
        ["--tolerance", "-1"],
        ["--tolerance", "nan"],
        ["--tolerance", "inf"],
        ["--max-size-group", "1"],
        ["--max-x-across", "0"],
        ["--frequency", "0"],
        ["--frequency", "-50"],
        ["--frequency", "NaN"],
        ["--frequency", "Infinity"],
        ["--frequency", "invalid"],
        ["--seed", "-1"],
        ["--speaker", "across"],
    ],
)
def test_invalid_arguments_fail_before_subprocess(monkeypatch: pytest.MonkeyPatch, arguments: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["compare_versions", "item", "features", "--max-size-group", "10", *arguments])

    def unexpected_run(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Invalid arguments must fail before running a subprocess")

    monkeypatch.setattr(subprocess, "run", unexpected_run)
    with pytest.raises(SystemExit) as exc:
        compare_versions.main()
    assert exc.value.code == 2


@pytest.mark.parametrize(
    ("current_score", "reference_failure", "expected_exit"),
    [(0.25, False, 0), (0.75, False, 1), (0.25 + 1e-10, False, 1), (0.25, True, 1)],
)
def test_command_exit_status_and_forwarded_arguments(
    monkeypatch: pytest.MonkeyPatch, current_score: float, *, reference_failure: bool, expected_exit: int
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_versions",
            "item",
            "features",
            "--max-size-group",
            "-1",
            "--max-x-across",
            "-1",
            "--speaker",
            "across",
            "--frequency",
            "49.95",
            "--ref-version",
            "0.9.0",
            "--tolerance",
            "0",
        ],
    )
    runners: list[Path] = []

    def fake_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[-1] == "--version":
            return subprocess.CompletedProcess(cmd, 0, stdout="fastabx 0.9.1\n")
        runners.append(Path(cmd[-2]))
        params = json.loads(cmd[-1])
        assert params["frequency"] == "49.95"
        assert params["max_size_group"] is None
        assert params["max_x_across"] is None
        reference = "--no-project" in cmd
        if reference and reference_failure:
            raise subprocess.CalledProcessError(1, cmd, stderr="reference failed")
        return subprocess.CompletedProcess(
            cmd, 0, stdout=json.dumps({"score": 0.25 if reference else current_score, "elapsed": 1.0})
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(Path(compare_versions.__file__)), run_name="__main__")
    assert exc.value.code == expected_exit
    assert runners
    assert all(not path.exists() for path in runners)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_runner_score_fails(monkeypatch: pytest.MonkeyPatch, score: float) -> None:
    def fake_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"score": score, "elapsed": 1.0}))

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="finite score"):
        compare_versions.run(["python", "runner.py"], {}, 1)
