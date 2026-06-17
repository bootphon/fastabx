"""Direct tests for ``fastabx.verify``, the CLI (``__main__``) and ``fastabx.utils``."""

import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest
import torch

from fastabx.utils import print_fastabx_output, with_librilight_bug
from fastabx.verify import (
    NDIM,
    CellErrorType,
    DuplicateConditionsError,
    EmptyDataPointsError,
    InputTypeError,
    InvalidCellError,
    InvalidLevelsError,
    LabelReservedNameError,
    LabelSuffixError,
    LevelsErrorType,
    format_score_levels,
    verify_cell,
    verify_dataset_labels,
    verify_empty_datapoints,
    verify_score_levels,
    verify_task_conditions,
)


def test_verify_task_conditions_duplicate_raises() -> None:
    with pytest.raises(DuplicateConditionsError):
        verify_task_conditions(["a", "a", "b"])


def test_verify_task_conditions_non_string_raises() -> None:
    with pytest.raises(InputTypeError):
        verify_task_conditions([1, "a"])  # ty: ignore[invalid-argument-type]


def test_verify_dataset_labels_reserved_name() -> None:
    df = pl.DataFrame({"index": [0]})
    with pytest.raises(LabelReservedNameError):
        verify_dataset_labels(df)


def test_verify_dataset_labels_invalid_suffix() -> None:
    df = pl.DataFrame({"phone_a": ["x"]})
    with pytest.raises(LabelSuffixError):
        verify_dataset_labels(df)


def test_verify_empty_datapoints() -> None:
    indices = {0: (0, 5), 1: (5, 5), 2: (5, 10), 3: (10, 9)}
    with pytest.raises(EmptyDataPointsError) as exc:
        verify_empty_datapoints(indices)
    msg = str(exc.value)
    assert "1" in msg
    assert "3" in msg


def test_verify_empty_datapoints_truncates_long_list() -> None:
    indices = dict.fromkeys(range(20), (0, 0))
    with pytest.raises(EmptyDataPointsError, match=r"\.\.\."):
        verify_empty_datapoints(indices)


def test_format_score_levels_normalises_strings() -> None:
    assert format_score_levels(["a", ("b", "c")]) == [("a",), ("b", "c")]


def test_format_score_levels_invalid_format_raises() -> None:
    with pytest.raises(InvalidLevelsError, match="list"):
        format_score_levels([123])  # ty: ignore[invalid-argument-type]


def test_verify_score_levels_duplicates() -> None:
    with pytest.raises(InvalidLevelsError, match="duplicates"):
        verify_score_levels(["x", "y"], [("x",), ("x",)])


def test_verify_score_levels_columns_missing() -> None:
    with pytest.raises(InvalidLevelsError, match="columns"):
        verify_score_levels(["x"], [("z",)])


def test_verify_cell_ndim() -> None:
    bad = torch.zeros(2, 3)  # 2D
    sa = torch.tensor([1, 1])
    with pytest.raises(InvalidCellError, match="3 dimensions"):
        verify_cell((bad, sa), (bad, sa), (bad, sa))


def test_verify_cell_feature_dim() -> None:
    a = torch.zeros(2, 1, 3)
    b = torch.zeros(2, 1, 4)
    s = torch.tensor([1, 1])
    with pytest.raises(InvalidCellError, match="feature dimension"):
        verify_cell((a, s), (b, s), (a, s))


def test_verify_cell_size() -> None:
    a = torch.zeros(2, 1, 3)
    bad_sizes = torch.tensor([1])
    s = torch.tensor([1, 1])
    with pytest.raises(InvalidCellError, match="size"):
        verify_cell((a, bad_sizes), (a, s), (a, s))


def test_invalid_cell_error_unknown_value_has_no_message() -> None:
    """Covers the no-case-matched exit of the ``match`` in ``InvalidCellError.__init__``.

    Not reachable via ``verify_cell`` (which only ever passes valid enum values), but the
    fall-through is in the source, so exercise it directly to keep branch coverage honest.
    """
    err = InvalidCellError(None)  # ty: ignore[invalid-argument-type]
    assert str(err) == "None"  # ValueError(None) stringifies as 'None'


def test_invalid_levels_error_unknown_value_has_no_message() -> None:
    """Same no-case-matched fall-through, for ``InvalidLevelsError``."""
    err = InvalidLevelsError(None)  # ty: ignore[invalid-argument-type]
    assert str(err) == "None"


def test_cell_error_type_enum_complete() -> None:
    assert {e.name for e in CellErrorType} == {"NDIM", "FEATURE_DIM", "SIZE"}


def test_levels_error_type_enum_complete() -> None:
    assert {e.name for e in LevelsErrorType} == {"FORMAT", "DUPLICATES", "COLUMNS"}


def test_ndim_constant() -> None:
    assert NDIM == 3


def test_with_librilight_bug_default_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FASTABX_WITH_LIBRILIGHT_BUG", raising=False)
    assert with_librilight_bug() is False
    monkeypatch.setenv("FASTABX_WITH_LIBRILIGHT_BUG", "1")
    assert with_librilight_bug() is True
    monkeypatch.setenv("FASTABX_WITH_LIBRILIGHT_BUG", "0")
    assert with_librilight_bug() is False


def test_print_fastabx_output_default_format(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FASTABX_OUTPUT", raising=False)
    print_fastabx_output(0.1234, item="foo")
    out = capsys.readouterr().out
    assert "ABX error rate" in out
    assert "12.340%" in out


def test_print_fastabx_output_json(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FASTABX_OUTPUT", "json")
    print_fastabx_output(0.5, item="x", count=3)
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"item": "x", "count": 3, "score": 0.5}


def test_librilight_bug_changes_frontiers(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastabx.dataset import item_frontiers

    df = pl.DataFrame(
        {
            "onset": [Decimal("0.1")],
            "offset": [Decimal("0.3")],
        }
    )

    monkeypatch.delenv("FASTABX_WITH_LIBRILIGHT_BUG", raising=False)
    _, end, *_ = item_frontiers(10, "onset", "offset")
    end_default = df.select(end)["end"][0]

    monkeypatch.setenv("FASTABX_WITH_LIBRILIGHT_BUG", "1")
    _, end, *_ = item_frontiers(10, "onset", "offset")
    end_buggy = df.select(end)["end"][0]
    assert end_default - end_buggy == 1


def test_cli_version() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "fastabx", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "fastabx" in result.stdout.lower()


def test_cli_requires_max_x_across_for_across_speaker(tmp_path: Path) -> None:
    item = tmp_path / "data.item"
    item.write_text("#file onset offset phone speaker prev-phone next-phone\nf1 0 0.1 a s1 # #\n")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fastabx",
            str(item),
            str(tmp_path),
            "--max-size-group",
            "5",
            "--speaker",
            "across",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "--max-x-across" in result.stderr


def _build_cli_dataset(tmp_path: Path) -> tuple[Path, Path]:
    """Build a tiny item file + matching features directory for CLI tests."""
    item = tmp_path / "data.item"
    item.write_text(
        "#file onset offset #phone prev-phone next-phone speaker\n"
        "f1 0.00 0.10 a # # s1\n"
        "f1 0.10 0.20 b # # s1\n"
        "f1 0.20 0.30 c # # s1\n"
        "f1 0.30 0.40 a # # s1\n"
        "f1 0.40 0.50 b # # s1\n"
        "f1 0.50 0.60 c # # s1\n"
        "f2 0.00 0.10 a # # s2\n"
        "f2 0.10 0.20 b # # s2\n"
        "f2 0.20 0.30 c # # s2\n"
        "f2 0.30 0.40 a # # s2\n"
        "f2 0.40 0.50 b # # s2\n"
        "f2 0.50 0.60 c # # s2\n"
    )
    feats_dir = tmp_path / "feats"
    feats_dir.mkdir()
    torch.manual_seed(0)
    torch.save(torch.randn(40, 4), feats_dir / "f1.pt")
    torch.save(torch.randn(40, 4), feats_dir / "f2.pt")
    return item, feats_dir


def test_main_missing_max_x_across_for_across_speaker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """In-process variant of the CLI error path, so it counts toward coverage."""
    from fastabx.__main__ import main

    item, feats = _build_cli_dataset(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fastabx",
            str(item),
            str(feats),
            "--max-size-group",
            "5",
            "--speaker",
            "across",
        ],
    )
    with pytest.raises(SystemExit):
        main()


def test_main_within_speaker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from fastabx.__main__ import main

    item, feats = _build_cli_dataset(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fastabx",
            str(item),
            str(feats),
            "--max-size-group",
            "-1",  # disabled
            "--frequency",
            "50",
            "--speaker",
            "within",
            "--context",
            "any",
            "--distance",
            "euclidean",
        ],
    )
    monkeypatch.setenv("TQDM_DISABLE", "1")
    main()
    out = capsys.readouterr().out
    assert "ABX error rate" in out


def test_main_across_speaker_with_disabled_x_across(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from fastabx.__main__ import main

    item, feats = _build_cli_dataset(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fastabx",
            str(item),
            str(feats),
            "--max-size-group",
            "-1",
            "--max-x-across",
            "-1",  # disabled
            "--frequency",
            "50",
            "--speaker",
            "across",
            "--context",
            "any",
            "--distance",
            "euclidean",
        ],
    )
    monkeypatch.setenv("TQDM_DISABLE", "1")
    main()
    out = capsys.readouterr().out
    assert "ABX error rate" in out
