"""Direct tests for ``fastabx.verify``, the CLI (``__main__``) and ``fastabx.utils``."""

import json
import subprocess
import sys
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest
import torch

from fastabx.__main__ import build_parser
from fastabx.accessor import InMemoryAccessor
from fastabx.utils import (
    InvalidEnvironmentVariableError,
    display_name,
    gather_chunk_rows,
    max_score_chunk_rows,
    print_fastabx_output,
    reduction_flush_cols,
    with_librilight_bug,
)
from fastabx.verify import (
    NDIM,
    CellErrorType,
    DuplicateConditionsError,
    EmptyDataPointsError,
    EmptyDatasetError,
    InputTypeError,
    InvalidCellError,
    InvalidLevelsError,
    LabelReservedNameError,
    LabelSuffixError,
    LevelsErrorType,
    NonContiguousIndicesError,
    PrecomputedCellsError,
    UnknownConditionError,
    format_score_levels,
    verify_cell,
    verify_conditions_exist,
    verify_dataset_labels,
    verify_empty_datapoints,
    verify_precomputed_cells,
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


def test_verify_empty_datapoints_rejects_empty_mapping() -> None:
    """An accessor with no datapoint at all is rejected here, rather than crashing on ``max(())`` below."""
    with pytest.raises(EmptyDatasetError, match="empty"):
        verify_empty_datapoints({})


@pytest.mark.parametrize(
    "indices",
    [
        {0: (0, 1), 2: (1, 2)},  # gap at 1
        {1: (0, 1), 2: (1, 2)},  # does not start at 0
        {0: (0, 1), 5: (1, 2)},  # highest beyond len - 1
    ],
)
def test_verify_empty_datapoints_rejects_non_contiguous(indices: dict[int, tuple[int, int]]) -> None:
    with pytest.raises(NonContiguousIndicesError):
        verify_empty_datapoints(indices)


def test_non_contiguous_indices_rejected_by_accessor() -> None:
    """A gap must raise rather than silently make the missing rows read as empty."""
    with pytest.raises(NonContiguousIndicesError):
        InMemoryAccessor({0: (0, 1), 2: (1, 2)}, torch.zeros(2, 3), torch.device("cpu"))


def test_format_score_levels_normalises_strings() -> None:
    assert format_score_levels(["a", ("b", "c")]) == [("a",), ("b", "c")]


def test_format_score_levels_invalid_format_raises() -> None:
    with pytest.raises(InvalidLevelsError, match="list"):
        format_score_levels([123])  # ty: ignore[invalid-argument-type]


def test_verify_score_levels_duplicates() -> None:
    with pytest.raises(InvalidLevelsError, match="duplicates"):
        verify_score_levels(["x", "y"], [("x",), ("x",)])


def _good_cells() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "header": ["h"],
            "description": ["d"],
            "index_a": [[0, 1]],
            "index_b": [[2]],
            "index_x": [[0, 1]],
        }
    )


def test_verify_precomputed_cells_accepts_well_formed() -> None:
    verify_precomputed_cells(_good_cells(), num_items=3, is_symmetric=True)


def test_verify_precomputed_cells_missing_column() -> None:
    bad = _good_cells().drop("description")
    with pytest.raises(PrecomputedCellsError, match="description"):
        verify_precomputed_cells(bad, num_items=3, is_symmetric=True)


def test_verify_precomputed_cells_wrong_index_dtype() -> None:
    bad = _good_cells().with_columns(pl.col("index_a").cast(pl.List(pl.Float64)))
    with pytest.raises(PrecomputedCellsError, match="list of integers"):
        verify_precomputed_cells(bad, num_items=3, is_symmetric=True)


def test_verify_precomputed_cells_negative_index() -> None:
    bad = pl.DataFrame(
        {
            "header": ["h"],
            "description": ["d"],
            "index_a": [[-1, 0]],
            "index_b": [[1]],
            "index_x": [[0]],
        }
    )
    with pytest.raises(PrecomputedCellsError, match="negative"):
        verify_precomputed_cells(bad, num_items=3, is_symmetric=False)


def test_verify_precomputed_cells_out_of_range() -> None:
    with pytest.raises(PrecomputedCellsError, match="only has 2"):
        verify_precomputed_cells(_good_cells(), num_items=2, is_symmetric=True)


def test_verify_precomputed_cells_empty_index_list() -> None:
    bad = pl.DataFrame(
        {
            "header": ["h"],
            "description": ["d"],
            "index_a": pl.Series([[]], dtype=pl.List(pl.Int64)),
            "index_b": [[1]],
            "index_x": pl.Series([[]], dtype=pl.List(pl.Int64)),
        }
    )
    with pytest.raises(PrecomputedCellsError, match="empty index lists"):
        verify_precomputed_cells(bad, num_items=3, is_symmetric=False)


def test_verify_precomputed_cells_symmetric_requires_a_equals_x() -> None:
    bad = _good_cells().with_columns(pl.col("index_a").list.reverse().alias("index_x"))  # [1, 0] != [0, 1]
    with pytest.raises(PrecomputedCellsError, match="index_a == index_x"):
        verify_precomputed_cells(bad, num_items=3, is_symmetric=True)
    # The very same cells are fine for an asymmetric task.
    verify_precomputed_cells(bad, num_items=3, is_symmetric=False)


def test_verify_precomputed_cells_symmetric_requires_two_a() -> None:
    """A symmetric cell with a single A has no triplet left once the diagonal is dropped."""
    bad = pl.DataFrame(
        {
            "header": ["h"],
            "description": ["d"],
            "index_a": [[0]],
            "index_b": [[1, 2]],
            "index_x": [[0]],
        }
    )
    with pytest.raises(PrecomputedCellsError, match="at least 2 rows in 'index_a'"):
        verify_precomputed_cells(bad, num_items=3, is_symmetric=True)
    # A single A is perfectly fine when A and X are different sets.
    verify_precomputed_cells(bad.with_columns(index_x=pl.lit([1], dtype=pl.List(pl.Int64))), 3, is_symmetric=False)


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


@pytest.mark.parametrize(
    ("reader", "name", "default"),
    [
        (max_score_chunk_rows, "FASTABX_MAX_SCORE_CHUNK_ROWS", 8192),
        (gather_chunk_rows, "FASTABX_GATHER_CHUNK_ROWS", 8192),
        (reduction_flush_cols, "FASTABX_REDUCTION_FLUSH_COLS", 262144),
    ],
)
def test_chunk_env_vars_are_read_lazily(
    monkeypatch: pytest.MonkeyPatch, reader: Callable[[], int], name: str, default: int
) -> None:
    """Read on every use, not once at import, so setting one after ``import fastabx`` still takes effect."""
    monkeypatch.delenv(name, raising=False)
    assert reader() == default
    monkeypatch.setenv(name, "7")
    assert reader() == 7


@pytest.mark.parametrize("value", ["", "eight", "3.5", "0", "-1"])
def test_chunk_env_vars_reject_bad_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """A bad value fails where it is used, naming the variable, rather than making ``import fastabx`` raise."""
    monkeypatch.setenv("FASTABX_GATHER_CHUNK_ROWS", value)
    with pytest.raises(InvalidEnvironmentVariableError, match="FASTABX_GATHER_CHUNK_ROWS"):
        gather_chunk_rows()


def test_with_librilight_bug_default_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FASTABX_WITH_LIBRILIGHT_BUG", raising=False)
    assert with_librilight_bug() is False
    monkeypatch.setenv("FASTABX_WITH_LIBRILIGHT_BUG", "1")
    assert with_librilight_bug() is True
    monkeypatch.setenv("FASTABX_WITH_LIBRILIGHT_BUG", "0")
    assert with_librilight_bug() is False


def test_hide_progress_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    """TQDM_DISABLE wins over the progress argument, in both directions."""
    from fastabx.utils import hide_progress

    monkeypatch.delenv("TQDM_DISABLE", raising=False)
    assert hide_progress(progress=True) is False
    assert hide_progress(progress=False) is True
    monkeypatch.setenv("TQDM_DISABLE", "1")
    assert hide_progress(progress=True) is True
    assert hide_progress(progress=False) is True


def test_print_fastabx_output_default_format(capsys: pytest.CaptureFixture[str]) -> None:
    print_fastabx_output(0.1234, item="foo")
    out = capsys.readouterr().out
    assert "ABX error rate" in out
    assert "12.340%" in out


def test_print_fastabx_output_json(capsys: pytest.CaptureFixture[str]) -> None:
    print_fastabx_output(0.5, "json", item="x", count=3)
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
    import importlib.metadata

    result = subprocess.run(
        [sys.executable, "-m", "fastabx", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    # The contract of ``--version`` is that the package version appears in stdout. The leading
    # program name depends on ``sys.argv[0]`` (and on argparse's interpretation of it across
    # Python versions / environments), so don't assert on it.
    assert importlib.metadata.version("fastabx") in result.stdout


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


def test_cli_parser_defaults_and_quiet() -> None:
    parser = build_parser()
    args = parser.parse_args(["a.item", "feats", "--max-size-group", "10"])
    assert args.frequency == 50
    assert args.quiet is False
    quiet = parser.parse_args(["a.item", "feats", "--max-size-group", "10", "--quiet"])
    assert quiet.quiet is True


@pytest.mark.parametrize("flag", ["--max-size-group", "--max-x-across"])
@pytest.mark.parametrize("value", ["0", "1"])
def test_cli_rejects_too_small_subsample_size(flag: str, value: str) -> None:
    """A size the ``Subsampler`` would reject is caught by the parser, with the usage rather than a traceback."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["a.item", "feats", "--max-size-group", "10", flag, value])


@pytest.mark.parametrize(
    ("flag", "value"),
    [("--max-size-group", "many"), ("--frequency", "0"), ("--frequency", "fast")],
)
def test_cli_rejects_malformed_numbers(flag: str, value: str) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["a.item", "feats", "--max-size-group", "10", flag, value])


def test_cli_accepts_negative_subsample_size_to_disable() -> None:
    """A negative size disables the subsampling, and must stay accepted."""
    parser = build_parser()
    args = parser.parse_args(["a.item", "feats", "--max-size-group", "-1", "--max-x-across", "-1"])
    assert args.max_size_group == -1
    assert args.max_x_across == -1


def test_cli_help_has_no_default_on_required_argument() -> None:
    """``--max-size-group`` is required, so a "(default: None)" next to it would be a lie."""
    help_text = build_parser().format_help()
    assert "--max-size-group" in help_text
    assert "(default: 50)" in help_text  # --frequency still shows its default
    assert "(default: None)" not in help_text.split("--max-x-across")[0]


def test_cli_quiet_hides_progress_bars(tmp_path: Path) -> None:
    """``--quiet`` must silence both bars; the score still goes to stdout."""
    item = tmp_path / "data.item"
    # Phone and speaker must not be correlated, or every cell is empty and there is nothing to score.
    rows = [
        "#file onset offset #phone speaker prev-phone next-phone",
        *(f"f{i} 0.00 0.10 {'ab'[i % 2]} s{(i // 2) % 2} p1 n1" for i in range(8)),
    ]
    item.write_text("\n".join(rows) + "\n")
    for i in range(8):
        torch.save(torch.randn(6, 4), tmp_path / f"f{i}.pt")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fastabx",
            str(item),
            str(tmp_path),
            "--max-size-group",
            "2",
            "--context",
            "any",
            "--quiet",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "ABX error rate" in result.stdout
    assert "Building dataset" not in result.stderr
    assert "Scoring each cell" not in result.stderr


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


def test_main_output_json_device_and_write_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--output json, --device and --write-csv all reach their destination."""
    import json as json_module

    from fastabx.__main__ import main

    item, feats = _build_cli_dataset(tmp_path)
    csv = tmp_path / "cells.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fastabx",
            str(item),
            str(feats),
            "--max-size-group",
            "-1",
            "--distance",
            "euclidean",
            "--context",
            "any",
            "--device",
            "cpu",
            "--output",
            "json",
            "--write-csv",
            str(csv),
        ],
    )
    monkeypatch.setenv("TQDM_DISABLE", "1")
    main()
    payload = json_module.loads(capsys.readouterr().out)
    assert 0.0 <= payload["score"] <= 1.0
    assert payload["device"] == "cpu"
    assert payload["write_csv"] == str(csv)
    assert "output" not in payload  # consumed by the formatter itself
    assert csv.exists()
    assert csv.read_text().splitlines()[0].endswith("score,size")


def test_main_defaults_to_text_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without --output, the CLI prints the human-readable line."""
    from fastabx.__main__ import main

    item, feats = _build_cli_dataset(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["fastabx", str(item), str(feats), "--max-size-group", "-1", "--context", "any"],
    )
    monkeypatch.setenv("TQDM_DISABLE", "1")
    main()
    assert "ABX error rate" in capsys.readouterr().out


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


# ``NoAcrossError`` guards ``cells_on_by_across`` against an empty ``across``, which ``Task`` never does:
# it is unreachable through the public API, so it stays out of the top-level namespace.
INTERNAL_EXCEPTIONS = {"NoAcrossError"}


def test_every_exception_is_exported() -> None:
    """Adding a user-facing exception without exporting it must fail here."""
    import importlib
    import inspect
    import pkgutil

    import fastabx

    defined = set()
    for module_info in pkgutil.iter_modules(fastabx.__path__):
        module = importlib.import_module(f"fastabx.{module_info.name}")
        for name, obj in vars(module).items():
            if inspect.isclass(obj) and issubclass(obj, Exception) and obj.__module__.startswith("fastabx"):
                defined.add(name)
    assert defined, "no exception class found in fastabx"
    missing = sorted(defined - set(fastabx.__all__) - INTERNAL_EXCEPTIONS)
    assert not missing, f"exceptions raised by the public API but not exported: {missing}"


def test_exported_exceptions_keep_standard_bases() -> None:
    """The exported exceptions are catchable as plain ValueError / TypeError, with no fastabx base class."""
    import fastabx

    exported = [getattr(fastabx, name) for name in fastabx.__all__ if name.endswith("Error")]
    assert len(exported) == 32
    assert all(issubclass(exc, Exception) for exc in exported)
    assert issubclass(fastabx.FeaturesSizeError, ValueError)
    assert issubclass(fastabx.FrequencyTypeError, TypeError)


def test_verify_task_conditions_checks_types_before_duplicates() -> None:
    """An unhashable condition raises InputTypeError, not the TypeError of building the set."""
    with pytest.raises(InputTypeError):
        verify_task_conditions([["a"], "b"])  # ty: ignore[invalid-argument-type]


def test_verify_conditions_exist_accepts_known_columns() -> None:
    verify_conditions_exist(["phone", "speaker"], ["phone", "speaker"])


def test_verify_conditions_exist_unknown_raises() -> None:
    # 'phoneme' is spelled correctly and is simply the wrong name: the column is 'phone'.
    with pytest.raises(UnknownConditionError, match="phoneme"):
        verify_conditions_exist(["phone", "speaker"], ["phoneme"])


def test_unknown_condition_error_lists_available_columns() -> None:
    with pytest.raises(UnknownConditionError, match="speaker"):
        verify_conditions_exist(["phone", "speaker"], ["phone", "spk"])


def test_display_name_of_str_and_callables() -> None:
    def custom(_a: object, _b: object) -> None: ...

    class Custom:
        def __call__(self, _a: object, _b: object) -> None: ...

    assert display_name("angular") == "angular"
    assert display_name(custom) == "custom"
    assert display_name(Custom()) == "Custom"
