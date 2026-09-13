# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and fastabx adheres to
[semantic versioning](https://semver.org/spec/v2.0.0.html).

Releases up to and including 0.8.0 predate this file and are documented at
<https://github.com/bootphon/fastabx/releases>.

## Unreleased

### Added

- `Accessor` is now a public protocol, and `Dataset.accessor` accepts any implementation of it.
- `Alignment` is a protocol too, and `Score` takes an `alignment` argument: either the name of a built-in
  one (only `"dtw"` for now) or a custom callable.
- `Score` and `abx_on_cell` accept a custom `Distance` callable.
- Every exception the public API raises is importable from the top-level `fastabx` namespace.
- `device` and `progress` arguments on the `Dataset` constructors, and `progress` on `Score` and
  `zerospeech_abx`.
- CLI: `--device`, `--output {text,json}`, `--write-csv PATH` and `--quiet`/`-q`.
- `Task` supports negative indices, so `task[-1]` is the last cell rather than an `IndexError`.

### Changed

- `Score.collapse` raises the new `EmptyScoreError` when every cell has a null score.
- `Cell.use_dtw` is now `Cell.needs_alignment`.
- `CollapseError` now lists the condition columns that are still present when `collapse` cannot pick an
  order on its own.
- A `Task` for which no cell can be built raises `EmptyTaskError`.
- The CLI rejects a `--max-size-group` or `--max-x-across` below 2, and a non-positive `--frequency`.

### Removed

- `FASTABX_OUTPUT` is gone; the CLI output format is now chosen with `--output {text,json}`.

### Fixed

- `pool_dataset` refuses a `Dataset` that a previous `"angular"` or `"cosine"` `Score` had L2-normalized in
  place.
- A condition column whose name starts with `index` (`indexer`, `index_of_speaker`, ...) is no
  longer mistaken for one of the internal `index_a` / `index_b` / `index_x` columns.
- `Task.from_cells` rejects a symmetric cell with fewer than two rows in `index_a`.
- An empty dataset raises `EmptyDatasetError`.
- `hamming_pooling` uses the symmetric window (`periodic=False`) instead of the periodic one.
- `apply_constraints` pins the order of each cell's `is_valid` mask.
