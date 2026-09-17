# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and fastabx adheres to
[semantic versioning](https://semver.org/spec/v2.0.0.html).

Releases up to and including 0.8.0 predate this file and are documented at
<https://github.com/bootphon/fastabx/releases>.

## Unreleased

### Added

- CI and releases smoke-test clean installations of both the wheel and source distribution outside the checkout.
- Distribution builds use an explicit source manifest so local corpora and unrelated untracked files cannot leak
  into release artifacts.
- Documentation covers dtype and count precision, timestamp contracts, normalization mutation and operation order,
  multi-ACROSS subsampling, bounded scoring intermediates, and hierarchical versus triplet-weighted averaging.
- All `Dataset.from_*` constructors accept `dtype=None` to preserve input precision or an explicit torch dtype.
- Dataset/accessor validation reports inconsistent row counts, invalid feature shapes and slice boundaries.
- Timestamp loading validates shape, finiteness and frame count; item intervals and frequencies are validated.

### Changed

- Grouped scoring trims excess chunk padding before computing each group's frame distances.
- The CLI accepts exact fractional feature frequencies and represents them as decimal strings in JSON output.
- Cell sizes use Int64 so large triplet totals remain valid when collapsing or exporting scores.
- Count reductions preserve half-integer contributions beyond float32 precision; final score storage stays float32.
  Small-group contributions stay float32 until a single promotion per flush.

- Tabular constructors preserve floating input precision by default. Pass `dtype=torch.float32` for the previous
  conversion behavior. Normalization and pooling require floating-point features; distance kernels defer dtype compatibility to PyTorch.
- Precomputed index lists reject nulls; duplicate and overlapping indices retain positional counting semantics.


### Fixed

- Timestamp loading honors custom column names and both boundary precisions.
- Callable distance objects need not be hashable; invalid distance/alignment arguments have actionable errors.
- Hamming pooling supports float64, and the score reducer handles float64 distance calculations.
- Duplicate normalized audio identifiers in units files are rejected instead of silently overwritten.

- Pooling preserves feature-to-label alignment after timestamp-based loading, including unsorted item files.
- Per-cell subsampling keeps distinct cells separate even when their labels contain hyphens or are identical.
- ACROSS subsampling uses collision-free grouping keys and samples complete observed X condition combinations.
- NumPy and dataframe constructors reject non-finite features, including overflow during float32 conversion.

## 0.9.0

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
