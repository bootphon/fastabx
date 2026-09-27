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
- `FASTABX_MAX_LATTICE_ELEMENTS` bounds the frame-level distance lattice built at once, splitting both the X and the
  target side of a group.
- `scripts/benchmark.py` measures the runtime, peak memory and score of synthetic scoring and loading workloads.

### Changed

- Angular/cosine distance now explicitly treats two zero frames as identical (distance 0) and a zero/nonzero
  pair as maximally distant (distance 1). Zero frames remain zero after normalization; the extra column is
  retained for layout compatibility. Scores involving zero frames can change. Pin the exact previous version
  and dependencies to reproduce historical behavior; no legacy-normalization switch is provided.
- Grouped scoring trims excess chunk padding before computing each group's frame distances.
- The CLI accepts exact fractional feature frequencies and represents them as decimal strings in JSON output.
- Cell sizes use Int64 so large triplet totals remain valid when collapsing or exporting scores.
- Win/tie counts are exact Int64 integers (a win counts 2, a tie 1).
- `Score.cells` stores the cell scores as Float64 instead of Float32, so collapsed scores no longer carry float32
  rounding noise (for example `0.4913194444444444` instead of `0.4913194353381793`).
- `Dataset.from_dataframe` reads every label column of a CSV file as a string, like the item files, so distinct
  labels such as speakers `01` and `1` are no longer merged. It raises `TypeError` for an unsupported `source`.
- `Task` raises `InputTypeError` when `by` or `across` is a single string instead of a list.
- The default `feature_maker` and `time_maker` load each file with `torch.load(path, map_location="cpu")`, so
  features saved from a GPU load on a CPU-only machine. They are now `None` in the signatures.
- Win/tie counting sorts each row of X-to-A distances and locates every B by binary search, instead of building
  every triplet: memory is proportional to the number of pairs of a group, not of its triplets. On a large
  unsubsampled pooled task, peak memory drops from 17.7 GB to 0.4 GB and runtime by 20x. Scores are unchanged.
- Constraints are evaluated once per distinct combination of the labels they use, and counted in chunks of
  triplets, instead of materialising a mask of every triplet of the task: 16x faster and 20x less memory on a
  large constrained task. Constraints must be row-wise expressions; aggregations over triplets are not supported.
- A group larger than `FASTABX_GATHER_CHUNK_ROWS` gathers its targets chunk by chunk as they are compared.
- `Dataset.from_item` allocates the output once and copies each item into place, holding at most one feature
  file besides it. Feature files with different dtypes now raise `InvalidFeaturesError` unless `dtype=` is given.
- `Dataset.from_item_with_times` locates each item by binary search in the timestamps, which must now be sorted in
  non-decreasing order (`InvalidTimesError` otherwise).
- `Dataset.from_numpy` converts the array to a tensor directly instead of through a polars DataFrame, and so accepts
  label columns of any name. The features are stored row-major.
- `pool_dataset` pools the items by batches of equal length (hamming pooling may differ by a few ulps from pooling
  each item), and raises `InvalidFeatureDtypeError` on integer features instead of a torch `RuntimeError`.
- `Subsampler` raises `ValueError` for a size below 2 (previously `TypeError`), `InputTypeError` for a non-integer
  size, and accepts NumPy integers.
- `abx_on_cell` accepts the name of a built-in alignment (`alignment="dtw"`, the default), like `Score`.
- The `Dataset.from_*` constructors return an instance of the class they are called on. `PooledDataset` is only
  built by `pool_dataset`: its inherited constructors raise `TypeError` instead of returning a plain `Dataset`.
- Unknown distance and pooling names raise a `ValueError` that lists the valid names.

- Tabular constructors preserve floating input precision by default. Pass `dtype=torch.float32` for the previous
  conversion behavior. Normalization and pooling require floating-point features; distance kernels defer dtype compatibility to PyTorch.
- Precomputed index lists reject nulls; duplicate and overlapping indices retain positional counting semantics.
- Every label column of a `.item` or `.csv` item file is read as a string instead of having its type inferred.
  Labels that were inferred as numbers, such as numeric speaker IDs, now have the `String` dtype.


### Fixed

- An item file whose onsets (or offsets) are all zero no longer crashes while its times are parsed as decimals.
- Times written in scientific notation (`5e-05`) are rejected with an `InvalidItemFileError`: the decimal
  parser silently read them as 0.
- An exception raised inside a user's `feature_maker` or `time_maker`, including a `KeyError`, now propagates
  as is instead of being reported as missing feature files.
- Chunked scoring preserves custom distance/alignment output dtypes instead of casting to the feature dtype,
  preventing chunk-size-dependent ties and scores.
- L2 normalization scales finite nonzero frames before computing norms, avoiding norm overflow and underflow.
  Low-precision norm accumulation uses float32. Extreme-magnitude and near-tie scores may change.
- Timestamp loading honors custom column names and both boundary precisions.
- NaN distances raise `NaNDistanceError` instead of being silently counted as ties. For example, `kl_symmetric`
  on features with negative values previously returned a meaningless score.
- Null values in ON, BY or ACROSS conditions raise `MissingLabelError` instead of silently dropping those rows
  from every cell.
- Item-file labels are no longer merged when they look like the same number (speakers `01` and `1`), and a
  label that stops looking numeric after the first 100 rows no longer fails to parse.
- `extension` must contain a dot: `"pt"` used to also match files such as `script`. Directories whose name ends
  with the extension are no longer taken for feature files.
- `Dataset.from_item_and_units` also strips Windows-style directories from the audio paths, and its duplicate
  identifiers error lists the duplicates. Its matching on base names is documented.
- Constraints strip exactly one `_a`/`_b`/`_x` suffix, so labels ending like a suffix (`mic_b`) can be
  constrained. A constraint column without a suffix, or naming an unknown label, raises a `NoConstraintsError`
  that names it. Constraints evaluating to null mark the triplet as invalid, explicitly.
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
