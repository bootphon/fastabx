# Contributing to fastabx

Contributions are very welcome! If you are unsure whether something belongs in fastabx, open an issue first:
the library aims to stay small and readable, so a new feature has to be worth the maintenance.

We are particularly interested in:

- **Other modalities**: Examples on non-speech dataset or APIs that make other modalities natural.
- **Documentation**: Any improvement is welcome. Places where a reader got lost are especially useful to hear about.
- **New alignments and accessors**: edit distance or another useful one, a memory-mapped and efficient accessor, etc.

## Development setup

Development is managed with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/bootphon/fastabx
cd fastabx
uv sync --group test          # Virtual environment with the test dependencies
uv run prek install           # Install the git hooks (lint, format, typos, lockfile)
```

## Checks

The following are what CI runs; they should all pass before a pull request is merged.

```bash
uv run pytest                 # Tests, with coverage (must stay above 99%)
uv run prek run --all-files   # ruff check, ruff format, ty, typos, tombi, zizmor, uv lock/audit
make docs                     # Build the documentation into docs/build
uv build                      # Build both release distributions
```

CI additionally installs the wheel and the source distribution into separate clean environments, then runs an
import, a minimal end-to-end evaluation and `fastabx --version` outside the checkout.

On a machine with a GPU, `pytest` runs every test twice: on the CPU, then on CUDA.

To compare scores and runtime against a published version, run:

```bash
uv run scripts/compare_versions.py path/to/file.item path/to/features --max-size-group 10 --runs 3
```

## Pull requests

Branch off `main`, keep the change focused, and make sure the checks above pass. The GitHub Actions CI runs
the linters and the test suite on every pull request.

## Changelog

User-visible changes go in `CHANGELOG.md`, under `## Unreleased`, in the pull request that makes them.

## Releases

Releases are automated: the version comes from the git tag (`hatch-vcs`), and pushing a tag triggers the
release workflow, which runs the checks, builds the wheel and the sdist, smoke-tests both in clean
environments outside the checkout, publishes to PyPI with trusted publishing, and creates the GitHub release.

The hosted CI is CPU-only. Run the test suite on a CUDA machine before releasing changes that affect
device-specific code.

Before tagging, rename the `## Unreleased` heading of `CHANGELOG.md` to the version being released and
commit it.
