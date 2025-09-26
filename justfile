build:
    uv build --no-build-isolation --wheel --verbose
    uv pip install ./dist/*.whl --no-deps
    python stable.py

sync:
    uv sync --index https://download.pytorch.org/whl/nightly/cu128 --index-strategy unsafe-best-match --prerelease=allow
