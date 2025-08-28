build:
    uv build --no-build-isolation --wheel --verbose
    uv pip install ./dist/*.whl --no-deps
    python stable.py
