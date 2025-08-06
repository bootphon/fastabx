build:
    uv build --no-build-isolation --wheel
    uv pip install ./dist/*.whl
    python stable.py
