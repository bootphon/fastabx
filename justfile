build:
    uv build --no-build-isolation --wheel
    uv pip install ./dist/fastabx-0.4.1-cp312-abi3-macosx_15_0_arm64.whl
    python stable.py
