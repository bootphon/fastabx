.PHONY: docs

SPHINXOPTS ?=

docs:
	rm -rf docs/build docs/examples/gallery
	uv run --group doc sphinx-build -b html $(SPHINXOPTS) docs docs/build
