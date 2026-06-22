.PHONY: docs

docs:
	rm -rf docs/build docs/examples/gallery
	uv run --group doc sphinx-build -b html docs docs/build
