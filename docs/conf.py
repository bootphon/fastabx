"""Sphinx configuration."""  # noqa: INP001

from datetime import UTC, datetime
from importlib.metadata import metadata

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_design",
    "nbsphinx",
]

project = "fastabx"
author = metadata(project)["Author"]
copyright = f"{datetime.now(tz=UTC).year}, {author}"  # noqa: A001
version = metadata(project)["Version"]
release = version

exclude_patterns = ["build"]
html_theme = "furo"
html_static_path = ["_static"]
