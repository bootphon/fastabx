# ruff: noqa: INP001, A001
"""Sphinx configuration."""

import functools
import inspect
from datetime import UTC, datetime
from importlib.metadata import metadata
from pathlib import Path
from typing import TypeAliasType

from packaging.version import parse

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.linkcode",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_design",
    "nbsphinx",
]

project = "fastabx"
author = metadata(project)["Author"]
copyright = f"{datetime.now(tz=UTC).year}, {author}"
version = parse(metadata(project)["Version"]).base_version
release = version

autodoc_typehints = "description"
add_function_parentheses = False
exclude_patterns = ["build"]
html_theme = "furo"
mathjax3_config = {"tex": {"macros": {"onset": "t_\\text{on}", "offset": "t_\\text{off}"}}}
toc_object_entries_show_parents = "hide"


class SourceCodeError(ValueError):
    """Some part of the source code cannot be found."""


@functools.cache
def linkcode_package() -> Path:
    """Path to the source of the package."""
    pkg = inspect.getsourcefile(__import__(project))
    if pkg is None:
        raise SourceCodeError
    return Path(pkg).parent


def linkcode_resolve(domain: str, info: dict) -> str | None:
    """Return the URL to source code."""
    if domain != "py" or not info["module"]:
        return None
    pkg = linkcode_package()
    module = __import__(info["module"], fromlist=[""])
    obj = module
    for part in info["fullname"].split("."):
        obj = getattr(obj, part)
    obj = inspect.unwrap(obj)
    if isinstance(obj, TypeAliasType):
        return None
    if isinstance(obj, property):
        obj = obj.fget
    elif isinstance(obj, functools.cached_property):
        obj = obj.func
    fn = inspect.getsourcefile(obj)
    if fn is None:
        raise SourceCodeError
    file = str(Path(fn).relative_to(pkg))
    source, start = inspect.getsourcelines(obj)
    end = start + len(source) - 1
    version = "main"  # To update to find correct version
    return f"https://github.com/bootphon/fastabx/blob/{version}/src/fastabx/{file}#L{start}-L{end}"
