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
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
    "sphinxarg.ext",
]

sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "examples/gallery",
    "remove_config_comments": True,
    "download_all_examples": False,
    "write_computation_times": False,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/main", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "polars": ("https://docs.pola.rs/api/python/stable", None),
}

project = "fastabx"
author = metadata(project)["Author"]
copyright = f"{datetime.now(tz=UTC).year}, {author}"
parsed_version = parse(metadata(project)["Version"])
version = parsed_version.base_version
release = version
is_released = not (parsed_version.is_devrelease or parsed_version.is_prerelease or parsed_version.local)
linkcode_ref = version if is_released else "main"

autodoc_typehints = "description"
autodoc_preserve_defaults = True
add_function_parentheses = False
exclude_patterns = ["build"]
nitpicky = True
nitpick_ignore = [
    ("py:class", "polars.dataframe.frame.DataFrame"),
    ("py:class", "fastabx.accessor.ArrayLike"),
    ("py:class", "numpy.int64"),
    ("py:class", "fastabx.verify.CellErrorType"),
    ("py:class", "fastabx.verify.LevelsErrorType"),
]
html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["sphinx_gallery_overrides.css"]
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
    return f"https://github.com/bootphon/fastabx/blob/{linkcode_ref}/src/fastabx/{file}#L{start}-L{end}"
