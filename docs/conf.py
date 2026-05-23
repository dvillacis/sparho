"""Sphinx configuration for sparho documentation."""

from __future__ import annotations

from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

# sparho is installed into the environment (pip / maturin develop) before docs
# are built; do not prepend `python/` to sys.path or the source tree will
# shadow the installed package and the compiled `_core` extension will be
# missing.

project = "sparho"
author = "David Villacis"
copyright = f"{datetime.now(tz=UTC).year}, {author}"

try:
    release = _pkg_version("sparho")
except PackageNotFoundError:
    release = "0.1.0.dev0"
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "numpydoc",
    "myst_parser",
    "sphinx_gallery.gen_gallery",
    "sphinxcontrib.bibtex",
]

# Single bibliography file backing both theory pages and docstring citations.
bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "alpha"
bibtex_reference_style = "author_year"

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"
exclude_patterns = ["_build", "examples/README.txt", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = f"sparho {release}"
html_static_path = ["_static"]

myst_enable_extensions = [
    "deflist",
    "colon_fence",
    "smartquotes",
    "dollarmath",
    "amsmath",
]
myst_heading_anchors = 3

# Autodoc / numpydoc behavior. numpydoc consumes numpy-style docstrings;
# napoleon stays on so that any incidentally-Google-styled docstrings also
# parse. Keep type-hint signatures out of the parameter list — the source
# already carries them via Python type annotations.
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_class_signature = "separated"
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# sphinx-gallery: render the runnable example scripts into docs/examples_built/.
sphinx_gallery_conf = {
    "examples_dirs": ["examples"],
    "gallery_dirs": ["examples_built"],
    "filename_pattern": r"/plot_",
    "ignore_pattern": r"__init__\.py",
    "remove_config_comments": True,
    "download_all_examples": False,
    "plot_gallery": "True",
    "abort_on_example_error": True,
    "show_memory": False,
    "thumbnail_size": (320, 224),
}

# Surface warnings as build errors when invoked with `-W`.
nitpicky = False  # numpydoc + Protocol references still produce noise on Sphinx 9
suppress_warnings = ["myst.header"]
