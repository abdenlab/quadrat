"""Sphinx configuration for the quadrat reference.

Reference-only for now. The docstrings in ``src`` are written for a
rendered build -- they carry ``:func:``, ``:data:`` and ``:class:``
roles throughout -- and until this existed nothing checked that any of
those references resolved, which is why several did not.
"""

project = "quadrat"
author = "Conrad"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
]

# Documentation Guide §3 assumes a bare `` `name` `` resolves to a Python
# object rather than rendering as emphasis, which is how the docstrings
# in `src` are written.
default_role = "py:obj"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "polars": ("https://docs.pola.rs/api/python/stable", None),
}
# Fetching an inventory is the only part of this build that needs the
# network. Keep it short so an offline build degrades to a warning
# quickly rather than hanging.
intersphinx_timeout = 5

# Deliberately off, and the reason is structural rather than a backlog.
# A nitpicky build fails on seven cross-references to members that
# exist in this tree and are deliberately *not* documented on this
# page -- `TrackWriter`, `plan_pileup`, `default_partitions`,
# `reporting`, and two module paths. Those docstrings are written for
# someone reading the source, where the reference resolves by being a
# name in scope; the page documents the published surface only.
#
# Turning this on would therefore mean either publishing those members
# or stripping the cross-references from the prose that explains them,
# and both are worse than an unresolved link on a page no part of
# which is about them. What it would catch -- a reference to a symbol
# that is nowhere at all -- the build already catches as a broken name
# in a docstring a reader is reading anyway.
nitpicky = False

html_theme = "sphinx_book_theme"
exclude_patterns = ["_build"]

# The one-line summaries in `src` are the documentation; do not let
# autodoc reorder them alphabetically and lose the foundational-to-
# derived order the modules are written in.
autodoc_member_order = "bysource"
