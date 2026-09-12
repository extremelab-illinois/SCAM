# SPDX-License-Identifier: MIT
"""Sphinx configuration for the SCAM manual.

This file is IDENTICAL on `main` and `public-release`. It must never contain
branch detection: ReadTheDocs builds from a shallow detached HEAD where branch
names are unreliable, and a branch-conditional conf.py would make the two
builds non-comparable. Branch differences are expressed only by which files
exist -- see the `:glob:` toctree in each section's index.md (docs/theory,
docs/verification, docs/api) and the `9x_` filename convention documented in
CLAUDE.md and CONTRIBUTING.md: any file matching `docs/**/9[0-9]_*.md` is
main-only and is dropped during promotion to public-release.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
# Make autodoc importable from any worktree without reinstalling the package.
sys.path.insert(0, str(_REPO))

# -- Project -----------------------------------------------------------------
project = "SCAM"
author = "Francesco Panerai and contributors"
copyright = "2026, The Board of Trustees of the University of Illinois"

# Version is parsed from *this tree's* pyproject.toml -- deliberately not from
# importlib.metadata (would return whichever copy happens to be pip-installed,
# wrong when building a second branch from a git worktree) and not from
# scam.__version__ (currently desynced from pyproject.toml on public-release).
# `main` and `public-release` intentionally carry different version numbers
# and must never be synchronised (see CLAUDE.md, "pyproject.toml" entry).
_pyproject_text = (_REPO / "pyproject.toml").read_text(encoding="utf-8")
_version_match = re.search(r'^version\s*=\s*"([^"]+)"', _pyproject_text, re.M)
release = _version_match.group(1) if _version_match else "0.0.0"
version = ".".join(release.split(".")[:2])

# -- General -------------------------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
]

source_suffix = {".md": "markdown"}
root_doc = "index"

exclude_patterns = [
    "_build",
    ".DS_Store",
    "Thumbs.db",
    "README.md",       # GitHub folder pointer, not a manual page
    "**/README.md",
    "planning/**",      # private working notes -- excluded on BOTH branches;
                        # never rendered even locally on main (see CLAUDE.md)
]

# The docs are full of `--flag` and '...' in prose; smart typography would
# silently turn `--verbose` into an en dash.
smartquotes = False

# Never enable: with autodoc_typehints="description" this produces hundreds of
# unresolvable numpy/scipy type cross-references and -W becomes permanently
# unachievable.
nitpicky = False

language = "en"

# -- MyST ----------------------------------------------------------------------
myst_enable_extensions = [
    "amsmath",       # unused today, free insurance for future theory chapters
    "attrs_inline",
    "colon_fence",    # ::: admonitions, survive nesting inside code fences
    "deflist",        # nomenclature / glossary tables
    "dollarmath",     # required: docs/verification/verification.md uses $$...$$
    "fieldlist",
    "tasklist",
]
myst_heading_anchors = 3   # keeps existing [text](file.md#anchor) links working
# Deliberately NOT enabled: "linkify" (extra dependency, mangles bare text),
# "substitution" (any literal {{ in the inherited docs would break the build).

# -- autodoc ---------------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
    "exclude-members": "__weakref__,__dict__,__module__",
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"
autodoc_preserve_defaults = True          # show `= 1e-3`, not the repr
autodoc_class_signature = "separated"

# `from __future__ import annotations` in most modules means annotations are
# strings; this keeps them readable rather than fully-qualified.
autodoc_type_aliases = {
    "ScalarBC": "scam.config.boundary.ScalarBC",
}

# Only the 2-D modules import an optional dependency at module scope (jax).
# Every 1-D optional import (cantera, h5py, pandas) is function-local, so the
# public API build needs no mocking at all -- see tests/unit/test_module_imports.py.
# Mocks are kept here as insurance against a future top-level import.
autodoc_mock_imports = ["jax", "jaxlib", "cantera", "h5py", "pandas", "mutationpp"]

# -- napoleon ----------------------------------------------------------------
napoleon_google_docstring = False    # the repo is NumPy-style throughout
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
# Deliberately NOT napoleon_attr_annotations=True: dataclass fields here are
# documented with autodoc-native `#:` attribute doc-comments
# (ModuleAnalyzer.find_attr_docs), not napoleon Attributes: sections.
# napoleon_attr_annotations independently injects an annotation-derived
# attribute doc for the same class members, and having both active
# registers each `#:`-commented field twice -> "duplicate object
# description" warnings under -W.
napoleon_attr_annotations = False

# -- intersphinx ---------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

# -- HTML ----------------------------------------------------------------------
html_theme = "furo"
html_title = f"SCAM {release}"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/extremelab-illinois/SCAM/",
    "source_branch": "main",
    "source_directory": "docs/",
}

# -- linkcheck (run on demand -- see docs/Makefile; never in the -W gate) ------
linkcheck_anchors = True
linkcheck_ignore = [r"^https://pato\.ac"]     # occasionally unreachable
linkcheck_timeout = 20
