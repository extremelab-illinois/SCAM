# SPDX-License-Identifier: MIT
"""Every public module must import with no optional dependency installed.

Sphinx's ``autodoc`` extension imports every module it documents. A future
top-level ``import cantera`` / ``import h5py`` / ``import pandas`` hoisted
into a 1-D module (all four are currently function-local — see
``docs/conf.py``'s ``autodoc_mock_imports`` comment) would silently break
the public API-reference build, and any user without the corresponding
extra installed. The 2-D modules are exempt: ``jax`` is a genuine
module-scope dependency there, they are excluded from the public API
reference (see ``docs/api/90_two_dimensional.md``), and they do not exist
at all on the ``public-release`` branch.

Run this in an environment with only the runtime dependencies (numpy,
scipy, pyyaml, matplotlib) plus pytest — no ``bprime``/``hdf5``/``jax``
extras — to actually exercise the guarantee; against the project's own
dev venv (which has every extra installed) it still catches an
accidentally-hoisted import, just not a missing one.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import scam

_EXEMPT = {"scam.__main__"}  # runs the CLI at import time (exits via SystemExit)


def _is_2d(name: str) -> bool:
    return "2d" in name


def _public_modules() -> list[str]:
    names = []
    for mod in pkgutil.walk_packages(scam.__path__, "scam."):
        if _is_2d(mod.name) or mod.name in _EXEMPT:
            continue
        names.append(mod.name)
    return sorted(names)


@pytest.mark.parametrize("module_name", _public_modules())
def test_module_imports(module_name: str) -> None:
    importlib.import_module(module_name)
