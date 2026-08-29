# SPDX-License-Identifier: MIT
"""Unit tests for physics/bprime_evaluator.py."""
import math
from pathlib import Path

import pytest


MATERIALS_DIR = Path(__file__).parent.parent.parent / "scam" / "materials"


def test_pyrolysis_target_fraction_tacot_config():
    pytest.importorskip("cantera")

    from scam.physics.bprime_evaluator import BprimeEvaluator

    try:
        ev = BprimeEvaluator.from_config(MATERIALS_DIR / "ablative_organic/tacot_v3.0_bprime_config.yaml")
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip(f"Cantera backend unavailable: {exc}")

    assert ev.pyrolysis_target_fraction("C") == pytest.approx(0.494996, abs=2e-5)
    assert ev.pyrolysis_target_fraction("C") == pytest.approx(0.494996, abs=2e-5)


def test_teflon_config_uses_fluorine_target():
    pytest.importorskip("cantera")

    from scam.physics.bprime_evaluator import BprimeEvaluator

    try:
        ev = BprimeEvaluator.from_config(MATERIALS_DIR / "ablative_organic/teflon_bprime_config.yaml")
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip(f"Cantera backend unavailable: {exc}")

    assert ev.target_element == "F"
    assert ev.surface_source_target_fraction == pytest.approx(0.759805)
    assert ev.pyrolysis_target_fraction("F") == pytest.approx(0.7598175, rel=1e-6)

    Bp, h_wall = ev.lookup(1500.0, 101325.0, 2.0)
    h_g, h_surface = ev.surface_enthalpies(1500.0, 101325.0)

    assert Bp >= 0.0
    assert math.isfinite(h_wall)
    assert h_surface == pytest.approx(h_g)
