# SPDX-License-Identifier: MIT
"""Unit tests for physics/kinetic_ablation.py — Kemp (1968) Eq. 12 closure.

Expected m_dot values below were computed once from the teflon.yaml card's
own constants (see scripts/manual verification in this session) and are
hardcoded here rather than imported from studies/teflon_ablation/, to keep
the test suite decoupled from the research/scratch directory. Cross-check
against studies/teflon_ablation/compare_teflon_kinetic_models.py if these
ever need to be regenerated.
"""
from pathlib import Path

import pytest

from scam.io.material_loader import load_material
from scam.physics.kinetic_ablation import (
    h_ablation_reaction,
    h_ablation_total,
    m_dot_kemp,
)
from scam.physics.properties import thermal_conductivity

MATERIALS_DIR = Path(__file__).parent.parent.parent / "scam" / "materials"


@pytest.fixture(scope="module")
def teflon_kinetic():
    mat, bpt = load_material(str(MATERIALS_DIR / "ablative_organic/teflon.yaml"))
    assert bpt is None
    assert mat.kinetic_ablation is not None
    return mat


def _m_dot_at(mat, T_w):
    ka = mat.kinetic_ablation
    k_w = thermal_conductivity(mat, T_w, mat.rho_char)
    h_tot = h_ablation_total(T_w, ka.h_ablation_total_coeffs)
    return m_dot_kemp(T_w, mat.rho_char, k_w, ka.B, ka.E_a, h_tot)


@pytest.mark.parametrize(
    "T_w,expected_m_dot",
    [
        (900.0, 0.04710021326416285),
        (950.0, 0.16929489767873304),
        (1000.0, 0.5365279984977894),
        (1050.0, 1.526121970151949),
        (1100.0, 3.953596143253835),
    ],
)
def test_m_dot_kemp_matches_teflon_card(teflon_kinetic, T_w, expected_m_dot):
    assert _m_dot_at(teflon_kinetic, T_w) == pytest.approx(expected_m_dot, rel=1e-9)


def test_kemp_fig1_order_of_magnitude_jump(teflon_kinetic):
    """Kemp's Fig. 1: 'nearly an order of magnitude increase in v_so and m_so
    as T_w goes from 950 to 1050 K' — reproduced here as a ~9x ratio.
    """
    m_950 = _m_dot_at(teflon_kinetic, 950.0)
    m_1050 = _m_dot_at(teflon_kinetic, 1050.0)
    assert m_1050 / m_950 == pytest.approx(9.0, rel=0.05)


def test_m_dot_kemp_monotonic_increasing(teflon_kinetic):
    temps = [700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0]
    values = [_m_dot_at(teflon_kinetic, T) for T in temps]
    assert all(b > a for a, b in zip(values, values[1:]))


def test_m_dot_kemp_nonphysical_inputs_return_zero():
    assert m_dot_kemp(0.0, 2310.0, 0.5, 3e19, 3.47e5, 2.0e6) == 0.0
    assert m_dot_kemp(1000.0, 2310.0, 0.5, 3e19, 3.47e5, 0.0) == 0.0
    assert m_dot_kemp(1000.0, 0.0, 0.5, 3e19, 3.47e5, 2.0e6) == 0.0
    assert m_dot_kemp(1000.0, 2310.0, 0.0, 3e19, 3.47e5, 2.0e6) == 0.0


def test_h_ablation_reaction_matches_card(teflon_kinetic):
    ka = teflon_kinetic.kinetic_ablation
    assert h_ablation_reaction(1000.0, ka.h_ablation_reaction_coeffs) == pytest.approx(
        1494641.27, rel=1e-9
    )
