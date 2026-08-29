# SPDX-License-Identifier: MIT
"""Unit tests for physics/properties.py."""
import numpy as np
import pytest

from scam.config.material import MaterialCard
from scam.physics.properties import (
    eps_virgin, thermal_conductivity, specific_heat,
    eval_k_array, eval_cp_array,
)


def _make_mat(kv: float = 2.0, kc: float = 0.5,
              cpv: float = 1000.0, cpc: float = 800.0,
              rho_v: float = 280.0, rho_c: float = 180.0) -> MaterialCard:
    T_pts = np.array([200.0, 3000.0])
    k_v = np.column_stack([T_pts, np.full(2, kv)])
    k_c = np.column_stack([T_pts, np.full(2, kc)])
    cp_v = np.column_stack([T_pts, np.full(2, cpv)])
    cp_c = np.column_stack([T_pts, np.full(2, cpc)])
    hg = np.column_stack([T_pts, np.zeros(2)])
    return MaterialCard(
        name="TEST", rho_virgin=rho_v, rho_char=rho_c, gamma_resin=0.0,
        components=[], k_virgin_table=k_v, k_char_table=k_c,
        cp_virgin_table=cp_v, cp_char_table=cp_c, h_g_table=hg,
        decomposing=False,
    )


def test_eps_virgin_full():
    mat = _make_mat()
    # pure virgin: eps = 1
    assert eps_virgin(mat, 280.0) == pytest.approx(1.0)


def test_eps_virgin_char():
    mat = _make_mat()
    assert eps_virgin(mat, 180.0) == pytest.approx(0.0)


def test_eps_virgin_mid():
    mat = _make_mat(rho_v=280.0, rho_c=180.0)
    # rho = 230 is midway between char and virgin
    assert eps_virgin(mat, 230.0) == pytest.approx(0.5)


def test_thermal_conductivity_pure_virgin():
    mat = _make_mat(kv=2.0, kc=0.5)
    k = thermal_conductivity(mat, 300.0, 280.0)
    assert k == pytest.approx(2.0)


def test_thermal_conductivity_pure_char():
    mat = _make_mat(kv=2.0, kc=0.5)
    k = thermal_conductivity(mat, 300.0, 180.0)
    assert k == pytest.approx(0.5)


def test_thermal_conductivity_mixed():
    mat = _make_mat(kv=2.0, kc=0.5, rho_v=280.0, rho_c=180.0)
    k = thermal_conductivity(mat, 300.0, 230.0)
    assert k == pytest.approx(0.5 * 2.0 + 0.5 * 0.5)


def test_specific_heat_pure_virgin():
    mat = _make_mat(cpv=1000.0, cpc=800.0)
    cp = specific_heat(mat, 300.0, 280.0)
    assert cp == pytest.approx(1000.0)


def test_eval_k_array():
    mat = _make_mat(kv=2.0, kc=0.5)
    T = np.array([300.0, 300.0, 300.0])
    rho = np.array([280.0, 230.0, 180.0])
    layer_id = np.array([0, 0, 0], dtype=int)
    k_arr = eval_k_array([mat], T, rho, layer_id)
    assert k_arr[0] == pytest.approx(2.0)
    assert k_arr[-1] == pytest.approx(0.5)
    assert k_arr[1] == pytest.approx(0.5 * 2.0 + 0.5 * 0.5)


def test_surface_emissivity_scalar_blend():
    """Scalar virgin/char emissivity blended by char fraction (backward compat)."""
    from scam.physics.properties import surface_emissivity
    mat = _make_mat()
    mat.emissivity = 0.8          # virgin
    mat.emissivity_char = 0.9     # char
    # rho_char=180 -> fully char -> 0.9 ; rho_virgin=280 -> virgin -> 0.8
    assert surface_emissivity(mat, 1500.0, 180.0) == pytest.approx(0.9)
    assert surface_emissivity(mat, 1500.0, 280.0) == pytest.approx(0.8)
    assert surface_emissivity(mat, 1500.0, 230.0) == pytest.approx(0.85)  # halfway


def test_surface_emissivity_temperature_tables():
    """Temperature-dependent emissivity tables interpolate and blend by phase."""
    from scam.physics.properties import surface_emissivity
    mat = _make_mat()
    mat.emissivity_virgin_table = np.array([[300.0, 0.80], [2000.0, 0.70]])
    mat.emissivity_char_table = np.array([[300.0, 0.85], [2000.0, 0.95]])
    # Fully charred surface follows the char table.
    assert surface_emissivity(mat, 300.0, 180.0) == pytest.approx(0.85)
    assert surface_emissivity(mat, 2000.0, 180.0) == pytest.approx(0.95)
    assert surface_emissivity(mat, 1150.0, 180.0) == pytest.approx(0.90)  # midpoint in T
    # Virgin surface follows the virgin table.
    assert surface_emissivity(mat, 2000.0, 280.0) == pytest.approx(0.70)


def test_surface_emissivity_single_value():
    """No char distinction -> single emissivity regardless of rho/T."""
    from scam.physics.properties import surface_emissivity
    mat = _make_mat()
    mat.emissivity = 0.88
    mat.emissivity_char = -1.0
    assert surface_emissivity(mat, 1500.0, 180.0) == pytest.approx(0.88)
    assert surface_emissivity(mat, 1500.0, 280.0) == pytest.approx(0.88)
