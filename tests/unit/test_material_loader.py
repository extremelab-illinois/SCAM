# SPDX-License-Identifier: MIT
"""Unit tests for io/material_loader.py."""
from pathlib import Path
import numpy as np
import pytest
import yaml

from scam.core.errors import SCAMInputError
from scam.io.material_loader import load_material, BPrimeTable, BPrimeTableRowFormat


MATERIALS_DIR = Path(__file__).parent.parent.parent / "scam" / "materials"


def test_tacot_component_cards():
    mat, _ = load_material(str(MATERIALS_DIR / "ablative_organic/tacot_v3.0.yaml"))
    comp_A = mat.components[0]
    comp_B = mat.components[1]
    assert comp_A.rho_0 > comp_A.rho_r   # rho_r < rho_0 validation
    assert comp_B.A_rate > 0
    assert comp_B.E_act > 0


def test_tacot_bprime_table_loaded():
    _, bpt = load_material(str(MATERIALS_DIR / "ablative_organic/tacot_v3.0.yaml"))
    assert bpt is not None


def test_bprime_lookup_in_range():
    _, bpt = load_material(str(MATERIALS_DIR / "ablative_organic/tacot_v3.0.yaml"))
    # Known grid point: T=1500, p=10000, Bg=0.2
    Bc, hw = bpt.lookup(T_w=1500.0, p_e=10000.0, B_g_prime=0.2)
    assert Bc >= 0.0
    assert hw > 0.0


def test_bprime_lookup_out_of_range_clamps():
    _, bpt = load_material(str(MATERIALS_DIR / "ablative_organic/tacot_v3.0.yaml"))
    # Temperatures below/above table range should clamp, not crash
    Bc_low,  _ = bpt.lookup(T_w=100.0,   p_e=10000.0, B_g_prime=0.0)
    Bc_high, _ = bpt.lookup(T_w=10000.0, p_e=10000.0, B_g_prime=0.0)
    assert Bc_low  >= 0.0
    assert Bc_high >= 0.0


def test_load_fiberform():
    mat, bpt = load_material(str(MATERIALS_DIR / "ablative_carbon/fiberform.yaml"))
    assert mat.name == "FiberForm"
    assert mat.decomposing is False
    assert bpt is None


def test_load_graphite_mersen2340():
    from scam.physics.properties import specific_heat, thermal_conductivity

    mat, bpt = load_material(str(MATERIALS_DIR / "ablative_carbon/graphite_mersen2340.yaml"))
    assert mat.name == "GRAPHITE_MERSEN_2340"
    assert mat.status == "provisional"
    assert mat.material_category == "ablative_carbon"
    assert mat.decomposing is False
    assert mat.components == []
    assert mat.rho_virgin == pytest.approx(1850.0)
    assert mat.rho_char == pytest.approx(1850.0)
    assert thermal_conductivity(mat, 300.0, mat.rho_virgin) == pytest.approx(102.0)
    assert specific_heat(mat, 1200.0, mat.rho_virgin) > 0.0
    assert bpt is not None
    assert mat.b_prime_ref is not None
    assert mat.b_prime_ref.path.endswith("calcarb_bprime_ace_air.yaml")


def test_load_pica():
    mat, bpt = load_material(str(MATERIALS_DIR / "ablative_organic/pica.yaml"))
    assert mat.name == "PICA"
    assert len(mat.components) >= 1


def test_load_teflon_estimate_card():
    mat, bpt = load_material(str(MATERIALS_DIR / "ablative_organic/teflon.yaml"))
    assert mat.name == "PTFE"
    assert mat.status == "estimate"
    assert mat.material_category == "ablative_organic"
    assert mat.rho_virgin == pytest.approx(2310.0)
    assert mat.rho_char == pytest.approx(mat.rho_virgin)
    assert mat.decomposing is False
    assert len(mat.components) == 0
    assert mat.pyro_elem_fracs is None
    # Kemp (1968) kinetic closure is now the canonical teflon.yaml path (no
    # B'-table); see test_load_teflon_bprime_legacy_card for the superseded
    # Gibbs-equilibrium card.
    assert bpt is None
    assert mat.b_prime_ref is None
    assert mat.kinetic_ablation is not None
    assert mat.kinetic_ablation.B == pytest.approx(3.0e19)
    assert mat.kinetic_ablation.E_a == pytest.approx(347272.0)


def test_load_teflon_bprime_legacy_card():
    mat, bpt = load_material(
        str(MATERIALS_DIR / "ablative_organic/teflon_bprime_legacy.yaml")
    )
    assert mat.name == "PTFE_bprime_legacy"
    assert mat.status == "estimate"
    assert mat.material_category == "ablative_organic"
    assert mat.rho_virgin == pytest.approx(2200.0)
    assert mat.rho_char == pytest.approx(mat.rho_virgin)
    assert mat.decomposing is False
    assert len(mat.components) == 0
    assert mat.pyro_elem_fracs is None
    assert mat.kinetic_ablation is None
    assert isinstance(bpt, BPrimeTable)
    assert bpt.target_element == "F"
    assert bpt.surface_source_target_fraction == pytest.approx(0.759805)


def test_kinetic_ablation_and_bprime_table_mutually_exclusive(tmp_path):
    bprime_abs_path = str(MATERIALS_DIR / "ablative_organic/teflon_bprime_air.yaml")
    bad_card = tmp_path / "bad_teflon.yaml"
    bad_card.write_text(
        f"""
name: BAD_PTFE
rho_virgin: 2310.0
rho_char: 2310.0
gamma_resin: 1.0
decomposing: false
components: []
k_virgin: [[300.0, 0.25], [1200.0, 0.7]]
k_char: [[300.0, 0.25], [1200.0, 0.7]]
cp_virgin: [[300.0, 1000.0], [1200.0, 2300.0]]
cp_char: [[300.0, 1000.0], [1200.0, 2300.0]]
h_g: [[300.0, 0.0], [1200.0, 1.0e6]]
b_prime_table: {bprime_abs_path}
kinetic_ablation:
  B: 3.0e19
  E_a: 347272.0
  h_ablation_total_coeffs: [1.0e6, 0.0, 0.0]
  h_ablation_reaction_coeffs: [1.0e6, 0.0]
"""
    )
    with pytest.raises(SCAMInputError, match="cannot specify both"):
        load_material(str(bad_card))


def test_load_inert_equal_density_material():
    mat, bpt = load_material(str(MATERIALS_DIR / "subsurface/fourier.yaml"))
    assert mat.name == "Fourier"
    assert mat.decomposing is False
    assert mat.rho_virgin == pytest.approx(1000.0)
    assert mat.rho_char == pytest.approx(1000.0)
    assert bpt is None


# ---------------------------------------------------------------------------
# TACOT 3.0 tests
# ---------------------------------------------------------------------------

def test_load_tacot():
    mat, bpt = load_material(str(MATERIALS_DIR / "ablative_organic/tacot_v3.0.yaml"))
    assert mat.name == "TACOT"
    assert mat.dataset_version == "3.0"
    assert mat.card_version == "1.0"
    assert mat.status == "verified"
    assert mat.rho_virgin == pytest.approx(280.0)
    assert mat.rho_char   == pytest.approx(220.0)
    assert len(mat.components) == 2   # inert reinforcement skipped
    assert mat.components[0].name == "resin_A"
    assert mat.components[0].rho_0  == pytest.approx(30.0)
    assert mat.components[0].rho_r  == pytest.approx(0.0)
    assert mat.components[0].E_act  == pytest.approx(71131.0, rel=1e-3)
    assert mat.components[1].rho_0  == pytest.approx(30.0)    # correct merge: 0.25*120, rho_r=0
    assert mat.components[1].rho_r  == pytest.approx(0.0)
    assert mat.components[1].A_rate == pytest.approx(4.97777e8, rel=1e-4)
    assert mat.components[1].E_act  == pytest.approx(169975.0, rel=1e-3)
    # in-plane conductivity tables stored
    assert mat.k_virgin_ip_table is not None
    assert mat.k_char_ip_table is not None


def test_tacot_bprime_table():
    _, bpt = load_material(str(MATERIALS_DIR / "ablative_organic/tacot_v3.0.yaml"))
    # New table is Cantera-generated 3-D grid format (BPrimeTable)
    assert isinstance(bpt, BPrimeTable)
    # Ablating condition: high T_wall, reasonable pressure
    Bc, hw = bpt.lookup(T_w=3000.0, p_e=101325.0, B_g_prime=0.5)
    assert np.isfinite(Bc)
    assert np.isfinite(hw)
    assert Bc >= 0.0
    h_g, h_c = bpt.surface_enthalpies(1500.0, 101325.0)
    assert np.isfinite(h_g)
    assert np.isfinite(h_c)


def test_bprime_table_interpolates_fixed_composition_surface_enthalpies(tmp_path):
    table_path = tmp_path / "bprime_3d.yaml"
    table_path.write_text(yaml.safe_dump({
        "T_wall_K": [1000.0, 2000.0],
        "p_e_Pa": [10000.0, 100000.0],
        "B_g_prime": [0.0, 1.0],
        "B_c_prime": np.zeros((2, 2, 2)).tolist(),
        "h_wall": np.zeros((2, 2, 2)).tolist(),
        "h_g": [[10.0, 20.0], [30.0, 40.0]],
        "h_c": [[100.0, 200.0], [300.0, 400.0]],
    }))

    table = BPrimeTable(str(table_path))
    h_g, h_c = table.surface_enthalpies(1500.0, 55000.0, Z_C_pyro=0.9)

    assert h_g == pytest.approx(25.0)
    assert h_c == pytest.approx(250.0)


def test_bprime_table_accepts_generic_surface_array(tmp_path):
    table_path = tmp_path / "bprime_generic.yaml"
    table_path.write_text(yaml.safe_dump({
        "T_wall_K": [1000.0, 2000.0],
        "p_e_Pa": [10000.0],
        "B_g_prime": [0.0, 1.0],
        "Bprime_surface": [[[0.1, 0.2]], [[0.3, 0.4]]],
        "h_wall": [[[10.0, 20.0]], [[30.0, 40.0]]],
        "target_element": "F",
        "surface_source_target_fraction": 0.75,
    }))

    table = BPrimeTable(str(table_path))
    Bp, hw = table.lookup(T_w=1500.0, p_e=10000.0, B_g_prime=0.5)

    assert table.target_element == "F"
    assert table.surface_source_target_fraction == pytest.approx(0.75)
    assert Bp == pytest.approx(0.25)
    assert hw == pytest.approx(25.0)


def test_bprime_table_interpolates_4d_surface_enthalpy_composition(tmp_path):
    table_path = tmp_path / "bprime_4d.yaml"
    table_path.write_text(yaml.safe_dump({
        "axes": {
            "T_wall_K": [1000.0, 2000.0],
            "p_e_Pa": [10000.0, 100000.0],
            "B_g_prime": [0.0, 1.0],
            "Z_C_pyro": [0.2, 0.6],
        },
        "B_c_prime": np.zeros((2, 2, 2, 2)).tolist(),
        "h_wall": np.zeros((2, 2, 2, 2)).tolist(),
        "h_g": [
            [[10.0, 30.0], [20.0, 40.0]],
            [[50.0, 70.0], [60.0, 80.0]],
        ],
        "h_c": [[100.0, 200.0], [300.0, 400.0]],
    }))

    table = BPrimeTable(str(table_path))
    h_g, h_c = table.surface_enthalpies(1500.0, 55000.0, Z_C_pyro=0.4)

    assert h_g == pytest.approx(45.0)
    assert h_c == pytest.approx(250.0)


def test_bprime_table_rejects_incomplete_surface_enthalpy_data(tmp_path):
    table_path = tmp_path / "bad_bprime.yaml"
    table_path.write_text(yaml.safe_dump({
        "T_wall_K": [1000.0, 2000.0],
        "p_e_Pa": [10000.0, 100000.0],
        "B_g_prime": [0.0, 1.0],
        "B_c_prime": np.zeros((2, 2, 2)).tolist(),
        "h_wall": np.zeros((2, 2, 2)).tolist(),
        "h_g": [[10.0, 20.0], [30.0, 40.0]],
    }))

    with pytest.raises(SCAMInputError, match="must be provided together"):
        BPrimeTable(str(table_path))


def test_tacot_e_over_r_conversion():
    """E_act from E_over_R must match R * E_over_R within 0.1%."""
    import math
    mat, _ = load_material(str(MATERIALS_DIR / "ablative_organic/tacot_v3.0.yaml"))
    R = 8.314
    assert mat.components[0].E_act == pytest.approx(8555.56 * R, rel=1e-3)
    assert mat.components[1].E_act == pytest.approx(20444.44 * R, rel=1e-3)
