# SPDX-License-Identifier: MIT
"""Focused tests for pretabulated B' surface enthalpies."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from scam.io.material_loader import BPrimeTable
from scam.tools.generate_bprime import cases_to_scam_yaml, csv_to_scam_yaml_4d


def test_cases_to_yaml_uses_explicit_nominal_zc_for_surface_enthalpy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[float | None] = []

    class FakeEvaluator:
        def pyrolysis_target_fraction(self, _element):
            return 0.495

        def surface_enthalpies(self, T, p, z_c):
            calls.append(z_c)
            return T + p + 1000.0 * z_c, T - p

    monkeypatch.setattr(
        "scam.physics.bprime_evaluator.BprimeEvaluator.from_config",
        lambda _path: FakeEvaluator(),
    )

    output = tmp_path / "table.yaml"
    cases_to_scam_yaml(
        T_axis=np.array([1000.0, 2000.0]),
        p_axis=np.array([10000.0, 100000.0]),
        Bg_axis=np.array([0.0, 1.0]),
        Bc_arr=np.zeros((2, 2, 2)),
        hw_arr=np.zeros((2, 2, 2)),
        output_path=output,
        bprime_config_path=tmp_path / "config.yaml",
    )

    raw = yaml.safe_load(output.read_text())
    assert raw["surface_enthalpy_Z_C_pyro"] == pytest.approx(0.495)
    assert calls == [0.495] * 4

    table = BPrimeTable(str(output))
    h_g, h_c = table.surface_enthalpies(1500.0, 55000.0)
    assert h_g == pytest.approx(56995.0)
    assert h_c == pytest.approx(-53500.0)


def test_cases_to_yaml_records_generic_target_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[float | None] = []

    class FakeEvaluator:
        target_element = "F"

        def pyrolysis_target_fraction(self, element):
            assert element == "F"
            return 0.76

        def surface_enthalpies(self, T, p, z_target):
            calls.append(z_target)
            return T + 100.0 * z_target, T + 200.0 * z_target

    monkeypatch.setattr(
        "scam.physics.bprime_evaluator.BprimeEvaluator.from_config",
        lambda _path: FakeEvaluator(),
    )

    output = tmp_path / "table.yaml"
    cases_to_scam_yaml(
        T_axis=np.array([1000.0]),
        p_axis=np.array([100000.0]),
        Bg_axis=np.array([0.0, 1.0]),
        Bc_arr=np.array([[[0.1, 0.2]]]),
        hw_arr=np.array([[[10.0, 20.0]]]),
        output_path=output,
        bprime_config_path=tmp_path / "config.yaml",
        target_element="F",
        surface_source_target_fraction=0.75,
    )

    raw = yaml.safe_load(output.read_text())
    assert raw["target_element"] == "F"
    assert raw["surface_source_target_fraction"] == pytest.approx(0.75)
    assert raw["surface_enthalpy_target_element"] == "F"
    assert raw["surface_enthalpy_target_fraction"] == pytest.approx(0.76)
    assert "surface_enthalpy_Z_C_pyro" not in raw
    assert raw["Bprime_surface"] == raw["B_c_prime"]
    assert calls == [0.76]


def test_4d_yaml_stores_composition_dependent_surface_enthalpy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeEvaluator:
        def surface_enthalpies(self, T, p, z_c):
            return T + p + 1000.0 * z_c, T - p

    monkeypatch.setattr(
        "scam.physics.bprime_evaluator.BprimeEvaluator.from_config",
        lambda _path: FakeEvaluator(),
    )

    csv_paths = {}
    for z_c in (0.2, 0.6):
        csv_path = tmp_path / f"zc_{z_c}.csv"
        rows = ["T_wall_K,pressure_Pa,Bg,Bprime_c_eq,h_wall_gas_J_kg,converged"]
        for T in (1000.0, 2000.0):
            for p in (10000.0, 100000.0):
                for bg in (0.0, 1.0):
                    rows.append(f"{T},{p},{bg},{z_c + bg},{T + p},True")
        csv_path.write_text("\n".join(rows) + "\n")
        csv_paths[z_c] = csv_path

    output = tmp_path / "table_4d.yaml"
    csv_to_scam_yaml_4d(
        csv_paths,
        output,
        bprime_config_path=tmp_path / "config.yaml",
    )

    table = BPrimeTable(str(output))
    h_g, h_c = table.surface_enthalpies(1500.0, 55000.0, Z_C_pyro=0.4)
    assert h_g == pytest.approx(56900.0)
    assert h_c == pytest.approx(-53500.0)
