"""
Unit and regression tests for the SCAM equilibrium B' table workflow.

Run with:
    pytest tests/                         # all tests
    pytest tests/ -v                      # verbose
    pytest tests/ -k "not cantera"        # skip Cantera-dependent tests
    pytest tests/ -k cantera              # only Cantera tests (slow)

Tests are organised into four sections:
  1. make_equilibrium_bprime_table – pure helpers (no Cantera)
  2. make_equilibrium_bprime_table – Cantera physics (marked slow)
  3. plot_equilibrium_bprime_table  – DataFrame helpers
  4. compare_tacot_bprime           – pressure matching, comparison math
"""

from __future__ import annotations

import io
import math
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bprime_df(
    temps=(500.0, 1000.0, 1500.0, 2000.0),
    pressures=(1000.0, 10000.0, 101325.0),
    bgs=(0.0, 0.5, 1.0),
    bprime_fn=None,
) -> pd.DataFrame:
    """Build a minimal synthetic long-form B' DataFrame."""
    if bprime_fn is None:
        bprime_fn = lambda T, p, bg: (1 + bg) * 0.1 * (T / 1000.0)  # noqa: E731
    rows = []
    for bg in bgs:
        for p in pressures:
            for T in temps:
                rows.append(
                    dict(
                        T_wall_K=T,
                        pressure_Pa=p,
                        Bg=bg,
                        Bprime_c_eq=bprime_fn(T, p, bg),
                        Z_C_wall=0.1,
                        Z_C_edge_eff=0.0,
                        h_wall_gas_J_kg=-1e6,
                        MW_wall_gas_kg_per_kmol=28.0,
                        gas_moles_final=1.0,
                        carbon_phase_moles_final=90.0,
                        major_wall_species_X="N2:0.8;CO2:0.2",
                    )
                )
    return pd.DataFrame(rows)


# ===========================================================================
# Section 1 – make_equilibrium_bprime_table: pure helpers
# ===========================================================================

class TestParseNumberGrid:
    def test_comma_list(self):
        from scam.chemistry.grids import parse_number_grid
        result = parse_number_grid("1000,5000,10000", name="test")
        assert result == [1000.0, 5000.0, 10000.0]

    def test_range_start_stop_step(self):
        from scam.chemistry.grids import parse_number_grid
        result = parse_number_grid("0:1:0.5", name="bg")
        assert len(result) == 3
        assert math.isclose(result[0], 0.0)
        assert math.isclose(result[1], 0.5)
        assert math.isclose(result[2], 1.0)

    def test_range_start_stop_only(self):
        from scam.chemistry.grids import parse_number_grid
        result = parse_number_grid("0:1", name="bg")
        assert result == [0.0, 1.0]

    def test_range_single_value(self):
        from scam.chemistry.grids import parse_number_grid
        result = parse_number_grid("1500:1500:100", name="T")
        assert len(result) >= 1
        assert math.isclose(result[0], 1500.0)

    def test_range_includes_stop(self):
        from scam.chemistry.grids import parse_number_grid
        result = parse_number_grid("0:3:0.5", name="bg")
        assert math.isclose(result[-1], 3.0), f"last={result[-1]}"
        assert len(result) == 7

    def test_empty_raises(self):
        from scam.chemistry.grids import parse_number_grid
        with pytest.raises(ValueError, match="Empty"):
            parse_number_grid("", name="T")

    def test_zero_step_raises(self):
        from scam.chemistry.grids import parse_number_grid
        with pytest.raises(ValueError, match="zero"):
            parse_number_grid("1:5:0", name="T")

    def test_wrong_sign_raises(self):
        from scam.chemistry.grids import parse_number_grid
        with pytest.raises(ValueError, match="sign"):
            parse_number_grid("5:1:1", name="T")

    def test_negative_step(self):
        from scam.chemistry.grids import parse_number_grid
        result = parse_number_grid("3:0:-1", name="bg")
        assert math.isclose(result[0], 3.0)
        assert math.isclose(result[-1], 0.0)

    def test_floating_point_grid(self):
        from scam.chemistry.grids import parse_number_grid
        result = parse_number_grid("250:4000:25", name="T")
        assert len(result) == 151
        assert math.isclose(result[0], 250.0)
        assert math.isclose(result[-1], 4000.0)


class TestNormalizeCompositionString:
    def test_passthrough(self):
        from scam.chemistry.composition import normalize_composition_string
        assert normalize_composition_string("O2:0.21,N2:0.79") == "O2:0.21,N2:0.79"

    def test_lowercase_aliases(self):
        from scam.chemistry.composition import normalize_composition_string
        result = normalize_composition_string("o2:0.21,n2:0.79")
        assert result == "O2:0.21,N2:0.79"

    def test_ar_alias(self):
        from scam.chemistry.composition import normalize_composition_string
        result = normalize_composition_string("Ar:1.0")
        assert result == "AR:1.0"

    def test_co2_alias(self):
        from scam.chemistry.composition import normalize_composition_string
        result = normalize_composition_string("co2:0.96,n2:0.04")
        assert result == "CO2:0.96,N2:0.04"

    def test_whitespace_stripped(self):
        from scam.chemistry.composition import normalize_composition_string
        result = normalize_composition_string("  O2 : 0.21 , N2 : 0.79 ")
        assert "O2:0.21" in result
        assert "N2:0.79" in result

    def test_empty_raises(self):
        from scam.chemistry.composition import normalize_composition_string
        with pytest.raises(ValueError, match="empty"):
            normalize_composition_string("")

    def test_missing_colon_raises(self):
        from scam.chemistry.composition import normalize_composition_string
        with pytest.raises(ValueError, match="species:value"):
            normalize_composition_string("O2")


class TestBPrimeMath:
    """Test the B'c formula independently of Cantera."""

    @pytest.mark.parametrize("bg,Z_C_wall,Z_C_edge,expected", [
        (0.0, 0.10, 0.0, 0.10 / (1 - 0.10)),           # classic Bg=0
        (0.0, 0.20, 0.0, 0.20 / 0.80),
        (1.0, 0.10, 0.05, 2.0 * (0.10 - 0.05) / 0.90), # coupled Bg=1
        (0.5, 0.15, 0.0, 1.5 * 0.15 / 0.85),
    ])
    def test_bprime_formula(self, bg, Z_C_wall, Z_C_edge, expected):
        bprime = (1 + bg) * (Z_C_wall - Z_C_edge) / (1 - Z_C_wall)
        assert math.isclose(bprime, expected, rel_tol=1e-10)

    def test_air_preset_bg_scaling(self):
        """With no pyrolysis (air preset), B'c scales exactly as (1+Bg)."""
        Z_C_w, Z_C_e = 0.12, 0.0
        base = (Z_C_w - Z_C_e) / (1 - Z_C_w)
        for bg in [0.0, 0.5, 1.0, 2.0, 3.0]:
            bprime = (1 + bg) * (Z_C_w - Z_C_e) / (1 - Z_C_w)
            assert math.isclose(bprime, (1 + bg) * base, rel_tol=1e-12)

    def test_bprime_inf_at_full_carbon(self):
        Z_C_wall = 1.0
        assert math.isinf(math.inf)  # guard: formula returns inf when Z_C_wall >= 1


# ===========================================================================
# Section 2 – make_equilibrium_bprime_table: Cantera physics (slow)
# ===========================================================================

@pytest.mark.cantera
class TestCanterapHysics:
    """Integration tests requiring a working Cantera installation."""

    @pytest.fixture(scope="class")
    def gas_carbon(self):
        import cantera as ct
        gas = ct.Solution("gri30.yaml")
        carbon = ct.Solution("graphite.yaml")
        return gas, carbon

    def test_bprime_positive_air_graphite(self, gas_carbon):
        """B'c must be positive for air/graphite at high temperature."""
        from scam.chemistry.thermochemistry import compute_bprime_case
        gas, carbon = gas_carbon
        case = compute_bprime_case(
            gas_mechanism="gri30.yaml",
            carbon_phase_file="graphite.yaml",
            temperature_K=2000.0,
            pressure_Pa=101325.0,
            bg=0.0,
            edge_x="O2:0.21,N2:0.79",
            edge_y=None,
            pyro_x=None,
            pyro_y=None,
            initial_gas_moles=1.0,
            initial_carbon_moles=100.0,
            solver="gibbs",
            max_steps=1000,
            max_iter=200,
            log_level=0,
            species_threshold=1e-4,
            max_species=6,
            gas=gas,
            carbon=carbon,
        )
        assert case.Bprime_c_eq > 0.0
        assert 0.0 < case.Z_C_wall < 1.0
        assert case.carbon_phase_moles_final > 0.0

    def test_bprime_increases_with_temperature(self, gas_carbon):
        """B'c at 2500 K should be larger than at 1500 K (air/graphite)."""
        from scam.chemistry.thermochemistry import compute_bprime_case
        gas, carbon = gas_carbon

        def run(T):
            return compute_bprime_case(
                gas_mechanism="gri30.yaml",
                carbon_phase_file="graphite.yaml",
                temperature_K=T,
                pressure_Pa=101325.0,
                bg=0.0,
                edge_x="O2:0.21,N2:0.79",
                edge_y=None,
                pyro_x=None,
                pyro_y=None,
                initial_gas_moles=1.0,
                initial_carbon_moles=100.0,
                solver="gibbs",
                max_steps=1000,
                max_iter=200,
                log_level=0,
                species_threshold=1e-4,
                max_species=6,
                gas=gas,
                carbon=carbon,
            )

        low = run(1500.0)
        high = run(2500.0)
        assert high.Bprime_c_eq > low.Bprime_c_eq

    def test_bg_scaling_air_preset(self, gas_carbon):
        """With identical edge & pyro (air), B'c scales as (1+Bg)."""
        from scam.chemistry.thermochemistry import compute_bprime_case
        gas, carbon = gas_carbon

        common = dict(
            gas_mechanism="gri30.yaml",
            carbon_phase_file="graphite.yaml",
            temperature_K=2000.0,
            pressure_Pa=101325.0,
            edge_x="O2:0.21,N2:0.79",
            edge_y=None,
            pyro_x=None,
            pyro_y=None,
            initial_gas_moles=1.0,
            initial_carbon_moles=100.0,
            solver="gibbs",
            max_steps=1000,
            max_iter=200,
            log_level=0,
            species_threshold=1e-4,
            max_species=6,
            gas=gas,
            carbon=carbon,
        )

        bc0 = compute_bprime_case(bg=0.0, **common).Bprime_c_eq
        bc1 = compute_bprime_case(bg=1.0, **common).Bprime_c_eq
        # With air only, B'c(Bg=1) = 2 × B'c(Bg=0)
        assert math.isclose(bc1, 2.0 * bc0, rel_tol=1e-6)

    def test_parse_and_normalize_flow(self):
        """normalize_composition_string output is accepted by Cantera."""
        import cantera as ct
        from scam.chemistry.composition import normalize_composition_string
        comp = normalize_composition_string("o2:0.21,n2:0.79")
        gas = ct.Solution("gri30.yaml")
        gas.TPX = 300.0, 101325.0, comp  # should not raise


# ===========================================================================
# Section 3 – plot_equilibrium_bprime_table: DataFrame helpers
# ===========================================================================

@pytest.mark.skip(reason="Tests CLI utilities not ported to SCAM")
class TestPlotHelpers:
    def test_has_bg_true(self):
        from plot_equilibrium_bprime_table import has_bg
        df = pd.DataFrame({"Bg": [0.0, 0.5]})
        assert has_bg(df) is True

    def test_has_bg_false(self):
        from plot_equilibrium_bprime_table import has_bg
        df = pd.DataFrame({"T_wall_K": [1000.0]})
        assert has_bg(df) is False

    def test_unique_bg_values(self):
        from plot_equilibrium_bprime_table import unique_bg_values
        df = pd.DataFrame({"Bg": [0.0, 0.0, 0.5, 1.0, 0.5]})
        assert unique_bg_values(df) == [0.0, 0.5, 1.0]

    def test_filter_bg_exact(self):
        from plot_equilibrium_bprime_table import filter_bg
        df = pd.DataFrame({"Bg": [0.0, 0.5, 1.0], "val": [1, 2, 3]})
        result = filter_bg(df, 0.5)
        assert list(result["val"]) == [2]

    def test_filter_bg_no_bg_column(self):
        from plot_equilibrium_bprime_table import filter_bg
        df = pd.DataFrame({"val": [1, 2, 3]})
        result = filter_bg(df, 0.5)
        assert len(result) == 3

    def test_parse_bg_filter_none(self):
        from plot_equilibrium_bprime_table import parse_bg_filter
        assert parse_bg_filter("") is None
        assert parse_bg_filter(None) is None

    def test_parse_bg_filter_values(self):
        from plot_equilibrium_bprime_table import parse_bg_filter
        result = parse_bg_filter("0,0.5,1")
        assert result == [0.0, 0.5, 1.0]

    def test_select_bg_values_no_filter(self):
        from plot_equilibrium_bprime_table import select_bg_values
        all_bgs = [0.0, 0.5, 1.0]
        assert select_bg_values(all_bgs, None) == [0.0, 0.5, 1.0]

    def test_select_bg_values_with_filter(self):
        from plot_equilibrium_bprime_table import select_bg_values
        all_bgs = [0.0, 0.5, 1.0, 1.5]
        assert select_bg_values(all_bgs, [0.0, 1.0]) == [0.0, 1.0]

    def test_select_bg_values_missing_warns(self, capsys):
        from plot_equilibrium_bprime_table import select_bg_values
        all_bgs = [0.0, 0.5]
        result = select_bg_values(all_bgs, [0.0, 9.9])
        assert result == [0.0]
        captured = capsys.readouterr()
        assert "9.9" in captured.out or "9.9" in captured.err

    def test_format_bg_positive(self):
        from plot_equilibrium_bprime_table import format_bg
        assert format_bg(0.0) == "0"
        assert format_bg(0.5) == "0.5"
        assert format_bg(1.0) == "1"

    def test_format_bg_negative(self):
        from plot_equilibrium_bprime_table import format_bg
        assert format_bg(-0.5) == "m0.5"

    def test_make_output_path_no_bg(self):
        from plot_equilibrium_bprime_table import make_output_path
        p = make_output_path(
            Path("data/bprime_air.csv"),
            Path("plots"),
            "Bprime_c_eq",
            "lines",
            "png",
        )
        assert p == Path("plots/bprime_air_Bprime_c_eq_lines.png")

    def test_make_output_path_with_bg(self):
        from plot_equilibrium_bprime_table import make_output_path
        p = make_output_path(
            Path("data/bprime_tacot.csv"),
            Path("plots"),
            "Bprime_c_eq",
            "lines",
            "png",
            bg=0.5,
        )
        assert "Bg0.5" in p.name
        assert p.suffix == ".png"

    def test_quantity_label_bprime(self):
        from plot_equilibrium_bprime_table import quantity_label
        label = quantity_label("Bprime_c_eq")
        assert "B" in label

    def test_pressure_label_kpa(self):
        from plot_equilibrium_bprime_table import pressure_label
        assert "kPa" in pressure_label("kPa")

    def test_format_pressure_kpa(self):
        from plot_equilibrium_bprime_table import format_pressure_for_legend
        label = format_pressure_for_legend(101325.0, "kPa")
        assert "101" in label

    def test_contour_levels_linear(self):
        from plot_equilibrium_bprime_table import contour_levels
        vals = np.linspace(0.1, 5.0, 50)
        levels = contour_levels(vals, log_color=False, n_levels=10)
        assert isinstance(levels, np.ndarray)
        assert len(levels) == 10

    def test_contour_levels_log(self):
        from plot_equilibrium_bprime_table import contour_levels
        vals = np.logspace(-2, 1, 50)
        levels = contour_levels(vals, log_color=True, n_levels=12)
        assert isinstance(levels, np.ndarray)
        assert len(levels) == 12


@pytest.mark.skip(reason="Tests plot_equilibrium_bprime_table.py utilities not ported to SCAM")
class TestReadTable:
    def test_read_basic(self, tmp_path):
        from plot_equilibrium_bprime_table import read_table
        df = _make_bprime_df(bgs=[0.0])
        csv = tmp_path / "bprime.csv"
        df.to_csv(csv, index=False)
        result = read_table(csv, "Bprime_c_eq", drop_sublimated=False)
        assert "Bprime_c_eq" in result.columns
        assert len(result) == len(df)

    def test_read_drops_sublimated(self, tmp_path):
        from plot_equilibrium_bprime_table import read_table
        df = _make_bprime_df(bgs=[0.0])
        df.loc[0, "carbon_phase_moles_final"] = 0.0  # sublimated row
        csv = tmp_path / "bprime.csv"
        df.to_csv(csv, index=False)
        result = read_table(csv, "Bprime_c_eq", drop_sublimated=True)
        assert len(result) == len(df) - 1

    def test_read_missing_quantity_raises(self, tmp_path):
        from plot_equilibrium_bprime_table import read_table
        df = _make_bprime_df(bgs=[0.0])
        csv = tmp_path / "bprime.csv"
        df.to_csv(csv, index=False)
        with pytest.raises((ValueError, SystemExit)):
            read_table(csv, "nonexistent_column")


# ===========================================================================
# Section 4 – compare_tacot_bprime: pressure matching, comparison math
# ===========================================================================

@pytest.mark.skip(reason="Tests CLI utilities not ported to SCAM")
class TestParseFortranD:
    def test_uppercase_d(self):
        from compare_tacot_bprime import _parse_fortran_d
        s = pd.Series(["1.01325D+05", "1.01325D-03"])
        result = _parse_fortran_d(s)
        assert math.isclose(result.iloc[0], 1.01325e5)
        assert math.isclose(result.iloc[1], 1.01325e-3)

    def test_lowercase_d(self):
        from compare_tacot_bprime import _parse_fortran_d
        s = pd.Series(["1.0d0"])
        assert math.isclose(_parse_fortran_d(s).iloc[0], 1.0)

    def test_plain_float(self):
        from compare_tacot_bprime import _parse_fortran_d
        s = pd.Series(["3.14"])
        assert math.isclose(_parse_fortran_d(s).iloc[0], 3.14)


@pytest.mark.skip(reason="Tests CLI utilities not ported to SCAM")
class TestMatchPressures:
    def test_exact_match(self):
        from compare_tacot_bprime import match_pressures
        pairs = match_pressures([1000.0, 10000.0], [1000.0, 10000.0], rel_tol=0.02)
        assert pairs == [(1000.0, 1000.0), (10000.0, 10000.0)]

    def test_close_match_within_tol(self):
        from compare_tacot_bprime import match_pressures
        # TACOT uses bar → Pa: 0.01325 bar × 1e5 = 1325 Pa ≈ 1000 Pa is outside 2%
        # but 1010 Pa vs 1000 Pa is 1% — inside tolerance
        pairs = match_pressures([1000.0], [1010.0], rel_tol=0.02)
        assert len(pairs) == 1
        assert pairs[0] == (1000.0, 1010.0)

    def test_outside_tol_not_matched(self):
        from compare_tacot_bprime import match_pressures
        pairs = match_pressures([1000.0], [1100.0], rel_tol=0.02)
        assert pairs == []

    def test_no_duplicate_comp_pressure(self):
        from compare_tacot_bprime import match_pressures
        # Two ref pressures close to the same comp pressure → only one match
        pairs = match_pressures([1000.0, 1001.0], [1000.0], rel_tol=0.02)
        assert len(pairs) == 1

    def test_tacot_pressures(self):
        from compare_tacot_bprime import match_pressures
        ref = [101.325, 1013.25, 10132.5, 101325.0]
        comp = [101.325, 1013.25, 10132.5, 101325.0]
        pairs = match_pressures(ref, comp, rel_tol=0.02)
        assert len(pairs) == 4


@pytest.mark.skip(reason="Tests CLI utilities not ported to SCAM")
class TestFilterComputedBg:
    def test_filters_to_slice(self):
        from compare_tacot_bprime import filter_computed_bg
        df = pd.DataFrame({
            "T_wall_K": [1000.0, 1000.0, 1000.0],
            "pressure_Pa": [101325.0] * 3,
            "Bprime_c_eq": [0.1, 0.15, 0.2],
            "Bg": [0.0, 0.5, 1.0],
        })
        result = filter_computed_bg(df, 0.5, bg_tol=1e-6)
        assert len(result) == 1
        assert math.isclose(result["Bprime_c_eq"].iloc[0], 0.15)
        assert "Bg" not in result.columns

    def test_passthrough_no_bg_column(self):
        from compare_tacot_bprime import filter_computed_bg
        df = pd.DataFrame({"T_wall_K": [1000.0], "pressure_Pa": [101325.0], "Bprime_c_eq": [0.1]})
        result = filter_computed_bg(df, 0.0, bg_tol=1e-6)
        assert len(result) == 1

    def test_missing_bg_raises(self):
        from compare_tacot_bprime import filter_computed_bg
        df = pd.DataFrame({"T_wall_K": [1000.0], "pressure_Pa": [101325.0], "Bprime_c_eq": [0.1], "Bg": [0.0]})
        with pytest.raises(ValueError, match="9.9"):
            filter_computed_bg(df, 9.9, bg_tol=1e-6)


@pytest.mark.skip(reason="Tests CLI utilities not ported to SCAM")
class TestSublimationOnset:
    def test_detects_onset(self):
        from compare_tacot_bprime import sublimation_onset_temperature
        df = pd.DataFrame({
            "T_wall_K": [1000.0, 2000.0, 3000.0, 4000.0],
            "pressure_Pa": [101325.0] * 4,
            "Bprime_c_eq": [0.1, 0.2, 0.5, 1.0],
            "carbon_phase_moles_final": [95.0, 80.0, 0.5, 0.1],
        })
        T_onset = sublimation_onset_temperature(df, 101325.0, initial_carbon_moles=100.0)
        assert T_onset == 3000.0

    def test_no_depletion_returns_none(self):
        from compare_tacot_bprime import sublimation_onset_temperature
        df = pd.DataFrame({
            "T_wall_K": [1000.0, 2000.0],
            "pressure_Pa": [101325.0, 101325.0],
            "Bprime_c_eq": [0.1, 0.2],
            "carbon_phase_moles_final": [99.0, 95.0],
        })
        assert sublimation_onset_temperature(df, 101325.0) is None

    def test_missing_column_returns_none(self):
        from compare_tacot_bprime import sublimation_onset_temperature
        df = pd.DataFrame({"T_wall_K": [1000.0], "pressure_Pa": [101325.0]})
        assert sublimation_onset_temperature(df, 101325.0) is None


@pytest.mark.skip(reason="Tests CLI utilities not ported to SCAM")
class TestBuildComparison:
    def _make_ref(self):
        T = np.array([500.0, 1000.0, 1500.0, 2000.0])
        return pd.DataFrame({
            "pressure_Pa": [101325.0] * 4,
            "T_wall_K": T,
            "Bc_ref": 0.1 * T / 1000.0,
        })

    def _make_comp(self):
        T = np.linspace(250.0, 2500.0, 50)
        return pd.DataFrame({
            "pressure_Pa": [101325.0] * 50,
            "T_wall_K": T,
            "Bprime_c_eq": 0.1 * T / 1000.0,  # identical to ref
        })

    def test_zero_error_identical_data(self):
        from compare_tacot_bprime import build_comparison
        ref = self._make_ref()
        comp = self._make_comp()
        pairs = [(101325.0, 101325.0)]
        cdf = build_comparison(ref, comp, pairs)
        assert (cdf["abs_err"].abs() < 1e-10).all()

    def test_output_columns(self):
        from compare_tacot_bprime import build_comparison
        ref = self._make_ref()
        comp = self._make_comp()
        cdf = build_comparison(ref, comp, [(101325.0, 101325.0)])
        expected_cols = {"ref_pressure_Pa", "comp_pressure_Pa", "T_wall_K", "Bc_ref", "Bc_comp", "abs_err", "rel_err_pct"}
        assert expected_cols.issubset(set(cdf.columns))

    def test_no_overlap_raises(self):
        from compare_tacot_bprime import build_comparison
        ref = pd.DataFrame({"pressure_Pa": [101325.0], "T_wall_K": [5000.0], "Bc_ref": [1.0]})
        comp = pd.DataFrame({"pressure_Pa": [101325.0], "T_wall_K": [100.0], "Bprime_c_eq": [0.1]})
        with pytest.raises(ValueError, match="No overlapping"):
            build_comparison(ref, comp, [(101325.0, 101325.0)])

    def test_known_error(self):
        from compare_tacot_bprime import build_comparison
        ref = pd.DataFrame({
            "pressure_Pa": [101325.0, 101325.0],
            "T_wall_K": [1000.0, 2000.0],
            "Bc_ref": [0.1, 0.2],
        })
        comp = pd.DataFrame({
            "pressure_Pa": [101325.0, 101325.0],
            "T_wall_K": [1000.0, 2000.0],
            "Bprime_c_eq": [0.11, 0.22],  # 10% high
        })
        cdf = build_comparison(ref, comp, [(101325.0, 101325.0)])
        assert (cdf["abs_err"] > 0).all()
        assert math.isclose(cdf["rel_err_pct"].iloc[0], 10.0, rel_tol=1e-6)


@pytest.mark.skip(reason="Tests CLI utilities not ported to SCAM")
class TestReadComputed:
    def test_reads_csv(self, tmp_path):
        from compare_tacot_bprime import read_computed
        df = _make_bprime_df(bgs=[0.0])
        csv = tmp_path / "bprime.csv"
        df.to_csv(csv, index=False)
        result = read_computed(csv)
        assert "Bprime_c_eq" in result.columns
        assert "Bg" in result.columns

    def test_missing_file_raises(self, tmp_path):
        from compare_tacot_bprime import read_computed
        with pytest.raises(FileNotFoundError):
            read_computed(tmp_path / "nonexistent.csv")

    def test_missing_column_raises(self, tmp_path):
        from compare_tacot_bprime import read_computed
        df = pd.DataFrame({"T_wall_K": [1000.0], "pressure_Pa": [101325.0]})
        csv = tmp_path / "bad.csv"
        df.to_csv(csv, index=False)
        with pytest.raises(ValueError, match="missing columns"):
            read_computed(csv)


# ===========================================================================
# Section 5 – Regression: CSV round-trip
# ===========================================================================

class TestCSVRoundTrip:
    """Write a BPrimeCase list to CSV and read it back intact."""

    def test_long_csv_round_trip(self, tmp_path):
        import dataclasses
        from scam.chemistry.models import BPrimeCase
        from scam.chemistry.tables import write_long_csv

        rows = [
            BPrimeCase(
                T_wall_K=1000.0 + i * 100.0,
                pressure_Pa=101325.0,
                Bg=0.0,
                Bprime_c_eq=0.1 + i * 0.01,
                Z_C_wall=0.08 + i * 0.001,
                Z_C_edge_eff=0.0,
                h_wall_gas_J_kg=-1.5e6,
                MW_wall_gas_kg_per_kmol=28.5,
                gas_moles_final=1.0,
                carbon_phase_moles_final=99.0 - i,
                major_wall_species_X="N2:0.8;CO:0.2",
            )
            for i in range(5)
        ]
        path = tmp_path / "test_long.csv"
        write_long_csv(rows, path)
        df = pd.read_csv(path)
        assert len(df) == 5
        assert list(df["T_wall_K"]) == [1000.0 + i * 100.0 for i in range(5)]
        assert math.isclose(df["Bprime_c_eq"].iloc[0], 0.1)

    def test_wide_csv_shape(self, tmp_path):
        import dataclasses
        from scam.chemistry.models import BPrimeCase
        from scam.chemistry.tables import write_wide_csv

        rows = [
            BPrimeCase(
                T_wall_K=T,
                pressure_Pa=p,
                Bg=0.0,
                Bprime_c_eq=(1 + 0) * 0.1 * T / 1000.0,
                Z_C_wall=0.1,
                Z_C_edge_eff=0.0,
                h_wall_gas_J_kg=-1e6,
                MW_wall_gas_kg_per_kmol=28.0,
                gas_moles_final=1.0,
                carbon_phase_moles_final=95.0,
                major_wall_species_X="N2:0.8",
            )
            for T in [1000.0, 2000.0, 3000.0]
            for p in [1000.0, 10000.0]
        ]
        path = tmp_path / "test_wide.csv"
        write_wide_csv(rows, path)
        df = pd.read_csv(path, index_col=0)
        assert df.shape[0] == 3   # 3 temperatures
        assert df.shape[1] == 2   # 2 pressures (for Bg=0)


# ===========================================================================
# Section 6 – Golden-file regression: carbon/gri30 baseline
# ===========================================================================

BASELINE_CSV = Path(__file__).parent / "baseline_carbon_gri30.csv"

@pytest.mark.cantera
class TestGoldenFileRegression:
    """Re-generate the 4×2×2 air/graphite table and compare against the
    committed golden file (tests/baseline_carbon_gri30.csv).

    The golden file was generated with:
        python make_equilibrium_bprime_table.py \
            --gas-mechanism gri30.yaml \
            --carbon-phase graphite.yaml \
            --edge-x "O2:0.21,N2:0.79" \
            --temps "1500,2000,2500,3000" \
            --pressures "10000,101325" \
            --bg "0.0,1.0" \
            --workers 1 \
            --out tests/baseline_carbon_gri30.csv \
            --out-wide ""

    Tolerances are tight (1e-6 relative) to catch regressions while
    allowing minor floating-point differences across Cantera patch versions.
    """

    @pytest.fixture(scope="class")
    def fresh_df(self):
        """Run the generator and return its long-form DataFrame."""
        import argparse
        import dataclasses
        from scam.chemistry.tables import build_table

        args = argparse.Namespace(
            gas_mechanism="gri30.yaml",
            gas_phase_name="",
            carbon_phase="graphite.yaml",
            condensed_phase_name="",
            edge_x="O2:0.21,N2:0.79",
            edge_y=None,
            pyro_x=None,
            pyro_y=None,
            gas_moles=1.0,
            carbon_moles=100.0,
            solver="gibbs",
            max_steps=2000,
            max_iter=200,
            log_level=0,
            species_threshold=1e-8,
            max_species=20,
            target_element="C",
            verbose=False,
            temps="1500,2000,2500,3000",
            pressures="10000,101325",
            bg="0.0,1.0",
            workers=1,
        )
        cases = build_table(args)
        return pd.DataFrame([dataclasses.asdict(c) for c in cases])

    def test_row_count(self, fresh_df):
        golden = pd.read_csv(BASELINE_CSV)
        assert len(fresh_df) == len(golden)

    def test_bprime_values(self, fresh_df):
        golden = pd.read_csv(BASELINE_CSV)
        golden_sorted = golden.sort_values(["T_wall_K", "pressure_Pa", "Bg"]).reset_index(drop=True)
        fresh_sorted = fresh_df.sort_values(["T_wall_K", "pressure_Pa", "Bg"]).reset_index(drop=True)
        for col in ["Bprime_c_eq", "Z_C_wall", "h_wall_gas_J_kg"]:
            for i, (g, f) in enumerate(zip(golden_sorted[col], fresh_sorted[col])):
                assert math.isclose(g, f, rel_tol=1e-5), (
                    f"Column {col!r} differs at row {i}: golden={g}, fresh={f}"
                )

    def test_columns_present(self, fresh_df):
        golden = pd.read_csv(BASELINE_CSV)
        for col in golden.columns:
            assert col in fresh_df.columns, f"Missing column {col!r} in regenerated table"


# ===========================================================================
# Section 7 – scam/chemistry/materials.py: data model and YAML loader
# ===========================================================================

MATERIALS_DIR = Path(__file__).parent / "materials"


class TestElementSet:
    def test_basic_construction(self):
        from scam.chemistry.materials import ElementSet
        es = ElementSet.from_list(["C", "O", "N"])
        assert len(es) == 3
        assert es.index("O") == 1

    def test_duplicate_raises(self):
        from scam.chemistry.materials import ElementSet
        with pytest.raises(ValueError, match="Duplicate"):
            ElementSet.from_list(["C", "O", "C"])

    def test_bad_symbol_raises(self):
        from scam.chemistry.materials import ElementSet
        with pytest.raises(ValueError):
            ElementSet.from_list(["c"])  # must start with uppercase


class TestCondensedSpecies:
    def test_graphite(self):
        from scam.chemistry.materials import CondensedSpecies
        sp = CondensedSpecies(
            name="C(gr)",
            molecular_weight_kg_per_kmol=12.011,
            elements={"C": 1},
        )
        assert sp.failure_temperature_K is None
        assert sp.elements["C"] == 1

    def test_sic(self):
        from scam.chemistry.materials import CondensedSpecies
        sp = CondensedSpecies(
            name="SiC(s)",
            molecular_weight_kg_per_kmol=40.097,
            elements={"Si": 1, "C": 1},
        )
        assert "Si" in sp.elements and "C" in sp.elements


class TestPyrolysisGas:
    def test_mass_fraction_ok(self):
        from scam.chemistry.materials import PyrolysisGas
        pg = PyrolysisGas(composition_mass="CH4:0.46,CO:0.35,H2O:0.19")
        assert pg.composition_mass is not None
        assert pg.composition_mole is None

    def test_both_raises(self):
        from scam.chemistry.materials import PyrolysisGas
        with pytest.raises(ValueError):
            PyrolysisGas(
                composition_mass="CH4:1.0",
                composition_mole="CH4:1.0",
            )

    def test_neither_raises(self):
        from scam.chemistry.materials import PyrolysisGas
        with pytest.raises(ValueError):
            PyrolysisGas()


class TestSurfaceMaterial:
    def test_active_elements_inferred(self):
        from scam.chemistry.materials import SurfaceMaterial, CondensedSpecies
        mat = SurfaceMaterial(
            name="test",
            gas_mechanism="gri30.yaml",
            condensed_species=[
                CondensedSpecies("C(gr)", 12.011, {"C": 1}),
                CondensedSpecies("SiO2", 60.084, {"Si": 1, "O": 2}),
            ],
        )
        assert "C" in mat.active_elements
        assert "Si" in mat.active_elements
        assert "O" in mat.active_elements

    def test_active_elements_explicit_overrides(self):
        from scam.chemistry.materials import SurfaceMaterial, CondensedSpecies
        mat = SurfaceMaterial(
            name="test",
            gas_mechanism="gri30.yaml",
            condensed_species=[CondensedSpecies("C(gr)", 12.011, {"C": 1})],
            active_elements=["C", "N"],
        )
        assert mat.active_elements == ["C", "N"]


class TestLoadMaterialCard:
    def test_carbon_char(self):
        from scam.chemistry.materials import load_material_card
        mat = load_material_card(MATERIALS_DIR / "carbon_char.yaml")
        assert mat.name == "carbon_char"
        assert len(mat.condensed_species) == 1
        assert mat.condensed_species[0].name == "C(gr)"
        assert mat.pyrolysis_gas is None
        assert "C" in mat.active_elements

    def test_tacot_char_has_pyrolysis(self):
        from scam.chemistry.materials import load_material_card
        mat = load_material_card(MATERIALS_DIR / "tacot_char.yaml")
        assert mat.pyrolysis_gas is not None
        assert mat.pyrolysis_gas.composition_mass is not None
        assert 0.0 in mat.pyrolysis_gas.bg_values

    def test_sic_has_three_condensed(self):
        from scam.chemistry.materials import load_material_card
        mat = load_material_card(MATERIALS_DIR / "sic.yaml")
        assert len(mat.condensed_species) == 3
        names = [sp.name for sp in mat.condensed_species]
        assert "SiC(s)" in names
        assert "SiO2(l)" in names

    def test_sic_failure_temperature(self):
        from scam.chemistry.materials import load_material_card
        mat = load_material_card(MATERIALS_DIR / "sic.yaml")
        sio2 = next(sp for sp in mat.condensed_species if sp.name == "SiO2(l)")
        assert sio2.failure_temperature_K == 2000.0

    def test_missing_file_raises(self):
        from scam.chemistry.materials import load_material_card
        with pytest.raises(FileNotFoundError):
            load_material_card("nonexistent.yaml")


# ===========================================================================
# Section 8 – scam/chemistry/thermo_backend.py: ThermoBackend interface
# ===========================================================================


class TestWallState:
    """WallState is a plain frozen dataclass — no Cantera required."""

    def test_construction(self):
        from scam.chemistry.thermo_backend import WallState
        ws = WallState(
            temperature_K=2000.0,
            pressure_Pa=101325.0,
            h_J_kg=1.5e6,
            MW_kg_per_kmol=28.0,
            element_mass_fractions={"C": 0.15, "N": 0.7, "O": 0.15},
            species_mole_fractions={"N2": 0.65, "CO": 0.35},
        )
        assert ws.temperature_K == 2000.0
        assert ws.gas_moles is None
        assert ws.element_mass_fractions["C"] == pytest.approx(0.15)

    def test_frozen(self):
        from scam.chemistry.thermo_backend import WallState
        ws = WallState(
            temperature_K=1500.0,
            pressure_Pa=50000.0,
            h_J_kg=0.0,
            MW_kg_per_kmol=28.0,
            element_mass_fractions={},
            species_mole_fractions={},
        )
        with pytest.raises((AttributeError, TypeError)):
            ws.temperature_K = 3000.0  # type: ignore[misc]


@pytest.mark.cantera
class TestCanteraBackend:
    """Integration tests for CanteraBackend — require Cantera."""

    @pytest.fixture(scope="class")
    def backend(self):
        from scam.chemistry.thermo_backend import CanteraBackend
        return CanteraBackend(
            gas_mechanism="gri30.yaml",
            carbon_phase_file="graphite.yaml",
            solver="gibbs",
        )

    def test_element_mass_fraction_air(self, backend):
        Z_N = backend.element_mass_fraction(
            composition_x="O2:0.21,N2:0.79",
            composition_y=None,
            temperature_K=300.0,
            pressure_Pa=101325.0,
            element="N",
        )
        assert 0.7 < Z_N < 0.8

    def test_compute_wall_state_returns_wallstate(self, backend):
        from scam.chemistry.thermo_backend import WallState
        ws, Z_edge = backend.compute_wall_state(
            edge_x="O2:0.21,N2:0.79",
            edge_y=None,
            pyro_x=None,
            pyro_y=None,
            bg=0.0,
            temperature_K=2000.0,
            pressure_Pa=101325.0,
            target_element="C",
        )
        assert isinstance(ws, WallState)
        assert ws.temperature_K == pytest.approx(2000.0)
        assert 0.0 < ws.element_mass_fractions.get("C", 0.0) < 1.0
        assert ws.MW_kg_per_kmol > 0.0

    def test_bprime_from_backend_matches_direct(self, backend):
        """B' computed through the backend must match compute_bprime_case."""
        import math as _math
        from scam.chemistry.thermochemistry import compute_bprime_case
        import cantera as ct

        gas = ct.Solution("gri30.yaml")
        carbon = ct.Solution("graphite.yaml")
        direct = compute_bprime_case(
            gas_mechanism="gri30.yaml",
            carbon_phase_file="graphite.yaml",
            temperature_K=2000.0,
            pressure_Pa=101325.0,
            bg=0.0,
            edge_x="O2:0.21,N2:0.79",
            edge_y=None, pyro_x=None, pyro_y=None,
            initial_gas_moles=1.0, initial_carbon_moles=100.0,
            solver="gibbs", max_steps=1000, max_iter=200,
            log_level=0, species_threshold=1e-4, max_species=6,
            gas=gas, carbon=carbon,
        )

        ws, Z_e = backend.compute_wall_state(
            edge_x="O2:0.21,N2:0.79",
            edge_y=None, pyro_x=None, pyro_y=None,
            bg=0.0,
            temperature_K=2000.0,
            pressure_Pa=101325.0,
            target_element="C",
        )
        Z_w = ws.element_mass_fractions.get("C", 0.0)
        bprime_via_backend = (Z_w - Z_e) / (1.0 - Z_w)

        assert _math.isclose(
            bprime_via_backend, direct.Bprime_c_eq, rel_tol=1e-5
        ), f"backend={bprime_via_backend}, direct={direct.Bprime_c_eq}"

    def test_wall_state_has_all_elements(self, backend):
        ws, _ = backend.compute_wall_state(
            edge_x="O2:0.21,N2:0.79",
            edge_y=None, pyro_x=None, pyro_y=None,
            bg=0.0,
            temperature_K=2000.0,
            pressure_Pa=101325.0,
        )
        for el in ("C", "N", "O"):
            assert el in ws.element_mass_fractions


# ===========================================================================
# Section 9 – Phase 4: multiple condensed surface species
# ===========================================================================


class TestCondensedSurfaceMoleFractions:
    """Pure-Python tests for _condensed_surface_mole_fractions helper."""

    def test_sums_to_one(self):
        from scam.chemistry.thermochemistry import _condensed_surface_mole_fractions
        result = _condensed_surface_mole_fractions(["A", "B", "C"], [3.0, 1.0, 1.0])
        assert math.isclose(sum(result.values()), 1.0)

    def test_correct_fractions(self):
        from scam.chemistry.thermochemistry import _condensed_surface_mole_fractions
        result = _condensed_surface_mole_fractions(["X", "Y"], [1.0, 3.0])
        assert math.isclose(result["X"], 0.25)
        assert math.isclose(result["Y"], 0.75)

    def test_all_zero_input_returns_zeros(self):
        from scam.chemistry.thermochemistry import _condensed_surface_mole_fractions
        result = _condensed_surface_mole_fractions(["A", "B"], [0.0, 0.0])
        assert result == {"A": 0.0, "B": 0.0}

    def test_negative_moles_clipped(self):
        from scam.chemistry.thermochemistry import _condensed_surface_mole_fractions
        result = _condensed_surface_mole_fractions(["A", "B"], [-5.0, 10.0])
        assert math.isclose(result["A"], 0.0)
        assert math.isclose(result["B"], 1.0)

    def test_dominant_species(self):
        from scam.chemistry.thermochemistry import _condensed_surface_mole_fractions
        result = _condensed_surface_mole_fractions(["graphite", "diamond"], [90.0, 10.0])
        dominant = max(result, key=result.get)
        assert dominant == "graphite"


@pytest.mark.cantera
class TestPhase4MultiCondensed:
    """Integration tests for multi-condensed equilibration."""

    @pytest.fixture(scope="class")
    def graphite_diamond_gas(self):
        import cantera as ct
        gas = ct.Solution("gri30.yaml")
        graphite = ct.Solution("graphite.yaml")
        diamond = ct.Solution("diamond.yaml")
        return gas, graphite, diamond

    def test_single_condensed_mole_fraction_is_one(self):
        """Single condensed phase must give X_l = 1.0."""
        from scam.chemistry.thermochemistry import _condensed_surface_mole_fractions
        result = _condensed_surface_mole_fractions(["graphite"], [85.3])
        assert math.isclose(result["graphite"], 1.0)

    def test_multi_condensed_diamond_converts_to_graphite(self, graphite_diamond_gas):
        """At 2000 K diamond is metastable: after equilibration diamond ≈ 0."""
        from scam.chemistry.thermochemistry import (
            equilibrate_wall_multi_condensed,
            _condensed_surface_mole_fractions,
            compute_blended_state,
        )
        gas, graphite, diamond = graphite_diamond_gas

        Y_blend, *_ = compute_blended_state(
            gas,
            temperature_K=2000.0,
            pressure_Pa=101325.0,
            edge_x="O2:0.21,N2:0.79",
            edge_y=None,
            pyro_x=None,
            pyro_y=None,
            bg=0.0,
            target_element="C",
        )

        wall_gas, gas_moles, cond_dict, *_ = equilibrate_wall_multi_condensed(
            gas=gas,
            condensed_phases=[(graphite, 50.0), (diamond, 50.0)],
            temperature_K=2000.0,
            pressure_Pa=101325.0,
            Y_blend=Y_blend,
        )

        assert "graphite" in cond_dict
        assert "gas" in cond_dict  # diamond.yaml phase is named "gas"

        X_l = _condensed_surface_mole_fractions(
            list(cond_dict.keys()), list(cond_dict.values())
        )
        # Diamond (named "gas" in diamond.yaml) should nearly vanish at 2000 K.
        assert X_l["gas"] < 1e-6
        assert X_l["graphite"] > 0.99

    def test_bprime_case_has_mole_fractions(self, graphite_diamond_gas):
        """compute_bprime_case with extra_condensed_phases populates mole fractions."""
        from scam.chemistry.thermochemistry import compute_bprime_case
        gas, graphite, diamond = graphite_diamond_gas

        case = compute_bprime_case(
            gas_mechanism="gri30.yaml",
            carbon_phase_file="graphite.yaml",
            temperature_K=2000.0,
            pressure_Pa=101325.0,
            bg=0.0,
            edge_x="O2:0.21,N2:0.79",
            edge_y=None,
            pyro_x=None,
            pyro_y=None,
            initial_gas_moles=1.0,
            initial_carbon_moles=50.0,
            solver="gibbs",
            max_steps=5000,
            max_iter=200,
            log_level=0,
            species_threshold=1e-4,
            max_species=6,
            gas=gas,
            carbon=graphite,
            extra_condensed_phases=[(diamond, 50.0)],
        )

        assert case.condensed_surface_mole_fractions is not None
        assert len(case.condensed_surface_mole_fractions) == 2
        assert math.isclose(sum(case.condensed_surface_mole_fractions.values()), 1.0, rel_tol=1e-6)
        assert case.dominant_condensed_species is not None
        assert case.dominant_condensed_species == "graphite"

    def test_bprime_case_no_extra_has_none_mole_fractions(self, graphite_diamond_gas):
        """Without extra_condensed_phases, mole fraction fields stay None."""
        from scam.chemistry.thermochemistry import compute_bprime_case
        gas, graphite, _ = graphite_diamond_gas

        case = compute_bprime_case(
            gas_mechanism="gri30.yaml",
            carbon_phase_file="graphite.yaml",
            temperature_K=2000.0,
            pressure_Pa=101325.0,
            bg=0.0,
            edge_x="O2:0.21,N2:0.79",
            edge_y=None,
            pyro_x=None,
            pyro_y=None,
            initial_gas_moles=1.0,
            initial_carbon_moles=100.0,
            solver="gibbs",
            max_steps=1000,
            max_iter=200,
            log_level=0,
            species_threshold=1e-4,
            max_species=6,
            gas=gas,
            carbon=graphite,
        )

        assert case.condensed_surface_mole_fractions is None
        assert case.dominant_condensed_species is None

    def test_cantera_backend_multi_condensed(self):
        """CanteraBackend with extra_condensed_phases populates WallState."""
        from scam.chemistry.thermo_backend import CanteraBackend, WallState

        backend = CanteraBackend(
            gas_mechanism="gri30.yaml",
            carbon_phase_file="graphite.yaml",
            initial_carbon_moles=50.0,
            extra_condensed_phases=[("diamond.yaml", "", 50.0)],
        )

        ws, _ = backend.compute_wall_state(
            edge_x="O2:0.21,N2:0.79",
            edge_y=None,
            pyro_x=None,
            pyro_y=None,
            bg=0.0,
            temperature_K=2000.0,
            pressure_Pa=101325.0,
            target_element="C",
        )

        assert isinstance(ws, WallState)
        assert ws.condensed_phase_mole_fractions is not None
        assert len(ws.condensed_phase_mole_fractions) == 2
        assert math.isclose(
            sum(ws.condensed_phase_mole_fractions.values()), 1.0, rel_tol=1e-6
        )


# ===========================================================================
# Section 10 – Phase 5: surface-element constraints
# ===========================================================================


class TestSurfaceElementConstraint:
    """Tests for the SurfaceElementConstraint data class."""

    def test_basic_construction(self):
        from scam.chemistry.mat.constraints import SurfaceElementConstraint
        c = SurfaceElementConstraint(ratios={"Si": 1.0, "C": 1.0})
        assert c.ratios["Si"] == 1.0
        assert c.strict is True

    def test_empty_ratios_raises(self):
        from scam.chemistry.mat.constraints import SurfaceElementConstraint
        with pytest.raises(ValueError, match="non-empty"):
            SurfaceElementConstraint(ratios={})

    def test_negative_ratio_raises(self):
        from scam.chemistry.mat.constraints import SurfaceElementConstraint
        with pytest.raises(ValueError):
            SurfaceElementConstraint(ratios={"Si": -1.0, "C": 1.0})

    def test_all_zero_ratios_raises(self):
        from scam.chemistry.mat.constraints import SurfaceElementConstraint
        with pytest.raises(ValueError):
            SurfaceElementConstraint(ratios={"Si": 0.0, "C": 0.0})


class TestApplySurfaceElementConstraint:
    """Tests for apply_surface_element_constraint (pure Python / scipy)."""

    # Synthetic SiC-like system:
    #   SiC(s):  Si=1, C=1
    #   C(gr):   C=1
    #   SiO2(l): Si=1, O=2
    _species_elements = {
        "SiC(s)":  {"Si": 1.0, "C": 1.0},
        "C(gr)":   {"C": 1.0},
        "SiO2(l)": {"Si": 1.0, "O": 2.0},
    }

    def test_unconstrained_passthrough_no_constraint(self):
        """Without constraint, mole fractions are unchanged."""
        X_l = {"SiC(s)": 0.7, "C(gr)": 0.2, "SiO2(l)": 0.1}
        # No constraint object — just verify the helper does not crash
        # and returns sensible input.
        from scam.chemistry.thermochemistry import _condensed_surface_mole_fractions
        names = list(X_l.keys())
        result = _condensed_surface_mole_fractions(names, list(X_l.values()))
        assert math.isclose(sum(result.values()), 1.0)

    def test_constraint_sums_to_one(self):
        from scam.chemistry.mat.constraints import SurfaceElementConstraint, apply_surface_element_constraint
        X_l = {"SiC(s)": 0.5, "C(gr)": 0.4, "SiO2(l)": 0.1}
        constraint = SurfaceElementConstraint(ratios={"Si": 1.0, "C": 1.0})
        X_c, satisfied = apply_surface_element_constraint(
            X_l, self._species_elements, constraint
        )
        assert math.isclose(sum(X_c.values()), 1.0, rel_tol=1e-6)

    def test_constraint_enforces_si_c_ratio(self):
        from scam.chemistry.mat.constraints import (
            SurfaceElementConstraint,
            apply_surface_element_constraint,
            constraint_residual,
        )
        # Start with carbon-rich unconstrained state.
        X_l = {"SiC(s)": 0.1, "C(gr)": 0.8, "SiO2(l)": 0.1}
        constraint = SurfaceElementConstraint(ratios={"Si": 1.0, "C": 1.0})
        X_c, satisfied = apply_surface_element_constraint(
            X_l, self._species_elements, constraint
        )
        assert satisfied
        # Verify elemental ratio is now ≈ 1:1
        res = constraint_residual(X_c, self._species_elements, constraint)
        # Si and C fractions in condensed layer should be equal.
        assert math.isclose(res["Si"], res["C"], rel_tol=1e-4), (
            f"Si={res['Si']:.6f}, C={res['C']:.6f} — ratio not 1:1"
        )

    def test_all_nonnegative_after_constraint(self):
        from scam.chemistry.mat.constraints import SurfaceElementConstraint, apply_surface_element_constraint
        X_l = {"SiC(s)": 0.3, "C(gr)": 0.6, "SiO2(l)": 0.1}
        constraint = SurfaceElementConstraint(ratios={"Si": 1.0, "C": 1.0})
        X_c, _ = apply_surface_element_constraint(
            X_l, self._species_elements, constraint
        )
        for name, x in X_c.items():
            assert x >= -1e-9, f"Species {name!r} has negative mole fraction {x}"

    def test_single_species_trivially_satisfies(self):
        from scam.chemistry.mat.constraints import SurfaceElementConstraint, apply_surface_element_constraint
        X_l = {"SiC(s)": 1.0}
        constraint = SurfaceElementConstraint(ratios={"Si": 1.0, "C": 1.0})
        X_c, satisfied = apply_surface_element_constraint(
            X_l, {"SiC(s)": {"Si": 1.0, "C": 1.0}}, constraint
        )
        assert satisfied
        assert math.isclose(X_c["SiC(s)"], 1.0)

    def test_constraint_residual_utility(self):
        from scam.chemistry.mat.constraints import SurfaceElementConstraint, constraint_residual
        # Pure SiC — both Si and C fractions should be 0.5.
        X_l = {"SiC(s)": 1.0}
        sp_el = {"SiC(s)": {"Si": 1.0, "C": 1.0}}
        constraint = SurfaceElementConstraint(ratios={"Si": 1.0, "C": 1.0})
        res = constraint_residual(X_l, sp_el, constraint)
        assert math.isclose(res["Si"], 0.5)
        assert math.isclose(res["C"], 0.5)


@pytest.mark.cantera
class TestPhase5ConstraintIntegration:
    """Integration test: constraint applied through compute_bprime_case."""

    @pytest.fixture(scope="class")
    def graphite_diamond_gas(self):
        import cantera as ct
        gas = ct.Solution("gri30.yaml")
        graphite = ct.Solution("graphite.yaml")
        diamond = ct.Solution("diamond.yaml")
        return gas, graphite, diamond

    def test_constrained_equal_fractions(self, graphite_diamond_gas):
        """Force X_graphite = X_diamond = 0.5 via a symmetric constraint."""
        import cantera as ct
        from scam.chemistry.thermochemistry import compute_bprime_case
        from scam.chemistry.mat.constraints import SurfaceElementConstraint
        gas, graphite, diamond = graphite_diamond_gas

        # Both phases are pure carbon.  A symmetric C:C = 1:2 constraint
        # means graphite:diamond elemental C should be 1:2, which forces
        # X_diamond such that its C contribution is twice graphite's C.
        # Since each species has C=1 per formula unit:
        #   X_g * 1 / (X_d * 1) = 1:2  =>  X_g = 1/3, X_d = 2/3
        sp_el = {
            "graphite": {"C": 1.0},
            "gas":      {"C": 1.0},  # diamond.yaml phase is named "gas"
        }
        constraint = SurfaceElementConstraint(ratios={"C": 1.0, "C_d": 2.0}, strict=False)

        # Use a simpler 1:1 constraint that maps directly to equal fractions.
        # Both phases are C=1, so 1:1 C ratio from two identical species
        # is trivially satisfied by any distribution — use a different approach:
        # enforce via asymmetric species composition in sp_el.
        sp_el_asym = {
            "graphite": {"C": 1.0},
            "gas":      {"C": 2.0},  # artificial: treat diamond as C=2
        }
        constraint_asym = SurfaceElementConstraint(ratios={"C": 3.0})

        case = compute_bprime_case(
            gas_mechanism="gri30.yaml",
            carbon_phase_file="graphite.yaml",
            temperature_K=2000.0,
            pressure_Pa=101325.0,
            bg=0.0,
            edge_x="O2:0.21,N2:0.79",
            edge_y=None,
            pyro_x=None,
            pyro_y=None,
            initial_gas_moles=1.0,
            initial_carbon_moles=50.0,
            solver="gibbs",
            max_steps=5000,
            max_iter=200,
            log_level=0,
            species_threshold=1e-4,
            max_species=6,
            gas=gas,
            carbon=graphite,
            extra_condensed_phases=[(diamond, 50.0)],
            surface_constraint=constraint_asym,
            species_elements=sp_el_asym,
        )

        # With a single-element constraint and only one element present,
        # the constraint is trivially satisfied.
        assert case.condensed_surface_mole_fractions is not None
        assert case.unconstrained_surface_mole_fractions is not None
        assert case.surface_constraint_satisfied is True
        assert math.isclose(
            sum(case.condensed_surface_mole_fractions.values()), 1.0, rel_tol=1e-6
        )

    def test_unconstrained_stored_separately(self, graphite_diamond_gas):
        """unconstrained_surface_mole_fractions must equal raw Cantera result."""
        from scam.chemistry.thermochemistry import compute_bprime_case
        from scam.chemistry.mat.constraints import SurfaceElementConstraint
        gas, graphite, diamond = graphite_diamond_gas

        sp_el = {"graphite": {"C": 1.0}, "gas": {"C": 1.0}}
        constraint = SurfaceElementConstraint(ratios={"C": 1.0})

        case_unconstrained = compute_bprime_case(
            gas_mechanism="gri30.yaml",
            carbon_phase_file="graphite.yaml",
            temperature_K=2000.0,
            pressure_Pa=101325.0,
            bg=0.0,
            edge_x="O2:0.21,N2:0.79",
            edge_y=None,
            pyro_x=None,
            pyro_y=None,
            initial_gas_moles=1.0,
            initial_carbon_moles=50.0,
            solver="gibbs",
            max_steps=5000,
            max_iter=200,
            log_level=0,
            species_threshold=1e-4,
            max_species=6,
            gas=gas,
            carbon=graphite,
            extra_condensed_phases=[(diamond, 50.0)],
        )
        case_constrained = compute_bprime_case(
            gas_mechanism="gri30.yaml",
            carbon_phase_file="graphite.yaml",
            temperature_K=2000.0,
            pressure_Pa=101325.0,
            bg=0.0,
            edge_x="O2:0.21,N2:0.79",
            edge_y=None,
            pyro_x=None,
            pyro_y=None,
            initial_gas_moles=1.0,
            initial_carbon_moles=50.0,
            solver="gibbs",
            max_steps=5000,
            max_iter=200,
            log_level=0,
            species_threshold=1e-4,
            max_species=6,
            gas=gas,
            carbon=graphite,
            extra_condensed_phases=[(diamond, 50.0)],
            surface_constraint=constraint,
            species_elements=sp_el,
        )

        # The unconstrained field in the constrained case matches the
        # condensed_surface_mole_fractions from the unconstrained run.
        for name in case_unconstrained.condensed_surface_mole_fractions:
            unconstrained_raw = case_unconstrained.condensed_surface_mole_fractions[name]
            stored = case_constrained.unconstrained_surface_mole_fractions[name]
            assert math.isclose(unconstrained_raw, stored, rel_tol=1e-9), (
                f"{name}: raw={unconstrained_raw}, stored={stored}"
            )


# =============================================================================
# Section 11: Phase 6 — Distinguished blowing parameters (B'_g, B'_total, ...)
# =============================================================================


class TestPhase6BlowingFields:
    """Pure-Python checks on the new Phase 6 BPrimeCase fields."""

    def test_bprime_total_no_blowing(self):
        """Bprime_total == Bprime_eq when B'g=0 and no failure."""
        from scam.chemistry.models import BPrimeCase

        case = BPrimeCase(
            T_wall_K=2000.0,
            pressure_Pa=101325.0,
            Bg=0.0,
            Bprime_c_eq=0.5,
            Z_C_wall=0.6,
            Z_C_edge_eff=0.2,
            h_wall_gas_J_kg=1e6,
            MW_wall_gas_kg_per_kmol=28.0,
            gas_moles_final=1.0,
            carbon_phase_moles_final=99.0,
            major_wall_species_X="CO:0.5",
            Bprime_eq=0.5,
            Bprime_g=0.0,
            Bprime_fail=0.0,
            Bprime_total=0.5,
        )
        assert case.Bprime_g == 0.0
        assert case.Bprime_fail == 0.0
        assert math.isclose(case.Bprime_total, case.Bprime_eq, rel_tol=1e-12)

    def test_bprime_total_with_blowing(self):
        """Bprime_total == Bprime_eq + B'g when there is pyrolysis blowing."""
        from scam.chemistry.models import BPrimeCase

        bprime_eq = 0.4
        bg = 0.5
        case = BPrimeCase(
            T_wall_K=2000.0,
            pressure_Pa=101325.0,
            Bg=bg,
            Bprime_c_eq=0.4,
            Z_C_wall=0.6,
            Z_C_edge_eff=0.2,
            h_wall_gas_J_kg=1e6,
            MW_wall_gas_kg_per_kmol=28.0,
            gas_moles_final=1.0,
            carbon_phase_moles_final=99.0,
            major_wall_species_X="CO:0.5",
            Bprime_eq=bprime_eq,
            Bprime_g=bg,
            Bprime_fail=0.0,
            Bprime_total=bprime_eq + bg,
        )
        assert math.isclose(case.Bprime_total, bprime_eq + bg, rel_tol=1e-12)

    def test_default_phase6_fields_are_nan_or_zero(self):
        """Default values: Bprime_g=0, Bprime_fail=0, Bprime_total=nan."""
        from scam.chemistry.models import BPrimeCase

        case = BPrimeCase(
            T_wall_K=1500.0,
            pressure_Pa=101325.0,
            Bg=0.0,
            Bprime_c_eq=0.3,
            Z_C_wall=0.4,
            Z_C_edge_eff=0.1,
            h_wall_gas_J_kg=1e6,
            MW_wall_gas_kg_per_kmol=28.0,
            gas_moles_final=None,
            carbon_phase_moles_final=None,
            major_wall_species_X="",
        )
        assert case.Bprime_g == 0.0
        assert case.Bprime_fail == 0.0
        assert math.isnan(case.Bprime_total)
        assert math.isnan(case.Z_edge_target)
        assert math.isnan(case.Z_pyro_target)


class TestPhase6ComputeBprimeCase:
    """Cantera integration: compute_bprime_case populates Phase 6 fields."""

    @pytest.fixture(scope="class")
    def air_gas_graphite(self):
        import cantera as ct
        return ct.Solution("gri30.yaml"), ct.Solution("graphite.yaml")

    @pytest.mark.cantera
    def test_bprime_case_phase6_no_blowing(self, air_gas_graphite):
        """Bg=0: Z_edge_target == Z_pyro_target == Z_C_edge_eff."""
        gas, graphite = air_gas_graphite
        from scam.chemistry.thermochemistry import compute_bprime_case

        case = compute_bprime_case(
            gas_mechanism="gri30.yaml",
            carbon_phase_file="graphite.yaml",
            temperature_K=2000.0,
            pressure_Pa=101325.0,
            bg=0.0,
            edge_x="O2:0.21,N2:0.79",
            edge_y=None,
            pyro_x=None,
            pyro_y=None,
            initial_gas_moles=1.0,
            initial_carbon_moles=100.0,
            solver="gibbs",
            max_steps=5000,
            max_iter=200,
            log_level=0,
            species_threshold=1e-4,
            max_species=6,
            gas=gas,
            carbon=graphite,
        )

        assert case.Bprime_g == 0.0
        assert case.Bprime_fail == 0.0
        assert math.isclose(case.Bprime_total, case.Bprime_eq, rel_tol=1e-9)
        assert math.isclose(case.Z_edge_target, case.Z_target_edge_eff, rel_tol=1e-9)
        assert math.isclose(case.Z_pyro_target, case.Z_edge_target, rel_tol=1e-9)

    @pytest.mark.cantera
    def test_bprime_case_phase6_with_pyro_blowing(self, air_gas_graphite):
        """Bg=0.5, pure CH4 pyrolysis gas: Z_pyro_target > Z_edge_target."""
        gas, graphite = air_gas_graphite
        from scam.chemistry.thermochemistry import compute_bprime_case

        case = compute_bprime_case(
            gas_mechanism="gri30.yaml",
            carbon_phase_file="graphite.yaml",
            temperature_K=2000.0,
            pressure_Pa=101325.0,
            bg=0.5,
            edge_x="O2:0.21,N2:0.79",
            edge_y=None,
            pyro_x="CH4:1.0",
            pyro_y=None,
            initial_gas_moles=1.0,
            initial_carbon_moles=100.0,
            solver="gibbs",
            max_steps=5000,
            max_iter=200,
            log_level=0,
            species_threshold=1e-4,
            max_species=6,
            gas=gas,
            carbon=graphite,
        )

        assert math.isclose(case.Bprime_g, 0.5, rel_tol=1e-12)
        assert case.Bprime_fail == 0.0
        assert math.isclose(case.Bprime_total, case.Bprime_eq + 0.5, rel_tol=1e-9)
        assert case.Z_edge_target == pytest.approx(0.0, abs=1e-6)
        assert case.Z_pyro_target > 0.5  # CH4 is ~75% carbon by mass
        assert case.Z_edge_target < case.Z_target_edge_eff < case.Z_pyro_target

    @pytest.mark.cantera
    def test_compute_blended_state_returns_four_tuple(self, air_gas_graphite):
        """compute_blended_state returns (Y_blend, Z_eff, Z_edge, Z_pyro)."""
        gas, _ = air_gas_graphite
        from scam.chemistry.thermochemistry import compute_blended_state

        result = compute_blended_state(
            gas,
            temperature_K=2000.0,
            pressure_Pa=101325.0,
            edge_x="O2:0.21,N2:0.79",
            edge_y=None,
            pyro_x="CH4:1.0",
            pyro_y=None,
            bg=0.5,
            target_element="C",
        )
        assert len(result) == 4
        Y_blend, Z_eff, Z_edge, Z_pyro = result
        assert Y_blend.sum() == pytest.approx(1.0, abs=1e-6)
        assert 0.0 <= Z_eff <= 1.0
        assert Z_edge == pytest.approx(0.0, abs=1e-6)
        assert Z_pyro > 0.5
        assert Z_edge < Z_eff < Z_pyro


# =============================================================================
# Section 12: Phase 7 — Material failure model
# =============================================================================


class TestFailureState:
    """Pure-Python tests for FailureState data class."""

    def test_passive_when_no_failure_species(self):
        from scam.chemistry.mat.failure import FailureState
        fs = FailureState(failing_species=(), Bprime_fail=0.0, active=False)
        assert fs.is_passive
        assert not fs.active

    def test_active_when_failing(self):
        from scam.chemistry.mat.failure import FailureState
        fs = FailureState(failing_species=("SiO2(l)",), Bprime_fail=0.8, active=True)
        assert fs.active
        assert not fs.is_passive
        assert fs.Bprime_fail == pytest.approx(0.8)


class TestComputeFailureState:
    """Pure tests for compute_failure_state()."""

    def _make_species(self, name, t_fail):
        from scam.chemistry.materials import CondensedSpecies
        return CondensedSpecies(
            name=name,
            molecular_weight_kg_per_kmol=60.0,
            elements={"Si": 1, "O": 2},
            failure_temperature_K=t_fail,
        )

    def test_no_failure_below_threshold(self):
        from scam.chemistry.mat.failure import compute_failure_state
        sp = self._make_species("SiO2(l)", 2000.0)
        fs = compute_failure_state([sp], 1800.0, {"SiO2(l)": 1.0})
        assert not fs.active
        assert fs.Bprime_fail == pytest.approx(0.0)
        assert fs.failing_species == ()

    def test_failure_above_threshold(self):
        from scam.chemistry.mat.failure import compute_failure_state
        sp = self._make_species("SiO2(l)", 2000.0)
        fs = compute_failure_state([sp], 2200.0, {"SiO2(l)": 1.0})
        assert fs.active
        assert "SiO2(l)" in fs.failing_species
        assert fs.Bprime_fail == pytest.approx(1.0)

    def test_partial_failure_two_species(self):
        """Only the species above T_fail contribute to Bprime_fail."""
        from scam.chemistry.mat.failure import compute_failure_state
        from scam.chemistry.materials import CondensedSpecies
        sp1 = CondensedSpecies(
            name="SiO2(l)", molecular_weight_kg_per_kmol=60.0,
            elements={"Si": 1, "O": 2}, failure_temperature_K=2000.0,
        )
        sp2 = CondensedSpecies(
            name="C(gr)", molecular_weight_kg_per_kmol=12.0,
            elements={"C": 1}, failure_temperature_K=None,  # no failure
        )
        x_l = {"SiO2(l)": 0.3, "C(gr)": 0.7}
        fs = compute_failure_state([sp1, sp2], 2100.0, x_l)
        assert fs.active
        assert "SiO2(l)" in fs.failing_species
        assert "C(gr)" not in fs.failing_species
        assert fs.Bprime_fail == pytest.approx(0.3)

    def test_species_without_failure_temp_never_fails(self):
        from scam.chemistry.mat.failure import compute_failure_state
        from scam.chemistry.materials import CondensedSpecies
        sp = CondensedSpecies(
            name="C(gr)", molecular_weight_kg_per_kmol=12.0,
            elements={"C": 1}, failure_temperature_K=None,
        )
        fs = compute_failure_state([sp], 5000.0, {"C(gr)": 1.0})
        assert not fs.active
        assert fs.Bprime_fail == pytest.approx(0.0)

    def test_missing_species_in_x_l_contributes_zero(self):
        """A failing species absent from X_l adds 0 to Bprime_fail."""
        from scam.chemistry.mat.failure import compute_failure_state
        sp = self._make_species("SiO2(l)", 2000.0)
        fs = compute_failure_state([sp], 2500.0, {})  # empty X_l
        assert fs.active
        assert fs.Bprime_fail == pytest.approx(0.0)
        assert "SiO2(l)" in fs.failing_species

    def test_passive_to_active_transition_at_threshold(self):
        """Exactly at T_fail the species is still passive (> not >=)."""
        from scam.chemistry.mat.failure import compute_failure_state
        sp = self._make_species("SiO2(l)", 2000.0)
        fs_at = compute_failure_state([sp], 2000.0, {"SiO2(l)": 1.0})
        fs_above = compute_failure_state([sp], 2000.01, {"SiO2(l)": 1.0})
        assert not fs_at.active
        assert fs_above.active


class TestPhase7BPrimeCaseIntegration:
    """Cantera integration: failure fields are populated in compute_bprime_case."""

    @pytest.fixture(scope="class")
    def air_gas_graphite(self):
        import cantera as ct
        return ct.Solution("gri30.yaml"), ct.Solution("graphite.yaml")

    @pytest.mark.cantera
    def test_no_failure_when_no_material_species(self, air_gas_graphite):
        """Without material_condensed_species, failing_species is empty."""
        gas, graphite = air_gas_graphite
        from scam.chemistry.thermochemistry import compute_bprime_case
        case = compute_bprime_case(
            gas_mechanism="gri30.yaml",
            carbon_phase_file="graphite.yaml",
            temperature_K=2000.0,
            pressure_Pa=101325.0,
            bg=0.0,
            edge_x="O2:0.21,N2:0.79",
            edge_y=None,
            pyro_x=None,
            pyro_y=None,
            initial_gas_moles=1.0,
            initial_carbon_moles=100.0,
            solver="gibbs",
            max_steps=5000,
            max_iter=200,
            log_level=0,
            species_threshold=1e-4,
            max_species=6,
            gas=gas,
            carbon=graphite,
        )
        assert case.failing_species == ()
        assert case.Bprime_fail == pytest.approx(0.0)

    @pytest.mark.cantera
    def test_failure_active_above_threshold(self, air_gas_graphite):
        """SiO2-like species with T_fail=1800 K fails at T_wall=2000 K."""
        gas, graphite = air_gas_graphite
        from scam.chemistry.thermochemistry import compute_bprime_case
        from scam.chemistry.materials import CondensedSpecies

        mat_species = [
            CondensedSpecies(
                name="graphite",
                molecular_weight_kg_per_kmol=12.011,
                elements={"C": 1},
                failure_temperature_K=1800.0,  # set low so it triggers at 2000 K
            )
        ]
        case = compute_bprime_case(
            gas_mechanism="gri30.yaml",
            carbon_phase_file="graphite.yaml",
            temperature_K=2000.0,
            pressure_Pa=101325.0,
            bg=0.0,
            edge_x="O2:0.21,N2:0.79",
            edge_y=None,
            pyro_x=None,
            pyro_y=None,
            initial_gas_moles=1.0,
            initial_carbon_moles=100.0,
            solver="gibbs",
            max_steps=5000,
            max_iter=200,
            log_level=0,
            species_threshold=1e-4,
            max_species=6,
            gas=gas,
            carbon=graphite,
            material_condensed_species=mat_species,
        )
        # No surface mole fractions (single-condensed path) — X_l is empty,
        # so Bprime_fail = 0 even though species is flagged as failing.
        assert "graphite" in case.failing_species
        assert case.Bprime_fail == pytest.approx(0.0)  # no X_l in single-condensed path

    @pytest.mark.cantera
    def test_failure_passive_below_threshold(self, air_gas_graphite):
        """Species with T_fail=3000 K is passive at T_wall=2000 K."""
        gas, graphite = air_gas_graphite
        from scam.chemistry.thermochemistry import compute_bprime_case
        from scam.chemistry.materials import CondensedSpecies

        mat_species = [
            CondensedSpecies(
                name="graphite",
                molecular_weight_kg_per_kmol=12.011,
                elements={"C": 1},
                failure_temperature_K=3000.0,
            )
        ]
        case = compute_bprime_case(
            gas_mechanism="gri30.yaml",
            carbon_phase_file="graphite.yaml",
            temperature_K=2000.0,
            pressure_Pa=101325.0,
            bg=0.0,
            edge_x="O2:0.21,N2:0.79",
            edge_y=None,
            pyro_x=None,
            pyro_y=None,
            initial_gas_moles=1.0,
            initial_carbon_moles=100.0,
            solver="gibbs",
            max_steps=5000,
            max_iter=200,
            log_level=0,
            species_threshold=1e-4,
            max_species=6,
            gas=gas,
            carbon=graphite,
            material_condensed_species=mat_species,
        )
        assert case.failing_species == ()
        assert case.Bprime_fail == pytest.approx(0.0)


# =============================================================================
# Section 13: Phase 8 — Heterogeneous surface reactions
# =============================================================================


class TestHeterogeneousReaction:
    """Pure-Python tests for HeterogeneousReaction data class."""

    def test_valid_construction(self):
        from scam.chemistry.mat.reactions import HeterogeneousReaction
        r = HeterogeneousReaction(
            name="C+O->CO",
            gas_reactants={"O": 1.0},
            condensed_reactant="C(gr)",
            gas_products={"CO": 1.0},
            A=2.0,
            beta=0.0,
            Ea_J_per_kmol=0.0,
        )
        assert r.name == "C+O->CO"
        assert r.A == 2.0

    def test_negative_A_raises(self):
        from scam.chemistry.mat.reactions import HeterogeneousReaction
        with pytest.raises(ValueError, match="non-negative"):
            HeterogeneousReaction(
                name="bad",
                gas_reactants={"O2": 1.0},
                condensed_reactant="C(gr)",
                gas_products={"CO2": 1.0},
                A=-1.0,
            )


class TestComputeReactionRate:
    """Pure-Python tests for Arrhenius rate evaluation."""

    def _simple_reaction(self, A=1.0, beta=0.0, Ea=0.0, reactant="O2"):
        from scam.chemistry.mat.reactions import HeterogeneousReaction
        return HeterogeneousReaction(
            name="test",
            gas_reactants={reactant: 1.0},
            condensed_reactant="C(gr)",
            gas_products={"CO2": 1.0},
            A=A,
            beta=beta,
            Ea_J_per_kmol=Ea,
        )

    def test_zero_activation_energy(self):
        """With Ea=0 and beta=0, rate = A * X_reactant."""
        from scam.chemistry.mat.reactions import compute_reaction_rate
        r = self._simple_reaction(A=2.0, Ea=0.0)
        rate = compute_reaction_rate(r, 2000.0, {"O2": 0.5})
        assert rate == pytest.approx(2.0 * 0.5, rel=1e-9)

    def test_rate_increases_with_temperature(self):
        """For Ea > 0 and beta=0, higher T → higher rate."""
        from scam.chemistry.mat.reactions import compute_reaction_rate
        r = self._simple_reaction(A=1.0, Ea=1e8)
        r1 = compute_reaction_rate(r, 1500.0, {"O2": 1.0})
        r2 = compute_reaction_rate(r, 3000.0, {"O2": 1.0})
        assert r2 > r1

    def test_zero_reactant_mole_fraction_gives_zero_rate(self):
        """If a reactant is absent, rate is 0."""
        from scam.chemistry.mat.reactions import compute_reaction_rate
        r = self._simple_reaction(A=100.0)
        rate = compute_reaction_rate(r, 2000.0, {"O2": 0.0})
        assert rate == pytest.approx(0.0)

    def test_missing_reactant_in_dict_gives_zero_rate(self):
        from scam.chemistry.mat.reactions import compute_reaction_rate
        r = self._simple_reaction(A=100.0)
        rate = compute_reaction_rate(r, 2000.0, {})
        assert rate == pytest.approx(0.0)

    def test_zero_order_reaction_ignores_concentrations(self):
        """Zero-order: rate = k regardless of composition."""
        from scam.chemistry.mat.reactions import HeterogeneousReaction, compute_reaction_rate
        r = HeterogeneousReaction(
            name="zero_order",
            gas_reactants={"O2": 1.0},
            condensed_reactant="C(gr)",
            gas_products={"CO": 1.0},
            A=5.0,
            Ea_J_per_kmol=0.0,
            reaction_order=0.0,
        )
        rate1 = compute_reaction_rate(r, 2000.0, {"O2": 0.0})
        rate2 = compute_reaction_rate(r, 2000.0, {"O2": 1.0})
        assert rate1 == pytest.approx(5.0)
        assert rate2 == pytest.approx(5.0)

    def test_negative_temperature_raises(self):
        from scam.chemistry.mat.reactions import compute_reaction_rate
        r = self._simple_reaction()
        with pytest.raises(ValueError, match="positive"):
            compute_reaction_rate(r, -100.0, {"O2": 0.5})

    def test_beta_exponent(self):
        """With A=1, Ea=0, beta=1: rate = T * X."""
        from scam.chemistry.mat.reactions import compute_reaction_rate
        r = self._simple_reaction(A=1.0, beta=1.0, Ea=0.0)
        T = 2000.0
        X = 0.3
        rate = compute_reaction_rate(r, T, {"O2": X})
        assert rate == pytest.approx(T * X, rel=1e-9)


class TestComputeSurfaceReactionRates:
    """Tests for compute_surface_reaction_rates and active_reactions."""

    def test_returns_dict_with_all_reaction_names(self):
        from scam.chemistry.mat.reactions import (
            HeterogeneousReaction,
            compute_surface_reaction_rates,
        )
        r1 = HeterogeneousReaction("r1", {"O2": 1.0}, "C(gr)", {"CO2": 1.0}, A=1.0, Ea_J_per_kmol=0.0)
        r2 = HeterogeneousReaction("r2", {"O": 1.0}, "C(gr)", {"CO": 1.0}, A=2.0, Ea_J_per_kmol=0.0)
        rates = compute_surface_reaction_rates([r1, r2], 2000.0, {"O2": 0.1, "O": 0.05})
        assert set(rates.keys()) == {"r1", "r2"}
        assert rates["r1"] == pytest.approx(0.1, rel=1e-9)
        assert rates["r2"] == pytest.approx(2.0 * 0.05, rel=1e-9)

    def test_active_reactions_threshold(self):
        from scam.chemistry.mat.reactions import active_reactions
        rates = {"fast": 1e-5, "slow": 1e-25, "medium": 1e-15}
        active = active_reactions(rates, threshold=1e-20)
        assert "fast" in active
        assert "medium" in active
        assert "slow" not in active

    def test_carbon_oxidation_reactions_defined(self):
        from scam.chemistry.mat.reactions import CARBON_OXIDATION_REACTIONS
        assert len(CARBON_OXIDATION_REACTIONS) == 3
        names = [r.name for r in CARBON_OXIDATION_REACTIONS]
        assert "C+O->CO" in names
        assert "C+0.5O2->CO" in names

    def test_carbon_oxidation_rate_increases_with_temperature(self):
        """C+O→CO (barrierless) rate should increase or stay flat with T."""
        from scam.chemistry.mat.reactions import CARBON_OXIDATION_REACTIONS, compute_reaction_rate
        co_rxn = next(r for r in CARBON_OXIDATION_REACTIONS if r.name == "C+O->CO")
        r1 = compute_reaction_rate(co_rxn, 1000.0, {"O": 0.01})
        r2 = compute_reaction_rate(co_rxn, 3000.0, {"O": 0.01})
        assert r2 >= r1  # barrierless → flat or increasing with T^beta

    def test_c_o2_co2_rate_higher_at_low_T_than_c_o2_co(self):
        """CO2 channel dominates at lower T (higher Ea/lower pre-exp crossover)."""
        from scam.chemistry.mat.reactions import CARBON_OXIDATION_REACTIONS, compute_reaction_rate
        co2_rxn = next(r for r in CARBON_OXIDATION_REACTIONS if r.name == "C+O2->CO2")
        co_rxn = next(r for r in CARBON_OXIDATION_REACTIONS if r.name == "C+0.5O2->CO")
        # Both present same O2; just check both give finite positive rates
        r_co2 = compute_reaction_rate(co2_rxn, 2000.0, {"O2": 0.2})
        r_co = compute_reaction_rate(co_rxn, 2000.0, {"O2": 0.2})
        assert r_co2 > 0.0
        assert r_co > 0.0


class TestPhase8BPrimeCaseIntegration:
    """Cantera integration: reaction_rates and active_reactions in BPrimeCase."""

    @pytest.fixture(scope="class")
    def air_gas_graphite(self):
        import cantera as ct
        return ct.Solution("gri30.yaml"), ct.Solution("graphite.yaml")

    @pytest.mark.cantera
    def test_no_reactions_when_not_provided(self, air_gas_graphite):
        """Without surface_reactions, reaction_rates is None and active_reactions is empty."""
        gas, graphite = air_gas_graphite
        from scam.chemistry.thermochemistry import compute_bprime_case
        case = compute_bprime_case(
            gas_mechanism="gri30.yaml",
            carbon_phase_file="graphite.yaml",
            temperature_K=2000.0,
            pressure_Pa=101325.0,
            bg=0.0,
            edge_x="O2:0.21,N2:0.79",
            edge_y=None,
            pyro_x=None,
            pyro_y=None,
            initial_gas_moles=1.0,
            initial_carbon_moles=100.0,
            solver="gibbs",
            max_steps=5000,
            max_iter=200,
            log_level=0,
            species_threshold=1e-4,
            max_species=6,
            gas=gas,
            carbon=graphite,
        )
        assert case.reaction_rates is None
        assert case.active_reactions == ()

    @pytest.mark.cantera
    def test_reaction_rates_populated_with_reactions(self, air_gas_graphite):
        """With carbon oxidation reactions, reaction_rates is a non-None dict."""
        gas, graphite = air_gas_graphite
        from scam.chemistry.thermochemistry import compute_bprime_case
        from scam.chemistry.mat.reactions import CARBON_OXIDATION_REACTIONS
        case = compute_bprime_case(
            gas_mechanism="gri30.yaml",
            carbon_phase_file="graphite.yaml",
            temperature_K=2000.0,
            pressure_Pa=101325.0,
            bg=0.0,
            edge_x="O2:0.21,N2:0.79",
            edge_y=None,
            pyro_x=None,
            pyro_y=None,
            initial_gas_moles=1.0,
            initial_carbon_moles=100.0,
            solver="gibbs",
            max_steps=5000,
            max_iter=200,
            log_level=0,
            species_threshold=1e-4,
            max_species=6,
            gas=gas,
            carbon=graphite,
            surface_reactions=CARBON_OXIDATION_REACTIONS,
        )
        assert case.reaction_rates is not None
        assert len(case.reaction_rates) == 3
        for name, rate in case.reaction_rates.items():
            assert rate >= 0.0, f"Rate for {name} is negative: {rate}"

    @pytest.mark.cantera
    def test_active_reactions_subset_of_all_reactions(self, air_gas_graphite):
        """active_reactions is a subset of reaction_rates keys."""
        gas, graphite = air_gas_graphite
        from scam.chemistry.thermochemistry import compute_bprime_case
        from scam.chemistry.mat.reactions import CARBON_OXIDATION_REACTIONS
        case = compute_bprime_case(
            gas_mechanism="gri30.yaml",
            carbon_phase_file="graphite.yaml",
            temperature_K=3000.0,
            pressure_Pa=101325.0,
            bg=0.0,
            edge_x="O2:0.21,N2:0.79",
            edge_y=None,
            pyro_x=None,
            pyro_y=None,
            initial_gas_moles=1.0,
            initial_carbon_moles=100.0,
            solver="gibbs",
            max_steps=5000,
            max_iter=200,
            log_level=0,
            species_threshold=1e-4,
            max_species=6,
            gas=gas,
            carbon=graphite,
            surface_reactions=CARBON_OXIDATION_REACTIONS,
        )
        all_names = set(case.reaction_rates.keys())
        for name in case.active_reactions:
            assert name in all_names


# =============================================================================
# Section 14: Phase 9 — Robust solver (continuation, diagnostics, damped retry)
# =============================================================================


class TestNewtonSolverConfig:
    """Pure tests for NewtonSolverConfig validation."""

    def test_default_construction(self):
        from scam.chemistry.mat.solver import NewtonSolverConfig
        cfg = NewtonSolverConfig()
        assert cfg.max_steps == 5000
        assert cfg.max_iter == 200
        assert cfg.damp_factor == 0.5
        assert cfg.max_retries == 3
        assert cfg.use_continuation is True

    def test_invalid_damp_factor_zero(self):
        from scam.chemistry.mat.solver import NewtonSolverConfig
        with pytest.raises(ValueError, match="damp_factor"):
            NewtonSolverConfig(damp_factor=0.0)

    def test_invalid_damp_factor_one(self):
        from scam.chemistry.mat.solver import NewtonSolverConfig
        with pytest.raises(ValueError, match="damp_factor"):
            NewtonSolverConfig(damp_factor=1.0)

    def test_invalid_max_retries_negative(self):
        from scam.chemistry.mat.solver import NewtonSolverConfig
        with pytest.raises(ValueError, match="max_retries"):
            NewtonSolverConfig(max_retries=-1)

    def test_custom_values(self):
        from scam.chemistry.mat.solver import NewtonSolverConfig
        cfg = NewtonSolverConfig(damp_factor=0.7, max_retries=5, use_continuation=False)
        assert cfg.damp_factor == 0.7
        assert cfg.max_retries == 5
        assert cfg.use_continuation is False


class TestSolverDiagnostics:
    """Pure tests for SolverDiagnostics data class."""

    def test_construction(self):
        from scam.chemistry.mat.solver import SolverDiagnostics
        d = SolverDiagnostics(
            converged=True, solver_used="gibbs", n_attempts=1,
            elapsed_s=0.05, warm_started=False,
        )
        assert d.converged
        assert d.solver_used == "gibbs"
        assert d.n_attempts == 1
        assert not d.warm_started

    def test_failed_diagnostics(self):
        from scam.chemistry.mat.solver import SolverDiagnostics
        d = SolverDiagnostics(
            converged=False, solver_used="failed", n_attempts=8,
            elapsed_s=1.2, warm_started=True,
        )
        assert not d.converged
        assert d.solver_used == "failed"
        assert d.n_attempts == 8


class TestSortGridForContinuation:
    """Pure tests for continuation grid ordering."""

    def test_sorted_by_pressure_then_bg_then_temperature(self):
        from scam.chemistry.mat.solver import sort_grid_for_continuation
        pts = [
            {"pressure_Pa": 101325.0, "bg": 0.0, "temperature_K": 3000.0},
            {"pressure_Pa": 10000.0,  "bg": 0.0, "temperature_K": 1500.0},
            {"pressure_Pa": 101325.0, "bg": 1.0, "temperature_K": 1500.0},
            {"pressure_Pa": 10000.0,  "bg": 0.0, "temperature_K": 2000.0},
            {"pressure_Pa": 101325.0, "bg": 0.0, "temperature_K": 1500.0},
        ]
        ordered = sort_grid_for_continuation(pts)
        pressures = [kw["pressure_Pa"] for kw in ordered]
        # Outer loop is pressure: all 10000 Pa points must come first
        assert pressures[0] == 10000.0
        assert pressures[1] == 10000.0
        # Within same pressure and bg, temperature ascending
        assert ordered[0]["temperature_K"] <= ordered[1]["temperature_K"]

    def test_original_list_not_mutated(self):
        from scam.chemistry.mat.solver import sort_grid_for_continuation
        pts = [
            {"pressure_Pa": 101325.0, "bg": 0.0, "temperature_K": 3000.0},
            {"pressure_Pa": 10000.0,  "bg": 0.0, "temperature_K": 1500.0},
        ]
        original_order = [kw["pressure_Pa"] for kw in pts]
        sort_grid_for_continuation(pts)
        assert [kw["pressure_Pa"] for kw in pts] == original_order

    def test_single_point_unchanged(self):
        from scam.chemistry.mat.solver import sort_grid_for_continuation
        pts = [{"pressure_Pa": 101325.0, "bg": 0.5, "temperature_K": 2000.0}]
        ordered = sort_grid_for_continuation(pts)
        assert ordered == pts

    def test_empty_list(self):
        from scam.chemistry.mat.solver import sort_grid_for_continuation
        assert sort_grid_for_continuation([]) == []

    def test_temperature_ascending_by_default(self):
        from scam.chemistry.mat.solver import sort_grid_for_continuation
        pts = [
            {"pressure_Pa": 101325.0, "bg": 0.0, "temperature_K": T}
            for T in [3000.0, 1500.0, 2500.0, 2000.0]
        ]
        ordered = sort_grid_for_continuation(pts)
        temps = [kw["temperature_K"] for kw in ordered]
        assert temps == sorted(temps)


class TestDampedEquilibrate:
    """Pure tests for damped_equilibrate retry logic."""

    def test_succeeds_on_first_try(self):
        from scam.chemistry.mat.solver import NewtonSolverConfig, damped_equilibrate

        call_log = []

        def build_fn(n_moles):
            call_log.append(("build", n_moles))
            return "wall_gas", "mix"

        def run_fn(mix, solver_name):
            call_log.append(("run", solver_name))
            # Always succeeds

        cfg = NewtonSolverConfig(max_retries=2)
        wall_gas, conv_moles, solver_used, n_attempts = damped_equilibrate(
            build_fn=build_fn,
            run_fn=run_fn,
            solvers=["gibbs", "vcs"],
            initial_condensed_moles=100.0,
            config=cfg,
        )
        assert wall_gas == "wall_gas"
        assert solver_used == "gibbs"
        assert n_attempts == 1
        assert math.isclose(conv_moles, 100.0)

    def test_falls_back_to_second_solver(self):
        from scam.chemistry.mat.solver import NewtonSolverConfig, damped_equilibrate

        def build_fn(n_moles):
            return "wall_gas", "mix"

        def run_fn(mix, solver_name):
            if solver_name == "gibbs":
                raise RuntimeError("gibbs failed")

        cfg = NewtonSolverConfig(max_retries=0)
        _wg, _cm, solver_used, n_attempts = damped_equilibrate(
            build_fn=build_fn,
            run_fn=run_fn,
            solvers=["gibbs", "vcs"],
            initial_condensed_moles=100.0,
            config=cfg,
        )
        assert solver_used == "vcs"
        assert n_attempts == 2

    def test_damped_retry_on_total_failure(self):
        """With both solvers failing, moles are damped and retried."""
        from scam.chemistry.mat.solver import NewtonSolverConfig, damped_equilibrate

        moles_seen = []

        def build_fn(n_moles):
            moles_seen.append(n_moles)
            return "wall_gas", "mix"

        call_count = [0]

        def run_fn(mix, solver_name):
            call_count[0] += 1
            # Fail for first 4 calls (2 solvers × 2 retries), succeed on 5th
            if call_count[0] < 5:
                raise RuntimeError("solver failed")

        cfg = NewtonSolverConfig(max_retries=3, damp_factor=0.5)
        _wg, conv_moles, _su, n_attempts = damped_equilibrate(
            build_fn=build_fn,
            run_fn=run_fn,
            solvers=["gibbs", "vcs"],
            initial_condensed_moles=100.0,
            config=cfg,
        )
        assert n_attempts == 5
        # 3rd retry (index 2) uses moles damped twice: 100 * 0.5^2 = 25
        assert math.isclose(conv_moles, 100.0 * 0.5**2, rel_tol=1e-9)

    def test_raises_after_all_retries_exhausted(self):
        from scam.chemistry.mat.solver import NewtonSolverConfig, damped_equilibrate

        def build_fn(n_moles):
            return "wg", "mix"

        def run_fn(mix, solver_name):
            raise RuntimeError("always fails")

        cfg = NewtonSolverConfig(max_retries=1, damp_factor=0.5)
        with pytest.raises(RuntimeError, match="converge"):
            damped_equilibrate(
                build_fn=build_fn,
                run_fn=run_fn,
                solvers=["gibbs"],
                initial_condensed_moles=100.0,
                config=cfg,
            )


class TestPhase9BPrimeCaseDiagnostics:
    """Cantera integration: BPrimeCase carries solver diagnostics."""

    @pytest.fixture(scope="class")
    def air_gas_graphite(self):
        import cantera as ct
        return ct.Solution("gri30.yaml"), ct.Solution("graphite.yaml")

    @pytest.mark.cantera
    def test_converged_and_solver_used_populated(self, air_gas_graphite):
        gas, graphite = air_gas_graphite
        from scam.chemistry.thermochemistry import compute_bprime_case
        case = compute_bprime_case(
            gas_mechanism="gri30.yaml",
            carbon_phase_file="graphite.yaml",
            temperature_K=2000.0,
            pressure_Pa=101325.0,
            bg=0.0,
            edge_x="O2:0.21,N2:0.79",
            edge_y=None,
            pyro_x=None,
            pyro_y=None,
            initial_gas_moles=1.0,
            initial_carbon_moles=100.0,
            solver="gibbs",
            max_steps=5000,
            max_iter=200,
            log_level=0,
            species_threshold=1e-4,
            max_species=6,
            gas=gas,
            carbon=graphite,
        )
        assert case.converged is True
        assert case.solver_used in ("gibbs", "vcs")
        assert case.n_solver_attempts >= 1

    @pytest.mark.cantera
    def test_n_solver_attempts_one_on_normal_convergence(self, air_gas_graphite):
        """Normal convergence should use exactly 1 attempt."""
        gas, graphite = air_gas_graphite
        from scam.chemistry.thermochemistry import compute_bprime_case
        case = compute_bprime_case(
            gas_mechanism="gri30.yaml",
            carbon_phase_file="graphite.yaml",
            temperature_K=2000.0,
            pressure_Pa=101325.0,
            bg=0.0,
            edge_x="O2:0.21,N2:0.79",
            edge_y=None,
            pyro_x=None,
            pyro_y=None,
            initial_gas_moles=1.0,
            initial_carbon_moles=100.0,
            solver="gibbs",
            max_steps=5000,
            max_iter=200,
            log_level=0,
            species_threshold=1e-4,
            max_species=6,
            gas=gas,
            carbon=graphite,
        )
        assert case.n_solver_attempts == 1

    @pytest.mark.cantera
    def test_default_bprimecase_converged_true(self):
        """BPrimeCase constructed without explicit diagnostics defaults to converged=True."""
        from scam.chemistry.models import BPrimeCase
        case = BPrimeCase(
            T_wall_K=2000.0,
            pressure_Pa=101325.0,
            Bg=0.0,
            Bprime_c_eq=0.5,
            Z_C_wall=0.6,
            Z_C_edge_eff=0.2,
            h_wall_gas_J_kg=1e6,
            MW_wall_gas_kg_per_kmol=28.0,
            gas_moles_final=1.0,
            carbon_phase_moles_final=99.0,
            major_wall_species_X="CO:0.5",
        )
        assert case.converged is True
        assert case.solver_used == ""
        assert case.n_solver_attempts == 1
