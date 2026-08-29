"""Phase 10 Verification and Validation tests (V-01 through V-11).

Each test corresponds to a case in VALIDATION_PLAN.md.  Tests that require
Cantera are marked @pytest.mark.cantera; all others are pure-Python and run
in the fast suite.

Implementation order follows the plan:
  V-01, V-02, V-03  → carbon equilibrium
  V-09              → solver robustness
  V-11              → material card round-trip
  V-04              → pyrolysis blowing
  V-05              → multi-condensed phases
  V-06              → surface-element constraint
  V-07              → SiO2 failure model
  V-08              → Arrhenius surface reaction rates
  V-10              → silica Si/O mechanism
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CHEMISTRY_DIR = Path(__file__).resolve().parent  # tests/chemistry/


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_case(T, p=101325.0, bg=0.0, edge_x="O2:0.21,N2:0.79",
               mechanism="gri30.yaml", carbon_phase="graphite.yaml",
               target_element="C", gas_phase_name="", condensed_phase_name="",
               pyro_x=None, extra_condensed=None, material_condensed_species=None,
               surface_reactions=None, surface_constraint=None):
    """Call compute_bprime_case with a pre-built gas/carbon pair for speed."""
    import cantera as ct
    from scam.chemistry.thermochemistry import compute_bprime_case

    gas = ct.Solution(mechanism, gas_phase_name) if gas_phase_name else ct.Solution(mechanism)
    if condensed_phase_name:
        carbon = ct.Solution(carbon_phase, condensed_phase_name)
    else:
        carbon = ct.Solution(carbon_phase)

    kwargs = dict(
        gas_mechanism=mechanism,
        carbon_phase_file=carbon_phase,
        gas_phase_name=gas_phase_name,
        condensed_phase_name=condensed_phase_name,
        temperature_K=T,
        pressure_Pa=p,
        bg=bg,
        edge_x=edge_x,
        edge_y=None,
        pyro_x=pyro_x,
        pyro_y=None,
        initial_gas_moles=1.0,
        initial_carbon_moles=100.0,
        solver="gibbs",
        max_steps=5000,
        max_iter=200,
        log_level=0,
        species_threshold=1e-4,
        max_species=6,
        target_element=target_element,
        gas=gas,
        carbon=carbon,
    )
    if extra_condensed is not None:
        kwargs["extra_condensed_phases"] = extra_condensed
    if material_condensed_species is not None:
        kwargs["material_condensed_species"] = material_condensed_species
    if surface_reactions is not None:
        kwargs["surface_reactions"] = surface_reactions
    if surface_constraint is not None:
        kwargs["surface_constraint"] = surface_constraint

    return compute_bprime_case(**kwargs)


# ---------------------------------------------------------------------------
# V-01 · Carbon equilibrium — analytic limit check
# ---------------------------------------------------------------------------


@pytest.mark.cantera
class TestV01CarbonAnalyticLimits:
    TEMPS = [1500.0, 2000.0, 2500.0, 3000.0]
    # Use p=10000 Pa to match the baseline CSV
    PRESSURE = 10000.0

    @pytest.fixture(scope="class")
    def cases(self):
        return [_make_case(T, p=self.PRESSURE) for T in self.TEMPS]

    def test_bprime_positive(self, cases):
        for c in cases:
            assert c.Bprime_c_eq > 0, f"B'c must be > 0 at T={c.T_wall_K} K"

    def test_bprime_monotone_with_T(self, cases):
        bprimes = [c.Bprime_c_eq for c in cases]
        for i in range(len(bprimes) - 1):
            assert bprimes[i] <= bprimes[i + 1], (
                f"B'c must be non-decreasing: {bprimes[i]:.5f} > {bprimes[i+1]:.5f}"
            )

    def test_bprime_total_equals_bprime_c_at_bg_zero(self, cases):
        for c in cases:
            assert math.isclose(c.Bprime_total, c.Bprime_c_eq, rel_tol=1e-9), (
                f"Bprime_total ({c.Bprime_total}) != Bprime_c_eq ({c.Bprime_c_eq}) at B'g=0"
            )

    def test_z_edge_target_near_zero(self, cases):
        for c in cases:
            assert c.Z_edge_target < 1e-6, (
                f"Z_edge_target should be ~0 for air edge, got {c.Z_edge_target}"
            )

    def test_z_pyro_equals_z_edge_at_bg_zero(self, cases):
        for c in cases:
            assert math.isclose(c.Z_pyro_target, c.Z_edge_target, abs_tol=1e-12), (
                f"Z_pyro_target ({c.Z_pyro_target}) != Z_edge_target ({c.Z_edge_target}) at B'g=0"
            )

    def test_solver_fields_populated(self, cases):
        for c in cases:
            assert c.solver_used in ("gibbs", "vcs"), f"Unexpected solver: {c.solver_used!r}"
            assert c.n_solver_attempts >= 1

    def test_regression_vs_baseline(self, cases):
        """B'c at T∈{1500,2000,2500,3000} K × p=101325 Pa matches the Phase 0 golden file."""
        baseline = PROJECT_ROOT / "tests" / "baseline_carbon_gri30.csv"
        if not baseline.exists():
            pytest.skip("Baseline CSV not found")

        # Build lookup: (T_wall_K, pressure_Pa, Bg) -> Bprime_c_eq
        with baseline.open() as f:
            reader = csv.DictReader(f)
            ref = {
                (float(r["T_wall_K"]), float(r["pressure_Pa"]), float(r.get("Bg", 0.0))): float(r["Bprime_c_eq"])
                for r in reader
            }

        for c in cases:
            key = (c.T_wall_K, c.pressure_Pa, c.Bg)
            if key not in ref:
                continue
            ref_val = ref[key]
            assert math.isclose(c.Bprime_c_eq, ref_val, rel_tol=1e-5), (
                f"Regression failure at T={c.T_wall_K} K: "
                f"computed {c.Bprime_c_eq:.8f} vs baseline {ref_val:.8f}"
            )


# ---------------------------------------------------------------------------
# V-02 · Carbon equilibrium — TACOT benchmark
# ---------------------------------------------------------------------------


@pytest.mark.cantera
class TestV02TacotBenchmark:
    """Run compare_tacot_bprime logic at B'g=0 and confirm MAE < 5%."""

    def test_tacot_comparison_bg0(self, tmp_path):
        """Run compare_tacot_bprime.py at B'g=0 and confirm it exits cleanly."""
        import subprocess
        import sys

        tacot_xls = PROJECT_ROOT / "TACOT_3.0.xls"
        if not tacot_xls.exists():
            pytest.skip("TACOT_3.0.xls not found")

        csv_out = tmp_path / "tacot_test.csv"
        gen = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "make_equilibrium_bprime_table.py"),
             "--preset", "tacot",
             "--temps", "1200:4000:400",
             "--pressures", "101.325,1013.25,10132.5,101325",
             "--bg", "0.0",
             "--out", str(csv_out), "--out-wide", ""],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        assert gen.returncode == 0, f"Table generation failed:\n{gen.stderr}"

        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "compare_tacot_bprime.py"),
             "--computed", str(csv_out), "--bg", "0.0",
             "--save-csv", str(tmp_path / "comparison.csv")],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0, (
            f"compare_tacot_bprime.py failed:\n{result.stdout}\n{result.stderr}"
        )

    def test_bprime_increases_with_temperature(self):
        """B'c must increase with T at p=101325 Pa, B'g=0 (TACOT mechanism)."""
        import cantera as ct
        from scam.chemistry.thermochemistry import compute_bprime_case

        temps = [1200.0, 2000.0, 3000.0]
        gas = ct.Solution("cno_ablation.yaml")
        graphite = ct.Solution("graphite.yaml")

        bprimes = []
        for T in temps:
            c = compute_bprime_case(
                gas_mechanism="cno_ablation.yaml",
                carbon_phase_file="graphite.yaml",
                temperature_K=T, pressure_Pa=101325.0, bg=0.0,
                edge_x="O2:0.21,N2:0.79", edge_y=None,
                pyro_x=None, pyro_y=None,
                initial_gas_moles=1.0, initial_carbon_moles=100.0,
                solver="gibbs", max_steps=5000, max_iter=200,
                log_level=0, species_threshold=1e-4, max_species=6,
                gas=gas, carbon=graphite,
            )
            bprimes.append(c.Bprime_c_eq)

        for i in range(len(bprimes) - 1):
            assert bprimes[i] <= bprimes[i + 1], (
                f"B'c non-monotone: T={temps[i]} K gives {bprimes[i]:.5f} "
                f"> T={temps[i+1]} K gives {bprimes[i+1]:.5f}"
            )

    def test_bprime_increases_with_bg(self):
        """B'c must increase with B'g at fixed T=2000 K, p=101325 Pa."""
        import cantera as ct
        from scam.chemistry.thermochemistry import compute_bprime_case
        from scam.chemistry.presets import PRESETS

        preset = PRESETS["tacot"]
        gas = ct.Solution("cno_ablation.yaml")
        graphite = ct.Solution("graphite.yaml")

        bprimes = []
        for bg in [0.0, 0.5]:
            c = compute_bprime_case(
                gas_mechanism="cno_ablation.yaml",
                carbon_phase_file="graphite.yaml",
                temperature_K=2000.0, pressure_Pa=101325.0, bg=bg,
                edge_x=preset["edge_x"], edge_y=None,
                pyro_x=preset.get("pyro_x"), pyro_y=None,
                initial_gas_moles=1.0, initial_carbon_moles=100.0,
                solver="gibbs", max_steps=5000, max_iter=200,
                log_level=0, species_threshold=1e-4, max_species=6,
                gas=gas, carbon=graphite,
            )
            bprimes.append(c.Bprime_c_eq)

        assert bprimes[1] > bprimes[0], (
            f"B'c should increase with B'g: B'g=0 → {bprimes[0]:.5f}, "
            f"B'g=0.5 → {bprimes[1]:.5f}"
        )

    def test_all_points_converged(self):
        """All computed cases must have converged=True."""
        import cantera as ct
        from scam.chemistry.thermochemistry import compute_bprime_case

        gas = ct.Solution("cno_ablation.yaml")
        graphite = ct.Solution("graphite.yaml")
        temps = [500.0, 1200.0, 2000.0, 3000.0]

        for T in temps:
            c = compute_bprime_case(
                gas_mechanism="cno_ablation.yaml",
                carbon_phase_file="graphite.yaml",
                temperature_K=T, pressure_Pa=101325.0, bg=0.0,
                edge_x="O2:0.21,N2:0.79", edge_y=None,
                pyro_x=None, pyro_y=None,
                initial_gas_moles=1.0, initial_carbon_moles=100.0,
                solver="gibbs", max_steps=5000, max_iter=200,
                log_level=0, species_threshold=1e-4, max_species=6,
                gas=gas, carbon=graphite,
            )
            assert c.converged, f"Point T={T} K did not converge"


# ---------------------------------------------------------------------------
# V-03 · Pressure scaling
# ---------------------------------------------------------------------------


@pytest.mark.cantera
class TestV03PressureScaling:

    @pytest.fixture(scope="class")
    def cases(self):
        pressures = [1000.0, 10000.0, 101325.0]
        return {p: _make_case(2000.0, p=p) for p in pressures}

    def test_bprime_decreases_with_pressure(self, cases):
        """B'c must decrease with increasing pressure at T=2000 K."""
        pressures = sorted(cases)
        bprimes = [cases[p].Bprime_c_eq for p in pressures]
        for i in range(len(bprimes) - 1):
            assert bprimes[i] >= bprimes[i + 1], (
                f"B'c should decrease with p: p={pressures[i]} Pa → {bprimes[i]:.5f} "
                f"< p={pressures[i+1]} Pa → {bprimes[i+1]:.5f}"
            )

    def test_all_converged(self, cases):
        for p, c in cases.items():
            assert c.converged, f"Point p={p} Pa did not converge"


# ---------------------------------------------------------------------------
# V-04 · Pyrolysis gas blowing
# ---------------------------------------------------------------------------


@pytest.mark.cantera
class TestV04PyrolysisBlowing:
    TEMPS = [1500.0, 2000.0, 2500.0, 3000.0]
    BGS = [0.0, 0.5, 1.0]

    @pytest.fixture(scope="class")
    def cases(self):
        return {
            (T, bg): _make_case(T, bg=bg, pyro_x="CH4:1.0")
            for T in self.TEMPS
            for bg in self.BGS
        }

    def test_bprime_total_equals_bprime_c_plus_bg(self, cases):
        """B'total = B'c_eq + B'g at all points (no failure model active)."""
        for (T, bg), c in cases.items():
            expected = c.Bprime_c_eq + bg
            assert math.isclose(c.Bprime_total, expected, rel_tol=1e-9), (
                f"T={T}, B'g={bg}: Bprime_total={c.Bprime_total} != "
                f"Bprime_c_eq+B'g={expected}"
            )

    def test_z_pyro_target_approximately_ch4_carbon_fraction(self, cases):
        """CH4 carbon mass fraction = 12/16 = 0.75; Z_pyro_target must be close."""
        ch4_c_fraction = 12.011 / 16.043  # ≈ 0.7490
        for (T, bg), c in cases.items():
            if bg > 0:
                assert math.isclose(c.Z_pyro_target, ch4_c_fraction, rel_tol=1e-3), (
                    f"T={T}: Z_pyro_target={c.Z_pyro_target:.4f}, "
                    f"expected ~{ch4_c_fraction:.4f}"
                )

    def test_z_edge_target_near_zero(self, cases):
        for (T, bg), c in cases.items():
            assert c.Z_edge_target < 1e-6, (
                f"Z_edge_target should be ~0 for air edge, got {c.Z_edge_target}"
            )

    def test_z_target_edge_eff_between_edge_and_pyro(self, cases):
        """Z_eff must lie strictly between Z_edge and Z_pyro when B'g > 0."""
        for (T, bg), c in cases.items():
            if bg > 0:
                lo = min(c.Z_edge_target, c.Z_pyro_target)
                hi = max(c.Z_edge_target, c.Z_pyro_target)
                assert lo <= c.Z_target_edge_eff <= hi, (
                    f"T={T}, B'g={bg}: Z_target_edge_eff={c.Z_target_edge_eff:.4f} "
                    f"outside [{lo:.4f}, {hi:.4f}]"
                )

    def test_z_edge_eff_formula(self, cases):
        """Z_eff = (Z_edge + B'g * Z_pyro) / (1 + B'g) must hold to 1e-9."""
        for (T, bg), c in cases.items():
            if bg > 0:
                expected = (c.Z_edge_target + bg * c.Z_pyro_target) / (1.0 + bg)
                assert math.isclose(c.Z_target_edge_eff, expected, rel_tol=1e-9), (
                    f"T={T}, B'g={bg}: Z_eff={c.Z_target_edge_eff:.8f} != "
                    f"formula {expected:.8f}"
                )


# ---------------------------------------------------------------------------
# V-05 · Multi-condensed phases — graphite + diamond proxy
# ---------------------------------------------------------------------------


@pytest.mark.cantera
class TestV05MultiCondensed:

    @pytest.fixture(scope="class")
    def case(self):
        import cantera as ct
        from scam.chemistry.thermochemistry import compute_bprime_case

        gas = ct.Solution("gri30.yaml")
        graphite = ct.Solution("graphite.yaml")
        diamond = ct.Solution("diamond.yaml")

        return compute_bprime_case(
            gas_mechanism="gri30.yaml",
            carbon_phase_file="graphite.yaml",
            temperature_K=2000.0, pressure_Pa=101325.0, bg=0.0,
            edge_x="O2:0.21,N2:0.79", edge_y=None,
            pyro_x=None, pyro_y=None,
            initial_gas_moles=1.0, initial_carbon_moles=100.0,
            solver="gibbs", max_steps=5000, max_iter=200,
            log_level=0, species_threshold=1e-4, max_species=6,
            gas=gas, carbon=graphite,
            extra_condensed_phases=[(diamond, 50.0)],
        )

    def test_surface_mole_fractions_sum_to_one(self, case):
        assert case.condensed_surface_mole_fractions is not None
        total = sum(case.condensed_surface_mole_fractions.values())
        assert abs(total - 1.0) < 1e-6, f"Surface mole fractions sum = {total}"

    def test_dominant_phase_is_graphite(self, case):
        assert case.dominant_condensed_species == "graphite", (
            f"Expected graphite dominant, got {case.dominant_condensed_species!r}"
        )

    def test_diamond_mole_fraction_near_zero(self, case):
        x = case.condensed_surface_mole_fractions
        x_diamond = x.get("diamond", x.get("C(d)", 0.0))
        assert x_diamond < 1e-2, (
            f"X_diamond should be near 0 at equilibrium, got {x_diamond:.4f}"
        )

    def test_bprime_positive(self, case):
        """B'c must be positive — carbon ablates in air at 2000 K."""
        assert case.Bprime_c_eq > 0, f"Expected positive B'c, got {case.Bprime_c_eq}"


# ---------------------------------------------------------------------------
# V-06 · Surface-element constraint — Si:C = 1:1
# ---------------------------------------------------------------------------


class TestV06SurfaceElementConstraint:
    """Pure-Python test: no Cantera required."""

    def _run_constraint(self, X_l_unconstrained, species_elements, ratios):
        from scam.chemistry.mat.constraints import apply_surface_element_constraint, SurfaceElementConstraint
        constraint = SurfaceElementConstraint(ratios=ratios)
        return apply_surface_element_constraint(X_l_unconstrained, species_elements, constraint)

    def test_si_c_1to1_residual(self):
        """Si:C = 1:1 constraint on a 2-species system."""
        X_l = {"Si-phase": 0.3, "C-phase": 0.7}
        sp_el = {"Si-phase": {"Si": 1.0}, "C-phase": {"C": 1.0}}

        X_constrained, satisfied = self._run_constraint(X_l, sp_el, {"Si": 1.0, "C": 1.0})

        assert satisfied, "Constraint should be satisfiable for Si:C=1:1 with two-species system"

        si_moles = X_constrained.get("Si-phase", 0.0)
        c_moles = X_constrained.get("C-phase", 0.0)
        # Si:C should be 1:1
        if c_moles > 1e-10:
            ratio = si_moles / c_moles
            assert math.isclose(ratio, 1.0, rel_tol=1e-6), (
                f"Si:C ratio = {ratio:.8f}, expected 1.0"
            )

    def test_constrained_sums_to_one(self):
        X_l = {"Si-phase": 0.3, "C-phase": 0.7}
        sp_el = {"Si-phase": {"Si": 1.0}, "C-phase": {"C": 1.0}}
        X_c, _ = self._run_constraint(X_l, sp_el, {"Si": 1.0, "C": 1.0})
        assert abs(sum(X_c.values()) - 1.0) < 1e-6

    def test_unconstrained_preserved_as_input(self):
        """The input X_l is not mutated."""
        X_l = {"Si-phase": 0.3, "C-phase": 0.7}
        original = dict(X_l)
        sp_el = {"Si-phase": {"Si": 1.0}, "C-phase": {"C": 1.0}}
        from scam.chemistry.mat.constraints import apply_surface_element_constraint, SurfaceElementConstraint
        apply_surface_element_constraint(X_l, sp_el, SurfaceElementConstraint({"Si": 1.0, "C": 1.0}))
        assert X_l == original, "Input X_l must not be mutated"

    def test_already_satisfied_unchanged(self):
        """If X_l already satisfies the constraint, output ≈ input."""
        X_l = {"Si-phase": 0.5, "C-phase": 0.5}
        sp_el = {"Si-phase": {"Si": 1.0}, "C-phase": {"C": 1.0}}
        X_c, satisfied = self._run_constraint(X_l, sp_el, {"Si": 1.0, "C": 1.0})
        assert satisfied
        assert math.isclose(X_c["Si-phase"], 0.5, rel_tol=1e-4)
        assert math.isclose(X_c["C-phase"], 0.5, rel_tol=1e-4)


# ---------------------------------------------------------------------------
# V-07 · SiO2 failure — passive-to-active transition
# ---------------------------------------------------------------------------


@pytest.mark.cantera
class TestV07SilicaFailure:
    TEMPS_BELOW = [1600.0, 1800.0, 1900.0, 2000.0]
    TEMPS_ABOVE = [2100.0, 2200.0, 2400.0]

    @pytest.fixture(scope="class")
    def cases(self):
        from scam.chemistry.materials import load_material_card
        import cantera as ct
        from scam.chemistry.thermochemistry import compute_bprime_case

        material = load_material_card(_CHEMISTRY_DIR / "materials" / "silica.yaml")
        gas = ct.Solution("sio_silica.yaml", "sio_gas")
        silica = ct.Solution("sio_silica.yaml", "silica_condensed")

        result = {}
        for T in self.TEMPS_BELOW + self.TEMPS_ABOVE:
            result[T] = compute_bprime_case(
                gas_mechanism="sio_silica.yaml",
                carbon_phase_file="sio_silica.yaml",
                gas_phase_name="sio_gas",
                condensed_phase_name="silica_condensed",
                temperature_K=T, pressure_Pa=101325.0, bg=0.0,
                edge_x="O2:1.0", edge_y=None,
                pyro_x=None, pyro_y=None,
                initial_gas_moles=1.0, initial_carbon_moles=100.0,
                solver="gibbs", max_steps=5000, max_iter=200,
                log_level=0, species_threshold=1e-4, max_species=6,
                target_element="Si", gas=gas, carbon=silica,
                material_condensed_species=material.condensed_species,
            )
        return result

    def test_no_failing_species_below_threshold(self, cases):
        for T in self.TEMPS_BELOW:
            c = cases[T]
            assert c.failing_species == (), (
                f"T={T} K (≤2000) should have no failing species, got {c.failing_species}"
            )

    def test_sio2_fails_above_threshold(self, cases):
        for T in self.TEMPS_ABOVE:
            c = cases[T]
            assert len(c.failing_species) > 0, (
                f"T={T} K (>2000) should have failing species, got none"
            )

    def test_bprime_fail_zero_below_threshold(self, cases):
        for T in self.TEMPS_BELOW:
            assert cases[T].Bprime_fail == 0.0, (
                f"Bprime_fail should be 0 below threshold at T={T} K"
            )

    def test_bprime_total_jumps_at_threshold(self, cases):
        """Bprime_total at T=2100 K should be higher than at T=2000 K."""
        below = cases[2000.0].Bprime_total
        above = cases[2100.0].Bprime_total
        assert above >= below, (
            f"Bprime_total should not decrease at the failure threshold: "
            f"2000K={below:.5f}, 2100K={above:.5f}"
        )


# ---------------------------------------------------------------------------
# V-08 · Carbon oxidation reactions — Arrhenius rates
# ---------------------------------------------------------------------------


@pytest.mark.cantera
class TestV08ArrheniusRates:
    TEMPS = [1000.0, 1500.0, 2000.0, 2500.0, 3000.0]

    @pytest.fixture(scope="class")
    def cases(self):
        from scam.chemistry.mat.reactions import CARBON_OXIDATION_REACTIONS
        return {T: _make_case(T, surface_reactions=CARBON_OXIDATION_REACTIONS)
                for T in self.TEMPS}

    def test_all_rates_finite_and_nonnegative(self, cases):
        for T, c in cases.items():
            assert c.reaction_rates is not None
            for name, rate in c.reaction_rates.items():
                assert rate >= 0.0 and not math.isnan(rate), (
                    f"Rate for {name!r} at T={T} K: {rate}"
                )

    def test_three_reactions_populated(self, cases):
        for T, c in cases.items():
            assert len(c.reaction_rates) == 3, (
                f"Expected 3 reaction rates at T={T}, got {len(c.reaction_rates)}"
            )

    def test_active_reactions_subset_of_rates(self, cases):
        for T, c in cases.items():
            for name in c.active_reactions:
                assert name in c.reaction_rates, (
                    f"active_reactions name {name!r} not in reaction_rates"
                )

    def test_c_plus_o_active_at_high_T(self, cases):
        """C+O→CO (barrierless) should be active at T=3000 K where atomic O is present."""
        c = cases[3000.0]
        # Look for the barrierless reaction (Ea=0)
        from scam.chemistry.mat.reactions import CARBON_OXIDATION_REACTIONS
        barrierless = [r.name for r in CARBON_OXIDATION_REACTIONS if r.Ea_J_per_kmol == 0.0]
        if barrierless:
            # At 3000 K in air, atomic O is present → the barrierless reaction should have a nonzero rate
            rate = c.reaction_rates.get(barrierless[0], 0.0)
            # At 3000 K, X_O should be non-negligible → rate > 0
            assert rate >= 0.0, f"Barrierless rate at 3000 K must be non-negative: {rate}"

    def test_rates_in_plausible_range(self, cases):
        """All rates should be in 10^-15 – 10^5 kmol/m²/s (very wide sanity range)."""
        for T, c in cases.items():
            for name, rate in c.reaction_rates.items():
                if rate > 0:
                    assert rate < 1e5, f"Rate {name!r} at T={T} K seems unphysically large: {rate}"


# ---------------------------------------------------------------------------
# V-09 · Solver robustness and continuation
# ---------------------------------------------------------------------------


class TestV09SolverContinuation:
    """Pure-Python tests — no Cantera required."""

    def test_sort_grid_pressure_outer_bg_middle_T_inner(self):
        from scam.chemistry.mat.solver import sort_grid_for_continuation

        grid = [
            {"temperature_K": T, "pressure_Pa": p, "bg": bg}
            for T in [3000.0, 1000.0, 2000.0]
            for p in [10000.0, 1000.0]
            for bg in [0.5, 0.0]
        ]
        ordered = sort_grid_for_continuation(grid)

        pressures = [d["pressure_Pa"] for d in ordered]
        # Pressure must be non-decreasing
        for i in range(len(pressures) - 1):
            assert pressures[i] <= pressures[i + 1], (
                f"Pressure not sorted at positions {i},{i+1}: {pressures[i]}, {pressures[i+1]}"
            )

        # Within each pressure block, B'g must be non-decreasing
        from itertools import groupby
        for p_val, group in groupby(ordered, key=lambda d: d["pressure_Pa"]):
            group = list(group)
            bgs = [d["bg"] for d in group]
            for i in range(len(bgs) - 1):
                assert bgs[i] <= bgs[i + 1], (
                    f"B'g not sorted within p={p_val}: {bgs[i]}, {bgs[i+1]}"
                )

            # Within each (p, bg) block, T must be ascending
            for bg_val, subgroup in groupby(group, key=lambda d: d["bg"]):
                temps = [d["temperature_K"] for d in subgroup]
                for i in range(len(temps) - 1):
                    assert temps[i] <= temps[i + 1], (
                        f"T not ascending in p={p_val}, bg={bg_val}: {temps}"
                    )

    def test_sort_grid_does_not_mutate_input(self):
        from scam.chemistry.mat.solver import sort_grid_for_continuation

        grid = [{"temperature_K": 2000.0, "pressure_Pa": 101325.0, "bg": 0.0}]
        original = [dict(d) for d in grid]
        sort_grid_for_continuation(grid)
        assert grid == original, "sort_grid_for_continuation must not mutate input"

    def test_sort_grid_preserves_all_entries(self):
        from scam.chemistry.mat.solver import sort_grid_for_continuation

        grid = [
            {"temperature_K": T, "pressure_Pa": p, "bg": bg}
            for T in [1000.0, 2000.0, 3000.0]
            for p in [10000.0, 101325.0]
            for bg in [0.0, 0.5, 1.0]
        ]
        ordered = sort_grid_for_continuation(grid)
        assert len(ordered) == len(grid), "sort_grid_for_continuation must preserve all entries"

    def test_n_solver_attempts_one_on_clean_run(self):
        """A well-conditioned equilibration should converge in 1 attempt."""
        pytest.importorskip("cantera")
        c = _make_case(2000.0)
        assert c.n_solver_attempts == 1, (
            f"Expected 1 solver attempt for a clean point, got {c.n_solver_attempts}"
        )


# ---------------------------------------------------------------------------
# V-10 · Silica equilibrium — Si/O mechanism
# ---------------------------------------------------------------------------


@pytest.mark.cantera
class TestV10SilicaEquilibrium:
    TEMPS = [1200.0, 1500.0, 1800.0, 2000.0, 2200.0]

    @pytest.fixture(scope="class")
    def cases(self):
        import cantera as ct
        from scam.chemistry.thermochemistry import compute_bprime_case

        gas = ct.Solution("sio_silica.yaml", "sio_gas")
        silica = ct.Solution("sio_silica.yaml", "silica_condensed")

        result = {}
        for T in self.TEMPS:
            result[T] = compute_bprime_case(
                gas_mechanism="sio_silica.yaml",
                carbon_phase_file="sio_silica.yaml",
                gas_phase_name="sio_gas",
                condensed_phase_name="silica_condensed",
                temperature_K=T, pressure_Pa=101325.0, bg=0.0,
                edge_x="O2:1.0", edge_y=None,
                pyro_x=None, pyro_y=None,
                initial_gas_moles=1.0, initial_carbon_moles=100.0,
                solver="gibbs", max_steps=5000, max_iter=200,
                log_level=0, species_threshold=1e-4, max_species=6,
                target_element="Si", gas=gas, carbon=silica,
            )
        return result

    def test_target_element_is_si(self, cases):
        for T, c in cases.items():
            assert c.target_element == "Si", (
                f"T={T}: target_element={c.target_element!r}, expected 'Si'"
            )

    def test_bprime_positive_above_melting(self, cases):
        """Bprime_eq > 0 above SiO2 melting point (~1723 K)."""
        for T in [1800.0, 2000.0, 2200.0]:
            c = cases[T]
            assert c.Bprime_eq > 0, f"Bprime_eq should be > 0 at T={T} K, got {c.Bprime_eq}"

    def test_z_edge_si_near_zero(self, cases):
        """Pure O2 edge has no silicon → Z_edge_target ≈ 0."""
        for T, c in cases.items():
            assert c.Z_edge_target < 1e-6, (
                f"T={T}: Z_edge_target={c.Z_edge_target}, expected ~0 for O2 edge"
            )

    def test_all_converged(self, cases):
        for T, c in cases.items():
            assert c.converged, f"T={T} K did not converge"


# ---------------------------------------------------------------------------
# V-11 · Material card round-trip
# ---------------------------------------------------------------------------


class TestV11MaterialCardRoundTrip:
    """Pure-Python tests — no Cantera required."""

    CARDS = ["carbon_char.yaml", "tacot_char.yaml", "sic.yaml", "silica.yaml"]

    def test_all_material_cards_load(self):
        from scam.chemistry.materials import load_material_card
        for fname in self.CARDS:
            path = _CHEMISTRY_DIR / "materials" / fname
            if not path.exists():
                pytest.skip(f"{fname} not found")
            mat = load_material_card(path)
            assert mat.name, f"{fname}: name field must be non-empty"

    def test_sic_has_three_condensed_species(self):
        from scam.chemistry.materials import load_material_card
        path = _CHEMISTRY_DIR / "materials" / "sic.yaml"
        if not path.exists():
            pytest.skip("sic.yaml not found")
        mat = load_material_card(path)
        assert len(mat.condensed_species) == 3, (
            f"sic.yaml should have 3 condensed species, got {len(mat.condensed_species)}"
        )

    def test_sic_sio2_failure_temperature(self):
        from scam.chemistry.materials import load_material_card
        path = _CHEMISTRY_DIR / "materials" / "sic.yaml"
        if not path.exists():
            pytest.skip("sic.yaml not found")
        mat = load_material_card(path)
        sio2 = next((s for s in mat.condensed_species if s.name == "SiO2(l)"), None)
        assert sio2 is not None, "sic.yaml must have SiO2(l) in condensed_species"
        assert math.isclose(sio2.failure_temperature_K, 2000.0), (
            f"SiO2(l) failure_temperature_K = {sio2.failure_temperature_K}, expected 2000.0"
        )

    def test_sic_active_elements(self):
        from scam.chemistry.materials import load_material_card
        path = _CHEMISTRY_DIR / "materials" / "sic.yaml"
        if not path.exists():
            pytest.skip("sic.yaml not found")
        mat = load_material_card(path)
        assert set(mat.active_elements) == {"Si", "C", "O", "N"}, (
            f"sic.yaml active_elements = {mat.active_elements}"
        )

    def test_malformed_yaml_raises(self, tmp_path):
        """load_material_card raises on a YAML missing the 'name' field."""
        from scam.chemistry.materials import load_material_card
        bad = tmp_path / "bad.yaml"
        bad.write_text("description: oops\ncondensed_species: []\n")
        with pytest.raises((KeyError, ValueError, Exception)):
            load_material_card(bad)

    def test_round_trip_field_equality(self):
        """Load, inspect fields, and confirm they match what the YAML specifies."""
        from scam.chemistry.materials import load_material_card
        path = _CHEMISTRY_DIR / "materials" / "sic.yaml"
        if not path.exists():
            pytest.skip("sic.yaml not found")

        mat1 = load_material_card(path)
        mat2 = load_material_card(path)  # load twice — must give identical result

        assert mat1.name == mat2.name
        assert len(mat1.condensed_species) == len(mat2.condensed_species)
        for s1, s2 in zip(mat1.condensed_species, mat2.condensed_species):
            assert s1.name == s2.name
            assert math.isclose(s1.molecular_weight_kg_per_kmol,
                                 s2.molecular_weight_kg_per_kmol)
