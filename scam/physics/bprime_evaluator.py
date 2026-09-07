# SPDX-License-Identifier: MIT
"""Live Cantera B' evaluator — drop-in replacement for BPrimeTable.

BprimeEvaluator satisfies the same ``lookup(T_wall, p_e, B_g, Z_C_pyro=None)``
interface as ``BPrimeTable`` but computes B'_c and h_wall on-the-fly via
Cantera equilibrium thermochemistry using ``scam.chemistry``.

Performance note: each call runs a multi-phase Cantera equilibration.  With
the current warm-started TACOT setup the base ablation2 validation path is
about 0.3 ms per lookup (~3–4 s for ~11.6k lookups).

Warm-start: Cantera Solution objects are created once at instantiation and
reused across calls (Cantera's internal state acts as a warm start).

Usage
-----
    from scam.physics.bprime_evaluator import BprimeEvaluator
    ev = BprimeEvaluator.from_config("scam/materials/ablative_organic/tacot_v3.0_bprime_config.yaml")
    B_c, h_wall = ev.lookup(2000.0, 101325.0, 0.3, Z_C_pyro=0.35)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Directory containing bundled mechanism files (scam/mechanisms/)
_MECHANISMS_DIR = Path(__file__).resolve().parent.parent / "mechanisms"

# Nominal TACOT pyrolysis gas composition (mole fractions).
# CH4:0.5551,CO:0.2418,H2O:0.2031 → C:0.206 H:0.679 O:0.115 (PATO tacot26)
_TACOT_PYRO_X_NOMINAL = "CH4:0.5551,CO:0.2418,H2O:0.2031"

# TACOT pyrolysis-gas elemental mass fractions — H and O ratio used when
# element-transport supplies only Z_C_pyro and the mechanism lacks atomic species.
_TACOT_PYRO_ZETA_H = 0.136912
_TACOT_PYRO_ZETA_O = 0.368092

_M = {"C": 12.011, "H": 1.008, "O": 15.999, "N": 14.007}


def _resolve_mechanism(name: str) -> str:
    """Resolve a mechanism file name.

    Priority:
      1. Absolute path that exists.
      2. Name found under scam/mechanisms/.
      3. Return as-is (Cantera handles built-in names like ``graphite.yaml``).
    """
    p = Path(name)
    if p.is_absolute() and p.exists():
        return str(p)
    candidate = _MECHANISMS_DIR / name
    if candidate.exists():
        return str(candidate)
    return name


def _pyro_x_from_z_c(
    Z_C_pyro: float,
    base_pyro_x: str = _TACOT_PYRO_X_NOMINAL,
) -> str:
    """Construct a pyrolysis-gas composition string with target carbon mass fraction.

    When the updated cno_ablation mechanism includes monatomic C, H, O species,
    this function returns ``"C:x,H:y,O:z"`` directly from the element mole
    fractions.  For older mechanisms without monatomic species it falls back to
    the CO/CH4/H2/C2H2 reconstruction.

    The caller (element-transport path) supplies only Z_C_pyro; H and O are
    reconstructed using TACOT's fixed H:O mass-fraction ratio.
    """
    Z_C_pyro = max(0.0, min(float(Z_C_pyro), 0.95))
    non_c = max(1.0 - Z_C_pyro, 0.0)
    ho_sum = _TACOT_PYRO_ZETA_H + _TACOT_PYRO_ZETA_O
    z_h = non_c * _TACOT_PYRO_ZETA_H / ho_sum
    z_o = non_c * _TACOT_PYRO_ZETA_O / ho_sum

    nC = Z_C_pyro / _M["C"]
    nH = z_h / _M["H"]
    nO = z_o / _M["O"]

    # If the mechanism has atomic C, H, O species we can express the composition
    # directly as element mole fractions — Cantera normalises them internally.
    # This is detected lazily: if _gas is available and has species "C", "H", "O".
    # The non-atomic fallback builds a valid CO/CH4/H2/C2H2 mixture instead.
    tot = nC + nH + nO
    if tot <= 0.0:
        return base_pyro_x
    return f"C:{nC/tot:.6f},H:{nH/tot:.6f},O:{nO/tot:.6f}"


def _pyro_x_from_z_c_molecular(
    Z_C_pyro: float,
    base_pyro_x: str = _TACOT_PYRO_X_NOMINAL,
) -> str:
    """Fallback CO/CH4/H2/C2H2 reconstruction for mechanisms without atomic species."""
    Z_C_pyro = max(0.0, min(float(Z_C_pyro), 0.95))
    non_c = max(1.0 - Z_C_pyro, 0.0)
    ho_sum = _TACOT_PYRO_ZETA_H + _TACOT_PYRO_ZETA_O
    z_h = non_c * _TACOT_PYRO_ZETA_H / ho_sum
    z_o = non_c * _TACOT_PYRO_ZETA_O / ho_sum

    n_c = Z_C_pyro / _M["C"]
    n_h = z_h / _M["H"]
    n_o = z_o / _M["O"]

    species: dict[str, float] = {}

    if n_o >= n_c:
        n_co = n_c
        n_o_rem = n_o - n_co
        n_h2o = min(n_o_rem, n_h / 2.0)
        n_h_rem = n_h - 2.0 * n_h2o
        if n_co > 0:
            species["CO"] = n_co
        if n_h2o > 0:
            species["H2O"] = n_h2o
        if n_h_rem > 0:
            species["H2"] = n_h_rem / 2.0
    else:
        n_co = n_o
        c_rem = n_c - n_co
        if n_co > 0:
            species["CO"] = n_co
        if n_h >= 4.0 * c_rem:
            n_ch4 = c_rem
            n_h2 = (n_h - 4.0 * n_ch4) / 2.0
            if n_ch4 > 0:
                species["CH4"] = n_ch4
            if n_h2 > 0:
                species["H2"] = n_h2
        elif n_h >= c_rem:
            n_ch4 = max((n_h - c_rem) / 3.0, 0.0)
            n_c2h2 = max((c_rem - n_ch4) / 2.0, 0.0)
            if n_ch4 > 0:
                species["CH4"] = n_ch4
            if n_c2h2 > 0:
                species["C2H2"] = n_c2h2
        else:
            n_c2h2 = n_h / 2.0
            c_after = max(c_rem - 2.0 * n_c2h2, 0.0)
            if n_c2h2 > 0:
                species["C2H2"] = n_c2h2
            if c_after > 0:
                species["C2"] = c_after / 2.0

    total = sum(species.values())
    if total <= 0.0:
        return base_pyro_x
    return ",".join(
        f"{name}:{moles / total:.6f}"
        for name, moles in species.items()
        if moles / total > 1e-10
    )


class BprimeEvaluator:
    """Live B' evaluator using Cantera equilibrium chemistry (scam.chemistry).

    Implements the same ``lookup()`` protocol as BPrimeTable so it can be
    passed as a drop-in wherever a BPrimeTable is expected.

    Mechanism files are resolved from:
      1. Absolute paths (if they exist on disk).
      2. ``scam/mechanisms/`` (bundled mechanism files).
      3. Cantera built-in data path (e.g. ``graphite.yaml``).
    """

    def __init__(
        self,
        gas_mechanism: str,
        carbon_phase_file: str,
        edge_x: str = "O2:0.21,N2:0.79",
        pyro_x: str = _TACOT_PYRO_X_NOMINAL,
        solver: str = "gibbs",
        max_steps: int = 2000,
        max_iter: int = 200,
        log_level: int = 0,
        species_threshold: float = 1e-8,
        max_species: int = 20,
        gas_phase_name: str | None = None,
        condensed_phase_name: str | None = None,
        initial_gas_moles: float = 1.0,
        initial_carbon_moles: float = 100.0,
        target_element: str = "C",
        surface_source_target_fraction: float = 1.0,
    ) -> None:
        from scam.chemistry.thermo_backend import CanteraBackend

        self._edge_x = edge_x
        self._pyro_x_nominal = pyro_x
        self.target_element = target_element.strip() or "C"
        self.surface_source_target_fraction = float(surface_source_target_fraction)
        self._species_threshold = species_threshold
        self._max_species = max_species
        self._pyro_target_fraction_cache: dict[str, float] = {}
        self._pyro_x_cache: dict[float, str] = {}

        self._backend = CanteraBackend(
            gas_mechanism=_resolve_mechanism(gas_mechanism),
            carbon_phase_file=_resolve_mechanism(carbon_phase_file),
            gas_phase_name=gas_phase_name or "",
            condensed_phase_name=condensed_phase_name or "",
            initial_gas_moles=initial_gas_moles,
            initial_carbon_moles=initial_carbon_moles,
            solver=solver,
            max_steps=max_steps,
            max_iter=max_iter,
            log_level=log_level,
        )
        # Keep direct references for surface_enthalpies and pyrolysis_target_fraction
        self._gas = self._backend._gas
        self._carbon = self._backend._carbon
        self._bprime_ran = False  # set True after first successful lookup

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
    ) -> "BprimeEvaluator":
        """Instantiate from a bprime config YAML (same file used by generate_bprime.py)."""
        import yaml

        with open(config_path, encoding="utf-8") as f:
            cfg: dict[str, Any] = yaml.safe_load(f)

        gen = cfg.get("generator", cfg)

        gas_mech = gen["gas_mechanism"]
        carbon_phase = gen["carbon_phase"]

        preset = gen.get("preset", "")
        if preset == "tacot":
            edge_x = "O2:0.21,N2:0.79"
            pyro_x = _TACOT_PYRO_X_NOMINAL
        else:
            edge_x = gen.get("edge_x") or gen.get("edge_y") or "O2:0.21,N2:0.79"
            pyro_x = gen.get("pyro_x") or gen.get("pyro_y") or _TACOT_PYRO_X_NOMINAL

        return cls(
            gas_mechanism=gas_mech,
            carbon_phase_file=carbon_phase,
            edge_x=edge_x,
            pyro_x=pyro_x,
            solver=gen.get("solver", "gibbs"),
            initial_gas_moles=1.0,
            initial_carbon_moles=float(gen.get("carbon_moles", 100.0)),
            target_element=gen.get("target_element", "C"),
            surface_source_target_fraction=float(
                gen.get("surface_source_target_fraction", 1.0)
            ),
        )

    def pyrolysis_target_fraction(self, target_element: str = "C") -> float:
        """Return nominal pyrolysis-gas elemental mass fraction for *target_element*."""
        element = target_element.strip()
        if not element:
            raise ValueError("target_element cannot be empty")
        if element not in self._pyro_target_fraction_cache:
            self._gas.TPX = 300.0, 101325.0, self._pyro_x_nominal
            self._pyro_target_fraction_cache[element] = float(
                self._gas.elemental_mass_fraction(element)
            )
        return self._pyro_target_fraction_cache[element]

    def _build_pyro_x(self, Z_C_pyro: float | None) -> str:
        """Select composition string for Z_C_pyro, trying atomic species first."""
        if Z_C_pyro is None or self.target_element != "C":
            return self._pyro_x_nominal
        cache_key = float(Z_C_pyro)
        cached = self._pyro_x_cache.get(cache_key)
        if cached is not None:
            return cached
        # Try atomic species (C, H, O) if mechanism supports them
        candidate = _pyro_x_from_z_c(Z_C_pyro, self._pyro_x_nominal)
        # Validate: attempt to parse with Cantera; fall back on error
        try:
            self._gas.TPX = 300.0, 101325.0, candidate
            result = candidate
        except Exception:
            result = _pyro_x_from_z_c_molecular(Z_C_pyro, self._pyro_x_nominal)
        self._pyro_x_cache[cache_key] = result
        return result

    def lookup(
        self,
        T_wall: float,
        p_e: float,
        B_g: float,
        Z_C_pyro: float | None = None,
    ) -> tuple[float, float]:
        """Compute B'_c and h_wall via Cantera equilibrium chemistry.

        Returns
        -------
        (B_c_prime, h_wall) — same convention as BPrimeTable.lookup()
        """
        from scam.chemistry.thermochemistry import compute_bprime_case

        pyro_x = self._build_pyro_x(Z_C_pyro)

        result = compute_bprime_case(
            gas_mechanism=self._backend._gas_mechanism,
            carbon_phase_file=self._backend._carbon_phase_file,
            temperature_K=T_wall,
            pressure_Pa=p_e,
            bg=B_g,
            edge_x=self._edge_x,
            edge_y=None,
            pyro_x=pyro_x,
            pyro_y=None,
            initial_gas_moles=self._backend._initial_gas_moles,
            initial_carbon_moles=self._backend._initial_carbon_moles,
            solver=self._backend._solver,
            max_steps=self._backend._max_steps,
            max_iter=self._backend._max_iter,
            log_level=self._backend._log_level,
            species_threshold=self._species_threshold,
            max_species=self._max_species,
            target_element=self.target_element,
            surface_source_target_fraction=self.surface_source_target_fraction,
            gas_phase_name=self._backend._gas_phase_name,
            condensed_phase_name=self._backend._condensed_phase_name,
            gas=self._gas,
            carbon=self._carbon,
        )

        if not result.converged:
            raise RuntimeError(
                f"BprimeEvaluator: did not converge at "
                f"T={T_wall:.1f} K, p={p_e:.0f} Pa, B'g={B_g:.3f}"
            )

        self._bprime_ran = True
        return float(result.Bprime_eq), float(result.h_wall_gas_J_kg)

    def surface_enthalpies(
        self,
        T_wall: float,
        p_e: float,
        Z_C_pyro: float | None = None,
    ) -> tuple[float, float]:
        """Pyrolysis-gas and char enthalpies at the wall [J/kg], Cantera-consistent.

        Returns ``(h_g, h_c)`` evaluated at ``(T_wall, p_e)`` on the SAME Cantera
        element reference as ``h_wall`` from :meth:`lookup`.  Used by the SEB
        advective terms ``qAdvPyro = mDotGw·(h_g − h_w)`` and
        ``qAdvChar = mDotCw·(h_c − h_w)`` (PATO's ``Bprime`` BC).

        h_g is the UNREACTED pyrolysis-gas enthalpy (Cantera NASA-9 reference).
        seb_residual overrides it with mat_surface.h_g_table when the material
        provides one (TACOT, PICA, …) — that table carries the PATO-consistent
        equilibrium enthalpy (including carbon condensation) and is on the same
        Cantera absolute reference as h_wall.  For materials without h_g_table
        this fallback value is used directly.

        If Z_C_pyro is provided (element transport), the pyrolysis composition is
        reconstructed from the carbon mass fraction first.
        """
        if Z_C_pyro is not None and self.target_element == "C":
            pyro_x = _pyro_x_from_z_c_molecular(Z_C_pyro, self._pyro_x_nominal)
        else:
            pyro_x = self._pyro_x_nominal
        T_safe = max(float(T_wall), 200.0)
        h_g = self._backend.gas_enthalpy(T_safe, float(p_e), pyro_x)
        h_c = (
            h_g
            if self.target_element != "C"
            else self._backend.condensed_enthalpy(T_safe, float(p_e))
        )
        return h_g, h_c
