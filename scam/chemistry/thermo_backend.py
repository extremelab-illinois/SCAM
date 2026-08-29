# SPDX-License-Identifier: MIT
"""ThermochemBackend — unified interface for thermodynamics, transport, and B' chemistry.

Defines the abstract base class ``ThermochemBackend`` and the concrete
``CanteraBackend`` implementation.  Future backends (Mutation++, native
NASA-9 evaluator, etc.) should subclass ``ThermochemBackend`` and implement
the same interface without exposing backend-specific objects to callers.

Capability groups
-----------------
1. **Equilibrium B' chemistry** — blend edge + pyrolysis streams, equilibrate
   against a condensed phase, return ``WallState``.  Used by
   ``compute_bprime_case`` and the surface solver.
2. **Gas thermodynamic properties** — mixture enthalpy and cp at a given (T, p,
   composition) without or with equilibration.  Used by ``equilibrium_hg_table``
   and the in-depth gas-phase energy terms.
3. **Gas transport properties** — viscosity, thermal conductivity, and
   diffusion coefficients.  Used by the pressure-driven Darcy flow module and
   future element-diffusion physics.

All composition inputs use Cantera-style strings (``"O2:0.21,N2:0.79"``
for mole fractions).  All quantities are SI.

``ThermoBackend`` is kept as a backward-compatible alias for ``ThermochemBackend``.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Result type returned by equilibrate()
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WallState:
    """Equilibrated wall gas state returned by a ThermochemBackend.

    Attributes
    ----------
    temperature_K:
        Wall temperature (equals the input, unchanged by TP equilibration).
    pressure_Pa:
        Pressure (unchanged).
    h_J_kg:
        Specific enthalpy of the wall gas mixture (J/kg).
    MW_kg_per_kmol:
        Mean molecular weight of the wall gas (kg/kmol).
    element_mass_fractions:
        Dict mapping element symbol → elemental mass fraction in the
        equilibrated wall gas.
    species_mole_fractions:
        Dict mapping species name → mole fraction (all species present in
        the mechanism, including those at zero).
    gas_moles:
        Total moles of gas phase after equilibration.  ``None`` if the
        backend does not track phase moles.
    condensed_moles:
        Total moles of the condensed phase after equilibration.
        ``None`` if the backend does not track phase moles.
    condensed_phase_mole_fractions:
        Per-phase mole fractions for multi-condensed runs.  ``None`` for
        single-condensed cases.
    """

    temperature_K: float
    pressure_Pa: float
    h_J_kg: float
    MW_kg_per_kmol: float
    element_mass_fractions: dict[str, float]
    species_mole_fractions: dict[str, float]
    gas_moles: float | None = None
    condensed_moles: float | None = None
    condensed_phase_mole_fractions: dict[str, float] | None = None


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class ThermochemBackend(abc.ABC):
    """Abstract interface for thermochemistry + transport backends.

    Backends are stateful (they cache Solution objects for warm-start), but
    each public method must be self-contained so that callers do not need to
    manage ordering.
    """

    # ------------------------------------------------------------------
    # Group 1 — Equilibrium B' chemistry (required)
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def element_mass_fraction(
        self,
        composition_x: str | None,
        composition_y: str | None,
        temperature_K: float,
        pressure_Pa: float,
        element: str,
    ) -> float:
        """Return the elemental mass fraction of *element* in a gas mixture.

        Exactly one of *composition_x* (mole fractions) or *composition_y*
        (mass fractions) must be provided.
        """

    @abc.abstractmethod
    def blend_streams(
        self,
        edge_x: str | None,
        edge_y: str | None,
        pyro_x: str | None,
        pyro_y: str | None,
        bg: float,
        temperature_K: float,
        pressure_Pa: float,
        target_element: str,
    ) -> tuple[object, float]:
        """Blend edge and pyrolysis streams by mass at ratio (1 : B'g).

        Returns
        -------
        blended_composition : backend-specific
        Z_target_edge_eff : float
        """

    @abc.abstractmethod
    def equilibrate(
        self,
        blended_composition: object,
        temperature_K: float,
        pressure_Pa: float,
    ) -> WallState:
        """Equilibrate the blended gas against the condensed phase at fixed T, P."""

    def compute_wall_state(
        self,
        *,
        edge_x: str | None,
        edge_y: str | None,
        pyro_x: str | None,
        pyro_y: str | None,
        bg: float,
        temperature_K: float,
        pressure_Pa: float,
        target_element: str = "C",
    ) -> tuple[WallState, float]:
        """Convenience wrapper: blend streams then equilibrate."""
        blended, Z_edge_eff = self.blend_streams(
            edge_x=edge_x,
            edge_y=edge_y,
            pyro_x=pyro_x,
            pyro_y=pyro_y,
            bg=bg,
            temperature_K=temperature_K,
            pressure_Pa=pressure_Pa,
            target_element=target_element,
        )
        wall_state = self.equilibrate(blended, temperature_K, pressure_Pa)
        return wall_state, Z_edge_eff

    # ------------------------------------------------------------------
    # Group 2 — Gas thermodynamic properties (optional, raise if unsupported)
    # ------------------------------------------------------------------

    def gas_enthalpy(self, T: float, p: float, composition_x: str) -> float:
        """Mixture enthalpy h [J/kg] at (T, p) without equilibration."""
        raise NotImplementedError(f"{type(self).__name__} does not implement gas_enthalpy")

    def gas_enthalpy_equilibrium(self, T: float, p: float, composition_x: str) -> float:
        """Mixture enthalpy h [J/kg] after equilibrating the gas at (T, p).

        Used for h_g(T) table generation.  The condensed phase is NOT included;
        this equilibrates only the gas phase (element-conserving speciation).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement gas_enthalpy_equilibrium"
        )

    def gas_cp(self, T: float, p: float, composition_x: str) -> float:
        """Mixture cp [J/kg/K] at (T, p) without equilibration."""
        raise NotImplementedError(f"{type(self).__name__} does not implement gas_cp")

    # ------------------------------------------------------------------
    # Group 3 — Gas transport properties (optional, raise if unsupported)
    # ------------------------------------------------------------------

    def viscosity(self, T: float, p: float, composition_x: str) -> float:
        """Dynamic viscosity µ [Pa·s]."""
        raise NotImplementedError(f"{type(self).__name__} does not implement viscosity")

    def thermal_conductivity(self, T: float, p: float, composition_x: str) -> float:
        """Thermal conductivity λ [W/m/K]."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement thermal_conductivity"
        )

    def diffusion_coefficients(
        self, T: float, p: float, composition_x: str
    ) -> dict[str, float]:
        """Mixture-averaged diffusion coefficients D_i [m²/s], keyed by species name."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement diffusion_coefficients"
        )


# Backward-compatible alias
ThermoBackend = ThermochemBackend


# ---------------------------------------------------------------------------
# Cantera implementation
# ---------------------------------------------------------------------------


class CanteraBackend(ThermochemBackend):
    """Cantera-based thermochemistry + transport backend.

    Parameters
    ----------
    gas_mechanism:
        Path or built-in name for the gas phase mechanism.
    carbon_phase_file:
        Path or built-in name for the condensed phase.
    gas_phase_name:
        Named phase inside *gas_mechanism* (empty → default phase).
    condensed_phase_name:
        Named phase inside *carbon_phase_file* (empty → default phase).
    initial_gas_moles:
        Initial moles of gas in the Cantera Mixture.
    initial_carbon_moles:
        Initial moles of condensed phase in the Cantera Mixture.
    solver:
        Equilibrium solver: ``"gibbs"``, ``"vcs"``, or ``"auto"``.
    max_steps:
        Maximum solver steps.
    max_iter:
        Maximum outer iterations.
    log_level:
        Cantera verbosity level (0 = silent).
    extra_condensed_phases:
        Optional list of ``(mechanism_file, phase_name_or_empty, initial_moles)``
        for additional competing condensed phases.
    transport_model:
        Cantera transport model for Group 3 methods (default
        ``"mixture-averaged"``).  Set to ``None`` to skip transport
        initialisation (Group 3 methods will raise ``NotImplementedError``).
    """

    def __init__(
        self,
        gas_mechanism: str,
        carbon_phase_file: str,
        gas_phase_name: str = "",
        condensed_phase_name: str = "",
        initial_gas_moles: float = 1.0,
        initial_carbon_moles: float = 100.0,
        solver: str = "gibbs",
        max_steps: int = 1000,
        max_iter: int = 200,
        log_level: int = 0,
        extra_condensed_phases: list[tuple[str, str, float]] | None = None,
        transport_model: str | None = "mixture-averaged",
    ) -> None:
        from .thermochemistry import require_cantera, _load_solution

        ct = require_cantera()
        self._ct = ct
        self._gas_mechanism = gas_mechanism
        self._carbon_phase_file = carbon_phase_file
        self._gas_phase_name = gas_phase_name or None
        self._condensed_phase_name = condensed_phase_name or None
        self._initial_gas_moles = initial_gas_moles
        self._initial_carbon_moles = initial_carbon_moles
        self._solver = solver
        self._max_steps = max_steps
        self._max_iter = max_iter
        self._log_level = log_level
        self._transport_model = transport_model

        self._gas = _load_solution(gas_mechanism, self._gas_phase_name)
        self._carbon = _load_solution(carbon_phase_file, self._condensed_phase_name)

        # Second gas Solution for transport/thermo queries so B' warm-start
        # state on self._gas is not disturbed by group-2/3 calls.
        self._gas_thermo = _load_solution(gas_mechanism, self._gas_phase_name)
        if transport_model is not None:
            try:
                self._gas_thermo.transport_model = transport_model
            except Exception:
                pass  # mechanism may not have transport data; methods raise on use

        self._extra_condensed: list[tuple[object, float]] = []
        if extra_condensed_phases:
            for mech, phase_name, n_moles in extra_condensed_phases:
                sol = _load_solution(mech, phase_name or None)
                self._extra_condensed.append((sol, float(n_moles)))

    # ------------------------------------------------------------------
    # Group 1 — B' surface chemistry
    # ------------------------------------------------------------------

    def element_mass_fraction(
        self,
        composition_x: str | None,
        composition_y: str | None,
        temperature_K: float,
        pressure_Pa: float,
        element: str,
    ) -> float:
        from .thermochemistry import _set_composition, _elemental_mass_fraction

        _set_composition(self._gas, temperature_K, pressure_Pa, composition_x, composition_y)
        return _elemental_mass_fraction(self._gas, element)

    def blend_streams(
        self,
        edge_x: str | None,
        edge_y: str | None,
        pyro_x: str | None,
        pyro_y: str | None,
        bg: float,
        temperature_K: float,
        pressure_Pa: float,
        target_element: str,
    ) -> tuple[object, float]:
        from .thermochemistry import compute_blended_state

        Y_blend, Z_edge_eff, _Z_edge, _Z_pyro = compute_blended_state(
            self._gas,
            temperature_K=temperature_K,
            pressure_Pa=pressure_Pa,
            edge_x=edge_x,
            edge_y=edge_y,
            pyro_x=pyro_x,
            pyro_y=pyro_y,
            bg=bg,
            target_element=target_element,
        )
        return Y_blend, Z_edge_eff

    def equilibrate(
        self,
        blended_composition: object,
        temperature_K: float,
        pressure_Pa: float,
    ) -> WallState:
        import numpy as np

        Y_blend = np.asarray(blended_composition)

        condensed_phase_mole_fractions: dict[str, float] | None = None

        if self._extra_condensed:
            from .thermochemistry import (
                equilibrate_wall_multi_condensed,
                _condensed_surface_mole_fractions,
            )
            all_condensed = [(self._carbon, self._initial_carbon_moles)] + list(
                self._extra_condensed
            )
            wall_gas, gas_moles, condensed_moles_dict, _su, _na = (
                equilibrate_wall_multi_condensed(
                    gas=self._gas,
                    condensed_phases=all_condensed,
                    temperature_K=temperature_K,
                    pressure_Pa=pressure_Pa,
                    Y_blend=Y_blend,
                    max_steps=self._max_steps,
                    max_iter=self._max_iter,
                    log_level=self._log_level,
                )
            )
            names = list(condensed_moles_dict.keys())
            moles = list(condensed_moles_dict.values())
            condensed_phase_mole_fractions = _condensed_surface_mole_fractions(names, moles)
            condensed_moles = sum(max(m, 0.0) for m in moles)
        else:
            from .thermochemistry import equilibrate_wall_with_condensed_phase

            wall_gas, gas_moles, condensed_moles, _su, _na = (
                equilibrate_wall_with_condensed_phase(
                    gas_mechanism=self._gas_mechanism,
                    carbon_phase_file=self._carbon_phase_file,
                    temperature_K=temperature_K,
                    pressure_Pa=pressure_Pa,
                    Y_blend=Y_blend,
                    initial_gas_moles=self._initial_gas_moles,
                    initial_carbon_moles=self._initial_carbon_moles,
                    solver=self._solver,
                    max_steps=self._max_steps,
                    max_iter=self._max_iter,
                    log_level=self._log_level,
                    gas_phase_name=self._gas_phase_name,
                    condensed_phase_name=self._condensed_phase_name,
                    gas=self._gas,
                    carbon=self._carbon,
                )
            )

        element_mf: dict[str, float] = {}
        for el in wall_gas.element_names:
            try:
                element_mf[el] = float(wall_gas.elemental_mass_fraction(el))
            except Exception:
                element_mf[el] = 0.0

        species_mf = dict(zip(wall_gas.species_names, wall_gas.X.tolist()))

        return WallState(
            temperature_K=temperature_K,
            pressure_Pa=pressure_Pa,
            h_J_kg=float(wall_gas.enthalpy_mass),
            MW_kg_per_kmol=float(wall_gas.mean_molecular_weight),
            element_mass_fractions=element_mf,
            species_mole_fractions=species_mf,
            gas_moles=gas_moles,
            condensed_moles=condensed_moles,
            condensed_phase_mole_fractions=condensed_phase_mole_fractions,
        )

    # ------------------------------------------------------------------
    # Group 2 — Gas thermodynamic properties
    # ------------------------------------------------------------------

    def _set_thermo(self, T: float, p: float, composition_x: str) -> None:
        """Set self._gas_thermo state without disturbing B' warm-start gas."""
        self._gas_thermo.TPX = float(T), float(p), composition_x

    def gas_enthalpy(self, T: float, p: float, composition_x: str) -> float:
        self._set_thermo(T, p, composition_x)
        return float(self._gas_thermo.enthalpy_mass)

    def gas_enthalpy_equilibrium(self, T: float, p: float, composition_x: str) -> float:
        self._set_thermo(T, p, composition_x)
        try:
            self._gas_thermo.equilibrate("TP")
        except Exception as exc:
            raise RuntimeError(
                f"CanteraBackend: gas-only equilibration failed at T={T:.1f} K, "
                f"p={p:.0f} Pa: {exc}"
            ) from exc
        return float(self._gas_thermo.enthalpy_mass)

    def gas_cp(self, T: float, p: float, composition_x: str) -> float:
        self._set_thermo(T, p, composition_x)
        return float(self._gas_thermo.cp_mass)

    # ------------------------------------------------------------------
    # Group 3 — Gas transport properties
    # ------------------------------------------------------------------

    def _check_transport(self) -> None:
        if self._transport_model is None:
            raise NotImplementedError(
                "CanteraBackend was initialised with transport_model=None; "
                "transport properties are unavailable."
            )

    def viscosity(self, T: float, p: float, composition_x: str) -> float:
        self._check_transport()
        self._set_thermo(T, p, composition_x)
        return float(self._gas_thermo.viscosity)

    def thermal_conductivity(self, T: float, p: float, composition_x: str) -> float:
        self._check_transport()
        self._set_thermo(T, p, composition_x)
        return float(self._gas_thermo.thermal_conductivity)

    def diffusion_coefficients(
        self, T: float, p: float, composition_x: str
    ) -> dict[str, float]:
        self._check_transport()
        self._set_thermo(T, p, composition_x)
        return dict(
            zip(self._gas_thermo.species_names, self._gas_thermo.mix_diff_coeffs.tolist())
        )

    def condensed_enthalpy(self, T: float, p: float) -> float:
        """Enthalpy of the condensed phase (graphite/char) at (T, p) [J/kg]."""
        self._carbon.TP = float(T), float(p)
        return float(self._carbon.enthalpy_mass)
