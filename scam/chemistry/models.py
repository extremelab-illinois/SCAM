from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BPrimeCase:
    T_wall_K: float
    pressure_Pa: float
    Bg: float

    # Backward-compatible carbon output
    Bprime_c_eq: float
    Z_C_wall: float
    Z_C_edge_eff: float

    h_wall_gas_J_kg: float
    MW_wall_gas_kg_per_kmol: float
    gas_moles_final: float | None
    carbon_phase_moles_final: float | None
    major_wall_species_X: str

    # Generic B-prime output for non-carbon materials, e.g. silica/Si.
    # Defaults preserve compatibility with older carbon-only tests and callers.
    Bprime_eq: float = math.nan
    target_element: str = "C"
    Z_target_wall: float = math.nan
    Z_target_edge_eff: float = math.nan
    condensed_phase_name: str = "graphite"
    condensed_phase_moles_final: float | None = None

    # Phase 4: surface mole fractions for competing condensed species.
    condensed_surface_mole_fractions: dict[str, float] | None = None
    dominant_condensed_species: str | None = None

    # Phase 5: surface-element constraint results.
    # When a SurfaceElementConstraint is applied, condensed_surface_mole_fractions
    # holds the CONSTRAINED X_l.  These fields record the pre-constraint values
    # and whether the constraint was satisfiable.
    unconstrained_surface_mole_fractions: dict[str, float] | None = None
    surface_constraint_satisfied: bool | None = None

    # Phase 6: distinguished blowing parameters and inlet elemental fractions.
    Bprime_g: float = 0.0
    Bprime_fail: float = 0.0
    Bprime_total: float = math.nan
    Z_edge_target: float = math.nan   # target-element mass fraction in edge gas
    Z_pyro_target: float = math.nan   # target-element mass fraction in pyrolysis gas
    Z_surface_source_target: float = 1.0  # target-element fraction in surface source

    # Phase 7: material failure state.
    # failing_species lists condensed species whose failure temperature was exceeded.
    failing_species: tuple[str, ...] = ()

    # Phase 8: heterogeneous surface reaction rates.
    # active_reactions lists reactions whose computed rate exceeds the active threshold.
    # reaction_rates maps reaction name → rate in kmol/m²/s.
    active_reactions: tuple[str, ...] = ()
    reaction_rates: dict[str, float] | None = None

    # Phase 9: solver diagnostics.
    # converged=False means the equilibration used a damped-moles fallback or failed.
    # solver_used records which Cantera solver succeeded ("gibbs", "vcs", or "failed").
    # n_solver_attempts counts total equilibrate() calls including retries.
    converged: bool = True
    solver_used: str = ""
    n_solver_attempts: int = 1
