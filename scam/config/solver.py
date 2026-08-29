# SPDX-License-Identifier: MIT
"""Solver options dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SolverOptions:
    """Numerical solver configuration.

    Time integration: semi-implicit Backward Euler for conduction;
    exact (analytical) Arrhenius integration at fixed T per nodelet for decomposition.
    Adaptive timestep based on temperature and density change rates.
    """

    t_start: float = 0.0       # simulation start time [s]
    t_end: float = 10.0        # simulation end time [s]

    dt_init: float = 1e-3      # initial timestep [s]
    dt_min: float = 1e-7       # minimum allowed timestep [s]
    dt_max: float = 1.0        # maximum allowed timestep [s]

    # Adaptive timestep criteria:
    dt_max_dT: float = 50.0        # max allowed |ΔT| per step in any cell [K]
    dt_max_drho_frac: float = 0.05 # max allowed |Δρ/ρ| per step in any nodelet [-]

    # Surface energy balance Newton convergence
    max_seb_iter: int = 50
    seb_tol: float = 1.0           # SEB residual tolerance [W/m^2]

    # Output
    output_dt: float = 0.1         # interval between saved results [s]

    # Mesh management
    # Trigger node drop when shrinking cell thickness < fraction * nominal_spacing
    node_drop_threshold: float = 0.1

    # Thermocouple probe depths [m] from original surface (interpolated at each output step)
    tc_positions: list = field(default_factory=list)  # list[float]

    # Set False to suppress surface recession (e.g. for in-depth-only comparisons
    # where pyrolysis gas exits via Darcy flow rather than driving surface retreat)
    allow_recession: bool = True

    # Energy storage formulation for the in-depth FVM equation.
    # True  (default): exact d(ρh)/dt — h_old computed at pre-decomposition ρ_old;
    #   the (ρ_old − ρ_new)·h̄_sensible density-change term is implicit on the LHS.
    #   Q_vol carries only h_bar_chemical (= h_bar_absolute − h̄_sensible).
    # False: PATO approximation ρ·dh/dt ≈ ρ·cp·dT/dt — Dc_thermal = ρ·cp·T^n·A·Δ/dt;
    #   the implicit h̄_sensible contribution is absent (slight energy inaccuracy for
    #   charring materials, but matches PATO's in-depth solver formulation).
    use_rho_old: bool = True

    # Enable element transport PDE (Z_C, Z_H, Z_O, Z_N through porous medium).
    # Requires MaterialCard.pyro_elem_fracs to be set for each ablating layer.
    # When True, Z_C_wall feeds the 4-D B' table (or BprimeEvaluator).
    element_transport: bool = False

    # Use live Cantera B' evaluation instead of the pre-computed YAML table.
    # Requires `pip install -e ".[bprime]"` and a configured BprimeEvaluator.
    # Only active when element_transport=True and a bprime_config is supplied.
    # WARNING: adds ~10–100 ms per SEB Newton call; use for reference runs only.
    bprime_runtime: bool = False

    # Picard (fixed-point) iteration on the in-depth tridiagonal solve.
    # The system is assembled with cp(T^k) and re-solved until T converges.
    # max_picard=1 reproduces the old single-pass behaviour.
    max_picard: int = 8
    picard_tol: float = 0.05   # convergence criterion: max |T_new - T_k| [K]

    # Experimental numerical-kernel backend.  NumPy remains the production
    # default; "jax" JIT-compiles selected fixed-shape kernels and requires
    # installation with `pip install -e ".[jax]"`.
    array_backend: str = "numpy"

    # Recession scheme.  Default (False) uses the Lagrangian fixed-grid model:
    # the surface cell shrinks and is dropped/merged when thin (discrete, one
    # cell at a time) — this produces a small sawtooth in T_wall because each
    # drop discretely exposes the next, cooler sub-surface node.  Set True to
    # use a continuous moving-mesh (ALE) scheme: the ablating layer's nodes are
    # redistributed between the receding surface and the (fixed) layer back face
    # every step and the fields are conservatively re-interpolated.  Node count
    # stays fixed, no discrete drops occur, and T_wall is smooth — matching
    # PATO's mesh-motion treatment.
    continuous_remap: bool = False

    # Prescribe a fixed surface recession rate [m/s] independently of the surface
    # mass balance.  When set, overrides the s_dot computed by the surface solver
    # every step.  Useful for verification cases where T_wall is prescribed but
    # the recession rate is also known analytically (e.g. Baer & Ambrosio 1961).
    s_dot_prescribed: float | None = None
