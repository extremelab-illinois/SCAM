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

    t_start: float = 0.0       #: simulation start time [s]
    t_end: float = 10.0        #: simulation end time [s]

    dt_init: float = 1e-3      #: initial timestep [s]
    dt_min: float = 1e-7       #: minimum allowed timestep [s]
    dt_max: float = 1.0        #: maximum allowed timestep [s]

    # Adaptive timestep criteria:
    dt_max_dT: float = 50.0        #: max allowed ``|delta T|`` per step in any cell [K]
    dt_max_drho_frac: float = 0.05 #: max allowed ``|delta rho / rho|`` per step in any nodelet [-]

    # Surface energy balance Newton convergence
    max_seb_iter: int = 50         #: max Newton iterations for the SEB solve
    seb_tol: float = 1.0           #: SEB residual tolerance [W/m^2]

    # Output
    output_dt: float = 0.1         #: interval between saved results [s]

    # Mesh management
    #: Trigger a node drop when the shrinking surface cell's thickness falls
    #: below this fraction of the nominal node spacing.
    node_drop_threshold: float = 0.1

    #: Thermocouple probe depths [m] from the original (t=0) front face;
    #: interpolated at each output step.
    tc_positions: list = field(default_factory=list)  # list[float]

    #: Set False to suppress surface recession (e.g. for in-depth-only
    #: comparisons where pyrolysis gas exits via Darcy flow rather than
    #: driving surface retreat).
    allow_recession: bool = True

    #: Energy storage formulation for the in-depth FVM equation. ``True``
    #: (default): exact ``d(rho h)/dt`` with ``h_old`` at pre-decomposition
    #: ``rho_old``. ``False``: PATO's ``rho*dh/dt ~= rho*cp*dT/dt``
    #: approximation. See the theory manual's energy-formulation comparison
    #: (:doc:`/theory/04_energy_formulation_comparison`).
    use_rho_old: bool = True

    #: Enable element transport PDE (Z_C, Z_H, Z_O, Z_N through porous
    #: medium). Requires ``MaterialCard.pyro_elem_fracs`` to be set for each
    #: ablating layer. When True, ``Z_C_wall`` feeds the 4-D B' table (or
    #: ``BprimeEvaluator``).
    element_transport: bool = False

    #: Use live Cantera B' evaluation instead of the pre-computed YAML
    #: table. Requires ``pip install -e ".[bprime]"`` and a configured
    #: ``BprimeEvaluator``. Only active when ``element_transport=True`` and
    #: a bprime config is supplied. Adds roughly 10-100 ms per SEB Newton
    #: call; use for reference runs only.
    bprime_runtime: bool = False

    #: Picard (fixed-point) iterations on the in-depth tridiagonal solve.
    #: The system is re-assembled with ``cp(T^k)`` and re-solved until T
    #: converges. ``max_picard=1`` reproduces the old single-pass behaviour.
    max_picard: int = 8
    picard_tol: float = 0.05   #: Picard convergence criterion: max ``|T_new - T_k|`` [K]

    #: Experimental numerical-kernel backend. NumPy remains the production
    #: default; ``"jax"`` JIT-compiles selected fixed-shape kernels and
    #: requires installation with ``pip install -e ".[jax]"``.
    array_backend: str = "numpy"

    #: Recession scheme. ``False`` (default): Lagrangian fixed grid -- the
    #: surface cell shrinks and is dropped/merged one cell at a time, which
    #: puts a small sawtooth on ``T_wall``. ``True``: continuous ALE moving
    #: mesh -- nodes are redistributed every step between the receding
    #: surface and the fixed back face, conservatively re-interpolated; node
    #: count stays fixed and ``T_wall`` is smooth. See the theory manual's
    #: recession chapter (:doc:`/theory/08_recession_and_mesh_motion`).
    continuous_remap: bool = False

    #: Prescribe a fixed surface recession rate [m/s] independently of the
    #: surface mass balance. When set, overrides the ``s_dot`` computed by
    #: the surface solver every step. Useful for verification cases where
    #: ``T_wall`` is prescribed but the recession rate is also known
    #: analytically (e.g. Baer & Ambrosio 1961).
    s_dot_prescribed: float | None = None
