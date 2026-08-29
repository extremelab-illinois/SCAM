from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from .composition import normalize_composition_string
from .models import BPrimeCase

if TYPE_CHECKING:
    import cantera as ct
    from .mat.constraints import SurfaceElementConstraint


def require_cantera():
    try:
        import cantera as ct
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Cantera is required for B' table generation. "
            "Install with: python -m pip install cantera"
        ) from exc
    return ct


def _load_solution(mechanism: str, phase_name: str | None = None):
    """Load a Cantera Solution with an optional named phase."""
    ct = require_cantera()

    if phase_name:
        return ct.Solution(mechanism, phase_name)

    return ct.Solution(mechanism)


def _set_composition(
    gas: "ct.Solution",
    temperature_K: float,
    pressure_Pa: float,
    comp_x: str | None,
    comp_y: str | None,
) -> None:
    if comp_x:
        gas.TPX = temperature_K, pressure_Pa, normalize_composition_string(comp_x)
    elif comp_y:
        gas.TPY = temperature_K, pressure_Pa, normalize_composition_string(comp_y)
    else:
        raise ValueError("Exactly one of comp_x or comp_y must be provided.")


def _elemental_mass_fraction(gas: "ct.Solution", element: str) -> float:
    """Return elemental mass fraction, or 0 if the element is absent."""
    try:
        return float(gas.elemental_mass_fraction(element))
    except Exception:
        return 0.0


def compute_blended_state(
    gas: "ct.Solution",
    *,
    temperature_K: float,
    pressure_Pa: float,
    edge_x: str | None,
    edge_y: str | None,
    pyro_x: str | None,
    pyro_y: str | None,
    bg: float,
    target_element: str = "C",
) -> tuple[np.ndarray, float, float, float]:
    """Blend edge and pyrolysis streams by mass.

    Returns
    -------
    Y_blend : np.ndarray
        Blended gas mass fractions.
    Z_target_edge_eff : float
        Effective inlet elemental mass fraction (blended edge + pyro).
    Z_target_edge : float
        Elemental mass fraction of target element in the edge gas alone.
    Z_target_pyro : float
        Elemental mass fraction of target element in the pyrolysis gas alone.
        Equals ``Z_target_edge`` when no separate pyrolysis stream is given.

    Notes
    -----
    The blend is mass-weighted: 1 unit of edge gas + B'g units of pyrolysis gas.
    For B'g < 0 or when no pyrolysis composition is supplied, the pyrolysis
    stream is replaced by the edge stream.
    """
    _set_composition(gas, temperature_K, pressure_Pa, edge_x, edge_y)
    Y_edge = gas.Y.copy()
    Z_edge = _elemental_mass_fraction(gas, target_element)

    if bg == 0.0:
        return Y_edge, Z_edge, Z_edge, Z_edge

    has_pyro = pyro_x or pyro_y
    if bg < 0.0 or not has_pyro:
        Y_pyro = Y_edge
        Z_pyro = Z_edge
    else:
        _set_composition(gas, temperature_K, pressure_Pa, pyro_x, pyro_y)
        Y_pyro = gas.Y.copy()
        Z_pyro = _elemental_mass_fraction(gas, target_element)

    total = 1.0 + bg
    if abs(total) < 1.0e-12:
        raise ValueError(f"B'g = {bg:.4g} yields zero total mass flux (1 + B'g ≈ 0).")

    Y_blend = (Y_edge + bg * Y_pyro) / total
    Y_blend = np.clip(Y_blend, 0.0, None)

    mass_sum = Y_blend.sum()
    if mass_sum > 0.0:
        Y_blend /= mass_sum

    Z_target_edge_eff = (Z_edge + bg * Z_pyro) / total
    return Y_blend, Z_target_edge_eff, Z_edge, Z_pyro


# ---------------------------------------------------------------------------
# Core equilibrium physics
# ---------------------------------------------------------------------------

def _get_phase_moles(mix: "ct.Mixture", phase_index: int) -> float | None:
    try:
        return float(mix.phase_moles(phase_index))
    except Exception:
        return None


def _condensed_surface_mole_fractions(
    names: list[str], moles: list[float]
) -> dict[str, float]:
    """Normalise condensed phase moles to surface mole fractions X_l.

    Negative moles (numerical noise) are clipped to zero before normalisation.
    Returns all-zero dict when the total is zero.
    """
    clipped = [max(m, 0.0) for m in moles]
    total = sum(clipped)
    if total <= 0.0:
        return {n: 0.0 for n in names}
    return {n: m / total for n, m in zip(names, clipped)}


def _major_species_string(
    gas: "ct.Solution",
    *,
    threshold: float,
    max_species: int,
) -> str:
    pairs = [
        (name, float(x))
        for name, x in zip(gas.species_names, gas.X)
        if float(x) >= threshold
    ]
    pairs.sort(key=lambda pair: pair[1], reverse=True)

    if max_species > 0:
        pairs = pairs[:max_species]

    return ";".join(f"{name}:{x:.6e}" for name, x in pairs)


def _build_mixture(
    gas_mechanism: str,
    carbon_phase_file: str,
    temperature_K: float,
    pressure_Pa: float,
    Y_blend: np.ndarray,
    initial_gas_moles: float,
    initial_carbon_moles: float,
    *,
    gas_phase_name: str | None = None,
    condensed_phase_name: str | None = None,
    gas: "ct.Solution | None" = None,
    carbon: "ct.Solution | None" = None,
) -> tuple["ct.Solution", "ct.Mixture"]:
    """Build a Cantera Mixture for gas + condensed phase equilibration."""
    ct = require_cantera()

    if gas is None:
        gas = _load_solution(gas_mechanism, gas_phase_name)

    if carbon is None:
        carbon = _load_solution(carbon_phase_file, condensed_phase_name)

    gas.TPY = temperature_K, pressure_Pa, Y_blend
    carbon.TP = temperature_K, pressure_Pa

    mix = ct.Mixture([(gas, initial_gas_moles), (carbon, initial_carbon_moles)])
    mix.T = temperature_K
    mix.P = pressure_Pa

    return gas, mix


def _run_equilibrate(
    mix: "ct.Mixture",
    solver: str,
    max_steps: int,
    max_iter: int,
    log_level: int,
) -> None:
    try:
        mix.equilibrate(
            "TP",
            solver=solver,
            max_steps=max_steps,
            max_iter=max_iter,
            log_level=log_level,
        )
    except TypeError:
        mix.equilibrate(
            "TP",
            solver=solver,
            max_steps=max_steps,
            log_level=log_level,
        )


def equilibrate_wall_with_condensed_phase(
    *,
    gas_mechanism: str,
    carbon_phase_file: str,
    temperature_K: float,
    pressure_Pa: float,
    Y_blend: np.ndarray,
    initial_gas_moles: float,
    initial_carbon_moles: float,
    solver: str,
    max_steps: int,
    max_iter: int,
    log_level: int,
    gas_phase_name: str | None = None,
    condensed_phase_name: str | None = None,
    gas: "ct.Solution | None" = None,
    carbon: "ct.Solution | None" = None,
    damp_factor: float = 0.5,
    max_retries: int = 3,
) -> tuple["ct.Solution", float | None, float | None, str, int]:
    """Equilibrate blended gas against a condensed reservoir at fixed T and P.

    Tries the primary solver then its complement (gibbs↔vcs).  On total
    failure, scales the initial condensed moles by ``damp_factor`` up to
    ``max_retries`` times before giving up.

    Returns
    -------
    wall_gas : ct.Solution
    gas_moles : float | None
    condensed_moles : float | None
    solver_used : str
        Name of the Cantera solver that converged.
    n_attempts : int
        Total equilibrate() calls made (including failed ones).
    """
    fallback = {"gibbs": "vcs", "vcs": "gibbs"}.get(solver)
    solvers = [solver] + ([fallback] if fallback else [])

    last_exc: Exception | None = None
    n_attempts = 0
    current_moles = float(initial_carbon_moles)

    for _retry in range(max_retries + 1):
        for i, solver_name in enumerate(solvers):
            n_attempts += 1
            # On the very first attempt, reuse the caller-supplied Solution
            # objects for warm-start (continuation).  On all subsequent
            # attempts (fallback solver or damped retry) build fresh objects
            # so the corrupted solver state doesn't propagate.
            reuse_gas = gas if (n_attempts == 1) else None
            reuse_carbon = carbon if (n_attempts == 1) else None

            wall_gas, mix = _build_mixture(
                gas_mechanism=gas_mechanism,
                carbon_phase_file=carbon_phase_file,
                temperature_K=temperature_K,
                pressure_Pa=pressure_Pa,
                Y_blend=Y_blend,
                initial_gas_moles=initial_gas_moles,
                initial_carbon_moles=current_moles,
                gas_phase_name=gas_phase_name,
                condensed_phase_name=condensed_phase_name,
                gas=reuse_gas,
                carbon=reuse_carbon,
            )
            try:
                _run_equilibrate(mix, solver_name, max_steps, max_iter, log_level)
                return (
                    wall_gas,
                    _get_phase_moles(mix, 0),
                    _get_phase_moles(mix, 1),
                    solver_name,
                    n_attempts,
                )
            except Exception as exc:
                last_exc = exc

        # All solvers failed at this moles level → damp and retry
        current_moles *= damp_factor
        if current_moles < 1e-12:
            break

    raise RuntimeError(
        f"Equilibrium did not converge at T={temperature_K:.1f} K, "
        f"p={pressure_Pa:.4g} Pa (all solvers and {max_retries} damped "
        "retries failed)."
    ) from last_exc


# Backward-compatible alias used by older imports/tests.
equilibrate_wall_with_graphite = equilibrate_wall_with_condensed_phase


# ---------------------------------------------------------------------------
# Multi-condensed equilibration
# ---------------------------------------------------------------------------


def _build_mixture_multi(
    gas: "ct.Solution",
    Y_blend: np.ndarray,
    temperature_K: float,
    pressure_Pa: float,
    condensed_phases: "list[tuple[ct.Solution, float]]",
) -> "tuple[ct.Solution, ct.Mixture]":
    """Build a Cantera Mixture with one gas phase and N condensed phases."""
    ct = require_cantera()
    gas.TPY = temperature_K, pressure_Pa, Y_blend
    for cond, _ in condensed_phases:
        cond.TP = temperature_K, pressure_Pa
    mix = ct.Mixture([(gas, 1.0)] + list(condensed_phases))
    mix.T = temperature_K
    mix.P = pressure_Pa
    return gas, mix


def equilibrate_wall_multi_condensed(
    *,
    gas: "ct.Solution",
    condensed_phases: "list[tuple[ct.Solution, float]]",
    temperature_K: float,
    pressure_Pa: float,
    Y_blend: np.ndarray,
    max_steps: int = 5000,
    max_iter: int = 200,
    log_level: int = 0,
    damp_factor: float = 0.5,
    max_retries: int = 3,
) -> "tuple[ct.Solution, float | None, dict[str, float], str, int]":
    """Equilibrate blended gas against N competing condensed phases at fixed T, P.

    Uses the ``vcs`` solver, which handles multi-phase systems reliably.
    Falls back to ``gibbs`` only if ``vcs`` fails.  On total solver failure,
    scales the initial condensed moles by ``damp_factor`` up to ``max_retries``
    times.

    Parameters
    ----------
    gas:
        Pre-created Cantera gas-phase Solution (will be mutated in place).
    condensed_phases:
        List of ``(Solution, initial_moles)`` pairs for each condensed phase.
        Phases are identified by their ``Solution.name``.
    temperature_K, pressure_Pa:
        Fixed thermodynamic state.
    Y_blend:
        Blended gas mass fractions (from ``compute_blended_state``).

    Returns
    -------
    wall_gas : ct.Solution
    gas_moles : float | None
    condensed_moles_dict : dict[str, float]
    solver_used : str
    n_attempts : int
    """
    solvers = ["vcs", "gibbs"]
    last_exc: Exception | None = None
    n_attempts = 0
    damp_scale = 1.0

    for _retry in range(max_retries + 1):
        scaled_phases = [(sol, moles * damp_scale) for sol, moles in condensed_phases]

        for solver_name in solvers:
            n_attempts += 1
            wall_gas, mix = _build_mixture_multi(
                gas, Y_blend, temperature_K, pressure_Pa, scaled_phases
            )
            try:
                _run_equilibrate(mix, solver_name, max_steps, max_iter, log_level)
                gas_moles = _get_phase_moles(mix, 0)
                condensed_moles_dict = {
                    mix.phase(i).name: float(mix.phase_moles(i))
                    for i in range(1, mix.n_phases)
                }
                return wall_gas, gas_moles, condensed_moles_dict, solver_name, n_attempts
            except Exception as exc:
                last_exc = exc

        damp_scale *= damp_factor
        if damp_scale < 1e-12:
            break

    raise RuntimeError(
        f"Multi-condensed equilibrium did not converge at T={temperature_K:.1f} K, "
        f"p={pressure_Pa:.4g} Pa (all solvers and {max_retries} damped retries failed)."
    ) from last_exc


def compute_bprime_case(
    *,
    gas_mechanism: str,
    carbon_phase_file: str,
    temperature_K: float,
    pressure_Pa: float,
    bg: float,
    edge_x: str | None,
    edge_y: str | None,
    pyro_x: str | None,
    pyro_y: str | None,
    initial_gas_moles: float,
    initial_carbon_moles: float,
    solver: str,
    max_steps: int,
    max_iter: int,
    log_level: int,
    species_threshold: float,
    max_species: int,
    gas_phase_name: str | None = None,
    condensed_phase_name: str | None = None,
    target_element: str = "C",
    surface_source_target_fraction: float = 1.0,
    gas: "ct.Solution | None" = None,
    carbon: "ct.Solution | None" = None,
    extra_condensed_phases: "list[tuple[ct.Solution, float]] | None" = None,
    surface_constraint: "SurfaceElementConstraint | None" = None,
    species_elements: "dict[str, dict[str, float]] | None" = None,
    material_condensed_species: "list | None" = None,
    surface_reactions: "list | None" = None,
    reaction_active_threshold: float = 1e-20,
) -> BPrimeCase:
    """Compute one B' table point.

    Parameters
    ----------
    extra_condensed_phases:
        Optional list of ``(ct.Solution, initial_moles)`` for additional
        competing condensed phases.  When provided the multi-condensed
        equilibration path is used (``vcs`` solver) and
        ``condensed_surface_mole_fractions`` / ``dominant_condensed_species``
        are populated in the returned ``BPrimeCase``.  The primary condensed
        phase (``carbon_phase_file``) is always the first phase in the
        Mixture; extras follow in order.
    surface_constraint:
        Optional ``SurfaceElementConstraint``.  When provided, the raw
        unconstrained surface mole fractions are projected onto the
        constraint manifold.  ``condensed_surface_mole_fractions`` in the
        returned ``BPrimeCase`` holds the constrained result;
        ``unconstrained_surface_mole_fractions`` holds the raw equilibrium.
        Only meaningful when ``extra_condensed_phases`` is also provided.
    species_elements:
        Elemental composition of each condensed species, used to evaluate
        and enforce the constraint.  Keys are species names (matching the
        phase names from Cantera); values are ``{element: atoms}`` dicts.
        If ``None``, falls back to an empty composition (constraint checks
        will always return zero elemental fractions).
    material_condensed_species:
        Optional list of ``CondensedSpecies`` objects (from
        ``scam.chemistry.materials``).  When provided, the failure model
        (Phase 7) is evaluated: species whose ``failure_temperature_K``
        is exceeded contribute to ``Bprime_fail`` and are listed in
        ``failing_species``.
    surface_reactions:
        Optional list of ``HeterogeneousReaction`` objects (from
        ``scam.chemistry.mat.reactions``).  When provided, each reaction rate is
        evaluated at the equilibrated wall state using the Arrhenius law.
        ``reaction_rates`` and ``active_reactions`` are populated in the
        returned ``BPrimeCase``.  Reaction rates are in kmol/m²/s.
    reaction_active_threshold:
        Minimum rate (kmol/m²/s) for a reaction to be listed in
        ``active_reactions`` (default 1e-20).
    """
    target_element = target_element.strip()
    if not target_element:
        raise ValueError("target_element cannot be empty")
    z_source = float(surface_source_target_fraction)
    if not (0.0 < z_source <= 1.0):
        raise ValueError(
            "surface_source_target_fraction must be in (0, 1], "
            f"got {surface_source_target_fraction!r}"
        )

    if gas is None:
        gas = _load_solution(gas_mechanism, gas_phase_name)

    Y_blend, Z_target_edge_eff, Z_edge_alone, Z_pyro_alone = compute_blended_state(
        gas,
        temperature_K=temperature_K,
        pressure_Pa=pressure_Pa,
        edge_x=edge_x,
        edge_y=edge_y,
        pyro_x=pyro_x,
        pyro_y=pyro_y,
        bg=bg,
        target_element=target_element,
    )

    # ── Choose single-condensed or multi-condensed equilibration path ────────
    surface_mole_fractions: dict[str, float] | None = None
    dominant_condensed: str | None = None

    if extra_condensed_phases:
        if carbon is None:
            carbon = _load_solution(carbon_phase_file, condensed_phase_name)
        all_condensed = [(carbon, float(initial_carbon_moles))] + list(extra_condensed_phases)

        wall_gas, gas_moles_final, condensed_moles_dict, _solver_used, _n_attempts = \
            equilibrate_wall_multi_condensed(
                gas=gas,
                condensed_phases=all_condensed,
                temperature_K=temperature_K,
                pressure_Pa=pressure_Pa,
                Y_blend=Y_blend,
                max_steps=max_steps,
                max_iter=max_iter,
                log_level=log_level,
            )
        names = list(condensed_moles_dict.keys())
        moles = list(condensed_moles_dict.values())
        surface_mole_fractions = _condensed_surface_mole_fractions(names, moles)
        dominant_condensed = max(surface_mole_fractions, key=surface_mole_fractions.get)  # type: ignore[arg-type]
        condensed_moles_final = condensed_moles_dict.get(
            next(iter(condensed_moles_dict)), None
        )
    else:
        wall_gas, gas_moles_final, condensed_moles_final, _solver_used, _n_attempts = \
            equilibrate_wall_with_condensed_phase(
                gas_mechanism=gas_mechanism,
                carbon_phase_file=carbon_phase_file,
                temperature_K=temperature_K,
                pressure_Pa=pressure_Pa,
                Y_blend=Y_blend,
                initial_gas_moles=initial_gas_moles,
                initial_carbon_moles=initial_carbon_moles,
                solver=solver,
                max_steps=max_steps,
                max_iter=max_iter,
                log_level=log_level,
                gas_phase_name=gas_phase_name,
                condensed_phase_name=condensed_phase_name,
                gas=gas,
                carbon=carbon,
            )

    try:
        Z_target_wall = float(wall_gas.elemental_mass_fraction(target_element))
    except Exception as exc:
        raise RuntimeError(
            f"The gas mechanism does not contain target element {target_element!r}. "
            "Use a mechanism containing the required ablation element."
        ) from exc

    denom_source = z_source - Z_target_wall
    if denom_source <= 0.0:
        bprime_eq = math.inf
    else:
        bprime_eq = (
            (1.0 + bg)
            * (Z_target_wall - Z_target_edge_eff)
            / denom_source
        )

    is_carbon = target_element == "C"
    carbon_value = float(bprime_eq) if is_carbon else math.nan
    carbon_wall = Z_target_wall if is_carbon else math.nan
    carbon_edge_eff = Z_target_edge_eff if is_carbon else math.nan
    carbon_moles_final = condensed_moles_final if is_carbon else math.nan

    # ── Phase 5: apply surface-element constraint if provided ───────────────
    unconstrained_mole_fractions: dict[str, float] | None = None
    constraint_satisfied: bool | None = None

    if surface_constraint is not None and surface_mole_fractions is not None:
        from .mat.constraints import apply_surface_element_constraint
        sp_el = species_elements or {}
        unconstrained_mole_fractions = dict(surface_mole_fractions)
        surface_mole_fractions, constraint_satisfied = apply_surface_element_constraint(
            surface_mole_fractions, sp_el, surface_constraint
        )
        if surface_mole_fractions:
            dominant_condensed = max(
                surface_mole_fractions, key=surface_mole_fractions.get  # type: ignore[arg-type]
            )

    # ── Phase 8: heterogeneous surface reaction rates ────────────────────────
    _reaction_rates: dict[str, float] | None = None
    _active_reactions: tuple[str, ...] = ()

    if surface_reactions:
        from .mat.reactions import compute_surface_reaction_rates, active_reactions as _active_fn
        wall_X = dict(zip(wall_gas.species_names, wall_gas.X.tolist()))
        _reaction_rates = compute_surface_reaction_rates(
            surface_reactions, temperature_K, wall_X
        )
        _active_reactions = tuple(_active_fn(_reaction_rates, reaction_active_threshold))

    # ── Phase 7: material failure model ─────────────────────────────────────
    _failing_species: tuple[str, ...] = ()
    _bprime_fail = 0.0

    if material_condensed_species:
        from .mat.failure import compute_failure_state
        x_l_for_failure = surface_mole_fractions or {}
        failure = compute_failure_state(
            material_condensed_species, temperature_K, x_l_for_failure
        )
        _failing_species = failure.failing_species
        _bprime_fail = failure.Bprime_fail

    return BPrimeCase(
        T_wall_K=temperature_K,
        pressure_Pa=pressure_Pa,
        Bg=bg,

        Bprime_eq=float(bprime_eq),
        target_element=target_element,
        Z_target_wall=Z_target_wall,
        Z_target_edge_eff=Z_target_edge_eff,

        Bprime_c_eq=carbon_value,
        Z_C_wall=carbon_wall,
        Z_C_edge_eff=carbon_edge_eff,

        h_wall_gas_J_kg=float(wall_gas.enthalpy_mass),
        MW_wall_gas_kg_per_kmol=float(wall_gas.mean_molecular_weight),
        gas_moles_final=gas_moles_final,

        condensed_phase_name=condensed_phase_name or "",
        condensed_phase_moles_final=condensed_moles_final,

        carbon_phase_moles_final=carbon_moles_final,

        major_wall_species_X=_major_species_string(
            wall_gas,
            threshold=species_threshold,
            max_species=max_species,
        ),

        condensed_surface_mole_fractions=surface_mole_fractions,
        dominant_condensed_species=dominant_condensed,
        unconstrained_surface_mole_fractions=unconstrained_mole_fractions,
        surface_constraint_satisfied=constraint_satisfied,

        Bprime_g=float(bg),
        Bprime_fail=_bprime_fail,
        Bprime_total=float(bprime_eq) + float(bg) + _bprime_fail,
        Z_edge_target=Z_edge_alone,
        Z_pyro_target=Z_pyro_alone,
        Z_surface_source_target=z_source,

        failing_species=_failing_species,

        active_reactions=_active_reactions,
        reaction_rates=_reaction_rates,

        converged=True,
        solver_used=_solver_used,
        n_solver_attempts=_n_attempts,
    )
