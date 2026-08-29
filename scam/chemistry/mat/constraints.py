"""Surface-element constraints for multicomponent ablation (Phase 5).

A ``SurfaceElementConstraint`` specifies target molar ratios for selected
elements in the condensed surface layer.  For example, a pure SiC surface
has Si:C = 1:1; a mixed SiC/SiO2 surface without free silicon would still
enforce Si:C = 1:1 if the bulk material is SiC.

The constraint is applied **post-equilibration**: Cantera's unconstrained
multi-condensed equilibrium gives raw surface mole fractions X_l, which are
then projected onto the constraint manifold (minimum L2 departure) using
scipy's SLSQP solver.  This is an approximation to the full constrained-
equilibrium formulation; it is exact when the constraint manifold is a
linear subspace of the simplex, which holds for linear elemental-ratio
constraints.

All quantities are in SI units or dimensionless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SurfaceElementConstraint:
    """Target molar ratios for elements in the condensed surface layer.

    Attributes
    ----------
    ratios:
        Mapping of element symbol → relative molar amount.  Only the
        ratios between values matter; absolute values are normalised
        internally.  Example: ``{"Si": 1.0, "C": 1.0}`` for SiC (Si:C = 1:1).
    strict:
        If ``True`` (default), raise ``ConstraintInfeasibleError`` when no
        feasible solution exists.  If ``False``, return the unconstrained
        X_l and set ``constraint_satisfied=False`` on the result.
    """

    ratios: dict[str, float]
    strict: bool = True

    def __post_init__(self) -> None:
        if not self.ratios:
            raise ValueError("SurfaceElementConstraint.ratios must be non-empty.")
        for el, val in self.ratios.items():
            if val < 0.0:
                raise ValueError(
                    f"Element ratio for {el!r} must be non-negative, got {val}."
                )
        if all(v == 0.0 for v in self.ratios.values()):
            raise ValueError("At least one ratio value must be positive.")


class ConstraintInfeasibleError(RuntimeError):
    """Raised when the surface-element constraint cannot be satisfied."""


# ---------------------------------------------------------------------------
# Core constraint application
# ---------------------------------------------------------------------------


def apply_surface_element_constraint(
    X_l: dict[str, float],
    species_elements: dict[str, dict[str, float]],
    constraint: SurfaceElementConstraint,
) -> tuple[dict[str, float], bool]:
    """Project unconstrained surface mole fractions onto the constraint manifold.

    Finds the distribution X_l* that:
    - Is closest (L2) to the unconstrained equilibrium X_l.
    - Satisfies the target elemental ratios in ``constraint.ratios``.
    - Has all non-negative entries summing to 1.

    Parameters
    ----------
    X_l:
        Unconstrained surface mole fractions, keyed by species name.
    species_elements:
        Mapping of species name → {element: atoms_per_formula_unit}.
        Only the species present in ``X_l`` are used.
    constraint:
        Target elemental ratio specification.

    Returns
    -------
    X_constrained : dict[str, float]
        Constrained surface mole fractions (sum = 1, all ≥ 0).
    satisfied : bool
        ``True`` if the optimiser converged to a feasible solution.
    """
    from scipy.optimize import minimize  # deferred import; scipy is optional

    names = list(X_l.keys())
    n = len(names)

    if n == 0:
        return {}, True

    if n == 1:
        # Only one species — trivially satisfies any constraint.
        return {names[0]: 1.0}, True

    constrained_elements = list(constraint.ratios.keys())
    r = np.array([constraint.ratios[el] for el in constrained_elements], dtype=float)
    r_sum = r.sum()
    if r_sum > 0.0:
        r = r / r_sum  # normalise

    # Build composition matrix A[i, j] = atoms of element j in species i.
    A = np.zeros((n, len(constrained_elements)))
    for i, name in enumerate(names):
        comp = species_elements.get(name, {})
        for j, el in enumerate(constrained_elements):
            A[i, j] = comp.get(el, 0.0)

    X0 = np.array([max(X_l[name], 0.0) for name in names])
    s = X0.sum()
    if s > 0.0:
        X0 /= s

    # Objective: minimise L2 distance from unconstrained equilibrium.
    def objective(X: np.ndarray) -> float:
        return float(np.dot(X - X0, X - X0))

    def jac(X: np.ndarray) -> np.ndarray:
        return 2.0 * (X - X0)

    # Equality constraints:
    #   1. sum(X) = 1
    #   2. For j = 1 … m-1: n_j / n_0 = r_j / r_0
    #      <=> sum_i X_i * (A[i,j] * r[0] - A[i,0] * r[j]) = 0
    eq_constraints: list[dict] = [
        {"type": "eq", "fun": lambda X: float(X.sum()) - 1.0},
    ]
    for j in range(1, len(constrained_elements)):
        coeff = A[:, j] * r[0] - A[:, 0] * r[j]
        eq_constraints.append(
            {"type": "eq", "fun": lambda X, c=coeff: float(np.dot(X, c))}
        )

    bounds = [(0.0, 1.0)] * n

    result = minimize(
        objective,
        X0,
        jac=jac,
        method="SLSQP",
        bounds=bounds,
        constraints=eq_constraints,
        options={"ftol": 1e-12, "maxiter": 2000},
    )

    satisfied = bool(result.success)

    if not satisfied and constraint.strict:
        raise ConstraintInfeasibleError(
            f"Surface-element constraint {constraint.ratios} could not be "
            f"satisfied for species {names}. Solver message: {result.message}"
        )

    X_out = np.clip(result.x, 0.0, None)
    total = X_out.sum()
    if total > 0.0:
        X_out /= total
    else:
        X_out = X0.copy()
        satisfied = False

    return dict(zip(names, X_out.tolist())), satisfied


# ---------------------------------------------------------------------------
# Convenience: check constraint residual on an existing X_l
# ---------------------------------------------------------------------------


def constraint_residual(
    X_l: dict[str, float],
    species_elements: dict[str, dict[str, float]],
    constraint: SurfaceElementConstraint,
) -> dict[str, float]:
    """Return the actual elemental molar fractions in the condensed layer.

    Useful for verifying that a constraint is satisfied after application.

    Returns
    -------
    dict mapping element symbol → molar fraction in the condensed layer.
    """
    names = list(X_l.keys())
    constrained_elements = list(constraint.ratios.keys())

    n_j: dict[str, float] = {el: 0.0 for el in constrained_elements}
    for name in names:
        x = X_l.get(name, 0.0)
        comp = species_elements.get(name, {})
        for el in constrained_elements:
            n_j[el] += x * comp.get(el, 0.0)

    total = sum(n_j.values())
    if total > 0.0:
        return {el: v / total for el, v in n_j.items()}
    return {el: 0.0 for el in constrained_elements}
