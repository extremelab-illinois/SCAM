"""Heterogeneous surface reaction model for surface equilibrium chemistry (Phase 8).

Implements finite-rate heterogeneous reactions at the ablating surface using
Arrhenius kinetics.  Each ``HeterogeneousReaction`` carries the stoichiometry
and rate parameters; ``compute_reaction_rate`` evaluates the rate at a given
wall state.

Rate law (modified Arrhenius):

    k = A * T^beta * exp(-Ea / (R * T))
    rate = k * product_i (X_i ^ n_i)   [kmol / m² / s]

where the product is over gas-phase reactant species.

Carbon oxidation examples (from Park / Zhluktov & Abe literature):

    C(s) + O2  → CO2       (low T, passive regime)
    C(s) + 1/2 O2 → CO     (dominant above ~1200 K)
    C(s) + O   → CO        (atomic oxygen attack)

These are provided as the ``CARBON_OXIDATION_REACTIONS`` convenience list.

All quantities use SI units: T in K, P in Pa, rates in kmol/m²/s, activation
energies in J/kmol.

References
----------
Park, C. (1976). "Viscous shock layer calculation..."
Zhluktov, S.V., Abe, T. (1999). "Viscous Shock-Layer Simulation..."
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Universal gas constant J / (kmol · K)
_R = 8314.46261815324


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeterogeneousReaction:
    """One finite-rate heterogeneous surface reaction."""

    name: str                            #: human-readable label (e.g. ``"C+O2->CO2"``)
    #: Mapping of gas-phase species name -> stoichiometric coefficient
    #: (used in the rate product), e.g. ``{"O2": 1.0}``.
    gas_reactants: dict[str, float]
    #: Name of the condensed-phase species consumed (informational only;
    #: does not affect the rate calculation).
    condensed_reactant: str
    #: Mapping of gas-phase product species -> stoichiometric coefficient
    #: (informational only).
    gas_products: dict[str, float]
    #: Pre-exponential factor (units depend on reaction order; consistent
    #: with the rate being in kmol/m^2/s when mole fractions are used).
    A: float
    beta: float = 0.0                    #: temperature exponent in ``T^beta`` (dimensionless)
    Ea_J_per_kmol: float = 0.0            #: activation energy [J/kmol]; 0 for a barrierless reaction
    #: Whether the reaction is reversible. Currently always treated as
    #: irreversible; flag reserved for future equilibrium-rate extension.
    reversible: bool = False
    #: Overall reaction order in gas-phase species; 0 means the rate
    #: depends only on temperature (zero-order, wall-limited).
    reaction_order: float = 1.0

    def __post_init__(self) -> None:
        if self.A < 0.0:
            raise ValueError(f"Pre-exponential A must be non-negative, got {self.A}.")


# ---------------------------------------------------------------------------
# Rate evaluation
# ---------------------------------------------------------------------------


def compute_reaction_rate(
    reaction: HeterogeneousReaction,
    T_K: float,
    species_mole_fractions: dict[str, float],
) -> float:
    """Evaluate the surface reaction rate at temperature T_K.

    Parameters
    ----------
    reaction:
        Reaction definition with Arrhenius parameters.
    T_K:
        Wall temperature (K).  Must be positive.
    species_mole_fractions:
        Gas-phase mole fractions at the wall, keyed by species name.
        Species not present in this dict are treated as X = 0.

    Returns
    -------
    float
        Reaction rate in kmol/m²/s.  Returns 0 when any reactant mole
        fraction is zero (or negative) and the reaction order is > 0.
    """
    if T_K <= 0.0:
        raise ValueError(f"Temperature must be positive, got {T_K} K.")

    k = reaction.A * (T_K ** reaction.beta) * math.exp(-reaction.Ea_J_per_kmol / (_R * T_K))

    if reaction.reaction_order == 0.0:
        return k

    rate = k
    for sp, nu in reaction.gas_reactants.items():
        x = max(species_mole_fractions.get(sp, 0.0), 0.0)
        if x <= 0.0:
            return 0.0
        rate *= x ** nu

    return rate


def compute_surface_reaction_rates(
    reactions: list[HeterogeneousReaction],
    T_K: float,
    species_mole_fractions: dict[str, float],
) -> dict[str, float]:
    """Evaluate rates for a list of surface reactions.

    Parameters
    ----------
    reactions:
        List of heterogeneous reactions to evaluate.
    T_K:
        Wall temperature (K).
    species_mole_fractions:
        Gas-phase mole fractions at the wall.

    Returns
    -------
    dict
        Mapping reaction name → rate (kmol/m²/s).
    """
    return {r.name: compute_reaction_rate(r, T_K, species_mole_fractions) for r in reactions}


def active_reactions(
    rates: dict[str, float],
    threshold: float = 1e-20,
) -> list[str]:
    """Return names of reactions whose rate exceeds *threshold*."""
    return [name for name, rate in rates.items() if rate > threshold]


# ---------------------------------------------------------------------------
# Built-in carbon oxidation reactions
# ---------------------------------------------------------------------------

#: Carbon oxidation reactions (Arrhenius parameters from literature).
#: These are approximate values suitable for qualitative comparison.
#: A in units giving rate in kmol/m²/s with X_i in mole fractions.
CARBON_OXIDATION_REACTIONS: list[HeterogeneousReaction] = [
    HeterogeneousReaction(
        name="C+O2->CO2",
        gas_reactants={"O2": 1.0},
        condensed_reactant="C(gr)",
        gas_products={"CO2": 1.0},
        A=1.0e-1,
        beta=0.0,
        Ea_J_per_kmol=1.67e8,   # ~167 kJ/mol
        reversible=False,
    ),
    HeterogeneousReaction(
        name="C+0.5O2->CO",
        gas_reactants={"O2": 1.0},
        condensed_reactant="C(gr)",
        gas_products={"CO": 1.0},
        A=3.6e-1,
        beta=0.0,
        Ea_J_per_kmol=1.25e8,   # ~125 kJ/mol
        reversible=False,
    ),
    HeterogeneousReaction(
        name="C+O->CO",
        gas_reactants={"O": 1.0},
        condensed_reactant="C(gr)",
        gas_products={"CO": 1.0},
        A=2.0e0,
        beta=0.0,
        Ea_J_per_kmol=0.0,      # barrierless at high T
        reversible=False,
    ),
]
