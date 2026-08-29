"""Material failure model for surface equilibrium chemistry (Phase 7).

Implements the simple step-function failure model: a condensed species that
has a ``failure_temperature_K`` is assumed to fail (melt/spall/actively oxidize)
when the wall temperature exceeds that threshold.

The failure blowing parameter B'_fail is computed as:

    B'_fail = sum_l  X_l   for all l where  T_wall > T_fail,l

This is a dimensionless quantity (sum of surface mole fractions of failing
species) that can be used as a qualitative indicator of the passive-to-active
transition.  A full quantitative model (bounded iterative maximum failure rate
per Phase 7+ roadmap) would replace this with a proper mass-flux calculation.

All quantities are in SI units or dimensionless.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FailureState:
    """Result of the failure model at a single (T_wall, X_l) point.

    Attributes
    ----------
    failing_species:
        Names of condensed species whose failure temperature is exceeded.
    Bprime_fail:
        Dimensionless failure blowing parameter:
        sum of surface mole fractions of all failing species.
    active:
        ``True`` when at least one species is failing.
    """

    failing_species: tuple[str, ...]
    Bprime_fail: float
    active: bool

    @property
    def is_passive(self) -> bool:
        return not self.active


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_failure_state(
    condensed_species: "list",
    T_wall_K: float,
    surface_mole_fractions: dict[str, float],
) -> FailureState:
    """Determine which condensed species are failing and compute B'_fail.

    Parameters
    ----------
    condensed_species:
        List of ``CondensedSpecies`` objects (from ``scam.chemistry.materials``).
        Only species with a non-``None`` ``failure_temperature_K`` are
        evaluated; others are always assumed stable.
    T_wall_K:
        Wall temperature (K).
    surface_mole_fractions:
        Current surface mole fractions X_l keyed by species name.
        Species absent from this dict are treated as X_l = 0.

    Returns
    -------
    FailureState
        With ``failing_species``, ``Bprime_fail``, and ``active`` populated.

    Notes
    -----
    Simple step model: a species fails if and only if
    ``T_wall_K > species.failure_temperature_K``.  The contribution to
    B'_fail is proportional to its surface mole fraction.
    """
    failing: list[str] = []
    bprime_fail = 0.0

    for sp in condensed_species:
        t_fail = sp.failure_temperature_K
        if t_fail is None:
            continue
        if T_wall_K > t_fail:
            x_l = surface_mole_fractions.get(sp.name, 0.0)
            failing.append(sp.name)
            bprime_fail += max(x_l, 0.0)

    return FailureState(
        failing_species=tuple(failing),
        Bprime_fail=bprime_fail,
        active=len(failing) > 0,
    )
