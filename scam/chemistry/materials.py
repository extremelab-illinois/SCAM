"""Material data model for the SCAM equilibrium chemistry module.

Provides data classes that describe surface materials, their condensed
phases, pyrolysis gases, and boundary-layer states.  These are the
building blocks for the MAT-style element flux balance (Phase 1 of the
development roadmap).

All quantities are in SI units.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Elemental and species descriptors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ElementSet:
    """Ordered set of chemical element symbols tracked in the surface balance.

    Example::

        ElementSet(symbols=["C", "O", "N", "Si"])
    """

    symbols: tuple[str, ...]

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for s in self.symbols:
            if not s or not s[0].isupper():
                raise ValueError(
                    f"Element symbol {s!r} must start with an uppercase letter."
                )
            if s in seen:
                raise ValueError(f"Duplicate element symbol: {s!r}")
            seen.add(s)

    def __len__(self) -> int:
        return len(self.symbols)

    def index(self, symbol: str) -> int:
        return list(self.symbols).index(symbol)

    @classmethod
    def from_list(cls, symbols: list[str]) -> "ElementSet":
        return cls(symbols=tuple(symbols))


@dataclass(frozen=True)
class CondensedSpecies:
    """A single condensed-phase species that may be present at the surface.

    Attributes
    ----------
    name:
        Cantera species name or a label for a custom phase.
    molecular_weight_kg_per_kmol:
        Molar mass in kg/kmol.
    elements:
        Mapping of element symbol → number of atoms per formula unit
        (e.g. ``{"Si": 1, "C": 1}`` for SiC).
    failure_temperature_K:
        Temperature above which this species is assumed to fail/melt/spall.
        ``None`` means no failure model.
    density_kg_per_m3:
        Bulk density.  Optional; used for volumetric failure-rate models.
    """

    name: str
    molecular_weight_kg_per_kmol: float
    elements: dict[str, float]
    failure_temperature_K: float | None = None
    density_kg_per_m3: float | None = None


@dataclass(frozen=True)
class PyrolysisGas:
    """Composition and blowing rate specification for a pyrolysis stream.

    Attributes
    ----------
    composition_mass:
        Mass-fraction composition string (Cantera format, e.g.
        ``"CH4:0.46,CO:0.35,H2O:0.19"``).  Exactly one of
        *composition_mass* or *composition_mole* must be provided.
    composition_mole:
        Mole-fraction composition string.
    bg_values:
        B'g values at which to evaluate this stream.  An empty sequence
        means only B'g = 0 is used.
    """

    composition_mass: str | None = None
    composition_mole: str | None = None
    bg_values: tuple[float, ...] = (0.0,)

    def __post_init__(self) -> None:
        if (self.composition_mass is None) == (self.composition_mole is None):
            raise ValueError(
                "Exactly one of composition_mass or composition_mole must be set."
            )


# ---------------------------------------------------------------------------
# Surface material card
# ---------------------------------------------------------------------------


@dataclass
class SurfaceMaterial:
    """Complete description of a TPS surface material.

    A material card bundles the gas mechanism, condensed-phase species,
    and optional pyrolysis gas into a single transferable object that
    can be passed to the solver or table generator.

    Attributes
    ----------
    name:
        Human-readable label (e.g. ``"carbon_char"``).
    gas_mechanism:
        Path or Cantera built-in name for the gas-phase kinetics file.
    gas_phase_name:
        Named phase inside *gas_mechanism* (for multi-phase YAML files).
        Empty string means use the default phase.
    condensed_species:
        Ordered list of condensed species present at the surface.
    active_elements:
        Elements tracked in the surface mass balance.  Defaults to the
        union of elements in all *condensed_species*.
    pyrolysis_gas:
        Optional pyrolysis gas specification.
    description:
        Free-text description.
    """

    name: str
    gas_mechanism: str
    condensed_species: list[CondensedSpecies] = field(default_factory=list)
    gas_phase_name: str = ""
    active_elements: list[str] = field(default_factory=list)
    pyrolysis_gas: PyrolysisGas | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not self.active_elements and self.condensed_species:
            seen: list[str] = []
            for sp in self.condensed_species:
                for el in sp.elements:
                    if el not in seen:
                        seen.append(el)
            self.active_elements = seen

    @property
    def primary_condensed_phase_file(self) -> str:
        """Return the gas_mechanism path (also used for condensed phases
        when both are defined in the same multi-phase YAML)."""
        return self.gas_mechanism

    @property
    def condensed_phase_names(self) -> list[str]:
        return [sp.name for sp in self.condensed_species]


# ---------------------------------------------------------------------------
# Boundary-layer state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundaryLayerState:
    """Thermodynamic state of the boundary-layer edge.

    Attributes
    ----------
    temperature_K:
        Edge total temperature.
    pressure_Pa:
        Local pressure (assumed uniform across the boundary layer).
    composition_mass:
        Edge gas mass-fraction string.
    composition_mole:
        Edge gas mole-fraction string.  Exactly one of *composition_mass*
        or *composition_mole* must be set.
    """

    temperature_K: float
    pressure_Pa: float
    composition_mass: str | None = None
    composition_mole: str | None = None

    def __post_init__(self) -> None:
        if (self.composition_mass is None) == (self.composition_mole is None):
            raise ValueError(
                "Exactly one of composition_mass or composition_mole must be set."
            )


# ---------------------------------------------------------------------------
# MAT solver control point and result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MATControlPoint:
    """One (T_wall, p, B'g) point at which the MAT solver is evaluated."""

    T_wall_K: float
    pressure_Pa: float
    Bg: float = 0.0


@dataclass
class MATResult:
    """Result at a single MAT control point.

    Populated incrementally by the solver; fields left as ``None`` were
    not computed (e.g. if finite-rate reactions are disabled).

    Attributes
    ----------
    control_point:
        The (T, p, B'g) input.
    converged:
        Whether the solver converged.
    Bprime_c:
        Ablation blowing parameter for the primary condensed element.
    Bprime_g:
        Pyrolysis gas blowing parameter (equals ``control_point.Bg``).
    Bprime_fail:
        Failure/melt/spall blowing parameter.
    Bprime_total:
        Sum of all blowing contributions.
    wall_gas_composition_X:
        Wall gas mole-fraction string.
    h_wall_gas_J_kg:
        Wall gas specific enthalpy (J/kg).
    MW_wall_gas_kg_per_kmol:
        Wall gas mean molecular weight.
    condensed_surface_mole_fractions:
        Mole fractions of each condensed species at the surface.
    active_reactions:
        Names of heterogeneous reactions that are active.
    reaction_rates:
        Reaction rates (kmol/m²/s) for each active reaction.
    convergence_residual:
        Final residual norm from the Newton solver.
    """

    control_point: MATControlPoint
    converged: bool = False

    Bprime_c: float | None = None
    Bprime_g: float | None = None
    Bprime_fail: float | None = None
    Bprime_total: float | None = None

    wall_gas_composition_X: str | None = None
    h_wall_gas_J_kg: float | None = None
    MW_wall_gas_kg_per_kmol: float | None = None

    condensed_surface_mole_fractions: dict[str, float] | None = None
    active_reactions: list[str] | None = None
    reaction_rates: dict[str, float] | None = None

    convergence_residual: float | None = None


# ---------------------------------------------------------------------------
# YAML material card loader
# ---------------------------------------------------------------------------


def _parse_condensed_species(raw: list[dict[str, Any]]) -> list[CondensedSpecies]:
    out = []
    for item in raw:
        out.append(
            CondensedSpecies(
                name=item["name"],
                molecular_weight_kg_per_kmol=float(item["molecular_weight_kg_per_kmol"]),
                elements={str(k): float(v) for k, v in item["elements"].items()},
                failure_temperature_K=item.get("failure_temperature_K"),
                density_kg_per_m3=item.get("density_kg_per_m3"),
            )
        )
    return out


def _parse_pyrolysis_gas(raw: dict[str, Any] | None) -> PyrolysisGas | None:
    if raw is None:
        return None
    return PyrolysisGas(
        composition_mass=raw.get("composition_mass"),
        composition_mole=raw.get("composition_mole"),
        bg_values=tuple(float(v) for v in raw.get("bg_values", [0.0])),
    )


def load_material_card(path: str | Path) -> SurfaceMaterial:
    """Load a SurfaceMaterial from a YAML file.

    Expected YAML structure::

        name: carbon_char
        gas_mechanism: cno_ablation.yaml
        gas_phase_name: ""          # optional
        description: "Carbon char in air"
        condensed_species:
          - name: C(gr)
            molecular_weight_kg_per_kmol: 12.011
            elements: {C: 1}
            failure_temperature_K: null
        pyrolysis_gas:              # optional
          composition_mass: "CH4:0.46,CO:0.35,H2O:0.19"
          bg_values: [0.0, 0.5, 1.0]
        active_elements: [C, O, N]  # optional; inferred if omitted
    """
    data = yaml.safe_load(Path(path).read_text())
    return SurfaceMaterial(
        name=data["name"],
        gas_mechanism=data["gas_mechanism"],
        gas_phase_name=data.get("gas_phase_name", ""),
        condensed_species=_parse_condensed_species(data.get("condensed_species", [])),
        active_elements=list(data.get("active_elements", [])),
        pyrolysis_gas=_parse_pyrolysis_gas(data.get("pyrolysis_gas")),
        description=data.get("description", ""),
    )
