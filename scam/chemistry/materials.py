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
    """A single condensed-phase species that may be present at the surface."""

    name: str                                    #: Cantera species name, or a label for a custom phase
    molecular_weight_kg_per_kmol: float          #: molar mass [kg/kmol]
    #: Mapping of element symbol -> number of atoms per formula unit
    #: (e.g. ``{"Si": 1, "C": 1}`` for SiC).
    elements: dict[str, float]
    #: Temperature above which this species is assumed to fail/melt/spall.
    #: None means no failure model.
    failure_temperature_K: float | None = None
    density_kg_per_m3: float | None = None       #: bulk density; optional, used for volumetric failure-rate models


@dataclass(frozen=True)
class PyrolysisGas:
    """Composition and blowing rate specification for a pyrolysis stream.

    Exactly one of ``composition_mass``/``composition_mole`` must be set
    (enforced in ``__post_init__``).
    """

    #: Mass-fraction composition string (Cantera format, e.g.
    #: ``"CH4:0.46,CO:0.35,H2O:0.19"``).
    composition_mass: str | None = None
    composition_mole: str | None = None          #: mole-fraction composition string
    #: B'g values at which to evaluate this stream; empty means only B'g = 0.
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
    """

    name: str                 #: human-readable label (e.g. ``"carbon_char"``)
    gas_mechanism: str        #: path or Cantera built-in name for the gas-phase kinetics file
    condensed_species: list[CondensedSpecies] = field(default_factory=list)  #: ordered list of condensed species present at the surface
    #: Named phase inside ``gas_mechanism`` (for multi-phase YAML files);
    #: empty string means use the default phase.
    gas_phase_name: str = ""
    #: Elements tracked in the surface mass balance; defaults to the
    #: union of elements in all ``condensed_species``.
    active_elements: list[str] = field(default_factory=list)
    pyrolysis_gas: PyrolysisGas | None = None     #: optional pyrolysis gas specification
    description: str = ""                          #: free-text description

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

    Exactly one of ``composition_mass``/``composition_mole`` must be set
    (enforced in ``__post_init__``).
    """

    temperature_K: float          #: edge total temperature [K]
    pressure_Pa: float            #: local pressure (assumed uniform across the boundary layer) [Pa]
    composition_mass: str | None = None    #: edge gas mass-fraction string
    composition_mole: str | None = None    #: edge gas mole-fraction string

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
    """

    control_point: MATControlPoint     #: the (T, p, B'g) input
    converged: bool = False             #: whether the solver converged

    Bprime_c: float | None = None       #: ablation blowing parameter for the primary condensed element
    Bprime_g: float | None = None       #: pyrolysis gas blowing parameter (equals ``control_point.Bg``)
    Bprime_fail: float | None = None    #: failure/melt/spall blowing parameter
    Bprime_total: float | None = None   #: sum of all blowing contributions

    wall_gas_composition_X: str | None = None      #: wall gas mole-fraction string
    h_wall_gas_J_kg: float | None = None            #: wall gas specific enthalpy [J/kg]
    MW_wall_gas_kg_per_kmol: float | None = None    #: wall gas mean molecular weight

    condensed_surface_mole_fractions: dict[str, float] | None = None   #: mole fractions of each condensed species at the surface
    active_reactions: list[str] | None = None                          #: names of heterogeneous reactions that are active
    reaction_rates: dict[str, float] | None = None                     #: reaction rates [kmol/m^2/s] for each active reaction

    convergence_residual: float | None = None       #: final residual norm from the Newton solver


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
