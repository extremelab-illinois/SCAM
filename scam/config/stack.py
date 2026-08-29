# SPDX-License-Identifier: MIT
"""Layer and stack configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LayerConfig:
    """Configuration for one layer in the material stack.

    layers[0] in a StackConfig is the surface (hot-face) layer.
    """

    material_name: str      # key into the material registry (matches MaterialCard.name)
    thickness: float        # initial layer thickness [m]
    n_nodes: int            # number of computational nodes in this layer (including boundaries)
    n_subcells: int = 4     # nodelets per thermal node for decomposition subgrid (must be >= 2, even)
    # Thermal contact resistance at the interface BEFORE this layer [m^2 K/W].
    # Only meaningful for layers[1] and beyond; ignored for layers[0].
    contact_resistance: float = 0.0
    # Mesh grading: ratio of back-face cell size to surface cell size.
    #   1.0  → uniform spacing (default)
    #   >1.0 → cells expand going into material (finer near hot face) — typical for ablation
    #   <1.0 → cells shrink going into material (finer near back face)
    # Equivalent to 1/simpleGrading in PATO's blockMeshDict (which uses the opposite direction).
    grading: float = 1.0


@dataclass
class StackConfig:
    """Ordered list of layers forming the material stack.

    layers[0] is the surface (hot-face) layer.
    layers[-1] is the back-face layer.
    """

    layers: list = field(default_factory=list)  # list[LayerConfig]
