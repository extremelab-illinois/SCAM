# SPDX-License-Identifier: MIT
"""Top-level simulation case dataclass (maps 1:1 to a YAML input deck)."""

from __future__ import annotations

from dataclasses import dataclass, field

from scam.config.stack import StackConfig
from scam.config.geometry import GeometryConfig
from scam.config.boundary import SurfaceBCConfig, BackBCConfig
from scam.config.solver import SolverOptions


@dataclass
class SimCase:
    """Complete specification of a SCAM simulation case.

    This dataclass mirrors the structure of a YAML input deck and is the
    canonical input to :func:`scam.solvers.material_response.run`.
    """

    name: str
    stack: StackConfig
    geometry: GeometryConfig
    surface_bc: SurfaceBCConfig
    back_bc: BackBCConfig
    options: SolverOptions

    # Paths to material card YAML files (resolved at load time)
    material_paths: list = field(default_factory=list)   # list[str]

    # Output configuration
    output_path: str = "results/"
    output_format: str = "csv"   # "csv" or "hdf5"
