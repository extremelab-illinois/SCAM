# SPDX-License-Identifier: MIT
"""Geometry configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np


class GeometryType(Enum):
    SLAB = auto()            # A(y) = 1.0 (per unit cross-sectional area)
    HOLLOW_CYLINDER = auto() # A(y) = r_inner + y  (cylindrical shell, area grows with depth)
    TABULATED = auto()       # A(y) from user-supplied table


@dataclass
class GeometryConfig:
    """Geometry configuration for the area function A(y).

    y is measured from the original surface position into the material.
    For HOLLOW_CYLINDER the area grows with depth (inner radius at hot face).
    """

    geometry_type: GeometryType = GeometryType.SLAB

    # For HOLLOW_CYLINDER: inner radius at the original surface position [m]
    r_inner: float = 0.0

    # For TABULATED: shape (N, 2) array [[y_m, A_m2], ...]
    area_table: Optional[np.ndarray] = None
