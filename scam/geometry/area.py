# SPDX-License-Identifier: MIT
"""Area function A(y) for different geometries.

y is measured from the original surface position into the material [m].
For slab geometry A(y) = 1 (per unit cross-sectional area).
For hollow-cylinder A(y) = r_inner + y (circumferential area grows with depth).
Tabulated geometry uses linear interpolation.

All functions return dimensioned areas [m^2] but for slab the convention is
A = 1.0 (dimensionless, equivalent to unit area) so that the energy equation
yields flux densities in [W/m^2] directly.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

from scam.config.geometry import GeometryConfig, GeometryType
from scam.core.errors import SCAMInputError


def build_area_function(cfg: GeometryConfig):
    """Return a callable ``A(y)`` that accepts a numpy array and returns areas.

    Parameters
    ----------
    cfg:
        GeometryConfig describing the domain geometry.

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        A vectorised function A(y) with the same shape as the input.
    """
    if cfg.geometry_type == GeometryType.SLAB:
        def _slab(y: np.ndarray) -> np.ndarray:
            return np.ones_like(np.asarray(y, dtype=float))
        return _slab

    elif cfg.geometry_type == GeometryType.HOLLOW_CYLINDER:
        r = float(cfg.r_inner)
        def _cylinder(y: np.ndarray) -> np.ndarray:
            return r + np.asarray(y, dtype=float)
        return _cylinder

    elif cfg.geometry_type == GeometryType.TABULATED:
        if cfg.area_table is None:
            raise SCAMInputError("GeometryType.TABULATED requires area_table to be set")
        tbl = np.asarray(cfg.area_table, dtype=float)
        if tbl.ndim != 2 or tbl.shape[1] != 2:
            raise SCAMInputError("area_table must have shape (N, 2): [[y, A], ...]")
        if not np.all(np.diff(tbl[:, 0]) > 0):
            raise SCAMInputError("area_table y-values must be strictly increasing")
        _interp = interp1d(
            tbl[:, 0], tbl[:, 1],
            kind="linear", bounds_error=False,
            fill_value=(tbl[0, 1], tbl[-1, 1]),
        )
        def _tabulated(y: np.ndarray) -> np.ndarray:
            return _interp(np.asarray(y, dtype=float))
        return _tabulated

    else:
        raise SCAMInputError(f"Unknown GeometryType: {cfg.geometry_type}")


def area_at(y: np.ndarray, cfg: GeometryConfig) -> np.ndarray:
    """Convenience wrapper: evaluate A(y) for the given geometry."""
    return build_area_function(cfg)(y)
