# SPDX-License-Identifier: MIT
"""Load and map the surface_bc.csv boundary condition file to hot-face cells.

surface_bc.csv format (AblationTestCase_3.x convention)
---------------------------------------------------------
Columns:
    s         arc length from nose stagnation point [cm]
    Y         radial coordinate [cm]  (0 = axis, 5 cm = r_cyl = 50 mm)
    Z         axial coordinate [cm]   (Z=0 at nose, Z<0 downward)
    q_w/q_w0  heat-transfer-coefficient ratio  (1.0 at stagnation)
    p_w/p_w0  pressure ratio                   (1.0 at stagnation)

All centimetre values are converted to metres on load.

The "axial" coordinate Z in the CSV is measured DOWN from the nose, while
the mesh uses h measured UP from the back face.  Conversion::

    h = h_nose + Z   where h_nose = 2 * r_cyl (nose tip axial position)
    e.g. Z=0 → h=h_nose=0.1 m;  Z=-9.992 cm → h=h_nose-0.09992 ≈ 0.00008 m

**Usage**

::

    from scam.io.surface_map import load_surface_map, map_to_hot_faces

    tbl = load_surface_map("examples/verification/ablation3/surface_bc.csv")
    rhoUeCH_faces, p_e_faces = map_to_hot_faces(tbl, mesh, rhoUeCH_ref=0.1, p_ref=405.3)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass
class SurfaceBCTable:
    """Parsed surface_bc.csv data in SI units."""

    s: NDArray          #: (N,) arc length from nose [m]
    r: NDArray           #: (N,) radial coordinate [m]
    h: NDArray           #: (N,) axial coordinate in mesh frame [m] (0 at back, h_nose at nose)
    q_ratio: NDArray     #: (N,) ``q_w / q_w(0)`` -- heat-flux / CH ratio
    p_ratio: NDArray     #: (N,) ``p_w / p_w(0)`` -- pressure ratio
    h_nose: float        #: axial position of the nose tip in mesh frame [m]


def load_surface_map(
    csv_path: str | Path,
    r_cyl: float = 0.05,
) -> SurfaceBCTable:
    """Load and parse surface_bc.csv.

    Parameters
    ----------
    csv_path : path to surface_bc.csv
    r_cyl : cylinder radius [m] (sets h_nose = 2 * r_cyl)

    Returns
    -------
    SurfaceBCTable in SI units
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"surface_bc.csv not found: {path}")

    data = np.genfromtxt(
        path,
        delimiter=None,
        skip_header=1,
        names=True,
        dtype=np.float64,
    )

    # Column names may vary; try standard names
    def _col(names, *candidates):
        for c in candidates:
            if c in names:
                return data[c]
        raise KeyError(f"None of {candidates} found in CSV header: {names}")

    names = data.dtype.names
    s_cm    = _col(names, "s", "arc_length", "S")
    Y_cm    = _col(names, "Y", "r", "R")
    Z_cm    = _col(names, "Z", "z")
    q_ratio = _col(names, "qwqw0", "q_wq_w0", "qw_ratio", "q_ratio")
    p_ratio = _col(names, "pwpw0", "p_wp_w0", "pw_ratio", "p_ratio")

    h_nose = 2.0 * r_cyl   # nose tip axial position in mesh frame [m]

    # Convert cm → m
    s = s_cm * 1e-2
    r = Y_cm * 1e-2
    # Z=0 at nose, negative downward → h = h_nose + Z (in metres)
    h = h_nose + Z_cm * 1e-2

    return SurfaceBCTable(
        s=s, r=r, h=h,
        q_ratio=q_ratio, p_ratio=p_ratio,
        h_nose=h_nose,
    )


def map_to_hot_faces(
    tbl: SurfaceBCTable,
    mesh,
    rhoUeCH_ref: float | NDArray,
    p_ref: float | NDArray,
) -> tuple[NDArray, NDArray]:
    """Interpolate surface BC table to hot-face cell centres by arc length.

    Parameters
    ----------
    tbl : SurfaceBCTable
    mesh : IsoQMesh
    rhoUeCH_ref : reference ρ_e u_e C_H at stagnation [kg/m²/s]
    p_ref : reference surface pressure at stagnation [Pa]

    Returns
    -------
    rhoUeCH_faces : (Fh,) [kg/m²/s]  per hot-face cell
    p_e_faces : (Fh,) [Pa]  per hot-face cell
    """
    Cf = mesh.hot_Cf   # (Fh, 2) face centres (r, h)
    r_faces = Cf[:, 0]
    h_faces = Cf[:, 1]

    # Compute arc length along hot face from nose tip to each face centre
    # Face centres are ordered: nose → shoulder → body in the mesh topology.
    # Compute cumulative arc length from the table-defined stagnation point.
    # Simple approach: find nearest table point by (r,h) distance, then use s.
    s_faces = _arc_length_by_distance(tbl, r_faces, h_faces)

    # Interpolate ratios at face arc-length positions
    q_ratio_f = np.interp(s_faces, tbl.s, tbl.q_ratio, left=tbl.q_ratio[0], right=tbl.q_ratio[-1])
    p_ratio_f = np.interp(s_faces, tbl.s, tbl.p_ratio, left=tbl.p_ratio[0], right=tbl.p_ratio[-1])

    rhoUeCH_faces = np.asarray(rhoUeCH_ref) * q_ratio_f
    p_e_faces = np.asarray(p_ref) * p_ratio_f

    return rhoUeCH_faces, p_e_faces


def _arc_length_by_distance(
    tbl: SurfaceBCTable,
    r_query: NDArray,
    h_query: NDArray,
) -> NDArray:
    """For each query point (r,h), find the closest table point and return its s.

    Falls back to nearest-neighbour when the query is off-arc.  For smooth
    body-fitted meshes this gives a very good interpolation of s(face_centre).
    """
    s_out = np.empty(len(r_query), dtype=np.float64)
    for i, (ri, hi) in enumerate(zip(r_query, h_query)):
        dist = np.sqrt((tbl.r - ri)**2 + (tbl.h - hi)**2)
        idx = int(np.argmin(dist))
        s_out[i] = tbl.s[idx]
    return s_out
