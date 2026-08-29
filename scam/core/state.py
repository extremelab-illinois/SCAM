# SPDX-License-Identifier: MIT
"""Simulation state dataclasses.

MeshState holds all geometric information.
SimState holds the full simulation state at one instant in time.
Both are designed to be copied rather than mutated in-place; individual solvers
return new (or modified) SimState objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class MeshState:
    """Geometric description of the current computational mesh.

    Coordinate convention:
    - y = 0 at the original (t=0) surface position.
    - y increases into the material (away from the hot face).
    - The ablating surface is currently at y = s_total.

    Node indexing is global across all layers (layer 0 is the surface layer).
    Within the ablating (surface) layer, node 0 is always the surface-most node.
    """

    n_nodes_total: int
    # Node center positions in the fixed material frame [m], shape (N,)
    y_nodes: np.ndarray
    # Nominal cell thickness for each node [m], shape (N,).
    # For the surface node: updated each step to reflect the shrinking cell.
    # For back node: half the inter-node spacing.
    delta_nodes: np.ndarray
    # Area function A(y_n) evaluated at node positions [m^2], shape (N,)
    area_nodes: np.ndarray
    # Layer index for each global node, dtype int, shape (N,)
    layer_id: np.ndarray
    # Global index of the ablating surface node (always 0 after any remap, kept explicit)
    shrinking_node_idx: int
    # Cumulative surface recession distance [m] (increases over time)
    s_total: float

    # Per-layer nodelet geometry (list indexed by layer index).
    # Entry l is None for non-decomposing layers.
    # Otherwise ndarray shape (n_nodes_in_layer_l, J_l) giving nodelet center y-positions [m].
    nodelet_y: list
    # Nodelet thicknesses [m], same structure as nodelet_y.
    nodelet_delta: list
    # List of (first_global_idx, last_global_idx+1) index ranges per layer.
    # Use layer_boundaries[l] to slice global arrays for layer l.
    layer_boundaries: list
    # Mesh grading ratio per layer: back_cell_size / surface_cell_size.
    # 1.0 = uniform; >1.0 = finer near hot face.  Set at build time; preserved through remaps.
    layer_gradings: list = field(default_factory=list)


@dataclass
class SimState:
    """Full simulation state at time ``time``.

    Arrays are allocated once at initialisation and updated in-place by solvers,
    except after a node-drop remap (which rebuilds MeshState and reallocates arrays).
    """

    time: float          # current simulation time [s]
    dt: float            # current timestep [s]

    # Nodal temperatures [K], shape (N,)
    T: np.ndarray
    # Volume-averaged mixture density [kg/m^3], shape (N,)
    rho: np.ndarray
    # Per-layer component density arrays on the decomposition subgrid.
    # rho_components[l] has shape (n_comp_l, n_nodes_l, J_l) or None if layer l is inert.
    rho_components: list

    # Surface state (scalars, updated by the surface solver each step)
    s_dot: float = 0.0          # surface recession rate [m/s]
    T_wall: float = 300.0       # surface temperature [K]
    q_cond: float = 0.0         # net conductive flux into material from surface [W/m^2]
    m_dot_pyro: float = 0.0     # total pyrolysis gas mass flux at surface [kg/m^2/s]
    m_dot_char: float = 0.0     # char erosion mass flux [kg/m^2/s]
    # Density at the surface node when s_dot was computed (pre-recession/drop).
    # snap.rho[0] may differ after apply_recession merges nodes; use this for SMB check.
    rho_wall: float = 0.0

    # Temperature field from the previous completed step [K], shape (N,) or None.
    # Used to estimate dT/dt for the gas expansion (Darcy) advection term.
    # None on the first step (Darcy contribution is zero on step 0).
    T_prev: Optional[np.ndarray] = None

    # Cumulative conservation trackers — updated every timestep (not just at output steps)
    # so that conservation_check gives exact per-interval residuals regardless of output_dt.
    cum_mass_out: float = 0.0    # ∫(m_dot_char + m_dot_pyro)*A_surface*dt  [kg]  (CMA model)
    cum_mass_exact: float = 0.0  # actual domain mass loss: ∑Δ(ρ·V) per step [kg]  (exact)
    cum_energy_in: float = 0.0   # ∫q_cond*A_surface*dt  [J]  (surface conduction only)

    # Elemental mass fractions in the porous gas phase, shape (4, N).
    # Index order: 0=C, 1=H, 2=O, 3=N.
    # None when element transport is disabled (default).
    Z_elem: Optional[np.ndarray] = None

    mesh: MeshState = field(default_factory=lambda: MeshState(
        n_nodes_total=0,
        y_nodes=np.array([]),
        delta_nodes=np.array([]),
        area_nodes=np.array([]),
        layer_id=np.array([], dtype=int),
        shrinking_node_idx=0,
        s_total=0.0,
        nodelet_y=[],
        nodelet_delta=[],
        layer_boundaries=[],
    ))
