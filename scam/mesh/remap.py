# SPDX-License-Identifier: MIT
"""Conservative node-drop remap for the ablating surface layer.

When the surface cell (node 0) has shrunk to below the drop threshold,
this module:

1. Merges node 0 (thin remnant) with node 1 (the cell just behind it)
   using a volume-weighted average for T and rho_components.
2. Removes node 0 from all global arrays — node 1 becomes the new node 0.
3. Updates the MeshState (layer_boundaries, etc.) accordingly.
4. Does NOT change s_total; the surface position is fixed.

Node-drop is ALWAYS from the FRONT of the ablating layer (the shrinking
surface cell is merged into the cell behind it), which means the new node 0
inherits a near-nominal cell thickness.  Doing it this way (rather than
discarding the thin cell outright) prevents the non-conservative
temperature discontinuity and associated recession-rate oscillations.

Interpolate_to_nodelets
-----------------------
Temperature is interpolated from the coarse thermal grid to the
decomposition subgrid (nodelets) using cumulative volume as the
independent variable, NOT physical distance.  This is the correct
choice for cylindrical geometry where A(y) varies.
"""

from __future__ import annotations

import copy

import numpy as np

from scam.core.state import MeshState, SimState
from scam.geometry.fvm import nodelet_cumulative_volume


# ---------------------------------------------------------------------------
# Temperature → nodelet interpolation
# ---------------------------------------------------------------------------

def interpolate_to_nodelets(
    T_left: float,
    T_center: float,
    T_right: float,
    V_cum_nodelets: np.ndarray,
    V_cell: float,
) -> np.ndarray:
    """Interpolate a nodal temperature to J nodelet positions.

    Uses piecewise linear interpolation with cumulative volume as the
    coordinate (eq. 6.1.54 in the reference CMA thesis):

    - Nodelets j = 0 .. J//2 - 1 interpolate between (T_left, T_center)
    - Nodelets j = J//2 .. J-1  interpolate between (T_center, T_right)

    For the surface node (no left neighbour) T_left = T_center.
    For the back node (no right neighbour) T_right = T_center.

    Parameters
    ----------
    T_left, T_center, T_right:
        Temperatures of the neighbouring nodes [K].
    V_cum_nodelets:
        Cumulative volume from the start of the cell for each nodelet [m^3],
        shape (J,).  V_cum_nodelets[-1] ≈ V_cell.
    V_cell:
        Total cell volume [m^3].

    Returns
    -------
    np.ndarray
        Nodelet temperatures [K], shape (J,).
    """
    J    = len(V_cum_nodelets)
    half = J // 2
    T_out = np.empty(J)

    # Left half: interpolate from T_left to T_center
    V_half_L = V_cum_nodelets[half - 1] if half > 0 else 1.0
    xi_L = np.clip(V_cum_nodelets[:half] / V_half_L if V_half_L > 0 else np.full(half, 0.5), 0.0, 1.0)
    T_out[:half] = (1.0 - xi_L) * T_left + xi_L * T_center

    # Right half: interpolate from T_center to T_right
    V_half_R = V_cell - (V_cum_nodelets[half - 1] if half > 0 else 0.0)
    xi_num   = V_cum_nodelets[half:] - (V_cum_nodelets[half - 1] if half > 0 else 0.0)
    xi_R     = np.clip(xi_num / V_half_R if V_half_R > 0 else np.full(J - half, 0.5), 0.0, 1.0)
    T_out[half:] = (1.0 - xi_R) * T_center + xi_R * T_right

    return T_out


# ---------------------------------------------------------------------------
# Conservative nodal-average of nodelets
# ---------------------------------------------------------------------------

def nodelets_to_nodal_density(
    rho_comp_nodelets: np.ndarray,  # (n_comp, J)
    delta_nodelets: np.ndarray,     # (J,)
    area_nodelets: np.ndarray,      # (J,)
    n_comp: int,
) -> np.ndarray:
    """Volume-weighted average of nodelet densities to get nodal density per component.

    Returns array of shape (n_comp,).
    """
    V_nodelets = delta_nodelets * area_nodelets   # shape (J,)
    V_total = V_nodelets.sum()
    if V_total == 0:
        return rho_comp_nodelets.mean(axis=1) if rho_comp_nodelets.ndim > 1 else rho_comp_nodelets
    return (rho_comp_nodelets * V_nodelets[np.newaxis, :]).sum(axis=1) / V_total


# ---------------------------------------------------------------------------
# Node drop: merge surface node into the node behind it
# ---------------------------------------------------------------------------

def drop_and_merge(
    state: SimState,
    mat_registry: dict | None = None,
) -> SimState:
    """Drop the shrinking surface node (node 0) by merging it into node 1.

    The merge is conservative in energy (volume-weighted T) and mass
    (volume-weighted rho and rho_components).

    Parameters
    ----------
    state:
        Current SimState; node 0 must be the thin surface node.
    mat_registry:
        Dict {name: MaterialCard} (optional) for nodelet allocation metadata.

    Returns
    -------
    SimState
        New SimState with one fewer node in the surface layer and all
        arrays updated consistently.
    """
    mesh = state.mesh
    N = mesh.n_nodes_total
    if N < 2:
        raise RuntimeError("Cannot drop: only one node in mesh")

    # ------------------------------------------------------------------
    # Volume-weighted merge of node 0 (thin) and node 1 (full)
    # ------------------------------------------------------------------
    V0 = mesh.delta_nodes[0] * mesh.area_nodes[0]
    V1 = mesh.delta_nodes[1] * mesh.area_nodes[1]
    V_tot = V0 + V1

    if V_tot <= 0:
        # Safety fallback: just drop node 0
        T_new_0 = state.T[1]
        rho_new_0 = state.rho[1]
    else:
        T_new_0   = (state.T[0]   * V0 + state.T[1]   * V1) / V_tot
        rho_new_0 = (state.rho[0] * V0 + state.rho[1] * V1) / V_tot

    # ------------------------------------------------------------------
    # New global T and rho arrays (node 0 removed, merged value at new node 0)
    # ------------------------------------------------------------------
    T_new   = np.empty(N - 1)
    rho_new = np.empty(N - 1)
    T_new[0]   = T_new_0
    rho_new[0] = rho_new_0
    T_new[1:]   = state.T[2:]
    rho_new[1:] = state.rho[2:]

    # ------------------------------------------------------------------
    # New mesh: drop node 0 from all geometry arrays
    # ------------------------------------------------------------------
    old_y     = mesh.y_nodes
    old_delta = mesh.delta_nodes
    old_area  = mesh.area_nodes
    old_lid   = mesh.layer_id

    new_y     = old_y[1:]
    new_delta = old_delta[1:].copy()
    new_area  = old_area[1:]
    new_lid   = old_lid[1:]

    # The new node 0 (old node 1) now extends from s_total to midpoint(y[1], y[2]).
    # Its new delta is recomputed from the current s_total.
    if len(new_y) >= 2:
        right_face_new0 = 0.5 * (new_y[0] + new_y[1])
        new_delta[0] = max(0.0, right_face_new0 - mesh.s_total)

    # Update layer_boundaries: each boundary decremented by 1 for global indices.
    new_boundaries = []
    for lo, hi in mesh.layer_boundaries:
        new_lo = max(0, lo - 1)
        new_hi = max(0, hi - 1)
        if new_hi > new_lo:
            new_boundaries.append((new_lo, new_hi))
        elif new_boundaries:  # this layer is now empty — merge into previous
            prev_lo, _ = new_boundaries[-1]
            new_boundaries[-1] = (prev_lo, new_hi)

    # ------------------------------------------------------------------
    # Rho_components: merge nodelets of node 0 and node 1
    # ------------------------------------------------------------------
    new_rho_comp: list = []
    for l_idx, rho_comp_l in enumerate(state.rho_components):
        if rho_comp_l is None:
            new_rho_comp.append(None)
            continue

        # Determine how many local nodes were in old node 0's layer
        old_lo, old_hi = mesh.layer_boundaries[l_idx]
        if old_lo == 0:
            # This is the surface layer; node 0 (local 0) is being dropped
            # Merge local node 0 and local node 1 in rho_comp_l
            n_comp, N_l, J = rho_comp_l.shape
            if N_l < 2:
                # Only one node in layer; just drop it
                new_rho_comp.append(rho_comp_l[:, 1:, :] if N_l > 1 else rho_comp_l[:, :0, :])
                continue

            nd0 = mesh.nodelet_delta[l_idx][0]   # shape (J,)
            nd1 = mesh.nodelet_delta[l_idx][1]   # shape (J,)
            na0 = mesh.area_nodes[0] * np.ones(J)  # rough: use nodal area for nodelets
            na1 = mesh.area_nodes[1] * np.ones(J)
            V_n0 = (nd0 * na0).sum()
            V_n1 = (nd1 * na1).sum()
            V_t  = V_n0 + V_n1

            rho_merged = np.zeros((n_comp, J))
            if V_t > 0:
                rho_merged = (
                    rho_comp_l[:, 0, :] * V_n0 + rho_comp_l[:, 1, :] * V_n1
                ) / V_t
            else:
                rho_merged = rho_comp_l[:, 1, :]

            new_l = np.empty((n_comp, N_l - 1, J))
            new_l[:, 0, :] = rho_merged
            new_l[:, 1:, :] = rho_comp_l[:, 2:, :]
            new_rho_comp.append(new_l)

            # Also update nodelet_y and nodelet_delta for this layer
            # (just shift — use node 1's values for the merged node)
            old_ny = mesh.nodelet_y[l_idx]    # (N_l, J)
            old_nd = mesh.nodelet_delta[l_idx] # (N_l, J)
            new_ny = old_ny[1:, :]
            new_nd = old_nd[1:, :]
            mesh.nodelet_y[l_idx]    = new_ny
            mesh.nodelet_delta[l_idx] = new_nd
        else:
            # No change for other layers
            new_rho_comp.append(rho_comp_l)

    new_mesh = MeshState(
        n_nodes_total=N - 1,
        y_nodes=new_y,
        delta_nodes=new_delta,
        area_nodes=new_area,
        layer_id=new_lid,
        shrinking_node_idx=0,
        s_total=mesh.s_total,
        nodelet_y=mesh.nodelet_y,
        nodelet_delta=mesh.nodelet_delta,
        layer_boundaries=new_boundaries,
        layer_gradings=list(mesh.layer_gradings),
    )

    new_state = copy.copy(state)
    new_state.T = T_new
    new_state.rho = rho_new
    new_state.rho_components = new_rho_comp
    new_state.mesh = new_mesh
    # Remap Z_elem: drop the surface node that was merged (index 0)
    if state.Z_elem is not None:
        new_state.Z_elem = state.Z_elem[:, 1:]
    return new_state
