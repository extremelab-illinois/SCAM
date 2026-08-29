# SPDX-License-Identifier: MIT
"""Surface recession kinematics.

Handles the advance of the ablating surface into the material.  The grid is
Lagrangian: node positions (y_nodes) are fixed in the material frame and do
not move.  The ablating surface position (mesh.s_total) is tracked as a
scalar; the surface cell's effective thickness decreases as s_total increases.

After each time step:
1. s_total is incremented by s_dot * dt.
2. delta_nodes[0] is updated from the new s_total via surface_delta().
3. area_nodes[0] is updated (important for cylindrical geometry).
4. If delta_nodes[0] < threshold * h_nominal, mesh/remap.drop_and_merge() is triggered.
"""

from __future__ import annotations

import copy

import numpy as np

from scam.core.state import MeshState, SimState
from scam.config.geometry import GeometryConfig
from scam.config.stack import StackConfig
from scam.config.solver import SolverOptions
from scam.geometry.area import build_area_function
from scam.mesh.grid import surface_delta, nominal_h, _graded_nodes


def apply_recession(
    state: SimState,
    s_dot: float,
    dt: float,
    geom: GeometryConfig,
) -> SimState:
    """Advance the surface position by s_dot * dt and update mesh geometry.

    Returns a new SimState (shallow copy of arrays, new MeshState).

    Parameters
    ----------
    state:
        Current simulation state.
    s_dot:
        Surface recession rate [m/s] (>= 0).
    dt:
        Timestep [s].
    geom:
        Geometry configuration (needed to recompute area_nodes).
    """
    ds = float(s_dot) * float(dt)
    if ds < 0:
        ds = 0.0

    mesh = state.mesh
    new_s_total = mesh.s_total + ds

    # Recompute geometry for the updated surface position
    A = build_area_function(geom)

    # Node positions are fixed; only the surface cell delta changes
    new_delta = mesh.delta_nodes.copy()
    new_y = mesh.y_nodes  # unchanged

    # Surface node effective thickness
    if mesh.n_nodes_total >= 2:
        right_face_0 = 0.5 * (new_y[0] + new_y[1])
        new_delta[0] = max(0.0, right_face_0 - new_s_total)
    else:
        new_delta[0] = max(0.0, new_delta[0] - ds)

    # Area at node 0 is evaluated at the current surface position (s_total),
    # since the surface has moved to y = new_s_total.
    new_area = mesh.area_nodes.copy()
    new_area[0] = float(A(np.array([new_s_total]))[0])

    new_mesh = copy.copy(mesh)
    new_mesh.s_total = new_s_total
    new_mesh.delta_nodes = new_delta
    new_mesh.area_nodes = new_area

    new_state = copy.copy(state)
    new_state.mesh = new_mesh
    new_state.s_dot = s_dot
    return new_state


def apply_recession_ale(
    state: SimState,
    s_dot: float,
    dt: float,
    geom: GeometryConfig,
    mat_surface=None,
) -> SimState:
    """Continuous moving-mesh (ALE) recession for the surface (ablating) layer.

    Instead of shrinking the surface cell and dropping it when thin (the
    Lagrangian scheme in :func:`apply_recession` + :func:`drop_and_merge`,
    which produces a sawtooth in T_wall), this redistributes the surface
    layer's nodes uniformly between the new surface position and the layer's
    (fixed) back face on every step, keeping the node count constant.  Fields
    are conservatively re-interpolated:

      * T, Z_elem : linear interpolation at the new node centres.
      * rho_components : linear interpolation at the new *nodelet* centres
        (preserving the within-cell decomposition sub-structure), from which
        the nodal density is recomposed.

    Because the surface moves by only ``s_dot*dt`` per step, the interpolation
    shift is tiny and T_wall evolves smoothly — there are no discrete drops.

    Only layer 0 (the surface layer) moves; deeper layers are left untouched.
    """
    ds = max(0.0, float(s_dot) * float(dt))
    mesh = state.mesh
    new_s = mesh.s_total + ds
    lo, hi = mesh.layer_boundaries[0]
    N_l = hi - lo

    # Layer back face is fixed in the lab frame.
    y_back = float(mesh.y_nodes[hi - 1])
    if new_s >= y_back:           # layer fully consumed — fall back to scalar shrink
        return apply_recession(state, s_dot, dt, geom)

    A = build_area_function(geom)

    old_y_layer = mesh.y_nodes[lo:hi]
    grading = mesh.layer_gradings[0] if mesh.layer_gradings else 1.0
    new_y_layer, new_delta_layer = _graded_nodes(new_s, y_back - new_s, N_l, grading)

    # --- Interpolate nodal temperature (and Z_elem) onto the new node centres.
    new_T = state.T.copy()
    new_T[lo:hi] = np.interp(new_y_layer, old_y_layer, state.T[lo:hi])

    new_Z = None
    if state.Z_elem is not None:
        new_Z = state.Z_elem.copy()
        for i in range(state.Z_elem.shape[0]):
            new_Z[i, lo:hi] = np.interp(new_y_layer, old_y_layer, state.Z_elem[i, lo:hi])

    # --- Re-interpolate component densities at new nodelet centres.
    new_rho_comp = list(state.rho_components)
    new_nodelet_y = list(mesh.nodelet_y)
    new_nodelet_delta = list(mesh.nodelet_delta)
    rho_comp_l = state.rho_components[0]
    if rho_comp_l is not None:
        n_comp, _, J = rho_comp_l.shape
        old_ny = mesh.nodelet_y[0]                      # (N_l, J)
        old_ny_flat = old_ny.ravel()
        order = np.argsort(old_ny_flat)
        old_ny_sorted = old_ny_flat[order]

        # New nodelet centres: J uniform per new cell (matches build_mesh).
        new_ny = np.zeros((N_l, J))
        new_nd = np.zeros((N_l, J))
        for n in range(N_l):
            if n == 0:
                cl, cr = new_y_layer[0], new_y_layer[0] + new_delta_layer[0]
            elif n == N_l - 1:
                cl, cr = new_y_layer[-1] - new_delta_layer[-1], new_y_layer[-1]
            else:
                cl = new_y_layer[n] - (new_y_layer[n] - new_y_layer[n - 1]) / 2.0
                cr = new_y_layer[n] + (new_y_layer[n + 1] - new_y_layer[n]) / 2.0
            dj = (cr - cl) / J
            new_ny[n, :] = cl + dj * (np.arange(J) + 0.5)
            new_nd[n, :] = dj

        new_rho_comp_l = np.empty_like(rho_comp_l)
        for ic in range(n_comp):
            vals = rho_comp_l[ic].ravel()[order]
            new_rho_comp_l[ic] = np.interp(
                new_ny.ravel(), old_ny_sorted, vals,
            ).reshape(N_l, J)
        new_rho_comp[0] = new_rho_comp_l
        new_nodelet_y[0] = new_ny
        new_nodelet_delta[0] = new_nd

    # --- New geometry arrays.
    new_y = mesh.y_nodes.copy()
    new_delta = mesh.delta_nodes.copy()
    new_y[lo:hi] = new_y_layer
    new_delta[lo:hi] = new_delta_layer
    new_area = mesh.area_nodes.copy()
    new_area[lo:hi] = A(new_y_layer)

    new_mesh = copy.copy(mesh)
    new_mesh.s_total = new_s
    new_mesh.y_nodes = new_y
    new_mesh.delta_nodes = new_delta
    new_mesh.area_nodes = new_area
    new_mesh.nodelet_y = new_nodelet_y
    new_mesh.nodelet_delta = new_nodelet_delta

    # --- Recompose nodal density from the moved nodelets (surface layer).
    new_rho = state.rho.copy()
    if rho_comp_l is not None and mat_surface is not None and mat_surface.decomposing:
        from scam.physics.decomposition import nodal_density_from_nodelets
        area_l = np.tile(new_area[lo:hi][:, None], (1, new_nodelet_delta[0].shape[1]))
        new_rho[lo:hi] = nodal_density_from_nodelets(
            mat_surface, new_rho_comp[0], new_nodelet_delta[0], area_l,
        )

    new_state = copy.copy(state)
    new_state.T = new_T
    new_state.rho = new_rho
    new_state.rho_components = new_rho_comp
    new_state.Z_elem = new_Z
    new_state.mesh = new_mesh
    new_state.s_dot = s_dot
    return new_state


def needs_drop(
    state: SimState,
    stack: StackConfig,
    options: SolverOptions,
) -> bool:
    """Return True when the surface cell is thin enough to trigger a node drop.

    The criterion is:
        delta_surface < threshold * h_nominal_surface_layer
    """
    delta_0 = state.mesh.delta_nodes[0]
    h = nominal_h(state.mesh, stack)
    return delta_0 < options.node_drop_threshold * h
