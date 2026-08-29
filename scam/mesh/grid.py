# SPDX-License-Identifier: MIT
"""Mesh construction for multi-layer material stacks.

Grid convention
---------------
- y = 0 at the original (t = 0) surface position.
- y increases into the material (away from the hot face).
- Node positions y_n are cell centres.
- The surface-most node of the surface layer (node 0) is the "shrinking node":
  its effective cell thickness decreases as the surface recedes.
- Layer interfaces are handled by adjacent half-nodes in neighbouring layers.
  The last node of layer l and the first node of layer l+1 share the same
  physical position (the layer boundary), but are stored as separate nodes
  with half-cell control volumes on each side. At the interface, the tridiagonal
  assembly uses the appropriate conductance formula (see geometry/fvm.py).

Nodelet (decomposition subgrid) convention
-------------------------------------------
Each thermal node n in a decomposing layer contains J = n_subcells nodelets.
Nodelet j is indexed 0..J-1 from the surface side to the back side.
The nodelet centres are positioned uniformly within the node's cell extent.
Cumulative volume is the interpolation coordinate for temperature assignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from scam.config.geometry import GeometryConfig
from scam.config.stack import StackConfig, LayerConfig
from scam.core.state import MeshState, SimState
from scam.core.errors import SCAMInputError, SCAMMeshError
from scam.geometry.area import build_area_function


# ---------------------------------------------------------------------------
# Graded node helper
# ---------------------------------------------------------------------------

def _graded_nodes(
    y_start: float,
    thickness: float,
    N: int,
    grading: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_layer, delta_layer) for a layer with geometric mesh grading.

    Parameters
    ----------
    y_start:   position of the first node (surface side) [m].
    thickness: total layer thickness [m].
    N:         number of nodes (N-1 cells).
    grading:   ratio of back-face cell size to surface cell size.
               1.0 → uniform; >1 → finer near hot face; <1 → finer near back.

    The cell spacings form a geometric series: gap[k] = gap[0] * g^k where
    g = grading^(1/(N-2)).  For N <= 2 or grading ≈ 1, falls back to uniform.
    """
    if N < 2:
        raise SCAMInputError(f"n_nodes must be >= 2, got {N}")

    if N == 2 or abs(grading - 1.0) < 1e-10:
        h = thickness / (N - 1)
        y_layer = y_start + np.linspace(0.0, thickness, N)
        delta_layer = np.full(N, h)
        delta_layer[0] = h / 2.0
        delta_layer[-1] = h / 2.0
        return y_layer, delta_layer

    # N-1 gaps; grading = gap[N-2] / gap[0] = g^(N-2)
    g = grading ** (1.0 / (N - 2))
    gap0 = thickness * (g - 1.0) / (g ** (N - 1) - 1.0)
    gaps = gap0 * g ** np.arange(N - 1)

    y_layer = y_start + np.concatenate([[0.0], np.cumsum(gaps)])

    delta_layer = np.empty(N)
    delta_layer[0] = gaps[0] / 2.0
    delta_layer[-1] = gaps[-1] / 2.0
    delta_layer[1:-1] = (gaps[:-1] + gaps[1:]) / 2.0

    return y_layer, delta_layer


# ---------------------------------------------------------------------------
# Build mesh from StackConfig
# ---------------------------------------------------------------------------

def build_mesh(
    stack: StackConfig,
    geom: GeometryConfig,
    s_total: float = 0.0,
    T_init: float = 300.0,
    mat_registry: dict | None = None,
) -> tuple[MeshState, SimState]:
    """Construct the initial computational mesh for a stack.

    Parameters
    ----------
    stack:
        Layer configurations (ordered surface → back).
    geom:
        Geometry configuration (slab / cylinder / tabulated).
    s_total:
        Initial cumulative recession [m] (0 for a fresh simulation).
    T_init:
        Uniform initial temperature [K].
    mat_registry:
        Dict {name: MaterialCard} used to determine which layers are decomposing
        and how many components each has.  May be None if all layers are treated
        as inert (no nodelet allocation).

    Returns
    -------
    (MeshState, SimState)
        Fully populated mesh and initial simulation state.
    """
    if not stack.layers:
        raise SCAMInputError("StackConfig must contain at least one layer")

    A = build_area_function(geom)
    n_layers = len(stack.layers)

    # ------------------------------------------------------------------
    # Build per-layer node arrays
    # ------------------------------------------------------------------
    # Each layer has n_nodes nodes.  The first and last nodes of each layer
    # are at the layer boundary positions (half-nodes).
    # Interior nodes are uniformly spaced between the boundaries.
    #
    # Layer l spans [y_start_l, y_end_l].
    # Node spacing within layer l: h_l = thickness_l / (n_nodes_l - 1)
    # Node positions: y_l_n = y_start_l + n * h_l,  n = 0..N_l-1
    # Cell thicknesses:
    #   n = 0 (left boundary, half-node):  delta = h_l / 2
    #   n = N-1 (right boundary, half-node): delta = h_l / 2
    #   n interior: delta = h_l
    #
    # Shared boundary between layers l and l+1:
    #   Last node of layer l   (half-node, delta = h_l/2)
    #   First node of layer l+1 (half-node, delta = h_{l+1}/2)
    # These are DIFFERENT nodes in the global array; the interface conductance
    # couples them in the tridiagonal system.

    all_y: list[np.ndarray] = []
    all_delta: list[np.ndarray] = []
    all_lid: list[np.ndarray] = []
    layer_boundaries: list[tuple[int, int]] = []
    layer_gradings: list[float] = []
    nodelet_y_list: list[np.ndarray | None] = []
    nodelet_delta_list: list[np.ndarray | None] = []

    y_cursor = s_total  # current layer start (surface = s_total)

    for l_idx, layer in enumerate(stack.layers):
        N = layer.n_nodes
        if N < 2:
            raise SCAMInputError(
                f"Layer {l_idx} ('{layer.material_name}'): n_nodes must be >= 2"
            )

        thickness = layer.thickness
        grading = getattr(layer, "grading", 1.0)

        y_layer, delta_layer = _graded_nodes(y_cursor, thickness, N, grading)
        layer_gradings.append(float(grading))

        first_global = sum(len(a) for a in all_y)
        all_y.append(y_layer)
        all_delta.append(delta_layer)
        all_lid.append(np.full(N, l_idx, dtype=int))
        layer_boundaries.append((first_global, first_global + N))

        # ------------------------------------------------------------------
        # Build nodelet subgrid for decomposing layers
        # ------------------------------------------------------------------
        mat = mat_registry.get(layer.material_name) if mat_registry else None
        is_decomp = (mat is not None and mat.decomposing and len(mat.components) > 0)

        if is_decomp:
            J = layer.n_subcells
            if J < 2 or J % 2 != 0:
                raise SCAMInputError(
                    f"Layer {l_idx}: n_subcells must be >= 2 and even; got {J}"
                )
            # For each node n, place J nodelets uniformly within its cell.
            # Boundary nodes are half-cells; interior nodes span the full gap.
            # Works for both uniform and graded spacing via actual y_layer positions.

            ny_layer = np.zeros((N, J))
            nd_layer = np.zeros((N, J))

            for n in range(N):
                if n == 0:
                    cell_left  = y_layer[0]
                    cell_right = y_layer[0] + delta_layer[0]
                elif n == N - 1:
                    cell_left  = y_layer[-1] - delta_layer[-1]
                    cell_right = y_layer[-1]
                else:
                    # Cell spans from mid-gap on left to mid-gap on right
                    cell_left  = y_layer[n] - (y_layer[n] - y_layer[n - 1]) / 2.0
                    cell_right = y_layer[n] + (y_layer[n + 1] - y_layer[n]) / 2.0

                dj = (cell_right - cell_left) / J
                ny_layer[n, :] = cell_left + dj * (np.arange(J) + 0.5)
                nd_layer[n, :] = dj

            nodelet_y_list.append(ny_layer)
            nodelet_delta_list.append(nd_layer)
        else:
            nodelet_y_list.append(None)
            nodelet_delta_list.append(None)

        y_cursor += thickness

    # ------------------------------------------------------------------
    # Assemble global arrays
    # ------------------------------------------------------------------
    y_nodes = np.concatenate(all_y)
    delta_nodes = np.concatenate(all_delta)
    layer_id = np.concatenate(all_lid)
    n_total = len(y_nodes)

    area_nodes = A(y_nodes)

    mesh = MeshState(
        n_nodes_total=n_total,
        y_nodes=y_nodes,
        delta_nodes=delta_nodes.copy(),
        area_nodes=area_nodes,
        layer_id=layer_id,
        shrinking_node_idx=0,
        s_total=s_total,
        nodelet_y=nodelet_y_list,
        nodelet_delta=nodelet_delta_list,
        layer_boundaries=layer_boundaries,
        layer_gradings=layer_gradings,
    )

    # ------------------------------------------------------------------
    # Initial simulation state
    # ------------------------------------------------------------------
    T_arr = np.full(n_total, T_init)
    rho_arr = np.zeros(n_total)

    rho_components: list = []
    for l_idx, layer in enumerate(stack.layers):
        mat = mat_registry.get(layer.material_name) if mat_registry else None
        is_decomp = (mat is not None and mat.decomposing and len(mat.components) > 0)
        if is_decomp:
            J = layer.n_subcells
            N_l = layer.n_nodes
            n_comp = len(mat.components)
            rho_comp_l = np.zeros((n_comp, N_l, J))
            for ic, comp in enumerate(mat.components):
                rho_comp_l[ic, :, :] = comp.rho_0
            rho_components.append(rho_comp_l)
            # Initial nodal density: fiber (non-reactive, = rho_char - sum(rho_r))
            # plus the initial reactive component densities sum(rho_0).  This
            # equals rho_char + sum(rho_0 - rho_r) = rho_virgin.  Do NOT use
            # rho_char + sum(rho_0) — that double-counts the residual density
            # rho_r of each component (visible bug when any rho_r > 0, e.g. v2.2).
            lo, hi = layer_boundaries[l_idx]
            rho_arr[lo:hi] = mat.rho_virgin
        else:
            rho_components.append(None)
            if mat is not None:
                lo, hi = layer_boundaries[l_idx]
                rho_arr[lo:hi] = mat.rho_virgin
            # If no mat info, leave at 0; caller can set rho manually.

    state = SimState(
        time=0.0,
        dt=0.0,
        T=T_arr,
        rho=rho_arr,
        rho_components=rho_components,
        s_dot=0.0,
        T_wall=T_init,
        q_cond=0.0,
        m_dot_pyro=0.0,
        m_dot_char=0.0,
        mesh=mesh,
    )
    return mesh, state


# ---------------------------------------------------------------------------
# Effective surface-cell delta (updated each step during recession)
# ---------------------------------------------------------------------------

def surface_delta(mesh: MeshState) -> float:
    """Current effective thickness of the ablating surface cell [m].

    As the surface recedes (s_total increases), the surface cell
    shrinks from its nominal half-cell thickness toward zero.
    The left face of node 0 is at the current surface (y = s_total);
    the right face is at the midpoint between node 0 and node 1.
    """
    y = mesh.y_nodes
    if mesh.n_nodes_total < 2:
        return 0.0
    right_face_0 = 0.5 * (y[0] + y[1])
    return max(0.0, right_face_0 - mesh.s_total)


def nominal_h(mesh: MeshState, stack: StackConfig) -> float:
    """Nominal inter-node spacing in the surface layer [m]."""
    return stack.layers[0].thickness / (stack.layers[0].n_nodes - 1)


def update_surface_delta(mesh: MeshState) -> MeshState:
    """Return a mesh with delta_nodes[0] updated for the current s_total."""
    new_delta = mesh.delta_nodes.copy()
    new_delta[0] = surface_delta(mesh)
    import copy
    m2 = copy.copy(mesh)
    m2.delta_nodes = new_delta
    return m2
