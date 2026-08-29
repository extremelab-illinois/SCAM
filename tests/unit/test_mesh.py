# SPDX-License-Identifier: MIT
"""Unit tests for mesh/grid.py."""
import numpy as np
import pytest

from scam.config.geometry import GeometryConfig, GeometryType
from scam.config.material import MaterialCard
from scam.config.stack import LayerConfig, StackConfig
from scam.mesh.grid import build_mesh, nominal_h
from scam.mesh.interpolation import interpolate_layer_to_nodelets
from scam.mesh.remap import interpolate_to_nodelets


def _make_inert_mat(name: str = "M", rho: float = 200.0) -> MaterialCard:
    T_pts = np.array([200.0, 3000.0])
    k_tab = np.column_stack([T_pts, np.full(2, 0.5)])
    cp_tab = np.column_stack([T_pts, np.full(2, 800.0)])
    hg_tab = np.column_stack([T_pts, np.zeros(2)])
    return MaterialCard(
        name=name, rho_virgin=rho, rho_char=rho, gamma_resin=0.0,
        components=[], k_virgin_table=k_tab, k_char_table=k_tab,
        cp_virgin_table=cp_tab, cp_char_table=cp_tab, h_g_table=hg_tab,
        emissivity=0.85, decomposing=False,
    )


def _single_layer_mesh(n_nodes: int = 11, thickness: float = 0.10):
    mat = _make_inert_mat()
    stack = StackConfig(layers=[
        LayerConfig("M", thickness=thickness, n_nodes=n_nodes, n_subcells=4),
    ])
    geom = GeometryConfig(geometry_type=GeometryType.SLAB)
    mat_registry = {"M": mat}
    return build_mesh(stack, geom, s_total=0.0, T_init=300.0, mat_registry=mat_registry)


def test_single_layer_total_thickness():
    mesh, state = _single_layer_mesh(n_nodes=11, thickness=0.10)
    total = float(mesh.delta_nodes.sum())
    assert abs(total - 0.10) < 1e-12, f"Total thickness {total} != 0.10"


def test_single_layer_node_count():
    mesh, _ = _single_layer_mesh(n_nodes=11)
    assert mesh.n_nodes_total == 11


def test_half_nodes_at_boundary():
    N = 11
    L = 0.10
    h = L / (N - 1)
    mesh, _ = _single_layer_mesh(n_nodes=N, thickness=L)
    # Surface node and back node should be h/2
    assert abs(mesh.delta_nodes[0]  - h / 2) < 1e-12
    assert abs(mesh.delta_nodes[-1] - h / 2) < 1e-12
    # Interior nodes should be h
    for i in range(1, N - 1):
        assert abs(mesh.delta_nodes[i] - h) < 1e-12


def test_initial_temperature():
    _, state = _single_layer_mesh()
    np.testing.assert_allclose(state.T, 300.0)


def test_two_layer_total_thickness():
    mat = _make_inert_mat()
    stack = StackConfig(layers=[
        LayerConfig("M", thickness=0.05, n_nodes=6),
        LayerConfig("M", thickness=0.05, n_nodes=6),
    ])
    geom = GeometryConfig(geometry_type=GeometryType.SLAB)
    mesh, _ = build_mesh(stack, geom, s_total=0.0, T_init=300.0, mat_registry={"M": mat})
    total = float(mesh.delta_nodes.sum())
    assert abs(total - 0.10) < 1e-12


def test_layer_id_assignment():
    mat = _make_inert_mat()
    stack = StackConfig(layers=[
        LayerConfig("M", thickness=0.05, n_nodes=6),
        LayerConfig("M", thickness=0.05, n_nodes=6),
    ])
    geom = GeometryConfig(geometry_type=GeometryType.SLAB)
    mesh, _ = build_mesh(stack, geom, s_total=0.0, T_init=300.0, mat_registry={"M": mat})
    # First 6 nodes → layer 0, next 6 → layer 1
    assert all(mesh.layer_id[:6] == 0)
    assert all(mesh.layer_id[6:] == 1)


def test_nominal_h():
    n, L = 11, 0.10
    mat = _make_inert_mat()
    stack = StackConfig(layers=[LayerConfig("M", thickness=L, n_nodes=n)])
    geom = GeometryConfig(geometry_type=GeometryType.SLAB)
    mesh, _ = build_mesh(stack, geom, s_total=0.0, T_init=300.0, mat_registry={"M": mat})
    h = nominal_h(mesh, stack)
    assert abs(h - L / (n - 1)) < 1e-12


@pytest.mark.parametrize("n_subcells", [2, 3, 4, 7])
def test_vectorized_nodelet_interpolation_matches_scalar(n_subcells):
    rng = np.random.default_rng(81 + n_subcells)
    n_nodes = 17
    T_left = rng.uniform(300.0, 2200.0, n_nodes)
    T_center = rng.uniform(300.0, 2200.0, n_nodes)
    T_right = rng.uniform(300.0, 2200.0, n_nodes)
    delta = rng.uniform(1e-6, 4e-4, (n_nodes, n_subcells))
    area = rng.uniform(0.2, 3.0, n_nodes)

    expected = np.empty_like(delta)
    for node in range(n_nodes):
        cumulative_volume = np.cumsum(delta[node] * area[node])
        expected[node] = interpolate_to_nodelets(
            T_left[node],
            T_center[node],
            T_right[node],
            cumulative_volume,
            cumulative_volume[-1],
        )

    actual = interpolate_layer_to_nodelets(
        T_left, T_center, T_right, delta, area,
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-12)
