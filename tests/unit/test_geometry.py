# SPDX-License-Identifier: MIT
"""Unit tests for geometry/area.py."""
import numpy as np
import pytest

from scam.config.geometry import GeometryConfig, GeometryType
from scam.geometry.area import build_area_function, area_at


def test_slab_area_is_one():
    cfg = GeometryConfig(geometry_type=GeometryType.SLAB)
    A = build_area_function(cfg)
    y = np.linspace(0, 0.1, 10)
    np.testing.assert_allclose(A(y), np.ones(10))


def test_cylinder_area():
    r_inner = 0.05
    cfg = GeometryConfig(geometry_type=GeometryType.HOLLOW_CYLINDER, r_inner=r_inner)
    A = build_area_function(cfg)
    y = np.array([0.0, 0.01, 0.02])
    expected = r_inner + y
    np.testing.assert_allclose(A(y), expected)


def test_tabulated_area():
    table = np.array([[0.0, 1.0], [0.05, 2.0], [0.10, 3.0]])
    cfg = GeometryConfig(
        geometry_type=GeometryType.TABULATED,
        area_table=table,
    )
    A = build_area_function(cfg)
    np.testing.assert_allclose(A(np.array([0.0])), [1.0])
    np.testing.assert_allclose(A(np.array([0.05])), [2.0])
    np.testing.assert_allclose(A(np.array([0.10])), [3.0])
    # Midpoint interpolation
    assert abs(A(np.array([0.025]))[0] - 1.5) < 0.01


def test_tabulated_flat_extrapolation():
    table = np.array([[0.0, 1.0], [0.1, 2.0]])
    cfg = GeometryConfig(
        geometry_type=GeometryType.TABULATED,
        area_table=table,
    )
    A = build_area_function(cfg)
    # Extrapolation should be flat (no crash, no divergence)
    assert A(np.array([-0.01]))[0] == pytest.approx(1.0)
    assert A(np.array([0.2]))[0]   == pytest.approx(2.0)


def test_area_at_wrapper():
    cfg = GeometryConfig(geometry_type=GeometryType.SLAB)
    assert area_at(np.array([0.0, 0.05]), cfg).sum() == pytest.approx(2.0)
