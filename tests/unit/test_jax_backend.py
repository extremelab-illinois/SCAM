# SPDX-License-Identifier: MIT
"""Parity tests for the optional JAX numerical kernels."""

import numpy as np
import pytest

pytest.importorskip("jax")

from scam.config.material import ComponentCard, MaterialCard
from scam.numerics.tridiagonal import TridiagSystem, solve_thomas
from scam.physics.decomposition import update_nodelet_densities


def test_jax_thomas_matches_numpy():
    rng = np.random.default_rng(42)
    n = 201
    A = -rng.uniform(0.1, 1.0, n)
    C = -rng.uniform(0.1, 1.0, n)
    A[0] = 0.0
    C[-1] = 0.0
    B = np.abs(A) + np.abs(C) + 1.0
    D = rng.normal(size=n)
    system = TridiagSystem(A=A, B=B, C=C, D=D)

    expected = solve_thomas(system)
    actual = solve_thomas(system, backend="jax")
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_jax_decomposition_matches_numpy():
    components = [
        ComponentCard(name="c1", A_rate=1.2e5, E_act=8.0e4, m_exp=1.0, rho_0=120.0, rho_r=20.0),
        ComponentCard(name="c2", A_rate=2.5e7, E_act=1.2e5, m_exp=2.0, rho_0=80.0, rho_r=10.0),
    ]
    material = MaterialCard(
        name="jax_test",
        rho_virgin=300.0,
        rho_char=100.0,
        gamma_resin=0.5,
        components=components,
        decomposing=True,
    )
    rho = np.empty((2, 16, 4))
    rho[0] = 115.0
    rho[1] = 75.0
    temperatures = np.linspace(500.0, 1800.0, 64).reshape(16, 4)
    delta = np.full((16, 4), 2.5e-5)
    shrinking = np.zeros(16, dtype=bool)
    shrinking[0] = True

    expected = update_nodelet_densities(
        material, rho, temperatures, 0.01, 1e-4, delta, shrinking
    )
    actual = update_nodelet_densities(
        material, rho, temperatures, 0.01, 1e-4, delta, shrinking,
        backend="jax",
    )
    for actual_array, expected_array in zip(actual, expected):
        np.testing.assert_allclose(actual_array, expected_array, rtol=1e-11, atol=1e-11)
