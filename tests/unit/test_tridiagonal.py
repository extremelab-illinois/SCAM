# SPDX-License-Identifier: MIT
"""Unit tests for numerics/tridiagonal.py."""
import numpy as np
import pytest

from scam.numerics.tridiagonal import TridiagSystem, solve_thomas


def _make_system(n: int, seed: int = 42) -> tuple[TridiagSystem, np.ndarray]:
    """Create a diagonally dominant tridiagonal system with known solution."""
    rng = np.random.default_rng(seed)
    A = -rng.uniform(0.5, 1.5, n)   # sub-diagonal (negative)
    C = -rng.uniform(0.5, 1.5, n)   # super-diagonal (negative)
    A[0] = 0.0                        # boundary
    C[-1] = 0.0                       # boundary
    # Diagonal: strict diagonal dominance
    B = np.abs(A) + np.abs(C) + rng.uniform(0.5, 1.0, n)
    x_exact = rng.uniform(1.0, 10.0, n)
    D = A * np.roll(x_exact, 1) + B * x_exact + C * np.roll(x_exact, -1)
    D[0]  = B[0] * x_exact[0]  + C[0]  * x_exact[1]
    D[-1] = A[-1] * x_exact[-2] + B[-1] * x_exact[-1]
    tri = TridiagSystem(A=A.copy(), B=B.copy(), C=C.copy(), D=D.copy())
    return tri, x_exact


def test_thomas_n2():
    tri = TridiagSystem(
        A=np.array([0.0, -1.0]),
        B=np.array([2.0,  2.0]),
        C=np.array([-1.0, 0.0]),
        D=np.array([1.0,  3.0]),
    )
    x = solve_thomas(tri)
    # Expected: [5/3, 7/3]
    assert x[0] == pytest.approx(5.0 / 3.0, rel=1e-12)
    assert x[1] == pytest.approx(7.0 / 3.0, rel=1e-12)


def test_thomas_n10():
    tri, x_exact = _make_system(10)
    x = solve_thomas(tri)
    np.testing.assert_allclose(x, x_exact, rtol=1e-10)


def test_thomas_n100():
    tri, x_exact = _make_system(100)
    x = solve_thomas(tri)
    np.testing.assert_allclose(x, x_exact, rtol=1e-10)


def test_thomas_matches_scipy():
    from scam.numerics.tridiagonal import solve_thomas_scipy
    tri, x_exact = _make_system(50, seed=7)
    x_thomas = solve_thomas(tri)
    x_scipy  = solve_thomas_scipy(tri)
    np.testing.assert_allclose(x_thomas, x_scipy, rtol=1e-10)
