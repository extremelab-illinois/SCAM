# SPDX-License-Identifier: MIT
"""Thomas algorithm (TDMA) for tridiagonal linear systems.

Solves:  A[n]*x[n-1] + B[n]*x[n] + C[n]*x[n+1] = D[n]

with  A[0] and C[-1] unused (boundary conditions absorbed into B and D).

The Thomas algorithm is O(N) and numerically stable for diagonally dominant
systems, which is always the case for the heat-conduction FVM stencil.

A scipy ``solve_banded`` fallback is provided for validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class TridiagSystem:
    """Coefficients of a tridiagonal linear system of size N.

    A[n] * T[n-1] + B[n] * T[n] + C[n] * T[n+1] = D[n]

    Conventions:
    - A[0] is unused (no node to the left of node 0).
    - C[N-1] is unused (no node to the right of node N-1).
    - B[n] is always the diagonal (must be non-zero).
    - D[n] is the right-hand side.
    """

    A: NDArray[np.float64]   # sub-diagonal, shape (N,)
    B: NDArray[np.float64]   # diagonal,     shape (N,)
    C: NDArray[np.float64]   # super-diagonal,shape (N,)
    D: NDArray[np.float64]   # RHS,          shape (N,)

    def __post_init__(self):
        N = len(self.B)
        for arr, name in ((self.A, "A"), (self.B, "B"), (self.C, "C"), (self.D, "D")):
            if len(arr) != N:
                raise ValueError(f"TridiagSystem: array '{name}' has length {len(arr)}, expected {N}")

    @property
    def N(self) -> int:
        return len(self.B)


def solve_thomas(
    sys: TridiagSystem,
    backend: str = "numpy",
) -> NDArray[np.float64]:
    """Solve the tridiagonal system using the Thomas (TDMA) algorithm.

    Parameters
    ----------
    sys:
        TridiagSystem with coefficients A, B, C, D.

    Returns
    -------
    np.ndarray
        Solution vector x, shape (N,).

    Raises
    ------
    RuntimeError
        If a zero pivot is encountered (system is singular).
    """
    if backend == "jax":
        from scam.numerics.jax_kernels import solve_thomas_jax
        return solve_thomas_jax(sys.A, sys.B, sys.C, sys.D)
    if backend != "numpy":
        raise ValueError(f"Unknown numerical backend: {backend!r}")

    N = sys.N
    A, B, C, D = sys.A, sys.B, sys.C, sys.D

    # Forward elimination (in-place on copies)
    c_prime = np.empty(N)
    d_prime = np.empty(N)

    if B[0] == 0.0:
        raise RuntimeError("Thomas algorithm: zero diagonal at node 0 (singular system)")
    c_prime[0] = C[0] / B[0]
    d_prime[0] = D[0] / B[0]

    for n in range(1, N):
        denom = B[n] - A[n] * c_prime[n - 1]
        if denom == 0.0:
            raise RuntimeError(f"Thomas algorithm: zero pivot at node {n} (singular system)")
        c_prime[n] = C[n] / denom
        d_prime[n] = (D[n] - A[n] * d_prime[n - 1]) / denom

    # Back substitution
    x = np.empty(N)
    x[-1] = d_prime[-1]
    for n in range(N - 2, -1, -1):
        x[n] = d_prime[n] - c_prime[n] * x[n + 1]

    return x


def solve_thomas_scipy(sys: TridiagSystem) -> NDArray[np.float64]:
    """Solve the tridiagonal system via scipy.linalg.solve_banded.

    Use for validation / debugging; the Thomas algorithm is faster.
    """
    from scipy.linalg import solve_banded

    N = sys.N
    # scipy banded format: ab[0] = super-diag (C), ab[1] = diag (B), ab[2] = sub-diag (A)
    ab = np.zeros((3, N))
    ab[0, 1:]  = sys.C[:-1]   # upper diagonal (offset +1)
    ab[1, :]   = sys.B         # main diagonal
    ab[2, :-1] = sys.A[1:]    # lower diagonal (offset -1)

    return solve_banded((1, 1), ab, sys.D)
