# SPDX-License-Identifier: MIT
"""Scalar nonlinear solvers (Newton, bisection).

Used primarily for the surface energy balance (SEB) Newton iteration on T_wall.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from scam.core.errors import SCAMNumericsError


def newton_scalar(
    f: Callable[[float], float],
    x0: float,
    tol: float = 1.0,
    max_iter: int = 50,
    dx_fd: float = 1.0,
    x_min: float = -np.inf,
    x_max: float = np.inf,
) -> float:
    """Scalar Newton iteration with finite-difference Jacobian.

    Convergence criterion: |f(x)| < tol.

    Parameters
    ----------
    f:
        Residual function f(x) → scalar. Converges when |f(x)| < tol.
    x0:
        Initial guess.
    tol:
        Absolute convergence tolerance on the residual |f(x)| [same units as f].
    max_iter:
        Maximum number of iterations before raising SCAMNumericsError.
    dx_fd:
        Step size for finite-difference Jacobian estimation [same units as x].
    x_min, x_max:
        Bounds on x; Newton step is clipped to [x_min, x_max].

    Returns
    -------
    float
        Converged solution x*.
    """
    x = float(x0)
    for _ in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return x
        # Finite-difference derivative
        fx_dx = f(x + dx_fd)
        df = (fx_dx - fx) / dx_fd
        if df == 0.0:
            # Try the other direction
            fx_dx2 = f(x - dx_fd)
            df = (fx - fx_dx2) / dx_fd
        if df == 0.0:
            raise SCAMNumericsError(
                f"Newton iteration: zero derivative at x={x:.3f}, f={fx:.3e}"
            )
        x_new = x - fx / df
        x = float(np.clip(x_new, x_min, x_max))

    fx_final = f(x)
    if abs(fx_final) >= tol:
        raise SCAMNumericsError(
            f"Newton iteration did not converge after {max_iter} iterations; "
            f"|f| = {abs(fx_final):.3e} > tol = {tol:.3e}"
        )
    return x


def bisect(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1.0,
    max_iter: int = 100,
) -> float:
    """Robust bisection solver for f(x) = 0 on [a, b].

    Requires that f(a) and f(b) have opposite signs.

    Parameters
    ----------
    f:
        Residual function.
    a, b:
        Bracket endpoints (f(a) * f(b) < 0 required).
    tol:
        Absolute convergence tolerance on |f(x)|.
    max_iter:
        Maximum iterations.
    """
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise SCAMNumericsError(
            f"bisect: f(a)={fa:.3e} and f(b)={fb:.3e} have the same sign; no bracket"
        )
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        fm = f(mid)
        if abs(fm) < tol:
            return mid
        if fa * fm < 0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
    return 0.5 * (a + b)
