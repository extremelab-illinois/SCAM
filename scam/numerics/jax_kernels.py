# SPDX-License-Identifier: MIT
"""Optional JAX implementations of fixed-shape numerical kernels.

This module is imported only when ``SolverOptions.array_backend == "jax"``.
Keeping the import lazy means JAX is not a runtime dependency for normal SCAM
use.  All kernels use float64 to preserve the solver's NumPy numerics.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def _jax_modules():
    try:
        import jax
        import jax.numpy as jnp
        from jax import lax
    except ImportError as exc:
        raise RuntimeError(
            "The JAX backend requires JAX; install it with "
            '`pip install -e ".[jax]"`.'
        ) from exc

    jax.config.update("jax_enable_x64", True)
    return jax, jnp, lax


@lru_cache(maxsize=1)
def _compiled_thomas():
    jax, jnp, lax = _jax_modules()

    @jax.jit
    def _solve(A, B, C, D):
        n = B.shape[0]
        c_prime = jnp.zeros_like(B).at[0].set(C[0] / B[0])
        d_prime = jnp.zeros_like(D).at[0].set(D[0] / B[0])

        def forward(i, carry):
            cp, dp = carry
            denom = B[i] - A[i] * cp[i - 1]
            cp = cp.at[i].set(C[i] / denom)
            dp = dp.at[i].set((D[i] - A[i] * dp[i - 1]) / denom)
            return cp, dp

        c_prime, d_prime = lax.fori_loop(1, n, forward, (c_prime, d_prime))
        x = jnp.zeros_like(D).at[-1].set(d_prime[-1])

        def backward(k, value):
            i = n - 2 - k
            return value.at[i].set(d_prime[i] - c_prime[i] * value[i + 1])

        return lax.fori_loop(0, n - 1, backward, x)

    return _solve


def solve_thomas_jax(A, B, C, D) -> np.ndarray:
    """Solve one tridiagonal system and return a host NumPy array."""
    solve = _compiled_thomas()
    result = solve(A, B, C, D)
    return np.asarray(result.block_until_ready())


@lru_cache(maxsize=1)
def _compiled_f_cond():
    jax, jnp, lax = _jax_modules()

    @jax.jit
    def _eliminate(A, B, C, D):
        n = B.shape[0]

        def backward(k, carry):
            bs, ds = carry
            i = n - 1 - k
            factor = C[i - 1] / bs[i]
            bs = bs.at[i - 1].add(-factor * A[i])
            ds = ds.at[i - 1].add(-factor * ds[i])
            return bs, ds

        bs, ds = lax.fori_loop(0, n - 1, backward, (B, D))
        return bs[0], ds[0]

    return _eliminate


def eliminate_f_cond_jax(A, B, C, D) -> tuple[float, float]:
    """Return the reduced surface diagonal and RHS after backward elimination."""
    eliminate = _compiled_f_cond()
    b0, d0 = eliminate(A, B, C, D)
    b0.block_until_ready()
    return float(b0), float(d0)


@lru_cache(maxsize=1)
def _compiled_decomposition():
    jax, jnp, lax = _jax_modules()

    @jax.jit
    def _update(
        rho,
        T_nodelets,
        dt,
        s_dot,
        delta_nodelets,
        is_shrinking,
        A_rate,
        E_act,
        rho_0,
        rho_r,
        m_exp,
        gas_yield,
        gas_yield_sum,
        r_universal,
    ):
        del gas_yield, gas_yield_sum
        n_comp, n_nodes, n_subcells = rho.shape
        j_arr = jnp.arange(n_subcells, dtype=rho.dtype) + 0.5
        v_j = jnp.where(
            is_shrinking[:, None],
            s_dot * (n_subcells - j_arr) / n_subcells,
            s_dot,
        )
        delta_cells = jnp.sum(delta_nodelets, axis=1)
        safe_delta = jnp.where(delta_cells > 0.0, delta_cells, 1.0)

        def update_component(rho_i, params):
            Ai, Ei, rho0i, rhori, mi = params
            excess = rho_i - rhori
            active = excess > 0.0
            k = Ai * jnp.exp(-Ei / (r_universal * T_nodelets))

            first_order = excess * jnp.exp(-k * dt)
            safe_xi = jnp.where(active, excess / rho0i, 1.0)
            xi_1m = safe_xi ** (1.0 - mi) + (mi - 1.0) * k * dt
            power_order = rho0i * jnp.where(
                xi_1m > 0.0,
                xi_1m ** (1.0 / (1.0 - mi)),
                0.0,
            )
            excess_new = lax.cond(
                mi == 1.0,
                lambda _: first_order,
                lambda _: power_order,
                operand=None,
            )
            excess_new = jnp.where(active, excess_new, 0.0)
            rho_reacted = jnp.clip(rhori + excess_new, rhori, rho_i)

            grad = jnp.zeros_like(rho_i)
            grad = grad.at[:, 1:].set(
                (rho_i[:, 1:] - rho_i[:, :-1]) / delta_nodelets[:, 1:]
            )
            drho_dt = (rho_reacted - rho_i) / dt
            drho_dt_y = drho_dt - v_j * grad
            rho_new_i = jnp.clip(rho_i + drho_dt_y * dt, rhori, rho_i)
            contribution = jnp.sum(drho_dt_y * delta_nodelets, axis=1)
            return rho_new_i, contribution / safe_delta

        params = (A_rate, E_act, rho_0, rho_r, m_exp)
        rho_new, rates = jax.vmap(update_component, in_axes=(0, 0))(
            rho, jnp.stack(params, axis=1)
        )
        return rho_new, jnp.sum(rates, axis=0), rates

    return _update


def update_nodelet_densities_jax(
    rho_comp_layer,
    T_nodelets,
    dt,
    s_dot,
    delta_nodelets,
    is_shrinking,
    components,
    r_universal,
):
    """JAX equivalent of the vectorized NumPy decomposition update."""
    update = _compiled_decomposition()
    arrays = [
        np.asarray([getattr(component, name) for component in components], dtype=float)
        for name in ("A_rate", "E_act", "rho_0", "rho_r", "m_exp")
    ]
    gas_yield = np.asarray(
        [getattr(component, "gas_yield", 0.0) for component in components],
        dtype=float,
    )
    result = update(
        rho_comp_layer,
        T_nodelets,
        dt,
        s_dot,
        delta_nodelets,
        is_shrinking,
        *arrays,
        gas_yield,
        float(gas_yield.sum()),
        r_universal,
    )
    result[0].block_until_ready()
    return tuple(np.asarray(value) for value in result)
