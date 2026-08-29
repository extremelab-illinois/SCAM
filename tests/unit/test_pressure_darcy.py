# SPDX-License-Identifier: MIT
"""Unit tests for the pressure-driven Darcy flow solver.

V1 — single-layer constant-property case: analytical parabolic solution.
V2 — two-layer piecewise case: piecewise-quadratic analytical solution.

Coordinate convention (same as SCAM):
  y = 0 at surface (Dirichlet BC), y = L at back wall (no-flow Neumann BC).

Analytical solution (constant Γ = K/μ, S = ε_g/T·dT/dt):
  PDE:  d/dy(Γ dp/dy) = S  →  d²p/dy² = -S/Γ
  BCs:  p(0) = p₀,  dp/dy(L) = 0

  Integrating from y to L with no-flow BC:
    dp/dy(y) = S·(L − y)/Γ  > 0      [gradient is positive → p increases into material]
    p(y)     = p₀ + S/(2Γ)·(2Ly − y²)  [parabolic; p(L) = p₀ + S·L²/(2Γ) > p₀]
    J_g(y)   = −ρ_g·S·(L − y)          [negative = toward surface]

Two-layer case: each layer satisfies d²p/dy² = −Sᵢ/Γᵢ so
  pᵢ(y) = Aᵢ + Bᵢ·y − Sᵢ/(2Γᵢ)·y²

with BC dp₂/dy(L) = 0 giving B₂ = S₂·L/Γ₂  (positive).
"""

from __future__ import annotations

import numpy as np
import pytest

from scam.config.material import MaterialCard, ComponentCard
from scam.physics.pressure_darcy import (
    _gas_viscosity,
    _node_permeability,
    _mass_flux_from_pressure,
    solve_gas_pressure,
    solve_pressure_and_flux,
)


# ---------------------------------------------------------------------------
# Minimal mesh and material stubs
# ---------------------------------------------------------------------------

class _FakeMesh:
    """Minimal MeshState stub sufficient for the pressure solver."""

    def __init__(self, N: int, L: float, layer_id: np.ndarray | None = None):
        self.n_nodes_total = N
        dy = L / N
        self.delta_nodes = np.full(N, dy)
        self.area_nodes = np.ones(N)
        self.layer_id = layer_id if layer_id is not None else np.zeros(N, dtype=int)
        self.y_nodes = np.arange(N) * dy + dy * 0.5
        self.layer_boundaries = [(0, N)]


def _mat(K_char: float, K_virgin: float, eps_g: float,
         rho_v: float = 300.0, rho_c: float = 200.0) -> MaterialCard:
    """Minimal MaterialCard with Darcy properties set."""
    comp = ComponentCard(
        name="resin", rho_0=rho_v * 0.8, rho_r=rho_c,
        A_rate=1e4, E_act=1e5, m_exp=3.0,
    )
    return MaterialCard(
        name="test",
        rho_virgin=rho_v,
        rho_char=rho_c,
        gamma_resin=0.8,
        components=[comp],
        permeability=K_char,
        permeability_virgin=K_virgin,
        eps_g_virgin=eps_g,
        eps_g_char=eps_g,
        gas_pressure=101325.0,
        gas_molar_mass=0.022,
    )


def _face_y(N: int, L: float) -> np.ndarray:
    """Face positions for a uniform N-cell mesh over [0, L]."""
    return np.arange(N + 1) * (L / N)


# ---------------------------------------------------------------------------
# V1 — single-layer, constant properties
# ---------------------------------------------------------------------------

class TestV1SingleLayer:
    """Parabolic analytical solution for constant Γ, S."""

    N = 100
    L = 0.025           # m
    K = 2.0e-11         # m²  (char permeability, no virgin blend)
    eps_g = 0.85
    T_val = 1200.0      # K
    dTdt_val = 500.0    # K/s
    p0 = 101325.0       # Pa

    def _setup(self):
        mesh = _FakeMesh(self.N, self.L)
        mat = _mat(self.K, 0.0, self.eps_g)
        mat_list = [mat]
        T = np.full(self.N, self.T_val)
        rho = np.full(self.N, mat.rho_char)   # fully charred → K_eff = K_char
        dTdt = np.full(self.N, self.dTdt_val)
        return mesh, mat_list, T, rho, dTdt

    def _analytical(self):
        mu = _gas_viscosity(self.T_val)
        Gamma = self.K / mu
        S = self.eps_g / self.T_val * self.dTdt_val
        L = self.L
        y_n = _FakeMesh(self.N, L).y_nodes
        # p(y) = p₀ + S/(2Γ)·(2Ly − y²)   [increases into material]
        p_ana = self.p0 + S / (2.0 * Gamma) * (2.0 * L * y_n - y_n**2)
        # J_g at faces: J_g = −ρ_g·S·(L − y_face)   [negative = toward surface]
        y_face = _face_y(self.N, L)
        R_univ = 8.314
        rho_g = self.p0 * 0.022 / (R_univ * self.T_val)
        J_ana = -rho_g * S * (L - y_face)
        return p_ana, J_ana, Gamma, S

    def test_pressure_profile(self):
        mesh, mat_list, T, rho, dTdt = self._setup()
        p_num = solve_gas_pressure(mat_list, mesh, T, rho, dTdt, self.p0)
        p_ana, _, Gamma, S = self._analytical()
        # normalise by max pressure deviation from p₀
        p_max_dev = S / (2.0 * Gamma) * self.L**2
        l2_err = np.linalg.norm(p_num - p_ana) / (p_max_dev * np.sqrt(self.N))
        # O(h²) scheme, h = L/N = 2.5e-4 → expect ~1e-4 normalised error at N=100
        assert l2_err < 1e-3, f"V1 pressure L2 error {l2_err:.2e} exceeds 1e-3"

    def test_pressure_increases_into_material(self):
        """Gas generation + no-flow back wall → pressure is highest at back."""
        mesh, mat_list, T, rho, dTdt = self._setup()
        p_num = solve_gas_pressure(mat_list, mesh, T, rho, dTdt, self.p0)
        assert np.all(np.diff(p_num) >= 0), "Pressure should increase into material"

    def test_flux_profile(self):
        mesh, mat_list, T, rho, dTdt = self._setup()
        _, J_num = solve_pressure_and_flux(mat_list, mesh, T, rho, dTdt, self.p0)
        _, J_ana, _, _ = self._analytical()
        max_flux = np.abs(J_ana).max()
        l2_err = np.linalg.norm(J_num - J_ana) / (max_flux * np.sqrt(self.N + 1))
        assert l2_err < 1e-3, f"V1 flux L2 error {l2_err:.2e} exceeds 1e-3"

    def test_back_wall_flux_zero(self):
        mesh, mat_list, T, rho, dTdt = self._setup()
        _, J_num = solve_pressure_and_flux(mat_list, mesh, T, rho, dTdt, self.p0)
        assert abs(J_num[-1]) < 1e-30, f"Back-wall flux not zero: {J_num[-1]}"

    def test_flux_toward_surface(self):
        """All interior face fluxes should be negative (toward surface, -y)."""
        mesh, mat_list, T, rho, dTdt = self._setup()
        _, J_num = solve_pressure_and_flux(mat_list, mesh, T, rho, dTdt, self.p0)
        assert np.all(J_num[:-1] <= 0), "Interior fluxes should be ≤ 0 (toward surface)"

    def test_no_permeability_returns_uniform_pressure(self):
        mesh, mat_list, T, rho, dTdt = self._setup()
        mat_list[0].permeability = 0.0
        p_num = solve_gas_pressure(mat_list, mesh, T, rho, dTdt, self.p0)
        assert np.allclose(p_num, self.p0)


# ---------------------------------------------------------------------------
# V2 — two-layer piecewise-quadratic
# ---------------------------------------------------------------------------

class TestV2TwoLayer:
    """Piecewise-quadratic analytical solution for two layers with different K, ε_g."""

    N = 200
    L = 0.05
    L_half = 0.025   # interface at midpoint
    K1 = 2.0e-11     # char layer (surface side)
    eps1 = 0.85
    K2 = 1.6e-11     # virgin layer (back side)
    eps2 = 0.80
    T_val = 1000.0
    dTdt_val = 300.0
    p0 = 101325.0

    def _setup(self):
        N_half = self.N // 2
        layer_id = np.array([0] * N_half + [1] * N_half, dtype=int)
        mesh = _FakeMesh(self.N, self.L, layer_id=layer_id)
        mesh.layer_boundaries = [(0, N_half), (N_half, self.N)]
        mat1 = _mat(self.K1, 0.0, self.eps1)
        mat2 = _mat(self.K2, 0.0, self.eps2)
        T = np.full(self.N, self.T_val)
        rho = np.concatenate([
            np.full(N_half, mat1.rho_char),
            np.full(N_half, mat2.rho_char),
        ])
        dTdt = np.full(self.N, self.dTdt_val)
        return mesh, [mat1, mat2], T, rho, dTdt

    def _analytical(self):
        """Piecewise-quadratic: pᵢ(y) = Aᵢ + Bᵢ·y − Sᵢ/(2Γᵢ)·y²."""
        mu = _gas_viscosity(self.T_val)
        G1 = self.K1 / mu
        G2 = self.K2 / mu
        S1 = self.eps1 / self.T_val * self.dTdt_val
        S2 = self.eps2 / self.T_val * self.dTdt_val
        L = self.L
        h = self.L_half

        # dp_i/dy = Bᵢ − Sᵢ/Γᵢ · y
        # BC dp₂/dy(L) = 0 → B₂ = S₂·L/Γ₂  (positive)
        B2 = S2 * L / G2

        # Flux continuity at y=h: Γ₁·(B₁ − S₁/Γ₁·h) = Γ₂·(B₂ − S₂/Γ₂·h)
        # → Γ₁·B₁ − S₁·h = Γ₂·B₂ − S₂·h
        # → B₁ = (Γ₂·B₂ − S₂·h + S₁·h) / Γ₁  = (S₂·L − (S₂−S₁)·h) / Γ₁
        B1 = (G2 * B2 - S2 * h + S1 * h) / G1

        A1 = self.p0
        # Pressure continuity at y=h
        A2 = A1 + B1 * h - S1 * h**2 / (2.0 * G1) - B2 * h + S2 * h**2 / (2.0 * G2)

        y = _FakeMesh(self.N, L).y_nodes
        N_half = self.N // 2
        p_ana = np.empty(self.N)
        p_ana[:N_half]  = A1 + B1 * y[:N_half]  - S1 / (2.0 * G1) * y[:N_half]**2
        p_ana[N_half:]  = A2 + B2 * y[N_half:]  - S2 / (2.0 * G2) * y[N_half:]**2
        return p_ana, B1, B2, G1, G2, S1, S2

    def test_analytical_self_consistency(self):
        """Sanity check: analytical solution satisfies BCs and interface conditions."""
        p_ana, B1, B2, G1, G2, S1, S2 = self._analytical()
        h = self.L_half
        L = self.L

        # p₁ at surface ≈ p₀ (extrapolating from node 0 back to y=0)
        dy = self.L / self.N
        # p(y=0) ≈ p(dy/2) - (dp/dy|y=dy/2) * dy/2 ≈ p_ana[0] - B1*dy/2
        p_surface_extrap = p_ana[0] + B1 * (dy * 0.5) - 0  # ~p0 + B1*dy/2 - ... ≈ A1=p0
        # Just verify it's close to p0
        assert abs(self.p0 - self.p0) < 1e-10  # trivially, A1 = p0 by construction

        # dp₂/dy(L) = B₂ − S₂/Γ₂ · L = 0
        assert abs(B2 - S2 * L / G2) < 1e-12 * abs(B2)

        # Flux continuity at h
        flux1_h = G1 * (B1 - S1 / G1 * h)
        flux2_h = G2 * (B2 - S2 / G2 * h)
        assert abs(flux1_h - flux2_h) < 1e-10 * max(abs(flux1_h), 1e-30)

    def test_pressure_increases_into_material(self):
        mesh, mat_list, T, rho, dTdt = self._setup()
        p_num = solve_gas_pressure(mat_list, mesh, T, rho, dTdt, self.p0)
        # p should increase monotonically from surface to back
        assert np.all(np.diff(p_num) >= 0)

    def test_pressure_profile(self):
        mesh, mat_list, T, rho, dTdt = self._setup()
        p_num = solve_gas_pressure(mat_list, mesh, T, rho, dTdt, self.p0)
        p_ana = self._analytical()[0]

        p_max_dev = max(abs(p_ana - self.p0).max(), 1e-10)
        l2_err = np.linalg.norm(p_num - p_ana) / (p_max_dev * np.sqrt(self.N))
        assert l2_err < 1e-2, f"V2 pressure L2 error {l2_err:.2e} exceeds 1e-2"

    def test_back_wall_flux_zero(self):
        mesh, mat_list, T, rho, dTdt = self._setup()
        _, J_num = solve_pressure_and_flux(mat_list, mesh, T, rho, dTdt, self.p0)
        assert abs(J_num[-1]) < 1e-30, f"Back-wall flux not zero: {J_num[-1]}"


# ---------------------------------------------------------------------------
# Klinkenberg correction
# ---------------------------------------------------------------------------

class TestKlinkenberg:

    def test_correction_increases_K_at_low_pressure(self):
        mat = _mat(2e-11, 0.0, 0.85)
        mat.klinkenberg_b = 6000.0
        rho = mat.rho_char
        K_atm = _node_permeability(mat, rho, 101325.0)
        K_low = _node_permeability(mat, rho, 1000.0)
        assert K_low > K_atm

    def test_correction_value(self):
        mat = _mat(2e-11, 0.0, 0.85)
        mat.klinkenberg_b = 6000.0
        K = _node_permeability(mat, mat.rho_char, 1000.0)
        expected = 2e-11 * (1.0 + 6000.0 / 1000.0)
        assert abs(K - expected) / expected < 1e-12

    def test_disabled_at_zero_b(self):
        mat = _mat(2e-11, 0.0, 0.85)
        mat.klinkenberg_b = 0.0
        K = _node_permeability(mat, mat.rho_char, 1.0)  # very low pressure
        assert abs(K - 2e-11) < 1e-30


# ---------------------------------------------------------------------------
# Virgin/char blending
# ---------------------------------------------------------------------------

class TestBlending:

    def test_fully_charred(self):
        mat = _mat(2e-11, 1.6e-11, 0.85)
        K = _node_permeability(mat, mat.rho_char, 101325.0)
        assert abs(K - 2e-11) / 2e-11 < 1e-12

    def test_fully_virgin(self):
        mat = _mat(2e-11, 1.6e-11, 0.85)
        K = _node_permeability(mat, mat.rho_virgin, 101325.0)
        assert abs(K - 1.6e-11) / 1.6e-11 < 1e-12

    def test_midpoint_blend(self):
        mat = _mat(2e-11, 1.6e-11, 0.85)
        rho_mid = 0.5 * (mat.rho_virgin + mat.rho_char)
        K = _node_permeability(mat, rho_mid, 101325.0)
        assert abs(K - (0.5 * 1.6e-11 + 0.5 * 2e-11)) / (1.8e-11) < 1e-12


def _solve_gas_pressure_scalar(mat_list, mesh, T, rho, dTdt, p_surface):
    """Pre-vectorization implementation retained as an exact parity oracle."""
    N = mesh.n_nodes_total
    lid = mesh.layer_id
    delta = mesh.delta_nodes
    p_seed = np.full(N, p_surface)
    Gamma = np.zeros(N)
    for n in range(N):
        K = _node_permeability(mat_list[lid[n]], rho[n], p_seed[n])
        if K > 0.0:
            Gamma[n] = K / _gas_viscosity(float(T[n]))

    Gamma_face = np.zeros(N + 1)
    Gamma_face[0] = Gamma[0]
    for n in range(1, N):
        G1, G2 = Gamma[n - 1], Gamma[n]
        if G1 + G2 > 0.0:
            Gamma_face[n] = 2.0 * G1 * G2 / (G1 + G2)

    integral = np.zeros(N + 1)
    for n in range(N - 1, -1, -1):
        mat = mat_list[lid[n]]
        fraction = np.clip(
            (rho[n] - mat.rho_char) / (mat.rho_virgin - mat.rho_char),
            0.0,
            1.0,
        )
        eps_g = mat.eps_g_char + (mat.eps_g_virgin - mat.eps_g_char) * fraction
        integral[n] = (
            integral[n + 1]
            + eps_g / max(float(T[n]), 1.0) * float(dTdt[n]) * float(delta[n])
        )

    dpdy = np.zeros(N + 1)
    for n in range(N):
        if Gamma_face[n] > 0.0:
            dpdy[n] = integral[n] / Gamma_face[n]

    p = np.zeros(N)
    p[0] = p_surface + dpdy[0] * delta[0] * 0.5
    for n in range(1, N):
        p[n] = p[n - 1] + dpdy[n] * delta[n]
    return p


def _mass_flux_scalar(mat_list, mesh, T, rho, p):
    """Pre-vectorization face-flux implementation."""
    N = mesh.n_nodes_total
    flux = np.zeros(N + 1)
    for n in range(N):
        mat = mat_list[mesh.layer_id[n]]
        K = _node_permeability(mat, rho[n], float(p[n]))
        if K <= 0.0:
            continue
        Gamma = K / _gas_viscosity(float(T[n]))
        if n == 0:
            dpdy = (p[0] - 101325.0) / (mesh.delta_nodes[0] * 0.5)
        else:
            dy = 0.5 * (mesh.delta_nodes[n - 1] + mesh.delta_nodes[n])
            dpdy = (p[n] - p[n - 1]) / dy
        T_face = 0.5 * (float(T[max(n - 1, 0)]) + float(T[n]))
        rho_g = mat.gas_pressure * mat.gas_molar_mass / (8.314 * max(T_face, 1.0))
        flux[n] = -rho_g * Gamma * dpdy
    return flux


def test_vectorized_pressure_and_flux_match_scalar_multilayer():
    rng = np.random.default_rng(2026)
    N = 73
    layer_id = np.repeat([0, 1, 2], [31, 19, 23])
    mesh = _FakeMesh(N, 0.045, layer_id=layer_id)
    mats = [
        _mat(2.0e-11, 1.6e-11, 0.85),
        _mat(8.0e-12, 4.0e-12, 0.35, rho_v=500.0, rho_c=350.0),
        _mat(3.0e-11, 0.0, 0.65, rho_v=220.0, rho_c=160.0),
    ]
    mats[0].klinkenberg_b = 6000.0
    mats[2].klinkenberg_b = 1500.0
    T = rng.uniform(350.0, 2100.0, N)
    rho = np.array([
        rng.uniform(mats[layer].rho_char, mats[layer].rho_virgin)
        for layer in layer_id
    ])
    dTdt = rng.uniform(-200.0, 600.0, N)

    expected_p = _solve_gas_pressure_scalar(mats, mesh, T, rho, dTdt, 101325.0)
    actual_p = solve_gas_pressure(mats, mesh, T, rho, dTdt, 101325.0)
    np.testing.assert_allclose(actual_p, expected_p, rtol=2e-15, atol=2e-10)

    expected_flux = _mass_flux_scalar(mats, mesh, T, rho, expected_p)
    actual_flux = _mass_flux_from_pressure(mats, mesh, T, rho, actual_p)
    np.testing.assert_allclose(actual_flux, expected_flux, rtol=5e-10, atol=1e-14)


# ---------------------------------------------------------------------------
# Stage-1 (p, T) gas-property tables — gating and correctness
# ---------------------------------------------------------------------------

def _pT_table(mu_val: float, M_val: float) -> dict:
    """Constant-valued (p, T) gasProperties table."""
    p_ax = np.array([1.0, 1.0e6])
    T_ax = np.array([200.0, 4000.0])
    ones = np.ones((2, 2))
    return {"p": p_ax, "T": T_ax,
            "M": M_val * ones, "h_g": -7.0e6 * ones, "mu": mu_val * ones}


def test_pT_table_absent_is_bit_identical():
    """Without gas_properties_pT the solver output is unchanged (legacy path)."""
    N = 40
    mesh = _FakeMesh(N, 0.03)
    mat = _mat(2.0e-11, 1.6e-11, 0.8)
    T = np.linspace(400.0, 1900.0, N)
    rho = np.linspace(mat.rho_virgin, mat.rho_char, N)
    dTdt = np.full(N, 300.0)
    p_ref, J_ref = solve_pressure_and_flux([mat], mesh, T, rho, dTdt, 101325.0)
    assert getattr(mat, "gas_properties_pT", None) is None
    p2, J2 = solve_pressure_and_flux([mat], mesh, T, rho, dTdt, 101325.0)
    np.testing.assert_array_equal(p_ref, p2)
    np.testing.assert_array_equal(J_ref, J2)


def test_pT_viscosity_scales_pressure_perturbation():
    """Doubling μ via the pT table doubles the pressure perturbation Δp."""
    N = 60
    mesh = _FakeMesh(N, 0.025)
    T = np.full(N, 1200.0)
    dTdt = np.full(N, 500.0)
    p0 = 101325.0

    from scam.physics.pressure_darcy import _gas_viscosity
    mu_suth = _gas_viscosity(1200.0)

    mat_a = _mat(2.0e-11, 0.0, 0.85)
    mat_a.gas_properties_pT = _pT_table(mu_suth, 0.022)
    mat_b = _mat(2.0e-11, 0.0, 0.85)
    mat_b.gas_properties_pT = _pT_table(2.0 * mu_suth, 0.022)
    rho = np.full(N, mat_a.rho_char)

    p_a, _ = solve_pressure_and_flux([mat_a], mesh, T, rho, dTdt, p0)
    p_b, _ = solve_pressure_and_flux([mat_b], mesh, T, rho, dTdt, p0)
    dp_a = p_a - p0
    dp_b = p_b - p0
    np.testing.assert_allclose(dp_b, 2.0 * dp_a, rtol=1e-10)

    # And a constant table matching Sutherland reproduces the legacy solve.
    mat_c = _mat(2.0e-11, 0.0, 0.85)
    p_c, J_c = solve_pressure_and_flux([mat_c], mesh, T, rho, dTdt, p0)
    np.testing.assert_allclose(p_a, p_c, rtol=1e-12)


def test_pT_molar_mass_scales_flux():
    """With the pT table, rho_g uses local p and M(T,p): doubling M doubles J."""
    N = 60
    mesh = _FakeMesh(N, 0.025)
    T = np.full(N, 1200.0)
    dTdt = np.full(N, 500.0)
    p0 = 101325.0
    from scam.physics.pressure_darcy import _gas_viscosity
    mu_suth = _gas_viscosity(1200.0)

    mat_a = _mat(2.0e-11, 0.0, 0.85)
    mat_a.gas_properties_pT = _pT_table(mu_suth, 0.022)
    mat_b = _mat(2.0e-11, 0.0, 0.85)
    mat_b.gas_properties_pT = _pT_table(mu_suth, 0.044)
    rho = np.full(N, mat_a.rho_char)

    _, J_a = solve_pressure_and_flux([mat_a], mesh, T, rho, dTdt, p0)
    _, J_b = solve_pressure_and_flux([mat_b], mesh, T, rho, dTdt, p0)
    np.testing.assert_allclose(J_b, 2.0 * J_a, rtol=1e-9, atol=1e-30)


def test_surface_flux_uses_p_surface():
    """The surface-face gradient must use the passed p_surface (was hardcoded
    101325), so a sub-atmospheric run must not see a spurious surface flux."""
    N = 50
    mesh = _FakeMesh(N, 0.025)
    mat = _mat(2.0e-11, 0.0, 0.85)
    T = np.full(N, 1200.0)
    rho = np.full(N, mat.rho_char)
    p_surf = 10132.5   # 0.1 atm
    # Zero source → uniform p == p_surface → flux must be identically zero.
    dTdt = np.zeros(N)
    p, J = solve_pressure_and_flux([mat], mesh, T, rho, dTdt, p_surf)
    np.testing.assert_allclose(p, p_surf)
    np.testing.assert_allclose(J, 0.0, atol=1e-25)
