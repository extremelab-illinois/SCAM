#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""V2 Darcy pressure verification — two-layer piecewise-quadratic.

Char layer (surface side, y ∈ [0, L/2]):  K₁ = 2.0×10⁻¹¹ m², ε_g1 = 0.85
Virgin layer (back side, y ∈ [L/2, L]):   K₂ = 1.6×10⁻¹¹ m², ε_g2 = 0.80

Both layers: T = 1000 K (isothermal snapshot), dT/dt = 300 K/s.

In each layer pᵢ(y) satisfies d²p/dy² = −Sᵢ/Γᵢ, giving:
    pᵢ(y) = Aᵢ + Bᵢ·y − Sᵢ/(2Γᵢ)·y²

The four constants are found from:
  1.  p₁(0)  = p₀                   (surface Dirichlet)
  2.  dp₂/dy(L)  = 0                (back-wall no-flow)   → B₂ = S₂·L/Γ₂
  3.  p₁(h)  = p₂(h)               (pressure continuity at interface)
  4.  Γ₁·dp₁/dy(h) = Γ₂·dp₂/dy(h) (flux continuity at interface)

This case tests the solver's harmonic-mean face conductance at the material
interface and validates the layer_id dispatch mechanism.

Run:
    MPLBACKEND=Agg python3 examples/verification/darcy_pressure/v2_twolayer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SCAM_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_SCAM_ROOT))

from scam.physics.pressure_darcy import _gas_viscosity, solve_pressure_and_flux
from scam.config.material import MaterialCard, ComponentCard

# ---------------------------------------------------------------------------
# Case parameters
# ---------------------------------------------------------------------------
N      = 200            # total nodes (100 per layer)
L      = 0.05           # m  total slab thickness
L_half = 0.025          # m  interface position (exact midpoint)
K1     = 2.0e-11        # m²  char permeability
eps1   = 0.85           # char gas porosity
K2     = 1.6e-11        # m²  virgin permeability
eps2   = 0.80           # virgin gas porosity
T_val  = 1000.0         # K
dTdt   = 300.0          # K/s
p0     = 101325.0       # Pa
R_univ = 8.314
M_gas  = 0.022          # kg/mol

# ---------------------------------------------------------------------------
# Mesh with two layers
# ---------------------------------------------------------------------------
class _Mesh:
    def __init__(self, N, L, layer_id):
        self.n_nodes_total = N
        dy = L / N
        self.delta_nodes = np.full(N, dy)
        self.area_nodes  = np.ones(N)
        self.layer_id    = layer_id
        self.y_nodes     = np.arange(N) * dy + dy * 0.5

N_half = N // 2
layer_id = np.array([0] * N_half + [1] * N_half, dtype=int)
mesh = _Mesh(N, L, layer_id)

# ---------------------------------------------------------------------------
# Material cards (K_virgin=0 → no blending, use K_char; both fully charred)
# ---------------------------------------------------------------------------
def _make_mat(K, eps, name):
    comp = ComponentCard(name="resin", rho_0=240.0, rho_r=200.0,
                         A_rate=1e4, E_act=1e5, m_exp=3.0)
    return MaterialCard(
        name=name, rho_virgin=300.0, rho_char=200.0, gamma_resin=0.8,
        components=[comp],
        permeability=K, permeability_virgin=0.0,
        eps_g_virgin=eps, eps_g_char=eps,
        gas_pressure=p0, gas_molar_mass=M_gas,
    )

mat1 = _make_mat(K1, eps1, "TACOT_char")
mat2 = _make_mat(K2, eps2, "TACOT_virgin")
mat_list = [mat1, mat2]

T_arr    = np.full(N, T_val)
rho_arr  = np.concatenate([np.full(N_half, mat1.rho_char),
                            np.full(N_half, mat2.rho_char)])
dTdt_arr = np.full(N, dTdt)

# ---------------------------------------------------------------------------
# Numerical solution
# ---------------------------------------------------------------------------
p_num, J_num = solve_pressure_and_flux(mat_list, mesh, T_arr, rho_arr, dTdt_arr, p0)

y_face  = np.arange(N + 1) * (L / N)
rho_g   = p0 * M_gas / (R_univ * T_val)

# Interstitial velocity (layer-specific eps_g)
eps_face = np.where(y_face <= L_half, eps1, eps2)
v_g_num  = J_num / (rho_g * eps_face)

# ---------------------------------------------------------------------------
# Analytical solution (piecewise-quadratic)
# ---------------------------------------------------------------------------
mu = _gas_viscosity(T_val)
G1 = K1 / mu
G2 = K2 / mu
S1 = eps1 / T_val * dTdt
S2 = eps2 / T_val * dTdt
h  = L_half

# BC at y=L: dp₂/dy(L) = 0 → B₂ = S₂·L/Γ₂
B2 = S2 * L / G2
# Flux continuity: Γ₁·(B₁ − S₁/Γ₁·h) = Γ₂·(B₂ − S₂/Γ₂·h)
B1 = (G2 * B2 - S2 * h + S1 * h) / G1
A1 = p0
# Pressure continuity at y=h
A2 = A1 + B1 * h - S1 * h**2 / (2.0 * G1) - B2 * h + S2 * h**2 / (2.0 * G2)

y_n = mesh.y_nodes
p_ana = np.empty(N)
p_ana[:N_half] = A1 + B1 * y_n[:N_half] - S1 / (2.0 * G1) * y_n[:N_half]**2
p_ana[N_half:] = A2 + B2 * y_n[N_half:] - S2 / (2.0 * G2) * y_n[N_half:]**2

# Analytical flux: J_g(y_face) = −Γᵢ(y)·ρ_g·dp/dy(y_face)
dp_ana = np.empty(N + 1)
for i, y in enumerate(y_face):
    if y <= L_half:
        dp_ana[i] = B1 - S1 / G1 * y
    else:
        dp_ana[i] = B2 - S2 / G2 * y
J_ana   = -rho_g * np.where(y_face <= L_half, G1, G2) * dp_ana
v_ana   = J_ana / (rho_g * eps_face)

# Verify BC: dp₂/dy(L) ≈ 0
dp_back_err = abs(B2 - S2 / G2 * L)
assert dp_back_err < 1e-8, f"Analytical back-wall BC error: {dp_back_err:.2e}"

# Verify flux continuity at interface
f1 = G1 * (B1 - S1 / G1 * h)
f2 = G2 * (B2 - S2 / G2 * h)
assert abs(f1 - f2) < 1e-10 * max(abs(f1), 1e-30), f"Flux continuity error: {abs(f1-f2):.2e}"

# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------
p_max_dev = max(abs(p_ana - p0).max(), 1e-30)
J_max     = max(abs(J_ana).max(), 1e-30)
v_max     = max(abs(v_ana).max(), 1e-30)

l2_p = np.linalg.norm(p_num - p_ana)   / (p_max_dev * np.sqrt(N))
l2_J = np.linalg.norm(J_num - J_ana)   / (J_max     * np.sqrt(N + 1))
l2_v = np.linalg.norm(v_g_num - v_ana) / (v_max     * np.sqrt(N + 1))

p_back_ana = p_ana[-1]
p_interface = 0.5 * (p_ana[N_half - 1] + p_ana[N_half])

print("=== V2 Darcy pressure verification (two-layer) ===")
print(f"  Char layer  : K₁={K1:.1e} m², ε_g1={eps1}")
print(f"  Virgin layer: K₂={K2:.1e} m², ε_g2={eps2}")
print(f"  T={T_val:.0f} K,  dT/dt={dTdt:.0f} K/s,  p₀={p0:.0f} Pa")
print()
print(f"  μ(T) = {mu:.3e} Pa·s")
print(f"  B₁ = {B1:.4f} Pa/m   B₂ = {B2:.4f} Pa/m")
print(f"  A₁ = {A1:.3f} Pa    A₂ = {A2:.3f} Pa")
print(f"  p(interface) ≈ {p_interface:.3f} Pa")
print(f"  p(back wall) ≈ {p_back_ana:.3f} Pa  (Δp = {(p_back_ana-p0)*1e3:.4f} mPa)")
print()
print(f"  L2 error — pressure: {l2_p:.2e}  (tolerance 1e-2)")
print(f"  L2 error — flux J_g: {l2_J:.2e}  (tolerance 1e-2)")
print(f"  L2 error — speed v_g: {l2_v:.2e} (tolerance 1e-2)")
print()

assert l2_p < 1e-2, f"Pressure L2 error {l2_p:.2e} exceeds 1e-2"
assert l2_J < 1e-2, f"Flux    L2 error {l2_J:.2e} exceeds 1e-2"
assert l2_v < 1e-2, f"Speed   L2 error {l2_v:.2e} exceeds 1e-2"
print("  All checks PASSED.")

# ---------------------------------------------------------------------------
# 3-panel figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle(
    f"V2 Darcy pressure — two-layer (char + virgin TACOT)\n"
    f"K₁={K1:.1e} m²  K₂={K2:.1e} m²  T={T_val:.0f} K  dT/dt={dTdt:.0f} K/s",
    fontsize=10,
)

y_mm  = y_n * 1e3
yf_mm = y_face * 1e3
interface_mm = L_half * 1e3

for ax in axes:
    ax.axvline(interface_mm, color="gray", lw=1, ls=":", label="Interface")

# Panel 1: pressure
ax = axes[0]
ax.plot(y_mm, (p_num - p0) * 1e3, "C0-",  lw=2, label="SCAM numerical")
ax.plot(y_mm, (p_ana - p0) * 1e3, "C1--", lw=2, label="Analytical")
ax.set_xlabel("Depth y  [mm]")
ax.set_ylabel("p − p_surface  [mPa]")
ax.set_title("Pressure perturbation")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.text(0.98, 0.02, "char | virgin", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=7, color="gray")

# Panel 2: mass flux
ax = axes[1]
ax.plot(yf_mm, J_num * 1e6, "C0-",  lw=2, label="SCAM numerical")
ax.plot(yf_mm, J_ana * 1e6, "C1--", lw=2, label="Analytical")
ax.set_xlabel("Depth y  [mm]")
ax.set_ylabel("J_g  [µg/(m²·s)]")
ax.set_title("Gas mass flux  (negative = toward surface)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 3: interstitial velocity
ax = axes[2]
ax.plot(yf_mm, v_g_num * 1e6, "C0-",  lw=2, label="SCAM numerical")
ax.plot(yf_mm, v_ana * 1e6,   "C1--", lw=2, label="Analytical")
ax.set_xlabel("Depth y  [mm]")
ax.set_ylabel("v_g  [µm/s]  (negative = toward surface)")
ax.set_title("Interstitial gas velocity")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = Path(__file__).parent / "v2_twolayer.png"
fig.savefig(out, dpi=150)
print(f"  Saved: {out}")
