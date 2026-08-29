#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""V1 Darcy pressure verification — single-layer, constant properties.

Compares SCAM's quasi-steady pressure solver against the closed-form
parabolic solution for a uniform TACOT char slab.

Governing PDE:  d/dy(Γ dp/dy) = S
  Γ = K/μ(T)            gas mobility [m²/(Pa·s)]
  S = ε_g/T · dT/dt     thermal-expansion source [1/s]

BCs: p(0) = p₀ (surface), dp/dy(L) = 0 (back wall, no-flow)

Analytical solution:
  dp/dy(y) = S·(L−y)/Γ           > 0  (pressure increases into material)
  p(y)     = p₀ + S/(2Γ)·(2Ly−y²)    [p(L) > p₀; max deviation at back wall]
  J_g(y)   = −ρ_g·S·(L−y)             [negative = toward surface]
  v_g(y)   = J_g / (ρ_g·ε_g) = −(dT/dt/T)·(L−y)   [m/s, independent of K]

Run:
    MPLBACKEND=Agg python3 examples/verification/darcy_pressure/v1_analytical.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow running from project root or this directory
_SCAM_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_SCAM_ROOT))

from scam.physics.pressure_darcy import (
    _gas_viscosity,
    solve_pressure_and_flux,
)
from scam.config.material import MaterialCard, ComponentCard

# ---------------------------------------------------------------------------
# Case parameters (fully-charred TACOT char)
# ---------------------------------------------------------------------------
N      = 100           # nodes
L      = 0.025         # m
K      = 2.0e-11       # m²  (TACOT char permeability, from XLS)
eps_g  = 0.85          # gas porosity of char
T_val  = 1200.0        # K   (isothermal snapshot)
dTdt   = 500.0         # K/s (uniform heating rate)
p0     = 101325.0      # Pa  (surface pressure)
R_univ = 8.314
M_gas  = 0.022         # kg/mol

# ---------------------------------------------------------------------------
# Minimal mesh stub
# ---------------------------------------------------------------------------
class _Mesh:
    def __init__(self, N, L):
        self.n_nodes_total = N
        dy = L / N
        self.delta_nodes = np.full(N, dy)
        self.area_nodes  = np.ones(N)
        self.layer_id    = np.zeros(N, dtype=int)
        self.y_nodes     = np.arange(N) * dy + dy * 0.5

mesh = _Mesh(N, L)

# ---------------------------------------------------------------------------
# Minimal MaterialCard (fully charred: rho = rho_char → K_eff = K_char)
# ---------------------------------------------------------------------------
comp = ComponentCard(name="resin", rho_0=240.0, rho_r=200.0,
                     A_rate=1e4, E_act=1e5, m_exp=3.0)
mat = MaterialCard(
    name="TACOT_char",
    rho_virgin=300.0, rho_char=200.0, gamma_resin=0.8,
    components=[comp],
    permeability=K, permeability_virgin=0.0,
    eps_g_virgin=eps_g, eps_g_char=eps_g,
    gas_pressure=p0, gas_molar_mass=M_gas,
)

T_arr    = np.full(N, T_val)
rho_arr  = np.full(N, mat.rho_char)
dTdt_arr = np.full(N, dTdt)

# ---------------------------------------------------------------------------
# Numerical solution
# ---------------------------------------------------------------------------
p_num, J_num = solve_pressure_and_flux([mat], mesh, T_arr, rho_arr, dTdt_arr, p0)

# Face positions (N+1 faces for N nodes)
y_face = np.arange(N + 1) * (L / N)
# Interstitial velocity: v = J_g / (ρ_g · ε_g)
rho_g_face = p0 * M_gas / (R_univ * T_val)
v_g_num    = J_num / (rho_g_face * eps_g)

# ---------------------------------------------------------------------------
# Analytical solution
# ---------------------------------------------------------------------------
mu    = _gas_viscosity(T_val)
Gamma = K / mu
S     = eps_g / T_val * dTdt
rho_g = p0 * M_gas / (R_univ * T_val)

p_ana  = p0 + S / (2.0 * Gamma) * (2.0 * L * mesh.y_nodes - mesh.y_nodes**2)
J_ana  = -rho_g * S * (L - y_face)
v_ana  = -(dTdt / T_val) * (L - y_face)  # = J_ana / (rho_g * eps_g)

# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------
p_dev   = S / (2.0 * Gamma) * L**2          # max pressure deviation from p0 [Pa]
J_max   = abs(J_ana).max()
v_max   = abs(v_ana).max()

l2_p = np.linalg.norm(p_num - p_ana)     / (p_dev  * np.sqrt(N))
l2_J = np.linalg.norm(J_num - J_ana)     / (J_max  * np.sqrt(N + 1))
l2_v = np.linalg.norm(v_g_num - v_ana)   / (v_max  * np.sqrt(N + 1))

print("=== V1 Darcy pressure verification (single-layer) ===")
print(f"  Material : TACOT char  K={K:.2e} m²  ε_g={eps_g}")
print(f"  T        : {T_val:.0f} K    dT/dt = {dTdt:.0f} K/s")
print(f"  μ(T)     : {mu:.3e} Pa·s    Γ = K/μ = {Gamma:.3e} m²/(Pa·s)")
print(f"  Source S : {S:.4e} 1/s    Δp_max = {p_dev*1e3:.4f} mPa")
print(f"  Gas flux at surface  J_g(0) = {J_num[0]:.3e} kg/(m²·s)")
print(f"  Gas speed at surface v_g(0) = {v_g_num[0]*1e6:.3f} µm/s")
print()
print(f"  L2 error — pressure : {l2_p:.2e}  (tolerance 1e-3)")
print(f"  L2 error — flux J_g : {l2_J:.2e}  (tolerance 1e-3)")
print(f"  L2 error — speed v_g: {l2_v:.2e}  (tolerance 1e-3)")
print()

assert l2_p < 1e-3, f"Pressure L2 error {l2_p:.2e} exceeds 1e-3"
assert l2_J < 1e-3, f"Flux    L2 error {l2_J:.2e} exceeds 1e-3"
assert l2_v < 1e-3, f"Speed   L2 error {l2_v:.2e} exceeds 1e-3"
print("  All checks PASSED.")

# ---------------------------------------------------------------------------
# 3-panel figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle(
    f"V1 Darcy pressure — single-layer TACOT char\n"
    f"K={K:.1e} m²,  T={T_val:.0f} K,  dT/dt={dTdt:.0f} K/s,  N={N}",
    fontsize=10,
)

y_mm = mesh.y_nodes * 1e3
yf_mm = y_face * 1e3

# Panel 1: pressure
ax = axes[0]
ax.plot(y_mm, (p_num - p0) * 1e3, "C0-", lw=2, label="SCAM numerical")
ax.plot(y_mm, (p_ana - p0) * 1e3, "C1--", lw=2, label="Analytical")
ax.set_xlabel("Depth y  [mm]")
ax.set_ylabel("p − p_surface  [mPa]")
ax.set_title("Pressure perturbation")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 2: mass flux
ax = axes[1]
ax.plot(yf_mm, J_num * 1e6, "C0-", lw=2, label="SCAM numerical")
ax.plot(yf_mm, J_ana * 1e6, "C1--", lw=2, label="Analytical")
ax.set_xlabel("Depth y  [mm]")
ax.set_ylabel("J_g  [µg/(m²·s)]")
ax.set_title("Gas mass flux  (negative = toward surface)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 3: interstitial velocity
ax = axes[2]
ax.plot(yf_mm, v_g_num * 1e6, "C0-", lw=2, label="SCAM numerical")
ax.plot(yf_mm, v_ana * 1e6, "C1--", lw=2, label="Analytical")
ax.set_xlabel("Depth y  [mm]")
ax.set_ylabel("v_g  [µm/s]  (negative = toward surface)")
ax.set_title("Interstitial gas velocity")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = Path(__file__).parent / "v1_analytical.png"
fig.savefig(out, dpi=150)
print(f"  Saved: {out}")
