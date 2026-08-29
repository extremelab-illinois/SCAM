# SPDX-License-Identifier: MIT
"""V2e — Hollow cylinder with prescribed inner-wall temperature and adiabatic outer wall.

Analytical reference (Green's function):
    T(r,t) = Tw + Σ_{m=1}^∞ C_m · φ_m(r) · exp(−α β_m² t)

where the eigenfunctions φ_m and eigenvalues β_m satisfy:

    φ_m(r) = J_0(β_m r) Y_1(β_m R2) − Y_0(β_m r) J_1(β_m R2)

    Eigenvalue equation:  J_0(β R1) Y_1(β R2) − Y_0(β R1) J_1(β R2) = 0

    Norm:  N_m = ∫_{R1}^{R2} r φ_m²(r) dr   (numerical)

    Coefficient:  C_m = (−Tw / N_m) ∫_{R1}^{R2} r φ_m(r) dr

Geometry:
    R1 = 5 mm  (inner radius, hot face)
    R2 = 55 mm (outer radius)
    Scyl = R2 − R1 = 50 mm (wall thickness)

Material properties (constant):
    ρ  = 1850 kg/m³
    cp = 2000 J/kg/K
    k  =   30 W/m/K
    α  = k / (ρ cp) ≈ 8.108 × 10⁻⁶ m²/s

Boundary / initial conditions:
    T(R1, t) = Tw = 3000 K  (inner wall, prescribed)
    dT/dr|_{R2} = 0          (outer wall, adiabatic)
    T(r, 0)    = T0 = 0 K   (uniform initial field)

Run:
    MPLBACKEND=Agg python3 examples/verification/conduction/v2e_hollow_cylinder.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import j0, j1, y0, y1

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from scam.config.boundary import BackBCConfig, BackBCType, SurfaceBCConfig, SurfaceBCType
from scam.config.geometry import GeometryConfig, GeometryType
from scam.config.material import MaterialCard
from scam.config.solver import SolverOptions
from scam.config.stack import LayerConfig, StackConfig
from scam.solvers.material_response import run


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
RHO    = 1850.0   # [kg/m³]
CP     = 2000.0   # [J/kg/K]
K      = 30.0     # [W/m/K]
ALPHA  = K / (RHO * CP)   # ≈ 8.108e-6 m²/s

T_WALL = 3000.0   # [K] inner-wall prescribed temperature
T_INIT =    0.0   # [K] uniform initial temperature

R1     = 0.005    # [m] inner radius
R2     = 0.055    # [m] outer radius
SCYL   = R2 - R1  # [m] wall thickness = 50 mm

N_NODES = 500
DT_MAX  = 0.01    # [s]

T_SNAPS = [10.0, 30.0, 60.0, 100.0]   # [s]

# Number of eigenvalues to use in the series (more → better early-time accuracy)
N_EIGEN = 300


# ---------------------------------------------------------------------------
# Analytical solution — Green's function
# ---------------------------------------------------------------------------

def _eigenvalue_func(beta):
    """Transcendental equation: J_0(β R1) Y_1(β R2) − Y_0(β R1) J_1(β R2) = 0."""
    return j0(beta * R1) * y1(beta * R2) - y0(beta * R1) * j1(beta * R2)


def _phi(beta, r):
    """Radial eigenfunction φ(β, r) = J_0(β r) Y_1(β R2) − Y_0(β r) J_1(β R2)."""
    r = np.asarray(r, dtype=float)
    return j0(beta * r) * y1(beta * R2) - y0(beta * r) * j1(beta * R2)


def _find_eigenvalues(n: int) -> np.ndarray:
    """Find the first n eigenvalues β_m > 0.

    Between consecutive poles of the transcendental function the zero of
    f(β) = J_0(β R1)Y_1(β R2) - Y_0(β R1)J_1(β R2) is found by Brent's method.

    The zeros are approximately spaced π / (R2 − R1) apart.
    """
    spacing = np.pi / SCYL          # ≈ 63 rad/m
    beta_scan = np.linspace(1.0, spacing * (n + 20), 20_000)
    f_scan = np.vectorize(_eigenvalue_func)(beta_scan)

    betas = []
    for i in range(len(f_scan) - 1):
        if f_scan[i] * f_scan[i + 1] < 0:
            try:
                beta_m = brentq(_eigenvalue_func, beta_scan[i], beta_scan[i + 1],
                                 xtol=1e-12, rtol=1e-12)
                betas.append(beta_m)
                if len(betas) == n:
                    break
            except ValueError:
                pass
    return np.array(betas)


def _build_series(n_eigen: int = N_EIGEN):
    """Pre-compute eigenvalues β_m and coefficients C_m.

    Returns (betas, coefficients, norms) as 1-D arrays of length n_eigen.
    """
    betas = _find_eigenvalues(n_eigen)

    coefficients = np.empty(len(betas))
    for i, bm in enumerate(betas):
        phi_vec = lambda r: _phi(bm, r)
        norm_integrand = lambda r: r * phi_vec(r) ** 2
        coeff_integrand = lambda r: r * phi_vec(r)

        norm, _  = quad(norm_integrand,  R1, R2, limit=200)
        proj, _  = quad(coeff_integrand, R1, R2, limit=200)

        # C_m = (T0 - Tw) / N_m * ∫ r φ_m dr  (T0 = 0)
        coefficients[i] = (-T_WALL / norm) * proj

    return betas, coefficients


# Cache the series so run_case() and greens_T() share it
_SERIES_CACHE: tuple | None = None


def _get_series():
    global _SERIES_CACHE
    if _SERIES_CACHE is None:
        print(f"  Building Green's function series ({N_EIGEN} eigenvalues) …", flush=True)
        _SERIES_CACHE = _build_series(N_EIGEN)
        print(f"  Done.  β_1 = {_SERIES_CACHE[0][0]:.4f} rad/m, "
              f"β_{N_EIGEN} = {_SERIES_CACHE[0][-1]:.1f} rad/m")
    return _SERIES_CACHE


def greens_T(r, t):
    """Evaluate the Green's function series T(r, t).

    Parameters
    ----------
    r : array_like, radial coordinate [m] in [R1, R2]
    t : float, time [s]
    """
    r = np.asarray(r, dtype=float)
    betas, coeffs = _get_series()

    T = np.full_like(r, T_WALL)
    for bm, cm in zip(betas, coeffs):
        exp_factor = np.exp(-ALPHA * bm ** 2 * t)
        if exp_factor < 1e-15:
            break
        T += cm * _phi(bm, r) * exp_factor
    return T


# ---------------------------------------------------------------------------
# Material builder
# ---------------------------------------------------------------------------

def _make_mat() -> MaterialCard:
    T_pts = np.array([1.0, 5000.0])   # wide range; T_init = 0 → use 1 K lower bound
    k_tab  = np.column_stack([T_pts, np.full(2, K)])
    cp_tab = np.column_stack([T_pts, np.full(2, CP)])
    hg_tab = np.column_stack([T_pts, np.zeros(2)])
    return MaterialCard(
        name="InertCyl", rho_virgin=RHO, rho_char=RHO, gamma_resin=0.0,
        components=[], k_virgin_table=k_tab, k_char_table=k_tab,
        cp_virgin_table=cp_tab, cp_char_table=cp_tab, h_g_table=hg_tab,
        decomposing=False,
    )


# ---------------------------------------------------------------------------
# Run SCAM
# ---------------------------------------------------------------------------

def run_case():
    """Run the hollow-cylinder case and return snapshots nearest each T_SNAP."""
    mat = _make_mat()
    geom = GeometryConfig(geometry_type=GeometryType.HOLLOW_CYLINDER, r_inner=R1)
    stack = StackConfig(layers=[LayerConfig("InertCyl", SCYL, N_NODES, 4)])
    surface_bc = SurfaceBCConfig(
        bc_type=SurfaceBCType.PRESCRIBED_TEMP, T_prescribed=lambda t: T_WALL,
    )
    back_bc = BackBCConfig(bc_type=BackBCType.ADIABATIC)
    options = SolverOptions(
        t_end=max(T_SNAPS),
        dt_init=0.02,
        dt_max=DT_MAX,
        dt_min=1e-4,
        dt_max_dT=1e4,
        output_dt=min(T_SNAPS) / 2.0,
    )

    print(f"\nGeometry: HOLLOW_CYLINDER, R1={R1*1e3:.1f} mm, R2={R2*1e3:.1f} mm, "
          f"Scyl={SCYL*1e3:.1f} mm")
    print(f"Material: ρ={RHO} kg/m³  cp={CP} J/kg/K  k={K} W/m/K  α={ALPHA:.4e} m²/s")
    print(f"N={N_NODES} nodes, dx={SCYL/(N_NODES-1)*1e3:.3f} mm, DT_MAX={DT_MAX} s")

    results = run(
        stack, {"InertCyl": mat}, {"InertCyl": None},
        geom, surface_bc, back_bc, options,
        initial_T=T_INIT, verbose=True,
    )

    saved_times = np.array([s.time for s in results.snapshots])
    snaps = {}
    for t_req in T_SNAPS:
        idx = int(np.argmin(np.abs(saved_times - t_req)))
        snaps[t_req] = results.snapshots[idx]
    return snaps


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def print_report(snaps):
    thresholds = {10.0: 3.0, 30.0: 2.0, 60.0: 1.5, 100.0: 1.0}
    # NOTE: back face dT is NOT checked here — with an adiabatic outer BC and
    # prescribed inner temperature the steady state is T=T_wall throughout the
    # cylinder.  The outer wall heats up naturally (it's supposed to).

    print("\n" + "=" * 90)
    print(f"{'t [s]':>8}  {'T(R2) [K]':>11}  {'max rel err [%]':>15}  "
          f"{'threshold [%]':>13}  status")
    print("=" * 90)
    all_pass = True
    for t_req in T_SNAPS:
        snap = snaps[t_req]
        t_act = snap.time
        # y_nodes is depth from original inner surface → r = R1 + y_nodes
        r_num = R1 + snap.mesh.y_nodes
        T_num = snap.T

        T_ana = greens_T(r_num, t_act)

        dT_ref = T_ana - T_INIT
        # Mask: 5% to 95% of current temperature span
        T_span_now = float(T_ana.max() - T_INIT)
        mask = (dT_ref > 0.05 * T_span_now) & (dT_ref < 0.95 * T_span_now)
        if mask.sum() == 0:
            mask = dT_ref > 10.0   # fallback for very early times

        if mask.sum() > 0:
            rel_err = np.abs((T_num[mask] - T_ana[mask]) / dT_ref[mask])
            max_err_pct = float(rel_err.max()) * 100.0
        else:
            max_err_pct = float("nan")

        thr = thresholds[t_req]
        ok  = (max_err_pct < thr)
        flag = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {t_req:6.1f}  {T_num[-1]:11.4f}  {max_err_pct:15.3f}  {thr:13.1f}  {flag}")

    print("=" * 90)
    print(f"Overall: {'PASS' if all_pass else 'FAIL'}")
    print(f"\nNotes:")
    print(f"  Adiabatic outer BC + prescribed inner T → steady state T(r,∞) = T_wall = {T_WALL:.0f} K")
    print(f"  Outer wall heats up (expected); only interior profile accuracy is asserted.")
    return all_pass


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_plot(snaps, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("(matplotlib not available — skipping plot)")
        return

    colors = cm.plasma(np.linspace(0.15, 0.85, len(T_SNAPS)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"V2e — Hollow cylinder, prescribed inner-wall temperature\n"
        f"R1={R1*1e3:.0f} mm, R2={R2*1e3:.0f} mm  "
        f"ρ={RHO} kg/m³  cp={CP} J/kg/K  k={K} W/m/K  "
        f"T_wall={T_WALL:.0f} K  T_init={T_INIT:.0f} K  (N={N_NODES})",
        fontsize=10,
    )

    ax = axes[0]
    for col, t_req in zip(colors, T_SNAPS):
        snap = snaps[t_req]
        r_num = R1 + snap.mesh.y_nodes
        T_num = snap.T
        T_ana = greens_T(r_num, snap.time)
        lbl = f"t={snap.time:.0f} s"
        ax.plot(r_num * 1e3, T_num, color=col, lw=2.0, label=f"SCAM {lbl}")
        ax.plot(r_num * 1e3, T_ana, color=col, lw=1.2, ls="--", label=f"Green {lbl}")
    ax.set_xlabel("Radial position r [mm]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("Profiles  (solid=SCAM, dashed=Green's function)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for col, t_req in zip(colors, T_SNAPS):
        snap = snaps[t_req]
        r_num = R1 + snap.mesh.y_nodes
        T_num = snap.T
        T_ana = greens_T(r_num, snap.time)
        dT_ref = T_ana - T_INIT
        T_span_now = float(T_ana.max() - T_INIT)
        mask = (dT_ref > 0.05 * T_span_now) & (dT_ref < 0.95 * T_span_now)
        if mask.sum() == 0:
            mask = dT_ref > 10.0
        if mask.sum() > 0:
            rel_err = np.abs((T_num[mask] - T_ana[mask]) / dT_ref[mask]) * 100.0
            ax.plot(r_num[mask] * 1e3, rel_err, color=col, lw=1.5,
                    label=f"t={snap.time:.0f} s")
    ax.axhline(2.0, color="k", ls="--", lw=1.0, label="2% guide")
    ax.set_xlabel("Radial position r [mm]")
    ax.set_ylabel("Relative error [%]")
    ax.set_title("SCAM vs Green's function  (5%–95% of ΔT span)")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("V2e — Hollow cylinder, prescribed inner wall temperature")
    print(f"Green's function: T(r,t) = {T_WALL} + Σ C_m φ_m(r) exp(−α β_m² t)")

    # Build series first (shared by run diagnostics and plot)
    _get_series()

    snaps = run_case()
    all_pass = print_report(snaps)

    out_png = Path(__file__).parent / "v2e_hollow_cylinder.png"
    make_plot(snaps, out_png)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
