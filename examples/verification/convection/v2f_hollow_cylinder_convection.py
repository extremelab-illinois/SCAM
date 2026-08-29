# SPDX-License-Identifier: MIT
"""V2f — Hollow cylinder with convection at the inner wall and adiabatic outer wall.

Analytical reference (Green's function, Carslaw & Jaeger §7):

    T(r,t) = T_env + Σ_{m=1}^∞ C_m · φ_m(r) · exp(−α β_m² t)

Eigenfunctions (built to satisfy adiabatic outer BC automatically):

    φ_m(r) = J_0(β_m r) Y_1(β_m R2) − Y_0(β_m r) J_1(β_m R2)
    φ_m'(r) = β_m [Y_1(β_m r) J_1(β_m R2) − J_1(β_m r) Y_1(β_m R2)]

    φ_m'(R2) = 0  (satisfied automatically for all β_m)

Eigenvalue equation (Robin BC at r=R1):

    −k φ_m'(R1) + h_w φ_m(R1) = 0
    i.e.  k β_m [J_1(β_m R1) Y_1(β_m R2) − Y_1(β_m R1) J_1(β_m R2)]
           + h_w [J_0(β_m R1) Y_1(β_m R2) − Y_0(β_m R1) J_1(β_m R2)] = 0

Coefficients (Sturm-Liouville, weight r):

    N_m = ∫_{R1}^{R2} r φ_m²(r) dr
    C_m = [(T0 − T_env) / N_m] ∫_{R1}^{R2} r φ_m(r) dr   (T0 = 0)

Geometry:
    R1 = 5 mm  (inner radius), R2 = 55 mm (outer), Scyl = 50 mm

Material properties (constant):
    ρ = 1850 kg/m³, cp = 2000 J/kg/K, k = 30 W/m/K, α ≈ 8.108e-6 m²/s

Boundary / initial conditions:
    −k ∂T/∂r|_{R1} = h_w (T_env − T_w)  [convection, h_w = 3000 W/m²/K]
    T_env = 5000 K  (environment temperature)
    ∂T/∂r|_{R2} = 0  (adiabatic outer)
    T(r, 0) = T0 = 0 K  (uniform initial)

Run:
    MPLBACKEND=Agg python3 examples/verification/conduction/v2f_hollow_cylinder_convection.py
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
RHO    = 1850.0    # [kg/m³]
CP     = 2000.0    # [J/kg/K]
K      = 30.0      # [W/m/K]
ALPHA  = K / (RHO * CP)   # ≈ 8.108e-6 m²/s

H_W    = 3000.0    # [W/m²/K] inner-wall convection coefficient
T_ENV  = 5000.0    # [K]      environment temperature
T_INIT =    0.0    # [K]      uniform initial temperature

R1     = 0.005     # [m] inner radius
R2     = 0.055     # [m] outer radius
SCYL   = R2 - R1   # [m] wall thickness = 50 mm

N_NODES = 500
DT_MAX  = 0.01     # [s]

T_SNAPS = [10.0, 30.0, 60.0, 100.0]   # [s]

N_EIGEN = 300      # number of eigenvalues in the series


# ---------------------------------------------------------------------------
# Analytical solution — Green's function with Robin inner + Neumann outer BCs
# ---------------------------------------------------------------------------

def _phi(beta: float, r):
    """Eigenfunction: J_0(βr)Y_1(βR2) − Y_0(βr)J_1(βR2).
    Satisfies φ'(R2) = 0 by construction.
    """
    r = np.asarray(r, dtype=float)
    return j0(beta * r) * y1(beta * R2) - y0(beta * r) * j1(beta * R2)


def _dphi_dr(beta: float, r):
    """dφ/dr = β[Y_1(βr)J_1(βR2) − J_1(βr)Y_1(βR2)]."""
    r = np.asarray(r, dtype=float)
    return beta * (y1(beta * r) * j1(beta * R2) - j1(beta * r) * y1(beta * R2))


def _eigen_func(beta: float) -> float:
    """Eigenvalue equation: −k φ'(R1) + h_w φ(R1) = 0."""
    return -K * _dphi_dr(beta, R1) + H_W * _phi(beta, R1)


def _find_eigenvalues(n: int) -> np.ndarray:
    """Find the first n positive eigenvalues by sign-change scanning + Brent."""
    spacing = np.pi / SCYL           # ≈ 63 rad/m between consecutive zeros
    beta_scan = np.linspace(0.1, spacing * (n + 30), 20 * (n + 30) * 10)
    fv = np.vectorize(_eigen_func)(beta_scan)

    betas: list[float] = []
    for i in range(len(fv) - 1):
        if fv[i] * fv[i + 1] < 0:
            try:
                bm = brentq(_eigen_func, beta_scan[i], beta_scan[i + 1], xtol=1e-12)
                betas.append(bm)
                if len(betas) == n:
                    break
            except ValueError:
                pass
    return np.array(betas)


def _build_series(n: int = N_EIGEN):
    """Pre-compute (betas, coefficients) for the Green's function series."""
    betas = _find_eigenvalues(n)

    coeffs = np.empty(len(betas))
    for i, bm in enumerate(betas):
        norm, _ = quad(lambda r: r * _phi(bm, r) ** 2, R1, R2, limit=200)
        proj, _ = quad(lambda r: r * _phi(bm, r),      R1, R2, limit=200)
        # C_m = (T0 - T_env) / N_m * ∫ r φ_m dr
        coeffs[i] = ((T_INIT - T_ENV) / norm) * proj

    return betas, coeffs


_SERIES_CACHE: tuple | None = None


def _get_series():
    global _SERIES_CACHE
    if _SERIES_CACHE is None:
        print(f"  Building Green's function series ({N_EIGEN} eigenvalues) …", flush=True)
        _SERIES_CACHE = _build_series(N_EIGEN)
        print(f"  Done.  β_1 = {_SERIES_CACHE[0][0]:.4f} rad/m, "
              f"β_{N_EIGEN} = {_SERIES_CACHE[0][-1]:.1f} rad/m")
    return _SERIES_CACHE


def greens_T(r, t: float) -> np.ndarray:
    """Evaluate T(r, t) from the Green's function series."""
    r = np.asarray(r, dtype=float)
    betas, coeffs = _get_series()

    T = np.full_like(r, T_ENV)
    for bm, cm in zip(betas, coeffs):
        exp_factor = np.exp(-ALPHA * bm ** 2 * t)
        if exp_factor < 1e-15:
            break
        T += cm * _phi(bm, r) * exp_factor
    return T


# ---------------------------------------------------------------------------
# Material builder (no radiation, no decomposition)
# ---------------------------------------------------------------------------

def _make_mat() -> MaterialCard:
    T_pts = np.array([1.0, 6000.0])
    k_tab  = np.column_stack([T_pts, np.full(2, K)])
    cp_tab = np.column_stack([T_pts, np.full(2, CP)])
    hg_tab = np.column_stack([T_pts, np.zeros(2)])
    return MaterialCard(
        name="InertCyl", rho_virgin=RHO, rho_char=RHO, gamma_resin=0.0,
        components=[], k_virgin_table=k_tab, k_char_table=k_tab,
        cp_virgin_table=cp_tab, cp_char_table=cp_tab, h_g_table=hg_tab,
        emissivity=0.0,    # no radiation
        decomposing=False,
    )


# ---------------------------------------------------------------------------
# Run SCAM
# ---------------------------------------------------------------------------

def run_case():
    """Run the hollow-cylinder convection case."""
    mat = _make_mat()
    geom = GeometryConfig(geometry_type=GeometryType.HOLLOW_CYLINDER, r_inner=R1)
    stack = StackConfig(layers=[LayerConfig("InertCyl", SCYL, N_NODES, 4)])
    # ENERGY_BALANCE with alpha_conv=h_w, T_aw=T_env, no radiation → Robin BC
    surface_bc = SurfaceBCConfig(
        bc_type=SurfaceBCType.ENERGY_BALANCE,
        alpha_conv=H_W,
        T_aw=T_ENV,
        emissivity=0.0,     # override any material card emissivity
        T_rad_in=0.0,
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

    print(f"\nGeometry: HOLLOW_CYLINDER, R1={R1*1e3:.1f} mm, R2={R2*1e3:.1f} mm")
    print(f"Material: ρ={RHO} kg/m³  cp={CP} J/kg/K  k={K} W/m/K  α={ALPHA:.4e} m²/s")
    print(f"BC inner: convection h_w={H_W} W/m²/K, T_env={T_ENV} K")
    print(f"BC outer: adiabatic")
    print(f"Biot number Bi = h_w·R1/k = {H_W*R1/K:.4f}")
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
    dx = SCYL / (N_NODES - 1)

    print("\n" + "=" * 100)
    print(f"{'t [s]':>8}  {'T_wall [K]':>10}  {'T(R2) [K]':>10}  "
          f"{'max rel err [%]':>15}  {'threshold [%]':>13}  status")
    print("=" * 100)
    all_pass = True
    for t_req in T_SNAPS:
        snap = snaps[t_req]
        t_act = snap.time
        r_num = R1 + snap.mesh.y_nodes
        T_num = snap.T

        T_ana  = greens_T(r_num, t_act)
        dT_ref = T_ana - T_INIT

        # Compare in the interior band (5%–95% of current span, excluding surface pin & cold tail)
        T_span_now = float(T_ana[0] - T_INIT)   # max at inner wall (surface heating from inside)
        mask = (dT_ref > 0.05 * T_span_now) & (dT_ref < 0.95 * T_span_now)
        if mask.sum() == 0:
            mask = dT_ref > 10.0

        if mask.sum() > 0:
            rel_err = np.abs((T_num[mask] - T_ana[mask]) / dT_ref[mask])
            max_err_pct = float(rel_err.max()) * 100.0
        else:
            max_err_pct = float("nan")

        thr = thresholds[t_req]
        ok  = max_err_pct < thr
        flag = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {t_req:6.1f}  {T_num[0]:10.3f}  {T_num[-1]:10.4f}  "
              f"{max_err_pct:15.3f}  {thr:13.1f}  {flag}")

    print("=" * 100)
    print(f"Overall: {'PASS' if all_pass else 'FAIL'}")
    print(f"\nNotes:")
    print(f"  Biot number Bi = h_w·R1/k = {H_W*R1/K:.4f}")
    print(f"  Steady state: T(r,∞) = T_env = {T_ENV:.0f} K (uniform, adiabatic outer)")
    print(f"  Inner-wall temperature rises toward T_env through convection (Robin BC).")
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

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        f"V2f — Hollow cylinder, convection at inner wall\n"
        f"R1={R1*1e3:.0f} mm, R2={R2*1e3:.0f} mm  "
        f"h_w={H_W:.0f} W/m²/K  T_env={T_ENV:.0f} K  T_init={T_INIT:.0f} K  "
        f"ρ={RHO:.0f} kg/m³  k={K:.0f} W/m/K  (N={N_NODES})",
        fontsize=9,
    )

    # Panel 1 — temperature profiles
    ax = axes[0]
    for col, t_req in zip(colors, T_SNAPS):
        snap = snaps[t_req]
        r_num = R1 + snap.mesh.y_nodes
        T_num = snap.T
        T_ana = greens_T(r_num, snap.time)
        lbl = f"t={snap.time:.0f} s"
        ax.plot(r_num * 1e3, T_num, color=col, lw=2.0, label=f"SCAM {lbl}")
        ax.plot(r_num * 1e3, T_ana, color=col, lw=1.2, ls="--", label=f"Green {lbl}")
    ax.axhline(T_ENV, color="grey", ls=":", lw=1, label=f"T_env = {T_ENV:.0f} K")
    ax.set_xlabel("Radial position r [mm]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("Profiles  (solid=SCAM, dashed=Green's function)")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 2 — relative error vs Green's function (interior band)
    ax = axes[1]
    for col, t_req in zip(colors, T_SNAPS):
        snap = snaps[t_req]
        r_num = R1 + snap.mesh.y_nodes
        T_num = snap.T
        T_ana = greens_T(r_num, snap.time)
        dT_ref = T_ana - T_INIT
        T_span_now = float(T_ana[0] - T_INIT)
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

    # Panel 3 — inner-wall temperature vs time
    ax = axes[2]
    t_dense = np.linspace(0.5, max(T_SNAPS), 300)
    T_wall_num = np.array([snaps[t_req].T[0] for t_req in T_SNAPS])
    T_wall_ana = np.array([greens_T(np.array([R1]), t)[0] for t in t_dense])
    ax.plot(t_dense, T_wall_ana, "k--", lw=1.5, label="Green's function T(R1)")
    ax.scatter([snaps[t].time for t in T_SNAPS], T_wall_num, color="tab:red",
               zorder=5, label="SCAM T_wall snapshots")
    ax.axhline(T_ENV, color="grey", ls=":", lw=1, label=f"T_env = {T_ENV:.0f} K")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Inner-wall temperature [K]")
    ax.set_title("T_wall vs time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("V2f — Hollow cylinder, convection at inner wall, adiabatic outer")
    print(f"Green's function: T(r,t) = {T_ENV} + Σ C_m φ_m(r) exp(−α β_m² t)")

    _get_series()
    snaps = run_case()
    all_pass = print_report(snaps)

    out_png = Path(__file__).parent / "v2f_hollow_cylinder_convection.png"
    make_plot(snaps, out_png)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
