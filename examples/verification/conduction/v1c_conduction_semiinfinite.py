# SPDX-License-Identifier: MIT
"""V1c — Semi-infinite slab verification with prescribed wall temperature.

Analytical reference:
    T(y, t) = T_init + (T_wall − T_init) · erfc(y / (2 √(α t)))

Material properties (constant, space- and time-independent):
    ρ  = 1850  kg/m³
    cp = 2000  J/kg/K
    k  =   30  W/m/K
    α  = k / (ρ cp) ≈ 8.108 × 10⁻⁶ m²/s

Boundary / initial conditions:
    T_wall = 1000 K  (front face, prescribed)
    T_init =  300 K  (uniform initial field; also back-face value)

Slab thickness:
    L = 0.5 m.  At t = 100 s the thermal penetration depth where ΔT > 0.01 K
    is ≈ 0.17 m (ξ = y / (2√(α t)) ≈ 3 → erfc(3) ≈ 2.2 × 10⁻⁵ × 700 K ≈ 0.02 K).
    With L = 0.5 m the back face sits at ξ ≈ 8.8, so its temperature change is
    negligibly small throughout the entire run.

Run:
    MPLBACKEND=Agg python3 examples/verification/conduction/v1c_conduction_semiinfinite.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import erfc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from scam.config.boundary import BackBCConfig, BackBCType, SurfaceBCConfig, SurfaceBCType
from scam.config.geometry import GeometryConfig, GeometryType
from scam.config.material import MaterialCard
from scam.config.solver import SolverOptions
from scam.config.stack import LayerConfig, StackConfig
from scam.solvers.material_response import run


# ---------------------------------------------------------------------------
# Physical constants (the values specified for this verification)
# ---------------------------------------------------------------------------
RHO   = 1850.0   # [kg/m³]
CP    = 2000.0   # [J/kg/K]
K     = 30.0     # [W/m/K]
ALPHA = K / (RHO * CP)   # ≈ 8.108e-6 m²/s
T_WALL = 1000.0  # [K] prescribed surface temperature
T_INIT =  300.0  # [K] uniform initial temperature

# Slab geometry — thick enough that the back face never heats up
THICKNESS = 0.50   # [m]
N_NODES   = 1001    # → dx = 0.5 mm
DT_MAX    = 0.01   # [s]

# Snapshot times at which to compare against the analytical solution
T_SNAPS = [10.0, 30.0, 60.0, 100.0]   # [s]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mat() -> MaterialCard:
    """Constant-property, non-decomposing material card."""
    T_pts = np.array([200.0, 4000.0])
    k_tab  = np.column_stack([T_pts, np.full(2, K)])
    cp_tab = np.column_stack([T_pts, np.full(2, CP)])
    hg_tab = np.column_stack([T_pts, np.zeros(2)])
    return MaterialCard(
        name="Inert", rho_virgin=RHO, rho_char=RHO, gamma_resin=0.0,
        components=[], k_virgin_table=k_tab, k_char_table=k_tab,
        cp_virgin_table=cp_tab, cp_char_table=cp_tab, h_g_table=hg_tab,
        decomposing=False,
    )


def _erfc_solution(y, t):
    """Semi-infinite limit: T(y,t) = T_init + (T_wall−T_init)·erfc(y/(2√(αt)))."""
    xi = y / (2.0 * np.sqrt(ALPHA * t))
    return T_INIT + (T_WALL - T_INIT) * erfc(xi)


def _fourier_solution(y, t, n_terms: int = 500):
    """Finite-slab Fourier series (prescribed T_wall at y=0, adiabatic at y=L).

    T(y,t) = T_init + (T_wall−T_init)·[1 − (4/π) Σ_{n=1,3,5,...}
             (1/n) exp(−(nπ/2)² αt/L²) sin(nπy/(2L))]

    Exact for the actual simulation BC.  Converges to erfc as L→∞; at
    Fo = αt/L² ≤ 3.2×10⁻³ both solutions agree to < 10⁻¹⁰ K here.
    """
    y = np.asarray(y, dtype=float)
    series = np.zeros_like(y)
    for k in range(n_terms):
        n = 2 * k + 1
        exp_coeff = np.exp(-(n * np.pi / 2.0) ** 2 * ALPHA * t / THICKNESS ** 2)
        if exp_coeff < 1e-15:
            break
        series += (1.0 / n) * exp_coeff * np.sin(n * np.pi * y / (2.0 * THICKNESS))
    return T_INIT + (T_WALL - T_INIT) * (1.0 - (4.0 / np.pi) * series)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_case():
    """Run the semi-infinite slab and return snapshots nearest each T_SNAP."""
    mat = _make_mat()
    geom = GeometryConfig(geometry_type=GeometryType.SLAB)
    stack = StackConfig(layers=[LayerConfig("Inert", THICKNESS, N_NODES, 4)])
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
        output_dt=min(T_SNAPS) / 2.0,   # save at least twice per shortest interval
    )

    print(f"\nMaterial: ρ={RHO} kg/m³  cp={CP} J/kg/K  k={K} W/m/K  α={ALPHA:.4e} m²/s")
    print(f"Slab: L={THICKNESS} m, N={N_NODES} nodes (dx={THICKNESS/(N_NODES-1)*1000:.2f} mm)")
    print(f"BCs: T_wall={T_WALL} K (front, prescribed)  T_init={T_INIT} K (adiabatic back)")

    results = run(
        stack, {"Inert": mat}, {"Inert": None},
        geom, surface_bc, back_bc, options,
        initial_T=T_INIT, verbose=True,
    )

    # Select snapshot nearest each requested time
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
    # Per-snapshot error thresholds — loosen at early times when only a few nodes
    # span the thermal front (backward-Euler spatial error ∝ 1/nodes_per_front²).
    # With N=1001 (dx=0.5 mm) all times are well-resolved (≥36 nodes/front).
    thresholds = {10.0: 3.0, 30.0: 2.0, 60.0: 1.5, 100.0: 1.0}

    dx = THICKNESS / (N_NODES - 1)
    print("\n" + "=" * 100)
    print(f"{'t [s]':>8}  {'nodes/front':>11}  {'back ΔT [K]':>11}  "
          f"{'vs erfc [%]':>11}  {'vs Fourier [%]':>14}  {'erfc−Fourier [K]':>17}  status")
    print("=" * 100)
    all_pass = True
    for t_req in T_SNAPS:
        snap = snaps[t_req]
        t_act = snap.time
        y, T_num = snap.mesh.y_nodes, snap.T

        nodes_per_front = 2.0 * np.sqrt(ALPHA * t_act) / dx
        back_dT = abs(T_num[-1] - T_INIT)

        T_erfc    = _erfc_solution(y, t_act)
        T_fourier = _fourier_solution(y, t_act)
        dT_ref    = T_fourier - T_INIT   # use Fourier as primary reference
        mask = dT_ref > 50.0

        def _rel_err_pct(T_ref):
            if mask.sum() == 0:
                return float("nan")
            return float(np.abs((T_num[mask] - T_ref[mask]) / dT_ref[mask]).max()) * 100.0

        err_erfc    = _rel_err_pct(T_erfc)
        err_fourier = _rel_err_pct(T_fourier)
        # Max absolute difference between the two analytical solutions (heated zone)
        ana_diff = float(np.abs(T_erfc[mask] - T_fourier[mask]).max()) if mask.sum() > 0 else float("nan")

        thr = thresholds[t_req]
        ok = back_dT < 0.01 and err_fourier < thr
        flag = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {t_req:6.1f}  {nodes_per_front:11.1f}  {back_dT:11.4f}  "
              f"{err_erfc:11.3f}  {err_fourier:14.3f}  {ana_diff:17.2e}  {flag}")

    print("=" * 100)
    print(f"Overall: {'PASS' if all_pass else 'FAIL'}")
    print(f"\nNotes:")
    print(f"  back face ΔT < 0.01 K  (L={THICKNESS} m is semi-infinite: ξ≥8.8 → erfc→0)")
    print(f"  erfc−Fourier ≈ 0 confirms the finite-slab solution agrees with the")
    print(f"  semi-infinite limit throughout (Fourier number Fo = αt/L² ≤ {ALPHA*max(T_SNAPS)/THICKNESS**2:.2e})")
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

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"V1c — Semi-infinite slab, prescribed wall temperature\n"
        f"ρ={RHO} kg/m³  cp={CP} J/kg/K  k={K} W/m/K  "
        f"T_wall={T_WALL} K  T_init={T_INIT} K  (N={N_NODES}, dx={THICKNESS/(N_NODES-1)*1000:.1f} mm)",
        fontsize=10,
    )

    # Panel 1 — temperature profiles: SCAM vs both analytical solutions
    ax = axes[0]
    for col, t_req in zip(colors, T_SNAPS):
        snap = snaps[t_req]
        y, T_num = snap.mesh.y_nodes, snap.T
        T_fourier = _fourier_solution(y, snap.time)
        T_erfc    = _erfc_solution(y, snap.time)
        delta = 2.0 * np.sqrt(ALPHA * snap.time)
        depth_max = min(3.0 * delta * 1000, THICKNESS * 1000)
        mp = y * 1000 <= depth_max
        lbl = f"t={snap.time:.0f} s"
        ax.plot(y[mp] * 1000, T_num[mp],     color=col, lw=2.0, label=f"SCAM {lbl}")
        ax.plot(y[mp] * 1000, T_fourier[mp], color=col, lw=1.2, ls="--",
                label=f"Fourier {lbl}")
        ax.plot(y[mp] * 1000, T_erfc[mp],    color=col, lw=0.8, ls=":",
                label=f"erfc {lbl}")
    ax.set_xlabel("Depth [mm]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("Profiles  (solid=SCAM, dashed=Fourier, dot=erfc)")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 2 — SCAM relative error vs Fourier (exact for actual BC).
    # Two-sided mask: exclude the surface node (pinned exactly) AND the steep
    # thermal front where backward-Euler curvature error peaks and dT_ref is
    # small (~50 K), which would make relative error diverge spuriously.
    # Interior band: 100 K < dT_ref < 0.95*(T_wall - T_init)
    dT_span = T_WALL - T_INIT
    ax = axes[1]
    for col, t_req in zip(colors, T_SNAPS):
        snap = snaps[t_req]
        y, T_num = snap.mesh.y_nodes, snap.T
        T_fourier = _fourier_solution(y, snap.time)
        dT_ref = T_fourier - T_INIT
        mask = (dT_ref > 0.15 * dT_span) & (dT_ref < 0.95 * dT_span)
        if mask.sum() > 0:
            rel_err = np.abs((T_num[mask] - T_fourier[mask]) / dT_ref[mask]) * 100.0
            ax.plot(y[mask] * 1000, rel_err, color=col, lw=1.5,
                    label=f"t={snap.time:.0f} s")
    ax.axhline(1.5, color="k", ls="--", lw=1.0, label="1.5 % guide")
    ax.set_xlabel("Depth [mm]")
    ax.set_ylabel("Relative error [%]")
    ax.set_title("SCAM vs Fourier  (interior: 15%–95% of ΔT span)")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # Panel 3 — difference between erfc and Fourier (validates semi-infinite thickness)
    ax = axes[2]
    for col, t_req in zip(colors, T_SNAPS):
        snap = snaps[t_req]
        y = snap.mesh.y_nodes
        T_fourier = _fourier_solution(y, snap.time)
        T_erfc    = _erfc_solution(y, snap.time)
        dT_ref = T_fourier - T_INIT
        mask = dT_ref > 1.0
        if mask.sum() > 0:
            diff_K = np.abs(T_erfc[mask] - T_fourier[mask])
            ax.semilogy(y[mask] * 1000, np.maximum(diff_K, 1e-15), color=col, lw=1.5,
                        label=f"t={snap.time:.0f} s")
    ax.set_xlabel("Depth [mm]")
    ax.set_ylabel("|erfc − Fourier|  [K]")
    ax.set_title("erfc − Fourier  (≈ 0 → L is semi-infinite)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("V1c — Semi-infinite slab, prescribed wall temperature")
    print(f"Analytical: T(y,t) = {T_INIT} + {T_WALL - T_INIT} · erfc(y / (2√(α t)))")

    snaps = run_case()
    all_pass = print_report(snaps)

    out_png = Path(__file__).parent / "v1c_conduction_semiinfinite.png"
    make_plot(snaps, out_png)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
