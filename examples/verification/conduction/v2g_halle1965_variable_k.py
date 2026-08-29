# SPDX-License-Identifier: MIT
"""V2g — Semi-infinite solid with variable thermal conductivity (Halle 1965).

Reference:
    H. Halle, "Exact solution of elementary transient heat conduction problem
    involving temperature dependent properties," Trans. ASME, Series C,
    J. Heat Transfer, vol. 87, no. 3, pp. 420–421, August 1965.

Analytical solution (similarity variable η = x/(2√(α₀t))):

    T(x,t) = T0 + (Tw − T0)·[A·erfc(η) + (1−A)·erfc(n·η)]

Thermal conductivity (parametric in η):

    k(η)/k₀ = [A·e^{−η²} + (1−A)/n·e^{−n²η²}]
               ───────────────────────────────────
               [A·e^{−η²} + (1−A)·n·e^{−n²η²}  ]

These form a closed-form k(T) relation (via T(η) ↔ η ↔ k(η)).

Verification passes:
    Case 1: A = 0.50, n = 2  →  k decreases from k₀ at T0 to 0.5 k₀ at Tw
    Case 2: A = 1.85, n = 2  →  k increases from k₀ at T0 to 9.5 k₀ at Tw

Material / geometry:
    ρ = 1850 kg/m³, cp = 2000 J/kg/K, k₀ = 30 W/m/K, α₀ ≈ 8.108e-6 m²/s
    T0 = 300 K, Tw = 1000 K
    Sslab = 100 mm  (semi-infinite up to t ~ 300 s)
    N = 1000 nodes, dt = 0.01 s

Run:
    MPLBACKEND=Agg python3 examples/verification/conduction/v2g_halle1965_variable_k.py
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
# Constants
# ---------------------------------------------------------------------------
RHO    = 1850.0    # [kg/m³]
CP     = 2000.0    # [J/kg/K]
K0     = 30.0      # [W/m/K]  reference (cold-side) conductivity
ALPHA0 = K0 / (RHO * CP)   # ≈ 8.108e-6 m²/s

T0     = 300.0     # [K] initial / far-field temperature
TW     = 1000.0    # [K] prescribed wall temperature

# Case 2 has k/k₀ = 9.5 at Tw → local α ≈ 9.5α₀ near the hot wall, so the
# thermal front penetrates much faster than the α₀-based estimate suggests.
# At t=100 s: 2√(9.5α₀·t) ≈ 175 mm >> 100 mm.  A 200 mm slab keeps the back
# face below 0.01 K change for both cases up to t=100 s.
SSLAB  = 0.20      # [m] numerical slab thickness (200 mm — semi-infinite for t ≤ 100 s)
N_NODES = 1000     # dx = 0.2 mm
DT_MAX  = 0.01     # [s]

T_SNAPS = [10.0, 30.0, 60.0, 100.0]   # [s]

CASES = [
    dict(A=0.50, n=2.0, label="Case 1  A=0.50, n=2"),
    dict(A=1.85, n=2.0, label="Case 2  A=1.85, n=2"),
]


# ---------------------------------------------------------------------------
# Analytical solution  (Halle 1965)
# ---------------------------------------------------------------------------

def theta_halle(eta, A, n):
    """Normalised temperature θ = (T−T0)/(Tw−T0) = A·erfc(η) + (1−A)·erfc(n·η)."""
    return A * erfc(eta) + (1 - A) * erfc(n * eta)


def k_ratio_halle(eta, A, n):
    """k(η)/k₀ from the Halle parametric conductivity formula."""
    e1 = np.exp(-eta ** 2)
    e2 = np.exp(-(n * eta) ** 2)
    num = A * e1 + (1 - A) / n * e2
    den = A * e1 + (1 - A) * n * e2
    return num / den


def T_analytical(x, t, A, n):
    """Temperature at position x [m] and time t [s]."""
    eta = np.asarray(x) / (2.0 * np.sqrt(ALPHA0 * t))
    return T0 + (TW - T0) * theta_halle(eta, A, n)


# ---------------------------------------------------------------------------
# k(T) table builder
# ---------------------------------------------------------------------------

def make_k_table(A, n, n_pts: int = 2000):
    """Build a k(T) table by sampling the parametric Halle curve (T(η), k(η)).

    η is sampled densely near 0 (T ≈ Tw, large-k region for case 2) and
    coarser at large η (T ≈ T0).
    """
    # Dense near η=0 where k/k₀ changes most rapidly (case 2: 9.5→1)
    eta_fine   = np.linspace(0.0, 0.2,  n_pts // 2)
    eta_coarse = np.linspace(0.2, 5.0,  n_pts - n_pts // 2)
    eta = np.unique(np.concatenate([eta_fine, eta_coarse]))

    T_eta  = T0 + (TW - T0) * theta_halle(eta, A, n)
    k_eta  = K0 * k_ratio_halle(eta, A, n)

    # Sort ascending in T; η=0 gives T=Tw, η→∞ gives T=T0
    idx = np.argsort(T_eta)
    T_s, k_s = T_eta[idx], k_eta[idx]

    # Deduplicate
    _, uniq = np.unique(np.round(T_s, 6), return_index=True)
    T_s, k_s = T_s[uniq], k_s[uniq]

    # Extend table to safely cover temperatures outside [T0, Tw]
    T_full = np.concatenate([[1.0],    T_s, [TW + 2000.0]])
    k_full = np.concatenate([[k_s[0]], k_s, [k_s[-1]]])

    return np.column_stack([T_full, k_full])


def make_variable_k_material(A, n) -> MaterialCard:
    """Non-decomposing material card with Halle k(T) relation."""
    k_tab  = make_k_table(A, n)
    cp_tab = np.column_stack([[1.0, TW + 2000.0], [CP, CP]])
    hg_tab = np.column_stack([[1.0, TW + 2000.0], [0.0, 0.0]])
    return MaterialCard(
        name="Halle", rho_virgin=RHO, rho_char=RHO, gamma_resin=0.0,
        components=[], k_virgin_table=k_tab, k_char_table=k_tab,
        cp_virgin_table=cp_tab, cp_char_table=cp_tab, h_g_table=hg_tab,
        emissivity=0.0, decomposing=False,
    )


# ---------------------------------------------------------------------------
# SCAM run
# ---------------------------------------------------------------------------

def run_case(A, n):
    """Run SCAM for one Halle case; return {t_req: snapshot}."""
    mat  = make_variable_k_material(A, n)
    geom = GeometryConfig(geometry_type=GeometryType.SLAB)
    stack = StackConfig(layers=[LayerConfig("Halle", SSLAB, N_NODES, 4)])
    surface_bc = SurfaceBCConfig(
        bc_type=SurfaceBCType.PRESCRIBED_TEMP,
        T_prescribed=lambda t: TW,
    )
    back_bc = BackBCConfig(bc_type=BackBCType.ADIABATIC)
    options = SolverOptions(
        t_end=max(T_SNAPS),
        dt_init=0.02,
        dt_max=DT_MAX,
        dt_min=1e-4,
        dt_max_dT=1e4,
        output_dt=min(T_SNAPS) / 2.0,
        max_picard=10,
    )

    results = run(
        stack, {"Halle": mat}, {"Halle": None},
        geom, surface_bc, back_bc, options,
        initial_T=T0, verbose=True,
    )

    saved_times = np.array([s.time for s in results.snapshots])
    return {
        t_req: results.snapshots[int(np.argmin(np.abs(saved_times - t_req)))]
        for t_req in T_SNAPS
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(snaps, A, n, label):
    # dx=0.2 mm (half the 1000-node 100mm grid) → thresholds ~2× vs V1c
    thresholds = {10.0: 3.0, 30.0: 2.0, 60.0: 1.5, 100.0: 2.0}

    print(f"\n{label}")
    print("=" * 80)
    print(f"{'t [s]':>8}  {'back ΔT [K]':>11}  {'max rel err [%]':>15}  "
          f"{'threshold [%]':>13}  status")
    print("=" * 80)
    all_pass = True
    for t_req in T_SNAPS:
        snap   = snaps[t_req]
        y_num  = snap.mesh.y_nodes
        T_num  = snap.T
        T_ana  = T_analytical(y_num, snap.time, A, n)
        dT_ref = T_ana - T0

        # Interior band: 5%–95% of current span
        T_span = float(T_ana[0] - T0)
        mask = (dT_ref > 0.05 * T_span) & (dT_ref < 0.95 * T_span)
        if mask.sum() == 0:
            mask = dT_ref > 10.0

        back_dT = abs(T_num[-1] - T0)
        if mask.sum() > 0:
            rel_err = np.abs((T_num[mask] - T_ana[mask]) / dT_ref[mask])
            max_err_pct = float(rel_err.max()) * 100.0
        else:
            max_err_pct = float("nan")

        thr  = thresholds[t_req]
        ok   = back_dT < 1.0 and max_err_pct < thr
        flag = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {t_req:6.1f}  {back_dT:11.4f}  {max_err_pct:15.3f}  {thr:13.1f}  {flag}")

    print("=" * 80)
    print(f"Overall: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_plot(case_results, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("(matplotlib not available — skipping plot)")
        return

    n_cases  = len(CASES)
    n_snaps  = len(T_SNAPS)
    colors   = cm.plasma(np.linspace(0.15, 0.85, n_snaps))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "V2g — Semi-infinite solid, variable conductivity (Halle 1965)\n"
        f"ρ={RHO} kg/m³  cp={CP} J/kg/K  k₀={K0} W/m/K  "
        f"T0={T0:.0f} K  Tw={TW:.0f} K  (N={N_NODES}, dt={DT_MAX} s)",
        fontsize=10,
    )

    # ----------------------------------------------------------------
    # Panel (0,0): k/k₀ vs θ = (T−T0)/(Tw−T0)
    # ----------------------------------------------------------------
    ax = axes[0, 0]
    eta_plot = np.linspace(0.0, 3.5, 2000)
    for c in CASES:
        A, n, lbl = c["A"], c["n"], c["label"]
        th_plot = theta_halle(eta_plot, A, n)
        kr_plot = k_ratio_halle(eta_plot, A, n)
        ax.plot(th_plot, kr_plot, lw=2, label=lbl)
    ax.set_xlabel(r"$(T-T_0)/(T_w-T_0)$")
    ax.set_ylabel(r"$k/k_0$")
    ax.set_title(r"$k/k_0$ vs normalised temperature")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ----------------------------------------------------------------
    # Panels (0,1) and (1,1): temperature profiles per case
    # Panels (0,2) and (1,2): relative error per case
    # ----------------------------------------------------------------
    row_axes = [(axes[0, 1], axes[0, 2]), (axes[1, 1], axes[1, 2])]

    for i_case, c in enumerate(CASES):
        A, n, lbl = c["A"], c["n"], c["label"]
        snaps = case_results[i_case]
        ax_T, ax_e = row_axes[i_case]

        for col, t_req in zip(colors, T_SNAPS):
            snap   = snaps[t_req]
            y_num  = snap.mesh.y_nodes
            T_num  = snap.T
            T_ana  = T_analytical(y_num, snap.time, A, n)

            delta  = 2.0 * np.sqrt(ALPHA0 * snap.time)
            depth_mm = min(3.0 * delta * 1000, SSLAB * 1000)
            mp = y_num * 1000 <= depth_mm

            tlbl = f"t={snap.time:.0f} s"
            ax_T.plot(y_num[mp] * 1e3, T_num[mp], color=col, lw=2.0,
                      label=f"SCAM {tlbl}")
            ax_T.plot(y_num[mp] * 1e3, T_ana[mp], color=col, lw=1.2, ls="--",
                      label=f"Ana. {tlbl}")

            dT_ref = T_ana - T0
            T_span = float(T_ana[0] - T0)
            mask = (dT_ref > 0.05 * T_span) & (dT_ref < 0.95 * T_span) & mp
            if mask.sum() > 0:
                rel_err = np.abs((T_num[mask] - T_ana[mask]) / dT_ref[mask]) * 100.0
                ax_e.plot(y_num[mask] * 1e3, rel_err, color=col, lw=1.5,
                          label=f"t={snap.time:.0f} s")

        ax_T.set_xlabel("Depth [mm]")
        ax_T.set_ylabel("Temperature [K]")
        ax_T.set_title(f"Temperature profiles — {lbl}\n(solid=SCAM, dashed=Analytical)")
        ax_T.legend(fontsize=6, ncol=2)
        ax_T.grid(True, alpha=0.3)

        ax_e.axhline(2.0, color="k", ls="--", lw=1.0, label="2% guide")
        ax_e.set_xlabel("Depth [mm]")
        ax_e.set_ylabel("Relative error [%]")
        ax_e.set_title(f"Error vs analytical — {lbl}")
        ax_e.legend(fontsize=8)
        ax_e.set_ylim(bottom=0)
        ax_e.grid(True, alpha=0.3)

    # Remove unused bottom-left panel
    axes[1, 0].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("V2g — Semi-infinite solid, variable conductivity (Halle 1965)")
    print(f"α₀ = k₀/(ρ·cp) = {ALPHA0:.4e} m²/s")
    print(f"Penetration at t=100s: 2√(α₀t) ≈ {2*np.sqrt(ALPHA0*100)*1e3:.1f} mm  "
          f"(<< Sslab={SSLAB*1e3:.0f} mm)")

    all_pass_overall = True
    case_results = []
    for c in CASES:
        A, n, label = c["A"], c["n"], c["label"]
        k_wall = K0 * k_ratio_halle(0.0, A, n)
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"  k at T0 (η→∞): {K0:.1f} W/m/K  |  k at Tw (η=0): {k_wall:.2f} W/m/K")
        print(f"{'='*60}")
        snaps = run_case(A, n)
        case_results.append(snaps)
        ok = print_report(snaps, A, n, label)
        all_pass_overall = all_pass_overall and ok

    out_png = Path(__file__).parent / "v2g_halle1965_variable_k.png"
    make_plot(case_results, out_png)

    return 0 if all_pass_overall else 1


if __name__ == "__main__":
    sys.exit(main())
