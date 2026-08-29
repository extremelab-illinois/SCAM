# SPDX-License-Identifier: MIT
"""V3 — Multi-material stack verification.

Verifies that SCAM correctly couples two layers at a shared interface and
handles an optional thermal contact resistance.  All sub-cases are run to
steady state so the reference is exact (no transient error).

Reference: one-dimensional series thermal resistance (Fourier's law, 1-D,
steady, constant properties in each layer):

    q = (T_front − T_back) / (L1/k1 + R + L2/k2)      [W/m²]
    dT/dy|layer = −q / k_layer                          [K/m]
    ΔT_contact  = q · R                                 [K]

The two layers have deliberately different conductivities (k1 = 2.0 W/m/K,
k2 = 0.5 W/m/K — a 4× ratio) so the per-layer slopes are visibly different and
interface effects are prominent.

Three sub-cases:
    1 — No contact resistance.  Verify q and per-layer slopes.
    2 — Contact resistance R = 1e-3 m²K/W.  Verify that the incremental
        interface temperature jump Δ(R>0) − Δ(R=0) equals q·R.
    3 — Grid convergence.  Error in q should halve on grid halving (first-order
        half-cell boundary treatment).

Note on the contact-resistance check (incremental, not absolute):
    The half-cell boundary treatment places a first-order artifact on the two
    nodes that straddle the interface even when R = 0.  Comparing the R>0
    jump directly against q·R would mix this artifact into the check.
    Instead, the *extra* jump attributable to the contact resistance
    (jump_R − jump_0) is compared to q·R; the artifact cancels exactly.

Run from the repo root:

    python3 examples/v3_multilayer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from scam.config.boundary import (
    BackBCConfig, BackBCType, SurfaceBCConfig, SurfaceBCType,
)
from scam.config.geometry import GeometryConfig, GeometryType
from scam.config.material import MaterialCard
from scam.config.solver import SolverOptions
from scam.config.stack import LayerConfig, StackConfig
from scam.solvers.material_response import run


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
K1, K2      = 2.0, 0.5        # W/m/K — front layer conductive, back insulating
RHO, CP     = 180.0, 710.0    # kg/m³, J/kg/K (same in both layers)
L1, L2      = 0.025, 0.025    # m  (equal thicknesses — total 50 mm)
T_FRONT     = 1500.0          # K  hot face
T_BACK      = 400.0           # K  cold face
R_CONTACT   = 1.0e-3          # m²K/W  for sub-case 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mat(name: str, k: float) -> MaterialCard:
    T_pts  = np.array([200.0, 4000.0])
    k_tab  = np.column_stack([T_pts, np.full(2, k)])
    cp_tab = np.column_stack([T_pts, np.full(2, CP)])
    hg_tab = np.column_stack([T_pts, np.zeros(2)])
    return MaterialCard(
        name=name, rho_virgin=RHO, rho_char=RHO, gamma_resin=0.0,
        components=[], k_virgin_table=k_tab, k_char_table=k_tab,
        cp_virgin_table=cp_tab, cp_char_table=cp_tab, h_g_table=hg_tab,
        decomposing=False,
    )


def _run(n1: int, n2: int, R: float = 0.0, verbose: bool = False):
    """Run two-layer slab to steady state; return final snapshot."""
    m1 = _mat("M1", K1)
    m2 = _mat("M2", K2)
    stack = StackConfig(layers=[
        LayerConfig("M1", L1, n1, 4),
        LayerConfig("M2", L2, n2, 4, contact_resistance=R),
    ])
    sbc = SurfaceBCConfig(
        bc_type=SurfaceBCType.PRESCRIBED_TEMP, T_prescribed=lambda t: T_FRONT,
    )
    bbc = BackBCConfig(bc_type=BackBCType.PRESCRIBED_TEMP, T_back=T_BACK)
    geom = GeometryConfig()
    # Run for 40 diffusion times of the slowest layer
    alpha_min = min(K1, K2) / (RHO * CP)
    t_diff    = (L1 + L2)**2 / alpha_min
    opts = SolverOptions(
        t_end=40.0 * t_diff, dt_init=0.1, dt_max=t_diff / 20.0, dt_min=1e-3,
        dt_max_dT=1e4, output_dt=40.0 * t_diff,
    )
    results = run(
        stack, {"M1": m1, "M2": m2}, {"M1": None, "M2": None},
        geom, sbc, bbc, opts, initial_T=T_BACK, verbose=verbose,
    )
    return results.snapshots[-1]


def _series_flux(R: float = 0.0) -> float:
    return (T_FRONT - T_BACK) / (L1 / K1 + R + L2 / K2)


def _interface_jump(snap) -> float:
    """Temperature drop across the interface: T(last M1 node) − T(first M2 node)."""
    lid = snap.mesh.layer_id
    i0 = int(np.where(lid == 0)[0][-1])
    i1 = int(np.where(lid == 1)[0][0])
    return float(snap.T[i0] - snap.T[i1])


def _layer_slope(snap, layer_idx: int) -> float:
    """Least-squares slope dT/dy in the interior of the specified layer."""
    lid = snap.mesh.layer_id
    idx = np.where(lid == layer_idx)[0][2:-2]   # skip 2 boundary nodes each end
    return float(np.polyfit(snap.mesh.y_nodes[idx], snap.T[idx], 1)[0])


# ---------------------------------------------------------------------------
# Sub-cases
# ---------------------------------------------------------------------------

def run_subcase1_flux_and_slopes():
    print("\n" + "=" * 60)
    print("Sub-case 1 — two layers, no contact resistance")
    print("=" * 60)
    snap    = _run(51, 51, R=0.0, verbose=True)
    q_ref   = _series_flux(0.0)
    q_num   = snap.q_cond
    rel_q   = abs(q_num - q_ref) / q_ref * 100

    print(f"\nSeries flux  q_ref  = {q_ref:.2f} W/m²")
    print(f"Solver       q_cond = {q_num:.2f} W/m²")
    print(f"Relative error      = {rel_q:.3f}%  (threshold 1%)")
    print("  " + ("PASS" if rel_q < 1.0 else "FAIL"))

    for li, (lname, k) in enumerate([("M1", K1), ("M2", K2)]):
        slope_num = _layer_slope(snap, li)
        slope_ref = -q_ref / k
        rel_s     = abs(slope_num - slope_ref) / abs(slope_ref) * 100
        print(f"\nLayer {lname} (k={k}):")
        print(f"  reference slope  = {slope_ref:.2f} K/m")
        print(f"  numerical slope  = {slope_num:.2f} K/m")
        print(f"  relative error   = {rel_s:.3f}%  (threshold 2%)")
        print("  " + ("PASS" if rel_s < 2.0 else "FAIL"))

    return snap, q_ref


def run_subcase2_contact_resistance():
    print("\n" + "=" * 60)
    print(f"Sub-case 2 — contact resistance R = {R_CONTACT:.0e} m²K/W")
    print("=" * 60)
    snap0  = _run(51, 51, R=0.0)
    snapR  = _run(51, 51, R=R_CONTACT)

    jump0  = _interface_jump(snap0)
    jumpR  = _interface_jump(snapR)
    delta  = jumpR - jump0          # incremental jump due to R alone
    expect = snapR.q_cond * R_CONTACT
    rel    = abs(delta - expect) / expect * 100

    print(f"\nR=0  interface jump  (artifact only) = {jump0:.4f} K")
    print(f"R>0  interface jump  (artifact + R)  = {jumpR:.4f} K")
    print(f"Incremental ΔT (jump_R − jump_0)    = {delta:.4f} K")
    print(f"Expected q·R = {snapR.q_cond:.2f} × {R_CONTACT:.0e} = {expect:.4f} K")
    print(f"Relative error = {rel:.3f}%  (threshold 5%)")
    print("  " + ("PASS" if rel < 5.0 else "FAIL"))

    return snap0, snapR, jump0, jumpR, delta, expect


def run_subcase3_grid_convergence():
    print("\n" + "=" * 60)
    print("Sub-case 3 — grid convergence in surface flux")
    print("=" * 60)
    snap_c = _run(26, 26)   # ~50 nodes total
    snap_f = _run(51, 51)   # ~100 nodes total
    q_ref  = _series_flux(0.0)
    err_c  = abs(snap_c.q_cond - q_ref) / q_ref * 100
    err_f  = abs(snap_f.q_cond - q_ref) / q_ref * 100
    ratio  = err_c / err_f if err_f > 0 else float("inf")

    print(f"\nCoarse (n=26+26) q_cond = {snap_c.q_cond:.2f} W/m²  err = {err_c:.4f}%")
    print(f"Fine   (n=51+51) q_cond = {snap_f.q_cond:.2f} W/m²  err = {err_f:.4f}%")
    print(f"Error ratio (coarse/fine) = {ratio:.2f}  (expected ~2.0)")
    ok = 1.5 < ratio < 2.5
    print("  " + ("PASS" if ok else "FAIL"))

    return snap_c, snap_f, err_c, err_f, ratio


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    snap1, q_ref      = run_subcase1_flux_and_slopes()
    snap0, snapR, jump0, jumpR, delta, expect = run_subcase2_contact_resistance()
    snap_c, snap_f, err_c, err_f, ratio       = run_subcase3_grid_convergence()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        fig.suptitle(
            f"V3 — Two-layer stack verification\n"
            f"M1: k={K1} W/m/K  L={L1*1e3:.0f} mm   |   "
            f"M2: k={K2} W/m/K  L={L2*1e3:.0f} mm,   "
            f"T_front={T_FRONT:.0f} K,  T_back={T_BACK:.0f} K",
            fontsize=11,
        )

        y_int = L1 * 1e3   # interface position in mm

        # ── [0,0]  Temperature profile, no contact resistance ────────────────
        ax = axes[0, 0]
        y1  = snap1.mesh.y_nodes * 1e3
        T1n = snap1.T
        # piecewise reference
        q_r = _series_flux(0.0)
        y_ref = np.array([0.0, L1, L1, L1 + L2]) * 1e3
        T_ref = np.array([
            T_FRONT,
            T_FRONT - q_r * L1 / K1,
            T_FRONT - q_r * L1 / K1,   # no contact jump
            T_BACK,
        ])
        ax.plot(y1,    T1n,  "b-",  lw=2,   label="SCAM numerical")
        ax.plot(y_ref, T_ref, "r--", lw=1.5, label="Series reference")
        ax.axvline(y_int, color="gray", ls=":", lw=1.0, label="Interface")
        ax.set_xlabel("Depth [mm]")
        ax.set_ylabel("Temperature [K]")
        ax.set_title("Steady profile  (R = 0)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── [0,1]  Profile with and without contact resistance ───────────────
        ax = axes[0, 1]
        y0  = snap0.mesh.y_nodes * 1e3
        yR  = snapR.mesh.y_nodes * 1e3
        ax.plot(y0, snap0.T, "b-",  lw=2,   label=f"R = 0")
        ax.plot(yR, snapR.T, "r-",  lw=2,   label=f"R = {R_CONTACT:.0e} m²K/W")
        ax.axvline(y_int, color="gray", ls=":", lw=1.0)
        ax.annotate("", xy=(y_int + 0.3, jumpR + T_FRONT - q_r*L1/K1 - jumpR/2),
                    xytext=(y_int + 0.3, T_FRONT - q_r*L1/K1 - jumpR/2),
                    arrowprops=dict(arrowstyle="<->", color="green", lw=1.5))
        ax.set_xlabel("Depth [mm]")
        ax.set_ylabel("Temperature [K]")
        ax.set_title(
            f"Contact resistance effect\n"
            f"ΔT_contact = {jumpR - jump0:.2f} K  (q·R = {expect:.2f} K)"
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── [0,2]  Grid convergence ──────────────────────────────────────────
        ax = axes[0, 2]
        grids  = ["coarse\n(26+26)", "fine\n(51+51)"]
        errors = [err_c, err_f]
        bars = ax.bar(grids, errors, color=["#aec7e8", "#1f77b4"], width=0.4)
        ax.bar_label(bars, labels=[f"{e:.4f}%" for e in errors], padding=3, fontsize=9)
        ax.set_ylabel("Flux relative error  [%]")
        ax.set_title(
            f"Grid convergence — q error\n"
            f"ratio = {ratio:.2f}  (target ~2.0, 1st-order)"
        )
        ax.grid(True, alpha=0.3, axis="y")

        # ── [1,0]  Per-layer temperature error from reference ────────────────
        ax = axes[1, 0]
        y    = snap1.mesh.y_nodes * 1e3
        lid  = snap1.mesh.layer_id
        T_ref_arr = np.where(
            lid == 0,
            T_FRONT - q_r / K1 * snap1.mesh.y_nodes,
            T_FRONT - q_r * L1 / K1 - q_r / K2 * (snap1.mesh.y_nodes - L1),
        )
        ax.plot(y, np.abs(snap1.T - T_ref_arr), "b-", lw=1.5)
        ax.axvline(y_int, color="gray", ls=":", lw=1.0, label="Interface")
        ax.set_xlabel("Depth [mm]")
        ax.set_ylabel("|T_num − T_ref|  [K]")
        ax.set_title("Pointwise error vs series reference  (R = 0)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── [1,1]  Slope check bar chart ─────────────────────────────────────
        ax = axes[1, 1]
        slopes_num = [_layer_slope(snap1, 0), _layer_slope(snap1, 1)]
        slopes_ref = [-q_r / K1, -q_r / K2]
        x = np.arange(2)
        w = 0.35
        b1 = ax.bar(x - w/2, slopes_num, w, color="#1f77b4", label="SCAM")
        b2 = ax.bar(x + w/2, slopes_ref, w, color="#d62728", label="−q/k (reference)", alpha=0.7)
        ax.bar_label(b1, labels=[f"{v:.1f}" for v in slopes_num], padding=3, fontsize=8)
        ax.bar_label(b2, labels=[f"{v:.1f}" for v in slopes_ref], padding=3, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"M1 (k={K1})", f"M2 (k={K2})"])
        ax.set_ylabel("dT/dy  [K/m]")
        ax.set_title("Per-layer interior slopes  (steeper = lower k)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

        # ── [1,2]  Interface jump breakdown ──────────────────────────────────
        ax = axes[1, 2]
        labels = ["R=0\n(artifact)", "R>0\ntotal jump", "Incremental\nΔ(R>0)−Δ(R=0)", "q·R\n(expected)"]
        values = [jump0, jumpR, delta, expect]
        colors_bar = ["#aec7e8", "#1f77b4", "#ff7f0e", "#d62728"]
        bars = ax.bar(labels, values, color=colors_bar, width=0.5)
        ax.bar_label(bars, labels=[f"{v:.3f} K" for v in values], padding=3, fontsize=8)
        ax.set_ylabel("Interface temperature jump  [K]")
        ax.set_title(
            f"Contact resistance check\n"
            f"ΔT_incremental = {delta:.3f} K  vs  q·R = {expect:.3f} K  "
            f"({abs(delta-expect)/expect*100:.2f}% err)"
        )
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        out = Path(__file__).parent / "v3_multilayer.png"
        plt.savefig(out, dpi=150)
        print(f"\nPlot saved → {out}")
        plt.show()

    except ImportError:
        print("\n(matplotlib not available — skipping plots)")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"\n(plot failed: {exc})")


if __name__ == "__main__":
    main()
