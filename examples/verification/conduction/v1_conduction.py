# SPDX-License-Identifier: MIT
"""V1 conduction verification — standalone runnable example.

Demonstrates and verifies the three sub-cases from the V1 rung of the SCAM
incremental verification ladder (tests/verification/test_v1_conduction_constant.py):

    Sub-case 1 — semi-infinite slab, prescribed surface FLUX
                 reference: erfc solution  T(y,t) = T0 + (2Q0/k)√(αt/π)e^{−ξ²}
                            − (Q0 y / k) erfc(ξ),  ξ = y/(2√(αt))

    Sub-case 2 — semi-infinite slab, prescribed surface TEMPERATURE
                 reference: erfc step-temperature  T(y,t) = T0 + (Tw−T0) erfc(ξ)

    Sub-case 3 — finite slab, both ends fixed, run to STEADY STATE
                 reference: linear profile  T(y) = T_front + (T_back−T_front) y/L
                 (first-order convergent under grid refinement — no bug,
                  the half-cell boundary nodes have a known O(h) offset)

Run from the repo root:

    python3 examples/v1_conduction_verification.py

Or with a non-interactive backend to suppress the display window:

    MPLBACKEND=Agg python3 examples/v1_conduction_verification.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import erfc

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
# Shared material / physical constants
# ---------------------------------------------------------------------------
K     = 1.0    # thermal conductivity [W/m/K]
RHO   = 180.0  # density [kg/m^3]
CP    = 710.0  # specific heat [J/kg/K]
ALPHA = K / (RHO * CP)   # thermal diffusivity [m^2/s]
T0    = 300.0  # initial uniform temperature [K]


# ---------------------------------------------------------------------------
# Material builder
# ---------------------------------------------------------------------------

def _inert_mat(name: str = "Inert") -> MaterialCard:
    """Constant-property inert slab (no decomposition)."""
    T_pts = np.array([200.0, 4000.0])
    k_tab  = np.column_stack([T_pts, np.full(2, K)])
    cp_tab = np.column_stack([T_pts, np.full(2, CP)])
    hg_tab = np.column_stack([T_pts, np.zeros(2)])
    return MaterialCard(
        name=name, rho_virgin=RHO, rho_char=RHO, gamma_resin=0.0,
        components=[], k_virgin_table=k_tab, k_char_table=k_tab,
        cp_virgin_table=cp_tab, cp_char_table=cp_tab, h_g_table=hg_tab,
        decomposing=False,
    )


# ---------------------------------------------------------------------------
# Analytical references
# ---------------------------------------------------------------------------

def _erfc_flux(y, t, Q0):
    """Semi-infinite erfc solution for step surface flux Q0 (Carslaw & Jaeger)."""
    xi = y / (2.0 * np.sqrt(ALPHA * t))
    return T0 + (2.0 * Q0 / K * np.sqrt(ALPHA * t / np.pi) * np.exp(-xi**2)
                 - Q0 * y / K * erfc(xi))


def _erfc_temp(y, t, T_wall):
    """Semi-infinite erfc solution for step surface temperature T_wall."""
    xi = y / (2.0 * np.sqrt(ALPHA * t))
    return T0 + (T_wall - T0) * erfc(xi)


def _linear_profile(y, L, T_front, T_back):
    """Steady-state linear temperature profile across a slab of thickness L."""
    return T_front + (T_back - T_front) * y / L


# ---------------------------------------------------------------------------
# Sub-case runners
# ---------------------------------------------------------------------------

def _run(stack, surface_bc, back_bc, options, initial_T=T0):
    mat = _inert_mat()
    geom = GeometryConfig(geometry_type=GeometryType.SLAB)
    return run(stack, {"Inert": mat}, {"Inert": None},
               geom, surface_bc, back_bc, options,
               initial_T=initial_T, verbose=True)


def run_subcase1_flux():
    """Sub-case 1: prescribed surface flux vs erfc."""
    print("\n" + "=" * 60)
    print("Sub-case 1 — semi-infinite slab, prescribed surface flux")
    print("=" * 60)
    Q0, t_end, thickness = 50_000.0, 20.0, 0.10
    stack = StackConfig(layers=[LayerConfig("Inert", thickness, 101, 4)])
    sbc = SurfaceBCConfig(
        bc_type=SurfaceBCType.PRESCRIBED_FLUX, q_prescribed=lambda t: Q0,
    )
    bbc = BackBCConfig(bc_type=BackBCType.ADIABATIC)
    opts = SolverOptions(
        t_end=t_end, dt_init=0.05, dt_max=1.0, dt_min=1e-4,
        dt_max_dT=200.0, output_dt=t_end,
    )
    results = _run(stack, sbc, bbc, opts)
    snap = results.snapshots[-1]
    y, T_num = snap.mesh.y_nodes, snap.T
    T_ana = _erfc_flux(y, t_end, Q0)
    dT = T_ana - T0
    mask = dT > 50.0
    rel_err = np.abs((T_num[mask] - T_ana[mask]) / dT[mask])
    max_pct = float(rel_err.max()) * 100.0
    print(f"\nMax relative error (dT > 50 K region): {max_pct:.2f}%")
    print("PASS" if max_pct < 4.0 else f"FAIL  (threshold 4%)")
    return y, T_num, T_ana, mask, rel_err, t_end, f"Prescribed flux  (Q₀ = {Q0/1e4:.1f} W/cm²)"


def run_subcase2_temp():
    """Sub-case 2: prescribed surface temperature vs erfc."""
    print("\n" + "=" * 60)
    print("Sub-case 2 — semi-infinite slab, prescribed surface temperature")
    print("=" * 60)
    T_wall, t_end, thickness = 2000.0, 20.0, 0.10
    stack = StackConfig(layers=[LayerConfig("Inert", thickness, 101, 4)])
    sbc = SurfaceBCConfig(
        bc_type=SurfaceBCType.PRESCRIBED_TEMP, T_prescribed=lambda t: T_wall,
    )
    bbc = BackBCConfig(bc_type=BackBCType.ADIABATIC)
    opts = SolverOptions(
        t_end=t_end, dt_init=0.02, dt_max=0.1, dt_min=1e-4,
        dt_max_dT=1e4, output_dt=t_end,
    )
    results = _run(stack, sbc, bbc, opts)
    snap = results.snapshots[-1]
    y, T_num = snap.mesh.y_nodes, snap.T
    T_ana = _erfc_temp(y, t_end, T_wall)
    pin_err = abs(T_num[0] - T_wall)
    print(f"Surface pinning error: {pin_err:.4f} K  (threshold 0.01 K)")
    dT = T_ana - T0
    mask = dT > 50.0
    rel_err = np.abs((T_num[mask] - T_ana[mask]) / dT[mask])
    max_pct = float(rel_err.max()) * 100.0
    print(f"Max relative error (dT > 50 K region): {max_pct:.2f}%")
    print("PASS" if (pin_err < 0.01 and max_pct < 5.0) else "FAIL")
    return y, T_num, T_ana, mask, rel_err, t_end, "Prescribed temp  (T_w = 1500 K)"


def run_subcase3_steady():
    """Sub-case 3: finite slab steady state, first-order convergence."""
    print("\n" + "=" * 60)
    print("Sub-case 3 — finite slab, fixed ends, steady state")
    print("=" * 60)
    T_front, T_back, L = 2000.0, 400.0, 0.05
    t_diff = L**2 / ALPHA

    def _run_n(n):
        stack = StackConfig(layers=[LayerConfig("Inert", L, n, 4)])
        sbc = SurfaceBCConfig(
            bc_type=SurfaceBCType.PRESCRIBED_TEMP, T_prescribed=lambda t: T_front,
        )
        bbc = BackBCConfig(bc_type=BackBCType.PRESCRIBED_TEMP, T_back=T_back)
        opts = SolverOptions(
            t_end=40.0 * t_diff, dt_init=0.1, dt_max=t_diff / 20.0, dt_min=1e-3,
            dt_max_dT=1e4, output_dt=40.0 * t_diff,
        )
        return _run(stack, sbc, bbc, opts, initial_T=T_back).snapshots[-1]

    snap_c = _run_n(51)
    snap_f = _run_n(101)

    ref_c = _linear_profile(snap_c.mesh.y_nodes, L, T_front, T_back)
    ref_f = _linear_profile(snap_f.mesh.y_nodes, L, T_front, T_back)
    err_c = float(np.abs(snap_c.T - ref_c).max())
    err_f = float(np.abs(snap_f.T - ref_f).max())
    ratio = err_c / err_f

    print(f"\nCoarse (n=51)  max|T − T_linear| = {err_c:.3f} K")
    print(f"Fine   (n=101) max|T − T_linear| = {err_f:.3f} K")
    print(f"Error ratio (coarse/fine): {ratio:.2f}  (expected ~2.0 for 1st-order)")
    note = ("This is expected first-order-accurate half-cell boundary "
            "discretization — not a bug.")
    print(f"  Note: {note}")
    # Threshold is 0.25% of the temperature span — absolute K threshold would
    # break if T_front is changed, because the half-cell error scales with span.
    span = T_front - T_back
    tol_K = 0.0025 * span
    ok = err_f < tol_K and 1.7 < ratio < 2.3
    print(f"Fine-grid threshold: {tol_K:.2f} K  (0.25% of {span:.0f} K span)")
    print("PASS" if ok else "FAIL")
    return snap_f.mesh.y_nodes, snap_f.T, ref_f, L, T_front, T_back


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    y1, T1n, T1a, mask1, re1, tend1, lbl1 = run_subcase1_flux()
    y2, T2n, T2a, mask2, re2, tend2, lbl2 = run_subcase2_temp()
    y3, T3n, T3r, L3, Tf3, Tb3          = run_subcase3_steady()

    try:
        import matplotlib
        matplotlib.use("Agg")          # non-interactive; works without a display
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        fig.suptitle("V1 — Conduction verification (constant properties)", fontsize=13)

        # ---- Sub-case 1 ----
        ax = axes[0, 0]
        ax.plot(y1 * 1e3, T1n, "b-",  lw=1.5, label="SCAM numerical")
        ax.plot(y1 * 1e3, T1a, "r--", lw=1.5, label="Analytical (erfc flux)")
        ax.set_xlabel("Depth [mm]")
        ax.set_ylabel("Temperature [K]")
        ax.set_title(lbl1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.semilogy(y1[mask1] * 1e3, re1 * 100, "b-", lw=1.5)
        ax.axhline(4.0, color="r", ls="--", lw=1, label="4% threshold")
        ax.set_xlabel("Depth [mm]")
        ax.set_ylabel("Relative error [%]")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ---- Sub-case 2 ----
        ax = axes[0, 1]
        ax.plot(y2 * 1e3, T2n, "b-",  lw=1.5, label="SCAM numerical")
        ax.plot(y2 * 1e3, T2a, "r--", lw=1.5, label="Analytical (erfc temp)")
        ax.set_xlabel("Depth [mm]")
        ax.set_ylabel("Temperature [K]")
        ax.set_title(lbl2)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.semilogy(y2[mask2] * 1e3, re2 * 100, "b-", lw=1.5)
        ax.axhline(5.0, color="r", ls="--", lw=1, label="5% threshold")
        ax.set_xlabel("Depth [mm]")
        ax.set_ylabel("Relative error [%]")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ---- Sub-case 3 ----
        ax = axes[0, 2]
        ax.plot(y3 * 1e3, T3n, "b-",  lw=1.5, label="SCAM numerical")
        ax.plot(y3 * 1e3, T3r, "r--", lw=1.5, label="Exact linear (steady)")
        ax.set_xlabel("Depth [mm]")
        ax.set_ylabel("Temperature [K]")
        ax.set_title(f"Fixed ends, steady state  ({Tf3:.0f}→{Tb3:.0f} K)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 2]
        ax.plot(y3 * 1e3, np.abs(T3n - T3r), "b-", lw=1.5)
        ax.axhline(3.0, color="r", ls="--", lw=1, label="3 K bound (fine grid)")
        ax.set_xlabel("Depth [mm]")
        ax.set_ylabel("|T_num − T_linear|  [K]")
        ax.set_title("1st-order boundary artifact\n(halves on grid doubling)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out = Path(__file__).parent / "v1_conduction_verification.png"
        plt.savefig(out, dpi=150)
        print(f"\nPlot saved → {out}")
        plt.show()   # no-op with Agg backend

    except ImportError:
        print("\n(matplotlib not available; skipping plots)")
    except Exception as exc:
        print(f"\n(plot failed: {exc})")


if __name__ == "__main__":
    main()
