# SPDX-License-Identifier: MIT
"""V2 conduction verification — temperature-dependent conductivity.

Verifies three aspects of the variable-k conduction path against the
Kirchhoff-transform analytical reference:

    Sub-case 1 — Steady profile
        k(T) = a + b T (5× variation across the slab), both ends fixed,
        run to steady state.  Reference: invert the Kirchhoff transform
        theta(T(y)) = theta_front * (1 - y/L) for a closed-form quadratic
        profile.  The numerical profile should lie on this curve everywhere.

    Sub-case 2 — Grid convergence
        Same case at n=51 and n=101 nodes.  Error must halve (first-order
        convergence) — the same half-cell boundary artifact as in V1.

    Sub-case 3 — Surface flux
        Reported q_cond must match the Kirchhoff flux q = theta_front / L
        within 2%.  This is the cleanest scalar check: q is constant through
        the slab at steady state and equals the integral of k over the whole
        temperature span divided by L.

Run from the repo root:

    python3 examples/v2_conduction_variable.py
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
from scam.config.solver import SolverOptions
from scam.config.stack import LayerConfig, StackConfig
from scam.solvers.material_response import run
from scam.config.material import MaterialCard


# ---------------------------------------------------------------------------
# Material / geometry constants
# ---------------------------------------------------------------------------
K0, K1   = 0.5, 2.5       # k(T) at T0_K and T1_K  [W/m/K]
T0_K, T1_K = 300.0, 3000.0
RHO, CP  = 180.0, 710.0   # [kg/m³], [J/kg/K]
T_FRONT  = 2500.0          # hot face temperature [K]
T_BACK   = 400.0           # cold face temperature [K]
L        = 0.05            # slab thickness [m]

# Derived linear-k coefficients  k(T) = _A + _B * T
_B = (K1 - K0) / (T1_K - T0_K)
_A = K0 - _B * T0_K


# ---------------------------------------------------------------------------
# Kirchhoff helpers
# ---------------------------------------------------------------------------

def theta(T):
    """Kirchhoff variable  theta(T) = ∫_{T_BACK}^{T} k dT'  for linear k."""
    return _A * (T - T_BACK) + 0.5 * _B * (T**2 - T_BACK**2)


def T_kirchhoff(y):
    """Reference profile: invert theta(T(y)) = theta_front * (1 - y/L)."""
    y = np.asarray(y)
    target = theta(T_FRONT) * (1.0 - y / L)
    out = np.empty_like(target)
    for i, tg in enumerate(target.flat):
        # 0.5 b T^2 + a T + c = 0
        c = -(_A * T_BACK + 0.5 * _B * T_BACK**2 + tg)
        disc = _A**2 - 4.0 * (0.5 * _B) * c
        out.flat[i] = (-_A + np.sqrt(max(disc, 0.0))) / _B
    return out


def kirchhoff_flux():
    """Steady surface flux q = theta(T_front) / L."""
    return theta(T_FRONT) / L


# ---------------------------------------------------------------------------
# Material builder
# ---------------------------------------------------------------------------

def _var_k_mat(name: str = "VarK") -> MaterialCard:
    T_pts  = np.array([T0_K, T1_K])
    k_tab  = np.column_stack([T_pts, [K0, K1]])
    cp_tab = np.column_stack([T_pts, [CP, CP]])
    hg_tab = np.column_stack([T_pts, [0., 0.]])
    return MaterialCard(
        name=name, rho_virgin=RHO, rho_char=RHO, gamma_resin=0.0,
        components=[], k_virgin_table=k_tab, k_char_table=k_tab,
        cp_virgin_table=cp_tab, cp_char_table=cp_tab, h_g_table=hg_tab,
        decomposing=False,
    )


def _run_steady(n_nodes: int):
    """Run to steady state with n_nodes and return the final snapshot."""
    # t_diff based on the slowest (lowest-k) end
    alpha_min = K0 / (RHO * CP)
    t_diff = L**2 / alpha_min
    mat = _var_k_mat()
    stack = StackConfig(layers=[LayerConfig("VarK", L, n_nodes, 4)])
    sbc = SurfaceBCConfig(
        bc_type=SurfaceBCType.PRESCRIBED_TEMP, T_prescribed=lambda t: T_FRONT,
    )
    bbc = BackBCConfig(bc_type=BackBCType.PRESCRIBED_TEMP, T_back=T_BACK)
    opts = SolverOptions(
        t_end=40.0 * t_diff, dt_init=0.1, dt_max=t_diff / 20.0, dt_min=1e-3,
        dt_max_dT=1e4, output_dt=40.0 * t_diff,
    )
    geom = GeometryConfig(geometry_type=GeometryType.SLAB)
    results = run(stack, {"VarK": mat}, {"VarK": None},
                  geom, sbc, bbc, opts, initial_T=T_BACK, verbose=True)
    return results.snapshots[-1]


# ---------------------------------------------------------------------------
# Sub-cases
# ---------------------------------------------------------------------------

def run_subcase1_profile():
    """Sub-case 1: steady temperature profile vs Kirchhoff reference."""
    print("\n" + "=" * 60)
    print("Sub-case 1 — steady profile, Kirchhoff reference")
    print("=" * 60)
    snap = _run_steady(101)
    y, T_num = snap.mesh.y_nodes, snap.T
    T_ref = T_kirchhoff(y)
    err_K = float(np.abs(T_num - T_ref).max())
    span  = T_FRONT - T_BACK
    err_pct = err_K / span * 100.0
    print(f"\nMax |T_num − T_Kirchhoff| = {err_K:.3f} K  ({err_pct:.2f}% of span)")
    print("PASS" if err_pct < 1.0 else f"FAIL  (threshold 1% of span = {0.01*span:.1f} K)")
    return y, T_num, T_ref


def run_subcase2_convergence():
    """Sub-case 2: grid convergence — error ratio ≈ 2 on grid doubling."""
    print("\n" + "=" * 60)
    print("Sub-case 2 — grid convergence")
    print("=" * 60)
    snap_c = _run_steady(51)
    snap_f = _run_steady(101)
    err_c = float(np.abs(snap_c.T - T_kirchhoff(snap_c.mesh.y_nodes)).max())
    err_f = float(np.abs(snap_f.T - T_kirchhoff(snap_f.mesh.y_nodes)).max())
    ratio = err_c / err_f
    span  = T_FRONT - T_BACK
    print(f"\nCoarse (n=51)  max err = {err_c:.3f} K  ({err_c/span*100:.2f}%)")
    print(f"Fine   (n=101) max err = {err_f:.3f} K  ({err_f/span*100:.2f}%)")
    print(f"Error ratio (coarse/fine): {ratio:.2f}  (expected ~2.0 for 1st-order)")
    ok = err_f / span < 0.01 and 1.7 < ratio < 2.3
    print("PASS" if ok else "FAIL")
    return snap_c.mesh.y_nodes, snap_c.T, snap_f.mesh.y_nodes, snap_f.T, err_c, err_f


def run_subcase3_flux():
    """Sub-case 3: reported surface flux vs Kirchhoff q = theta_front / L."""
    print("\n" + "=" * 60)
    print("Sub-case 3 — surface flux vs Kirchhoff q")
    print("=" * 60)
    snap   = _run_steady(101)
    q_num  = snap.q_cond
    q_ref  = kirchhoff_flux()
    rel    = abs(q_num - q_ref) / q_ref * 100.0
    print(f"\nKirchhoff q_ref = {q_ref:.1f} W/m²")
    print(f"Solver   q_cond = {q_num:.1f} W/m²")
    print(f"Relative error  = {rel:.2f}%  (threshold 2%)")
    print("PASS" if rel < 2.0 else "FAIL")
    return q_num, q_ref


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    y1, T1n, T1r             = run_subcase1_profile()
    yc, Tc, yf, Tf, ec, ef  = run_subcase2_convergence()
    q_num, q_ref             = run_subcase3_flux()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(
            f"V2 — Variable-k conduction verification\n"
            f"k(T) = {_A:.3f} + {_B:.4f}·T  ({K0}→{K1} W/m/K),  "
            f"T_front={T_FRONT:.0f} K,  T_back={T_BACK:.0f} K",
            fontsize=11,
        )

        # ---- Sub-case 1: profiles ----
        ax = axes[0, 0]
        ax.plot(y1 * 1e3, T1n, "b-",  lw=1.8, label="SCAM numerical")
        ax.plot(y1 * 1e3, T1r, "r--", lw=1.5, label="Kirchhoff reference")
        # constant-k reference for comparison
        T_const_k = T_FRONT + (T_BACK - T_FRONT) * y1 / L
        ax.plot(y1 * 1e3, T_const_k, "k:", lw=1.2, label="Const-k linear (for ref)")
        ax.set_xlabel("Depth [mm]")
        ax.set_ylabel("Temperature [K]")
        ax.set_title("Steady profile  (n=101)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ---- Sub-case 1: temperature error ----
        ax = axes[1, 0]
        ax.plot(y1 * 1e3, np.abs(T1n - T1r), "b-", lw=1.5)
        tol = 0.01 * (T_FRONT - T_BACK)
        ax.axhline(tol, color="r", ls="--", lw=1.2, label=f"1% span ({tol:.0f} K)")
        ax.set_xlabel("Depth [mm]")
        ax.set_ylabel("|T_num − T_Kirchhoff|  [K]")
        ax.set_title("Profile error")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ---- Sub-case 2: convergence overlay ----
        ax = axes[0, 1]
        T_ref_c = T_kirchhoff(yc)
        T_ref_f = T_kirchhoff(yf)
        ax.plot(yf * 1e3, T_ref_f, "r--", lw=1.5, label="Kirchhoff ref")
        ax.plot(yc * 1e3, Tc, "g-",  lw=1.5, label=f"n=51  (err={ec:.1f} K)")
        ax.plot(yf * 1e3, Tf, "b-",  lw=1.5, label=f"n=101 (err={ef:.1f} K)")
        ax.set_xlabel("Depth [mm]")
        ax.set_ylabel("Temperature [K]")
        ax.set_title("Grid convergence (1st-order)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ---- Sub-case 2: pointwise errors ----
        ax = axes[1, 1]
        ax.plot(yc * 1e3, np.abs(Tc - T_ref_c), "g-", lw=1.5, label=f"n=51")
        ax.plot(yf * 1e3, np.abs(Tf - T_ref_f), "b-", lw=1.5, label=f"n=101")
        ax.axhline(0.01*(T_FRONT-T_BACK), color="r", ls="--", lw=1.2, label="1% span")
        ax.set_xlabel("Depth [mm]")
        ax.set_ylabel("|T_num − T_ref|  [K]")
        ax.set_title(f"Error ratio = {ec/ef:.2f}  (target ~2.0)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ---- Sub-case 3: Kirchhoff variable theta ----
        ax = axes[0, 2]
        theta_num = theta(y1 * 0.0)   # just allocate
        theta_num = np.array([theta(T) for T in T1n])
        theta_lin = theta(T_FRONT) * (1.0 - y1 / L)
        ax.plot(y1 * 1e3, theta_num, "b-",  lw=1.8, label="θ(T_num(y))")
        ax.plot(y1 * 1e3, theta_lin, "r--", lw=1.5, label="Linear reference θ_f(1−y/L)")
        ax.set_xlabel("Depth [mm]")
        ax.set_ylabel("θ(T)  =  ∫k dT  [W/m]")
        ax.set_title("Kirchhoff variable  (must be linear at s.s.)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ---- Sub-case 3: flux bar chart ----
        ax = axes[1, 2]
        vals  = [q_ref, q_num]
        lbls  = ["Kirchhoff\nq = θ_f/L", "Solver\nq_cond"]
        colors = ["#d62728", "#1f77b4"]
        bars = ax.bar(lbls, [v / 1e4 for v in vals], color=colors, width=0.4)
        ax.bar_label(bars, labels=[f"{v/1e4:.2f}" for v in vals], padding=3, fontsize=9)
        ax.set_ylabel("Heat flux  [W/cm²]")
        rel = abs(q_num - q_ref) / q_ref * 100.0
        ax.set_title(f"Surface flux  (rel. err = {rel:.2f}%,  threshold 2%)")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        out = Path(__file__).parent / "v2_conduction_variable.png"
        plt.savefig(out, dpi=150)
        print(f"\nPlot saved → {out}")
        plt.show()

    except ImportError:
        print("\n(matplotlib not available — skipping plots)")
    except Exception as exc:
        print(f"\n(plot failed: {exc})")


if __name__ == "__main__":
    main()
