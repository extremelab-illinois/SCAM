# SPDX-License-Identifier: MIT
"""V2b — Transient approach to the Kirchhoff steady state.

Extends v2_conduction_variable.py by tracking the evolution of the temperature
field and the Kirchhoff variable theta(y) from the initial condition toward the
steady-state reference at four dimensionless time snapshots:

    Fo ≈ 0.25, 1, 5, 20   (expressed as multiples of t_diff = L²/α_ref)

Key insight: theta(y) = ∫_{T_back}^{T(y)} k dT' must become LINEAR in y as
steady state is approached — regardless of how nonlinear k(T) is.  Plotting
theta(y,t) reveals this convergence far more clearly than the temperature
profile itself, because the nonlinear shape of T(y) can mask the error while
the residual of theta from linear is a direct measure of how far from steady
state the solver is.

Three panel pairs (top = profile, bottom = error/linearity):
  Left   — temperature profiles converging to the Kirchhoff reference
  Centre — theta(y,t) converging to the linear reference θ_front*(1-y/L)
  Right  — error of theta from linear over time  (convergence to zero)

Run from the repo root:

    python3 examples/v2b_conduction_variable_transient.py
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
# Material / geometry constants
# ---------------------------------------------------------------------------
K0, K1     = 0.5, 2.5        # k(T) endpoints  [W/m/K]
T0_K, T1_K = 300.0, 3000.0
RHO, CP    = 180.0, 710.0
T_FRONT    = 2500.0           # hot face  [K]
T_BACK     = 400.0            # cold face [K]
L          = 0.05             # slab thickness  [m]

# Representative diffusivity (geometric mean of k range)
K_REF  = np.sqrt(K0 * K1)
ALPHA  = K_REF / (RHO * CP)
T_DIFF = L**2 / ALPHA         # reference diffusion time [s]

# Snapshot times as multiples of T_DIFF.
# The fixed-end slab equilibrates as exp(-pi^2 * Fo), so it is essentially
# at steady state by Fo~0.5.  The interesting transient is Fo < 0.5.
FO_SNAPS  = [0.02, 0.08, 0.20, 0.60]
T_SNAPS   = [fo * T_DIFF for fo in FO_SNAPS]

N_NODES = 101

# Linear-k coefficients
_B = (K1 - K0) / (T1_K - T0_K)
_A = K0 - _B * T0_K


# ---------------------------------------------------------------------------
# Kirchhoff helpers
# ---------------------------------------------------------------------------

def theta(T):
    """Kirchhoff transform  theta(T) = ∫_{T_BACK}^{T} k dT'  for linear k."""
    T = np.asarray(T)
    return _A * (T - T_BACK) + 0.5 * _B * (T**2 - T_BACK**2)


def T_kirchhoff(y):
    """Steady-state reference: invert theta(T(y)) = theta_front * (1 - y/L)."""
    y = np.asarray(y)
    target = theta(T_FRONT) * (1.0 - y / L)
    out = np.empty_like(target)
    for i, tg in enumerate(target.flat):
        c = -(_A * T_BACK + 0.5 * _B * T_BACK**2 + tg)
        disc = _A**2 - 4.0 * (0.5 * _B) * c
        out.flat[i] = (-_A + np.sqrt(max(disc, 0.0))) / _B
    return out


# ---------------------------------------------------------------------------
# Material builder & runner
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


def _run_transient():
    """Run to t_end=20*T_DIFF and collect snapshots at FO_SNAPS multiples."""
    t_end = max(T_SNAPS) * 1.01
    alpha_min = K0 / (RHO * CP)
    # dt_max: Fo_step ≈ 0.1 for the slowest diffusion time
    dt_max = L**2 / alpha_min / 100.0
    dt_out = min(np.diff([0.0] + T_SNAPS))

    mat = _var_k_mat()
    geom = GeometryConfig(geometry_type=GeometryType.SLAB)
    stack = StackConfig(layers=[LayerConfig("VarK", L, N_NODES, 4)])
    sbc = SurfaceBCConfig(
        bc_type=SurfaceBCType.PRESCRIBED_TEMP, T_prescribed=lambda t: T_FRONT,
    )
    bbc = BackBCConfig(bc_type=BackBCType.PRESCRIBED_TEMP, T_back=T_BACK)
    opts = SolverOptions(
        t_end=t_end, dt_init=1.0, dt_max=dt_max, dt_min=1e-2,
        dt_max_dT=1e4, output_dt=dt_out,
    )
    results = run(stack, {"VarK": mat}, {"VarK": None},
                  geom, sbc, bbc, opts, initial_T=T_BACK, verbose=False)

    # Pick the snapshot closest to each requested time
    times = np.array([s.time for s in results.snapshots])
    snaps = []
    for t in T_SNAPS:
        idx = int(np.argmin(np.abs(times - t)))
        snaps.append(results.snapshots[idx])
    return snaps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"t_diff = {T_DIFF:.1f} s  (L²/α_ref, α_ref = {ALPHA:.2e} m²/s)")
    print(f"Snapshot times: {[f'{t:.0f}s (Fo={fo:.2f})' for fo, t in zip(FO_SNAPS, T_SNAPS)]}")
    print("\nRunning transient...")
    snaps = _run_transient()

    theta_front = float(theta(T_FRONT))

    # Error table
    print(f"\n{'Fo':>6}  {'t [s]':>7}  {'max |T−T_ref| [K]':>20}  {'max |θ−linear|/θ_f':>22}")
    for snap, fo in zip(snaps, FO_SNAPS):
        y, T_num = snap.mesh.y_nodes, snap.T
        T_ref = T_kirchhoff(y)
        theta_num = theta(T_num)
        theta_lin = theta_front * (1.0 - y / L)
        err_T  = float(np.abs(T_num - T_ref).max())
        err_th = float(np.abs(theta_num - theta_lin).max()) / theta_front
        print(f"{fo:6.2f}  {snap.time:7.1f}  {err_T:20.2f}  {err_th:22.4f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        colors = cm.plasma(np.linspace(0.1, 0.85, len(FO_SNAPS)))

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        fig.suptitle(
            f"V2b — Variable-k transient  "
            f"k(T)={K0}→{K1} W/m/K,  T_front={T_FRONT:.0f} K,  T_back={T_BACK:.0f} K\n"
            f"solid = SCAM,  dashed = steady-state Kirchhoff reference",
            fontsize=11,
        )

        # Steady reference on fine grid
        y_ref = np.linspace(0, L, 300)
        T_ss  = T_kirchhoff(y_ref)
        th_ss = theta(T_ss)
        th_lin = theta_front * (1.0 - y_ref / L)

        # ---- Temperature profiles ----
        ax = axes[0, 0]
        ax.plot(y_ref * 1e3, T_ss, "k--", lw=1.5, label="Steady-state ref")
        for snap, fo, c in zip(snaps, FO_SNAPS, colors):
            ax.plot(snap.mesh.y_nodes * 1e3, snap.T, "-", color=c, lw=1.5,
                    label=f"Fo={fo:.2f}  (t={snap.time:.0f}s)")
        ax.set_xlabel("Depth [mm]")
        ax.set_ylabel("Temperature [K]")
        ax.set_title("Temperature T(y,t)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ---- Temperature error from steady reference ----
        ax = axes[1, 0]
        for snap, fo, c in zip(snaps, FO_SNAPS, colors):
            y = snap.mesh.y_nodes
            err = np.abs(snap.T - T_kirchhoff(y))
            ax.plot(y * 1e3, err, "-", color=c, lw=1.5, label=f"Fo={fo:.2f}")
        ax.axhline(0.01*(T_FRONT-T_BACK), color="r", ls="--", lw=1.2, label="1% span")
        ax.set_xlabel("Depth [mm]")
        ax.set_ylabel("|T − T_ss|  [K]")
        ax.set_title("Error vs steady reference")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ---- Kirchhoff variable theta ----
        ax = axes[0, 1]
        ax.plot(y_ref * 1e3, th_lin, "k--", lw=1.5, label="Linear (steady state)")
        for snap, fo, c in zip(snaps, FO_SNAPS, colors):
            th = theta(snap.T)
            ax.plot(snap.mesh.y_nodes * 1e3, th, "-", color=c, lw=1.5,
                    label=f"Fo={fo:.2f}")
        ax.set_xlabel("Depth [mm]")
        ax.set_ylabel("θ(T)  =  ∫k dT  [W/m]")
        ax.set_title("Kirchhoff variable  θ(y,t)\n(converges to linear)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ---- theta linearity residual ----
        ax = axes[1, 1]
        for snap, fo, c in zip(snaps, FO_SNAPS, colors):
            y = snap.mesh.y_nodes
            th = theta(snap.T)
            th_l = theta_front * (1.0 - y / L)
            ax.plot(y * 1e3, np.abs(th - th_l) / theta_front * 100, "-",
                    color=c, lw=1.5, label=f"Fo={fo:.2f}")
        ax.set_xlabel("Depth [mm]")
        ax.set_ylabel("|θ − linear| / θ_front  [%]")
        ax.set_title("θ linearity residual  (→ 0 at s.s.)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ---- Max theta residual vs Fourier number ----
        ax = axes[0, 2]
        fo_arr  = np.array(FO_SNAPS)
        res_arr = np.array([
            float(np.abs(theta(s.T) - theta_front*(1.0 - s.mesh.y_nodes/L)).max())
            / theta_front * 100
            for s in snaps
        ])
        ax.semilogy(fo_arr, res_arr, "o-", lw=1.8, ms=7, color="#1f77b4")
        ax.set_xlabel("Fourier number  Fo = α t / (L²/α/α_ref)")
        ax.set_ylabel("max |θ − linear| / θ_front  [%]")
        ax.set_title("Convergence to steady state")
        ax.grid(True, alpha=0.3, which="both")

        # ---- k(T) curve to show non-linearity ----
        ax = axes[1, 2]
        T_range = np.linspace(T_BACK, T_FRONT, 200)
        k_range = _A + _B * T_range
        ax.plot(T_range, k_range, "b-", lw=2)
        ax.axvline(T_BACK,  color="gray", ls="--", lw=1, label=f"T_back={T_BACK:.0f}K")
        ax.axvline(T_FRONT, color="gray", ls=":",  lw=1, label=f"T_front={T_FRONT:.0f}K")
        ax.set_xlabel("Temperature [K]")
        ax.set_ylabel("k(T)  [W/m/K]")
        ax.set_title(f"Conductivity  k={K0}→{K1} W/m/K  (5× variation)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out = Path(__file__).parent / "v2b_conduction_variable_transient.png"
        plt.savefig(out, dpi=150)
        print(f"\nPlot saved → {out}")
        plt.show()

    except ImportError:
        print("\n(matplotlib not available — skipping plots)")
    except Exception as exc:
        print(f"\n(plot failed: {exc})")


if __name__ == "__main__":
    main()
