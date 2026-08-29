# SPDX-License-Identifier: MIT
"""V2c — Transient conduction: constant vs temperature-dependent k and cp.

Runs the same 1-D slab problem twice:

    Case A — constant properties  k = k_eff,  cp = cp_mean
    Case B — variable properties  k(T) = K0→K1,  cp(T) = CP0→CP1

and compares the transient evolution side by side.

Property definitions
────────────────────
    k(T)  = K0 + (K1−K0)/(T1K−T0K) · (T−T0K)       [0.5 → 2.5 W/m/K]
    cp(T) = CP0 + (CP1−CP0)/(T1K−T0K) · (T−T0K)     [600 → 1400 J/kg/K]
    ρ     = 180 kg/m³  (constant in both cases)

The constant-property case uses the Kirchhoff-equivalent conductivity (the k
that yields the same steady-state heat flux as the variable-k profile) and the
arithmetic-mean specific heat:

    k_eff = ∫_{T_back}^{T_front} k dT / (T_front − T_back)
           = k((T_front + T_back)/2)                 [for linear k(T)]
    cp_mean = (CP0·(T_front−T0K) + ½(CP1−CP0)·(T_front−T0K)·f + ...) / ...
            ≈ cp((T_front + T_back)/2)               [for linear cp(T)]

so both cases share the same steady-state flux  q_ss = k_eff·(T_front−T_back)/L.

What the comparison reveals
────────────────────────────
During the transient the hot surface (T ≈ T_front) has k ≈ K1 = 2.5 W/m/K and
cp ≈ CP1 = 1400 J/kg/K.  The net diffusivity α(T) = k(T)/(ρ·cp(T)) is not the
same as α_eff = k_eff/(ρ·cp_mean) evaluated at the mean.  Variable-property
heat penetrates unevenly: faster near the hot surface (high k), slower in the
cold interior (low k and low cp).

Four panels compare the two cases at Fo = 0.02, 0.08, 0.20, 0.60 (using a
shared reference diffusion time α_ref·t/L²):

    [0,0]  Temperature profiles at each snapshot
    [0,1]  Surface heat flux q_cond(t) — both cases
    [1,0]  Pointwise temperature difference T_var(y,t) − T_const(y,t)
    [1,1]  Local diffusivity α(T) = k(T)/(ρ·cp(T)) across the slab at steady state
    [0,2]  Kirchhoff variable θ(y,t) for the variable case (must → linear)
    [1,2]  k(T) and cp(T) curves with operating range marked

Run from the repo root:

    python3 examples/v2c_conduction_variable_comparison.py
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
# Property definitions
# ---------------------------------------------------------------------------
T0_K, T1_K = 300.0,  3000.0   # table endpoints for property tables [K]
K0,   K1   = 0.5,    2.5      # k(T) range  [W/m/K]
CP0,  CP1  = 600.0,  1400.0   # cp(T) range [J/kg/K]
RHO        = 180.0             # density [kg/m³]  (constant)

# Boundary / geometry
T_FRONT = 2500.0   # hot face [K]
T_BACK  = 400.0    # cold face [K]
L       = 0.05     # slab thickness [m]
N_NODES = 101

# Linear interpolation coefficients
_Bk  = (K1  - K0)  / (T1_K - T0_K)
_Ak  = K0  - _Bk  * T0_K
_Bcp = (CP1 - CP0) / (T1_K - T0_K)
_Acp = CP0 - _Bcp * T0_K


def k_of_T(T):
    return np.clip(_Ak + _Bk * np.asarray(T), K0, K1)


def cp_of_T(T):
    return np.clip(_Acp + _Bcp * np.asarray(T), CP0, CP1)


# Kirchhoff-equivalent constant k (= k at arithmetic-mean temperature)
T_MID   = (T_FRONT + T_BACK) / 2.0
K_EFF   = float(k_of_T(T_MID))     # same steady flux as variable-k case
CP_MEAN = float(cp_of_T(T_MID))    # cp at mean temperature

# Reference diffusion time using the constant-case properties
ALPHA_REF = K_EFF / (RHO * CP_MEAN)
T_DIFF    = L**2 / ALPHA_REF

FO_SNAPS = [0.02, 0.08, 0.20, 0.60]
T_SNAPS  = [fo * T_DIFF for fo in FO_SNAPS]


# ---------------------------------------------------------------------------
# Kirchhoff transform (variable-k case only)
# ---------------------------------------------------------------------------

def theta(T):
    """θ(T) = ∫_{T_BACK}^{T} k(T') dT'  for linear k."""
    T = np.asarray(T, dtype=float)
    return _Ak * (T - T_BACK) + 0.5 * _Bk * (T**2 - T_BACK**2)


# ---------------------------------------------------------------------------
# Material builders
# ---------------------------------------------------------------------------

def _const_mat(name: str = "Const") -> MaterialCard:
    T_pts  = np.array([T0_K, T1_K])
    k_tab  = np.column_stack([T_pts, [K_EFF,   K_EFF]])
    cp_tab = np.column_stack([T_pts, [CP_MEAN, CP_MEAN]])
    hg_tab = np.column_stack([T_pts, [0.0, 0.0]])
    return MaterialCard(
        name=name, rho_virgin=RHO, rho_char=RHO, gamma_resin=0.0,
        components=[], k_virgin_table=k_tab, k_char_table=k_tab,
        cp_virgin_table=cp_tab, cp_char_table=cp_tab, h_g_table=hg_tab,
        decomposing=False,
    )


def _var_mat(name: str = "VarK") -> MaterialCard:
    T_pts  = np.array([T0_K, T1_K])
    k_tab  = np.column_stack([T_pts, [K0,  K1]])
    cp_tab = np.column_stack([T_pts, [CP0, CP1]])
    hg_tab = np.column_stack([T_pts, [0.0, 0.0]])
    return MaterialCard(
        name=name, rho_virgin=RHO, rho_char=RHO, gamma_resin=0.0,
        components=[], k_virgin_table=k_tab, k_char_table=k_tab,
        cp_virgin_table=cp_tab, cp_char_table=cp_tab, h_g_table=hg_tab,
        decomposing=False,
    )


# ---------------------------------------------------------------------------
# Solver runner
# ---------------------------------------------------------------------------

def _run(mat: MaterialCard, t_end: float, verbose: bool = False):
    """Run to t_end, saving snapshots at each T_SNAPS (plus a final output)."""
    name  = mat.name
    geom  = GeometryConfig(geometry_type=GeometryType.SLAB)
    stack = StackConfig(layers=[LayerConfig(name, L, N_NODES, 4)])
    sbc   = SurfaceBCConfig(
        bc_type=SurfaceBCType.PRESCRIBED_TEMP, T_prescribed=lambda t: T_FRONT,
    )
    bbc   = BackBCConfig(bc_type=BackBCType.PRESCRIBED_TEMP, T_back=T_BACK)
    alpha_min = K0 / (RHO * CP1)   # slowest diffusivity in table
    dt_max    = L**2 / alpha_min / 100.0
    dt_out    = min(np.diff([0.0] + T_SNAPS))
    opts  = SolverOptions(
        t_end=t_end, dt_init=1.0, dt_max=dt_max, dt_min=1e-3,
        dt_max_dT=1e4, output_dt=dt_out,
    )
    return run(
        stack, {name: mat}, {name: None},
        geom, sbc, bbc, opts, initial_T=T_BACK, verbose=verbose,
    )


def _pick_snaps(results, times):
    """Return snapshot closest to each requested time."""
    t_arr = np.array([s.time for s in results.snapshots])
    return [results.snapshots[int(np.argmin(np.abs(t_arr - t)))] for t in times]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_end = max(T_SNAPS) * 1.02

    print(f"k_eff  = {K_EFF:.4f} W/m/K   (= k at T_mid = {T_MID:.0f} K)")
    print(f"cp_mean = {CP_MEAN:.1f} J/kg/K  (= cp at T_mid)")
    print(f"α_ref  = {ALPHA_REF:.3e} m²/s,  t_diff = {T_DIFF:.1f} s")
    print(f"Snapshot Fo: {FO_SNAPS}  →  t [s]: {[f'{t:.1f}' for t in T_SNAPS]}")

    print("\nRunning constant-k/cp case...")
    res_const = _run(_const_mat(), t_end)
    snaps_const = _pick_snaps(res_const, T_SNAPS)

    print("Running variable-k/cp case...")
    res_var   = _run(_var_mat(),   t_end)
    snaps_var = _pick_snaps(res_var, T_SNAPS)

    # Collect surface-flux history from both runs
    t_const = np.array([s.time for s in res_const.snapshots])
    q_const = np.array([s.q_cond for s in res_const.snapshots])
    t_var   = np.array([s.time for s in res_var.snapshots])
    q_var   = np.array([s.q_cond for s in res_var.snapshots])

    # Steady-state flux reference (same for both by construction)
    q_ss = K_EFF * (T_FRONT - T_BACK) / L

    # Print comparison table
    theta_front = float(theta(T_FRONT))
    print(f"\n{'Fo':>5}  {'t [s]':>7}  {'T_surf_const':>12}  {'T_surf_var':>10}  "
          f"{'ΔT_surf [K]':>11}  {'θ-residual [%]':>14}")
    for sc, sv, fo in zip(snaps_const, snaps_var, FO_SNAPS):
        dT_surf = float(sv.T[0] - sc.T[0])
        th      = theta(sv.T)
        th_lin  = theta_front * (1.0 - sv.mesh.y_nodes / L)
        res_th  = float(np.abs(th - th_lin).max()) / theta_front * 100
        print(f"{fo:5.2f}  {sc.time:7.1f}  {sc.T[0]:12.2f}  {sv.T[0]:10.2f}  "
              f"{dT_surf:11.2f}  {res_th:14.3f}")

    print(f"\nSteady flux (both cases, by construction): q_ss = {q_ss:.1f} W/m²")
    print(f"  Const-k  q_cond final = {snaps_const[-1].q_cond:.1f} W/m²")
    print(f"  Var-k    q_cond final = {snaps_var[-1].q_cond:.1f} W/m²")

    # ── Plotting ─────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        colors = cm.plasma(np.linspace(0.1, 0.85, len(FO_SNAPS)))
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        fig.suptitle(
            f"V2c — Transient conduction: constant vs variable k(T) and cp(T)\n"
            f"k: {K0}→{K1} W/m/K,  cp: {CP0:.0f}→{CP1:.0f} J/kg/K,  "
            f"k_eff={K_EFF:.3f} W/m/K,  cp_mean={CP_MEAN:.0f} J/kg/K  "
            f"(both give same q_ss = {q_ss/1e4:.2f} W/cm²)",
            fontsize=10,
        )

        # ── [0,0]  Temperature profiles ──────────────────────────────────────
        ax = axes[0, 0]
        for sc, sv, fo, c in zip(snaps_const, snaps_var, FO_SNAPS, colors):
            y = sc.mesh.y_nodes * 1e3
            ax.plot(y, sc.T, "-",  color=c, lw=1.5, label=f"const  Fo={fo:.2f}")
            ax.plot(y, sv.T, "--", color=c, lw=2.0)
        # Legend proxy
        from matplotlib.lines import Line2D
        ax.plot([], [], "k-",  lw=1.5, label="solid = const k/cp")
        ax.plot([], [], "k--", lw=2.0, label="dash  = var k(T)/cp(T)")
        ax.set_xlabel("Depth [mm]")
        ax.set_ylabel("Temperature [K]")
        ax.set_title("Temperature profiles  (colour = Fo snapshot)")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        # ── [0,1]  Surface flux vs time ───────────────────────────────────────
        ax = axes[0, 1]
        fo_c = ALPHA_REF * t_const / L**2
        fo_v = ALPHA_REF * t_var   / L**2
        ax.plot(fo_c, q_const / 1e4, "b-",  lw=2,   label="const k/cp")
        ax.plot(fo_v, q_var   / 1e4, "r--", lw=2,   label="var k(T)/cp(T)")
        ax.axhline(q_ss / 1e4, color="gray", ls=":", lw=1.2,
                   label=f"q_ss = {q_ss/1e4:.2f} W/cm²")
        for fo in FO_SNAPS:
            ax.axvline(fo, color="gray", ls="--", lw=0.7, alpha=0.5)
        ax.set_xlabel("Fourier number  Fo = α_ref·t/L²")
        ax.set_ylabel("Surface heat flux  [W/cm²]")
        ax.set_title("Surface flux evolution\n(var-k starts higher: k(T_front) > k_eff)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── [0,2]  Kirchhoff θ(y,t) for variable case ─────────────────────────
        ax = axes[0, 2]
        y_ref   = np.linspace(0, L, 300)
        theta_f = float(theta(T_FRONT))
        ax.plot(y_ref * 1e3, theta_f * (1.0 - y_ref / L), "k--",
                lw=1.5, label="linear (steady state)")
        for sv, fo, c in zip(snaps_var, FO_SNAPS, colors):
            th = theta(sv.T)
            ax.plot(sv.mesh.y_nodes * 1e3, th, "-", color=c, lw=1.5,
                    label=f"Fo={fo:.2f}")
        ax.set_xlabel("Depth [mm]")
        ax.set_ylabel("θ(T) = ∫k dT  [W/m]")
        ax.set_title("Kirchhoff variable (var case)\n→ linear at steady state")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── [1,0]  Temperature difference T_var − T_const ─────────────────────
        ax = axes[1, 0]
        for sc, sv, fo, c in zip(snaps_const, snaps_var, FO_SNAPS, colors):
            y    = sc.mesh.y_nodes * 1e3
            diff = sv.T - sc.T
            ax.plot(y, diff, "-", color=c, lw=1.5, label=f"Fo={fo:.2f}")
        ax.axhline(0, color="gray", ls="--", lw=0.8)
        ax.set_xlabel("Depth [mm]")
        ax.set_ylabel("T_var − T_const  [K]")
        ax.set_title(
            "Temperature difference  (var − const)\n"
            "> 0: var heats faster;  < 0: var heats slower"
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── [1,1]  Local diffusivity α(T) at steady state ─────────────────────
        ax = axes[1, 1]
        # Use the variable-k steady-state profile (last snapshot)
        sv_ss = snaps_var[-1]
        T_ss  = sv_ss.T
        k_ss  = k_of_T(T_ss)
        cp_ss = cp_of_T(T_ss)
        alpha_ss = k_ss / (RHO * cp_ss)
        y_ss = sv_ss.mesh.y_nodes * 1e3
        ax.plot(y_ss, alpha_ss * 1e6, "r-",  lw=2,   label="var α(T) at s.s.")
        ax.axhline(ALPHA_REF * 1e6, color="b", ls="--", lw=1.5,
                   label=f"const α_ref = {ALPHA_REF*1e6:.2f} mm²/s")
        ax_right = ax.twinx()
        ax_right.plot(y_ss, T_ss, "gray", lw=1, ls=":", alpha=0.6)
        ax_right.set_ylabel("T(y) at s.s.  [K]", color="gray", fontsize=8)
        ax_right.tick_params(axis="y", colors="gray")
        ax.set_xlabel("Depth [mm]")
        ax.set_ylabel("α(T) = k/(ρ·cp)  [mm²/s]")
        ax.set_title("Local diffusivity α(T) at steady state\n"
                     "var-k faster near hot face, slower in cold interior")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── [1,2]  k(T) and cp(T) property curves ─────────────────────────────
        ax = axes[1, 2]
        T_range = np.linspace(T0_K, T1_K, 300)
        k_r     = k_of_T(T_range)
        ax.plot(T_range, k_r, "r-", lw=2, label="k(T)  [W/m/K]")
        ax.axhline(K_EFF, color="r", ls="--", lw=1.2, label=f"k_eff = {K_EFF:.3f}")
        ax.axvline(T_BACK,  color="gray", ls="--", lw=0.8)
        ax.axvline(T_FRONT, color="gray", ls="--", lw=0.8)
        ax.axvspan(T_BACK, T_FRONT, alpha=0.06, color="gray", label="operating range")
        ax.set_xlabel("Temperature [K]")
        ax.set_ylabel("k(T)  [W/m/K]", color="r")
        ax.tick_params(axis="y", colors="r")

        ax2 = ax.twinx()
        cp_r = cp_of_T(T_range)
        ax2.plot(T_range, cp_r, "b-", lw=2, label="cp(T)  [J/kg/K]")
        ax2.axhline(CP_MEAN, color="b", ls="--", lw=1.2,
                    label=f"cp_mean = {CP_MEAN:.0f}")
        ax2.set_ylabel("cp(T)  [J/kg/K]", color="b")
        ax2.tick_params(axis="y", colors="b")

        # merge legends
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, fontsize=7, loc="upper left")
        ax.set_title(f"Property curves  k: {K0}→{K1} W/m/K,  cp: {CP0:.0f}→{CP1:.0f} J/kg/K")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out = Path(__file__).parent / "v2c_conduction_variable_comparison.png"
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
