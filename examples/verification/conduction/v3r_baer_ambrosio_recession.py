# SPDX-License-Identifier: MIT
"""V3r — Semi-infinite receding solid: Baer & Ambrosio (1961).

Reference:
    D. Baer and A. Ambrosio, "Heat conduction in a semi-infinite slab with
    sublimation at the surface," Planetary and Space Science, vol. 4,
    pp. 436–446, 1961.

Problem:
    A semi-infinite solid initially at uniform temperature T₀ is exposed at
    t=0 to a prescribed surface temperature Tw AND a constant surface
    recession rate ṡ.  The analytical solution in terms of depth x measured
    from the CURRENT (receding) surface is:

        T = T₀ + (Tw − T₀)/2 · [ erfc((x + ṡt)/(2√(αt)))
                                  + exp(−xṡ/α) · erfc((x − ṡt)/(2√(αt))) ]

    In SCAM's y-coordinate (measured from the ORIGINAL front face at t=0):
        x = y − ṡt   (depth below current surface; ṡt = s_total for const ṡ)

    So the formula becomes:
        T = T₀ + (Tw − T₀)/2 · [ erfc(y / (2√(αt)))
                                  + exp(−(y − ṡt)·ṡ/α) · erfc((y − 2ṡt) / (2√(αt))) ]

    Steady-state (t → ∞, in frame of receding surface):
        T = T₀ + (Tw − T₀) · exp(−xṡ/α)

Parameters:
    T₀ = 300 K, Tw = 4000 K, ṡ = 1 mm/s
    ρ = 1850 kg/m³, cp = 2000 J/kg/K, k = 30 W/m/K
    α = k/(ρcp) ≈ 8.108×10⁻⁶ m²/s
    Thermal length scale α/ṡ ≈ 8.1 mm  (profile ≈ steady by t ≳ 30 s)

SCAM setup:
    PRESCRIBED_TEMP BC (T_wall = Tw) + SolverOptions.s_dot_prescribed = ṡ.
    ALE (continuous_remap=True) gives smooth T_wall; CFL: dt ≤ 0.1·dx/ṡ = 0.01 s.
    Slab 200 mm, 2000 nodes → dx = 0.1 mm.  At t=100 s, back face ΔT < 0.01 K.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scam.config.boundary import BackBCConfig, BackBCType, SurfaceBCConfig, SurfaceBCType
from scam.config.geometry import GeometryConfig, GeometryType
from scam.config.material import MaterialCard
from scam.config.solver import SolverOptions
from scam.config.stack import LayerConfig, StackConfig
from scam.solvers.material_response import run

# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------
RHO   = 1850.0
CP    = 2000.0
K     = 30.0
ALPHA = K / (RHO * CP)   # ≈ 8.108e-6 m²/s

T0    = 300.0             # [K] initial / far-field
TW    = 4000.0            # [K] prescribed surface temperature
SDOT  = 1e-3              # [m/s] constant recession rate (1 mm/s)

SSLAB   = 0.20            # [m] 200 mm — semi-infinite for t ≤ 100 s
N_NODES = 2000            # dx = 0.1 mm
DT_MAX  = 0.01            # [s] satisfies ALE CFL: 0.1·(0.1mm)/(1mm/s) = 0.01 s

T_SNAPS = [10.0, 30.0, 60.0, 100.0]   # [s]


# ---------------------------------------------------------------------------
# Analytical solution
# ---------------------------------------------------------------------------

def baer_ambrosio_T(y, t):
    """Transient analytical solution in SCAM y-coordinate (from original front face).

    Valid for y ≥ ṡ·t (material that has not yet been ablated).
    """
    y = np.asarray(y, dtype=float)
    s = SDOT * t                      # total recession at time t
    sqrt_at2 = 2.0 * np.sqrt(ALPHA * t)
    exp_fac  = np.exp(-(y - s) * SDOT / ALPHA)
    return T0 + 0.5 * (TW - T0) * (erfc(y / sqrt_at2) + exp_fac * erfc((y - 2.0*s) / sqrt_at2))


def baer_ambrosio_steady(x):
    """Steady-state temperature as function of depth x from the current surface."""
    x = np.asarray(x, dtype=float)
    return T0 + (TW - T0) * np.exp(-x * SDOT / ALPHA)


# ---------------------------------------------------------------------------
# SCAM simulation
# ---------------------------------------------------------------------------

def _make_material():
    cp_tab = np.column_stack([[1.0, TW + 2000.0], [CP, CP]])
    k_tab  = np.column_stack([[1.0, TW + 2000.0], [K,  K]])
    hg_tab = np.column_stack([[1.0, TW + 2000.0], [0.0, 0.0]])
    return MaterialCard(
        name="BaerAmbrosio", rho_virgin=RHO, rho_char=RHO, gamma_resin=0.0,
        components=[], k_virgin_table=k_tab, k_char_table=k_tab,
        cp_virgin_table=cp_tab, cp_char_table=cp_tab, h_g_table=hg_tab,
        emissivity=0.0, decomposing=False,
    )


def run_scam():
    mat   = _make_material()
    geom  = GeometryConfig(geometry_type=GeometryType.SLAB)
    stack = StackConfig(layers=[LayerConfig("BaerAmbrosio", SSLAB, N_NODES, 1)])
    surface_bc = SurfaceBCConfig(
        bc_type=SurfaceBCType.PRESCRIBED_TEMP, T_prescribed=lambda t: TW,
    )
    back_bc = BackBCConfig(bc_type=BackBCType.ADIABATIC)
    options = SolverOptions(
        t_end=max(T_SNAPS),
        dt_init=1e-4,
        dt_max=DT_MAX,
        dt_min=1e-6,
        dt_max_dT=1e4,
        output_dt=min(T_SNAPS) / 2.0,
        s_dot_prescribed=SDOT,
        continuous_remap=True,
        node_drop_threshold=0.2,
    )
    return run(stack, {"BaerAmbrosio": mat}, {"BaerAmbrosio": None},
               geom, surface_bc, back_bc, options, initial_T=T0, verbose=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results = run_scam()
    saved_times = np.array([s.time for s in results.snapshots])
    snaps = {
        t_req: results.snapshots[int(np.argmin(np.abs(saved_times - t_req)))]
        for t_req in T_SNAPS
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax_prof, ax_ss, ax_err, ax_ss_log = axes.flat

    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(T_SNAPS)))

    # --- panel 1: transient profiles ---
    for color, (t_req, snap) in zip(colors, snaps.items()):
        y = snap.mesh.y_nodes
        s = snap.mesh.s_total
        x = y - s                          # depth below current surface

        mask = x >= 0.0
        T_num = snap.T[mask]
        T_ana = baer_ambrosio_T(y[mask], snap.time)
        x_plt = x[mask] * 1e3             # → mm

        ax_prof.plot(x_plt, T_ana, '-',  color=color, lw=1.5,
                     label=f't={t_req:.0f} s (analytical)')
        ax_prof.plot(x_plt, T_num, '--', color=color, lw=1.0,
                     label=f't={t_req:.0f} s (SCAM)')

    T_ss_x = np.linspace(0, 60e-3, 500)
    ax_prof.plot(T_ss_x * 1e3, baer_ambrosio_steady(T_ss_x), 'k:',
                 lw=1.5, label='Steady state (t→∞)')
    ax_prof.set_xlabel('Depth from current surface x [mm]')
    ax_prof.set_ylabel('Temperature [K]')
    ax_prof.set_title('Transient profiles (SCAM vs analytical)')
    ax_prof.set_xlim(left=0)
    # deduplicate legend (show only analytical + SCAM + ss)
    handles, labels = ax_prof.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_prof.legend(by_label.values(), by_label.keys(), fontsize=8)

    # --- panel 2: steady-state comparison (t=100 s) ---
    snap100 = snaps[100.0]
    y100 = snap100.mesh.y_nodes
    x100 = (y100 - snap100.mesh.s_total) * 1e3    # mm
    T_ss_num = snap100.T
    T_ss_ana = baer_ambrosio_steady((y100 - snap100.mesh.s_total))
    ax_ss.plot(x100, T_ss_ana, 'k-',  lw=2, label='Analytical steady state')
    ax_ss.plot(x100, T_ss_num, 'r--', lw=1, label='SCAM t=100 s')
    ax_ss.set_xlabel('Depth from current surface x [mm]')
    ax_ss.set_ylabel('Temperature [K]')
    ax_ss.set_title('Steady-state profile (t=100 s)')
    ax_ss.set_xlim(0, 60)
    ax_ss.legend(fontsize=9)

    # --- panel 3: relative error vs time ---
    for color, (t_req, snap) in zip(colors, snaps.items()):
        y = snap.mesh.y_nodes
        s = snap.mesh.s_total
        x = y - s
        T_num = snap.T
        T_ana = baer_ambrosio_T(y, snap.time)
        dT_ref = T_ana - T0
        mask = dT_ref > 0.05 * (TW - T0)
        if mask.sum() > 0:
            rel_err = np.abs((T_num[mask] - T_ana[mask]) / dT_ref[mask]) * 100.0
            ax_err.plot(x[mask] * 1e3, rel_err, color=color, label=f't={t_req:.0f} s')
    ax_err.set_xlabel('Depth from current surface x [mm]')
    ax_err.set_ylabel('Relative error [%]')
    ax_err.set_title('Profile relative error (comparison band ΔT > 5% of ΔTmax)')
    ax_err.axhline(3.0, color='gray', ls='--', lw=1, label='3% threshold')
    ax_err.set_xlim(left=0)
    ax_err.legend(fontsize=8)

    # --- panel 4: steady-state log-scale ---
    x_ss = (snap100.mesh.y_nodes - snap100.mesh.s_total) * 1e3
    T_ss = snap100.T
    theta_num = (T_ss - T0) / (TW - T0)
    theta_ana = np.exp(-x_ss * 1e-3 * SDOT / ALPHA)
    mask_pos = theta_num > 1e-6
    ax_ss_log.semilogy(x_ss[mask_pos], theta_num[mask_pos], 'r-', lw=1.5, label='SCAM t=100 s')
    ax_ss_log.semilogy(x_ss[mask_pos], theta_ana[mask_pos], 'k--', lw=1.5,
                       label=r'exp(−xṡ/α)')
    ax_ss_log.set_xlabel('Depth from current surface x [mm]')
    ax_ss_log.set_ylabel(r'$(T - T_0)/(T_w - T_0)$')
    ax_ss_log.set_title('Steady-state log scale (t=100 s)')
    ax_ss_log.set_xlim(0, 60)
    ax_ss_log.legend(fontsize=9)

    fig.suptitle(
        f'Baer & Ambrosio (1961) recession verification\n'
        f'ṡ={SDOT*1e3:.1f} mm/s, Tw={TW:.0f} K, α/ṡ={ALPHA/SDOT*1e3:.1f} mm, '
        f'N={N_NODES}, dx={SSLAB/N_NODES*1e3:.2f} mm',
        fontsize=11,
    )
    fig.tight_layout()

    out = Path(__file__).with_suffix('.png')
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
