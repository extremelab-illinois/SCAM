# SPDX-License-Identifier: MIT
"""Compare SCAM vs PATO AblationTestCase_1.0.

Prescribed surface temperature, no surface ablation, 5 cm TACOT slab.
Material loaded from ``scam/materials/ablative_organic/tacot_v3.0.yaml``.
h_g replaced with PATO gasProperties 1-atm sub-table for consistency.

Run:
    MPLBACKEND=Agg python3 examples/verification/ablation1/compare_pato_ablation1.py
"""
from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO))

from scam.config.boundary import BackBCConfig, BackBCType, SurfaceBCConfig, SurfaceBCType
from scam.config.geometry import GeometryConfig, GeometryType
from scam.config.solver import SolverOptions
from scam.config.stack import LayerConfig, StackConfig
from scam.io.material_loader import load_material
from scam.solvers.material_response import run

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PATO_CASE = Path.home() / "PATO-dev/tutorials/1D/AblationTestCase_1.0"
_LOCAL_REF_CASE = Path(__file__).resolve().parent / "pato_reference/AblationTestCase_1.0"
# Repo-bundled copy of the small PATO reference outputs (see README's "Note on
# PATO reference data") is used when present; falls back to a local PATO-dev
# checkout for regenerating/extending the bundled set.
PATO_REF_CASE = _LOCAL_REF_CASE if _LOCAL_REF_CASE.exists() else Path.home() / (
    "PATO-dev/src/applications/utilities/tests/testsuites/tutorials/ref/1D/AblationTestCase_1.0"
)
PATO_MAT = Path.home() / "PATO-dev/data/Materials/Composites/TACOT"
PATO_TA  = PATO_REF_CASE / "output/porousMat/scalar/Ta_plot"
PATO_RHO = PATO_REF_CASE / "output/porousMat/scalar/rho_s_plot"
# FIAT reference is bundled too (it lives under the PATO *tutorials* tree, a
# different root from the testsuites `ref` tree above, so it needs its own
# bundled-first resolution). Without this, a checkout with no local PATO-dev
# could not run this script at all.
_LOCAL_FIAT = Path(__file__).resolve().parent / "pato_reference/AblationTestCase_1.0/data/ref/FIAT/T"
FIAT_T   = _LOCAL_FIAT if _LOCAL_FIAT.exists() else PATO_CASE / "data/ref/FIAT/T"

TACOT_YAML = REPO / "scam/materials/ablative_organic/tacot_v3.0.yaml"
OUT_PNG      = Path(__file__).parent / "compare_pato_ablation1.png"

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
THICKNESS  = 0.05     # m
N_NODES    = 501
T_INIT     = 300.0    # K
T_SURF     = 1644.0   # K
T_END      = 120.0    # s
RHO_VIRGIN = 280.0    # kg/m³
RHO_CHAR   = 220.0    # kg/m³

TC_DEPTHS = [0.001, 0.002, 0.004, 0.008, 0.012, 0.016, 0.024]  # m

# ---------------------------------------------------------------------------
# Load h_g from PATO gasProperties (1 atm column)
# ---------------------------------------------------------------------------

def _load_pato_hg() -> tuple[np.ndarray, np.ndarray]:
    gas_file = PATO_MAT / "gasProperties"
    if gas_file.exists():
        rows = []
        with open(gas_file) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 5:
                    try:
                        p, T, M, hg, nu = [float(x) for x in parts]
                        if p > 1e5:
                            rows.append((T, hg))
                    except ValueError:
                        pass
        rows.sort()
        return np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
    # fallback subset if PATO not installed
    T   = np.array([200., 300., 500., 700., 800., 900., 1000., 1100., 1200.,
                    1300., 1400., 1500., 1644., 2000., 3000., 4000.])
    hg  = np.array([-7.247e6, -7.090e6, -6.715e6, -6.005e6, -5.014e6, -3.335e6,
                    -2.170e6, -1.789e6, -1.199e6, -5.255e5,  1.299e5,  1.137e6,
                     2.625e6,  4.400e6,  1.100e7,  2.200e7])
    return T, hg


# ---------------------------------------------------------------------------
# Temperature BC: step at 0.1 s, step back at 60.1 s
# ---------------------------------------------------------------------------

_T_BC = np.array([
    [0.0,   T_INIT],
    [0.099, T_INIT],
    [0.1,   T_SURF],
    [60.0,  T_SURF],
    [60.1,  T_INIT],
    [T_END, T_INIT],
])


def _T_surf(t: float) -> float:
    return float(np.interp(t, _T_BC[:, 0], _T_BC[:, 1]))


# ---------------------------------------------------------------------------
# Run PATO (if output not already present)
# ---------------------------------------------------------------------------

def run_pato() -> None:
    if PATO_TA.exists():
        print(f"PATO output found at {PATO_TA}; skipping re-run.")
        return
    print("Running PATO AblationTestCase_1.0 ...")
    cmd = (
        f"cd {PATO_CASE} && "
        "cp -r origin.0 0 && "
        "blockMesh -region porousMat && "
        "PATOx"
    )
    result = subprocess.run(
        ["bash", "-c", cmd],
        env={**__import__("os").environ,
             "PATH": f"{Path.home()}/PATO-dev/bin:" +
                     __import__("os").environ.get("PATH", "")},
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("PATO stderr:", result.stderr[-2000:])
        raise RuntimeError("PATO run failed")
    print("PATO run complete.")


# ---------------------------------------------------------------------------
# Parse PATO output
# ---------------------------------------------------------------------------

def load_pato_T() -> tuple[np.ndarray, np.ndarray]:
    rows = []
    with open(PATO_TA) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                vals = [float(x) for x in line.split()]
                if len(vals) == 8:
                    rows.append(vals)
            except ValueError:
                continue
    arr = np.array(rows)
    return arr[:, 0], arr[:, 1:]


def load_pato_rho() -> tuple[np.ndarray, np.ndarray]:
    rows = []
    with open(PATO_RHO) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                vals = [float(x) for x in line.split()]
                if len(vals) == 8:
                    rows.append(vals)
            except ValueError:
                continue
    arr = np.array(rows)
    return arr[:, 0], arr[:, 1:]


def load_fiat() -> tuple[np.ndarray, np.ndarray]:
    rows = []
    with open(FIAT_T) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("/"):
                continue
            try:
                vals = [float(x) for x in line.split()]
                if len(vals) >= 7:
                    rows.append(vals[:7])
            except ValueError:
                continue
    arr = np.array(rows)
    return arr[:, 0], arr[:, 1:]


# ---------------------------------------------------------------------------
# Run SCAM
# ---------------------------------------------------------------------------

def run_scam():
    mat, _ = load_material(str(TACOT_YAML))
    HG_T, HG_VAL = _load_pato_hg()
    mat = dataclasses.replace(mat,
                              h_g_table=np.column_stack([HG_T, HG_VAL]),
                              h_g_abs_offset=None)

    mat_cards      = {mat.name: mat}
    b_prime_tables = {mat.name: None}

    stack = StackConfig(layers=[
        LayerConfig(mat.name, thickness=THICKNESS, n_nodes=N_NODES, n_subcells=4),
    ])
    geom = GeometryConfig(geometry_type=GeometryType.SLAB)

    surface_bc = SurfaceBCConfig(
        bc_type=SurfaceBCType.PRESCRIBED_TEMP,
        T_prescribed=_T_surf,
    )
    back_bc = BackBCConfig(bc_type=BackBCType.ADIABATIC)

    options = SolverOptions(
        t_end=T_END,
        dt_init=0.01,
        dt_max=0.5,
        dt_min=1e-4,
        dt_max_dT=20.0,
        output_dt=0.5,
        tc_positions=TC_DEPTHS,
        allow_recession=False,
        use_rho_old=True,
    )

    print(f"Running SCAM (TACOT 3.0 YAML, prescribed-T, no ablation) ...")
    return run(
        stack, mat_cards, b_prime_tables,
        geom, surface_bc, back_bc, options,
        initial_T=T_INIT,
        verbose=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    run_pato()

    print("Loading PATO output ...")
    t_pato, T_pato = load_pato_T()
    _,      rho_pato = load_pato_rho()

    print("Loading FIAT reference ...")
    t_fiat, T_fiat = load_fiat()

    results = run_scam()

    t_scam = results.times_array()
    tc_arr = results.tc_array()   # (n_tc, n_steps)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    depths_mm = [d * 1000 for d in TC_DEPTHS]
    plot_idx  = [0, 2, 4]   # 1 mm, 4 mm, 12 mm
    colors    = ["tab:blue", "tab:orange", "tab:green"]
    fiat_col_map = {0: 1, 2: 3, 4: 5}   # SCAM tc idx → FIAT column

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        "SCAM vs PATO AblationTestCase_1.0 vs FIAT — TACOT 3.0 YAML material\n"
        "rho_char=220 kg/m³ | prescribed surface T | no surface ablation",
        fontsize=11,
    )

    # Panel 1 — Temperature history 1, 4, 12 mm (full run)
    ax = axes[0, 0]
    for idx, col in zip(plot_idx, colors):
        d_mm = depths_mm[idx]
        ax.plot(t_scam, tc_arr[idx],   color=col, lw=2,   label=f"SCAM {d_mm:.0f} mm")
        ax.plot(t_pato, T_pato[:, idx], color=col, lw=1.5, ls="--", label=f"PATO {d_mm:.0f} mm")
    ax.axvline(60, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Temperature [K]")
    ax.set_title("T history — 1, 4, 12 mm depth")
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    # Panel 2 — Heating phase (0–60 s) + FIAT
    ax = axes[0, 1]
    for idx, col in zip(plot_idx, colors):
        d_mm  = depths_mm[idx]
        ms    = t_scam <= 60
        mp    = t_pato <= 60
        ax.plot(t_scam[ms], tc_arr[idx][ms],       color=col, lw=2,   label=f"SCAM {d_mm:.0f} mm")
        ax.plot(t_pato[mp], T_pato[mp, idx],        color=col, lw=1.5, ls="--", label=f"PATO {d_mm:.0f} mm")
        fi = fiat_col_map.get(idx)
        if fi is not None:
            ax.plot(t_fiat, T_fiat[:, fi], color=col, lw=1.0, ls=":", marker="o",
                    ms=2, label=f"FIAT {d_mm:.0f} mm")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Temperature [K]")
    ax.set_title("Heating phase vs FIAT (0–60 s)")
    ax.legend(fontsize=6, ncol=3); ax.grid(True, alpha=0.3)

    # Panel 3 — T profiles at t≈30 s and t≈90 s
    ax = axes[0, 2]
    snap_30 = snap_90 = None
    for snap in results.snapshots:
        if abs(snap.time - 30) < 0.3 and snap_30 is None:
            snap_30 = snap
        if abs(snap.time - 90) < 0.3 and snap_90 is None:
            snap_90 = snap
    i30 = int(np.argmin(np.abs(t_pato - 30)))
    i90 = int(np.argmin(np.abs(t_pato - 90)))
    if snap_30 is not None:
        ax.plot(snap_30.mesh.y_nodes * 1000, snap_30.T, "b-", lw=2, label="SCAM t=30 s")
    if snap_90 is not None:
        ax.plot(snap_90.mesh.y_nodes * 1000, snap_90.T, "r-", lw=2, label="SCAM t=90 s")
    ax.plot(depths_mm, T_pato[i30, :], "b--o", ms=4, lw=1.5, label=f"PATO t={t_pato[i30]:.1f} s")
    ax.plot(depths_mm, T_pato[i90, :], "r--o", ms=4, lw=1.5, label=f"PATO t={t_pato[i90]:.1f} s")
    ax.set_xlabel("Depth from surface [mm]"); ax.set_ylabel("Temperature [K]")
    ax.set_title("T profiles at t=30 s and t=90 s")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 4 — SCAM − PATO temperature difference
    ax = axes[1, 0]
    for idx, col in zip(plot_idx, colors):
        d_mm = depths_mm[idx]
        T_pi = np.interp(t_scam, t_pato, T_pato[:, idx])
        ax.plot(t_scam, tc_arr[idx] - T_pi, color=col, lw=1.5, label=f"{d_mm:.0f} mm")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(60, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("SCAM − PATO  [K]")
    ax.set_title("Temperature difference SCAM − PATO")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 5 — Density evolution at 1 mm and 4 mm
    ax = axes[1, 1]
    rho_scam_tc = []
    for snap in results.snapshots:
        y = snap.mesh.y_nodes
        rho_scam_tc.append([
            float(snap.rho[int(np.argmin(np.abs(y - d)))])
            for d in [TC_DEPTHS[0], TC_DEPTHS[2]]
        ])
    rho_scam_tc = np.array(rho_scam_tc)
    for j, (idx, col, lbl) in enumerate(zip([0, 2], colors[:2], ["1 mm", "4 mm"])):
        ax.plot(t_scam, rho_scam_tc[:, j], color=col, lw=2,   label=f"SCAM {lbl}")
        ax.plot(t_pato, rho_pato[:, idx],  color=col, lw=1.5, ls="--", label=f"PATO {lbl}")
    ax.axhline(RHO_CHAR,   color="k", lw=0.8, ls=":",  label=f"ρ_char={RHO_CHAR} kg/m³")
    ax.axhline(RHO_VIRGIN, color="k", lw=0.8, ls="-.", label=f"ρ_virgin={RHO_VIRGIN} kg/m³")
    ax.axvline(60, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Density [kg/m³]")
    ax.set_title("Density evolution at 1 mm and 4 mm depth")
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    # Panel 6 — SCAM − FIAT (heating phase)
    ax = axes[1, 2]
    for idx, col in zip(plot_idx, colors):
        fi = fiat_col_map.get(idx)
        if fi is None:
            continue
        d_mm = depths_mm[idx]
        t_heat = t_scam[t_scam <= 60]
        T_s    = tc_arr[idx][t_scam <= 60]
        T_fi   = np.interp(t_heat, t_fiat, T_fiat[:, fi])
        ax.plot(t_heat, T_s - T_fi, color=col, lw=1.5, label=f"{d_mm:.0f} mm")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("SCAM − FIAT  [K]")
    ax.set_title("Temperature difference SCAM − FIAT (heating)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f"\nPlot saved → {OUT_PNG}")

    # --- summary stats ---
    print("\n--- SCAM vs PATO (max |ΔT| over full run) ---")
    for idx in range(len(TC_DEPTHS)):
        T_pi   = np.interp(t_scam, t_pato, T_pato[:, idx])
        dT_max = float(np.abs(tc_arr[idx] - T_pi).max())
        print(f"  depth {TC_DEPTHS[idx]*1000:5.1f} mm :  max |SCAM−PATO| = {dT_max:6.2f} K")

    print("\n--- SCAM vs FIAT (max |ΔT| over heating phase t≤60 s) ---")
    for tc_idx, fi, d in [(0, 1, 1), (2, 3, 4), (4, 5, 12)]:
        T_fi   = np.interp(t_scam[t_scam <= 60], t_fiat, T_fiat[:, fi])
        dT_max = float(np.abs(tc_arr[tc_idx][t_scam <= 60] - T_fi).max())
        print(f"  depth {d:5.0f} mm :  max |SCAM−FIAT|  = {dT_max:6.2f} K")


if __name__ == "__main__":
    main()
