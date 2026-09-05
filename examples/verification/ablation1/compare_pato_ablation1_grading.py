# SPDX-License-Identifier: MIT
"""Compare SCAM vs PATO AblationTestCase_1.0_grading.

Same geometry as the base case but PATO uses a GradedPorous material model
with a third phase (coating, rhoI=1200 kg/m³) whose volume fraction varies
with depth from the surface:

    d=0 mm: epsI[3]=0.10  → rho_coating=120 kg/m³  (total ≈ 400 kg/m³)
    d=1 mm: epsI[3]=0.04  → rho_coating= 48 kg/m³  (total ≈ 328 kg/m³)
    d=4 mm: epsI[3]=0.005 → rho_coating=  6 kg/m³  (total ≈ 286 kg/m³)
   d=10 mm: epsI[3]=0.00  → rho_coating=  0 kg/m³  (total = 280 kg/m³)

SCAM does not support spatially graded materials.  Two SCAM configurations
are compared:
  (a) Uniform TACOT 3.0 — baseline, ignores grading
  (b) Graded multilayer stack — 10 thin TACOT layers with depth-varying
      initial densities that approximate the phase-3 grading profile

Material loaded from ``scam/materials/ablative_organic/tacot_v3.0.yaml``.

Run:
    MPLBACKEND=Agg python3 examples/verification/ablation1/compare_pato_ablation1_grading.py
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO))

from scam.config.boundary import BackBCConfig, BackBCType, SurfaceBCConfig, SurfaceBCType
from scam.config.geometry import GeometryConfig
from scam.config.solver import SolverOptions
from scam.config.stack import LayerConfig, StackConfig
from scam.io.material_loader import load_material
from scam.solvers.material_response import run

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PATO_MAT     = Path.home() / (
    "anaconda3/envs/pato/src/volume_pato/pato-3.1/data/Materials/Composites/TACOT"
)
TACOT_YAML = REPO / "scam/materials/ablative_organic/tacot_v3.0.yaml"

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
THICKNESS  = 0.05       # m
T_INIT     = 300.0      # K
T_SURF     = 1644.0     # K
T_END      = 120.0      # s
RHO_VIRGIN = 280.0      # kg/m³ — TACOT 3.0 base virgin density
RHO_CHAR   = 220.0      # kg/m³ — TACOT 3.0 base char density
TC_DEPTHS  = [0.001, 0.002, 0.004, 0.008, 0.012, 0.016, 0.024]
N_NODES_UNIFORM = 501   # nodes for the single-layer uniform run
N_NODES_GRADED  = 51    # nodes per 1 mm graded layer (10 layers)
N_NODES_BASE    = 151   # nodes for the remaining 40 mm uniform base layer


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
    T   = np.array([200., 300., 500., 700., 800., 900., 1000., 1100., 1200.,
                    1300., 1400., 1500., 1644., 2000., 3000., 4000.])
    hg  = np.array([-7.247e6, -7.090e6, -6.715e6, -6.005e6, -5.014e6, -3.335e6,
                    -2.170e6, -1.789e6, -1.199e6, -5.255e5,  1.299e5,  1.137e6,
                     2.625e6,  4.400e6,  1.100e7,  2.200e7])
    return T, hg


# Load base material once at module level for use in graded layer construction
def _load_base_mat():
    mat, _ = load_material(str(TACOT_YAML))
    hg_T, hg_val = _load_pato_hg()
    return dataclasses.replace(mat,
                               h_g_table=np.column_stack([hg_T, hg_val]),
                               h_g_abs_offset=None)


_BASE_MAT = _load_base_mat()

# ---------------------------------------------------------------------------
# Temperature BC: step at 0.1 s → 1644 K, step back at 60.1 s → 300 K
# ---------------------------------------------------------------------------
_T_BC_TABLE = np.array([
    [0.0,   T_INIT],
    [0.099, T_INIT],
    [0.1,   T_SURF],
    [60.0,  T_SURF],
    [60.1,  T_INIT],
    [T_END, T_INIT],
])


def _T_surf(t: float) -> float:
    return float(np.interp(t, _T_BC_TABLE[:, 0], _T_BC_TABLE[:, 1]))

# ---------------------------------------------------------------------------
# PATO reference data
# ---------------------------------------------------------------------------
_LOCAL_REF = (
    Path(__file__).resolve().parent
    / "pato_reference/AblationTestCase_1.0_grading/output/porousMat/scalar"
)
# Repo-bundled copy of the small PATO reference outputs (see README's "Note on
# PATO reference data") is used when present; falls back to a local PATO-dev
# checkout for regenerating/extending the bundled set.
PATO_REF = _LOCAL_REF if _LOCAL_REF.exists() else (
    Path.home()
    / "PATO-dev/src/applications/utilities/tests/testsuites/tutorials/ref"
    / "1D/AblationTestCase_1.0_grading/output/porousMat/scalar"
)
PATO_TA = PATO_REF / "Ta_plot"


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


# ---------------------------------------------------------------------------
# Phase-3 grading: depth-from-surface → epsI[3]
# rho_coating(d) = 1200 * epsI3(d)
# ---------------------------------------------------------------------------
_GRADING_D   = np.array([0.0, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.008, 0.010])
_GRADING_EPS = np.array([0.10, 0.06,  0.04,  0.02,  0.008, 0.005, 0.0005, 0.0])
_RHO_COATING_INTRINSIC = 1200.0  # kg/m³ (rhoI[3])


def graded_rho_virgin(depth_from_surface: float) -> float:
    """Total virgin density (base + coating) at given depth from surface."""
    eps3 = float(np.interp(depth_from_surface, _GRADING_D, _GRADING_EPS,
                           left=_GRADING_EPS[0], right=0.0))
    return RHO_VIRGIN + _RHO_COATING_INTRINSIC * eps3


def _make_graded_layer(label: str, d_center: float, thickness: float,
                       n_nodes: int) -> tuple[str, object]:
    """Build a MaterialCard with coating density matching depth d_center."""
    rho_v = graded_rho_virgin(d_center)
    rho_c = RHO_CHAR + (rho_v - RHO_VIRGIN)
    scale = rho_v / RHO_VIRGIN
    new_comps = [dataclasses.replace(c, rho_0=c.rho_0 * scale)
                 for c in _BASE_MAT.components]
    gamma = sum(c.rho_0 for c in new_comps) / rho_v
    card = dataclasses.replace(
        _BASE_MAT,
        name=label,
        rho_virgin=rho_v,
        rho_char=rho_c,
        gamma_resin=gamma,
        components=new_comps,
    )
    return label, card


def _make_graded_stack() -> tuple[StackConfig, dict]:
    """Multi-layer approximation of graded TACOT.

    The graded zone (0–10 mm) is split into 10 thin layers of 1 mm each.
    The remaining 40 mm is one uniform TACOT layer.
    """
    layers: list[LayerConfig] = []
    mat_cards: dict = {}

    n_graded_layers = 10
    graded_thickness = 0.001
    for i in range(n_graded_layers):
        d_center = (i + 0.5) * graded_thickness
        label = f"TACOT_graded_{i}"
        _, card = _make_graded_layer(label, d_center, graded_thickness, n_nodes=N_NODES_GRADED)
        layers.append(LayerConfig(label, thickness=graded_thickness, n_nodes=N_NODES_GRADED, n_subcells=4))
        mat_cards[label] = card

    uniform_name = _BASE_MAT.name + "_uniform"
    mat_cards[uniform_name] = dataclasses.replace(_BASE_MAT, name=uniform_name)
    layers.append(LayerConfig(uniform_name, thickness=0.040, n_nodes=N_NODES_BASE, n_subcells=4))

    return StackConfig(layers=layers), mat_cards


def _solver_options(tc_positions) -> SolverOptions:
    return SolverOptions(
        t_end=T_END,
        dt_init=0.01,
        dt_max=0.5,
        dt_min=1e-4,
        dt_max_dT=20.0,
        output_dt=0.5,
        tc_positions=tc_positions,
        allow_recession=False,
        use_rho_old=True,
    )


def run_scam_uniform() -> object:
    mat = _BASE_MAT
    stack = StackConfig(layers=[
        LayerConfig(mat.name, thickness=THICKNESS, n_nodes=N_NODES_UNIFORM, n_subcells=4),
    ])
    print("Running SCAM (uniform TACOT 3.0, no grading)...")
    return run(
        stack, {mat.name: mat}, {mat.name: None},
        GeometryConfig(),
        SurfaceBCConfig(bc_type=SurfaceBCType.PRESCRIBED_TEMP, T_prescribed=_T_surf),
        BackBCConfig(bc_type=BackBCType.ADIABATIC),
        _solver_options(TC_DEPTHS), initial_T=T_INIT, verbose=False,
    )


def run_scam_graded() -> object:
    stack, mat_cards = _make_graded_stack()
    b_prime = {k: None for k in mat_cards}
    print("Running SCAM (graded multilayer approximation)...")
    return run(
        stack, mat_cards, b_prime,
        GeometryConfig(),
        SurfaceBCConfig(bc_type=SurfaceBCType.PRESCRIBED_TEMP, T_prescribed=_T_surf),
        BackBCConfig(bc_type=BackBCType.ADIABATIC),
        _solver_options(TC_DEPTHS), initial_T=T_INIT, verbose=False,
    )


def main() -> None:
    print(f"Loading PATO grading reference from {PATO_TA}")
    t_pato, T_pato = load_pato_T()

    res_u = run_scam_uniform()
    res_g = run_scam_graded()

    t_u = res_u.times_array()
    tc_u = res_u.tc_array()
    t_g = res_g.times_array()
    tc_g = res_g.tc_array()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    depths_mm = [d * 1000 for d in TC_DEPTHS]
    plot_idx = [0, 2, 4]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        "SCAM vs PATO AblationTestCase_1.0_grading\n"
        "PATO: graded 3-phase TACOT  |  SCAM-u: uniform  |  SCAM-g: multilayer approx.",
        fontsize=10,
    )

    ax = axes[0, 0]
    for idx, col in zip(plot_idx, colors):
        d = depths_mm[idx]
        ax.plot(t_u, tc_u[idx], color=col, lw=2, label=f"SCAM-u {d:.0f} mm")
        ax.plot(t_g, tc_g[idx], color=col, lw=1.8, ls="-.", label=f"SCAM-g {d:.0f} mm")
        ax.plot(t_pato, T_pato[:, idx], color=col, lw=1.5, ls="--", label=f"PATO {d:.0f} mm")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("T history — 1, 4, 12 mm")
    ax.legend(fontsize=6, ncol=3)
    ax.axvline(60, color="k", lw=0.8, ls=":")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    d_prof = np.linspace(0, 0.01, 200)
    rho_grad = np.array([graded_rho_virgin(d) for d in d_prof])
    ax.plot(d_prof * 1000, rho_grad, "k-", lw=2, label="PATO rho_virgin(d)")
    ax.axhline(RHO_VIRGIN, color="gray", lw=1, ls="--", label=f"Uniform = {RHO_VIRGIN} kg/m³")
    for i in range(10):
        d_c = (i + 0.5) * 0.001
        ax.axvline(d_c * 1000, color="tab:blue", lw=0.5, alpha=0.4)
    ax.set_xlabel("Depth from surface [mm]")
    ax.set_ylabel("Virgin density [kg/m³]")
    ax.set_title("Grading profile (phase-3 coating density)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 12)

    ax = axes[1, 0]
    for idx, col in zip(plot_idx, colors):
        d = depths_mm[idx]
        diff_u = tc_u[idx] - np.interp(t_u, t_pato, T_pato[:, idx])
        diff_g = tc_g[idx] - np.interp(t_g, t_pato, T_pato[:, idx])
        ax.plot(t_u, diff_u, color=col, lw=1.5, label=f"uniform {d:.0f} mm")
        ax.plot(t_g, diff_g, color=col, lw=1.2, ls="-.", label=f"graded {d:.0f} mm")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(60, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("SCAM − PATO [K]")
    ax.set_title("Difference: uniform [solid] and graded-approx [dash-dot]")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    snap_30_u = snap_30_g = None
    for snap in res_u.snapshots:
        if abs(snap.time - 30) < 0.3 and snap_30_u is None:
            snap_30_u = snap
    for snap in res_g.snapshots:
        if abs(snap.time - 30) < 0.3 and snap_30_g is None:
            snap_30_g = snap
    i30 = int(np.argmin(np.abs(t_pato - 30)))
    if snap_30_u is not None:
        ax.plot(snap_30_u.mesh.y_nodes * 1000, snap_30_u.T, "b-", lw=2, label="SCAM-u t=30 s")
    if snap_30_g is not None:
        ax.plot(snap_30_g.mesh.y_nodes * 1000, snap_30_g.T, "b-.", lw=1.8, label="SCAM-g t=30 s")
    pato_d_mm = [d * 1000 for d in TC_DEPTHS]
    ax.plot(pato_d_mm, T_pato[i30, :], "b--o", ms=4, lw=1.5, label=f"PATO t={t_pato[i30]:.0f} s")
    ax.set_xlabel("Depth from surface [mm]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("T profile at t=30 s")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(__file__).parent / "compare_pato_ablation1_grading.png"
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved → {out}")

    print("\n--- SCAM vs PATO-grading (max |ΔT| over full run) ---")
    print(f"  {'depth':>8}  {'uniform':>12}  {'graded-approx':>14}")
    for idx in range(len(TC_DEPTHS)):
        dT_u = float(np.abs(tc_u[idx] - np.interp(t_u, t_pato, T_pato[:, idx])).max())
        dT_g = float(np.abs(tc_g[idx] - np.interp(t_g, t_pato, T_pato[:, idx])).max())
        print(f"  {TC_DEPTHS[idx]*1000:5.1f} mm     {dT_u:8.2f} K     {dT_g:10.2f} K")


if __name__ == "__main__":
    main()
