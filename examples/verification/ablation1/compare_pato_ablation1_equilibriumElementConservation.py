# SPDX-License-Identifier: MIT
"""Compare SCAM vs PATO AblationTestCase_1.0_equilibriumElementConservation.

Same geometry as the base case (TACOT_v3, 5cm slab, 100 cells graded 0.1,
no surface ablation) but PATO's gas model is switched from Tabulated to
Equilibrium element conservation (CEA-based thermochemistry at each cell).
Surface BC and TACOT material are the same.

SCAM matches this by computing the pyrolysis-gas enthalpy h_g(T) from LIVE
Cantera equilibrium: the TACOT pyrolysis-gas elemental composition (zeta) is held
fixed and the species are re-equilibrated at each (T,p), enforcing element
conservation while the speciation follows chemical equilibrium (the live-Cantera
analogue of PATO's Equilibrium element-conservation gas model).  If Cantera
is unavailable it falls back to PATO's tabulated h_g.
The remaining differences between SCAM and base PATO also apply here.

Run:
    MPLBACKEND=Agg python3 examples/compare_pato_ablation1_equilibriumElementConservation.py
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

TACOT_YAML = REPO / "scam/materials/ablative_organic/tacot_v3.0.yaml"

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
THICKNESS  = 0.05       # m
T_INIT     = 300.0      # K
T_SURF     = 1644.0     # K
T_END      = 120.0      # s
TC_DEPTHS  = [0.001, 0.002, 0.004, 0.008, 0.012, 0.016, 0.024]
NODES = 501

# ---------------------------------------------------------------------------
# Fallback PATO gasProperties h_g (1 atm column), used when Cantera unavailable
# ---------------------------------------------------------------------------
_PATO_MAT = Path.home() / (
    "anaconda3/envs/pato/src/volume_pato/pato-3.1/data/Materials/Composites/TACOT"
)

def _load_pato_hg() -> tuple[np.ndarray, np.ndarray]:
    gas_file = _PATO_MAT / "gasProperties"
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

_HG_T, _HG_VAL = _load_pato_hg()


# ---------------------------------------------------------------------------
# Equilibrium element-conservation gas enthalpy via Cantera (PATO's Equilibrium
# gas model).  The TACOT pyrolysis-gas elemental composition (zeta, by mass) is
# held fixed and the species are re-equilibrated at each (T, p), so element
# conservation is enforced while the speciation — and hence h_g(T), cp_g(T) —
# follows chemical equilibrium.  This is the live-Cantera analogue of PATO's
# "Equilibrium element conservation" model, replacing the fixed gasProperties
# table the base case uses.  Falls back to PATO's tabulated h_g if Cantera
# is unavailable.  (The reference coincides with PATO's: equil h_g(300 K) =
# −7.09 MJ/kg = PATO tabulated, so no offset alignment is needed.)
_PYRO_ZETA = {"C": 0.494996, "H": 0.136912, "O": 0.368092}   # TACOT, mass fractions
_GAS_P = 101325.0
_CNO_MECH = str(Path(__file__).resolve().parents[3] / "scam" / "mechanisms" / "cno_ablation.yaml")


def _pyro_x_from_zeta(zeta: dict) -> dict:
    """Species mole-fraction dict with the given elemental mass composition.

    Any feasible speciation works since equilibrate('TP') conserves elements;
    a CO / CH4 / H2 basis reproduces the TACOT C/H/O ratio exactly.
    """
    M = {"C": 12.011, "H": 1.008, "O": 15.999}
    nC, nH, nO = zeta["C"] / M["C"], zeta["H"] / M["H"], zeta["O"] / M["O"]
    nCO = min(nO, nC)
    nCH4 = max(nC - nCO, 0.0)
    nH2 = max(nH - 4.0 * nCH4, 0.0) / 2.0
    raw = {"CO": nCO, "CH4": nCH4, "H2": nH2}
    tot = sum(raw.values())
    return {k: v / tot for k, v in raw.items() if v > 0}


def _equilibrium_hg_table(T_grid: np.ndarray) -> np.ndarray:
    import cantera as ct
    gas = ct.Solution(_CNO_MECH)
    X = _pyro_x_from_zeta(_PYRO_ZETA)
    hg = np.empty(len(T_grid))
    for i, T in enumerate(T_grid):
        gas.TPX = max(float(T), 250.0), _GAS_P, X
        gas.equilibrate("TP")
        hg[i] = gas.enthalpy_mass
    return hg


_HG_T_EQ = np.arange(250.0, 4001.0, 50.0)
try:
    _HG_VAL_EQ = _equilibrium_hg_table(_HG_T_EQ)
    _USING_CANTERA_GAS = True
    print("  Gas model: LIVE Cantera equilibrium element conservation "
          f"(re-speciated TACOT pyrolysis gas, {len(_HG_T_EQ)} T-points)")
except Exception as _e:  # noqa: BLE001 — Cantera optional
    _HG_T_EQ, _HG_VAL_EQ = _HG_T, _HG_VAL
    _USING_CANTERA_GAS = False
    print(f"  Cantera equilibrium gas unavailable ({type(_e).__name__}: {_e}); "
          "falling back to PATO tabulated h_g")


def make_tacot_material():
    mat, _ = load_material(str(TACOT_YAML))
    hg_tab = np.column_stack([_HG_T_EQ, _HG_VAL_EQ])   # Cantera equilibrium gas
    return dataclasses.replace(mat, h_g_table=hg_tab, h_g_abs_offset=None)


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
# Reference data (PATO run with equilibrium element conservation gas model)
# ---------------------------------------------------------------------------
_REF_BASE = (
    Path.home()
    / "PATO-dev/src/applications/utilities/tests/testsuites/tutorials/ref"
    / "1D/AblationTestCase_1.0_equilibriumElementConservation/output/porousMat/scalar"
)
# Tabulated PATO reference (base case) for comparison of gas-model effect
_BASE_CASE = (
    Path.home()
    / "PATO-dev/src/applications/utilities/tests/testsuites/tutorials/ref"
    / "1D/AblationTestCase_1.0/output/porousMat/scalar"
)

PATO_TA_EQ  = _REF_BASE / "Ta_plot"
PATO_TA_TAB = _BASE_CASE / "Ta_plot"


def _load_ta(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    with open(path) as f:
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


def run_scam():
    mat = make_tacot_material()
    stack = StackConfig(layers=[
        LayerConfig(mat.name, thickness=THICKNESS, n_nodes=NODES, n_subcells=4),
    ])
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
    gas_label = "Cantera equilibrium gas" if _USING_CANTERA_GAS else "tabulated gas (Cantera unavailable)"
    print(f"Running SCAM ({gas_label})...")
    return run(
        stack, {mat.name: mat}, {mat.name: None},
        GeometryConfig(),
        SurfaceBCConfig(bc_type=SurfaceBCType.PRESCRIBED_TEMP, T_prescribed=_T_surf),
        BackBCConfig(bc_type=BackBCType.ADIABATIC),
        options, initial_T=T_INIT, verbose=True,
    )


def main() -> None:
    print(f"Loading PATO equilibrium reference from {PATO_TA_EQ}")
    t_eq, T_eq = _load_ta(PATO_TA_EQ)
    tab_exists = PATO_TA_TAB.exists()
    if tab_exists:
        t_tab, T_tab = _load_ta(PATO_TA_TAB)
        print("  (base Tabulated reference also loaded for gas-model comparison)")

    results = run_scam()
    t_scam = results.times_array()
    tc_arr = results.tc_array()

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
        "SCAM vs PATO AblationTestCase_1.0_equilibriumElementConservation\n"
        + ("SCAM: Cantera equilibrium gas" if _USING_CANTERA_GAS else "SCAM: tabulated gas")
        + "  |  PATO: equilibrium element conservation",
        fontsize=10,
    )

    ax = axes[0, 0]
    for idx, col in zip(plot_idx, colors):
        d = depths_mm[idx]
        ax.plot(t_scam, tc_arr[idx], color=col, lw=2, label=f"SCAM {d:.0f} mm")
        ax.plot(t_eq, T_eq[:, idx], color=col, lw=1.5, ls="--", label=f"PATO-eq {d:.0f} mm")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("T history — 1, 4, 12 mm")
    ax.legend(fontsize=7, ncol=2)
    ax.axvline(60, color="k", lw=0.8, ls=":")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for idx, col in zip(plot_idx, colors):
        d = depths_mm[idx]
        ax.plot(t_scam, tc_arr[idx] - np.interp(t_scam, t_eq, T_eq[:, idx]),
                color=col, lw=1.5, label=f"{d:.0f} mm (vs eq)")
        if tab_exists:
            ax.plot(t_eq, T_eq[:, idx] - np.interp(t_eq, t_tab, T_tab[:, idx]),
                    color=col, lw=1.0, ls=":", label=f"{d:.0f} mm (eq−tab)")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(60, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("ΔT [K]")
    ax.set_title("SCAM−PATO(eq) [solid]  and  PATO(eq)−PATO(tab) [dotted]")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for idx, col in zip(plot_idx, colors):
        d = depths_mm[idx]
        T_ref = np.interp(t_scam, t_eq, T_eq[:, idx])
        ax.plot(t_scam, tc_arr[idx] - T_ref, color=col, lw=1.5, label=f"{d:.0f} mm")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(60, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("SCAM − PATO(eq) [K]")
    ax.set_title("SCAM − PATO equilibrium difference")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    snap_30 = snap_90 = None
    for snap in results.snapshots:
        if abs(snap.time - 30) < 0.3 and snap_30 is None:
            snap_30 = snap
        if abs(snap.time - 90) < 0.3 and snap_90 is None:
            snap_90 = snap
    pato_d_mm = [d * 1000 for d in TC_DEPTHS]
    i30 = int(np.argmin(np.abs(t_eq - 30)))
    i90 = int(np.argmin(np.abs(t_eq - 90)))
    if snap_30 is not None:
        ax.plot(snap_30.mesh.y_nodes * 1000, snap_30.T, "b-", lw=2, label="SCAM t=30 s")
    if snap_90 is not None:
        ax.plot(snap_90.mesh.y_nodes * 1000, snap_90.T, "r-", lw=2, label="SCAM t=90 s")
    ax.plot(pato_d_mm, T_eq[i30, :], "b--o", ms=4, lw=1.5, label=f"PATO-eq t={t_eq[i30]:.0f} s")
    ax.plot(pato_d_mm, T_eq[i90, :], "r--o", ms=4, lw=1.5, label=f"PATO-eq t={t_eq[i90]:.0f} s")
    ax.set_xlabel("Depth from surface [mm]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("T profiles at t=30 s and t=90 s")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(__file__).parent / "compare_pato_ablation1_equilibriumElementConservation.png"
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved → {out}")

    print("\n--- SCAM vs PATO-equilibrium (max |ΔT| over full run) ---")
    for idx in range(len(TC_DEPTHS)):
        T_ref = np.interp(t_scam, t_eq, T_eq[:, idx])
        dT = float(np.abs(tc_arr[idx] - T_ref).max())
        print(f"  depth {TC_DEPTHS[idx]*1000:5.1f} mm :  max |SCAM−PATO-eq| = {dT:6.2f} K")

    if tab_exists:
        print("\n--- PATO-equilibrium vs PATO-tabulated (gas-model effect) ---")
        for idx in range(len(TC_DEPTHS)):
            T_tab_i = np.interp(t_eq, t_tab, T_tab[:, idx])
            dT = float(np.abs(T_eq[:, idx] - T_tab_i).max())
            print(f"  depth {TC_DEPTHS[idx]*1000:5.1f} mm :  max |PATO-eq−PATO-tab| = {dT:6.2f} K")


if __name__ == "__main__":
    main()
