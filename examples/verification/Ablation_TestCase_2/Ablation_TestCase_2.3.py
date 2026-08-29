#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""SCAM verification for Ablation Test Case 2.3 using TACOT v2.2.

Identical to TC 2.2 except the recovery enthalpy is h_r = 2.5×10⁷ J/kg
(higher aerothermal heating rate).  Full char ablation and recession are
enabled.  Blowing correction: Lees log formula with blown B′_g.

Reference: Amaryllis results from the Ablation Test Case series.

Verification status
-------------------
* Backend ``"refxls"`` (Amaryllis/XLS source table) gives the best agreement
  with Amaryllis for TC 2.3 and is the default.
* Live backends (``"mpp"``, ``"cantera"``) produce noticeably larger
  discrepancies at peak heating (~50 K above Amaryllis near t = 60 s) — this
  is attributed to differences in the thermodynamic database at high T (carbon
  sublimation regime).  Root cause not yet confirmed; to be investigated later.

Run from the repository root with::

    MPLBACKEND=Agg python3 \\
        examples/verification/Ablation_TestCase_2/Ablation_TestCase_2.3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
REF_DIR = HERE / "ReferenceData"
MATERIAL_FILE         = REPO / "scam/materials/ablative_organic/tacot_v2.2.yaml"
BPRIME_CONFIG_CANTERA = REPO / "scam/materials/ablative_organic/tacot_v2.2_bprime_config.yaml"
BPRIME_CONFIG_MPP     = REPO / "scam/materials/ablative_organic/tacot_v2.2_mpp_config.yaml"
BPRIME_REFXLS         = REPO / "scam/materials/ablative_organic/tacot_v2.2_bprime_from_refXLS.yaml"
OUT_PNG = HERE / "Ablation_TestCase_2.3.png"

sys.path.insert(0, str(REPO))

from scam.config.boundary import (  # noqa: E402
    BackBCConfig, BackBCType, SurfaceBCConfig, SurfaceBCType,
)
from scam.config.geometry import GeometryConfig, GeometryType  # noqa: E402
from scam.config.solver import SolverOptions  # noqa: E402
from scam.config.stack import LayerConfig, StackConfig  # noqa: E402
from scam.io.material_loader import load_material  # noqa: E402
from scam.solvers.material_response import run  # noqa: E402

T_INIT     = 300.0
T_END      = 120.0
THICKNESS  = 0.05
N_NODES    = 201
P_EDGE     = 101325.0
RHOUE_CH   = 0.3
H_RECOVERY = 2.5e7    # TC 2.3: 10× higher than TC 2.2's 1.5e6 J/kg
TC_DEPTHS  = np.array([0.001, 0.002, 0.004, 0.008, 0.012, 0.016, 0.024])
TC_LABELS  = ["1 mm", "2 mm", "4 mm", "8 mm", "12 mm", "16 mm", "24 mm"]
BACKEND    = "refxls"  # "refxls" | "table" | "cantera" | "mpp"

_BC_TIMES        = np.array([0.0, 0.1, 60.0, 60.1, T_END])
_RHOUECH         = np.array([0.003, RHOUE_CH, RHOUE_CH, 0.003, 0.003])
_CHEMISTRY_RHOUE = np.array([0.003, RHOUE_CH, RHOUE_CH, 0.0, 0.0])
_H_RECOVERY      = np.array([0.0, H_RECOVERY, H_RECOVERY, 0.0, 0.0])


def _interp_bc(t: float, values: np.ndarray) -> float:
    return float(np.interp(t, _BC_TIMES, values))


def _rho_ue_ch(t: float) -> float:
    return _interp_bc(t, _RHOUECH)


def _chemistry_rho_ue(t: float) -> float:
    return _interp_bc(t, _CHEMISTRY_RHOUE)


def _h_recovery(t: float) -> float:
    return _interp_bc(t, _H_RECOVERY)


def make_bprime_backend(bprime_table):
    """Return a B′ backend according to BACKEND.

    "refxls"  — XLS reference table (Amaryllis / Ablation Test Case series source).
    "mpp"     — Mutation++ warm-start grid (PATO's thermo database).
    "cantera" — live Cantera CNO equilibrium.
    "table"   — pre-computed Cantera B′ table (fast; omits surface_enthalpies).

    Falls back to "table" automatically if the selected backend is unavailable.
    """
    if BACKEND == "refxls":
        from scam.io.material_loader import BPrimeTable
        print("  Surface chemistry: XLS reference B′ table (tacot_v2.2_bprime_from_refXLS.yaml)")
        return BPrimeTable(str(BPRIME_REFXLS))

    if BACKEND == "mpp":
        try:
            from scam.physics.mpp_evaluator import MutationppEvaluator
            ev = MutationppEvaluator.from_config(str(BPRIME_CONFIG_MPP))
            ev.lookup(1500.0, P_EDGE, 0.05)
            print("  Surface chemistry: Mutation++ (warm-start grid)")
            return ev
        except Exception as exc:
            print(f"  WARNING: Mutation++ unavailable ({exc}); trying Cantera.")

    if BACKEND in ("mpp", "cantera"):
        try:
            from scam.physics.bprime_evaluator import BprimeEvaluator
            ev = BprimeEvaluator.from_config(str(BPRIME_CONFIG_CANTERA))
            ev.lookup(1500.0, P_EDGE, 0.05, None)
            print("  Surface chemistry: LIVE Cantera")
            return ev
        except Exception as exc:
            print(f"  WARNING: Cantera unavailable ({exc}); falling back to B′ table.")

    print("  Surface chemistry: pre-computed B′ table")
    return bprime_table


def _load_reference(name: str) -> np.ndarray:
    path = REF_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Reference data not found: {path}")
    data = np.genfromtxt(path, skip_header=1)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Could not parse reference data: {path}")
    return data


def load_reference_data() -> dict[str, np.ndarray]:
    return {
        "ama_energy": _load_reference("Amaryllis_Energy_TestCase_2.3.txt"),
        "ama_mass":   _load_reference("Amaryllis_Mass_TestCase_2.3.txt"),
    }


def run_scam():
    material, bprime_table = load_material(str(MATERIAL_FILE))
    bprime = make_bprime_backend(bprime_table)

    stack = StackConfig(layers=[
        LayerConfig(material.name, thickness=THICKNESS, n_nodes=N_NODES, n_subcells=4),
    ])
    geometry = GeometryConfig(geometry_type=GeometryType.SLAB)
    surface_bc = SurfaceBCConfig(
        bc_type=SurfaceBCType.ENERGY_BALANCE,
        rhoUeCH=_rho_ue_ch,
        h_r=_h_recovery,
        emissivity=-1.0,
        view_factor=1.0,
        T_rad_in=T_INIT,
        rho_e_u_e=_chemistry_rho_ue,
        C_M=1.0,
        p_e=P_EDGE,
        lambda_blowing=0.5,
        blowing_model="lees",   # PATO-consistent: log(1+Phi)/Phi + blown B'_g
    )
    back_bc = BackBCConfig(bc_type=BackBCType.ADIABATIC)
    options = SolverOptions(
        t_end=T_END,
        dt_init=0.01,
        dt_max=0.25,
        dt_min=1e-5,
        dt_max_dT=30.0,
        dt_max_drho_frac=0.05,
        output_dt=0.1,
        tc_positions=[],
        allow_recession=True,
        continuous_remap=True,
        use_rho_old=True,
    )

    print(f"Running SCAM Ablation Test Case 2.3 with TACOT v2.2 [backend={BACKEND}] ...")
    return run(
        stack,
        {material.name: material},
        {material.name: bprime},
        geometry,
        surface_bc,
        back_bc,
        options,
        initial_T=T_INIT,
        verbose=True,
    ), material


def temperatures_at_original_depths(results) -> np.ndarray:
    histories = []
    for snap in results.snapshots:
        values = np.full(TC_DEPTHS.shape, np.nan)
        inside = TC_DEPTHS >= snap.mesh.s_total
        values[inside] = np.interp(TC_DEPTHS[inside], snap.mesh.y_nodes, snap.T)
        histories.append(values)
    return np.asarray(histories).T


def _crossing_depth(y, rho, target):
    if rho[0] >= target:
        return float(y[0])
    if rho[-1] <= target:
        return float(y[-1])
    delta = rho - target
    crossings = np.flatnonzero(delta[:-1] * delta[1:] <= 0.0)
    if crossings.size == 0:
        return float("nan")
    i = int(crossings[0])
    if rho[i + 1] == rho[i]:
        return float(y[i])
    w = (target - rho[i]) / (rho[i + 1] - rho[i])
    return float(y[i] + w * (y[i + 1] - y[i]))


def pyrolysis_fronts(results, material):
    density_span = material.rho_virgin - material.rho_char
    rho_98 = material.rho_char + 0.98 * density_span
    rho_02 = material.rho_char + 0.02 * density_span
    virgin_front, char_front = [], []
    for snap in results.snapshots:
        virgin_front.append(_crossing_depth(snap.mesh.y_nodes, snap.rho, rho_98))
        char_front.append(_crossing_depth(snap.mesh.y_nodes, snap.rho, rho_02))
    return np.asarray(virgin_front), np.asarray(char_front)


def _valid_ref(values):
    out = values.astype(float, copy=True)
    out[out <= 1.0] = np.nan
    return out


def make_plot(results, material, ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t            = results.times_array()
    wall_t       = results.T_wall_array()
    tc_t         = temperatures_at_original_depths(results)
    recession    = results.s_array()
    mdot_g       = np.array([snap.m_dot_pyro for snap in results.snapshots])
    mdot_c       = np.array([snap.m_dot_char for snap in results.snapshots])
    virgin_front, char_front = pyrolysis_fronts(results, material)

    ama_e = ref["ama_energy"]
    ama_m = ref["ama_mass"]

    colors = plt.cm.tab10(np.linspace(0.0, 0.9, len(TC_DEPTHS)))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "Ablation Test Case 2.3 — TACOT v2.2  (h_r = 2.5×10⁷ J/kg)\n"
        "SCAM compared with Amaryllis",
        fontsize=14,
    )

    # --- temperatures ---
    ax = axes[0, 0]
    ax.plot(t, wall_t, color="black", lw=2.2, label="SCAM wall")
    ax.plot(ama_e[:, 0], ama_e[:, 1], color="black", ls=":", lw=1.8, label="Amaryllis wall")
    for i, (label, color) in enumerate(zip(TC_LABELS, colors)):
        ax.plot(t, tc_t[i], color=color, lw=1.5, label=f"SCAM {label}")
        ax.plot(ama_e[:, 0], _valid_ref(ama_e[:, i + 2]), color=color, ls=":", lw=1.0)
    ax.axvline(60.0, color="0.4", ls=":", lw=1)
    ax.set(title="Wall and thermocouple temperatures",
           xlabel="Time [s]", ylabel="Temperature [K]", xlim=(0, T_END))
    ax.legend(fontsize=7, ncol=3)
    ax.grid(alpha=0.25)

    # --- mass fluxes ---
    ax = axes[0, 1]
    ax.plot(t, mdot_g, color="tab:blue", lw=2, label=r"SCAM $\dot m_g$")
    ax.plot(t, mdot_c, color="tab:red",  lw=2, label=r"SCAM $\dot m_c$")
    ax.plot(ama_m[:, 0], ama_m[:, 1], color="tab:blue", ls=":", lw=1.8,
            label=r"Amaryllis $\dot m_g$")
    ax.plot(ama_m[:, 0], ama_m[:, 2], color="tab:red", ls=":", lw=1.8,
            label=r"Amaryllis $\dot m_c$")
    ax.axvline(60.0, color="0.4", ls=":", lw=1)
    ax.set(title="Surface mass fluxes", xlabel="Time [s]",
           ylabel=r"Mass flux [kg m$^{-2}$ s$^{-1}$]", xlim=(0, T_END))
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)

    # --- pyrolysis fronts ---
    ax = axes[1, 0]
    ax.plot(t, virgin_front * 1e3, color="tab:orange", lw=2, label="SCAM 98% virgin")
    ax.plot(t, char_front   * 1e3, color="tab:green",  lw=2, label="SCAM 2% virgin")
    ax.plot(ama_m[:, 0], ama_m[:, 3] * 1e3, color="tab:orange", ls=":", lw=1.8,
            label="Amaryllis 98% virgin")
    ax.plot(ama_m[:, 0], ama_m[:, 4] * 1e3, color="tab:green", ls=":", lw=1.8,
            label="Amaryllis 2% virgin")
    ax.set(title="Pyrolysis-zone fronts", xlabel="Time [s]",
           ylabel="Depth from original surface [mm]", xlim=(0, T_END))
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)

    # --- recession ---
    ax = axes[1, 1]
    ax.plot(t, recession * 1e3, color="black", lw=2.2, label="SCAM")
    ax.plot(ama_m[:, 0], ama_m[:, 5] * 1e3, color="black", ls=":", lw=1.8,
            label="Amaryllis")
    ax.axvline(60.0, color="0.4", ls=":", lw=1)
    ax.set(title="Surface recession", xlabel="Time [s]",
           ylabel="Recession [mm]", xlim=(0, T_END))
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    tmp = OUT_PNG.with_name(f".{OUT_PNG.stem}.tmp{OUT_PNG.suffix}")
    fig.savefig(tmp, dpi=160)
    tmp.replace(OUT_PNG)
    plt.close(fig)


def print_summary(results, ref):
    t         = results.times_array()
    wall      = results.T_wall_array()
    recession = results.s_array()
    ama_e     = ref["ama_energy"]
    ama_m     = ref["ama_mass"]

    print("\nComparison vs Amaryllis:")
    for target in (10.0, 30.0, 60.0, 90.0, 120.0):
        scam_t = float(np.interp(target, t, wall))
        ama_t  = float(np.interp(target, ama_e[:, 0], ama_e[:, 1]))
        print(f"  T_wall({target:5.1f} s): SCAM={scam_t:8.2f} K  "
              f"Amaryllis={ama_t:8.2f} K  Δ={scam_t - ama_t:+.2f} K")
    scam_rec = recession[-1] * 1e3
    ama_rec  = float(np.interp(T_END, ama_m[:, 0], ama_m[:, 5])) * 1e3
    print(f"  Recession at {T_END:.0f} s: SCAM={scam_rec:.3f} mm  "
          f"Amaryllis={ama_rec:.3f} mm  Δ={scam_rec - ama_rec:+.3f} mm")


def main():
    ref = load_reference_data()
    results, material = run_scam()
    make_plot(results, material, ref)
    print_summary(results, ref)
    print(f"\nPlot saved to {OUT_PNG}")


if __name__ == "__main__":
    main()
