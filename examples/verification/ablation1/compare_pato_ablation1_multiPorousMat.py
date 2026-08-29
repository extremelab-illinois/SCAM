# SPDX-License-Identifier: MIT
"""Compare SCAM vs PATO AblationTestCase_1.0_multiPorousMat.

Two-layer composite slab (prescribed surface temperature, no surface ablation):

    Surface (top) → T_prescribed (same ramp as base case)
    ┌──────────────────────────────────────┐ y=0.02 m
    │  porousMat1 : TACOT porous (1 cm)   │  — inner layer (next to surface)
    │  LinearArrhenius matrix pyrolysis   │
    ├──────────────────────────────────────┤ y=0.01 m
    │  porousMat2 : Cork (1 cm)            │  — outer layer (back)
    │  k=0.0805 W/m/K, cp=2625 J/kg/K    │
    │  ρ=290 kg/m³  (no pyrolysis)        │
    └──────────────────────────────────────┘ y=0.00 m
    Back (bottom) → adiabatic

PATO reference probes:
    porousMat1 probe at y=0.015 m → 5 mm from surface
    porousMat2 probe at y=0.005 m → 15 mm from surface

SCAM models this as a two-layer StackConfig: pyrolyzing TACOT over inert cork.

Root-cause note — porousMat1 is NOT inert:
    PATO's porousMat1 uses `PyrolysisType LinearArrhenius` with the TACOT_v3
    matrix reactions.  Earlier SCAM versions treated this layer as inert, which
    made the 5 mm probe run 70–90 K too hot after the initial transient.  SCAM
    now enables the same reduced TACOT decomposition model used by the other
    TACOT examples (two resin components: 0.25 and 0.19+0.06 of the matrix
    density).  The density evolves from 280 kg/m³ toward the physical char
    density 220 kg/m³, and the material properties blend from virgin to char.

PATO's property blending — "virginOrChar char" root cause:
    PATO's porousMat1Properties sets `virginOrChar char`, which forces tau=0
    everywhere in the blending formula:

        field = field_c + (field_v - field_c) * tau

    tau=0 means ALL thermal properties (k, cp, eps_g, h_s) are frozen at their
    char values throughout the run, regardless of the actual current density.
    In particular:
      • cp = cp_char (787 J/kg/K at 300 K) rather than cp_virgin (984 J/kg/K)
        → 25 % lower thermal mass → 25 % faster heating
      • hs = 0 in the pyrolysis energy (h_char − h_char = 0); only hp = −4 MJ/kg
        contributes (same magnitude as SCAM's total h_bar ≈ −4 MJ/kg at 300 K)

    This is physically inconsistent: the material starts as virgin (ρ = 280 kg/m³)
    but uses char thermal properties.  SCAM correctly blends cp from virgin to char
    based on the current density.  The `virginOrChar char` setting in PATO was
    intended for a material already fully charred at initialisation, not one that
    decomposes during the run.

Implementation in this script:
    `make_tacot_char_material()` replicates PATO's blending by overriding virgin
    tables with char tables before building the MaterialCard:
        k_virgin  ← k_char,   cp_virgin ← cp_char,   eps_g_virgin ← eps_g_char,
        h_virgin_sensible ← h_char_sensible  (→ h_bar_sensible = 0; only hp = −4 MJ/kg)
    This reduces the 5 mm peak gap from 73 K down to ~48 K.  The residual ~48 K
    is undiagnosed but the leading suspect is the decomposition energy: with
    h_bar_sensible=0, SCAM's Q_vol carries h_bar_chemical = h_bar_absolute ≈
    −4429 kJ/kg, whereas PATO with hs=0 applies only hp = −4000 kJ/kg.  Closing
    this further would require also zeroing h_bar_absolute (overriding both
    absolute enthalpy tables) while injecting hp directly — impractical and still
    physically wrong.
    The physical SCAM configuration (virgin→char blending) is documented separately
    by the 73 K vs 48 K numbers above.

Gas energy storage:
    PATO's PyrolysisEnergyModel includes fvc::ddt(epsgRhogEg) explicitly.  SCAM
    replicates this with gas_storage_implicit=False.  The correction is ~4 J/kg/K
    at 300 K (negligible; gas density is small at low T).

Density note:
    PATO initialises the TACOT layer from constantProperties
    (epsI[1]=0.1, epsI[2]=0.1) → rho_s = 1600×0.1 + 1200×0.1 = 280 kg/m³.
    During LinearArrhenius pyrolysis, the matrix residue leaves the layer near
    the physical char bulk density 220 kg/m³.

Pressure-driven Darcy:
    PATO solves a full 1D pressure equation (MassType DarcyLaw).  The pressure
    perturbations are tiny: Δp < 1 Pa = 0.001% of atmospheric (TACOT
    K=2×10⁻¹¹ m²).  The resulting gas energy flux is ~22 W/m² vs 62,640 W/m²
    conduction = 0.036%.  Implemented in physics/pressure_darcy.py; effect < 0.1 K.

Run:
    MPLBACKEND=Agg python3 examples/ablation1/compare_pato_ablation1_multiPorousMat.py
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
from scam.config.material import MaterialCard
from scam.config.solver import SolverOptions
from scam.config.stack import LayerConfig, StackConfig
from scam.io.material_loader import load_material
from scam.solvers.material_response import run

TACOT_YAML = REPO / "scam/materials/ablative_organic/tacot_v3.0.yaml"

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
T_INIT = 300.0    # K
T_SURF = 1644.0   # K
T_END  = 120.0    # s

_T_BC_TABLE = np.array([
    [0.0,   T_INIT],
    [0.1,   T_SURF],   # linear ramp 0→0.1 s matching PATO uniformValue table
    [60.0,  T_SURF],
    [60.1,  T_INIT],
    [T_END, T_INIT],
])


def _T_surf(t: float) -> float:
    return float(np.interp(t, _T_BC_TABLE[:, 0], _T_BC_TABLE[:, 1]))


# ---------------------------------------------------------------------------
# PATO reference data
# ---------------------------------------------------------------------------
_REF = (
    Path.home()
    / "PATO-dev/src/applications/utilities/tests/testsuites/tutorials/ref"
    / "1D/AblationTestCase_1.0_multiPorousMat/output"
)
PATO_TA_MAT1   = _REF / "porousMat1/scalar/Ta_plot"          # TACOT porous layer (inner)
PATO_TA_MAT2   = _REF / "porousMat2/scalar/Ta_plot"          # Cork layer (outer/back)
PATO_TA_SURF   = _REF / "porousMat1/scalar/Ta_surfacePatch"  # Surface patch T


def _load_ta(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a PATO Ta_plot with 1 probe column → (times, T[:,0])."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                vals = [float(x) for x in line.split()]
                if len(vals) >= 2:
                    rows.append(vals[:2])
            except ValueError:
                continue
    arr = np.array(rows)
    return arr[:, 0], arr[:, 1]


# ---------------------------------------------------------------------------
# Material definitions
# ---------------------------------------------------------------------------

_TACOT_PERM = 2e-11  # m² — char permeability K_c from PATO constantProperties

# Cork: constant k and cp (no pyrolysis, no T-dependence in PATO data)
_CORK_K   = 0.0805   # W/m/K
_CORK_CP  = 2625.0   # J/kg/K
_CORK_RHO = 290.0    # kg/m³
_EPS_G_CORK = 0.8    # Cork porosity (PATO Cork_noPyrolysis constantProperties eps_g_v)

# Gas properties at 1 atm (p=101325 Pa) — same table for both materials in PATO.
# Both layers use EnergyType Pyrolysis, which adds a gas energy storage term:
#   d(eps_g * rho_g * h_g)/dt = eps_g * rho_g(T) * (cp_g(T) - h_g(T)/T) * dT/dt
# At uniform pressure (neglecting pressure-driven gas flow), absorbing this into an effective cp makes
# SCAM match PATO's energy equation.  The correction is ~10% at 300K, <1% above 1000K.
_GAS_M   = 0.022     # kg/mol (PATO gasProperties M column)
_GAS_R   = 8.314     # J/(mol·K)
_GAS_P   = 101325.0  # Pa
_GAS_T   = np.array([200., 225., 250., 275., 300., 325., 350., 375., 400., 425.,
                      450., 475., 500., 525., 550., 575., 600., 625., 650., 700.,
                      800., 900., 1000., 1100., 1200., 1300., 1400., 1500., 1644.,
                      2000., 3000., 4000.])
_GAS_HG  = np.array([-7.247e6, -7.208e6, -7.170e6, -7.130e6, -7.090e6, -7.049e6,
                      -7.006e6, -6.963e6, -6.917e6, -6.870e6, -6.821e6, -6.770e6,
                      -6.715e6, -6.657e6, -6.595e6, -6.527e6, -6.451e6, -6.365e6,
                      -6.270e6, -6.005e6, -5.014e6, -3.335e6, -2.170e6, -1.789e6,
                      -1.199e6, -5.255e5,  1.299e5,  1.137e6,  2.625e6,  4.400e6,
                       1.100e7,  2.200e7])


def make_tacot_char_material() -> MaterialCard:
    """Build the TACOT layer with PATO's 'virginOrChar char' property blending.

    PHYSICAL NOTE: This is intentionally physically incorrect — PATO's
    porousMat1Properties sets `virginOrChar char`, which forces tau=0 in
    its blending formula `field = field_c + (field_v - field_c)*tau`.
    The result is that ALL thermal properties (k, cp, eps_g) are frozen at
    char values and the sensible h_bar contribution (hs = h_v - h_c) is zero.
    The material starts at virgin density (280 kg/m³) but heats as if already
    fully charred.  SCAM is physically correct; this override is applied here
    only to match the PATO reference data for verification purposes.
    """
    mat, _ = load_material(str(TACOT_YAML))
    # --- PATO virginOrChar char: freeze all properties at char values (tau=0) ---
    mat = dataclasses.replace(
        mat,
        k_virgin_table=mat.k_char_table,        # PATO: k = k_c (no virgin blend)
        cp_virgin_table=mat.cp_char_table,       # PATO: cp = cp_c (25% lower thermal mass)
        eps_g_virgin=mat.eps_g_char,             # PATO: eps_g = eps_g_c
        h_virgin_sensible=mat.h_char_sensible,   # PATO: hs = h_c - h_c = 0; only hp contributes
    )
    return dataclasses.replace(
        mat,
        name="TACOT_char",
        h_g_table=np.column_stack([_GAS_T, _GAS_HG]),
        h_g_abs_offset=None,
        gas_storage_implicit=False,   # PATO uses fvc::ddt(epsgRhogEg) explicitly
        permeability=_TACOT_PERM,
    )


def make_cork_material() -> MaterialCard:
    T_pts  = np.array([200.0, 300.0, 500.0, 800.0, 1200.0, 2000.0, 4000.0])
    k_tab  = np.column_stack([T_pts, np.full(len(T_pts), _CORK_K)])
    cp_tab = np.column_stack([T_pts, np.full(len(T_pts), _CORK_CP)])
    hg_tab = np.column_stack([_GAS_T, _GAS_HG])
    return MaterialCard(
        name="Cork",
        rho_virgin=_CORK_RHO,
        rho_char=_CORK_RHO,
        gamma_resin=0.0,
        components=[],
        k_virgin_table=k_tab,
        k_char_table=k_tab,
        cp_virgin_table=cp_tab,
        cp_char_table=cp_tab,
        h_g_table=hg_tab,
        eps_g_virgin=_EPS_G_CORK,
        eps_g_char=_EPS_G_CORK,
        gas_molar_mass=_GAS_M,
        gas_pressure=_GAS_P,
        permeability=_TACOT_PERM,
        gas_storage_implicit=False,   # PATO uses fvc::ddt(epsgRhogEg) explicitly
        emissivity=0.9,
        decomposing=False,
    )


# ---------------------------------------------------------------------------
# TC depths: 5 mm (in TACOT porous layer) and 15 mm (in Cork layer)
# PATO probes: porousMat1 at y=0.015m, porousMat2 at y=0.005m
# Depths from surface = 0.02-0.015=0.005m and 0.02-0.005=0.015m
# ---------------------------------------------------------------------------
TC_DEPTHS_MULTI = [0.005, 0.015]
THICKNESS_MAT1 = 0.010   # 1 cm TACOT char (inner, next to surface)
THICKNESS_MAT2 = 0.010   # 1 cm Cork (back)
N_NODES_MAT1   = 600
N_NODES_MAT2   = 600


def run_scam() -> object:
    mat1 = make_tacot_char_material()
    mat2 = make_cork_material()
    stack = StackConfig(layers=[
        LayerConfig("TACOT_char", thickness=THICKNESS_MAT1, n_nodes=N_NODES_MAT1, n_subcells=4),
        LayerConfig("Cork",       thickness=THICKNESS_MAT2, n_nodes=N_NODES_MAT2, n_subcells=1),
    ])
    options = SolverOptions(
        t_end=T_END,
        dt_init=0.01,
        dt_max=0.5,
        dt_min=1e-4,
        dt_max_dT=20.0,
        dt_max_drho_frac=0.05,
        output_dt=0.5,
        tc_positions=TC_DEPTHS_MULTI,
        allow_recession=False,
        use_rho_old=True,       # exact d(ρh)/dt storage (default); False → PATO ρ·dh/dt approximation
    )
    print("Running SCAM (pyrolyzing TACOT + Cork, two-layer conduction)...")
    return run(
        stack,
        {"TACOT_char": mat1, "Cork": mat2},
        {"TACOT_char": None, "Cork": None},
        GeometryConfig(),
        SurfaceBCConfig(bc_type=SurfaceBCType.PRESCRIBED_TEMP, T_prescribed=_T_surf),
        BackBCConfig(bc_type=BackBCType.ADIABATIC),
        options, initial_T=T_INIT, verbose=True,
    )


def main() -> None:
    print(f"Loading PATO multiPorousMat references...")
    t1_p, T1_p = _load_ta(PATO_TA_MAT1)   # 5 mm from surface (TACOT char)
    t2_p, T2_p = _load_ta(PATO_TA_MAT2)   # 15 mm from surface (Cork)
    try:
        t_surf_p, T_surf_p = _load_ta(PATO_TA_SURF)
    except FileNotFoundError:
        t_surf_p = np.array([0.0, T_END])
        T_surf_p = np.array([T_INIT, T_INIT])

    results = run_scam()
    t_scam  = results.times_array()
    tc_arr  = results.tc_array()           # [0]=5 mm, [1]=15 mm
    T_wall  = results.T_wall_array()       # surface temperature

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        "SCAM vs PATO AblationTestCase_1.0_multiPorousMat\n"
        "Two-layer conduction: TACOT pyrolyzing porous layer (0–1 cm) + Cork (1–2 cm)",
        fontsize=10,
    )

    labels   = ["5 mm (TACOT porous)", "15 mm (Cork)"]
    colors   = ["tab:blue", "tab:orange"]
    pato_t   = [t1_p, t2_p]
    pato_T   = [T1_p, T2_p]

    # Panel 1: full temperature history including surface T
    ax = axes[0, 0]
    ax.plot(t_scam, T_wall, color="tab:red", lw=2, label="SCAM surface (prescribed)")
    ax.plot(t_surf_p, T_surf_p, color="tab:red", lw=1.5, ls="--", label="PATO surface")
    for j, (col, lab) in enumerate(zip(colors, labels)):
        ax.plot(t_scam, tc_arr[j], color=col, lw=2, label=f"SCAM {lab}")
        ax.plot(pato_t[j], pato_T[j], color=col, lw=1.5, ls="--",
                label=f"PATO {lab}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("T history — surface, 5 mm and 15 mm depth")
    ax.legend(fontsize=7, ncol=2)
    ax.axvline(60, color="k", lw=0.8, ls=":")
    ax.grid(True, alpha=0.3)

    # Panel 2: temperature difference (in-depth probes only)
    ax = axes[0, 1]
    for j, (col, lab) in enumerate(zip(colors, labels)):
        T_ref = np.interp(t_scam, pato_t[j], pato_T[j])
        ax.plot(t_scam, tc_arr[j] - T_ref, color=col, lw=1.5, label=lab)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(60, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("SCAM − PATO [K]")
    ax.set_title("Temperature difference SCAM − PATO")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: early-time zoom (0–30 s) to visualise onset of 5 mm divergence
    ax = axes[1, 0]
    ax.plot(t_scam, T_wall, color="tab:red", lw=1.5, label="SCAM surface")
    ax.plot(t_surf_p, T_surf_p, color="tab:red", lw=1, ls="--", label="PATO surface")
    for j, (col, lab) in enumerate(zip(colors, labels)):
        ax.plot(t_scam, tc_arr[j], color=col, lw=2, label=f"SCAM {lab}")
        ax.plot(pato_t[j], pato_T[j], color=col, lw=1.5, ls="--",
                label=f"PATO {lab}")
    ax.set_xlim(0, 30)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("Early-time zoom (0–30 s) — onset of divergence at 5 mm")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 4: T profile snapshots
    ax = axes[1, 1]
    snaps = {}
    for snap in results.snapshots:
        for t_target in [30, 60, 90]:
            if abs(snap.time - t_target) < 0.3 and t_target not in snaps:
                snaps[t_target] = snap
    plot_colors_prof = {"30": "tab:blue", "60": "tab:orange", "90": "tab:green"}
    for t_target, snap in snaps.items():
        col = plot_colors_prof[str(t_target)]
        y_mm = snap.mesh.y_nodes * 1000
        ax.plot(y_mm, snap.T, color=col, lw=2, label=f"SCAM t={t_target} s")
    ax.axvline(THICKNESS_MAT1 * 1000, color="k", lw=0.8, ls=":",
               label=f"Layer interface ({THICKNESS_MAT1*1000:.0f} mm)")
    ax.set_xlabel("Depth from surface [mm]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("T profiles at t=30, 60, 90 s")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(__file__).parent / "compare_pato_ablation1_multiPorousMat.png"
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved → {out}")

    print("\n--- SCAM vs PATO multiPorousMat (max |ΔT| over full run) ---")
    for j, (lab, tp, Tp) in enumerate(zip(labels, pato_t, pato_T)):
        T_ref = np.interp(t_scam, tp, Tp)
        dT = float(np.abs(tc_arr[j] - T_ref).max())
        print(f"  {lab:<26}:  max |SCAM−PATO| = {dT:6.2f} K")


if __name__ == "__main__":
    main()
