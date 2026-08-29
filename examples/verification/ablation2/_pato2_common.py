# SPDX-License-Identifier: MIT
"""Shared helpers for PATO AblationTestCase_2.x comparison examples.

Provides:
  - PATO output file loaders (Ta_plot, Ta_surfacePatch, mass)
  - PATO case runner
  - Fresh/atomic PNG output helpers for comparison plots
  - Post-processing helper to evaluate T at fixed original-surface depths
    (needed because PATO probes are fixed in space while SCAM's y-axis is
    relative to the current receding surface)
"""
from __future__ import annotations

import subprocess
import os
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PATO_DIR  = Path.home() / "PATO-dev"
PATO_TUTS = PATO_DIR / "tutorials/1D"
PATO_REF  = PATO_DIR / "src/applications/utilities/tests/testsuites/tutorials/ref/1D"
PATO_MAT  = PATO_DIR / "data/Materials/Composites/TACOT"
REPO      = Path(__file__).resolve().parent.parent.parent.parent


def prepare_plot_output(path: str | Path) -> Path:
    """Remove an old comparison PNG before a long run starts.

    If the run fails or is interrupted, the missing output is less misleading
    than a stale image from an earlier physics/configuration state.
    """
    out = Path(path)
    if out.exists():
        out.unlink()
        print(f"Removed stale plot → {out}")
    return out


def save_plot_atomic(fig, path: str | Path, **savefig_kwargs) -> Path:
    """Write a plot through a temporary PNG, then atomically replace the target."""
    out = Path(path)
    tmp = out.with_name(f".{out.stem}.tmp{out.suffix}")
    fig.savefig(tmp, **savefig_kwargs)
    tmp.replace(out)
    return out

# ---------------------------------------------------------------------------
# PATO 2.x convective BC parameters (enthalpy-based formulation)
#
# PATO SEB uses the B' wall enthalpy while chemistry is active.  The 2.x table
# keeps a small rhoUeCH=0.3e-2 value when chemistryOn=0, but the Bprime
# temperature BC ignores rhoUeCH/h_r in that branch and uses hconv*(Tedge-T).
#
# h_gas(T_w) is read from the material's h_g_table (same gasProperties file
# used for pyrolysis gas energy transport in-depth).  At 300 K, h_gas = -7.09 MJ/kg, so the
# active heating pulse is much larger than a linear cp estimate.
#
# For the B' table blowing: B' = m_dot / rhoUeCH  (C_H = C_M assumed).
# Pass rho_e_u_e = rhoUeCH, C_M = 1.0 so the product equals rhoUeCH.
# ---------------------------------------------------------------------------
RHOUE_CH = 0.3          # PATO rhoUeCH = ρ_e·u_e·C_H [kg/m²/s]
H_R      = 1.5e6        # PATO recovery enthalpy [J/kg]
P_EDGE   = 101325.0     # edge pressure [Pa]
RHO_E_UE = RHOUE_CH    # product rho_e_u_e * C_M = rhoUeCH (when C_M = 1)
C_M      = 1.0          # mass transfer Stanton number

# Linearised enthalpy-based BC parameters — used only by chemistryOff.
# PATO applies rhoUeCH*(h_r - h_gas(T_w)) which linearises to
# alpha_enthalpy*(T_aw_enthalpy - T_w) with cp_ref ≈ 1150 J/kg/K.
_CP_REF       = 1150.0          # J/kg/K  (h_g slope in the 1000-1500 K range)
ALPHA_ENTHALPY = RHOUE_CH * _CP_REF   # ≈ 345 W/m²/K
T_AW_ENTHALPY  = H_R / _CP_REF        # ≈ 1304 K  (T where q_conv_enthalpy = 0)

# Timing shared by all 2.x cases
T_END   = 120.0   # s
T_INIT  = 300.0   # K

# TC probe depths from original surface [m] — match PATO plotDict
TC_DEPTHS = [0.001, 0.002, 0.004, 0.008, 0.012, 0.016, 0.024]

def _load_pato_hg() -> tuple[np.ndarray, np.ndarray]:
    """Load h_g(T) from PATO gasProperties (1 atm sub-table), or fall back."""
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
        if rows:
            rows.sort()
            return (np.array([r[0] for r in rows]),
                    np.array([r[1] for r in rows]))
    # Representative fallback subset
    T   = np.array([200., 300., 500., 700., 800., 900., 1000., 1100., 1200.,
                    1300., 1400., 1500., 1644., 2000., 3000., 4000.])
    hg  = np.array([-7.247e6, -7.090e6, -6.715e6, -6.005e6, -5.014e6, -3.335e6,
                    -2.170e6, -1.789e6, -1.199e6, -5.255e5, 1.299e5, 1.137e6,
                     2.625e6,  4.400e6,  1.100e7,  2.200e7])
    return T, hg


# PATO zeta: elemental composition of pyrolysis gas (all 3 reactions identical)
# zeta[C]=0.494996, zeta[H]=0.136912, zeta[O]=0.368092, zeta[N]=0.0
_PYRO_ZETA = np.array([0.494996, 0.136912, 0.368092, 0.0])   # [C, H, O, N]

_MECHANISMS_DIR = Path(__file__).resolve().parents[3] / "scam" / "mechanisms"
_CNO_MECH = str(_MECHANISMS_DIR / "cno_ablation.yaml")


def equilibrium_hg_table(T_grid=None, gas_pressure: float = 101325.0):
    """Pyrolysis-gas enthalpy h_g(T) from live Cantera equilibrium element conservation.

    Holds the TACOT pyrolysis-gas elemental composition (``_PYRO_ZETA``, by mass)
    fixed and re-equilibrates the species at each (T, p) — element-conserving by
    construction, with speciation (hence h_g, cp_g) following chemical
    equilibrium.  This is the live-Cantera analogue of PATO's ``Equilibrium``
    gas-properties model (used by ``AblationTestCase_2.x_equilibriumElementConservation``).

    Returns an ``[[T_K, h_g_J_kg], ...]`` table, or ``None`` if Cantera is
    unavailable (caller should then keep the tabulated h_g).  The reference
    coincides with PATO's: equil h_g(300 K) = −7.09 MJ/kg = PATO tabulated.
    """
    if T_grid is None:
        T_grid = np.arange(250.0, 4001.0, 50.0)
    try:
        from scam.chemistry.thermo_backend import CanteraBackend
        backend = CanteraBackend(
            gas_mechanism=_CNO_MECH,
            carbon_phase_file="graphite.yaml",
        )
    except Exception:
        return None
    # Atomic mole fractions from _PYRO_ZETA — no CO/CH4/H2 reconstruction needed.
    # Monatomic C, H, O species in cno_ablation.yaml carry the element ratios
    # exactly; equilibrate("TP") then conserves those elements.
    M = {"C": 12.011, "H": 1.008, "O": 15.999}
    nC = _PYRO_ZETA[0] / M["C"]
    nH = _PYRO_ZETA[1] / M["H"]
    nO = _PYRO_ZETA[2] / M["O"]
    tot = nC + nH + nO
    pyro_x = f"C:{nC/tot:.6f},H:{nH/tot:.6f},O:{nO/tot:.6f}"
    try:
        hg = np.array([
            backend.gas_enthalpy_equilibrium(max(float(T), 250.0), gas_pressure, pyro_x)
            for T in T_grid
        ])
    except Exception:
        return None
    return np.column_stack([T_grid, hg])


# ---------------------------------------------------------------------------
# PATO output file loaders
# ---------------------------------------------------------------------------

def load_pato_ta_plot(path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a PATO Ta_plot file.

    Returns (times [N], T_probes [N, n_probes]).
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                vals = [float(x) for x in line.split()]
                if len(vals) >= 2:
                    rows.append(vals)
            except ValueError:
                continue
    arr = np.array(rows)
    return arr[:, 0], arr[:, 1:]


def load_pato_ta_surfacepatch(path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a PATO Ta_surfacePatch file.

    Returns (times [N], T_patches [N, n_patches]).
    """
    return load_pato_ta_plot(path)


def load_pato_mass(path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse a PATO mass file.

    Returns (times, m_dot_g, m_dot_c, recession_m).
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                vals = [float(x) for x in line.split()]
                if len(vals) >= 5:
                    rows.append(vals[:6])
            except ValueError:
                continue
    arr = np.array(rows)
    # columns: t, m_dot_g, m_dot_c, frac_virgin, frac_char, recession
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 5]


# ---------------------------------------------------------------------------
# PATO runner
# ---------------------------------------------------------------------------

def run_pato(case_dir: Path, output_check: Path) -> None:
    """Run PATO for *case_dir* if *output_check* file does not yet exist."""
    if output_check.exists():
        print(f"PATO output found at {output_check}; skipping re-run.")
        return
    print(f"Running PATO {case_dir.name} ...")
    pato_bin = Path.home() / "PATO-dev/bin"
    cmd = (
        f"cd {case_dir} && "
        "cp -r origin.0 0 && "
        "blockMesh -region porousMat && "
        "PATOx"
    )
    env = {**os.environ, "PATH": f"{pato_bin}:{os.environ.get('PATH', '')}"}
    result = subprocess.run(["bash", "-c", cmd], env=env,
                            capture_output=True, text=True)
    if result.returncode != 0:
        print("PATO stderr:", result.stderr[-2000:])
        raise RuntimeError("PATO run failed")
    print("PATO run complete.")


# ---------------------------------------------------------------------------
# Post-processing: T at fixed original-surface depths
#
# PATO probes are fixed in absolute space; SCAM's y_nodes are relative to the
# current (receded) surface.  To compare, shift by s_total:
#   y_original = y_current + s_total
# ---------------------------------------------------------------------------

def T_at_original_depths(results, depths: list[float]) -> np.ndarray:
    """Evaluate T at fixed original-surface depths for every snapshot.

    Parameters
    ----------
    results : Results
        SCAM results object.
    depths : list[float]
        Depths from the original surface [m].

    Returns
    -------
    arr : np.ndarray, shape (n_depths, n_steps)
        Temperature [K].  NaN where the surface has receded past the probe.
    """
    n_d = len(depths)
    n_t = len(results.snapshots)
    arr = np.full((n_d, n_t), np.nan)
    for j, snap in enumerate(results.snapshots):
        # mesh.y_nodes is the absolute coordinate measured from the ORIGINAL
        # front face: the back face is fixed and the surface node sits at
        # y_nodes[0] == s_total as the front recedes into the material.  A probe
        # at original-surface depth d is therefore evaluated directly in
        # y_nodes, and is ablated once the surface has receded past it
        # (d < y_nodes[0] == s_total).  (Do NOT add s_total again — that would
        # double-count the recession and shift the profile inward.)
        y_orig = snap.mesh.y_nodes
        for i, d in enumerate(depths):
            if d < y_orig[0]:
                continue   # probe ablated away (surface has receded past it)
            arr[i, j] = float(np.interp(d, y_orig, snap.T,
                                        left=np.nan, right=snap.T[-1]))
    return arr
