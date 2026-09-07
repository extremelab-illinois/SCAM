# SPDX-License-Identifier: MIT
"""Shared machinery for the Amar (2006) Chapter 8 verification cases.

The §8.8 carbon-phenolic case (`compare_amar_cp.py`) and the §8.7
carbon-carbon case run the same ballistic-reentry trajectory (Figs 8.14-8.15)
and the same Amar et al. (2008) Stanton-number corrections.  That common code
lives here so neither case depends on the other -- in particular so §8.8 runs
standalone.  (§8.7 is still unresolved and its script is not part of the public
distribution; §8.8 does not need it.)

Reference:
    Amar, Adam J. (2006), "Modeling of One-Dimensional Ablation with Porous Flow
    Using Finite Control Volume Procedure", PhD thesis, Chapter 8.
    Amar, A., Blackwell, B., Edwards, J. (2008), Eqs. (37)-(42).

The underlying Stanton correlations themselves are part of the SCAM package
(`scam.physics.stanton`); only the trajectory-specific glue is here.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scam.physics.stanton import (
    eckert_reference_enthalpy,
    laminar_wall_factor,
    turbulent_eckert_wall_factor,
)

# ---------------------------------------------------------------------------
# Trajectory constants shared by §8.7 and §8.8 (Figs 8.14-8.15)
# ---------------------------------------------------------------------------
T_INIT = 297.04          # [K]  ~534°R initial/cold-wall temperature
T_TRANSITION = 23.5      # [s]  laminar -> turbulent transition

# Unit conversions
BTU_LBM_TO_J_KG    = 2326.0     # h_r  [Btu/lbm -> J/kg]
LBM_FT2S_TO_KG_M2S = 4.8824     # HTC  [lbm/ft²·s -> kg/m²·s]
ATM_TO_PA          = 101325.0   # p_e  [atm -> Pa]
IN_S_TO_M_S        = 0.0254     # ṡ    [in/s -> m/s]
R_TO_K             = 5.0 / 9.0  # T    [°R -> K]


def _load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load two-column CSV, sort by time, return (t, v)."""
    data = np.loadtxt(str(path), delimiter=",")
    order = np.argsort(data[:, 0])
    return data[order, 0], data[order, 1]


def smooth_laminar_htc(
    t: np.ndarray, htc: np.ndarray, t_split: float = T_TRANSITION, window: int = 9
) -> np.ndarray:
    """Smooth the digitization staircase in the laminar portion of the HTC curve.

    In the laminar phase (t < t_split) the true HTC is small and sits on the plot
    axis, so the digitized Fig 8.15 came out as a quantized staircase.  Combined
    with the large laminar recovery enthalpy (q_conv ~ HTC*h_r), each step injects
    a visible kink into the surface-temperature rate.  A centered moving average
    over the laminar segment removes the staircase; the turbulent segment and the
    sharp laminar->turbulent jump at t_split are left untouched.
    """
    out = htc.copy()
    idx = np.where(t < t_split)[0]
    if idx.size >= window:
        seg = out[idx]
        pad = window // 2
        kernel = np.ones(window) / window
        out[idx] = np.convolve(np.pad(seg, pad, mode="edge"), kernel, mode="valid")
    return out


class AmarWallCorrection:
    """Amar et al. (2008) Eqs. (37)-(42), using equilibrium-air properties.

    The original code used tabulated air properties.  This reproduction uses
    GRI-Mech 3.0 as the closest available source, so small differences from the
    paper's property tables are expected.
    """

    _AIR_X = "O2:0.21,N2:0.79"

    def __init__(self, h_r_fn, p_e_fn, u_e_fn, cold_wall_T: float = T_INIT):
        try:
            import cantera as ct
        except ImportError as exc:
            raise RuntimeError(
                "The Amar hot-wall correction requires Cantera; "
                'install with pip install -e ".[bprime]"'
            ) from exc
        self._gas = ct.Solution("gri30.yaml")
        self._h_r_fn = h_r_fn
        self._p_e_fn = p_e_fn
        self._u_e_fn = u_e_fn
        self._cold_wall_T = cold_wall_T

    def _air_tp(self, T: float, p: float) -> tuple[float, float, float, float]:
        self._gas.TPX = max(float(T), 200.0), max(float(p), 1.0), self._AIR_X
        self._gas.equilibrate("TP")
        pr = self._gas.cp_mass * self._gas.viscosity / self._gas.thermal_conductivity
        return (
            float(self._gas.enthalpy_mass),
            float(self._gas.density),
            float(self._gas.viscosity),
            float(pr),
        )

    def _air_hp(self, h: float, p: float) -> tuple[float, float]:
        self._gas.TPX = 300.0, max(float(p), 1.0), self._AIR_X
        self._gas.HP = float(h), max(float(p), 1.0)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"ChemEquil::equilibrate: Temperature .* outside valid range",
            )
            self._gas.equilibrate("HP")
        return float(self._gas.density), float(self._gas.viscosity)

    def __call__(self, T_wall: float, time: float, h_wall: float) -> float:
        # h_wall is part of the generic callback protocol but is not h_hw from
        # Amar Eq. (39).  h_hw is the nonablating air enthalpy at T_wall.
        del h_wall
        p_e = self._p_e_fn(time)
        h_hot, rho_hot, mu_hot, pr_hot = self._air_tp(T_wall, p_e)
        h_cold, rho_cold, mu_cold, _pr_cold = self._air_tp(self._cold_wall_T, p_e)

        if time < T_TRANSITION:
            return laminar_wall_factor(mu_hot, rho_hot, mu_cold, rho_cold)

        recovery_factor = pr_hot ** (1.0 / 3.0)
        u_e = self._u_e_fn(time)
        h_edge = self._h_r_fn(time) - recovery_factor * u_e**2 / 2.0
        h_hot_ref = eckert_reference_enthalpy(
            h_edge, h_hot, u_e, recovery_factor,
        )
        h_cold_ref = eckert_reference_enthalpy(
            h_edge, h_cold, u_e, recovery_factor,
        )
        rho_hot_ref, mu_hot_ref = self._air_hp(h_hot_ref, p_e)
        rho_cold_ref, mu_cold_ref = self._air_hp(h_cold_ref, p_e)
        return turbulent_eckert_wall_factor(
            mu_hot_ref, rho_hot_ref, mu_cold_ref, rho_cold_ref,
        )
