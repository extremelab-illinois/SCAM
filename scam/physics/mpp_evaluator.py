# SPDX-License-Identifier: MIT
"""Mutation++ B' evaluator — drop-in replacement for BPrimeTable / BprimeEvaluator.

MutationppEvaluator runs the ``bprime`` CLI once at instantiation over a full
T × P × B'g grid (warm-start strategy) and builds a ``RegularGridInterpolator``
for B'c and h_wall.  Subsequent ``lookup()`` calls are O(1) interpolations.

``surface_enthalpies()`` delegates to a Cantera ``CanteraBackend`` (same NASA-9
reference as Mutation++, so h_g − h_wall differences are physically consistent).

**Usage**::

    from scam.physics.mpp_evaluator import MutationppEvaluator
    ev = MutationppEvaluator.from_config(
        "scam/materials/ablative_organic/tacot_v3.0_mpp_config.yaml"
    )
    B_c, h_wall = ev.lookup(2500.0, 101325.0, 0.3)
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator

# Default Mutation++ installation paths.
#
# Resolved from the environment so this package carries no machine-specific
# path.  Both variables below are Mutation++'s own, set by its `install.sh` /
# environment script, so a standard upstream install needs no extra setup:
#
#   MPP_DIRECTORY       Mutation++ root (the cloned/installed tree)
#   MPP_DATA_DIRECTORY  data directory (mixtures/, thermo/, transport/)
#
# The fallback is ~/Mutationpp, which is the layout Mutation++'s install guide
# produces.  Any of the three can still be overridden per-call or per-config
# via the `mpp_bin` / `mpp_data` / `mpp_lib` keys.
_MPP_ROOT = Path(os.environ.get("MPP_DIRECTORY") or Path.home() / "Mutationpp")
_MPP_BIN_DEFAULT  = _MPP_ROOT / "install" / "bin" / "bprime"
_MPP_DATA_DEFAULT = Path(os.environ.get("MPP_DATA_DIRECTORY") or _MPP_ROOT / "data")
_MPP_LIB_DEFAULT  = _MPP_ROOT / "install" / "lib"

# Mechanism files directory for Cantera surface_enthalpies
_MECHANISMS_DIR = Path(__file__).resolve().parent.parent / "mechanisms"

# Default grid (matches the pre-computed Cantera table range)
_DEFAULT_T_GRID  = np.arange(250.0, 4000.1, 25.0)           # 151 points
_DEFAULT_P_GRID  = np.array([101.325, 1013.25, 10132.5, 101325.0])
_DEFAULT_BG_GRID = np.array([0.0, 0.1, 0.25, 0.5, 1.0, 2.0])

# Nominal TACOT pyro composition string (for surface_enthalpies fallback)
_TACOT_PYRO_X = "CH4:0.5551,CO:0.2418,H2O:0.2031"


class MutationppEvaluator:
    """Warm-start Mutation++ B' evaluator.

    On ``__init__`` the full T × P × B'g grid is swept via the ``bprime`` CLI.
    ``lookup()`` interpolates from the pre-built grid.
    ``surface_enthalpies()`` uses Cantera for h_g and h_c (same NASA-9 reference).
    """

    def __init__(
        self,
        mpp_bin: str | Path = _MPP_BIN_DEFAULT,
        mpp_data: str | Path = _MPP_DATA_DEFAULT,
        mpp_lib: str | Path = _MPP_LIB_DEFAULT,
        mixture_name: str = "tacot_air_bprime",
        mixture_src: str | Path | None = None,
        bl_name: str = "air",
        pyro_name: str = "tacot_pyro",
        cp_name: str = "carbon",
        T_grid: np.ndarray | None = None,
        P_grid: np.ndarray | None = None,
        Bg_grid: np.ndarray | None = None,
        cno_mechanism: str = "cno_ablation.yaml",
        carbon_phase_file: str = "graphite.yaml",
        timeout: int = 300,
    ) -> None:
        self._mpp_bin  = Path(mpp_bin)
        self._mpp_data = Path(mpp_data)
        self._mpp_lib  = Path(mpp_lib)
        self._mixture  = mixture_name
        self._bl       = bl_name
        self._pyro     = pyro_name
        self._cp       = cp_name
        self._timeout  = timeout

        self._T_grid  = np.asarray(T_grid  if T_grid  is not None else _DEFAULT_T_GRID,  dtype=float)
        self._P_grid  = np.asarray(P_grid  if P_grid  is not None else _DEFAULT_P_GRID,  dtype=float)
        self._Bg_grid = np.asarray(Bg_grid if Bg_grid is not None else _DEFAULT_BG_GRID, dtype=float)

        if not self._mpp_bin.exists():
            raise FileNotFoundError(f"Mutation++ bprime binary not found: {self._mpp_bin}")

        # Install mixture XML if a source is given
        if mixture_src is not None:
            dst = self._mpp_data / "mixtures" / Path(mixture_src).name
            if not dst.exists() or dst.stat().st_mtime < Path(mixture_src).stat().st_mtime:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(mixture_src, dst)

        # Build the warm-start grid
        self._itp_bc, self._itp_hw = self._build_interpolators()

        # Cantera backend for surface_enthalpies
        self._cno_mechanism  = cno_mechanism
        self._carbon_phase   = carbon_phase_file
        self._cantera_backend: Any = None   # lazy-loaded

    # ------------------------------------------------------------------
    # Config loader
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_path: str | Path) -> "MutationppEvaluator":
        """Instantiate from a YAML config file."""
        import yaml
        from scam.chemistry.grids import parse_number_grid

        with open(config_path, encoding="utf-8") as f:
            cfg: dict = yaml.safe_load(f)

        mpp = cfg.get("mpp", cfg)

        mpp_bin  = mpp.get("mpp_bin",  str(_MPP_BIN_DEFAULT))
        mpp_data = mpp.get("mpp_data", str(_MPP_DATA_DEFAULT))
        mpp_lib  = mpp.get("mpp_lib",  str(_MPP_LIB_DEFAULT))
        mixture  = mpp.get("mixture",  "tacot_air_bprime")
        bl_name  = mpp.get("bl_name",  "air")
        pyro_name = mpp.get("pyro_name", "tacot_pyro")
        cp_name  = mpp.get("cp_name",   "carbon")
        timeout  = int(mpp.get("timeout", 300))

        # Resolve mixture_src relative to the config file if given
        mixture_src_raw = mpp.get("mixture_src")
        mixture_src: Path | None = None
        if mixture_src_raw:
            p = Path(mixture_src_raw)
            if not p.is_absolute():
                p = Path(config_path).parent / p
            mixture_src = p

        # Grid specs (comma list or start:stop:step)
        T_grid  = parse_number_grid(mpp.get("T_grid",  "250:4000:25"), name="T")
        P_grid  = parse_number_grid(mpp.get("P_grid",  "101.325,1013.25,10132.5,101325"), name="P")
        Bg_grid = parse_number_grid(mpp.get("Bg_grid", "0.0,0.1,0.25,0.5,1.0,2.0"), name="Bg")

        # Cantera mechanism for surface_enthalpies
        cno_mechanism  = mpp.get("cno_mechanism",  "cno_ablation.yaml")
        carbon_phase   = mpp.get("carbon_phase_file", "graphite.yaml")

        return cls(
            mpp_bin=mpp_bin,
            mpp_data=mpp_data,
            mpp_lib=mpp_lib,
            mixture_name=mixture,
            mixture_src=mixture_src,
            bl_name=bl_name,
            pyro_name=pyro_name,
            cp_name=cp_name,
            T_grid=np.array(T_grid, dtype=float),
            P_grid=np.array(P_grid, dtype=float),
            Bg_grid=np.array(Bg_grid, dtype=float),
            cno_mechanism=cno_mechanism,
            carbon_phase_file=carbon_phase,
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Public interface (matches BPrimeTable / BprimeEvaluator)
    # ------------------------------------------------------------------

    def lookup(
        self,
        T_wall: float,
        p_e: float,
        B_g: float,
        Z_C_pyro: float | None = None,
    ) -> tuple[float, float]:
        """Return (B'_c, h_wall [J/kg]) by interpolation from the warm-start grid.

        ``Z_C_pyro`` is accepted for API compatibility but ignored — Mutation++
        uses fixed pyrolysis gas element fractions defined in the mixture XML.
        """
        pt = np.array([[float(T_wall), float(p_e), float(B_g)]])
        b_c  = float(self._itp_bc(pt)[0])
        h_w  = float(self._itp_hw(pt)[0])
        return b_c, h_w

    def pyrolysis_target_fraction(self, element: str) -> float:
        """Return the mass fraction of *element* in the nominal pyrolysis gas.

        Mirrors ``BprimeEvaluator.pyrolysis_target_fraction()`` so both backends
        can be wrapped in ``_DefaultZCBackend`` with the same pattern.
        """
        backend = self._get_cantera_backend()
        gas = backend._gas
        gas.TPX = 300.0, 101325.0, _TACOT_PYRO_X
        elem_fracs = gas.elemental_mass_fractions
        for i in range(gas.n_elements):
            if gas.element_name(i) == element:
                return float(elem_fracs[i])
        raise ValueError(f"Element {element!r} not found in gas mechanism")

    def surface_enthalpies(
        self,
        T_wall: float,
        p_e: float,
        Z_C_pyro: float | None = None,
    ) -> tuple[float, float]:
        """Return (h_g, h_c) [J/kg] using Cantera (same NASA-9 reference).

        h_g is the UNREACTED pyrolysis-gas enthalpy (Cantera NASA-9 reference).
        seb_residual overrides it with mat_surface.h_g_table when the material
        provides one (TACOT, PICA, …) — that table carries the PATO-consistent
        equilibrium enthalpy and is on the same Cantera absolute reference as
        h_wall.  This method is therefore only the fallback for materials without
        an h_g_table.
        """
        backend = self._get_cantera_backend()
        T_safe = max(float(T_wall), 200.0)
        p_safe = float(p_e)
        if Z_C_pyro is not None:
            from scam.physics.bprime_evaluator import _pyro_x_from_z_c_molecular
            pyro_x = _pyro_x_from_z_c_molecular(float(Z_C_pyro), _TACOT_PYRO_X)
        else:
            pyro_x = _TACOT_PYRO_X
        h_g = backend.gas_enthalpy(T_safe, p_safe, pyro_x)
        h_c = backend.condensed_enthalpy(T_safe, p_safe)
        return h_g, h_c

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mpp_env(self) -> dict:
        env = os.environ.copy()
        lib_path = str(self._mpp_lib)
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"]    = f"{lib_path}:{existing}" if existing else lib_path
        env["MPP_DATA_DIRECTORY"] = str(self._mpp_data)
        return env

    def _run_bprime_slice(self, P: float, Bg: float) -> tuple[np.ndarray, np.ndarray]:
        """Run ``bprime`` for one (P, Bg) pair; return (Bc_arr, hw_arr_J_kg)."""
        T1, dT, T2 = self._T_grid[0], self._T_grid[1] - self._T_grid[0], self._T_grid[-1]
        cmd = [
            str(self._mpp_bin),
            "-T", f"{T1:.1f}:{dT:.1f}:{T2:.1f}",
            "-P", f"{P:.6g}",
            "-b", f"{Bg:.6g}",
            "-m", self._mixture,
            "-bl", self._bl,
            "-py", self._pyro,
            "-cp", self._cp,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            env=self._mpp_env(), timeout=self._timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"bprime failed (P={P:.0f} Pa, B'g={Bg}):\n{result.stderr}"
            )

        Bc_list, hw_list = [], []
        for line in result.stdout.splitlines():
            cols = line.split()
            if not cols or cols[0].startswith('"'):
                continue
            try:
                Bc_list.append(float(cols[1]))
                hw_list.append(float(cols[2]) * 1e6)  # MJ/kg → J/kg
            except (ValueError, IndexError):
                continue

        nT = len(self._T_grid)
        if len(Bc_list) != nT:
            raise RuntimeError(
                f"bprime returned {len(Bc_list)} rows, expected {nT} "
                f"(P={P:.0f} Pa, B'g={Bg})"
            )
        return np.array(Bc_list), np.array(hw_list)

    def _build_interpolators(
        self,
    ) -> tuple[RegularGridInterpolator, RegularGridInterpolator]:
        """Sweep the full grid and build interpolators."""
        nT  = len(self._T_grid)
        nP  = len(self._P_grid)
        nBg = len(self._Bg_grid)
        Bc  = np.zeros((nT, nP, nBg))
        Hw  = np.zeros((nT, nP, nBg))

        for jp, P in enumerate(self._P_grid):
            for jb, Bg in enumerate(self._Bg_grid):
                bc_arr, hw_arr = self._run_bprime_slice(P, Bg)
                Bc[:, jp, jb] = bc_arr
                Hw[:, jp, jb] = hw_arr

        itp_bc = RegularGridInterpolator(
            (self._T_grid, self._P_grid, self._Bg_grid),
            Bc, method="linear", bounds_error=False, fill_value=None,
        )
        itp_hw = RegularGridInterpolator(
            (self._T_grid, self._P_grid, self._Bg_grid),
            Hw, method="linear", bounds_error=False, fill_value=None,
        )
        return itp_bc, itp_hw

    def _resolve_mechanism(self, name: str) -> str:
        p = Path(name)
        if p.is_absolute() and p.exists():
            return str(p)
        candidate = _MECHANISMS_DIR / name
        if candidate.exists():
            return str(candidate)
        return name

    def _get_cantera_backend(self):
        if self._cantera_backend is None:
            from scam.chemistry.thermo_backend import CanteraBackend
            self._cantera_backend = CanteraBackend(
                gas_mechanism=self._resolve_mechanism(self._cno_mechanism),
                carbon_phase_file=self._resolve_mechanism(self._carbon_phase),
            )
        return self._cantera_backend

    def _build_pyro_x(self, Z_C_pyro: float | None) -> str:
        if Z_C_pyro is None:
            return _TACOT_PYRO_X
        # Delegate to the same atomic composition logic as BprimeEvaluator
        from scam.physics.bprime_evaluator import _pyro_x_from_z_c
        candidate = _pyro_x_from_z_c(Z_C_pyro, _TACOT_PYRO_X)
        try:
            backend = self._get_cantera_backend()
            backend._gas.TPX = 300.0, 101325.0, candidate
            return candidate
        except Exception:
            from scam.physics.bprime_evaluator import _pyro_x_from_z_c_molecular
            return _pyro_x_from_z_c_molecular(Z_C_pyro, _TACOT_PYRO_X)
