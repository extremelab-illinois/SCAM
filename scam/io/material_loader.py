# SPDX-License-Identifier: MIT
"""Load MaterialCard and BPrimeTable from YAML files.

YAML material card format:

    name: TACOT
    version: "3.0"
    description: "..."

    rho_virgin: 280.0          # [kg/m^3]
    rho_char:   180.0          # [kg/m^3]
    gamma_resin: 0.2           # [-]

    components:
      - name: resin_A
        rho_0: 60.0            # [kg/m^3]
        rho_r: 0.0             # [kg/m^3]
        A_rate: 1.2e4          # [1/s]
        E_act:  8.7e4          # [J/mol]
        m_exp:  3.0            # [-]
        h_decomp: 1.63e6       # [J/kg]

    k_virgin:   [[200, 0.337], [400, 0.401], ...]   # [[T_K, k_W_mK], ...]
    k_char:     [[200, 1.05],  ...]
    cp_virgin:  [[200, 711.0], ...]
    cp_char:    [[200, 720.0], ...]
    h_g:        [[300, 0.0],   ...]                 # [[T_K, h_g_J_kg], ...]

    emissivity: 0.85                                # virgin scalar emissivity
    emissivity_char: 0.90                           # optional: char scalar; blended by char fraction
    # Optional temperature-dependent emissivity (overrides the scalars above);
    # interpolated at T_wall per phase then blended by char fraction:
    emissivity_virgin: [[300, 0.80], [2000, 0.78], ...]   # [[T_K, eps], ...]
    emissivity_char_table: [[300, 0.90], [2000, 0.92], ...]
    b_prime_table: tacot_v3.0_bprime_air.yaml          # relative to this file

B' table YAML format (3-D, legacy):

    description: "..."
    T_wall_K:  [500, 700, ...]
    p_e_Pa:    [1000, 10000, ...]
    B_g_prime: [0.0, 0.1, ...]
    B_c_prime: [[[...], ...], ...]   # shape (n_T, n_p, n_Bg)
    h_wall:    [[[...], ...], ...]   # shape (n_T, n_p, n_Bg) [J/kg]
    h_g:       [[...], ...]          # optional shape (n_T, n_p) [J/kg]
    h_c:       [[...], ...]          # optional shape (n_T, n_p) [J/kg]
    target_element: C                 # optional, defaults to C
    surface_source_target_fraction: 1.0

B' table YAML format (row-format, TACOT 3.0-style):

    columns: ["p_e_Pa", "B_g_prime", "B_c_prime", ..., "T_wall_K", ..., "h_wall_J_kg", "ablating?"]
    rows:
      - [101325.0, 0.5, 0.001, ..., 3200.0, ..., 1.2e7, "ablating"]
      - ...

B' table YAML format (4-D, element conservation):

    description: "..."
    axes:
      T_wall_K:  [500, 700, ...]
      p_e_Pa:    [1000, 10000, ...]
      B_g_prime: [0.0, 0.5, 1.0]
      Z_C_pyro:  [0.15, 0.25, 0.35, 0.50, 0.65]
    B_c_prime: [[[[...], ...], ...], ...]   # shape (n_T, n_p, n_Bg, n_ZC)
    h_wall:    [[[[...], ...], ...], ...]   # shape (n_T, n_p, n_Bg, n_ZC) [J/kg]
    h_g:       [[[...], ...], ...]           # optional shape (n_T, n_p, n_ZC) [J/kg]
    h_c:       [[...], ...]                  # optional shape (n_T, n_p) [J/kg]
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from scam.config.material import ComponentCard, MaterialCard, BPrimeTableRef, KineticAblationCard
from scam.core.errors import SCAMInputError


# ---------------------------------------------------------------------------
# B' table (loaded on demand from physics.chemistry)
# ---------------------------------------------------------------------------

class BPrimeTable:
    """3-D or 4-D B' table lookup.

    3-D mode (default): (T_wall, p_e, B'_g) → (B'_c, h_wall)
    4-D mode: (T_wall, p_e, B'_g, Z_C_pyro) → (B'_c, h_wall)

    4-D mode is activated when the YAML file contains an ``axes.Z_C_pyro`` key
    (new schema) or a top-level ``Z_C_pyro`` key.  The 3-D schema (legacy)
    uses top-level ``T_wall_K`` / ``p_e_Pa`` / ``B_g_prime`` keys.

    Uses scipy RegularGridInterpolator with linear interpolation and
    boundary clamping (no extrapolation beyond table edges — clamp to edge value).
    """

    def __init__(self, path: str) -> None:
        self._path = path
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        from scipy.interpolate import RegularGridInterpolator

        # Support both legacy flat schema and new axes-dict schema
        if "axes" in raw:
            axes = raw["axes"]
            self._T   = np.asarray(axes["T_wall_K"],   dtype=float)
            self._p   = np.asarray(axes["p_e_Pa"],     dtype=float)
            self._Bg  = np.asarray(axes["B_g_prime"],  dtype=float)
            self._is_4d = "Z_C_pyro" in axes
            if self._is_4d:
                self._ZC = np.asarray(axes["Z_C_pyro"], dtype=float)
                self._ZC_nominal = float(self._ZC[len(self._ZC) // 2])
        else:
            self._T   = np.asarray(raw["T_wall_K"],   dtype=float)
            self._p   = np.asarray(raw["p_e_Pa"],     dtype=float)
            self._Bg  = np.asarray(raw["B_g_prime"],  dtype=float)
            self._is_4d = "Z_C_pyro" in raw
            if self._is_4d:
                self._ZC = np.asarray(raw["Z_C_pyro"], dtype=float)
                self._ZC_nominal = float(self._ZC[len(self._ZC) // 2])

        self.target_element = str(raw.get("target_element", "C"))
        self.surface_source_target_fraction = float(
            raw.get("surface_source_target_fraction", 1.0)
        )
        bprime_key = "B_c_prime" if "B_c_prime" in raw else "Bprime_surface"
        if bprime_key not in raw:
            raise SCAMInputError(
                f"B' table {path}: missing B_c_prime or Bprime_surface array"
            )
        Bc = np.asarray(raw[bprime_key], dtype=float)
        hw = np.asarray(raw["h_wall"],    dtype=float)

        if self._is_4d:
            expected = (len(self._T), len(self._p), len(self._Bg), len(self._ZC))
            if Bc.shape != expected or hw.shape != expected:
                raise SCAMInputError(
                    f"B' table {path}: 4-D arrays must have shape {expected}, "
                    f"got {Bc.shape} and {hw.shape}"
                )
            grid = (self._T, self._p, self._Bg, self._ZC)
        else:
            expected = (len(self._T), len(self._p), len(self._Bg))
            if Bc.shape != expected or hw.shape != expected:
                raise SCAMInputError(
                    f"B' table {path}: B_c_prime and h_wall must have shape {expected}, "
                    f"got {Bc.shape} and {hw.shape}"
                )
            grid = (self._T, self._p, self._Bg)

        self._interp_Bc = RegularGridInterpolator(
            grid, Bc, method="linear", bounds_error=False, fill_value=None,
        )
        self._interp_hw = RegularGridInterpolator(
            grid, hw, method="linear", bounds_error=False, fill_value=None,
        )

        # Optional surface-enthalpy arrays.  Fixed-composition tables store
        # h_g(T,p), while element-transport tables store h_g(T,p,Z_C_pyro).
        # Condensed-char enthalpy is composition independent and remains h_c(T,p).
        # Presence of these arrays activates BPrimeTable.surface_enthalpies(), which
        # makes the SEB advective terms (qAdvPyro, qAdvChar) active for table runs,
        # matching the physics of the live BprimeEvaluator path.
        self._interp_hg = self._interp_hc = None
        has_hg = "h_g" in raw
        has_hc = "h_c" in raw
        if has_hg != has_hc:
            raise SCAMInputError(
                f"B' table {path}: optional h_g and h_c arrays must be provided together"
            )
        if has_hg:
            hg = np.asarray(raw["h_g"], dtype=float)
            hc = np.asarray(raw["h_c"], dtype=float)
            expected_hg = (
                (len(self._T), len(self._p), len(self._ZC))
                if self._is_4d
                else (len(self._T), len(self._p))
            )
            expected_hc = (len(self._T), len(self._p))
            if hg.shape != expected_hg or hc.shape != expected_hc:
                raise SCAMInputError(
                    f"B' table {path}: h_g and h_c must have shapes "
                    f"{expected_hg} and {expected_hc}, got {hg.shape} and {hc.shape}"
                )
            hg_grid = (self._T, self._p, self._ZC) if self._is_4d else (self._T, self._p)
            self._interp_hg = RegularGridInterpolator(
                hg_grid, hg, method="linear", bounds_error=False, fill_value=None,
            )
            self._interp_hc = RegularGridInterpolator(
                (self._T, self._p), hc, method="linear",
                bounds_error=False, fill_value=None,
            )

    def lookup(
        self,
        T_w: float,
        p_e: float,
        B_g_prime: float,
        Z_C_pyro: float | None = None,
    ) -> tuple[float, float]:
        """Return (B'_c, h_wall [J/kg]) for given conditions.

        Parameters
        ----------
        T_w:       wall temperature [K]
        p_e:       edge pressure [Pa]
        B_g_prime: pyrolysis gas blowing parameter [-]
        Z_C_pyro:  carbon mass fraction in pyrolysis gas [-] (4-D tables only).
                   Ignored for 3-D tables.  If None and table is 4-D, uses the
                   nominal (middle) Z_C value.

        Out-of-range inputs are clamped to the table boundary.
        """
        T_c  = float(np.clip(T_w,       self._T[0],  self._T[-1]))
        p_c  = float(np.clip(p_e,       self._p[0],  self._p[-1]))
        Bg_c = float(np.clip(B_g_prime, self._Bg[0], self._Bg[-1]))

        if self._is_4d:
            ZC = Z_C_pyro if Z_C_pyro is not None else self._ZC_nominal
            ZC_c = float(np.clip(ZC, self._ZC[0], self._ZC[-1]))
            pt = [[T_c, p_c, Bg_c, ZC_c]]
        else:
            pt = [[T_c, p_c, Bg_c]]

        Bc = float(self._interp_Bc(pt)[0])
        hw = float(self._interp_hw(pt)[0])
        return Bc, hw

    def surface_enthalpies(
        self,
        T_wall: float,
        p_e: float,
        Z_C_pyro: float | None = None,
    ) -> tuple[float, float]:
        """Return stored/interpolated ``(h_g, h_c)`` [J/kg].

        Only available when the table was generated with surface-enthalpy columns
        (h_g, h_c stored alongside B_c_prime / h_wall).  For a 4-D table, h_g is
        interpolated over Z_C_pyro using the same clamp/default convention as
        lookup(); h_c is composition independent.  Raises AttributeError for
        backward compatibility when the optional arrays are absent.
        """
        if self._interp_hg is None:
            raise AttributeError("BPrimeTable: h_g/h_c columns not present in table")
        T_c = float(np.clip(T_wall, self._T[0], self._T[-1]))
        p_c = float(np.clip(p_e,    self._p[0], self._p[-1]))
        if self._is_4d:
            ZC = Z_C_pyro if Z_C_pyro is not None else self._ZC_nominal
            ZC_c = float(np.clip(ZC, self._ZC[0], self._ZC[-1]))
            hg_pt = [[T_c, p_c, ZC_c]]
        else:
            hg_pt = [[T_c, p_c]]
        hc_pt = [[T_c, p_c]]
        return float(self._interp_hg(hg_pt)[0]), float(self._interp_hc(hc_pt)[0])


class BPrimeTableRowFormat:
    """B' table from flat-row YAML (TACOT 3.0-style).

    Format::

        columns: ["p_e_Pa", "B_g_prime", "B_c_prime", ..., "T_wall_K", ...,
                  "h_wall_J_kg", "ablating?"]
        rows:
          - [101325.0, 0.5, 0.001, ..., 3200.0, ..., 1.2e7, "ablating"]
          - ...

    Lookup uses ``scipy.interpolate.LinearNDInterpolator`` on a 3-D point
    cloud in ``(T_wall, log10(p_e), log10(B'_g))`` space.  B'_g is clamped
    to the table's minimum before taking the log to handle B'_g = 0 inputs.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        columns: list[str] = raw["columns"]
        rows: list = raw["rows"]

        col = {name: i for i, name in enumerate(columns)}
        ablating_col = col.get("ablating?", None)

        # Parse numeric columns; string "ablating?" column handled separately
        n_numeric = len(columns) - (1 if ablating_col is not None else 0)
        numeric_cols = [i for i in range(len(columns)) if i != ablating_col]

        data = np.array(
            [[float(r[i]) for i in numeric_cols] for r in rows],
            dtype=float,
        )

        # Re-map column indices after dropping the string column
        remap = {orig: new for new, orig in enumerate(numeric_cols)}

        p_e  = data[:, remap[col["p_e_Pa"]]]
        B_g  = data[:, remap[col["B_g_prime"]]]
        B_c  = data[:, remap[col["B_c_prime"]]]
        T_w  = data[:, remap[col["T_wall_K"]]]
        h_w  = data[:, remap[col["h_wall_J_kg"]]]

        # Build 3-D interpolator in (T_wall, log10 p_e, log10 B'_g) space.
        # Using log scale for p_e and B'_g because both span multiple orders of
        # magnitude; linear scale in T_wall (it's already well-distributed).
        from scipy.interpolate import LinearNDInterpolator

        log_p  = np.log10(np.maximum(p_e, 1e-30))
        log_Bg = np.log10(np.maximum(B_g, 1e-30))
        pts = np.column_stack([T_w, log_p, log_Bg])

        self._interp_Bc = LinearNDInterpolator(pts, B_c)
        self._interp_hw = LinearNDInterpolator(pts, h_w)

        self._T_range    = (float(T_w.min()),    float(T_w.max()))
        self._log_p_range  = (float(log_p.min()),  float(log_p.max()))
        self._log_Bg_range = (float(log_Bg.min()), float(log_Bg.max()))

    def _interp_at(
        self, T_c: float, lp_c: float, lBg_c: float, *, _raw: bool = False
    ) -> tuple[float, float]:
        """Evaluate interpolators at a clamped (T, log_p, log_Bg) point.

        With ``_raw=True`` the raw (possibly NaN) interpolator values are
        returned without the zero-fallback, so callers can detect hull misses.
        """
        pt = [[T_c, lp_c, lBg_c]]
        Bc = float(self._interp_Bc(pt)[0])
        hw = float(self._interp_hw(pt)[0])
        if not _raw:
            if np.isnan(Bc):
                Bc = 0.0
            if np.isnan(hw):
                hw = 0.0
        return Bc, hw

    def in_hull(self, T_w: float, p_e: float, B_g_prime: float) -> bool:
        """Return True when (T_w, p_e, B'_g) is inside the interpolator's convex hull.

        Points outside the hull fall back to 0.0 in :meth:`lookup`; use this
        to exclude those points from quality metrics.
        """
        log_p  = np.log10(max(p_e,        1e-30))
        log_Bg = np.log10(max(B_g_prime,  10.0 ** self._log_Bg_range[0]))
        lp_c  = float(np.clip(log_p,  *self._log_p_range))
        lBg_c = float(np.clip(log_Bg, *self._log_Bg_range))
        T_min, T_max = self._T_range
        T_c   = float(np.clip(T_w, T_min, T_max))
        Bc, _ = self._interp_at(T_c, lp_c, lBg_c, _raw=True)
        return not np.isnan(Bc)

    def lookup(
        self,
        T_w: float,
        p_e: float,
        B_g_prime: float,
        Z_C_pyro: float | None = None,
    ) -> tuple[float, float]:
        """Return (B'_c, h_wall [J/kg]).  B'_g = 0 is clamped to table minimum.

        T_w outside the table range uses linear extrapolation from the boundary
        so that the Newton finite-difference Jacobian sees a non-zero gradient.
        """
        log_p  = np.log10(max(p_e,        1e-30))
        log_Bg = np.log10(max(B_g_prime,  10.0 ** self._log_Bg_range[0]))

        lp_c  = float(np.clip(log_p,  *self._log_p_range))
        lBg_c = float(np.clip(log_Bg, *self._log_Bg_range))

        T_min, T_max = self._T_range
        if T_w < T_min:
            # Linear extrapolation below table minimum using a 5 K finite-diff slope.
            _dT = 5.0
            Bc0, hw0 = self._interp_at(T_min,        lp_c, lBg_c)
            Bc1, hw1 = self._interp_at(T_min + _dT,  lp_c, lBg_c)
            alpha = (T_w - T_min) / _dT
            return Bc0 + alpha * (Bc1 - Bc0), hw0 + alpha * (hw1 - hw0)

        if T_w > T_max:
            _dT = 5.0
            Bc0, hw0 = self._interp_at(T_max - _dT,  lp_c, lBg_c)
            Bc1, hw1 = self._interp_at(T_max,         lp_c, lBg_c)
            alpha = (T_w - T_max) / _dT
            return Bc1 + alpha * (Bc1 - Bc0), hw1 + alpha * (hw1 - hw0)

        return self._interp_at(T_w, lp_c, lBg_c)


def _load_b_prime_table(bp_path: str):
    """Load a B' table YAML file, auto-detecting 3-D grid vs row format."""
    with open(bp_path, "r", encoding="utf-8") as f:
        peek = yaml.safe_load(f)
    if "rows" in peek and "columns" in peek:
        return BPrimeTableRowFormat(bp_path)
    return BPrimeTable(bp_path)


def _load_gas_properties_pT(path: str) -> dict:
    """Load a (p, T) pyrolysis-gas equilibrium property table.

    Companion-file format (see ``tacot_v3.0_gasProperties_pT.yaml``)::

        p_Pa: [p_0, p_1, ...]        # ascending, [Pa]
        T_K:  [T_0, T_1, ...]        # ascending, [K]
        M_kg_mol:  [[...], ...]      # shape (n_p, n_T)
        h_g_J_kg:  [[...], ...]      # shape (n_p, n_T), ABSOLUTE reference
        mu_Pa_s:   [[...], ...]      # shape (n_p, n_T)

    Returns the dict consumed by ``MaterialCard.gas_properties_pT`` and
    ``properties.py::interp_gas_pT``: keys "p", "T" (1-D) and "M", "h_g",
    "mu" (2-D, shape (n_p, n_T)).
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    p_ax = np.asarray(raw["p_Pa"], dtype=float)
    T_ax = np.asarray(raw["T_K"], dtype=float)
    M = np.asarray(raw["M_kg_mol"], dtype=float)
    H = np.asarray(raw["h_g_J_kg"], dtype=float)
    MU = np.asarray(raw["mu_Pa_s"], dtype=float)
    for name, arr in (("M_kg_mol", M), ("h_g_J_kg", H), ("mu_Pa_s", MU)):
        if arr.shape != (len(p_ax), len(T_ax)):
            raise SCAMInputError(
                f"{path}: {name} has shape {arr.shape}, expected "
                f"({len(p_ax)}, {len(T_ax)}) = (n_p, n_T)"
            )
    return {"p": p_ax, "T": T_ax, "M": M, "h_g": H, "mu": MU}


# ---------------------------------------------------------------------------
# Material card loader
# ---------------------------------------------------------------------------

def _parse_table(raw, name: str, path: str) -> np.ndarray:
    """Parse a property table from YAML (list of [T, value] pairs)."""
    try:
        arr = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise SCAMInputError(f"{path}: cannot parse table '{name}': {exc}") from exc
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise SCAMInputError(
            f"{path}: table '{name}' must be a list of [T, value] pairs; got shape {arr.shape}"
        )
    if not np.all(np.diff(arr[:, 0]) > 0):
        raise SCAMInputError(f"{path}: table '{name}' temperatures must be strictly increasing")
    return arr


_R_GAS = 8.314  # universal gas constant [J/mol/K]


def _parse_component(raw: dict, idx: int, card_path: str) -> ComponentCard | None:
    """Parse one decomposing component from a YAML dict.

    Returns None for inert components (A_rate == 0 and rho_r == rho_0), which
    are silently skipped — their density is already implicit in rho_char.

    Accepts either ``E_act`` [J/mol] or ``E_over_R`` [K] for the activation
    energy; ``E_act = E_over_R * R`` where R = 8.314 J/mol/K.
    """
    required_base = ("name", "rho_0", "rho_r", "A_rate", "m_exp")
    for key in required_base:
        if key not in raw:
            raise SCAMInputError(f"{card_path}: component {idx} missing required field '{key}'")
    if "E_act" not in raw and "E_over_R" not in raw:
        raise SCAMInputError(
            f"{card_path}: component {idx} must have either 'E_act' [J/mol] or 'E_over_R' [K]"
        )
    try:
        rho_0  = float(raw["rho_0"])
        rho_r  = float(raw["rho_r"])
        A_rate = float(raw["A_rate"])
        m_exp  = float(raw["m_exp"])
        if "E_act" in raw:
            E_act = float(raw["E_act"])
        else:
            E_act = float(raw["E_over_R"]) * _R_GAS
        comp = ComponentCard(
            name=str(raw["name"]),
            rho_0=rho_0,
            rho_r=rho_r,
            A_rate=A_rate,
            E_act=E_act,
            m_exp=m_exp,
            h_decomp=float(raw.get("h_decomp", 0.0)),
        )
    except (TypeError, ValueError) as exc:
        raise SCAMInputError(f"{card_path}: component {idx}: {exc}") from exc

    # Inert component (fiber / reinforcement): rho_r == rho_0, A_rate == 0.
    # Its density contribution is already captured in rho_char; skip it.
    if comp.A_rate == 0.0 and comp.rho_r == comp.rho_0:
        return None

    if comp.rho_r >= comp.rho_0:
        raise SCAMInputError(
            f"{card_path}: component '{comp.name}': rho_r ({comp.rho_r}) must be < rho_0 ({comp.rho_0})"
        )
    if comp.A_rate <= 0:
        raise SCAMInputError(f"{card_path}: component '{comp.name}': A_rate must be positive")
    if comp.E_act <= 0:
        raise SCAMInputError(f"{card_path}: component '{comp.name}': E_act must be positive")
    return comp


def load_material(path: str) -> tuple[MaterialCard, Optional[BPrimeTable]]:
    """Load a MaterialCard (and optional BPrimeTable) from a YAML file.

    Returns ``(material_card, b_prime_table_or_None)``.
    """
    path = str(Path(path).resolve())
    card_dir = os.path.dirname(path)

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise SCAMInputError(f"{path}: empty material card")

    required_top = ("name", "rho_virgin", "rho_char", "gamma_resin")
    for key in required_top:
        if key not in raw:
            raise SCAMInputError(f"{path}: missing required field '{key}'")

    try:
        rho_virgin = float(raw["rho_virgin"])
        rho_char   = float(raw["rho_char"])
        gamma_resin = float(raw["gamma_resin"])
    except (TypeError, ValueError) as exc:
        raise SCAMInputError(f"{path}: {exc}") from exc

    decomposing = bool(raw.get("decomposing", True))
    if rho_char > rho_virgin or (rho_char == rho_virgin and decomposing):
        raise SCAMInputError(
            f"{path}: rho_char ({rho_char}) must be < rho_virgin ({rho_virgin}) "
            "for decomposing materials"
        )

    # Components (inert entries return None and are skipped)
    components = []
    for i, comp_raw in enumerate(raw.get("components", [])):
        comp = _parse_component(comp_raw, i, path)
        if comp is not None:
            components.append(comp)

    # Property tables (required)
    required_tables = ("k_virgin", "k_char", "cp_virgin", "cp_char", "h_g")
    tables = {}
    for tname in required_tables:
        if tname not in raw:
            raise SCAMInputError(f"{path}: missing required property table '{tname}'")
        tables[tname] = _parse_table(raw[tname], tname, path)

    emissivity = float(raw.get("emissivity", 0.85))
    emissivity_char = float(raw.get("emissivity_char", -1.0))
    # Optional temperature-dependent emissivity tables [[T_K, eps], ...].
    emissivity_virgin_table = (
        _parse_table(raw["emissivity_virgin"], "emissivity_virgin", path)
        if "emissivity_virgin" in raw else None
    )
    emissivity_char_table = (
        _parse_table(raw["emissivity_char_table"], "emissivity_char_table", path)
        if "emissivity_char_table" in raw else None
    )

    # Elemental mass fractions of pyrolysis gas (optional)
    pyro_elem_fracs = None
    if "pyro_elem_fracs" in raw:
        pef_raw = raw["pyro_elem_fracs"]
        if pef_raw is not None:
            elem_order = ("C", "H", "O", "N")
            tables_pef = []
            for elem in elem_order:
                if elem not in pef_raw:
                    raise SCAMInputError(f"{path}: pyro_elem_fracs missing element '{elem}'")
                tables_pef.append(_parse_table(pef_raw[elem], f"pyro_elem_fracs.{elem}", path))
            # stack to shape (4, n_T, 2)
            pyro_elem_fracs = np.stack(tables_pef, axis=0)

    # Optional absolute enthalpy tables [[T_K, h_J_kg], ...] for solid phases.
    h_virgin_table = (
        _parse_table(raw["h_virgin"], "h_virgin", path)
        if "h_virgin" in raw else None
    )
    h_char_table = (
        _parse_table(raw["h_char"], "h_char", path)
        if "h_char" in raw else None
    )

    # Optional gas porosity and transport parameters.
    # Porosity may be given as a [[beta, eps], ...] table (beta = char fraction = 0 virgin, 1 char);
    # if so, use the endpoints: beta=0 → eps_g_virgin, beta=1 → eps_g_char.
    _poros_raw = raw.get("porosity", None)
    if isinstance(_poros_raw, list):
        eps_g_virgin = float(raw.get("eps_g_virgin", _poros_raw[0][-1]))
        eps_g_char   = float(raw.get("eps_g_char",   _poros_raw[-1][-1]))
    else:
        eps_g_virgin = float(raw.get("eps_g_virgin", 0.0))
        eps_g_char   = float(raw.get("eps_g_char",   0.0))
    gas_molar_mass = float(raw.get("gas_molar_mass", 0.022))
    gas_molar_mass_table = (
        _parse_table(raw["gas_molar_mass_table"], "gas_molar_mass_table", path)
        if "gas_molar_mass_table" in raw else None
    )
    gas_viscosity_table = (
        _parse_table(raw["gas_viscosity_table"], "gas_viscosity_table", path)
        if "gas_viscosity_table" in raw else None
    )
    # Optional full (p, T) equilibrium gas-property table (companion file,
    # relative to this card, same resolution convention as b_prime_table).
    # When present, the in-depth gas-energy path (assembly h_g_cache, gas
    # storage), the SEB q_adv, and the 1D/2D Darcy transport all interpolate
    # bilinearly in (p, T) instead of using the 1-atm gas_*_table slices /
    # h_g_abs_offset — see physics/gas_enthalpy.py::pyrolysis_gas_enthalpy_abs.
    gas_properties_pT = None
    if "gas_properties_pT" in raw:
        gpT_path = str(Path(card_dir) / str(raw["gas_properties_pT"]))
        gas_properties_pT = _load_gas_properties_pT(gpT_path)
    # permeability may be a scalar or a [[beta, K], ...] table (beta = char fraction);
    # if a table, use the fully-charred endpoint as char permeability and the first row as virgin.
    _perm_raw = raw.get("permeability", 0.0)
    if isinstance(_perm_raw, list):
        permeability         = float(_perm_raw[-1][-1])
        _perm_virgin_default = float(_perm_raw[0][-1])
    else:
        permeability         = float(_perm_raw)
        _perm_virgin_default = 0.0
    permeability_virgin  = float(raw.get("permeability_virgin", _perm_virgin_default))
    klinkenberg_b        = float(raw.get("klinkenberg_b", 0.0))
    tortuosity           = float(raw.get("tortuosity", 1.0))

    # Initial elemental fracs [C,H,O,N] inside material (element transport)
    initial_Z_elem: Optional[np.ndarray] = None
    if "initial_Z_elem" in raw:
        z = raw["initial_Z_elem"]
        if len(z) != 4:
            raise SCAMInputError(f"{path}: initial_Z_elem must have 4 entries [C, H, O, N]")
        initial_Z_elem = np.array([float(v) for v in z], dtype=float)

    # Optional in-plane conductivity tables (stored but not used by 1-D solver)
    k_virgin_ip_table = (
        _parse_table(raw["k_virgin_ip"], "k_virgin_ip", path)
        if "k_virgin_ip" in raw else None
    )
    k_char_ip_table = (
        _parse_table(raw["k_char_ip"], "k_char_ip", path)
        if "k_char_ip" in raw else None
    )

    # B' table reference (auto-detects 3-D grid vs row format)
    b_prime_ref = None
    b_prime_table = None
    if "b_prime_table" in raw:
        bp_rel = str(raw["b_prime_table"])
        bp_path = str(Path(card_dir) / bp_rel)
        b_prime_ref = BPrimeTableRef(path=bp_path)
        b_prime_table = _load_b_prime_table(bp_path)

    # Kemp (1968) kinetic ablation closure (reaction-rate-limited, mutually
    # exclusive with the B'-table equilibrium closure above).
    kinetic_ablation = None
    if "kinetic_ablation" in raw:
        if "b_prime_table" in raw:
            raise SCAMInputError(
                f"{path}: cannot specify both 'kinetic_ablation' and "
                "'b_prime_table' (ambiguous ablation closure — pick one)"
            )
        ka_raw = raw["kinetic_ablation"]
        try:
            h_tot = ka_raw["h_ablation_total_coeffs"]
            h_rxn = ka_raw["h_ablation_reaction_coeffs"]
            if len(h_tot) != 3:
                raise SCAMInputError(
                    f"{path}: h_ablation_total_coeffs must have 3 entries [a,b,c]"
                )
            if len(h_rxn) != 2:
                raise SCAMInputError(
                    f"{path}: h_ablation_reaction_coeffs must have 2 entries [a,b]"
                )
            kinetic_ablation = KineticAblationCard(
                B=float(ka_raw["B"]),
                E_a=float(ka_raw["E_a"]),
                h_ablation_total_coeffs=(float(h_tot[0]), float(h_tot[1]), float(h_tot[2])),
                h_ablation_reaction_coeffs=(float(h_rxn[0]), float(h_rxn[1])),
                rho_sw_override=(
                    float(ka_raw["rho_sw_override"])
                    if "rho_sw_override" in ka_raw else None
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SCAMInputError(f"{path}: kinetic_ablation: {exc}") from exc
        if kinetic_ablation.B <= 0 or kinetic_ablation.E_a <= 0:
            raise SCAMInputError(f"{path}: kinetic_ablation: B and E_a must be positive")

    material_category = raw.get("material_category", None)
    # dataset_version: accept new key or legacy "version" key for backward compat
    dataset_version = raw.get("dataset_version") or (str(raw["version"]) if "version" in raw else None)
    card_version = str(raw.get("card_version", "1.0"))
    status = str(raw.get("status", "provisional"))

    h_g_abs_offset = (float(raw["h_g_abs_offset"]) if "h_g_abs_offset" in raw else None)

    from scam.physics.properties import build_sensible_enthalpy_table as _bsh

    card = MaterialCard(
        name=str(raw["name"]),
        rho_virgin=rho_virgin,
        rho_char=rho_char,
        gamma_resin=gamma_resin,
        dataset_version=dataset_version,
        card_version=card_version,
        status=status,
        components=components,
        k_virgin_table=tables["k_virgin"],
        k_char_table=tables["k_char"],
        k_virgin_ip_table=k_virgin_ip_table,
        k_char_ip_table=k_char_ip_table,
        cp_virgin_table=tables["cp_virgin"],
        cp_char_table=tables["cp_char"],
        h_g_table=tables["h_g"],
        h_virgin_table=h_virgin_table,
        h_char_table=h_char_table,
        emissivity=emissivity,
        emissivity_char=emissivity_char,
        emissivity_virgin_table=emissivity_virgin_table,
        emissivity_char_table=emissivity_char_table,
        eps_g_virgin=eps_g_virgin,
        eps_g_char=eps_g_char,
        gas_molar_mass=gas_molar_mass,
        gas_molar_mass_table=gas_molar_mass_table,
        gas_viscosity_table=gas_viscosity_table,
        gas_properties_pT=gas_properties_pT,
        permeability=permeability,
        permeability_virgin=permeability_virgin,
        klinkenberg_b=klinkenberg_b,
        tortuosity=tortuosity,
        decomposing=decomposing,
        b_prime_ref=b_prime_ref,
        kinetic_ablation=kinetic_ablation,
        pyro_elem_fracs=pyro_elem_fracs,
        initial_Z_elem=initial_Z_elem,
        h_virgin_sensible=_bsh(tables["cp_virgin"]),
        h_char_sensible=_bsh(tables["cp_char"]),
        material_category=material_category,
        h_g_abs_offset=h_g_abs_offset,
    )
    return card, b_prime_table


def load_materials(paths: list[str]) -> tuple[dict, dict]:
    """Load multiple material cards.

    Returns ``(cards, b_prime_tables)`` where both are dicts keyed by ``MaterialCard.name``.
    """
    cards: dict[str, MaterialCard] = {}
    b_prime_tables: dict[str, Optional[BPrimeTable]] = {}
    for p in paths:
        card, bpt = load_material(p)
        if card.name in cards:
            raise SCAMInputError(f"Duplicate material name '{card.name}' from {p}")
        cards[card.name] = card
        b_prime_tables[card.name] = bpt
    return cards, b_prime_tables
