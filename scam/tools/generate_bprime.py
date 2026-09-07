# SPDX-License-Identifier: MIT
"""Generate a SCAM-format B' YAML table from an equilibrium chemistry backend.

Two backends are supported (``--backend`` flag):

  cantera  (default) — uses the embedded ``scam.chemistry`` Cantera/CNO engine.
                    Reads a ``*_bprime_config.yaml`` file.
  mpp             — uses the Mutation++ ``bprime`` CLI subprocess.
                    Reads a ``*_mpp_config.yaml`` file.

3-D usage (fixed pyrolysis gas composition):
    python -m scam.tools.generate_bprime \\
        --config scam/materials/ablative_organic/tacot_v3.0_bprime_config.yaml \\
        --output scam/materials/ablative_organic/tacot_v3.0_bprime_air.yaml

4-D usage (variable pyrolysis gas carbon fraction for element conservation):
    python -m scam.tools.generate_bprime \\
        --config scam/materials/ablative_organic/tacot_v3.0_bprime_config.yaml \\
        --output scam/materials/ablative_organic/tacot_bprime_air_4d.yaml \\
        --z-c-pyro "0.15,0.25,0.35,0.50,0.65" \\
        [--workers 4]

Mutation++ backend:
    python -m scam.tools.generate_bprime \\
        --backend mpp \\
        --config scam/materials/ablative_organic/tacot_v3.0_mpp_config.yaml \\
        --output scam/materials/ablative_organic/tacot_v3.0_bprime_mpp_air.yaml
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Pyrolysis gas composition helper for 4-D sweeps
# ---------------------------------------------------------------------------

_CORRECT_TACOT_PYRO_X = "CH4:0.5551,CO:0.2418,H2O:0.2031"


def _base_pyro_x_from_config(cfg: dict) -> str:
    """Extract the nominal pyrolysis gas mole fractions from a bprime config."""
    g = cfg.get("generator", cfg)
    preset = g.get("preset", "")
    if preset == "tacot":
        return _CORRECT_TACOT_PYRO_X
    return g.get("pyro_x") or g.get("pyro_y") or _CORRECT_TACOT_PYRO_X


def _pyro_x_for_z_c(z_c: float, base_pyro_x: str) -> str:
    """Return a Cantera mole-fraction string for pyrolysis gas at carbon mass fraction z_c."""
    from scam.physics.bprime_evaluator import _pyro_x_from_z_c
    return _pyro_x_from_z_c(z_c, base_pyro_x)


# ---------------------------------------------------------------------------
# Cantera backend (uses embedded scam.chemistry)
# ---------------------------------------------------------------------------

def _build_cantera_namespace(cfg: dict, workers: int) -> argparse.Namespace:
    """Build an argparse.Namespace suitable for scam.chemistry.tables.build_table."""
    g = cfg.get("generator", cfg)
    from scam.chemistry.grids import parse_number_grid

    preset = g.get("preset", "")
    if preset == "tacot":
        edge_x  = "O2:0.21,N2:0.79"
        pyro_x  = _CORRECT_TACOT_PYRO_X
    else:
        edge_x = g.get("edge_x") or g.get("edge_y") or "O2:0.21,N2:0.79"
        pyro_x = g.get("pyro_x") or g.get("pyro_y") or _CORRECT_TACOT_PYRO_X

    # Resolve mechanism names; _resolve_mechanism handles scam/mechanisms/ lookup
    from scam.physics.bprime_evaluator import _resolve_mechanism
    gas_mech   = _resolve_mechanism(g["gas_mechanism"])
    carbon_phs = _resolve_mechanism(g["carbon_phase"])

    temps_str = g.get("temps",     "250:4000:25")
    press_str = g.get("pressures", "101.325,1013.25,10132.5,101325")
    bg_str    = g.get("bg",        "0.0,0.1,0.25,0.5,1.0,2.0")

    return argparse.Namespace(
        gas_mechanism=gas_mech,
        gas_phase_name=g.get("gas_phase_name", ""),
        carbon_phase=carbon_phs,
        condensed_phase_name=g.get("condensed_phase_name", ""),
        edge_x=edge_x,
        edge_y=None,
        pyro_x=pyro_x,
        pyro_y=None,
        gas_moles=float(g.get("gas_moles",    1.0)),
        carbon_moles=float(g.get("carbon_moles", 100.0)),
        solver=g.get("solver", "gibbs"),
        max_steps=int(g.get("max_steps", 2000)),
        max_iter=int(g.get("max_iter",   200)),
        log_level=int(g.get("log_level", 0)),
        species_threshold=float(g.get("species_threshold", 1e-8)),
        max_species=int(g.get("max_species", 20)),
        target_element=g.get("target_element", "C"),
        surface_source_target_fraction=float(
            g.get("surface_source_target_fraction", 1.0)
        ),
        verbose=False,
        temps=temps_str,
        pressures=press_str,
        bg=bg_str,
        workers=workers,
    )


def run_cantera_native(cfg: dict, workers: int) -> list:
    """Run Cantera table generation using embedded scam.chemistry; return list[BPrimeCase]."""
    from scam.chemistry.tables import build_table
    ns = _build_cantera_namespace(cfg, workers)
    return build_table(ns)


# ---------------------------------------------------------------------------
# Mutation++ backend
# ---------------------------------------------------------------------------

def run_mpp(mpp_config_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run Mutation++ bprime sweep; return (T_axis, p_axis, Bg_axis, Bc_arr, Hw_arr)."""
    from scam.physics.mpp_evaluator import MutationppEvaluator
    ev = MutationppEvaluator.from_config(mpp_config_path)
    return ev._T_grid, ev._P_grid, ev._Bg_grid, \
           ev._itp_bc.values, ev._itp_hw.values


# ---------------------------------------------------------------------------
# CSV → SCAM YAML converter (shared)
# ---------------------------------------------------------------------------

def cases_to_scam_yaml(
    T_axis: np.ndarray,
    p_axis: np.ndarray,
    Bg_axis: np.ndarray,
    Bc_arr: np.ndarray,
    hw_arr: np.ndarray,
    output_path: Path,
    description: str = "",
    source: str = "",
    bprime_config_path: Path | None = None,
    target_element: str = "C",
    surface_source_target_fraction: float = 1.0,
) -> None:
    """Write SCAM BPrimeTable YAML from pre-built (T, p, Bg) arrays.

    Bc_arr and hw_arr must have shape (nT, nP, nBg).
    """
    nT, nP, nBg = len(T_axis), len(p_axis), len(Bg_axis)
    assert Bc_arr.shape == (nT, nP, nBg), f"Bc_arr shape {Bc_arr.shape} != ({nT},{nP},{nBg})"

    n_nan = int(np.isnan(Bc_arr).sum())
    if n_nan:
        print(f"WARNING: {n_nan} NaN entries in B_c_prime — forward-filling along T axis.")
        for i in range(1, nT):
            mask = np.isnan(Bc_arr[i])
            Bc_arr[i][mask] = Bc_arr[i - 1][mask]
            hw_arr[i][mask] = hw_arr[i - 1][mask]

    def arr_to_nested(a: np.ndarray) -> list:
        return [
            [list(round(float(v), 10) for v in a[i, j, :]) for j in range(nP)]
            for i in range(nT)
        ]

    def arr_to_nested_2d(a: np.ndarray) -> list:
        return [[round(float(a[i, j]), 10) for j in range(nP)] for i in range(nT)]

    doc = {
        "description": description or "SCAM B' table",
        "source": source,
        "T_wall_K":  [float(v) for v in T_axis],
        "p_e_Pa":    [float(v) for v in p_axis],
        "B_g_prime": [float(v) for v in Bg_axis],
        "B_c_prime": arr_to_nested(Bc_arr),
        "Bprime_surface": arr_to_nested(Bc_arr),
        "h_wall":    arr_to_nested(hw_arr),
        "target_element": target_element,
        "surface_source_target_fraction": float(surface_source_target_fraction),
    }

    # Optionally compute surface enthalpies h_g(T,p), h_c(T,p) via BprimeEvaluator.
    # Use the explicit nominal elemental fraction so this follows the same
    # molecular-composition reconstruction as live validation runs.
    if bprime_config_path is not None:
        print(f"Computing surface enthalpies h_g, h_c at {nT}×{nP} grid points ...")
        from scam.physics.bprime_evaluator import BprimeEvaluator
        ev = BprimeEvaluator.from_config(bprime_config_path)
        target_for_enthalpy = getattr(ev, "target_element", "C")
        zc_nominal = ev.pyrolysis_target_fraction(target_for_enthalpy)

        hg_arr = np.full((nT, nP), np.nan)
        hc_arr = np.full((nT, nP), np.nan)
        n_done = 0
        for i, T in enumerate(T_axis):
            for j, p in enumerate(p_axis):
                try:
                    hg_arr[i, j], hc_arr[i, j] = ev.surface_enthalpies(
                        T, p, zc_nominal,
                    )
                except Exception:
                    pass
                n_done += 1
                if n_done % 100 == 0 or n_done == nT * nP:
                    print(f"  {n_done}/{nT*nP} surface-enthalpy points done")

        n_nan_hg = int(np.isnan(hg_arr).sum())
        if n_nan_hg:
            print(f"WARNING: {n_nan_hg} NaN h_g/h_c entries — forward-filling.")
            for i in range(1, nT):
                for arr in (hg_arr, hc_arr):
                    mask = np.isnan(arr[i])
                    arr[i][mask] = arr[i - 1][mask]

        doc["h_g"] = arr_to_nested_2d(hg_arr)
        doc["h_c"] = arr_to_nested_2d(hc_arr)
        doc["surface_enthalpy_target_element"] = target_for_enthalpy
        doc["surface_enthalpy_target_fraction"] = float(zc_nominal)
        if target_for_enthalpy == "C":
            doc["surface_enthalpy_Z_C_pyro"] = float(zc_nominal)
        print("Surface enthalpies stored in table.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(doc, f, default_flow_style=None, sort_keys=False, width=120)
    print(f"SCAM B' YAML written → {output_path}")


def csv_to_scam_yaml(
    csv_path: Path,
    output_path: Path,
    description: str = "",
    source: str = "",
    bprime_config_path: Path | None = None,
    ceat_dir: Path | None = None,   # kept for backward compat, unused
) -> None:
    """Convert a BPrimeCase long-form CSV to a SCAM BPrimeTable YAML.

    This thin wrapper reads the CSV and delegates to ``cases_to_scam_yaml``.
    It exists for backward compatibility with callers that already produced a CSV.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    required = {"T_wall_K", "pressure_Pa", "Bg", "Bprime_c_eq", "h_wall_gas_J_kg"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    if "converged" in df.columns:
        n_bad = (~df["converged"]).sum()
        if n_bad:
            print(f"WARNING: {n_bad} unconverged points dropped.")
        df = df[df["converged"]]

    T_axis  = np.array(sorted(df["T_wall_K"].unique()),    dtype=float)
    p_axis  = np.array(sorted(df["pressure_Pa"].unique()), dtype=float)
    Bg_axis = np.array(sorted(df["Bg"].unique()),          dtype=float)
    nT, nP, nBg = len(T_axis), len(p_axis), len(Bg_axis)

    T_idx  = {v: i for i, v in enumerate(T_axis)}
    p_idx  = {v: i for i, v in enumerate(p_axis)}
    Bg_idx = {v: i for i, v in enumerate(Bg_axis)}

    Bc_arr = np.full((nT, nP, nBg), np.nan)
    hw_arr = np.full((nT, nP, nBg), np.nan)
    for row in df.itertuples(index=False):
        i = T_idx.get(row.T_wall_K)
        j = p_idx.get(row.pressure_Pa)
        k = Bg_idx.get(row.Bg)
        if i is not None and j is not None and k is not None:
            bprime_value = (
                row.Bprime_eq
                if "Bprime_eq" in df.columns and not np.isnan(row.Bprime_eq)
                else row.Bprime_c_eq
            )
            Bc_arr[i, j, k] = bprime_value
            hw_arr[i, j, k] = row.h_wall_gas_J_kg

    cases_to_scam_yaml(
        T_axis, p_axis, Bg_axis, Bc_arr, hw_arr,
        output_path, description=description, source=source,
        bprime_config_path=bprime_config_path,
    )


def csv_to_scam_yaml_4d(
    csv_paths_by_zc: dict[float, Path],
    output_path: Path,
    description: str = "",
    source: str = "",
    bprime_config_path: Path | None = None,
) -> None:
    """Convert multiple BPrimeCase CSVs (one per Z_C_pyro) to a 4-D SCAM YAML."""
    import pandas as pd

    ZC_axis = np.array(sorted(csv_paths_by_zc.keys()), dtype=float)
    first_csv = csv_paths_by_zc[ZC_axis[0]]
    df0 = pd.read_csv(first_csv)
    if "converged" in df0.columns:
        df0 = df0[df0["converged"]]

    T_axis  = np.array(sorted(df0["T_wall_K"].unique()),    dtype=float)
    p_axis  = np.array(sorted(df0["pressure_Pa"].unique()),  dtype=float)
    Bg_axis = np.array(sorted(df0["Bg"].unique()),           dtype=float)
    nT, nP, nBg, nZC = len(T_axis), len(p_axis), len(Bg_axis), len(ZC_axis)
    print(f"4-D table: T({nT}) × p({nP}) × Bg({nBg}) × ZC({nZC}) = {nT*nP*nBg*nZC} points")

    Bc_arr = np.full((nT, nP, nBg, nZC), np.nan)
    hw_arr = np.full((nT, nP, nBg, nZC), np.nan)
    T_idx  = {v: i for i, v in enumerate(T_axis)}
    p_idx  = {v: i for i, v in enumerate(p_axis)}
    Bg_idx = {v: i for i, v in enumerate(Bg_axis)}

    for zc_k, csv_path in sorted(csv_paths_by_zc.items()):
        k = int(np.searchsorted(ZC_axis, zc_k))
        df = pd.read_csv(csv_path)
        if "converged" in df.columns:
            df = df[df["converged"]]
        for row in df.itertuples(index=False):
            i = T_idx.get(row.T_wall_K)
            j = p_idx.get(row.pressure_Pa)
            m = Bg_idx.get(row.Bg)
            if i is not None and j is not None and m is not None:
                Bc_arr[i, j, m, k] = row.Bprime_c_eq
                hw_arr[i, j, m, k] = row.h_wall_gas_J_kg

    n_nan = int(np.isnan(Bc_arr).sum())
    if n_nan:
        print(f"WARNING: {n_nan} NaN entries — forward-filling along T axis.")
        for i in range(1, nT):
            mask = np.isnan(Bc_arr[i])
            Bc_arr[i][mask] = Bc_arr[i - 1][mask]
            hw_arr[i][mask] = hw_arr[i - 1][mask]

    def arr_to_nested_4d(a: np.ndarray) -> list:
        return [
            [
                [list(round(float(v), 10) for v in a[i, j, m, :]) for m in range(nBg)]
                for j in range(nP)
            ]
            for i in range(nT)
        ]

    def arr_to_nested_3d(a: np.ndarray) -> list:
        return [
            [list(round(float(v), 10) for v in a[i, j, :]) for j in range(nP)]
            for i in range(nT)
        ]

    def arr_to_nested_2d(a: np.ndarray) -> list:
        return [[round(float(a[i, j]), 10) for j in range(nP)] for i in range(nT)]

    doc = {
        "description": description or "4-D TACOT B' table with Z_C_pyro axis",
        "source": source or "Cantera equilibrium thermochemistry (scam.chemistry)",
        "axes": {
            "T_wall_K":  [float(v) for v in T_axis],
            "p_e_Pa":    [float(v) for v in p_axis],
            "B_g_prime": [float(v) for v in Bg_axis],
            "Z_C_pyro":  [float(v) for v in ZC_axis],
        },
        "B_c_prime": arr_to_nested_4d(Bc_arr),
        "h_wall":    arr_to_nested_4d(hw_arr),
    }

    if bprime_config_path is not None:
        print(
            "Computing surface enthalpies h_g(T,p,Z_C_pyro) and h_c(T,p) "
            f"at {nT}×{nP}×{nZC} grid points ..."
        )
        from scam.physics.bprime_evaluator import BprimeEvaluator
        ev = BprimeEvaluator.from_config(bprime_config_path)
        hg_arr = np.full((nT, nP, nZC), np.nan)
        hc_arr = np.full((nT, nP), np.nan)
        for i, T in enumerate(T_axis):
            for j, p in enumerate(p_axis):
                for k, zc in enumerate(ZC_axis):
                    hg, hc = ev.surface_enthalpies(T, p, float(zc))
                    hg_arr[i, j, k] = hg
                    hc_arr[i, j] = hc
        doc["h_g"] = arr_to_nested_3d(hg_arr)
        doc["h_c"] = arr_to_nested_2d(hc_arr)
        print("Composition-dependent surface enthalpies stored in table.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(doc, f, default_flow_style=None, sort_keys=False, width=120)
    print(f"SCAM 4-D B' YAML written → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate a SCAM B' YAML table from a chemistry backend."
    )
    parser.add_argument(
        "--backend", choices=["cantera", "mpp"], default="cantera",
        help="Chemistry backend: 'cantera' uses scam.chemistry (Cantera CNO), "
             "'mpp' uses Mutation++ bprime CLI. Default: cantera.",
    )
    parser.add_argument(
        "--config", required=True, metavar="YAML",
        help="Backend config YAML (e.g. tacot_v3.0_bprime_config.yaml or tacot_v3.0_mpp_config.yaml)",
    )
    parser.add_argument(
        "--output", required=True, metavar="YAML",
        help="Destination path for the SCAM-format B' YAML",
    )
    parser.add_argument(
        "--workers", type=int, default=1, metavar="N",
        help="Number of parallel Cantera workers (cantera backend only, default: 1)",
    )
    parser.add_argument(
        "--description", default="", metavar="STR",
        help="Optional description string written into the output YAML header",
    )
    parser.add_argument(
        "--z-c-pyro", default="", metavar="LIST",
        help=(
            "Comma-separated Z_C_pyro values for a 4-D table sweep "
            "(e.g. '0.15,0.25,0.35,0.50,0.65').  cantera backend only."
        ),
    )
    # Backward compat alias
    parser.add_argument("--ceat-config", dest="config_alias", default=None,
                        help=argparse.SUPPRESS)

    args = parser.parse_args(argv)

    # Support old --ceat-config flag
    if args.config_alias and not args.config:
        args.config = args.config_alias

    config_path = Path(args.config).resolve()
    output_path = Path(args.output).resolve()

    if not config_path.exists():
        parser.error(f"Config not found: {config_path}")

    # ── Mutation++ backend ─────────────────────────────────────────────────
    if args.backend == "mpp":
        print(f"Backend: Mutation++ bprime CLI")
        print(f"Config:  {config_path}")

        from scam.physics.mpp_evaluator import MutationppEvaluator
        ev = MutationppEvaluator.from_config(config_path)

        # Extract arrays from the already-built interpolators
        T_axis  = ev._T_grid
        p_axis  = ev._P_grid
        Bg_axis = ev._Bg_grid
        Bc_arr  = ev._itp_bc.values.copy()
        hw_arr  = ev._itp_hw.values.copy()

        cases_to_scam_yaml(
            T_axis, p_axis, Bg_axis, Bc_arr, hw_arr,
            output_path,
            description=args.description or "Mutation++ B' table (NASA-9 thermo)",
            source="Mutation++ bprime CLI",
        )
        return

    # ── Cantera backend ────────────────────────────────────────────────────
    print(f"Backend: scam.chemistry (Cantera)")
    print(f"Config:  {config_path}")

    with open(config_path, encoding="utf-8") as f:
        cfg_base = yaml.safe_load(f)

    # Parse optional Z_C_pyro sweep
    zc_values: list[float] = []
    if args.z_c_pyro.strip():
        try:
            zc_values = [float(x.strip()) for x in args.z_c_pyro.split(",") if x.strip()]
        except ValueError as exc:
            parser.error(f"Invalid --z-c-pyro value: {exc}")

    if not zc_values:
        # 3-D mode
        print(f"Running 3-D Cantera table generation ({args.workers} worker(s)) ...")
        cases = run_cantera_native(cfg_base, args.workers)
        gen_cfg = cfg_base.get("generator", cfg_base)
        target_element = gen_cfg.get("target_element", "C")
        surface_source_target_fraction = float(
            gen_cfg.get("surface_source_target_fraction", 1.0)
        )

        T_vals  = sorted({c.T_wall_K    for c in cases})
        p_vals  = sorted({c.pressure_Pa for c in cases})
        Bg_vals = sorted({c.Bg          for c in cases})
        nT, nP, nBg = len(T_vals), len(p_vals), len(Bg_vals)

        T_idx  = {v: i for i, v in enumerate(T_vals)}
        p_idx  = {v: i for i, v in enumerate(p_vals)}
        Bg_idx = {v: i for i, v in enumerate(Bg_vals)}

        Bc_arr = np.full((nT, nP, nBg), np.nan)
        hw_arr = np.full((nT, nP, nBg), np.nan)
        for c in cases:
            i = T_idx.get(c.T_wall_K)
            j = p_idx.get(c.pressure_Pa)
            k = Bg_idx.get(c.Bg)
            if i is not None and j is not None and k is not None:
                Bc_arr[i, j, k] = c.Bprime_eq
                hw_arr[i, j, k] = c.h_wall_gas_J_kg

        cases_to_scam_yaml(
            np.array(T_vals), np.array(p_vals), np.array(Bg_vals),
            Bc_arr, hw_arr,
            output_path,
            description=args.description or "Cantera B' table (scam.chemistry)",
            source="scam.chemistry equilibrium thermochemistry (Cantera)",
            bprime_config_path=config_path,
            target_element=target_element,
            surface_source_target_fraction=surface_source_target_fraction,
        )

    else:
        # 4-D mode: one Cantera run per Z_C_pyro value — produces intermediate CSVs
        import copy
        import tempfile

        base_pyro_x = _base_pyro_x_from_config(cfg_base)
        csv_paths: dict[float, Path] = {}

        for zc in zc_values:
            print(f"\n--- Running Cantera for Z_C_pyro = {zc:.3f} ---")
            pyro_x = _pyro_x_for_z_c(zc, base_pyro_x)
            print(f"    pyro_x = {pyro_x}")

            cfg_zc = copy.deepcopy(cfg_base)
            g = cfg_zc.setdefault("generator", {})
            g["pyro_x"] = pyro_x
            g.pop("preset", None)

            cases_zc = run_cantera_native(cfg_zc, args.workers)

            tmpdir  = Path(tempfile.mkdtemp(prefix="scam_bprime_"))
            csv_out = tmpdir / f"bprime_zc{zc:.3f}.csv"

            from scam.chemistry.tables import write_long_csv
            from scam.chemistry.models import BPrimeCase
            write_long_csv(cases_zc, csv_out)
            csv_paths[zc] = csv_out

        csv_to_scam_yaml_4d(
            csv_paths, output_path,
            description=args.description or "4-D Cantera B' table (scam.chemistry)",
            bprime_config_path=config_path,
        )


if __name__ == "__main__":
    main()
