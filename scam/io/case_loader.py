# SPDX-License-Identifier: MIT
"""YAML case deck → SimCase dataclass.

The YAML case deck is the primary human-facing input format.  This module
parses it into the typed Python dataclasses that the solver consumes.

Example deck (minimal)::

    name: "my_case"
    geometry:
      type: slab
    stack:
      layers:
        - material: TACOT
          thickness: 0.05
          n_nodes: 50
    surface_bc:
      type: energy_balance
      alpha_conv: 5000.0
      T_aw: 8000.0
    solver:
      t_end: 120.0
      output_dt: 1.0
    materials:
      - materials/ablative_organic/tacot_v3.0.yaml
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scam.config.boundary import (
    BackBCConfig, BackBCType,
    SurfaceBCConfig, SurfaceBCType,
)
from scam.config.case import SimCase
from scam.config.geometry import GeometryConfig, GeometryType
from scam.config.solver import SolverOptions
from scam.config.stack import LayerConfig, StackConfig
from scam.core.errors import SCAMInputError
from scam.io.material_loader import load_material


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_scalar_bc(raw_val):
    """Return a constant float or a time-interpolating callable.

    Accepts:
      - float / int              → constant, returned as float
      - [[t0, v0], [t1, v1], …] → lambda that linearly interpolates at t
    """
    if isinstance(raw_val, (int, float)):
        return float(raw_val)
    pairs = list(raw_val)
    ts = np.array([p[0] for p in pairs], dtype=float)
    vs = np.array([p[1] for p in pairs], dtype=float)
    return lambda t, _ts=ts, _vs=vs: float(np.interp(t, _ts, _vs))


def _req(d: dict, key: str, section: str) -> Any:
    """Get required key from dict, raise SCAMInputError if missing."""
    if key not in d:
        raise SCAMInputError(f"[{section}] required field '{key}' missing")
    return d[key]


def _parse_geometry(raw: dict) -> GeometryConfig:
    gtype_str = raw.get("type", "slab").lower()
    gtype_map = {
        "slab": GeometryType.SLAB,
        "hollow_cylinder": GeometryType.HOLLOW_CYLINDER,
        "tabulated": GeometryType.TABULATED,
    }
    if gtype_str not in gtype_map:
        raise SCAMInputError(
            f"Unknown geometry type '{gtype_str}'. "
            f"Valid: {list(gtype_map.keys())}"
        )
    gtype = gtype_map[gtype_str]
    r_inner = float(raw.get("r_inner", 0.0))
    area_table = raw.get("area_table")
    return GeometryConfig(geometry_type=gtype, r_inner=r_inner, area_table=area_table)


def _parse_stack(raw: dict) -> StackConfig:
    layers_raw = _req(raw, "layers", "stack")
    layers = []
    for i, lr in enumerate(layers_raw):
        mat_name = _req(lr, "material", f"stack.layers[{i}]")
        thickness = float(_req(lr, "thickness", f"stack.layers[{i}]"))
        n_nodes   = int(_req(lr, "n_nodes", f"stack.layers[{i}]"))
        n_subcells = int(lr.get("n_subcells", 4))
        contact_r  = float(lr.get("contact_resistance", 0.0))
        grading    = float(lr.get("grading", 1.0))
        layers.append(LayerConfig(
            material_name=str(mat_name),
            thickness=thickness,
            n_nodes=n_nodes,
            n_subcells=n_subcells,
            contact_resistance=contact_r,
            grading=grading,
        ))
    return StackConfig(layers=layers)


def _parse_surface_bc(raw: dict) -> SurfaceBCConfig:
    type_str = raw.get("type", "energy_balance").lower()
    type_map = {
        "energy_balance": SurfaceBCType.ENERGY_BALANCE,
        "prescribed_flux": SurfaceBCType.PRESCRIBED_FLUX,
        "prescribed_temp": SurfaceBCType.PRESCRIBED_TEMP,
    }
    if type_str not in type_map:
        raise SCAMInputError(
            f"Unknown surface_bc type '{type_str}'. Valid: {list(type_map.keys())}"
        )
    bc_type = type_map[type_str]

    kwargs: dict[str, Any] = dict(bc_type=bc_type)
    for key in ("alpha_conv", "T_aw", "rhoUeCH", "h_r",
                "T_rad_in", "rho_e_u_e", "C_M", "p_e"):
        if key in raw:
            kwargs[key] = _make_scalar_bc(raw[key])
    for key in ("emissivity", "view_factor", "lambda_blowing"):
        if key in raw:
            kwargs[key] = float(raw[key])

    if "q_prescribed" in raw:
        kwargs["q_prescribed"] = _make_scalar_bc(raw["q_prescribed"])

    if "T_prescribed" in raw:
        kwargs["T_prescribed"] = _make_scalar_bc(raw["T_prescribed"])

    return SurfaceBCConfig(**kwargs)


def _parse_back_bc(raw: dict) -> BackBCConfig:
    type_str = raw.get("type", "adiabatic").lower()
    type_map = {
        "adiabatic": BackBCType.ADIABATIC,
        "prescribed_temp": BackBCType.PRESCRIBED_TEMP,
        "prescribed_flux": BackBCType.PRESCRIBED_FLUX,
        "radiation": BackBCType.RADIATION,
    }
    if type_str not in type_map:
        raise SCAMInputError(
            f"Unknown back_bc type '{type_str}'. Valid: {list(type_map.keys())}"
        )
    bc_type = type_map[type_str]
    kwargs: dict[str, Any] = dict(bc_type=bc_type)
    if "T_back" in raw:
        kwargs["T_back"] = _make_scalar_bc(raw["T_back"])
    if "q_back" in raw:
        kwargs["q_back"] = _make_scalar_bc(raw["q_back"])
    # RADIATION back face
    if "emissivity_back" in raw:
        kwargs["emissivity_back"] = float(raw["emissivity_back"])
    if "view_factor_back" in raw:
        kwargs["view_factor_back"] = float(raw["view_factor_back"])
    if "T_env_back" in raw:
        kwargs["T_env_back"] = _make_scalar_bc(raw["T_env_back"])
    if "h_back" in raw:
        kwargs["h_back"] = _make_scalar_bc(raw["h_back"])
    if bc_type is BackBCType.RADIATION and "emissivity_back" not in raw and "h_back" not in raw:
        raise SCAMInputError(
            "back_bc type 'radiation' requires 'emissivity_back' (and/or 'h_back'); "
            "with both zero the BC is identical to 'adiabatic'."
        )
    return BackBCConfig(**kwargs)


def _parse_solver(raw: dict) -> SolverOptions:
    kwargs: dict[str, Any] = {}
    float_keys = ("t_start", "t_end", "dt_init", "dt_min", "dt_max",
                  "dt_max_dT", "dt_max_drho_frac", "seb_tol", "output_dt",
                  "node_drop_threshold", "picard_tol")
    int_keys = ("max_seb_iter", "max_picard")
    bool_keys = ("allow_recession", "use_rho_old", "element_transport",
                 "bprime_runtime", "continuous_remap")
    for k in float_keys:
        if k in raw:
            kwargs[k] = float(raw[k])
    for k in int_keys:
        if k in raw:
            kwargs[k] = int(raw[k])
    for k in bool_keys:
        if k in raw:
            kwargs[k] = bool(raw[k])
    if "tc_positions" in raw:
        kwargs["tc_positions"] = [float(x) for x in raw["tc_positions"]]
    return SolverOptions(**kwargs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_case(path: str | os.PathLike) -> tuple[SimCase, dict, dict]:
    """Parse a YAML case deck and load all referenced material files.

    Parameters
    ----------
    path:
        Path to the YAML case deck.

    Returns
    -------
    (case, mat_cards, b_prime_tables)
        case:           SimCase dataclass
        mat_cards:      {name: MaterialCard} dict
        b_prime_tables: {name: BPrimeTable | None} dict
    """
    path = Path(path).resolve()
    case_dir = path.parent

    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if raw is None:
        raise SCAMInputError(f"Empty YAML case file: {path}")

    name     = raw.get("name", path.stem)
    geom_raw = raw.get("geometry", {})
    stack_raw = _req(raw, "stack", "top-level")
    sbc_raw  = raw.get("surface_bc", {})
    bbc_raw  = raw.get("back_bc", {})
    sol_raw  = raw.get("solver", {})
    out_raw  = raw.get("output", {})

    geometry   = _parse_geometry(geom_raw)
    stack      = _parse_stack(stack_raw)
    surface_bc = _parse_surface_bc(sbc_raw)
    back_bc    = _parse_back_bc(bbc_raw)
    options    = _parse_solver(sol_raw)
    output_path   = out_raw.get("path", "results/")
    output_format = out_raw.get("format", "csv")

    # Resolve material file paths relative to the case deck directory
    mat_paths_raw = raw.get("materials", [])
    material_paths = [str(case_dir / p) for p in mat_paths_raw]

    # Load materials
    mat_cards: dict = {}
    b_prime_tables: dict = {}
    for mp in material_paths:
        mat, bpt = load_material(mp)
        mat_cards[mat.name] = mat
        b_prime_tables[mat.name] = bpt

    # Validate that all stack layers have a loaded material
    for layer in stack.layers:
        if layer.material_name not in mat_cards:
            raise SCAMInputError(
                f"Stack layer material '{layer.material_name}' not found in "
                f"loaded materials {list(mat_cards.keys())}. "
                f"Add the YAML path to the 'materials' list in the case deck."
            )

    case = SimCase(
        name=name,
        stack=stack,
        geometry=geometry,
        surface_bc=surface_bc,
        back_bc=back_bc,
        options=options,
        material_paths=material_paths,
        output_path=output_path,
        output_format=output_format,
    )
    return case, mat_cards, b_prime_tables
