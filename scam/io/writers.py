# SPDX-License-Identifier: MIT
"""Output writers: CSV (default) and optional HDF5.

Write Results objects to disk in two formats:

CSV (always available)
----------------------
- ``{output_path}/time_history.csv``  — per-output-step scalars
  (time, T_wall, s_total, s_dot, q_cond, m_dot_pyro, m_dot_char,
   tc_0, tc_1, ...)
- ``{output_path}/temperature_profiles.csv``  — T(y) at each saved step
  (first column = node depth [m] at t=0, subsequent columns = T [K] at each time)
- ``{output_path}/depth_profiles.csv``  — y_nodes [m] at each saved step
  (first column = node depth at t=0, subsequent columns = actual node positions
   at each snapshot; differs from temperature_profiles depth col for ALE cases)
- ``{output_path}/char_fraction_profiles.csv``  — β(y) at each saved step
  (β = (ρ_virgin − ρ) / (ρ_virgin − ρ_char); written only when stack and
   mat_cards are supplied to write_csv / write_results)

HDF5 (requires h5py, enabled via format="hdf5")
------------------------------------------------
- ``{output_path}/results.h5`` — single file with datasets:
    /time [s], /T_wall [K], /s_total [m], /s_dot [m/s],
    /q_cond [W/m2], /m_dot_pyro [kg/m2/s], /m_dot_char [kg/m2/s],
    /tc_temps [n_tc, n_steps] [K],
    /T_profiles [n_steps, n_nodes] [K]
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from scam.core.results import Results


def _ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_csv(
    results: Results,
    output_path: str | os.PathLike,
    stack=None,
    mat_cards: dict | None = None,
) -> None:
    """Write results to CSV files in output_path/.

    When *stack* and *mat_cards* are provided, also writes
    ``char_fraction_profiles.csv`` (β = (ρ_v − ρ) / (ρ_v − ρ_c) per node).
    """
    out = _ensure_dir(output_path)

    times       = results.times_array()
    T_wall      = results.T_wall_array()
    s_total     = results.s_array()
    s_dot       = np.array([s.s_dot      for s in results.snapshots])
    q_cond      = np.array([s.q_cond     for s in results.snapshots])
    m_dot_pyro  = np.array([s.m_dot_pyro for s in results.snapshots])
    m_dot_char  = np.array([s.m_dot_char for s in results.snapshots])

    # --- time_history.csv ---
    tc_data = results.tc_array()   # (n_tc, n_steps) or None
    header_cols = ["time_s", "T_wall_K", "s_total_m", "s_dot_m_s",
                   "q_cond_W_m2", "m_dot_pyro_kg_m2_s", "m_dot_char_kg_m2_s"]
    data_cols = [times, T_wall, s_total, s_dot, q_cond, m_dot_pyro, m_dot_char]

    if tc_data is not None:
        n_tc = tc_data.shape[0]
        for i in range(n_tc):
            pos = results.tc_positions[i] if i < len(results.tc_positions) else i
            header_cols.append(f"tc_{i}_at_{pos*1000:.1f}mm_K")
            data_cols.append(tc_data[i])   # time series for TC i

    matrix = np.column_stack(data_cols)
    header = ",".join(header_cols)
    np.savetxt(out / "time_history.csv", matrix, delimiter=",",
               header=header, comments="", fmt="%.6e")

    # --- temperature_profiles.csv ---
    # Rows = nodes (original depth), columns = snapshots.
    # Profiles may differ in length after node drops; pad to max length.
    first_y = results.snapshots[0].mesh.y_nodes
    n_max   = max(len(s.T) for s in results.snapshots)

    padded = []
    for snap in results.snapshots:
        T_row = snap.T
        if len(T_row) < n_max:
            T_row = np.pad(T_row, (n_max - len(T_row), 0), constant_values=np.nan)
        padded.append(T_row)

    T_matrix = np.column_stack(padded)     # (n_nodes, n_steps)
    y_col = np.full(n_max, np.nan)
    y_col[:len(first_y)] = first_y
    T_matrix_with_y = np.column_stack([y_col, T_matrix])

    time_header = ",".join(["depth_m"] + [f"t={t:.3f}s" for t in times])
    np.savetxt(out / "temperature_profiles.csv", T_matrix_with_y,
               delimiter=",", header=time_header, comments="", fmt="%.6e")

    # --- depth_profiles.csv — actual y_nodes at each snapshot ---
    # Identical format to temperature_profiles but stores node positions.
    # For Lagrangian cases this is approximately first_y; for ALE it reflects
    # node redistribution and is the correct depth axis for profile animations.
    padded_y = []
    for snap in results.snapshots:
        y_snap = snap.mesh.y_nodes
        if len(y_snap) < n_max:
            y_snap = np.pad(y_snap, (n_max - len(y_snap), 0), constant_values=np.nan)
        padded_y.append(y_snap)
    y_matrix = np.column_stack(padded_y)
    np.savetxt(out / "depth_profiles.csv",
               np.column_stack([y_col, y_matrix]),
               delimiter=",", header=time_header, comments="", fmt="%.6e")

    # --- char_fraction_profiles.csv (optional) ---
    if stack is not None and mat_cards is not None:
        layer_names = [lyr.material_name for lyr in stack.layers]
        rho_v_by_layer = np.array([mat_cards[n].rho_virgin for n in layer_names])
        rho_c_by_layer = np.array([mat_cards[n].rho_char   for n in layer_names])

        padded_beta = []
        for snap in results.snapshots:
            lid   = snap.mesh.layer_id          # (n_nodes,)
            rho_v = rho_v_by_layer[lid]
            rho_c = rho_c_by_layer[lid]
            denom = rho_v - rho_c
            beta  = np.where(denom > 0, (rho_v - snap.rho) / denom, 0.0)
            beta  = np.clip(beta, 0.0, 1.0)
            if len(beta) < n_max:
                beta = np.pad(beta, (n_max - len(beta), 0), constant_values=np.nan)
            padded_beta.append(beta)

        beta_matrix = np.column_stack(padded_beta)
        beta_matrix_with_y = np.column_stack([y_col, beta_matrix])
        np.savetxt(out / "char_fraction_profiles.csv", beta_matrix_with_y,
                   delimiter=",", header=time_header, comments="", fmt="%.6e")

        # --- comp{c}_beta_profiles.csv — per-component char fraction ---
        # rho_components[l] has shape (n_comp_l, n_nodes_l, J_l); volume-weight over nodelets.
        n_comp_max = max((len(mat_cards[n].components) for n in layer_names), default=0)
        for c_idx in range(n_comp_max):
            padded_cb = []
            for snap in results.snapshots:
                n_curr = snap.mesh.n_nodes_total
                offset = n_max - n_curr  # front-pad offset (consistent with T/rho padding)
                cb_global = np.full(n_max, np.nan)
                for l_idx, lname in enumerate(layer_names):
                    rc_l = snap.rho_components[l_idx]
                    if rc_l is None or c_idx >= len(mat_cards[lname].components):
                        continue
                    comp = mat_cards[lname].components[c_idx]
                    g0, g1 = snap.mesh.layer_boundaries[l_idx]
                    nd_l = snap.mesh.nodelet_delta[l_idx]   # (n_nodes_l, J)
                    rho_c = rc_l[c_idx]                      # (n_nodes_l, J)
                    rho_c_node = (rho_c * nd_l).sum(axis=1) / nd_l.sum(axis=1)
                    denom = comp.rho_0 - comp.rho_r
                    if denom > 0:
                        bc = np.clip((comp.rho_0 - rho_c_node) / denom, 0.0, 1.0)
                    else:
                        bc = np.zeros(len(rho_c_node))
                    cb_global[offset + g0: offset + g1] = bc
                padded_cb.append(cb_global)
            cb_matrix = np.column_stack(padded_cb)
            np.savetxt(out / f"comp{c_idx}_beta_profiles.csv",
                       np.column_stack([y_col, cb_matrix]),
                       delimiter=",", header=time_header, comments="", fmt="%.6e")

    print(f"CSV output written to: {out}")


def write_hdf5(results: Results, output_path: str | os.PathLike) -> None:
    """Write results to a single HDF5 file.  Requires h5py."""
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required for HDF5 output. Install with: pip install h5py"
        ) from exc

    out = _ensure_dir(output_path)
    fpath = out / "results.h5"

    times       = results.times_array()
    T_wall      = results.T_wall_array()
    s_total     = results.s_array()
    s_dot       = np.array([s.s_dot      for s in results.snapshots])
    q_cond      = np.array([s.q_cond     for s in results.snapshots])
    m_dot_pyro  = np.array([s.m_dot_pyro for s in results.snapshots])
    m_dot_char  = np.array([s.m_dot_char for s in results.snapshots])
    tc_data     = results.tc_array()   # (n_tc, n_steps) or None

    with h5py.File(fpath, "w") as f:
        f.create_dataset("time",        data=times)
        f["time"].attrs["units"] = "s"
        f.create_dataset("T_wall",      data=T_wall)
        f["T_wall"].attrs["units"] = "K"
        f.create_dataset("s_total",     data=s_total)
        f["s_total"].attrs["units"] = "m"
        f.create_dataset("s_dot",       data=s_dot)
        f["s_dot"].attrs["units"] = "m/s"
        f.create_dataset("q_cond",      data=q_cond)
        f["q_cond"].attrs["units"] = "W/m2"
        f.create_dataset("m_dot_pyro",  data=m_dot_pyro)
        f["m_dot_pyro"].attrs["units"] = "kg/m2/s"
        f.create_dataset("m_dot_char",  data=m_dot_char)
        f["m_dot_char"].attrs["units"] = "kg/m2/s"

        if tc_data is not None:
            f.create_dataset("tc_temps", data=tc_data)
            f["tc_temps"].attrs["units"] = "K"
            f["tc_temps"].attrs["shape"] = "(n_tc, n_steps)"
            if results.tc_positions:
                f["tc_temps"].attrs["tc_positions_m"] = np.array(results.tc_positions)

        first_y = results.snapshots[0].mesh.y_nodes
        n_max = max(len(s.T) for s in results.snapshots)
        padded = []
        for snap in results.snapshots:
            row = snap.T
            if len(row) < n_max:
                row = np.pad(row, (n_max - len(row), 0), constant_values=np.nan)
            padded.append(row)
        f.create_dataset("T_profiles", data=np.array(padded))
        f["T_profiles"].attrs["units"] = "K"
        f["T_profiles"].attrs["shape"] = "(n_steps, n_nodes)"
        f.create_dataset("initial_depth_m", data=first_y)

    print(f"HDF5 output written to: {fpath}")


def write_results(
    results: Results,
    output_path: str | os.PathLike,
    fmt: str = "csv",
    stack=None,
    mat_cards: dict | None = None,
) -> None:
    """Write results in the requested format.

    Parameters
    ----------
    results:
        Results object from ``run()``.
    output_path:
        Directory where output files will be written.
    fmt:
        ``"csv"`` (default) or ``"hdf5"``.
    stack:
        StackConfig from the case.  When provided together with *mat_cards*,
        enables ``char_fraction_profiles.csv`` output (CSV only).
    mat_cards:
        ``{name: MaterialCard}`` dict.  See *stack*.
    """
    fmt = fmt.lower()
    if fmt == "csv":
        write_csv(results, output_path, stack=stack, mat_cards=mat_cards)
    elif fmt in ("hdf5", "h5"):
        write_hdf5(results, output_path)
    else:
        raise ValueError(f"Unknown output format '{fmt}'. Valid: 'csv', 'hdf5'")
