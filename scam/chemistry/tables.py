from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

from .grids import parse_number_grid
from .models import BPrimeCase
from .thermochemistry import compute_bprime_case, require_cantera


# ---------------------------------------------------------------------------
# Table builder
# ---------------------------------------------------------------------------

def build_table(args: argparse.Namespace) -> list[BPrimeCase]:
    import time as _time

    temperatures = parse_number_grid(args.temps, name="temperature")
    pressures    = parse_number_grid(args.pressures, name="pressure")
    bg_values    = parse_number_grid(args.bg, name="B'g")

    n_total = len(pressures) * len(bg_values) * len(temperatures)
    workers = int(args.workers)
    if workers < 1:
        raise ValueError("--workers must be at least 1")

    # Build the flat list of per-point keyword dicts (no Solution objects —
    # those are created inside each worker via _worker_init).
    base_kwargs = dict(
        gas_mechanism=args.gas_mechanism,
        gas_phase_name=getattr(args, "gas_phase_name", ""),
        carbon_phase_file=args.carbon_phase,
        condensed_phase_name=getattr(args, "condensed_phase_name", ""),
        edge_x=args.edge_x,
        edge_y=args.edge_y,
        pyro_x=args.pyro_x,
        pyro_y=args.pyro_y,
        initial_gas_moles=args.gas_moles,
        initial_carbon_moles=args.carbon_moles,
        solver=args.solver,
        max_steps=args.max_steps,
        max_iter=args.max_iter,
        log_level=args.log_level,
        species_threshold=args.species_threshold,
        max_species=args.max_species,
        target_element=getattr(args, "target_element", "C"),
        surface_source_target_fraction=getattr(
            args, "surface_source_target_fraction", 1.0
        ),
    )
    all_kwargs = [
        {**base_kwargs, "temperature_K": t, "pressure_Pa": p, "bg": bg}
        for p in pressures
        for bg in bg_values
        for t in temperatures
    ]

    t0 = _time.perf_counter()
    n_workers_actual = min(workers, n_total)
    mode = "serial" if n_workers_actual == 1 else f"{n_workers_actual} workers"
    print(f"Computing {n_total} points ({mode}) …", file=sys.stderr)

    results: list[dict | None] = []
    errors: list[str] = []

    if n_workers_actual == 1:
        ct = require_cantera()
        # ── Single-process path: create Solution objects once and reuse ──────
        # Apply continuation ordering so physically similar points are adjacent
        # and the implicit warm-start (reusing last-equilibrated Solution) helps.
        from .mat.solver import sort_grid_for_continuation
        all_kwargs = sort_grid_for_continuation(all_kwargs)

        gas_phase_name = getattr(args, "gas_phase_name", "")
        condensed_phase_name = getattr(args, "condensed_phase_name", "")

        gas = (
            ct.Solution(args.gas_mechanism, gas_phase_name)
            if gas_phase_name
            else ct.Solution(args.gas_mechanism)
        )

        carbon = (
            ct.Solution(args.carbon_phase, condensed_phase_name)
            if condensed_phase_name
            else ct.Solution(args.carbon_phase)
        )
        for i, kw in enumerate(all_kwargs, 1):
            if args.verbose:
                print(
                    f"  [{i:6d}/{n_total:6d}]  T={kw['temperature_K']:.1f} K"
                    f"  p={kw['pressure_Pa']:.5g} Pa  B'g={kw['bg']:.4g}",
                    file=sys.stderr, end="",
                )
            try:
                row = compute_bprime_case(**kw, gas=gas, carbon=carbon)
                results.append(asdict(row))
                if args.verbose:
                    print("", file=sys.stderr)
            except Exception as exc:
                err = str(exc)
                errors.append(f"T={kw['temperature_K']:.1f} K p={kw['pressure_Pa']:.4g} Pa B'g={kw['bg']:.4g}: {err}")
                print(f"  SKIPPED: {err}", file=sys.stderr)
    else:
        # ── Multi-process path: pool of workers, each with its own Solutions ──
        # Sort once so each chunk covers a contiguous, physically coherent slice
        # of the grid (same ordering as the serial path). Workers then process
        # their chunk serially, enabling implicit warm-starting between adjacent
        # points. This also reduces IPC pickling from n_total round-trips to
        # n_workers round-trips.
        from .mat.solver import sort_grid_for_continuation
        all_kwargs = sort_grid_for_continuation(all_kwargs)
        chunks = list(_split_chunks(all_kwargs, n_workers_actual))

        with ProcessPoolExecutor(
            max_workers=n_workers_actual,
            initializer=_worker_init,
            initargs=(
                args.gas_mechanism,
                getattr(args, "gas_phase_name", ""),
                args.carbon_phase,
                getattr(args, "condensed_phase_name", ""),
            ),
        ) as pool:
            futures = [pool.submit(_worker_compute_batch, chunk) for chunk in chunks]
            n_done = 0
            for fut in futures:
                for result, err in fut.result():
                    n_done += 1
                    if result is not None:
                        results.append(result)
                    else:
                        errors.append(err or "unknown error")
                    if args.verbose and n_done % max(1, n_total // 50) == 0:
                        print(f"  [{n_done:6d}/{n_total:6d}]  …", file=sys.stderr)

    dt = _time.perf_counter() - t0
    n_ok = len(results)
    n_skip = len(errors)
    print(
        f"Done: {n_ok} points in {dt:.1f}s ({dt/max(n_ok,1)*1000:.1f} ms/point)"
        + (f"  — {n_skip} skipped" if n_skip else ""),
        file=sys.stderr,
    )
    if errors:
        print(f"WARNING: {n_skip}/{n_total} points skipped due to convergence failures.", file=sys.stderr)

    rows = [BPrimeCase(**r) for r in results]
    rows.sort(key=lambda r: (r.pressure_Pa, r.Bg, r.T_wall_K))
    return rows

_WORKER_GAS = None
_WORKER_CARBON = None


def _worker_init(
    gas_mechanism: str,
    gas_phase_name: str,
    carbon_phase_file: str,
    condensed_phase_name: str,
) -> None:
    global _WORKER_GAS, _WORKER_CARBON

    ct = require_cantera()

    _WORKER_GAS = (
        ct.Solution(gas_mechanism, gas_phase_name)
        if gas_phase_name
        else ct.Solution(gas_mechanism)
    )

    _WORKER_CARBON = (
        ct.Solution(carbon_phase_file, condensed_phase_name)
        if condensed_phase_name
        else ct.Solution(carbon_phase_file)
    )

def _worker_compute(kwargs: dict) -> tuple[dict | None, str | None]:
    try:
        row = compute_bprime_case(**kwargs, gas=_WORKER_GAS, carbon=_WORKER_CARBON)
        return asdict(row), None
    except Exception as exc:
        return None, str(exc)


def _worker_compute_batch(
    batch: list[dict],
) -> list[tuple[dict | None, str | None]]:
    """Process a contiguous chunk of grid points serially within a worker.

    Passing the same Solution objects through every point lets Cantera reuse
    its internally cached Jacobian and species data, giving an implicit
    warm-start for physically adjacent points.
    """
    out = []
    for kw in batch:
        try:
            row = compute_bprime_case(**kw, gas=_WORKER_GAS, carbon=_WORKER_CARBON)
            out.append((asdict(row), None))
        except Exception as exc:
            out.append((None, str(exc)))
    return out


def _split_chunks(lst: list, n: int):
    """Yield n contiguous, roughly equal chunks from lst."""
    k, r = divmod(len(lst), n)
    i = 0
    for j in range(n):
        size = k + (1 if j < r else 0)
        yield lst[i : i + size]
        i += size


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_long_csv(rows: list[BPrimeCase], path: Path) -> None:
    """Write long-form CSV: one row per T / pressure / B'g point."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = (
        list(asdict(rows[0]).keys())
        if rows
        else list(BPrimeCase.__annotations__.keys())
    )

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(asdict(row))


def write_wide_csv(rows: list[BPrimeCase], path: Path) -> None:
    """Write wide CSV: rows are T_wall_K, columns are one per (B'g, pressure)."""
    path.parent.mkdir(parents=True, exist_ok=True)

    temps = sorted({r.T_wall_K for r in rows})
    pressures = sorted({r.pressure_Pa for r in rows})
    bg_values = sorted({r.Bg for r in rows})

    lookup = {
        (r.T_wall_K, r.pressure_Pa, r.Bg): r.Bprime_c_eq
        for r in rows
    }

    with path.open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            ["T_wall_K"]
            + [
                f"p={p:.12g}_Pa_Bg={bg:.6g}"
                for bg in bg_values
                for p in pressures
            ]
        )

        for t in temps:
            writer.writerow(
                [f"{t:.12g}"]
                + [
                    lookup.get((t, p, bg), "")
                    for bg in bg_values
                    for p in pressures
                ]
            )


def write_json(rows: list[BPrimeCase], path: Path) -> None:
    """Write table rows to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        json.dump([asdict(row) for row in rows], f, indent=2)


def _resolve_output(name: str, outdir: Path) -> Path:
    """Resolve output filename under outdir unless it is already absolute."""
    p = Path(name)
    return p if p.is_absolute() else outdir / p


def write_outputs(rows: list[BPrimeCase], args: argparse.Namespace) -> None:
    """Write all requested table outputs."""
    outdir = Path(args.outdir)

    out_path = _resolve_output(args.out, outdir)
    write_long_csv(rows, out_path)
    print(f"Wrote long-form table: {out_path}")

    if args.out_wide:
        wide_path = _resolve_output(args.out_wide, outdir)
        write_wide_csv(rows, wide_path)
        print(f"Wrote wide table: {wide_path}")

    if args.out_json:
        json_path = _resolve_output(args.out_json, outdir)
        write_json(rows, json_path)
        print(f"Wrote JSON table: {json_path}")
