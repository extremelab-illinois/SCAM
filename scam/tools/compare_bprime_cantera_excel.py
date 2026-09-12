# SPDX-License-Identifier: MIT
"""Compare Cantera-generated B' values against the TACOT 3.0 reference Excel B' table.

The Excel-derived file (``tacot_bprime_from_ref_TACOT_3.0.xls.yaml``) is a
tab-separated text file with Fortran D-notation floats and columns:

    p (bar) \\t B'g \\t Temp (K) \\t B'c \\t Hw (kJ/kg)

**Usage**::

    python scam/tools/compare_bprime_cantera_excel.py --bg-plot 0.5 1.0

Options
-------

- ``--bg-plot B [B ...]``: B'g values to evaluate and plot. If omitted all
  B'g values in the file are used (slow).
- ``--workers N``: parallel Cantera workers (default: 4).
- ``--output FILE``: output PNG (default:
  ``<script_dir>/compare_bprime_cantera_excel.png``).
- ``--p-atm P [P ...]``: pressures to plot in atm (default: all found in file).
- ``--bg-floor FLOOR``: floor for B'g=0 when calling Cantera (default: 1e-6).
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

BPRIME_CONFIG = REPO / "scam/materials/ablative_organic/tacot_v3.0_bprime_config.yaml"
EXCEL_FILE  = REPO / "scam/materials/ablative_organic/tacot_bprime_from_ref_TACOT_3.0.xls.yaml"


def _parse_excel_table(path: Path) -> dict[tuple, list]:
    """Return groups[(p_Pa, Bg)] = sorted list of (T_K, Bc, hw_J_kg)."""
    groups: dict[tuple, list] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("p"):
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            try:
                p_bar, bg, T, bc, hw_kJkg = [
                    float(x.replace("D", "e").replace("d", "e")) for x in parts[:5]
                ]
            except ValueError:
                continue
            groups[(p_bar * 1e5, bg)].append((T, bc, hw_kJkg * 1e3))
    for key in groups:
        groups[key].sort()
    return dict(groups)


def _cantera_worker(args: tuple) -> tuple:
    """Evaluate one (T, p, Bg) point; returns (T, p, Bg, Bc, hw)."""
    T, p, Bg, config_path, bg_floor = args
    sys.path.insert(0, str(REPO))
    from scam.physics.bprime_evaluator import BprimeEvaluator
    ev = BprimeEvaluator.from_config(config_path)
    try:
        Bc, hw = ev.lookup(T, p, max(Bg, bg_floor))
    except Exception:
        Bc, hw = float("nan"), float("nan")
    return (T, p, Bg, Bc, hw)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bg-plot",  type=float, nargs="+", dest="bg_plot", default=None,
                    help="B'g values to compute and plot (omit = all, slow)")
    ap.add_argument("--workers",  type=int, default=4)
    ap.add_argument("--output",   type=str,
                    default=str(Path(__file__).parent / "compare_bprime_cantera_excel.png"))
    ap.add_argument("--p-atm",    type=float, nargs="+", default=None, dest="p_atm")
    ap.add_argument("--bg-floor", type=float, default=1e-6, dest="bg_floor")
    return ap.parse_args()


def main():
    args = parse_args()

    print(f"Parsing {EXCEL_FILE.name} ...")
    groups = _parse_excel_table(EXCEL_FILE)
    all_p  = sorted(set(k[0] for k in groups))
    all_bg = sorted(set(k[1] for k in groups))
    print(f"  {sum(len(v) for v in groups.values())} rows, "
          f"{len(all_p)} pressures, {len(all_bg)} B'g values")

    # Select pressures
    if args.p_atm is not None:
        sel_p = sorted({min(all_p, key=lambda x: abs(x - p * 101325)) for p in args.p_atm})
    else:
        sel_p = all_p

    # Select B'g values to compute (and plot)
    if args.bg_plot is not None:
        sel_bg = sorted({min(all_bg, key=lambda x: abs(x - b)) for b in args.bg_plot})
    else:
        print("  WARNING: no --bg-plot given; evaluating all B'g values (slow)")
        sel_bg = all_bg

    print(f"  Pressures: {[f'{p/101325:.4g} atm' for p in sel_p]}")
    print(f"  B'g:       {sel_bg}")

    # Build Cantera jobs
    jobs, job_keys = [], []
    for p in sel_p:
        for Bg in sel_bg:
            if (p, Bg) not in groups:
                continue
            for i, (T, _, _) in enumerate(groups[(p, Bg)]):
                jobs.append((float(T), float(p), float(Bg), str(BPRIME_CONFIG), args.bg_floor))
                job_keys.append((p, Bg, i))

    print(f"\nTotal Cantera evaluations: {len(jobs)}")
    print(f"Running Cantera ({args.workers} workers) ...")
    t0 = time.time()

    # cantera[(p, Bg)][i] = (Bc, hw)
    cantera: dict[tuple, dict[int, tuple]] = defaultdict(dict)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_cantera_worker, job): jk for job, jk in zip(jobs, job_keys)}
        n_done = 0
        for fut in as_completed(futs):
            p, Bg, i = futs[fut]
            _, _, _, Bc, hw = fut.result()
            cantera[(p, Bg)][i] = (Bc, hw)
            n_done += 1
            if n_done % 200 == 0 or n_done == len(jobs):
                elapsed = time.time() - t0
                eta = (len(jobs) - n_done) / max(n_done / max(elapsed, 1e-9), 1e-9)
                print(f"  {n_done}/{len(jobs)}  ({elapsed:.0f} s, ETA {eta:.0f} s)")

    print(f"Cantera done in {time.time() - t0:.1f} s.\n")

    # Summary
    print("--- Comparison summary ---")
    print(f"  {'p (atm)':>8}  {'B_g':>7}  {'N':>5}"
          f"  {'Bc|max':>8}  {'Bc|mean':>8}  {'hw|max MJ':>10}  {'hw|mean MJ':>11}")
    for p in sel_p:
        for Bg in sel_bg:
            key = (p, Bg)
            if key not in cantera:
                continue
            pts = groups[key]
            res = cantera[key]
            Bc_xl = np.array([pts[i][1] for i in sorted(res)])
            hw_xl = np.array([pts[i][2] for i in sorted(res)])
            Bc_ct = np.array([res[i][0] for i in sorted(res)])
            hw_ct = np.array([res[i][1] for i in sorted(res)])
            ok = np.isfinite(Bc_ct) & np.isfinite(Bc_xl)
            N = ok.sum()
            if N == 0:
                continue
            dBc = np.abs(Bc_ct[ok] - Bc_xl[ok])
            dhw = np.abs(hw_ct[ok] - hw_xl[ok]) / 1e6
            print(f"  {p/101325:>8.4f}  {Bg:>7.3f}  {N:>5}"
                  f"  {dBc.max():>8.4f}  {dBc.mean():>8.4f}"
                  f"  {dhw.max():>9.3f}    {dhw.mean():>9.3f}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    colors = cm.tab10(np.linspace(0, 0.9, len(sel_bg)))
    fig, axes = plt.subplots(4, len(sel_p), figsize=(4.5 * len(sel_p), 14), squeeze=False)
    fig.suptitle(
        f"TACOT 3.0 B' — Cantera vs Excel ({EXCEL_FILE.name})\nsolid = Cantera   □ = Excel",
        fontsize=10,
    )

    for ip, p in enumerate(sel_p):
        axes[0, ip].set_title(f"p = {p/101325:.4g} atm", fontsize=9)
        axes[0, ip].set_ylabel("B'_c  [-]",           fontsize=8)
        axes[0, ip].set_yscale("log")
        axes[1, ip].set_ylabel("ΔB'_c (Cantera−Excel)",  fontsize=8)
        axes[2, ip].set_ylabel("h_wall  [MJ/kg]",      fontsize=8)
        axes[3, ip].set_ylabel("Δh_wall  [MJ/kg]",     fontsize=8)
        axes[3, ip].set_xlabel("T_wall  [K]",           fontsize=8)

        for ib, Bg in enumerate(sel_bg):
            key = (p, Bg)
            if key not in cantera:
                continue
            pts = groups[key]
            res = cantera[key]
            idx = sorted(res)
            T   = np.array([pts[i][0] for i in idx])
            Bc_xl = np.array([pts[i][1] for i in idx])
            hw_xl = np.array([pts[i][2] for i in idx]) / 1e6
            Bc_ct = np.array([res[i][0] for i in idx])
            hw_ct = np.array([res[i][1] for i in idx]) / 1e6
            c   = colors[ib]
            lbl = f"B'g={Bg:.3g}"

            ok_bc = np.isfinite(Bc_ct) & (Bc_ct > 0)
            ok_hw = np.isfinite(hw_ct)

            axes[0, ip].plot(T[ok_bc], Bc_ct[ok_bc], "-", color=c, lw=1.8, label=lbl)
            axes[0, ip].plot(T, Bc_xl, "s", color=c, ms=4, markerfacecolor="none", markeredgewidth=0.8)
            axes[1, ip].plot(T[ok_bc & np.isfinite(Bc_xl)],
                             (Bc_ct - Bc_xl)[ok_bc & np.isfinite(Bc_xl)], "-", color=c, lw=1.5, label=lbl)
            axes[2, ip].plot(T[ok_hw], hw_ct[ok_hw], "-", color=c, lw=1.8)
            axes[2, ip].plot(T, hw_xl, "s", color=c, ms=4, markerfacecolor="none", markeredgewidth=0.8)
            axes[3, ip].plot(T[ok_hw & np.isfinite(hw_xl)],
                             (hw_ct - hw_xl)[ok_hw & np.isfinite(hw_xl)], "-", color=c, lw=1.5)

        for ax in axes[:, ip]:
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
        axes[1, ip].axhline(0, color="k", lw=0.8, ls=":")
        axes[3, ip].axhline(0, color="k", lw=0.8, ls=":")
        axes[0, ip].legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {args.output}")


if __name__ == "__main__":
    main()
