# SPDX-License-Identifier: MIT
"""Shared plotting helpers for B' comparison scripts."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import RegularGridInterpolator


# ── Colours ──────────────────────────────────────────────────────────────────
C_REF  = "#d62728"   # red dashed  — XLS reference (Mutation++)
C_TBL  = "#1f77b4"   # blue        — pre-computed Cantera table
C_LIVE = "#2ca02c"   # green dots  — live Cantera evaluator


# ── TSV parser ───────────────────────────────────────────────────────────────
def _flt(s: str) -> float:
    return float(s.replace("D", "e").replace("d", "e"))


def load_ref_tsv(path: str, col_order: str) -> RegularGridInterpolator:
    """Parse a raw XLS-copy TSV and return a RegularGridInterpolator pair.

    col_order: "p_bar_bg_T_Bc_hw_kJ"  (v3.0 TSV format)
            or "p_Pa_bg_Bc_T_hw_J"    (v2.2 TSV format)

    Returns two callables: (lookup_bc, lookup_hw) where each takes (T, p_Pa, bg).
    Actually returns a single callable that returns (bc, hw).
    """
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            try:
                _flt(parts[0])
                rows.append(parts)
            except ValueError:
                pass

    if col_order == "p_bar_bg_T_Bc_hw_kJ":
        # v3.0: p(bar)  B'g  T(K)  B'c  hw(kJ/kg)
        p_vals  = sorted(set(_flt(r[0]) for r in rows))
        bg_vals = sorted(set(_flt(r[1]) for r in rows))
        T_vals  = sorted(set(_flt(r[2]) for r in rows))
        p_Pa    = [p * 1e5 for p in p_vals]
        nT, nP, nBg = len(T_vals), len(p_vals), len(bg_vals)
        bc_arr = np.empty((nT, nP, nBg))
        hw_arr = np.empty((nT, nP, nBg))
        Ti = {v: i for i, v in enumerate(T_vals)}
        Pi = {v: i for i, v in enumerate(p_vals)}
        Bi = {v: i for i, v in enumerate(bg_vals)}
        for r in rows:
            it, ip, ib = Ti[_flt(r[2])], Pi[_flt(r[0])], Bi[_flt(r[1])]
            bc_arr[it, ip, ib] = _flt(r[3])
            hw_arr[it, ip, ib] = _flt(r[4]) * 1e3
    else:
        # v2.2: p(Pa)  B'g  B'c  T(K)  hw(J/kg)
        p_Pa    = sorted(set(_flt(r[0]) for r in rows))
        bg_vals = sorted(set(_flt(r[1]) for r in rows))
        T_vals  = sorted(set(_flt(r[3]) for r in rows))
        nT, nP, nBg = len(T_vals), len(p_Pa), len(bg_vals)
        bc_arr = np.empty((nT, nP, nBg))
        hw_arr = np.empty((nT, nP, nBg))
        Ti = {v: i for i, v in enumerate(T_vals)}
        Pi = {v: i for i, v in enumerate(p_Pa)}
        Bi = {v: i for i, v in enumerate(bg_vals)}
        for r in rows:
            it, ip, ib = Ti[_flt(r[3])], Pi[_flt(r[0])], Bi[_flt(r[1])]
            bc_arr[it, ip, ib] = _flt(r[2])
            hw_arr[it, ip, ib] = _flt(r[4])

    itp_bc = RegularGridInterpolator(
        (T_vals, p_Pa, bg_vals), bc_arr,
        method="linear", bounds_error=False, fill_value=None,
    )
    itp_hw = RegularGridInterpolator(
        (T_vals, p_Pa, bg_vals), hw_arr,
        method="linear", bounds_error=False, fill_value=None,
    )
    T_range = (T_vals[0], T_vals[-1])
    bg_range = (bg_vals[0], bg_vals[-1])
    return itp_bc, itp_hw, T_range, bg_range


def make_lookup(itp_bc, itp_hw):
    """Return a lookup(T, p, bg) -> (bc, hw) callable from two interpolators."""
    def lookup(T: float, p: float, bg: float):
        pt = np.array([[T, p, bg]])
        return float(itp_bc(pt)[0]), float(itp_hw(pt)[0])
    return lookup


# ── Query helpers ─────────────────────────────────────────────────────────────
P_ATM = 101325.0  # Pa


def sweep_T(lookup_fn, T_arr, bg, p=P_ATM):
    bc = np.array([lookup_fn(float(T), p, float(bg))[0] for T in T_arr])
    hw = np.array([lookup_fn(float(T), p, float(bg))[1] for T in T_arr])
    return bc, hw


def sweep_Bg(lookup_fn, bg_arr, T, p=P_ATM):
    bc = np.array([lookup_fn(T, p, float(bg))[0] for bg in bg_arr])
    hw = np.array([lookup_fn(T, p, float(bg))[1] for bg in bg_arr])
    return bc, hw


# ── Figure builder ────────────────────────────────────────────────────────────
def make_figure(
    title: str,
    BG_COMPARE: list,
    T_SLICES: list,
    lookup_ref, lookup_tbl, lookup_live,
    T_MIN: float, T_MAX: float,
    label_ref: str, label_tbl: str, label_live: str,
):
    """Build a 4-row comparison figure. Returns (fig, axes)."""
    T_FINE = np.linspace(T_MIN, T_MAX, 300)
    T_LIVE = np.linspace(T_MIN, T_MAX, 75)
    BG_SWEEP = np.linspace(0.01, 2.0, 150)
    BG_LIVE  = np.linspace(0.01, 2.0,  40)

    # Pre-compute live Cantera (slow)
    print("Pre-computing live Cantera sweep over T…")
    live_bc_T, live_hw_T = {}, {}
    for bg in BG_COMPARE:
        live_bc_T[bg], live_hw_T[bg] = sweep_T(lookup_live, T_LIVE, bg)
        print(f"  B'g={bg} done")
    print("Pre-computing live Cantera sweep over B'g…")
    live_bc_Bg = {}
    for T_s in T_SLICES:
        live_bc_Bg[T_s], _ = sweep_Bg(lookup_live, BG_LIVE, T_s)
        print(f"  T={T_s:.0f} K done")

    N_BG, N_T = len(BG_COMPARE), len(T_SLICES)
    NCOLS = max(N_BG, N_T)
    LW = 1.5
    fig, axes = plt.subplots(4, NCOLS, figsize=(3.0 * NCOLS, 14),
                             constrained_layout=True)
    fig.suptitle(title, fontsize=9)

    # Row 0: B'c vs T
    for col, bg in enumerate(BG_COMPARE):
        ax = axes[0, col]
        bc_ref, _ = sweep_T(lookup_ref, T_FINE, bg)
        bc_tbl, _ = sweep_T(lookup_tbl, T_FINE, bg)
        bc_lv     = live_bc_T[bg]
        ax.axhline(0, color="k", lw=0.6, ls=":")
        ax.fill_between(T_LIVE, bc_lv, 0, where=(bc_lv < 0),
                        alpha=0.12, color=C_LIVE, label="deposition (live)")
        ax.plot(T_FINE, bc_ref, color=C_REF,  lw=LW, ls="--", label=label_ref)
        ax.plot(T_FINE, bc_tbl, color=C_TBL,  lw=LW,          label=label_tbl)
        ax.plot(T_LIVE, bc_lv,  color=C_LIVE, lw=LW, marker="o", ms=2.5, label=label_live)
        ax.set_title(f"B′g = {bg}")
        ax.set_xlabel("T_wall  [K]")
        if col == 0:
            ax.set_ylabel("B′c  [–]")
            ax.legend(fontsize=6.5)
        ax.set_xlim(T_MIN, T_MAX)
        ax.grid(True, alpha=0.3)
    for col in range(N_BG, NCOLS):
        axes[0, col].set_visible(False)

    # Row 1: h_wall vs T
    for col, bg in enumerate(BG_COMPARE):
        ax = axes[1, col]
        _, hw_ref = sweep_T(lookup_ref, T_FINE, bg)
        _, hw_tbl = sweep_T(lookup_tbl, T_FINE, bg)
        hw_lv = live_hw_T[bg]
        ax.plot(T_FINE, hw_ref / 1e6, color=C_REF,  lw=LW, ls="--", label=label_ref)
        ax.plot(T_FINE, hw_tbl / 1e6, color=C_TBL,  lw=LW,          label=label_tbl)
        ax.plot(T_LIVE, hw_lv  / 1e6, color=C_LIVE, lw=LW, marker="o", ms=2.5, label=label_live)
        ax.set_title(f"B′g = {bg}")
        ax.set_xlabel("T_wall  [K]")
        if col == 0:
            ax.set_ylabel("h_wall  [MJ/kg]")
            ax.legend(fontsize=6.5)
        ax.set_xlim(T_MIN, T_MAX)
        ax.grid(True, alpha=0.3)
    for col in range(N_BG, NCOLS):
        axes[1, col].set_visible(False)

    # Row 2: B'c vs B'g
    for col, T_s in enumerate(T_SLICES):
        ax = axes[2, col]
        bc_ref_bg, _ = sweep_Bg(lookup_ref, BG_SWEEP, T_s)
        bc_tbl_bg, _ = sweep_Bg(lookup_tbl, BG_SWEEP, T_s)
        bc_lv_bg = live_bc_Bg[T_s]
        ax.axhline(0, color="k", lw=0.6, ls=":")
        ax.plot(BG_SWEEP, bc_ref_bg, color=C_REF,  lw=LW, ls="--", label=label_ref)
        ax.plot(BG_SWEEP, bc_tbl_bg, color=C_TBL,  lw=LW,          label=label_tbl)
        ax.plot(BG_LIVE,  bc_lv_bg,  color=C_LIVE, lw=LW, marker="o", ms=3, label=label_live)
        ax.set_title(f"T_wall = {T_s:.0f} K")
        ax.set_xlabel("B′g  [–]")
        if col == 0:
            ax.set_ylabel("B′c  [–]")
            ax.legend(fontsize=6.5)
        ax.set_xlim(BG_SWEEP[0], BG_SWEEP[-1])
        ax.grid(True, alpha=0.3)
    for col in range(N_T, NCOLS):
        axes[2, col].set_visible(False)

    # Row 3: diagnostics — Δh_wall, ΔB'c, pressure sensitivity
    BG_DIAG   = [0.1, 0.5, 1.0, 2.0]
    colors_dh = plt.cm.plasma(np.linspace(0.15, 0.85, len(BG_DIAG)))

    AX_DH = axes[3, 0]
    for i, bg in enumerate(BG_DIAG):
        _, hw_ref = sweep_T(lookup_ref, T_LIVE, bg)
        _, hw_tbl = sweep_T(lookup_tbl, T_LIVE, bg)
        hw_lv = live_hw_T[bg]
        AX_DH.plot(T_LIVE, (hw_ref - hw_lv) / 1e6, color=C_REF,  lw=1.2, ls="--", alpha=0.85)
        AX_DH.plot(T_LIVE, (hw_tbl - hw_lv) / 1e6, color=C_TBL,  lw=1.2, ls="-",  alpha=0.85)
    AX_DH.axhline(0, color="k", lw=0.6, ls=":")
    AX_DH.legend(handles=[
        Line2D([0],[0], color=C_REF,  lw=1.5, ls="--", label=f"ref − live"),
        Line2D([0],[0], color=C_TBL,  lw=1.5,           label=f"Cantera tbl − live"),
        *[Line2D([0],[0], color=colors_dh[i], lw=1.5, label=f"B′g={bg}") for i,bg in enumerate(BG_DIAG)],
    ], fontsize=6)
    AX_DH.set_xlabel("T_wall  [K]");  AX_DH.set_ylabel("Δh_wall  [MJ/kg]  (source − live)")
    AX_DH.set_title("h_wall offset from live Cantera");  AX_DH.set_xlim(T_MIN, T_MAX)
    AX_DH.grid(True, alpha=0.3)

    AX_DBC = axes[3, 1]
    for i, bg in enumerate(BG_DIAG):
        bc_ref, _ = sweep_T(lookup_ref, T_LIVE, bg)
        bc_tbl, _ = sweep_T(lookup_tbl, T_LIVE, bg)
        bc_lv = live_bc_T[bg]
        AX_DBC.plot(T_LIVE, bc_ref - bc_lv, color=C_REF, lw=1.2, ls="--", alpha=0.85)
        AX_DBC.plot(T_LIVE, bc_tbl - bc_lv, color=C_TBL, lw=1.2, ls="-",  alpha=0.85)
    AX_DBC.axhline(0, color="k", lw=0.6, ls=":")
    AX_DBC.legend(handles=[
        Line2D([0],[0], color=C_REF,  lw=1.5, ls="--", label="ref − live"),
        Line2D([0],[0], color=C_TBL,  lw=1.5,           label="Cantera tbl − live"),
        *[Line2D([0],[0], color=colors_dh[i], lw=1.5, label=f"B′g={bg}") for i,bg in enumerate(BG_DIAG)],
    ], fontsize=6)
    AX_DBC.set_xlabel("T_wall  [K]");  AX_DBC.set_ylabel("ΔB′c  (source − live)")
    AX_DBC.set_title("B′c offset from live Cantera");  AX_DBC.set_xlim(T_MIN, T_MAX)
    AX_DBC.grid(True, alpha=0.3)

    # Pressure sensitivity panel (Cantera table only — ref may be single-pressure)
    AX_PRES = axes[3, 2]
    P_LEVELS = [(101.325, "0.001 atm"), (1013.25, "0.01 atm"),
                (10132.5, "0.1 atm"),   (101325., "1 atm")]
    colors_p = plt.cm.cool(np.linspace(0.1, 0.9, len(P_LEVELS)))
    bg_pres = 1.0
    for i, (p, lbl) in enumerate(P_LEVELS):
        bc_tbl_p = np.array([lookup_tbl(float(T), p, bg_pres)[0] for T in T_FINE])
        AX_PRES.plot(T_FINE, bc_tbl_p, color=colors_p[i], lw=LW, ls="-", label=f"tbl {lbl}")
    # add reference at 1 atm
    bc_ref_1atm, _ = sweep_T(lookup_ref, T_FINE, bg_pres)
    AX_PRES.plot(T_FINE, bc_ref_1atm, color=C_REF,  lw=2, ls="--", label="ref 1 atm")
    AX_PRES.plot(T_LIVE, live_bc_T[bg_pres], color=C_LIVE, lw=2,
                 marker="o", ms=3, label="live 1 atm")
    AX_PRES.axhline(0, color="k", lw=0.6, ls=":")
    AX_PRES.set_xlabel("T_wall  [K]");  AX_PRES.set_ylabel("B′c  [–]")
    AX_PRES.set_title(f"Pressure sensitivity  (B′g = {bg_pres})")
    AX_PRES.set_xlim(T_MIN, T_MAX);  AX_PRES.legend(fontsize=6.5)
    AX_PRES.grid(True, alpha=0.3)

    for col in range(3, NCOLS):
        axes[3, col].set_visible(False)

    return fig, axes, T_FINE, T_LIVE, live_bc_T, live_hw_T


def print_summary(lookup_ref, lookup_tbl, lookup_live, label_ref, label_tbl, label_live,
                  T_vals=(2000, 2500, 3000, 3500), Bg_vals=(0.5, 1.0, 2.0), p=P_ATM):
    w = max(len(label_ref), len(label_tbl), len(label_live))
    print(f"\nB'c at 1 atm  (B'c < 0 = carbon deposition)")
    print(f"{'T_wall':>8}  {'B′g':>5}  {label_ref:>{w}}  {label_tbl:>{w}}  {label_live:>{w}}  {'ref−live':>9}  {'tbl−live':>9}")
    for T_s in T_vals:
        for bg in Bg_vals:
            bc_ref, _ = lookup_ref(float(T_s), p, bg)
            bc_tbl, _ = lookup_tbl(float(T_s), p, bg)
            bc_lv, _  = lookup_live(float(T_s), p, bg)
            dep = " *" if bc_lv < 0 else "  "
            print(f"{T_s:>8}  {bg:>5.2f}  {bc_ref:>{w}.4f}  {bc_tbl:>{w}.4f}  {bc_lv:>{w}.4f}{dep}  {bc_ref-bc_lv:>9.4f}  {bc_tbl-bc_lv:>9.4f}")
    print("  * deposition regime (live Cantera B'c < 0)")

    print(f"\nh_wall [MJ/kg] at 1 atm, B'g=1")
    print(f"{'T_wall':>8}  {label_ref:>{w}}  {label_tbl:>{w}}  {label_live:>{w}}  {'ref−live':>9}  {'tbl−live':>9}")
    for T_s in list(T_vals) + [4000]:
        _, hw_ref = lookup_ref(float(T_s), p, 1.0)
        _, hw_tbl = lookup_tbl(float(T_s), p, 1.0)
        _, hw_lv  = lookup_live(float(T_s), p, 1.0)
        print(f"{T_s:>8}  {hw_ref/1e6:>{w}.3f}  {hw_tbl/1e6:>{w}.3f}  {hw_lv/1e6:>{w}.3f}  {(hw_ref-hw_lv)/1e6:>9.3f}  {(hw_tbl-hw_lv)/1e6:>9.3f}")
