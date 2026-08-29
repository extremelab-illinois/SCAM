# SPDX-License-Identifier: MIT
"""Compare live Cantera, pretabulated, and live Mutation++ backends."""

from __future__ import annotations

from contextlib import redirect_stdout
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import compare_pato_ablation2 as case


HERE = Path(__file__).resolve().parent
OUT = HERE / "compare_pato_ablation2_live_vs_pretab.png"


def _run_mode(mode: str):
    log = Path("/tmp") / f"compare_pato_ablation2_{mode}.log"
    with log.open("w") as stream, redirect_stdout(stream):
        result = case.run_scam(mode)
    return result, log


def main() -> None:
    pretab, pretab_log = _run_mode("table")
    live, live_log     = _run_mode("live_cantera_zc_default")
    mpp, mpp_log       = _run_mode("live_mpp")

    t_table  = pretab.times_array();  tw_table = pretab.T_wall_array();  s_table = pretab.s_array()
    t_live   = live.times_array();    tw_live  = live.T_wall_array();    s_live  = live.s_array()
    t_mpp    = mpp.times_array();     tw_mpp   = mpp.T_wall_array();     s_mpp   = mpp.s_array()

    t_pato, T_pato_all = case.load_pato_ta_surfacepatch(case.TA_SURF)
    tw_pato = T_pato_all[:, 0]
    t_mass, _mdg, _mdc, s_pato = case.load_pato_mass(case.MASS_FILE)

    # Interpolate everything onto a common grid for difference plots.
    common_t    = np.linspace(0.0, case.T_END, 1201)
    tw_pato_i   = np.interp(common_t, t_pato, tw_pato)
    s_pato_i    = np.interp(common_t, t_mass, s_pato)
    tw_table_i  = np.interp(common_t, t_table, tw_table)
    tw_live_i   = np.interp(common_t, t_live,  tw_live)
    tw_mpp_i    = np.interp(common_t, t_mpp,   tw_mpp)
    s_table_i   = np.interp(common_t, t_table, s_table)
    s_live_i    = np.interp(common_t, t_live,  s_live)
    s_mpp_i     = np.interp(common_t, t_mpp,   s_mpp)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        "AblationTestCase_2.x: surface chemistry backend comparison vs PATO"
    )

    # --- absolute T_wall ---
    ax = axes[0, 0]
    ax.plot(t_pato,  tw_pato,  "k--",  label="PATO",              linewidth=1.5)
    ax.plot(t_live,  tw_live,  label="SCAM live Cantera")
    ax.plot(t_table, tw_table, label="SCAM pretabulated",          alpha=0.85)
    ax.plot(t_mpp,   tw_mpp,   label="SCAM live Mutation++",       alpha=0.85)
    ax.set(xlabel="Time [s]", ylabel="T_wall [K]", title="Surface temperature")
    ax.grid(alpha=0.3)
    ax.legend()

    # --- T_wall differences vs PATO ---
    ax = axes[0, 1]
    ax.plot(common_t, tw_live_i  - tw_pato_i, color="tab:blue",   label="Cantera − PATO")
    ax.plot(common_t, tw_mpp_i   - tw_pato_i, color="tab:red",    label="Mutation++ − PATO", alpha=0.85)
    ax.plot(common_t, tw_table_i - tw_pato_i, color="tab:purple", label="Pretabulated − PATO", alpha=0.85)
    ax.axhline(0.0, color="k", linewidth=0.8)
    ax.set(
        xlabel="Time [s]",
        ylabel="SCAM − PATO [K]",
        title="Surface temperature difference vs PATO",
    )
    ax.grid(alpha=0.3)
    ax.legend()

    # --- absolute recession ---
    ax = axes[1, 0]
    ax.plot(t_mass,  1000.0 * s_pato,  "k--",  label="PATO",              linewidth=1.5)
    ax.plot(t_live,  1000.0 * s_live,  label="SCAM live Cantera")
    ax.plot(t_table, 1000.0 * s_table, label="SCAM pretabulated",          alpha=0.85)
    ax.plot(t_mpp,   1000.0 * s_mpp,   label="SCAM live Mutation++",       alpha=0.85)
    ax.set(xlabel="Time [s]", ylabel="Recession [mm]", title="Surface recession")
    ax.grid(alpha=0.3)
    ax.legend()

    # --- recession differences vs PATO ---
    ax = axes[1, 1]
    ax.plot(common_t, 1000.0 * (s_live_i  - s_pato_i), color="tab:blue",   label="Cantera − PATO")
    ax.plot(common_t, 1000.0 * (s_mpp_i   - s_pato_i), color="tab:red",    label="Mutation++ − PATO", alpha=0.85)
    ax.plot(common_t, 1000.0 * (s_table_i - s_pato_i), color="tab:purple", label="Pretabulated − PATO", alpha=0.85)
    ax.axhline(0.0, color="k", linewidth=0.8)
    ax.set(
        xlabel="Time [s]",
        ylabel="SCAM − PATO [mm]",
        title="Recession difference vs PATO",
    )
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    plt.close(fig)

    # --- summary table ---
    heating = (common_t >= 2.0) & (common_t <= 59.9)
    print(f"\n{'t':>6}  {'Tw_PATO':>9}  {'Tw_Cant':>9}  {'Tw_MPP':>9}  {'Tw_tab':>9}  "
          f"{'ΔT_Cant':>8}  {'ΔT_MPP':>8}  {'ΔT_tab':>8}  "
          f"{'s_PATO':>8}  {'s_Cant':>8}  {'Δs_Cant':>8}  {'Δs_MPP':>8}  {'Δs_tab':>8}")
    print("-" * 130)
    for target in (2.0, 10.0, 30.0, 60.0, 90.0, 120.0):
        Tp   = float(np.interp(target, t_pato,  tw_pato))
        Tl   = float(np.interp(target, t_live,  tw_live))
        Tm   = float(np.interp(target, t_mpp,   tw_mpp))
        Tt   = float(np.interp(target, t_table, tw_table))
        sp   = 1000.0 * float(np.interp(target, t_mass,  s_pato))
        sl   = 1000.0 * float(np.interp(target, t_live,  s_live))
        sm   = 1000.0 * float(np.interp(target, t_mpp,   s_mpp))
        st   = 1000.0 * float(np.interp(target, t_table, s_table))
        print(f"{target:6.1f}  {Tp:9.3f}  {Tl:9.3f}  {Tm:9.3f}  {Tt:9.3f}  "
              f"{Tl-Tp:+8.3f}  {Tm-Tp:+8.3f}  {Tt-Tp:+8.3f}  "
              f"{sp:8.4f}  {sl:8.4f}  {sl-sp:+8.4f}  {sm-sp:+8.4f}  {st-sp:+8.4f}")

    print()
    for label, tw_i, s_i in [
        ("Cantera",      tw_live_i,  s_live_i),
        ("Mutation++",   tw_mpp_i,   s_mpp_i),
        ("Pretabulated", tw_table_i, s_table_i),
    ]:
        max_dT = float(np.max(np.abs(tw_i[heating] - tw_pato_i[heating])))
        max_ds = 1000.0 * float(np.max(np.abs(s_i - s_pato_i)))
        print(f"max |{label:13s} − PATO|  heating T = {max_dT:.3f} K   recession = {max_ds:.5f} mm")

    print(f"\nplot: {OUT}")
    print(f"logs: {pretab_log}, {live_log}, {mpp_log}")


if __name__ == "__main__":
    main()
