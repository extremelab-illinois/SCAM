# SPDX-License-Identifier: MIT
"""Compare SCAM vs pyropy — multiple parallel decomposition reactions.

Full TACOT charring material with two independent resin components driven
through a linear temperature ramp.  Each component obeys the same Arrhenius
ODE as in compare_pyrolysis_single.py; here they are solved together.

TACOT composition (Lachaud & Mansour JTHT 2014):
    rho_virgin = 280 kg/m³
    ├─ resin_A:  ρ₀=60,  ρᵣ=0, A=1.2e4 1/s, E=87 000 J/mol, m=3
    ├─ resin_B:  ρ₀=40,  ρᵣ=0, A=4.48e9 1/s, E=150 440 J/mol, m=3
    └─ fiber:   ρ_char=180 (inert, does not decompose)

Why two components appear as ONE peak in the total dρ/dT
─────────────────────────────────────────────────────────
Kinetic compensation: resin_B has a much larger activation energy (E=150 kJ/mol
vs 87 kJ/mol) BUT also a far larger pre-exponential (A=4.48e9 vs 1.2e4).  These
effects cancel: both reactions reach 50% conversion within ~58 K of each other
(T₅₀≈693 K for resin_B, ≈751 K for resin_A).  Their individual −dρ/dT peaks are
only ~39 K apart and together span the same 640–960 K window, so the *total* rate
always shows a single merged hump.  The per-component rate subplot (bottom-centre)
reveals both peaks in separate colours on a zoomed axis.

Integration accuracy note
─────────────────────────
SCAM evaluates the Arrhenius rate at the *start-of-step* temperature (T constant
over Δt), while pyropy's Radau integrator adapts internally.  With N_PTS=10 000
(ΔT≈0.15 K/step) the constant-T error is < 0.6% per step → max per-component
density difference < 0.02 kg/m³ (< 0.05%) — effectively invisible.

Convention note — for ρᵣ = 0 both models share the same equation:
    SCAM:   dρ_i/dt = −k_i·(ρ_i)^m / ρ₀_i^(m-1)
    pyropy: dα_i/dt =  k_i·(1−α_i)^n        (α_i = 1 − ρ_i/ρ₀_i, n=m)
These are identical when ρᵣ=0; any residual difference is purely from the
integration method (analytical step vs Radau).

Run from the repo root:

    python3 examples/pyropy_verification/compare_pyrolysis_multi.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
PYROPY_REPO = REPO.parent / "pyropy"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(PYROPY_REPO))

from scam.io.material_loader import load_material
from scam.physics.decomposition import update_nodelet_densities

from pyropy import PyrolysisParallel, ReactManager


# ---------------------------------------------------------------------------
# Temperature ramp
# ---------------------------------------------------------------------------
T0      = 300.0    # [K]
T_END   = 1800.0   # [K]
BETA    = 20.0     # heating rate [K/min]
BETA_KS = BETA / 60
# 10 000 points → Δt = 0.45 s, ΔT = 0.15 K/step → SCAM step error < 0.6%
N_PTS   = 10_000


# ---------------------------------------------------------------------------
# pyropy ReactManager built from TACOT parameters without file I/O
# ---------------------------------------------------------------------------

def _pyropy_react_manager(mat) -> ReactManager:
    rho_v = mat.rho_virgin
    comps = mat.components

    rm = ReactManager()
    rm.solids = [c.name for c in comps] + ["fiber"]
    rm.n_reactions = len(comps)
    rm.n_solids = len(rm.solids)
    rm.rhoIni = rho_v
    rm.solid_reactant = [c.name for c in comps]
    rm.solid_product  = [f"Gas_{c.name}" for c in comps]
    rm.rhs = rm.solid_product[:]
    rm.dict_params = {
        "E": [c.E_act              for c in comps],
        "A": [np.log10(c.A_rate)   for c in comps],
        "n": [c.m_exp              for c in comps],
        "F": [c.rho_0 / rho_v      for c in comps],
    }
    return rm


# ---------------------------------------------------------------------------
# SCAM decomposition loop
# ---------------------------------------------------------------------------

def run_scam(mat, t: np.ndarray):
    """Step update_nodelet_densities along the temperature ramp.

    T is evaluated at the start of each step (SCAM's actual behaviour).
    With ΔT=0.15 K/step the constant-T error is negligible.

    Returns:
        rho_total  — total nodal density [kg/m³] at each output time
        rho_each   — per-component density, shape (n_comp, N_PTS)
    """
    n_comp = len(mat.components)
    rho_comp = np.zeros((n_comp, 1, 1))
    for ic, c in enumerate(mat.components):
        rho_comp[ic, 0, 0] = c.rho_0

    T_nod  = np.array([[T0]])
    delta  = np.array([[1e-3]])
    shrink = np.array([False])

    rho_each  = np.empty((n_comp, len(t)))
    rho_total = np.empty(len(t))

    for ic, c in enumerate(mat.components):
        rho_each[ic, 0] = c.rho_0
    rho_total[0] = mat.rho_virgin

    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        T_nod[0, 0] = T0 + BETA_KS * t[i - 1]   # start-of-step temperature
        rho_comp, _, _comp = update_nodelet_densities(mat, rho_comp, T_nod, dt, 0.0, delta, shrink)
        for ic in range(n_comp):
            rho_each[ic, i] = float(rho_comp[ic, 0, 0])
        rho_total[i] = mat.rho_char + float(rho_comp.sum())

    return rho_total, rho_each


# ---------------------------------------------------------------------------
# pyropy parallel solve + per-component extraction
# ---------------------------------------------------------------------------

def run_pyropy(rm: ReactManager, mat, t: np.ndarray):
    """Radau ODE solver; returns total rho and per-component rho arrays."""
    model = PyrolysisParallel(
        temp_0=T0, temp_end=T_END, beta=BETA, n_points=N_PTS,
        reaction_scheme_obj=rm,
    )
    model.solve_system()
    rho_total = model.rho_solid

    # per-component: re-solve to extract α_i(t) with tight tolerances
    from scipy.integrate import solve_ivp
    betaKs = BETA / 60
    y0  = np.zeros(rm.n_reactions)
    sol = solve_ivp(
        fun=lambda tt, z: model.pyro_rates(z, tt, T0, betaKs),
        t_span=(0, t[-1]), y0=y0, t_eval=t,
        method="Radau", max_step=5 / betaKs * 100, rtol=1e-7,
    )
    rho_each = np.array([
        mat.components[ic].rho_0 * (1.0 - sol.y[ic])
        for ic in range(len(mat.components))
    ])
    return rho_total, rho_each


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _peak_T(rho_i: np.ndarray, T: np.ndarray) -> float:
    """Temperature of the peak −dρᵢ/dT for one component."""
    rate = -np.gradient(rho_i, T)
    return float(T[np.argmax(rate)])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    mat, _ = load_material("scam/materials/ablative_organic/tacot_v3.0.yaml")
    rm = _pyropy_react_manager(mat)

    t = np.linspace(0.0, (T_END - T0) / BETA_KS, N_PTS)
    T = T0 + BETA_KS * t
    dt_step = t[1] - t[0]

    print(f"N_PTS={N_PTS},  Δt={dt_step:.3f} s,  ΔT={dt_step*BETA_KS:.3f} K/step")
    print("Running SCAM decomposition kernel (2 components)...")
    rho_scam_total, rho_each_scam = run_scam(mat, t)

    print("Running pyropy PyrolysisParallel (Radau, 2 reactions)...")
    rho_pyropy_total, rho_each_pyropy = run_pyropy(rm, mat, t)

    max_diff_total = float(np.abs(rho_scam_total - rho_pyropy_total).max())
    print(f"\nTotal density:")
    print(f"  max |ρ_total_SCAM − ρ_total_pyropy| = {max_diff_total:.4f} kg/m³")

    peak_Ts: dict[str, float] = {}
    for ic, c in enumerate(mat.components):
        d = float(np.abs(rho_each_scam[ic] - rho_each_pyropy[ic]).max())
        pT = _peak_T(rho_each_scam[ic], T)
        peak_Ts[c.name] = pT
        T50_s = float(T[np.argmin(np.abs(rho_each_scam[ic]   - c.rho_0 / 2))])
        T50_p = float(T[np.argmin(np.abs(rho_each_pyropy[ic] - c.rho_0 / 2))])
        print(f"\n  {c.name}:")
        print(f"    peak −dρ/dT at T = {pT:.1f} K")
        print(f"    T₅₀  SCAM={T50_s:.1f} K  pyropy={T50_p:.1f} K")
        print(f"    max |ρ_SCAM − ρ_pyropy| = {d:.4f} kg/m³  ({d/c.rho_0*100:.3f}% of ρ₀)")

    names = list(peak_Ts.keys())
    peak_sep = abs(peak_Ts[names[0]] - peak_Ts[names[1]])
    print(f"\n  Peak separation = {peak_sep:.0f} K  "
          f"(kinetic compensation — both reactions overlap → one merged hump in total)")
    print(f"  Final rho_total  SCAM   = {rho_scam_total[-1]:.3f} kg/m³")
    print(f"  Final rho_total  pyropy = {rho_pyropy_total[-1]:.3f} kg/m³  "
          f"(expected ≈{mat.rho_char:.1f} kg/m³)")

    # ── Plotting ─────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        COLORS = {"resin_A": "#1f77b4", "resin_B": "#ff7f0e"}

        fig.suptitle(
            "Multiple parallel decomposition reactions — SCAM (analytical step) vs pyropy (Radau)\n"
            "TACOT: resin_A (A=1.2e4, E=87 kJ/mol, m=3) + "
            "resin_B (A=4.48e9, E=150 kJ/mol, m=3),  β=20 K/min",
            fontsize=11,
        )

        # zoom window for decomposition subplots
        T_lo, T_hi = 550.0, 1050.0
        mask = (T >= T_lo) & (T <= T_hi)
        vkw  = dict(lw=1.1, ls="--", alpha=0.65)

        # ── [0,0]  total density — full range ────────────────────────────────
        ax = axes[0, 0]
        ax.plot(T, rho_scam_total, "b-", lw=1.5, label="SCAM (total)")
        ax.plot(T, rho_pyropy_total, "r--", lw=1.5, label="pyropy (total)")
        ax.axhline(mat.rho_char, color="gray", ls=":", lw=1.2,
                   label=f"ρ_char = {mat.rho_char:.0f} kg/m³")
        ax.set_xlabel("Temperature [K]")
        ax.set_ylabel("Total density  [kg/m³]")
        ax.set_title("Total material density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── [0,1]  per-component density — zoomed ────────────────────────────
        ax = axes[0, 1]
        for ic, c in enumerate(mat.components):
            col = COLORS.get(c.name, f"C{ic}")
            ax.plot(T[mask], rho_each_scam[ic][mask],   "-",  color=col, lw=2,
                    label=f"SCAM {c.name}")
            ax.plot(T[mask], rho_each_pyropy[ic][mask], "--", color=col, lw=1.5,
                    label=f"pyropy {c.name}")
        ax.set_xlabel("Temperature [K]")
        ax.set_ylabel("Component density  [kg/m³]")
        ax.set_title("Per-component density  (decomposition zone)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── [0,2]  total density difference ──────────────────────────────────
        ax = axes[0, 2]
        ax.semilogy(T, np.abs(rho_scam_total - rho_pyropy_total) + 1e-12, "g-", lw=1.5)
        ax.set_xlabel("Temperature [K]")
        ax.set_ylabel("|Δρ_total|  [kg/m³]")
        ax.set_title(f"Total density difference\nmax = {max_diff_total:.4f} kg/m³")
        ax.grid(True, alpha=0.3, which="both")

        # ── [1,0]  total decomposition rate — zoomed — shows merged peak ─────
        ax = axes[1, 0]
        drho_scam   = -np.gradient(rho_scam_total,   T)
        drho_pyropy = -np.gradient(rho_pyropy_total, T)
        ax.plot(T[mask], drho_scam[mask],   "b-", lw=1.5,   label="SCAM")
        ax.plot(T[mask], drho_pyropy[mask], "r--", lw=1.5, label="pyropy")
        y_top = float(drho_scam[mask].max()) * 1.05
        for ic, c in enumerate(mat.components):
            pT  = peak_Ts[c.name]
            col = COLORS.get(c.name, f"C{ic}")
            ax.axvline(pT, color=col, **vkw)
            ax.text(pT + 3, y_top * 0.02, f"{c.name}\n{pT:.0f} K",
                    color=col, fontsize=7.5, va="bottom")
        ax.set_xlim(T_lo, T_hi)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Temperature [K]")
        ax.set_ylabel("−dρ/dT  [kg/m³/K]")
        ax.set_title(
            "Total rate — zoomed to 550–1050 K\n"
            "Two overlapping reactions → single merged hump"
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── [1,1]  per-component rate — zoomed — reveals two distinct peaks ──
        ax = axes[1, 1]
        for ic, c in enumerate(mat.components):
            col = COLORS.get(c.name, f"C{ic}")
            rate_s = -np.gradient(rho_each_scam[ic],   T)
            rate_p = -np.gradient(rho_each_pyropy[ic], T)
            ax.plot(T[mask], rate_s[mask], "-",  color=col, lw=2,
                    label=f"SCAM {c.name}  (peak {peak_Ts[c.name]:.0f} K)")
            ax.plot(T[mask], rate_p[mask], "--", color=col, lw=1.5,
                    label=f"pyropy {c.name}")
            ax.axvline(peak_Ts[c.name], color=col, **vkw)
        ax.set_xlim(T_lo, T_hi)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Temperature [K]")
        ax.set_ylabel("−dρ_i/dT  [kg/m³/K]")
        ax.set_title(
            f"Per-component rate — zoomed\n"
            f"Peak separation = {peak_sep:.0f} K  (kinetic compensation)"
        )
        ax.legend(fontsize=7.5)
        ax.grid(True, alpha=0.3)

        # ── [1,2]  per-component absolute difference ─────────────────────────
        ax = axes[1, 2]
        for ic, c in enumerate(mat.components):
            col = COLORS.get(c.name, f"C{ic}")
            diff  = np.abs(rho_each_scam[ic] - rho_each_pyropy[ic]) + 1e-12
            d_max = float(diff.max())
            ax.semilogy(T, diff, "-", color=col, lw=1.5,
                        label=f"{c.name}  (max {d_max:.4f} kg/m³)")
        ax.set_xlabel("Temperature [K]")
        ax.set_ylabel("|Δρ_i|  [kg/m³]")
        ax.set_title(
            "Per-component absolute difference\n"
            "(analytical constant-T step vs Radau)"
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which="both")

        plt.tight_layout()
        out = Path(__file__).parent / "compare_pyrolysis_multi.png"
        plt.savefig(out, dpi=150)
        print(f"\nPlot saved → {out}")
        plt.show()

    except ImportError:
        print("\n(matplotlib not available — skipping plots)")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"\n(plot failed: {exc})")


if __name__ == "__main__":
    main()
