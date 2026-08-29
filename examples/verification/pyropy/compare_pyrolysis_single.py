# SPDX-License-Identifier: MIT
"""Compare SCAM vs pyropy — single decomposition reaction.

A single resin component (TACOT resin_A kinetics) is driven through a
linear temperature ramp.  Both solvers see exactly the same Arrhenius ODE:

    dρ/dt = -k(T) · ρ^m / ρ₀^(m-1)      [SCAM, with ρᵣ=0]
    dα/dt =  k(T) · (1-α)^n               [pyropy, with α=(ρ₀-ρ)/ρ₀, n=m]

For ρᵣ = 0 these are identical (both normalise by ρ₀^(m-1)).  Any remaining
difference is purely numerical: SCAM uses a closed-form analytical step
integration while pyropy uses scipy Radau.

Run from the repo root:

    python3 examples/pyropy_verification/compare_pyrolysis_single.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
PYROPY_REPO = REPO.parent / "pyropy"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(PYROPY_REPO))

from scam.config.material import ComponentCard, MaterialCard
from scam.physics.decomposition import update_nodelet_densities

from pyropy import PyrolysisParallel, ReactManager


# ---------------------------------------------------------------------------
# Kinetic parameters (TACOT resin_A, Lachaud & Mansour JTHT 2014)
# ---------------------------------------------------------------------------
RHO_0    = 60.0       # initial virgin density of this component [kg/m³]
RHO_R    = 0.0        # residual (char) density [kg/m³] — pure gas-phase product
A_RATE   = 1.200e4    # Arrhenius pre-exponential [1/s]
E_ACT    = 87_000.0   # activation energy [J/mol]
M_EXP    = 3.0        # reaction order

# Temperature ramp
T0       = 300.0      # initial temperature [K]
T_END    = 1800.0     # final temperature [K]
BETA     = 20.0       # heating rate [K/min]
BETA_KS  = BETA / 60  # [K/s]
N_POINTS = 2000       # number of output points


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _time_vec():
    return np.linspace(0.0, (T_END - T0) / BETA_KS, N_POINTS)


def _T_vec(t):
    return T0 + BETA_KS * t


def _scam_material() -> MaterialCard:
    """One-component inert material carrying only resin_A kinetics."""
    comp = ComponentCard(
        name="resin_A",
        rho_0=RHO_0, rho_r=RHO_R,
        A_rate=A_RATE, E_act=E_ACT, m_exp=M_EXP,
        h_decomp=0.0,
    )
    T_pts = np.array([200.0, 4000.0])
    k_tab  = np.column_stack([T_pts, [1.0, 1.0]])
    cp_tab = np.column_stack([T_pts, [800.0, 800.0]])
    hg_tab = np.column_stack([T_pts, [0.0, 0.0]])
    return MaterialCard(
        name="SingleResin",
        rho_virgin=RHO_0, rho_char=RHO_R, gamma_resin=1.0,
        components=[comp],
        k_virgin_table=k_tab, k_char_table=k_tab,
        cp_virgin_table=cp_tab, cp_char_table=cp_tab,
        h_g_table=hg_tab,
        decomposing=True,
    )


def _pyropy_react_manager() -> ReactManager:
    """Build a ReactManager for a single parallel reaction without file I/O."""
    rm = ReactManager()
    rm.solids = ["resin_A"]
    rm.n_reactions = 1
    rm.n_solids = 1
    rm.rhoIni = RHO_0
    rm.solid_reactant = ["resin_A"]
    rm.solid_product  = ["Gas"]
    rm.rhs = ["Gas"]
    # pyropy stores log10(A); n matches SCAM m_exp; F=1 (100% converts)
    rm.dict_params = {
        "E": [E_ACT],
        "A": [np.log10(A_RATE)],
        "n": [M_EXP],
        "F": [1.0],
    }
    return rm


# ---------------------------------------------------------------------------
# Run SCAM decomposition kernel
# ---------------------------------------------------------------------------

def run_scam(mat: MaterialCard, t: np.ndarray) -> np.ndarray:
    """Drive update_nodelet_densities step-by-step along the temperature ramp."""
    rho_comp = np.array([[[RHO_0]]])
    T_nod    = np.array([[T0]])
    delta    = np.array([[1e-3]])
    shrink   = np.array([False])

    rho_out  = np.empty(len(t))
    rho_out[0] = RHO_0

    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        T_nod[0, 0] = T0 + BETA_KS * t[i - 1]
        rho_comp, _, _comp = update_nodelet_densities(mat, rho_comp, T_nod, dt, 0.0, delta, shrink)
        rho_out[i] = float(rho_comp[0, 0, 0])

    return rho_out


# ---------------------------------------------------------------------------
# Run pyropy PyrolysisParallel
# ---------------------------------------------------------------------------

def run_pyropy(rm: ReactManager, t: np.ndarray) -> np.ndarray:
    """Solve via pyropy's Radau ODE integrator."""
    model = PyrolysisParallel(
        temp_0=T0, temp_end=T_END,
        beta=BETA, n_points=N_POINTS,
        reaction_scheme_obj=rm,
    )
    model.solve_system()
    return model.rho_solid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t   = _time_vec()
    T   = _T_vec(t)
    mat = _scam_material()
    rm  = _pyropy_react_manager()

    print("Running SCAM decomposition kernel...")
    rho_scam   = run_scam(mat, t)

    print("Running pyropy PyrolysisParallel (Radau)...")
    rho_pyropy = run_pyropy(rm, t)

    alpha_scam   = (RHO_0 - rho_scam)   / RHO_0
    alpha_pyropy = (RHO_0 - rho_pyropy) / RHO_0

    max_diff_rho   = float(np.abs(rho_scam - rho_pyropy).max())
    max_diff_alpha = float(np.abs(alpha_scam - alpha_pyropy).max())

    print(f"\nResults:")
    print(f"  max |ρ_SCAM − ρ_pyropy|   = {max_diff_rho:.4f} kg/m³")
    print(f"  max |α_SCAM − α_pyropy|   = {max_diff_alpha:.2e}")
    print(f"  Final density  SCAM   = {rho_scam[-1]:.4f} kg/m³")
    print(f"  Final density  pyropy = {rho_pyropy[-1]:.4f} kg/m³")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle(
            f"Single decomposition reaction — SCAM vs pyropy\n"
            f"resin_A: A={A_RATE:.2e} 1/s,  E={E_ACT:.0f} J/mol,  m=n={M_EXP:.0f},  "
            f"β={BETA:.0f} K/min",
            fontsize=11,
        )

        ax = axes[0]
        ax.plot(T, rho_scam,   "b-",  lw=2,   label="SCAM (analytical step)")
        ax.plot(T, rho_pyropy, "r--", lw=1.5, label="pyropy (Radau ODE)")
        ax.set_xlabel("Temperature [K]")
        ax.set_ylabel("Density  ρ  [kg/m³]")
        ax.set_title("Density evolution")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(T, alpha_scam * 100,   "b-",  lw=2,   label="SCAM")
        ax.plot(T, alpha_pyropy * 100, "r--", lw=1.5, label="pyropy")
        ax.set_xlabel("Temperature [K]")
        ax.set_ylabel("Conversion  α  [%]")
        ax.set_title("Conversion degree  α = (ρ₀ − ρ) / ρ₀")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        ax.semilogy(T, np.abs(rho_scam - rho_pyropy) + 1e-12, "g-", lw=1.5)
        ax.set_xlabel("Temperature [K]")
        ax.set_ylabel("|ρ_SCAM − ρ_pyropy|  [kg/m³]")
        ax.set_title(
            f"Pointwise difference\n(analytical step vs Radau)\n"
            f"max = {max_diff_rho:.4f} kg/m³"
        )
        ax.grid(True, alpha=0.3, which="both")

        plt.tight_layout()
        out = Path(__file__).parent / "compare_pyrolysis_single.png"
        plt.savefig(out, dpi=150)
        print(f"\nPlot saved → {out}")
        plt.show()

    except ImportError:
        print("\n(matplotlib not available — skipping plots)")
    except Exception as exc:
        print(f"\n(plot failed: {exc})")


if __name__ == "__main__":
    main()
