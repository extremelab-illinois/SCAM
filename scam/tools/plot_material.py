# SPDX-License-Identifier: MIT
"""Plot thermophysical properties of a SCAM material card.

**Usage**::

    # SI (default) — show only
    python scam/tools/plot_material.py scam/materials/ablative_organic/tacot_v3.0.yaml

    # Imperial with °R temperature (k in BTU/ft/h/°R, cp in BTU/lb/°R)
    python scam/tools/plot_material.py scam/materials/ablative_organic/pica.yaml --imperial

    # Imperial with °F temperature (k in BTU/ft/h/°F, cp in BTU/lb/°F)
    python scam/tools/plot_material.py scam/materials/ablative_organic/pica.yaml --fahrenheit

    # Save to file (still shows the window)
    python scam/tools/plot_material.py scam/materials/ablative_organic/tacot_v3.0.yaml --save tacot_props.png

    # Save without showing (useful in headless environments)
    python scam/tools/plot_material.py scam/materials/ablative_organic/tacot_v3.0.yaml --save tacot_props.png --no-show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Allow running from the repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scam.io.material_loader import load_material


# ---------------------------------------------------------------------------
# Unit conversion helpers
# ---------------------------------------------------------------------------

def _si_to_imperial(T_K, k_WmK, cp_JkgK, rho_kgm3, h_Jkg):
    """Return (T_R, k_BTU, cp_BTU, rho_slug, h_BTU) — all Imperial."""
    T_R     = T_K * 9.0 / 5.0                  # K → °R
    k_BTU   = k_WmK * 0.577789                  # W/m/K → BTU/ft/h/°R
    cp_BTU  = cp_JkgK * 2.38846e-4              # J/kg/K → BTU/lb/°R
    rho_slug = rho_kgm3 * 0.0685218             # kg/m³ → slug/ft³
    h_BTU   = h_Jkg * 4.29923e-4               # J/kg → BTU/lb
    return T_R, k_BTU, cp_BTU, rho_slug, h_BTU


# ---------------------------------------------------------------------------
# Main plotter
# ---------------------------------------------------------------------------

def plot_material(mat_path: str | Path, imperial: bool = False,
                  fahrenheit: bool = False,
                  save: str | None = None, show: bool = True) -> None:
    mat_path = Path(mat_path)
    card, bpt = load_material(mat_path)

    use_imp = imperial or fahrenheit  # °F implies imperial for all other units
    use_f   = fahrenheit              # selects °F vs °R within imperial mode

    # --- unit labels --------------------------------------------------------
    if use_imp:
        _deg     = "°F" if use_f else "°R"
        T_label   = f"Temperature [{_deg}]"
        k_label   = f"Thermal conductivity [BTU/ft/h/{_deg}]"
        cp_label  = f"Specific heat [BTU/lb/{_deg}]"
        hg_label  = "Pyrolysis gas enthalpy [BTU/lb]"
        rho_label = "Density [slug/ft³]"
    else:
        T_label   = "Temperature [K]"
        k_label   = "Thermal conductivity [W/m/K]"
        cp_label  = "Specific heat [J/kg/K]"
        hg_label  = "Pyrolysis gas enthalpy [J/kg]"
        rho_label = "Density [kg/m³]"

    def _T(arr):
        T = arr[:, 0]
        if use_f:
            return T * 9.0 / 5.0 - 459.67   # K → °F
        if use_imp:
            return T * 9.0 / 5.0             # K → °R
        return T

    def _k(arr):
        v = arr[:, 1]
        return v * 0.577789 if use_imp else v

    def _cp(arr):
        v = arr[:, 1]
        return v * 2.38846e-4 if use_imp else v

    def _hg(arr):
        v = arr[:, 1]
        return v * 4.29923e-4 if use_imp else v

    def _rho(val):
        return val * 0.0685218 if use_imp else val

    # --- figure layout ------------------------------------------------------
    has_hg  = card.h_g_table is not None and not np.allclose(card.h_g_table[:, 1], 0)
    has_eps = hasattr(card, "emissivity_virgin_table") and card.emissivity_virgin_table is not None
    has_eps_char = hasattr(card, "emissivity_char_table") and card.emissivity_char_table is not None

    n_rows = 2
    n_cols = 3 if has_hg else 2

    fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    fig.suptitle(f"{card.name}  —  material properties", fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.42, wspace=0.35)

    # --- row 0: k_virgin, k_char | cp_virgin, cp_char | (h_g) ---------------
    ax_k  = fig.add_subplot(gs[0, 0])
    ax_cp = fig.add_subplot(gs[0, 1])

    ax_k.plot(_T(card.k_virgin_table), _k(card.k_virgin_table),
              "b-o", ms=3, label="virgin")
    ax_k.plot(_T(card.k_char_table),   _k(card.k_char_table),
              "r-s", ms=3, label="char")
    ax_k.set_xlabel(T_label); ax_k.set_ylabel(k_label)
    ax_k.set_title("Thermal conductivity")
    ax_k.legend(fontsize=8); ax_k.grid(True, alpha=0.3)

    ax_cp.plot(_T(card.cp_virgin_table), _cp(card.cp_virgin_table),
               "b-o", ms=3, label="virgin")
    ax_cp.plot(_T(card.cp_char_table),   _cp(card.cp_char_table),
               "r-s", ms=3, label="char")
    ax_cp.set_xlabel(T_label); ax_cp.set_ylabel(cp_label)
    ax_cp.set_title("Specific heat")
    ax_cp.legend(fontsize=8); ax_cp.grid(True, alpha=0.3)

    if has_hg:
        ax_hg = fig.add_subplot(gs[0, 2])
        ax_hg.plot(_T(card.h_g_table), _hg(card.h_g_table),
                   "g-o", ms=3)
        ax_hg.axhline(0, color="k", lw=0.6, ls="--")
        ax_hg.set_xlabel(T_label); ax_hg.set_ylabel(hg_label)
        ax_hg.set_title("Pyrolysis gas enthalpy")
        ax_hg.grid(True, alpha=0.3)

    # --- row 1: emissivity | decomposition summary | density bar -------------
    ax_eps  = fig.add_subplot(gs[1, 0])
    ax_comp = fig.add_subplot(gs[1, 1])
    if n_cols == 3:
        ax_rho = fig.add_subplot(gs[1, 2])
    else:
        ax_rho = ax_comp  # will be overwritten below if no components

    # emissivity
    eps_plotted = False
    if has_eps:
        et = np.array(card.emissivity_virgin_table)
        ax_eps.plot(_T(et), et[:, 1], "b-o", ms=3, label="virgin")
        eps_plotted = True
    if has_eps_char:
        et = np.array(card.emissivity_char_table)
        ax_eps.plot(_T(et), et[:, 1], "r-s", ms=3, label="char")
        eps_plotted = True
    if not eps_plotted:
        # scalar only — draw flat lines
        T_range = np.array([card.k_virgin_table[0, 0], card.k_virgin_table[-1, 0]])
        if use_imp:
            T_range = T_range * 9.0 / 5.0
        ax_eps.axhline(card.emissivity, color="b", ls="-", label=f"virgin ε={card.emissivity:.2f}")
        if card.emissivity_char >= 0:
            ax_eps.axhline(card.emissivity_char, color="r", ls="--",
                           label=f"char ε={card.emissivity_char:.2f}")
    ax_eps.set_xlabel(T_label); ax_eps.set_ylabel("Emissivity [-]")
    ax_eps.set_ylim(0, 1.05)
    ax_eps.set_title("Surface emissivity")
    ax_eps.legend(fontsize=8); ax_eps.grid(True, alpha=0.3)

    # decomposition components — horizontal bar chart of density contributions
    if card.components:
        names  = [c.name for c in card.components]
        rho0s  = [_rho(c.rho_0) for c in card.components]
        rho_rs = [_rho(c.rho_r) for c in card.components]
        x = np.arange(len(names))
        w = 0.35
        ax_comp.bar(x - w/2, rho0s,  w, label="ρ₀ (virgin)",  color="steelblue")
        ax_comp.bar(x + w/2, rho_rs, w, label="ρᵣ (residual)", color="firebrick")
        ax_comp.set_xticks(x); ax_comp.set_xticklabels(names, fontsize=8, rotation=15)
        ax_comp.set_ylabel(rho_label)
        ax_comp.set_title("Component densities")
        ax_comp.legend(fontsize=8); ax_comp.grid(True, alpha=0.3, axis="y")

        # text box: Arrhenius summary
        summary = "\n".join(
            f"{c.name[:14]}: A={c.A_rate:.1e}, Ea={c.E_act/1e3:.0f} kJ/mol"
            for c in card.components
        )
        ax_comp.text(0.01, 0.99, summary, transform=ax_comp.transAxes,
                     fontsize=6.5, va="top", family="monospace",
                     bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))
    else:
        ax_comp.text(0.5, 0.5, "Non-decomposing\n(inert material)",
                     ha="center", va="center", transform=ax_comp.transAxes, fontsize=11)
        ax_comp.set_title("Decomposition")
        ax_comp.axis("off")

    # density summary bar (row 1, col 2 only when n_cols == 3)
    if n_cols == 3:
        rho_v = _rho(card.rho_virgin)
        rho_c = _rho(card.rho_char)
        ax_rho.bar(["virgin", "char"], [rho_v, rho_c],
                   color=["steelblue", "firebrick"], width=0.5)
        ax_rho.set_ylabel(rho_label)
        ax_rho.set_title("Bulk density")
        ax_rho.grid(True, alpha=0.3, axis="y")
        for i, v in enumerate([rho_v, rho_c]):
            ax_rho.text(i, v * 1.01, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    # --- annotations --------------------------------------------------------
    unit_str = "Imperial (°F)" if use_f else ("Imperial (°R)" if use_imp else "SI")
    bpt_str  = f"B′ table: {Path(bpt._path).name}" if bpt else "No B′ table"
    fig.text(0.99, 0.01, f"{mat_path.name}  |  {unit_str}  |  {bpt_str}",
             ha="right", va="bottom", fontsize=7, color="grey")

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"Saved → {save}")

    if show:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot SCAM material thermophysical properties.")
    p.add_argument("material", help="Path to material YAML (e.g. scam/materials/ablative_organic/tacot_v3.0.yaml)")
    p.add_argument("--imperial", action="store_true", default=False,
                   help="Display in Imperial units with °R temperature (default: SI)")
    p.add_argument("--fahrenheit", action="store_true", default=False,
                   help="Display in Imperial units with °F temperature (implies --imperial)")
    p.add_argument("--save", metavar="FILE", default=None,
                   help="Save figure to FILE (PNG/PDF/SVG; inferred from extension)")
    p.add_argument("--no-show", action="store_true", default=False,
                   help="Do not open an interactive window (useful with --save in headless mode)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    plot_material(
        mat_path=args.material,
        imperial=args.imperial,
        fahrenheit=args.fahrenheit,
        save=args.save,
        show=not args.no_show,
    )
