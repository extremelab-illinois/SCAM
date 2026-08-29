# SPDX-License-Identifier: MIT
"""Compare TACOT v3.0 B' tables: XLS reference, Cantera pre-computed table, Cantera live.

Material: TACOT v3.0  (tacot_v3.0.yaml / tacot_v3.0_3rxn.yaml)
  Both the merged 2-component and 3-reaction kinetics variants share the same
  pyrolysis gas composition and therefore the same B' chemistry.
Edge gas: air  O2:0.21 / N2:0.79  (molar)
Pyro gas: C:0.206  H:0.679  O:0.115  (elemental mass fracs)

Three sources:

  XLS reference  — tacot_v3.0_bprime_from_refXLS.TSV / .yaml
    From TACOT_3.0.xls, computed with Mutation++ (Cantera CNO thermodynamics).
    4 pressures (0.001–1 atm); 25 B'g levels (0–10); 151 T points (250–4000 K).

  Cantera table  — tacot_v3.0_bprime_air.yaml
    Pre-computed with generate_bprime.py / Cantera CNO mechanism.
    Same 4 pressures; 6 B'g levels (0, 0.1, 0.25, 0.5, 1, 2);
    151 T points (250–4000 K).
    Queried via RegularGridInterpolator; may overshoot near zero crossings
    due to the coarser B'g grid relative to the reference.

  Cantera live  — BprimeEvaluator from tacot_v3.0_bprime_config.yaml
    Direct Cantera equilibrium calls; ground truth for the Cantera
    chemistry, exposing table interpolation artefacts.

Physics note — deposition regime (B'c < 0):
  At low T and moderate-to-high B'g equilibrium favours surface carbon
  deposition, giving B'c < 0.  All three sources capture this, but the
  pre-computed table overshoots (more negative) near the transition while
  the XLS reference may smooth or clip depending on the B'g resolution.

Run:
    MPLBACKEND=Agg python3 examples/verification/bprime/compare_bprime_tacot_v3.0.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO))

from _bprime_plot import (
    load_ref_tsv, make_lookup, make_figure, print_summary, P_ATM,
)
from scam.io.material_loader import load_material
from scam.physics.bprime_evaluator import BprimeEvaluator

MAT_DIR = REPO / "scam" / "materials" / "ablative_organic"

# ── Load sources ─────────────────────────────────────────────────────────────
itp_bc, itp_hw, T_range_ref, _ = load_ref_tsv(
    str(MAT_DIR / "tacot_v3.0_bprime_from_refXLS.TSV"),
    col_order="p_bar_bg_T_Bc_hw_kJ",
)
lookup_ref = make_lookup(itp_bc, itp_hw)

_, bp_tbl = load_material(str(MAT_DIR / "tacot_v3.0.yaml"))
lookup_tbl = bp_tbl.lookup

print("Loading Cantera live evaluator…")
ev_live = BprimeEvaluator.from_config(str(MAT_DIR / "tacot_v3.0_bprime_config.yaml"))
lookup_live = ev_live.lookup
print("Cantera live evaluator ready.")

# ── Plot ──────────────────────────────────────────────────────────────────────
T_MIN, T_MAX = T_range_ref   # 250–4000 K

fig, axes, *_ = make_figure(
    title=(
        "TACOT v3.0 B′ comparison at 1 atm\n"
        "XLS ref / Mutation++ (red dashed) · Cantera table (blue) · Cantera live (green)\n"
        "air O₂:0.21 N₂:0.79 · pyro gas C:0.206 H:0.679 O:0.115"
    ),
    BG_COMPARE=[0.1, 0.25, 0.5, 1.0, 2.0],
    T_SLICES=[2000.0, 2500.0, 3000.0, 3500.0],
    lookup_ref=lookup_ref,
    lookup_tbl=lookup_tbl,
    lookup_live=lookup_live,
    T_MIN=T_MIN,
    T_MAX=T_MAX,
    label_ref="XLS ref (Mutation++)",
    label_tbl="Cantera table",
    label_live="Cantera live",
)

out = Path(__file__).parent / "bprime_tacot_v3.0.png"
fig.savefig(out, dpi=150)
print(f"\nSaved: {out}")

print_summary(
    lookup_ref, lookup_tbl, lookup_live,
    label_ref="XLS ref",
    label_tbl="Cantera tbl",
    label_live="Cantera live",
)
