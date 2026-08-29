# SPDX-License-Identifier: MIT
"""Compare TACOT B' from current Mutation++ (NASA-9 thermo) vs Cantera (CNO).

Both compute the same surface mass balance (SMB):

    Z_wall = (Z_edge + B'g·Z_pyro + B'c·Z_cond) / (1 + B'g + B'c)

then find the multi-phase equilibrium at (T, P, Z_wall) and read off B'c from
the gas-phase carbon fraction.  They differ in thermodynamic databases and
species sets:

  Mutation++ (current, /opt/Mutationpp/):
    - NASA-9 thermodynamic database
    - 35-species TACOT/air gas mixture + C(gr) condensed phase
    - Multi-phase equilibrium: MultiPhaseEquilSolver partitions C between gas
      and solid graphite.  C(gr) MUST be in the species list — without it the
      solver is gas-phase-only, carbon is conserved, and the LargeNumber=100
      shortcut in surfaceMassBalance returns B'c ≈ 100 (non-physical).
    - B' computed by: bprime -T T1:dT:T2 -P P -b Bg
                             -m tacot_air_bprime -bl air -py tacot_pyro -cp carbon

  Cantera (CNO mechanism):
    - cno_ablation.yaml thermodynamic mechanism
    - Multi-phase: gas phase + graphite condensed phase (carbon_moles=100)
    - Pre-computed table: tacot_v3.0_bprime_air.yaml  (regenerated 2026-06-16)
    - Live evaluator:     tacot_v3.0_bprime_config.yaml

Edge gas:      air  O2:0.21, N2:0.79 (molar)
Pyrolysis gas: CH4:0.5551, CO:0.2418, H2O:0.2031 (molar)
Condensed:     pure graphite C(gr) / C(s)

Run (from SCAM repo root):
    MPLBACKEND=Agg python3 examples/verification/bprime/compare_bprime_mpp_vs_cantera.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO))

from _bprime_plot import make_figure, print_summary
from scam.io.material_loader import load_material
from scam.physics.bprime_evaluator import BprimeEvaluator
from scam.physics.mpp_evaluator import MutationppEvaluator

# ── paths ─────────────────────────────────────────────────────────────────────
MAT_DIR    = REPO / "scam" / "materials" / "ablative_organic"
MPP_CONFIG = MAT_DIR / "tacot_v3.0_mpp_config.yaml"

# ── main ──────────────────────────────────────────────────────────────────────

if not MPP_CONFIG.exists():
    print(f"ERROR: Mpp config not found: {MPP_CONFIG}")
    sys.exit(1)

# ── Cantera sources ──────────────────────────────────────────────────────────────
print("Loading Cantera pre-computed table (tacot_v3.0_bprime_air.yaml)…")
_, bp_tbl = load_material(str(MAT_DIR / "tacot_v3.0.yaml"))
lookup_tbl = bp_tbl.lookup

print("Loading Cantera live evaluator…")
ev_live = BprimeEvaluator.from_config(str(MAT_DIR / "tacot_v3.0_bprime_config.yaml"))
lookup_live = ev_live.lookup
print("Cantera live evaluator ready.")

# ── Mutation++ sweep (warm-start grid via MutationppEvaluator) ─────────────────
print("\nBuilding MutationppEvaluator (warm-start grid)…")
ev_mpp = MutationppEvaluator.from_config(MPP_CONFIG)
lookup_mpp = ev_mpp.lookup
print("Mutation++ sweep complete.")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, *_ = make_figure(
    title=(
        "TACOT B′ comparison — Mutation++ (NASA-9 + C(gr)) vs Cantera (CNO) at 1 atm\n"
        "Mutation++ bprime CLI (red dashed) · Cantera pre-comp table (blue) · Cantera live (green)\n"
        "air O₂:0.21 N₂:0.79 · pyro CH₄:0.5551 CO:0.2418 H₂O:0.2031\n"
        "NOTE: C(gr) condensed phase required in species list for physical B′c values."
    ),
    BG_COMPARE=[0.1, 0.25, 0.5, 1.0, 2.0],
    T_SLICES=[2000.0, 2500.0, 3000.0, 3500.0],
    lookup_ref=lookup_mpp,
    lookup_tbl=lookup_tbl,
    lookup_live=lookup_live,
    T_MIN=ev_mpp._T_grid[0],
    T_MAX=ev_mpp._T_grid[-1],
    label_ref="Mut++ / NASA-9",
    label_tbl="Cantera table",
    label_live="Cantera live",
)

out = Path(__file__).parent / "bprime_mpp_vs_cantera.png"
fig.savefig(out, dpi=150)
print(f"\nSaved: {out}")

# ── Console summary ───────────────────────────────────────────────────────────
print_summary(
    lookup_mpp, lookup_tbl, lookup_live,
    label_ref="Mut++/NASA-9",
    label_tbl="Cantera tbl",
    label_live="Cantera live",
)
