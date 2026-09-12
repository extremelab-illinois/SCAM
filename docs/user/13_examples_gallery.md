<!-- SPDX-License-Identifier: MIT -->
# Examples gallery

## Annotated templates (start here)

`examples/templates/` — see
[`04_case_deck_walkthrough.md`](04_case_deck_walkthrough.md):

| File | Case |
|---|---|
| `tacot_arcjet_template.yaml` | Every supported deck field, with defaults marked. |
| `ablation1_template.yaml` | Prescribed surface temperature, no recession (PATO `AblationTestCase_1.x`). |
| `ablation2_template.yaml` | Full SEB + B′ chemistry + ALE recession (PATO `AblationTestCase_2.x`). |
| `plot_results.py` | Standalone plotter — static PNG, `--video`, `--slab`. |

## Python-API worked examples

`examples/other_examples/` — see
[`11_python_api_tutorial.md`](11_python_api_tutorial.md):

| Script | Demonstrates |
|---|---|
| `tacot_arcjet.yaml` | A minimal, lightly-commented deck (the quickstart example). |
| `flat_plate_no_ablation.py` | Inert conduction vs. the analytical erfc solution. |
| `full_ablation_tacot.py` | Full TACOT arc-jet case, built entirely from the Python API. |
| `two_layer_backup.py` | A two-layer ablator-over-backing stack. |

## Verification against PATO

SCAM is verified against PATO's `AblationTestCase_1.x` (prescribed surface
temperature) and `2.x` (full energy balance with recession) benchmark
families. These are code-to-code comparisons (verification), not
code-to-experiment comparisons (validation) — see the
[verification record](../verification/index.md) for the full record of
physics fixes and residual gaps.

| Script | Case |
|---|---|
| `examples/verification/ablation1/compare_pato_ablation1.py` | Base case — prescribed T, no recession. |
| `examples/verification/ablation1/compare_pato_ablation1_grading.py` | Depth-graded density (multilayer approximation). |
| `examples/verification/ablation1/compare_pato_ablation1_multiPorousMat.py` | Two-material ablative stack. |
| `examples/verification/ablation1/compare_pato_ablation1_function.py` | Time-varying boundary conditions via Python callables. |
| `examples/verification/ablation1/compare_pato_ablation1_equilibriumElementConservation.py` | Element transport + 4-D B′ table. |
| `examples/verification/ablation2/compare_pato_ablation2.py` | Full SEB + recession. |
| `examples/verification/ablation2/compare_pato_ablation2_multiMat.py` | 3-layer TACOT 3.0 + inert sublayers. |
| `examples/verification/ablation2/compare_pato_ablation2_chemistryOff.py` | No surface chemistry. |
| `examples/verification/ablation2/compare_pato_ablation2_equilibriumElementConservation.py` | Cantera + element transport. |

Run any of these with:

```bash
MPLBACKEND=Agg python3 examples/verification/ablation2/compare_pato_ablation2.py
```

### You do not need PATO installed

The PATO reference outputs these scripts plot against are small text files,
committed to this repository under
`examples/verification/ablation1/pato_reference/` and
`examples/verification/ablation2/pato_reference/` (plus the FIAT reference
used by the ablation1 base case). Every script resolves the bundled copy
first and only falls back to a local `~/PATO-dev` checkout if it is absent
— that fallback exists for regenerating or extending the bundled set, not
for normal use. The same applies to PATO's TACOT `gasProperties`: the
conversion is committed as
`scam/materials/ablative_organic/tacot_v3.0_gasProperties_pT.yaml`.

Two exceptions:

- `examples/verification/ablation2/compare_pato_ablation2_live_vs_pretab.py`
  requires a Mutation++ build — see
  [`01_installation.md`](01_installation.md#optional-mutation-not-a-pip-dependency).
  Cantera alone is sufficient for every other case, including the other
  live-chemistry ones.
- The 2-D `AblationTestCase_3.x` comparison reads a live PATO case directory
  and is not part of this distribution.
