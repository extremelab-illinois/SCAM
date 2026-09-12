<!-- SPDX-License-Identifier: MIT -->
# Quickstart

## Run a built-in example

```bash
python3 -m scam examples/other_examples/tacot_arcjet.yaml
```

This runs a 5 cm TACOT slab under representative arc-jet stagnation heating
(`alpha_conv = 5000 W/(m²·K)`, `T_aw = 8000 K`) for 120 s and writes CSV
output to `results/`.

## Or drive it from Python

```bash
MPLBACKEND=Agg python3 examples/other_examples/full_ablation_tacot.py
```

`MPLBACKEND=Agg` avoids blocking on `plt.show()` in a headless environment.
See [`11_python_api_tutorial.md`](11_python_api_tutorial.md) for the
Python-API equivalent of the CLI path.

## Run the annotated templates

`examples/templates/` holds three heavily-commented input decks, each
mirroring a PATO `AblationTestCase` family — read them alongside
[`04_case_deck_walkthrough.md`](04_case_deck_walkthrough.md):

```bash
python3 -m scam examples/templates/tacot_arcjet_template.yaml
python3 -m scam examples/templates/ablation1_template.yaml
python3 -m scam examples/templates/ablation2_template.yaml
```

Then plot the results:

```bash
python3 examples/templates/plot_results.py results/ablation2_template/ --video --speed 30
```

See [`10_plotting_results.md`](10_plotting_results.md) for the full plotting
options, and the animated examples in the project README.

## Run the test suite

```bash
python3 -m pytest tests/ -q
```

Use `python3 -m pytest`, not a bare `pytest`, so the interpreter matching
your active environment is guaranteed to run.

## A minimal case deck

A flat plate with temperature-based convection, from scratch:

```yaml
name: "My Case"

geometry:
  type: slab

stack:
  layers:
    - material: TACOT
      thickness: 0.05   # metres
      n_nodes: 51
      n_subcells: 4

surface_bc:
  type: energy_balance
  alpha_conv: 5000.0    # W/m²/K
  T_aw: 8000.0          # K
  emissivity: 0.85
  rho_e_u_e: 0.1        # kg/m²/s  (edge mass flux for blowing)
  p_e: 10000.0          # Pa

back_bc:
  type: adiabatic

solver:
  t_end: 120.0
  dt_max: 0.5

materials:
  - path/to/scam/materials/ablative_organic/tacot_v3.0.yaml
```

Run it with `python3 -m scam my_case.yaml`. See
[`04_case_deck_walkthrough.md`](04_case_deck_walkthrough.md) and
[`01_yaml_case_deck.md`](../reference/01_yaml_case_deck.md) for every field
this deck could have used.
