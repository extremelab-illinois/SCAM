<!-- SPDX-License-Identifier: MIT -->
# Adding a material card

A walkthrough for writing a new material YAML card from scratch. See
[`02_material_card_schema.md`](../reference/02_material_card_schema.md) for
the exhaustive field reference this page points into.

## 1. Pick a category and a filename

Place the file under the matching subdirectory of `scam/materials/`:
`ablative_organic/`, `ablative_carbon/`, `ablative_hybrid/`,
`ablative_silicone/`, `ablative_silica/`, or `subsurface/` (inert backing
materials). Follow the naming convention:

```text
{material}.yaml                    # a single known version
{material}_v{X}.yaml               # when multiple dataset versions coexist
{material}_v{X}_{variant}.yaml     # same dataset, different model variant
```

See [`06_material_library.md`](06_material_library.md) for worked examples
(`tacot_v3.0.yaml`, `tacot_v3.0_3rxn.yaml`).

## 2. Fill in the required fields

Every card needs `name`, `rho_virgin`, `rho_char`, `gamma_resin`, and the
five thermal property tables: `k_virgin`, `k_char`, `cp_virgin`, `cp_char`,
`h_g`. Loading fails immediately (`SCAMInputError`, naming the missing
table) if any of the five is absent — there is no default for these.

```yaml
name: MyMaterial

rho_virgin: 280.0     # [kg/m^3]
rho_char:   180.0     # [kg/m^3]
gamma_resin: 0.2      # [-]

k_virgin:  [[300, 0.30], [1000, 0.35], [3000, 0.40]]   # [[T_K, k_W_mK], ...]
k_char:    [[300, 0.90], [1000, 1.00], [3000, 1.10]]
cp_virgin: [[300, 1200], [1000, 1500], [3000, 1800]]   # [[T_K, cp_J_kgK], ...]
cp_char:   [[300, 1200], [1000, 1500], [3000, 1800]]
h_g:       [[300, 0.0], [3000, 3.0e6]]                 # [[T_K, h_g_J_kg], ...] sensible ref
```

All tables interpolate linearly and extrapolate flat (the edge value is
held) — a table only needs enough points to capture the real curvature, not
full coverage of every temperature the solver might reach.

## 3. Decide whether it decomposes

If the material pyrolyzes, add `components:` — one entry per Arrhenius
reaction, ordered fastest-first by convention:

```yaml
components:
  - name: resin_A
    rho_0: 60.0      # [kg/m^3] initial density of this component
    rho_r: 0.0       # [kg/m^3] residual density; 0 if it fully decomposes
    A_rate: 1.2e4    # [1/s]
    E_act: 8.7e4      # [J/mol]  (or E_over_R: <E_act/R> in [K] instead)
    m_exp: 3.0
    h_decomp: 1.63e6  # [J/kg], endothermic > 0
```

For a purely inert layer (a structural backing, an insulator with no
chemistry), set `decomposing: false` and leave `components` empty — leaving
`decomposing` at its default `true` with no components raises
`SCAMInputError`.

## 4. Choose a surface ablation closure — or none

If this material can be the **surface** (hot-face) layer and loses mass at
the wall, pick exactly one of:

- **`b_prime_table:`** for phase-equilibrium, transport-limited surface
  chemistry (the common case: carbon/organic ablators oxidizing or
  subliming under boundary-layer mass transfer). Point it at a companion B′
  table file (generate one with `scam/tools/generate_bprime.py`, or write a
  live-Cantera `*_bprime_config.yaml` — see
  [`08_surface_chemistry_backends.md`](08_surface_chemistry_backends.md)).
- **`kinetic_ablation:`** for reaction-rate-limited surface ablation (e.g.
  PTFE), where the mass-loss rate is a pure function of `T_w` and material
  constants, not of the boundary-layer mass-transfer coefficient. See the
  worked example in `scam/materials/ablative_organic/teflon.yaml`.
- Neither, if the material is never the hot-face layer (an inert backing, a
  sublayer under an ablator).

Setting both raises `SCAMInputError` ("ambiguous ablation closure — pick
one").

## 5. Set emissivity and porosity as needed

`emissivity` (virgin, default `0.85`) and `emissivity_char` (default `-1`,
meaning "same as virgin") are the minimum. If the material has meaningful
gas porosity (affects the in-depth gas energy-storage term), set
`eps_g_virgin`/`eps_g_char`; leaving both at `0.0` (default) skips that
correction entirely, which is correct for a dense, low-porosity material.

## 6. Add metadata and a status

```yaml
material_category: ablative_organic
dataset_version: "1.0"     # upstream source release, if there is one
card_version: "1.0"        # bump this when you revise the card
status: provisional        # verified | provisional | estimate
```

Be honest about `status`: `estimate` means values are placeholders or
analogy-based and must say so in the card's own comments;  `provisional`
means literature values not yet checked against a reference solution;
`verified` means checked against PATO or an equivalent independent
solution — reserve it for cards that have actually been through that
process. Update
[`06_material_library.md`](06_material_library.md)'s inventory table (kept
in `scam/materials/MATERIALS.md`) when you add or requalify a card.

## 7. Reference it from a case deck

```yaml
materials:
  - ../../scam/materials/ablative_organic/mymaterial.yaml

stack:
  layers:
    - material: MyMaterial   # matches the card's `name:` field, not the filename
      thickness: 0.03
      n_nodes: 51
```

Run the deck (`scam my_case.yaml`) and check the loader accepts it before
trusting any results — a typo in `material:` fails immediately with
`SCAMInputError` naming the loaded material set, which is the fastest way to
catch a mismatch between the card's `name:` and what the deck expects.
