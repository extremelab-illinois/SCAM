<!-- SPDX-License-Identifier: MIT -->
# Case deck walkthrough

`examples/templates/` holds three fully-annotated YAML decks, each mirroring
a PATO `AblationTestCase` family. They are the best starting point for a new
case — copy one and edit it, rather than writing a deck from a blank file.
This page walks through what each demonstrates; for the exhaustive field
list see [`01_yaml_case_deck.md`](../reference/01_yaml_case_deck.md).

## `tacot_arcjet_template.yaml` — every field, with defaults

This deck's own header states its purpose: **every supported field is
listed, with `[default: …]` markers on anything optional**. Read it top to
bottom the first time you write a deck from scratch.

```{literalinclude} ../../examples/templates/tacot_arcjet_template.yaml
:language: yaml
```

## `ablation1_template.yaml` — prescribed surface temperature

The PATO `AblationTestCase_1.x` equivalent: pure in-depth response
(conduction + Arrhenius pyrolysis), with the surface temperature imposed as
a step function rather than solved from a surface energy balance. No
recession, no SEB Newton iteration. Useful for verifying in-depth physics
in isolation.

Differences from the arcjet template above:

- `surface_bc.type: prescribed_temp` — no SEB Newton iteration.
- `allow_recession: false` — the surface does not recede.
- A much finer mesh (501 nodes vs. 51) and shorter `output_dt` (0.5 s vs.
  10 s), both for verification-grade resolution.

```{literalinclude} ../../examples/templates/ablation1_template.yaml
:language: yaml
```

## `ablation2_template.yaml` — full SEB, B′ chemistry, ALE recession

The PATO `AblationTestCase_2.x` equivalent: the full coupled response,
including the surface energy balance, B′ table surface chemistry, and
continuous (ALE) recession.

Differences from the arcjet template:

- **Enthalpy-based convection** (`rhoUeCH` + `h_r`) instead of
  temperature-based (`alpha_conv` + `T_aw`). When `rhoUeCH > 0`,
  `alpha_conv`/`T_aw` are ignored and
  `q_conv = rhoUeCH · (h_r − h_wall(T_w))` is used instead — see
  [`05_boundary_conditions.md`](05_boundary_conditions.md).
- `rho_e_u_e` is dropped to `0` during cooldown, gating the B′ chemistry
  lookup off — reproducing PATO's `chemistryOn = 0` flag for the cooldown
  phase. This avoids mixing enthalpy references when `h_r` is also zero; see
  the "cooldown SEB reference consistency" note in `CLAUDE.md`.
- `continuous_remap: true` — the ALE moving mesh, for a smooth `T_wall`
  instead of the Lagrangian scheme's small node-drop sawtooth.
- `blowing_model: "lees"` is set in this deck, matching PATO's
  `constantLambdaBlowingCorrectionModel` — but see the warning below.

```{warning}
This deck sets `blowing_model: "lees"`, and that value is **not currently
read by the YAML loader** — see the warning in
the `surface_bc` section of [`01_yaml_case_deck.md`](../reference/01_yaml_case_deck.md). The
solver runs with the dataclass default (`"rational"`) regardless. This is a
known gap in the loader, not something specific to this template; it is
called out here because this is the one shipped deck that happens to set the
field.
```

```{literalinclude} ../../examples/templates/ablation2_template.yaml
:language: yaml
```
