<!-- SPDX-License-Identifier: MIT -->
# Changelog

All notable changes to SCAM are documented here. Versioning follows
[Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-08-29

Initial public release.

### Included

- 1-D Backward Euler FVM ablation solver (decomposition, pyrolysis gas flux,
  in-depth conduction, surface energy balance, recession).
- Slab, hollow-cylinder, and tabulated geometries.
- Optional element transport, Darcy/pressure-driven gas flow, and live
  Cantera equilibrium surface chemistry (B′).
- Material library across `ablative_organic/`, `ablative_carbon/`,
  `ablative_hybrid/`, `ablative_silicone/`, `ablative_silica/`, and
  `subsurface/` classes — see `scam/materials/MATERIALS.md` for per-material
  verification status.
- Verification suite: the V1–V5 analytical ladder, and PATO
  `AblationTestCase_1.x`/`2.x` code-to-code comparisons.

### Not included (withheld pending further validation)

- The 2-D axisymmetric solver — under active development, not yet
  converged against its PATO reference cases.
- Three verification cases that are still work in progress: the Amar (2006)
  carbon-carbon comparison, the Bianchi thesis case (not yet configured),
  and the CHyPS BlaineTest comparison (open discrepancy in deep in-depth
  probes).

These will be promoted into a future release once validated.
