# SCAM Material Database

## Card fields

Every material YAML has three metadata fields (not used by the solver):

| Field | Type | Meaning |
|---|---|---|
| `dataset_version` | string | Upstream dataset release (e.g. `"3.0"` for TACOT 3.0). Omitted when no single authoritative version exists. |
| `card_version` | string | SCAM-internal revision. Increment when the card changes (start: `"1.0"`). |
| `status` | string | Data quality — see below. |

### Status values

- **`verified`** — validated against reference data (e.g. PATO ablation test cases). Suitable for production runs.
- **`provisional`** — values from peer-reviewed literature, loaded into SCAM but not yet independently validated against a reference solution. Use with care.
- **`estimate`** — placeholder or approximate values, explicitly noted in the description. Do not use for quantitative predictions.

---

## Naming convention

```
{material}.yaml              # single known version
{material}_v{X}.yaml         # when multiple dataset versions coexist (X = major version)
{material}_v{X}_{variant}.yaml  # same dataset, different model variant
```

**Examples**

| File | Dataset | Variant |
|---|---|---|
| `tacot_v2.2.yaml` | TACOT 2.2 | Goldstein 2-reaction kinetics |
| `tacot_v3.0.yaml` | TACOT 3.0 | 2-component Arrhenius (canonical) |
| `tacot_v3.0_3rxn.yaml` | TACOT 3.0 | 3-reaction form matching PATO `constantProperties` exactly |

Companion B′ files follow the pattern `{prefix}_{suffix}.yaml`:

| Suffix | Meaning |
|---|---|
| `_bprime_from_refXLS.TSV` | Raw copy-paste from the reference XLS dataset |
| `_bprime_from_refXLS.yaml` | Same data in SCAM `BPrimeTable` format (built from TSV) |
| `_bprime_air.yaml` | Cantera-generated pre-computed table (what SCAM uses in simulation) |
| `_bprime_config.yaml` | Config for `generate_bprime.py` to regenerate the Cantera B' table |

TACOT v2.2 and v3.0 share identical pyrolysis gas elemental composition (C:0.206, H:0.679,
O:0.115), so their bprime configs and `_bprime_air.yaml` files are chemically equivalent.

---

## Material classes

### `ablative_organic/`

Carbon- or polymer-matrix composites that pyrolyze under heating, forming a porous char layer and releasing pyrolysis gases. The dominant class for spacecraft heat shields. Includes TACOT (theoretical benchmark), PICA (Stardust, MSL), AVCOAT (Apollo, Orion), HEEET (multi-layer woven), carbon phenolic (high-ballistic-coefficient entries), cork and cork-phenolic composites (Ariane, IXV fairing), ASTERM and ZURAM (ESA re-entry), NorCoat Liège (ExoMars/IXV), and silica-fiber phenolic.

### `ablative_carbon/`

Non-decomposing porous carbon substrates. No pyrolysis; ablation occurs only through surface oxidation and sublimation. Used as the rigid fibrous preform beneath organic ablators (e.g. FiberForm under PICA) or as stand-alone insulator in arc-jet experiments.

### `ablative_hybrid/`

Silica-particle-filled organic matrix ablators where the char zone contains both carbon and silica residue. Surface chemistry is more complex (Si species in addition to C/O). SLA-561V (Mars landers) is the primary representative.

### `ablative_silicone/`

Silicone-rubber or silicone-based foam ablators. Char is predominantly SiO₂; surface B′ chemistry requires Si-capable equilibrium solvers. Used on secondary structures and lower heat-flux surfaces (RTV-560, SIRCA, SLA-220).

### `ablative_silica/`

Pure silica-fiber or quartz-based insulators. Ablation is negligible at typical entry heat fluxes; primary role is standoff insulation behind the primary ablator. No decomposition kinetics; thermal properties only.

### `subsurface/`

Inert structural or insulating sublayers that back the ablative stack. No chemistry, no recession. Includes synthetic templates for conduction verification and placeholder cards for bonding layers or structural panels.

---

## Inventory

| File | Name | Status | `dataset_version` | Primary source |
|---|---|---|---|---|
| `ablative_organic/tacot_v2.2.yaml` | TACOT | provisional | 2.2 | TACOT_2.2.xls; Goldstein kinetics |
| `ablative_organic/tacot_v3.0.yaml` | TACOT | verified | 3.0 | TACOT_3.0.xls; PATO constantProperties |
| `ablative_organic/tacot_v3.0_3rxn.yaml` | TACOT_3rxn | verified | 3.0 | Same as above, exact 3-rxn form |
| `ablative_organic/pica.yaml` | PICA | provisional | — | Tran 1997 / Milos & Chen 2010 |
| `ablative_organic/avcoat.yaml` | AVCOAT | provisional | — | Chen & Milos 1999; Dec & Braun 2006 |
| `ablative_organic/asterm.yaml` | ASTERM | provisional | — | Zanetti et al. 2015; Lachaud et al. 2017 |
| `ablative_organic/cork.yaml` | Cork | provisional | — | Bouilly 2006; Natali 2011; Candau 2012 |
| `ablative_organic/heeet.yaml` | heeet | provisional | — | Venkatapathy et al. 2009 |
| `ablative_organic/heeet_inner.yaml` | heeet_inner | provisional | — | Same source, inner-layer variant |
| `ablative_organic/carbon_phenolic.yaml` | CARBON_PHENOLIC | provisional | — | Amar 2006 thesis Appendix D, **except char cp** from Sutton NASA TN D-5930 Table VI(b) — Amar's Table D.4 cp column is erroneous (209 J/kg·K at 278 K); see docs/verification/verification.md §8.8 |
| `ablative_organic/narmco_4028.yaml` | NARMCO_4028 | estimate | TN D-5930 | Narmco 4028 carbon phenolic, **single-source** from Sutton NASA TN D-5930 Table VI. Thermal properties fully sourced; kinetics component split, reaction order, virgin emissivity and gas molar mass are flagged assumptions (Table VI gives no component densities). No B′ table — Sutton closes the surface with finite-rate char oxidation, recorded as metadata. See the KNOWN GAPS block in the card |
| `ablative_organic/silica_phenolic.yaml` | silica_phenolic | provisional | — | Multiple sources |
| `ablative_organic/teflon.yaml` | PTFE | estimate | — | Kemp 1968 kinetic (Eq. 12) ablation closure; validated vs. Steg 1962 / Yurevich 1973 within ±6% — see studies/teflon_ablation/ |
| `ablative_organic/teflon_bprime_legacy.yaml` | PTFE_bprime_legacy | estimate | — | Superseded Gibbs-equilibrium fluorine B′ table; kept for regression/comparison only (overpredicts kinetic rate by 4-9 orders — not physically endorsed) |
| `ablative_organic/zuram.yaml` | ZURAM | provisional | — | AblaNTIS TN-2.2 / VKI+DLR measurements (card_version 2.0) |
| `ablative_organic/norcoat_liege.yaml` | NorcoatLiege | estimate | — | Values estimated |
| `ablative_carbon/calcarb.yaml` | Calcarb | provisional | — | Mersen Calcarb technical guide; VKI AblaNTIS TN-2.2 |
| `ablative_carbon/carbon_carbon_amar2006.yaml` | CARBON_CARBON_AMAR2006 | provisional | — | Amar 2006 PhD thesis Table 8.12; k/cp from Potter & Kuntz (Sandia NL) |
| `ablative_carbon/carbon_carbon_amar2006_ace.yaml` | CARBON_CARBON_AMAR2006_ACE | provisional | — | Same as above, ACE-style B′ table (gri30, no C2/C3 sublimation) |
| `ablative_carbon/fiberform.yaml` | FiberForm | provisional | — | Panerai et al. 2017; Weng et al. 2020 |
| `ablative_carbon/graphite_mersen2340.yaml` | GRAPHITE_MERSEN_2340 | provisional | — | Ringel 2025 Ch. 4; Mersen grade 2340 brochure |
| `ablative_silicone/rtv560.yaml` | rtv560 | provisional | — | Tran et al. 1992; Milos 1997 |
| `ablative_silicone/sirca.yaml` | SIRCA | provisional | — | Balboni et al. 1999 |
| `ablative_silicone/sla220.yaml` | SLA-220 | provisional | — | Balakrishnan et al. 1999 |
| `ablative_hybrid/sla561v.yaml` | SLA-561V | provisional | — | Milos & Chen 2004 |
| `subsurface/fourier.yaml` | Fourier | verified | — | Synthetic template (constant k, cp, ρ) |
