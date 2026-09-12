<!-- SPDX-License-Identifier: MIT -->
# Material library

```{danger}
**UNVERIFIED MATERIAL DATA.** The material property cards in
`scam/materials/` are compiled from open literature sources. They have
**not** been independently verified against experimental data or validated
for use in engineering analysis. Property values marked `# ESTIMATE` in the
YAML files are based on engineering judgement or analogy from related
materials, not primary measurements. B′ tables marked `PROVISIONAL` were
generated with estimated pyrolysis gas compositions and have not been
validated against arc-jet data.

**These materials must not be used for scientific, engineering, or design
analysis until the relevant entries in the inventory below have been
independently checked and signed off.**
```

Materials are grouped into subdirectories by class under `scam/materials/`:
`ablative_organic/` (TACOT, PICA, AVCOAT, HEEET, cork, carbon phenolic,
ASTERM, ZURAM, NorCoat Liège), `ablative_carbon/` (FiberForm, Calcarb,
carbon-carbon), `ablative_hybrid/` (SLA-561V), `ablative_silicone/`
(RTV560, SIRCA, SLA-220), `ablative_silica/`, and `subsurface/` (inert
backing materials).

## Verification status

Materials are grouped by class below. **Key:** ✅ verified against
reference data · ⬜ provisional (literature values, not independently
validated) · n/a not applicable.

### Organic ablators (`ablative_organic/`)

| Material | File | Thermal props | Kinetics | B′ table | Primary source |
| --- | --- | :---: | :---: | :---: | --- |
| TACOT 2.2 | `tacot_v2.2.yaml` | ✅ | ✅ | ✅ | TACOT_2.2.xls; Goldstein kinetics |
| TACOT 3.0 | `tacot_v3.0.yaml` | ✅ | ✅ | ✅ | TACOT_3.0.xls; PATO AblationTestCase |
| TACOT 3.0 (3-rxn) | `tacot_v3.0_3rxn.yaml` | ✅ | ✅ | ✅ | Same; exact 3-reaction form |
| PICA (original) | `pica.yaml` | ⬜ | ⬜ | ⬜ | Tran 1997; Milos & Chen 2010 |
| AVCOAT | `avcoat.yaml` | ⬜ | ⬜ | ⬜ | Chen & Milos 1999; Si fiberglass neglected in B′ |
| Carbon phenolic | `carbon_phenolic.yaml` | ✅ | ⬜ | ⬜ | Amar 2006 App. D (mixed sources); char cp from Sutton NASA TN D-5930. Thermal props verified vs Amar §8.8; B′ table still unvalidated |
| Silica phenolic | `silica_phenolic.yaml` | ⬜ | ⬜ | n/a | MX-2600 analogy; Si surface chemistry not modelled |
| ASTERM | `asterm.yaml` | ⬜ | ⬜ | ⬜ | Zanetti 2015; IXV heritage |
| ZURAM 18/50 | `zuram.yaml` | ⬜ | ⬜ | ⬜ | AblaNTIS TN-2.2 / VKI+DLR measurements; card_version 2.0 |
| Cork | `cork.yaml` | ⬜ | ⬜ | ⬜ | Bouilly 2006; Natali 2012; estimates |
| Norcoat Liège | `norcoat_liege.yaml` | ⬜ | ⬜ | ⬜ | ExoMars/IXV heritage; estimates |
| HEEET outer | `heeet.yaml` | ⬜ | ⬜ | ⬜ | Venkatapathy 2009; 3-D woven architecture not captured |
| HEEET inner | `heeet_inner.yaml` | ⬜ | ⬜ | ⬜ | Same; PICA-like porous layer |

### Carbon ablators (`ablative_carbon/`)

| Material | File | Thermal props | Kinetics | B′ table | Primary source |
| --- | --- | :---: | :---: | :---: | --- |
| Calcarb CBCF 18/2000 | `calcarb.yaml` | ⬜ | n/a | ⬜ | Mersen technical guide; VKI AblaNTIS TN-2.2 |
| FiberForm | `fiberform.yaml` | ⬜ | n/a | n/a | Panerai et al. 2017; Weng & Martin 2014 |

### Silicone ablators (`ablative_silicone/`)

| Material | File | Thermal props | Kinetics | B′ table | Primary source |
| --- | --- | :---: | :---: | :---: | --- |
| RTV-560 | `rtv560.yaml` | ⬜ | ⬜ | n/a | Tran 1992; Milos 1997; Si B′ not modelled |
| SIRCA | `sirca.yaml` | ⬜ | ⬜ | n/a | Balboni 1999; Si B′ not modelled |
| SLA-220 | `sla220.yaml` | ⬜ | ⬜ | n/a | Balakrishnan 1999; Si B′ not modelled |

### Hybrid ablators (`ablative_hybrid/`)

| Material | File | Thermal props | Kinetics | B′ table | Primary source |
| --- | --- | :---: | :---: | :---: | --- |
| SLA-561V | `sla561v.yaml` | ⬜ | ⬜ | n/a | Milos & Chen 2004; Si B′ not modelled |

To update the table, replace ⬜ with ✅ once the relevant properties have
been checked against independent experimental or computational data, and
record the reference in the material YAML header.

## Card metadata, naming convention, and full inventory

The full per-file inventory, the `status`/`card_version`/`dataset_version`
metadata fields, and the B′ companion-file naming convention live in
`scam/materials/MATERIALS.md`, included here in full:

```{include} ../../scam/materials/MATERIALS.md
:start-line: 2
```
