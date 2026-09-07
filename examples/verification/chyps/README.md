<!-- SPDX-License-Identifier: MIT -->
# CHyPS BlaineTest comparison

Code-to-code comparison of SCAM against **CHyPS** on a 5 cm TACOT slab under
two-phase heating (0–60 s low flux, 60–90 s high flux, 90–100 s cooldown).

## Reference data — credit

The CHyPS results in `BlaineTest_chyps_results/` are **courtesy of
Dr. Blaine Vollmer, University of Illinois at Urbana-Champaign**, and are
redistributed here with permission for verification purposes.

They are *not* SCAM output and are not covered by SCAM's MIT licence in the
sense of authorship — please credit Dr. Vollmer and CHyPS in any work that uses
them, and do not treat them as SCAM-generated results.

## Note on the probe files

`BlaineTest_chyps_results/probes/T0`–`T8` are **column-trimmed** copies of the
original CHyPS probe output, reduced from the full 32–35 instrumentation columns
to only the quantities this comparison reads:

| file | columns kept |
|---|---|
| `T0` (surface) | `time [s]`, `mesh_displacement [m]`, `temperature [K]` |
| `T1`–`T8` | `time [s]`, `temperature [K]` |

All 2049 time samples are retained, so the comparison is **numerically
identical** to the one against the full files — this is a size reduction (14 MB
→ 0.4 MB), not a resampling. The discarded columns are CHyPS internals (species
fractions, permeability, per-solid cp/enthalpy, …) that the comparison never
reads.

`compare_chyps_blaine.py` accepts **either** layout: if a probe file has 32 or
more columns it is treated as raw CHyPS output, otherwise as the trimmed form.
So an original CHyPS run can be dropped in unchanged. Contact Dr. Vollmer for
the full-instrumentation output.

## Probe layout

Fixed depths from the original surface:

```
T0 = surface (Lagrangian, tracks recession)
T1 = 1 mm    T2 = 2 mm    T3 = 4 mm    T4 = 8 mm
T5 = 12 mm   T6 = 16 mm   T7 = 24 mm   T8 = back face (50 mm)
```

T1–T5 are consumed by the receding surface; T6–T8 survive to t = 100 s.

## Running it

```bash
MPLBACKEND=Agg python3 examples/verification/chyps/compare_chyps_blaine.py
```

Requires Cantera (`pip install -e ".[bprime]"`) for the live B′ backend.
Status and results: `docs/verification/verification.md`, §CHyPS BlaineTest.
