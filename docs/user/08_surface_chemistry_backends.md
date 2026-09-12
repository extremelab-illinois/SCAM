<!-- SPDX-License-Identifier: MIT -->
# Surface chemistry backends

The surface solver needs an object with a
`lookup(T_wall, p_e, B'_g, Z_C_pyro=None) -> (B'_c, h_wall)` method.
Two interchangeable backends implement it, keyed by material name in the
`b_prime_tables` dict that `load_case`/`load_material` build for you; a
third exists for cross-checking against an external code.

## `BPrimeTable` — pre-computed table (default)

A `RegularGridInterpolator` over a pre-computed YAML table (3-D `(T_wall,
p_e, B'_g)`, or 4-D with `Z_C_pyro` when element transport is enabled).
Fast, and needs no Cantera at runtime. Generated offline by
`scam/tools/generate_bprime.py`. Referenced from a material card via
`b_prime_table:` — see
[`02_material_card_schema.md`](../reference/02_material_card_schema.md).

```{danger}
**Out-of-range lookups are silently clamped to the table edge.** `T_wall`,
`p_e`, `B'_g`, and `Z_C_pyro` are each passed through `np.clip` to the
table's own axis range before interpolation — there is no warning and no
extrapolation. A table whose pressure axis stops at 1 atm will quietly
return 1-atm chemistry for a 2-atm boundary layer, with nothing in the log
to say so.

Check a table's axis coverage against your case before trusting it near the
edges: for example, `calcarb_bprime_air.yaml` spans only 101 Pa–1 atm, while
`calcarb_bprime_air_mp.yaml` and `calcarb_bprime_ace_air.yaml` span
101 Pa–5 atm. A ballistic-reentry case peaking near 1.6 atm needs the 5-atm
table — the 1-atm one will run without complaint and simply be wrong.
```

## `BprimeEvaluator` — live Cantera equilibrium

A drop-in replacement for `BPrimeTable` (same `lookup` signature), backed by
live Cantera chemical-equilibrium evaluation instead of a pre-computed
table. Build it with:

```python
from scam.physics.bprime_evaluator import BprimeEvaluator
ev = BprimeEvaluator.from_config(
    "scam/materials/ablative_organic/tacot_v3.0_bprime_config.yaml"
)
```

Requires `pip install -e ".[bprime]"`. Warm-started (~0.3 ms/call in the
TACOT verification path). Enable it in a case deck with
`solver.bprime_runtime: true` together with `element_transport: true` (see
[`03_solver_options.md`](../reference/03_solver_options.md)); expect
roughly 10–100 ms per SEB Newton call, so use it for reference runs, not
everyday ones.

**This is the only backend that exposes `surface_enthalpies(T_w, p,
Z_C_pyro) -> (h_g, h_c)`**, and that method is what activates the SEB's
advective enthalpy terms — PATO's `qAdvPyro`/`qAdvChar`, the pyrolysis-gas
and char-oxidation enthalpy flux at the wall. A pre-computed `BPrimeTable`
only carries these terms if it was generated with the optional pretabulated
`h_g`/`h_c` columns; legacy tables omit them and the advective terms are
simply not applied.

## `MutationppEvaluator` — cross-check only

A third backend (`scam/physics/mpp_evaluator.py`) wraps the external
Mutation++ `bprime` tool behind the same `lookup(...)` interface. It is
optional and used only for cross-checking Cantera against Mutation++ — see
`examples/verification/ablation2/compare_pato_ablation2_live_vs_pretab.py`
and [`01_installation.md`](01_installation.md#optional-mutation-not-a-pip-dependency)
for the Mutation++ build/environment setup it needs.

## Which one runs for a given case

`load_case` builds the `b_prime_tables` dict from each material's
`b_prime_table:` companion file, so **`BPrimeTable` is what runs by
default** for any material card that sets that field. `BprimeEvaluator`
only replaces it when `bprime_runtime: true` is set on the solver options
*and* the material's chemistry is configured for element transport (see
[`03_solver_options.md`](../reference/03_solver_options.md)) — there is no
YAML switch that swaps backends for a plain table-only case; that swap
happens in the Python API (see the verification scripts under
`examples/verification/ablation2/` for worked examples of both paths).
