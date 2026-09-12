<!-- SPDX-License-Identifier: MIT -->
# Surface Chemistry Modes

This note documents the `chemistry_mode` options used by
`examples/verification/ablation2/compare_pato_ablation2.py`.

All modes solve the same surface energy balance and use the same TACOT 3.0
material card.  They differ only in where the surface chemistry quantities come
from:

- `B'_c`: char blowing / char mass-loss coefficient
- `h_wall`: wall gas enthalpy used in the enthalpy-form convective term
- `h_g`, `h_c`: pyrolysis-gas and char enthalpies used by the PATO-style
  advective terms `qAdvPyro` and `qAdvChar`
- the fallback pyrolysis-gas composition when element transport does not provide
  `Z_C_pyro`

## Recommended Mode

Use `live_cantera_zc_default` for the base `AblationTestCase_2.x` validation.

The base case does not transport elements, so SCAM has no in-depth
`Z_C_pyro` field to pass to Cantera.  This mode asks Cantera for the carbon mass
fraction of the configured nominal TACOT 3.0 pyrolysis stream, then passes that
value back as the fallback `Z_C_pyro`.  Cantera therefore reconstructs the gas
through the same `Z_C_pyro -> pyro_x` path used by the element-transport cases.

This keeps the live Cantera `B'_c` / `h_wall` behavior close to the tabulated TACOT
chemistry while giving the PATO-consistent `qAdvPyro` enthalpy behavior.

## Mode Summary

| Mode | `B'_c` / `h_wall` source | `h_g` / `h_c` source | Fallback composition | Use |
|---|---|---|---|---|
| `live_cantera_zc_default` | live Cantera | live Cantera | Cantera-derived nominal `Z_C_pyro` | Default validation mode for base ablation2 |
| `live_cantera` | live Cantera | live Cantera | raw configured Cantera species string | Diagnostic raw-Cantera mode |
| `table_bc_live_h` | table `B'_c`, live Cantera `h_wall` | live Cantera | Cantera-derived nominal `Z_C_pyro` | Separates tabulated char mass loss from live enthalpies |
| `table_hybrid` | table `B'_c` / table `h_wall` | Cantera enthalpy differences shifted onto table `h_wall` | raw Cantera species string | Recession/mass-loss diagnostic |
| `table` | table `B'_c` / table `h_wall` | table-stored `h_g` / `h_c` | table generation's nominal `Z_C_pyro` | Runtime-Cantera-free table backend |

## Detailed Differences

### `live_cantera_zc_default`

This is the default:

```python
z_c_pyro = ev.pyrolysis_target_fraction("C")
backend = _DefaultZCBackend(ev, z_c_pyro)
```

When the surface solver calls `lookup(..., Z_C_pyro=None)` or
`surface_enthalpies(..., Z_C_pyro=None)`, the wrapper replaces `None` with the
nominal carbon mass fraction.  Cantera then calls its `Z_C_pyro` reconstruction
path rather than using the raw configured species string directly.

### `live_cantera`

This returns the raw `BprimeEvaluator`.

When `Z_C_pyro=None`, Cantera uses the configured nominal pyrolysis species string
directly.  This is useful for isolating Cantera's raw `B'_c` and `h_wall` behavior,
but for the TACOT 3.0 base ablation2 comparison its pyrolysis-gas enthalpy makes
the surface too cold.

### `table_bc_live_h`

This mode uses the table only for `B'_c`, so recession and char mass loss follow
the tabulated TACOT 3.0 B' data.  It gets `h_wall`, `h_g`, and `h_c` from live
Cantera using the nominal `Z_C_pyro` fallback.

Use this mode when you want to test whether a discrepancy is coming from char
mass loss (`B'_c`) or from the enthalpy terms in the surface energy balance.

### `table_hybrid`

This mode uses table `B'_c` and table `h_wall`, but takes Cantera enthalpy
differences for the advective terms:

```text
h_g = h_wall_table + (h_g_cantera - h_wall_cantera)
h_c = h_wall_table + (h_c_cantera - h_wall_cantera)
```

That keeps the table wall-enthalpy reference while adding PATO-style advective
enthalpy differences.  In the current script this mode uses raw Cantera composition
for the enthalpy differences, so it is mainly a diagnostic.

### `table`

This is the table-backend diagnostic mode.  `B'_c` and `h_wall` come from the
TACOT 3.0 B' table loaded from the material YAML.

The checked-in table stores `h_g(T,p)` and `h_c(T,p)` generated with the explicit
nominal `Z_C_pyro` convention used by the live validation backend.  No Cantera
calls are made at runtime.  This makes `table` the mode for verifying that
pretabulation reproduces the live surface-energy behavior.

On the 120 s base ablation2 comparison, the pretabulated and
`live_cantera_zc_default` paths give respectively 1568.11 K and 1567.48 K at
60 s (0.63 K difference), with only 0.004 mm difference in recession.  The
remaining spread is interpolation of the tabulated B′/enthalpy fields.

## Practical Debugging Order

For surface-temperature debugging, a useful order is:

1. `table`: check the tabulated B' backend and interpolation path.
2. `live_cantera_zc_default`: check the intended live validation path.
3. `table_bc_live_h`: hold `B'_c` fixed to the table while changing enthalpies.
4. `live_cantera`: inspect raw Cantera nominal-composition behavior.
5. `table_hybrid`: inspect table wall enthalpy plus Cantera enthalpy differences.

If `table` and `live_cantera_zc_default` agree closely, the B' table, live Cantera
lookup, and advective enthalpy references are likely consistent; remaining
surface-temperature differences should be sought in the SEB terms, radiation,
cooldown branch, or in-depth coupling.
