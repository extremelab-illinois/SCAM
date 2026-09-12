<!-- SPDX-License-Identifier: MIT -->
# Python API tutorial

The CLI (`scam case.yaml`) is a thin wrapper around three calls:
`load_case` → `run` → `write_results`. Driving SCAM from Python directly
gives you two things the YAML deck cannot: the two solver options with no
YAML path (`array_backend`, `s_dot_prescribed` — see
[`03_solver_options.md`](../reference/03_solver_options.md)), and full
control over initial conditions and post-processing in the same script.

## Minimal end-to-end example

```python
from scam.config.boundary import BackBCConfig, SurfaceBCConfig, SurfaceBCType
from scam.config.geometry import GeometryConfig
from scam.config.solver import SolverOptions
from scam.config.stack import LayerConfig, StackConfig
from scam.io.material_loader import load_material
from scam.io.writers import write_results
from scam.solvers.material_response import run

mat, bpt = load_material("scam/materials/ablative_organic/tacot_v3.0.yaml")
mat_cards = {mat.name: mat}
b_prime_tables = {mat.name: bpt}

stack = StackConfig(layers=[
    LayerConfig(mat.name, thickness=0.05, n_nodes=51, n_subcells=4),
])
geom = GeometryConfig()

surface_bc = SurfaceBCConfig(
    alpha_conv=5000.0,
    T_aw=8000.0,
    rho_e_u_e=0.1,
    C_M=0.01,
    p_e=10000.0,
)
back_bc = BackBCConfig()

options = SolverOptions(t_end=120.0, dt_init=0.01, output_dt=1.0)

results = run(
    stack=stack,
    mat_cards=mat_cards,
    b_prime_tables=b_prime_tables,
    geom=geom,
    surface_bc=surface_bc,
    back_bc=back_bc,
    options=options,
    initial_T=300.0,
    verbose=True,
)

write_results(results, "results/my_run/", fmt="csv",
              stack=stack, mat_cards=mat_cards)
```

`write_results(..., stack=stack, mat_cards=mat_cards)` is what enables
`char_fraction_profiles.csv` and the per-component beta files — omit them
and you silently lose those two files (see
[`09_output_files.md`](09_output_files.md)).

## The outer time loop, in outline

`scam.solvers.material_response.run` does, per step:

1. Solve one in-depth timestep (mass + energy equations).
2. Apply surface recession (advance `s_total` by `s_dot · dt`).
3. Check the node-drop trigger; apply `drop_and_merge` if needed.
4. Adapt the timestep for the next step.
5. Save results at output intervals.
6. Repeat until `t_end`.

## Two options only reachable from Python

`array_backend` (default `"numpy"`; `"jax"` JIT-compiles selected
fixed-shape kernels, requires `pip install -e ".[jax]"`) and
`s_dot_prescribed` (a fixed recession rate `[m/s]` that overrides the
surface-mass-balance-computed `s_dot` every step, useful for verification
cases with an analytically known recession rate) are real `SolverOptions`
fields that `case_loader.py` does not parse from YAML — set them by
constructing `SolverOptions(...)` directly, as above.

## Inert, non-decomposing materials from scratch

For a synthetic material with no material card file at all (useful for
verification against an analytical solution), build a `MaterialCard`
directly:

```python
import numpy as np
from scam.config.material import MaterialCard

mat = MaterialCard(
    name="FiberFormLike",
    rho_virgin=180.0, rho_char=180.0, gamma_resin=0.0,
    decomposing=False,
    k_virgin_table=np.array([[300.0, 1.0], [3000.0, 1.0]]),
    k_char_table=np.array([[300.0, 1.0], [3000.0, 1.0]]),
    cp_virgin_table=np.array([[300.0, 710.0], [3000.0, 710.0]]),
    cp_char_table=np.array([[300.0, 710.0], [3000.0, 710.0]]),
)
```

`decomposing=False` skips the mass equation and decomposition subgrid
entirely — the pattern used by
`examples/other_examples/flat_plate_no_ablation.py` to validate pure
conduction against the analytical semi-infinite-slab erfc solution.

## Further examples

`examples/other_examples/` has three complete, runnable scripts:

| Script | Demonstrates |
|---|---|
| `flat_plate_no_ablation.py` | Inert conduction validated against the analytical erfc solution. |
| `full_ablation_tacot.py` | Full TACOT arc-jet case built entirely from the Python API. |
| `two_layer_backup.py` | A two-layer stack (ablator over a structural backing). |

Run any of them directly (`MPLBACKEND=Agg python3 examples/other_examples/full_ablation_tacot.py`)
to see a complete, working reference.
