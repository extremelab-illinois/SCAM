<!-- SPDX-License-Identifier: MIT -->
# Installation

Requires Python 3.10+.

```bash
git clone <repo>
cd SCAM
pip install -e ".[dev]"
```

`.[dev]` pulls in `pytest`, `pytest-cov`, and `h5py` on top of the runtime
dependencies (`numpy`, `scipy`, `pyyaml`, `matplotlib`). This is the extra
you want for day-to-day development and running the test suite.

## Optional extras

| Extra | Adds | Needed for |
|---|---|---|
| `hdf5` | `h5py` | `output.format: hdf5` (also included in `dev`). |
| `bprime` | `cantera`, `pandas` | Live Cantera surface-chemistry evaluation (`SolverOptions.bprime_runtime`) and regenerating B′ tables with `scam/tools/generate_bprime.py`. |
| `jax` | `jax[cpu]` | The experimental `SolverOptions.array_backend = "jax"` kernel path. |
| `docs` | `sphinx`, `myst-parser`, `furo`, `sphinx-copybutton` | Building this documentation site locally. |

```bash
pip install -e ".[hdf5]"
pip install -e ".[bprime]"
```

## Optional: Mutation++ (not a pip dependency)

[Mutation++](https://github.com/mutationpp/Mutationpp) is a compiled C++
library with its own CMake build, so it **cannot** be installed via `pip` and
is deliberately absent from the extras above. It is entirely optional —
Cantera covers every live-chemistry path in SCAM. You only need Mutation++ to
run the `MutationppEvaluator` backend, used by
`examples/verification/ablation2/compare_pato_ablation2_live_vs_pretab.py`
and by the `*_mpp_config.yaml` material configs.

Build it per its own instructions, then make sure its environment is on the
path. SCAM locates it via Mutation++'s own variables, falling back to
`~/Mutationpp`:

```bash
export MPP_DIRECTORY=/path/to/Mutationpp          # install root
export MPP_DATA_DIRECTORY="$MPP_DIRECTORY/data"   # mixtures/, thermo/, transport/
```

If it is missing, only that one comparison script fails, with a message
naming the binary it looked for.

## Troubleshooting

**`ModuleNotFoundError: No module named 'scam'`** — you installed into a
different Python than the one you are running. Confirm with
`python3 -c "import scam; print(scam.__file__)"` using the same interpreter
you invoke `python3 -m scam` with; if it's a venv, activate it or call
`.venv/bin/python3 -m scam ...` explicitly.

**Plots or animations do nothing / block forever** — set
`MPLBACKEND=Agg` before running any script that plots (`plot_results.py`,
the `examples/` scripts). Matplotlib's default interactive backend calls
`plt.show()`, which blocks in a headless environment.

**HDF5 output fails with `ImportError: h5py is required`** — install the
`hdf5` extra (`pip install -e ".[hdf5]"`); `dev` already includes it.

**A live-chemistry run is very slow** — `SolverOptions.bprime_runtime=True`
(or `element_transport=True` with a `BprimeEvaluator`) calls Cantera on every
SEB Newton iteration, at roughly 10–100 ms per call. This is expected for
reference-quality runs; use the pre-computed `BPrimeTable` (the default) for
everyday runs. See [`08_surface_chemistry_backends.md`](08_surface_chemistry_backends.md).

**A material fails to load with "cannot specify both 'kinetic_ablation' and
'b_prime_table'"** — a material card may declare at most one surface
ablation closure; see
[`02_material_card_schema.md`](../reference/02_material_card_schema.md).

**A case fails to load with "Stack layer material '...' not found"** — the
`material` name in a `stack.layers[]` entry must match a loaded card's
`name` field, not its filename. Make sure the card's YAML path is also
listed under the deck's `materials:` key.
