<!-- SPDX-License-Identifier: MIT -->
# CLI reference

```bash
scam CASE.yaml [options]
# or, equivalently:
python3 -m scam CASE.yaml [options]
```

`CASE.yaml` is positional and **optional**: with no case given, `scam` prints
help and exits with status `1`.

## Options

| Flag | Notes |
|---|---|
| `--version` | Print the installed version and exit. |
| `--verbose`, `-v` | Print solver progress (default: on). |
| `--quiet`, `-q` | Suppress progress output; overrides `--verbose`. |
| `--output DIR`, `-o DIR` | Override the output directory from the case deck's `output.path`. |
| `--format {csv,hdf5}`, `-f {csv,hdf5}` | Override the output format from the case deck's `output.format`. |
| `--no-output` | Run the simulation without writing any output files. |
| `--initial-T K` | Uniform initial temperature [K] (default: `300.0`). |

```{note}
`--initial-T` is the **only** way to set the initial temperature — there is
no `initial_T` key in the YAML deck. If you need a non-uniform initial
temperature field, use the Python API directly; see
[`11_python_api_tutorial.md`](11_python_api_tutorial.md).
```

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success. |
| `1` | No case file given. |
| `2` | Case failed to load (bad YAML, missing required field, unknown material). |
| `3` | Simulation succeeded but writing output failed. |

A simulation exception (as opposed to a load or write failure) is re-raised
after printing an `ERROR during simulation:` message, so the Python
traceback is visible — the CLI does not swallow solver errors.

## Examples

```bash
# Run with default output location and format from the deck
scam examples/templates/ablation2_template.yaml

# Override output directory and format
scam examples/templates/ablation2_template.yaml -o /tmp/run1 -f hdf5

# Dry run: check the deck loads and the solver runs, write nothing
scam examples/templates/ablation2_template.yaml --no-output

# Start from a hotter initial state
scam examples/templates/ablation1_template.yaml --initial-T 500
```
