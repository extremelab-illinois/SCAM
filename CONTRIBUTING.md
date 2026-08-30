<!-- SPDX-License-Identifier: MIT -->
# Contributing to SCAM

Thanks for your interest in SCAM. This is a small research-code project
maintained by the [Extreme Lab at UIUC](https://github.com/extremelab-illinois);
contributions are welcome but reviewed on a best-effort basis.

## Before you start

- Open an issue first for anything beyond a small fix — a new physics model,
  a new material card, or a change to solver behavior — so we can agree on
  the approach before you invest time in it.
- For bug fixes and small improvements, a pull request directly is fine.

## Getting set up

```bash
git clone https://github.com/extremelab-illinois/SCAM.git
cd SCAM
pip install -e ".[dev]"
```

Optional extras, only needed if you're touching the relevant code path:

```bash
pip install -e ".[hdf5]"    # HDF5 output support
pip install -e ".[bprime]"  # live Cantera surface chemistry
```

## Running tests

```bash
pytest tests/ -q
```

Run a single file or test while iterating:

```bash
pytest tests/unit/test_decomposition.py -v
pytest tests/unit/test_tridiagonal.py::test_thomas_simple -v
```

Test layout, so you know where a new test belongs:

- `tests/unit/` — single-function correctness (geometry, Thomas algorithm,
  Arrhenius kinetics, properties)
- `tests/integration/` — physics-level checks (analytical erfc solution,
  energy conservation drift)
- `tests/verification/` — the analytical verification ladder V1–V5
  (conduction, multilayer, prescribed-T decomposition, surface balance)
- `tests/validation/` — full TACOT arcjet run against physics-calibrated
  bounds

A PR that changes solver behavior should include or update a test in the
relevant tier above. If you touch surface energy balance, recession, or
pyrolysis physics, also rerun the PATO verification scripts under
`examples/verification/` (`MPLBACKEND=Agg python3 <script>.py`) and check
the resulting plots against the reference — see `CLAUDE.md` for details on
what each script covers.

## Material data contributions

Material cards in `scam/materials/` are tracked as verified, provisional, or
estimate — see the disclaimer and verification table in `README.md`. If you
add or update a material:

- Mark new/unverified property values `# ESTIMATE` in the YAML and record
  the source in the card header.
- Do not flip a table entry to ✅ verified without an independent source
  (published data, an experimental comparison, or a code-to-code check) —
  say what that source is in the PR description.
- Follow the existing naming convention: `{material}_v{X}.yaml` when
  multiple dataset versions coexist, `_{variant}` suffix for model variants
  of the same dataset.

## Code style

- No new dependencies without discussion — this is deliberately a
  pure-Python core with optional extras.
- Follow the module dependency order documented in `CLAUDE.md` (`core` →
  `config` → `state`/`geometry`/`mesh`/`io` → `physics` → `numerics` →
  `solvers` → `cli`); don't add upward imports.
- Prefer small, focused commits and PRs — one physics fix or one feature
  per PR, not a mix.

## What's not in this repo

This is a sanitized public subset of a larger private codebase. The 2-D
axisymmetric solver and a few work-in-progress verification cases are
withheld pending further validation (see `CHANGELOG.md`) — don't be
surprised if an issue referencing 2-D functionality gets a "not yet public"
response rather than a fix.

## License

By contributing, you agree your contribution is licensed under the MIT
License that covers this repository (see `LICENSE`).
