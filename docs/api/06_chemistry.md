<!-- SPDX-License-Identifier: MIT -->
# Equilibrium chemistry (`scam.chemistry`)

The Cantera-backed equilibrium B′ computation, offline table generation, and
the newer MAT-style multicomponent surface equilibrium chemistry modules
(`scam.chemistry.mat`) -- element flux balance, surface constraints, failure
model, and heterogeneous surface reactions. This package is what
`scam/tools/generate_bprime.py` and `scam.physics.bprime_evaluator` build on.

## `scam.chemistry`

```{eval-rst}
.. automodule:: scam.chemistry
   :no-members:
```

The package re-exports `ThermochemBackend`, `ThermoBackend`, `CanteraBackend`,
`compute_bprime_case`, and `BPrimeCase` from the submodules documented below
(members are shown once, on their defining module, to avoid an ambiguous
cross-reference between the re-export and the definition).

## `scam.chemistry.composition`

```{eval-rst}
.. automodule:: scam.chemistry.composition
```

## `scam.chemistry.grids`

```{eval-rst}
.. automodule:: scam.chemistry.grids
```

## `scam.chemistry.models`

```{eval-rst}
.. automodule:: scam.chemistry.models
```

## `scam.chemistry.presets`

```{eval-rst}
.. automodule:: scam.chemistry.presets
```

## `scam.chemistry.thermo_backend`

```{eval-rst}
.. automodule:: scam.chemistry.thermo_backend
```

## `scam.chemistry.thermochemistry`

```{eval-rst}
.. automodule:: scam.chemistry.thermochemistry
```

## `scam.chemistry.tables`

```{eval-rst}
.. automodule:: scam.chemistry.tables
```

## `scam.chemistry.materials`

```{eval-rst}
.. automodule:: scam.chemistry.materials
```

## `scam.chemistry.mat.constraints`

```{eval-rst}
.. automodule:: scam.chemistry.mat.constraints
```

## `scam.chemistry.mat.failure`

```{eval-rst}
.. automodule:: scam.chemistry.mat.failure
```

## `scam.chemistry.mat.reactions`

```{eval-rst}
.. automodule:: scam.chemistry.mat.reactions
```

## `scam.chemistry.mat.solver`

```{eval-rst}
.. automodule:: scam.chemistry.mat.solver
```
