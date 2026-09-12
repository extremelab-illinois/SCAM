<!-- SPDX-License-Identifier: MIT -->
# Schema reference

Authoritative, hand-maintained reference tables for the YAML input formats:
the case deck, the material card, `SolverOptions`, and the output file
layout. These are written against what `scam.io.case_loader` and
`scam.io.material_loader` actually parse, not against the Python dataclasses
they parse into — the two are not always the same set of fields (see the
"documented but not read" notes below), and a schema page generated from the
dataclasses would silently advertise fields the loader ignores.

```{toctree}
:maxdepth: 2
:glob:

*
```
