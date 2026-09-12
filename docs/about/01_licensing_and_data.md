<!-- SPDX-License-Identifier: MIT -->
# Licensing and third-party data

SCAM itself is distributed under the MIT License (see `LICENSE` at the
repository root).

**PATO.** [PATO](https://pato.ac/) is a separate project (built on
OpenFOAM, GPL-licensed) and is **not** redistributed here. This repository
ships only small, derived numerical outputs used as reference values for
the code-to-code comparisons in
[`13_examples_gallery.md`](../user/13_examples_gallery.md) — not PATO
source code. Obtain PATO itself from its own repository under its own
license if you need to reproduce the reference runs.

**CHyPS.** The CHyPS reference data used by
`examples/verification/chyps/` is courtesy of Dr. Blaine Vollmer
(University of Illinois at Urbana-Champaign), redistributed with
permission; see `examples/verification/chyps/README.md`.

**Amar (2006) digitized figure data.** The digitized figure data used by
the `examples/verification/amar_thesis/` comparisons is this project's own
digitization of a publicly distributed thesis.

```{note}
Reference data from a third party needs explicit clearance before it is
published — an `SPDX: MIT` header on a file does not make that file's
*contents* ours to license. Check this page (and the relevant script's
README) before adding another third-party dataset.
```
