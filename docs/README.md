<!-- SPDX-License-Identifier: MIT -->
# SCAM documentation

This folder is the source for the Sphinx documentation site (built with
`sphinx-build`, config in `conf.py`). It is organized by document purpose:

- `theory/` — the theory manual: derivations and analysis of SCAM's physics
  models, plus formulations extracted from PATO.
- `verification/` — verification ladders, PATO comparisons, and benchmark notes.
  Files matching `9[0-9]_*.md` are private (main-only); see CLAUDE.md.
- `planning/` — private roadmaps and implementation plans (excluded from the
  built site on every branch; see `conf.py`'s `exclude_patterns`).

To build the site locally:

```bash
pip install -e ".[docs]"
sphinx-build -b html -W --keep-going docs docs/_build/html
```

Then open `docs/_build/html/index.html`.
