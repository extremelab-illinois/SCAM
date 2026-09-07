<!-- SPDX-License-Identifier: MIT -->
# AGENTS.md

**The project instructions for this repository live in [CLAUDE.md](CLAUDE.md).**

Read that file before making changes. It is the single source of truth for:

- build, test and example commands
- architecture and the module dependency order
- the critical numerical details (sign conventions, energy formulation,
  coordinate conventions) that are easy to get wrong
- material-card and B′-table conventions
- verification scripts and where their reference data lives

This file used to carry its own copy of that material. The copy drifted out of
date, so it was replaced with this pointer: two files describing one project
will always diverge, and a stale instruction file is worse than none. Everything
that was uniquely here has been merged into `CLAUDE.md`.

If you are adding project-wide guidance, put it in `CLAUDE.md`, not here.
