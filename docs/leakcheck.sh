#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Grep a built HTML tree for private (main-only) identifiers that must never
# reach a public build. Run via `make -C docs leakcheck` after `make pubsim`,
# and always before promoting to public-release (see CLAUDE.md).
#
# This is a triage aid, not a fully automated oracle: it is allowed to
# mention a private module's *name* in prose explaining an exclusion (the
# CLAUDE.md exclusion list itself does exactly that), so a bare name match
# is not automatically a leak. Two such matches are known-audited and
# allowlisted below by exact built-file path; anything else that matches
# fails the check and needs the same kind of look before being cleared.
#
# Usage: docs/leakcheck.sh <built-html-dir>
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "usage: $0 <built-html-dir>" >&2
    exit 2
fi
OUT="$1"

if [ ! -d "$OUT" ]; then
    echo "leakcheck: '$OUT' is not a directory" >&2
    exit 2
fi

# Audited 2026-09: both are plain-text NAME mentions, not content leaks.
#   - scam/physics/gas_enthalpy.py (public, shared) has always had a
#     docstring comment naming `physics2d/gas_transport_2d.py` in passing
#     ("mirrors ..."); this predates the docs work entirely and already
#     ships in public-release's source today, so viewcode surfacing it is
#     not a new exposure.
#   - docs/api/02_config.md, 04_io.md, and theory/06_pressure_darcy_solve.md
#     each say, in prose, that a named private module/page "is documented
#     separately" on the main-only page -- no link, no content, by design
#     (see CLAUDE.md's "no public page links to a 9x_ page" rule, which the
#     -W build enforces separately from this script).
ALLOWLIST=(
    "_modules/scam/physics/gas_enthalpy.html"
    "api/05_physics.html"   # renders the same gas_enthalpy.py docstring via napoleon
    "api/02_config.html"
    "api/04_io.html"
    "theory/06_pressure_darcy_solve.html"
    "_sources/api/02_config.md.txt"
    "_sources/api/04_io.md.txt"
    "_sources/theory/06_pressure_darcy_solve.md.txt"
    "searchindex.js"   # site-wide index; necessarily contains every allowlisted page's text too
)

PAT='case2d|case_loader_2d|mesh2d|numerics2d|physics2d|solvers2d|state2d|two_dimensional|verification2d|uq_solver_development|uq_verification_plan|teflon_plan|scam_nd_roadmap'

is_allowlisted() {
    local f="$1"
    for a in "${ALLOWLIST[@]}"; do
        [[ "$f" == *"$a" ]] && return 0
    done
    return 1
}

fail=0
unexpected=()
while IFS= read -r -d '' f; do
    if ! is_allowlisted "$f"; then
        unexpected+=("$f")
    fi
done < <(grep -rilZE --include='*.html' --include='*.js' --include='*.txt' "$PAT" "$OUT" 2>/dev/null || true)

if [ "${#unexpected[@]}" -gt 0 ]; then
    echo "LEAK: private identifier found outside the audited allowlist:" >&2
    printf '  %s\n' "${unexpected[@]}" >&2
    fail=1
fi

if [ -d "$OUT/_modules/scam" ] && ls "$OUT/_modules/scam" 2>/dev/null | grep -qiE '2d'; then
    echo "LEAK: private module source under $OUT/_modules/scam/" >&2
    fail=1
fi

# The private repo's own origin URL must never appear in a public build
# (e.g. via a stray absolute link or a copied html_theme_options value).
if grep -rilE "github\.com/fpanerai/SCAM" "$OUT" 2>/dev/null; then
    echo "LEAK: private repo origin URL found in built HTML" >&2
    fail=1
fi

if [ "$fail" -eq 0 ]; then
    echo "leakcheck: clean -- $OUT (allowlisted, pre-audited mentions excluded)"
fi
exit "$fail"
