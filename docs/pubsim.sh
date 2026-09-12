#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Simulate a public-release checkout from the CURRENT WORKING TREE (not
# git HEAD -- so it picks up uncommitted docs work too) and build it,
# without ever switching this repo's own branch. This is the single
# highest-value check in the docs workflow: it proves a build is
# promotion-safe *before* you promote, not after. See CLAUDE.md's
# "The docs/ Sphinx build" section.
#
# Usage: docs/pubsim.sh [output-dir]   (default: /tmp/scam-pubsim)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-/tmp/scam-pubsim}"

echo "pubsim: building a simulated public-release tree at $OUT_DIR"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

rsync -a "$REPO_ROOT/docs/" "$OUT_DIR/docs/" --exclude _build
rsync -a "$REPO_ROOT/scam/" "$OUT_DIR/scam/" --exclude '__pycache__'
rsync -a "$REPO_ROOT/examples/" "$OUT_DIR/examples/" \
    --exclude '__pycache__' --exclude '*.png' --exclude '*.gif' \
    --exclude '*.mp4' --exclude 'results*'
rsync -a "$REPO_ROOT/tests/" "$OUT_DIR/tests/" --exclude '__pycache__'
cp "$REPO_ROOT/pyproject.toml" "$REPO_ROOT/README.md" "$OUT_DIR/"

# Drop everything public-release excludes -- see CLAUDE.md's exclusion list.
find "$OUT_DIR/docs" -regex '.*/9[0-9]_.*\.md' -delete
rm -rf "$OUT_DIR/docs/planning"
rm -rf "$OUT_DIR/scam"/{mesh2d,numerics2d,physics2d,solvers2d,state2d}
rm -f  "$OUT_DIR/scam/config/case2d.py" "$OUT_DIR/scam/io/case_loader_2d.py"

echo "pubsim: building docs"
python3 -m sphinx -b html -W --keep-going \
    -d "$OUT_DIR/docs/_build/doctrees" \
    "$OUT_DIR/docs" "$OUT_DIR/docs/_build/html"

echo "pubsim: checking for leaks"
"$REPO_ROOT/docs/leakcheck.sh" "$OUT_DIR/docs/_build/html"

echo "pubsim: checking module imports"
( cd "$OUT_DIR" && python3 -m pytest tests/unit/test_module_imports.py -q )

echo "pubsim: OK -- $OUT_DIR is a clean, promotion-safe simulation"
