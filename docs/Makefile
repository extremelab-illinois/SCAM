# SPDX-License-Identifier: MIT
# Documentation build targets. Run from the repo root as `make -C docs <target>`,
# or from this directory as `make <target>`. Requires `pip install -e ".[docs]"`.

SPHINXBUILD ?= python3 -m sphinx
SRCDIR      := $(CURDIR)
BUILDDIR    := $(SRCDIR)/_build
PUBSIM_DIR  ?= /tmp/scam-pubsim

.PHONY: help html strict clean pubsim leakcheck linkcheck coverage

help:
	@echo "Targets:"
	@echo "  html       Build the HTML site (warnings shown, not fatal)."
	@echo "  strict     Build with -W --keep-going (the real gate; matches RTD)."
	@echo "  pubsim     Simulate a public-release build from the working tree,"
	@echo "             without switching branches. See docs/pubsim.sh."
	@echo "  leakcheck  Grep an already-built HTML dir for private identifiers."
	@echo "             Usage: make leakcheck DIR=path/to/html"
	@echo "  linkcheck  Check external links (run on demand, not in CI)."
	@echo "  coverage   Docstring-coverage report (docs/_build/coverage)."
	@echo "  clean      Remove docs/_build."

html:
	$(SPHINXBUILD) -b html -d "$(BUILDDIR)/doctrees" "$(SRCDIR)" "$(BUILDDIR)/html"

strict:
	$(SPHINXBUILD) -b html -W --keep-going -d "$(BUILDDIR)/doctrees" "$(SRCDIR)" "$(BUILDDIR)/html"

pubsim:
	bash "$(SRCDIR)/pubsim.sh" "$(PUBSIM_DIR)"

leakcheck:
	bash "$(SRCDIR)/leakcheck.sh" "$(DIR)"

linkcheck:
	$(SPHINXBUILD) -b linkcheck "$(SRCDIR)" "$(BUILDDIR)/linkcheck"

coverage:
	$(SPHINXBUILD) -b coverage "$(SRCDIR)" "$(BUILDDIR)/coverage"
	@cat "$(BUILDDIR)/coverage/python.txt"

clean:
	rm -rf "$(BUILDDIR)"
