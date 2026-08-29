# SPDX-License-Identifier: MIT
"""CLI entry point: ``python -m scam case.yaml`` or ``scam case.yaml``.

Usage
-----
    scam case.yaml                 # run case, write output to path in deck
    scam case.yaml --verbose       # extra progress output
    scam case.yaml --output out/   # override output directory
    scam case.yaml --format hdf5   # override output format
    scam case.yaml --no-output     # run without writing files
    scam --version                 # print version and exit
"""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scam",
        description=(
            "SCAM — Simple Code for Ablative Materials.\n"
            "Run a 1-D material-response/ablation simulation from a YAML case deck."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("case", nargs="?", metavar="CASE.yaml",
                   help="Path to the YAML case deck.")
    p.add_argument("--version", action="store_true",
                   help="Print version and exit.")
    p.add_argument("--verbose", "-v", action="store_true", default=True,
                   help="Print solver progress (default: True).")
    p.add_argument("--quiet", "-q", action="store_true",
                   help="Suppress progress output.")
    p.add_argument("--output", "-o", metavar="DIR",
                   help="Override output directory from the case deck.")
    p.add_argument("--format", "-f", choices=["csv", "hdf5"],
                   help="Override output format (csv or hdf5).")
    p.add_argument("--no-output", action="store_true",
                   help="Run without writing any output files.")
    p.add_argument("--initial-T", type=float, default=300.0,
                   metavar="K",
                   help="Uniform initial temperature [K] (default: 300 K).")
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point for CLI and ``python -m scam``."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Lazy imports so --version doesn't need to load scipy etc.
    from scam import __version__

    if args.version:
        print(f"scam {__version__}")
        return 0

    if not args.case:
        parser.print_help()
        return 1

    # Deferred imports
    from scam.io.case_loader import load_case
    from scam.io.writers import write_results
    from scam.solvers.material_response import run

    verbose = args.verbose and not args.quiet

    try:
        case, mat_cards, b_prime_tables = load_case(args.case)
    except Exception as exc:
        print(f"ERROR loading case '{args.case}': {exc}", file=sys.stderr)
        return 2

    if verbose:
        print(f"Case: {case.name}")
        print(f"  t_end  = {case.options.t_end} s")
        print(f"  Layers = {[l.material_name for l in case.stack.layers]}")

    try:
        results = run(
            stack=case.stack,
            mat_cards=mat_cards,
            b_prime_tables=b_prime_tables,
            geom=case.geometry,
            surface_bc=case.surface_bc,
            back_bc=case.back_bc,
            options=case.options,
            initial_T=args.initial_T,
            verbose=verbose,
        )
    except Exception as exc:
        print(f"ERROR during simulation: {exc}", file=sys.stderr)
        raise  # re-raise so stack trace is visible

    if not args.no_output:
        out_path = args.output or case.output_path
        out_fmt  = args.format  or case.output_format
        try:
            write_results(results, out_path, fmt=out_fmt,
                          stack=case.stack, mat_cards=mat_cards)
        except Exception as exc:
            print(f"ERROR writing output: {exc}", file=sys.stderr)
            return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())
