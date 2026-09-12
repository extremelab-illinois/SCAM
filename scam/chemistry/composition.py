# SPDX-License-Identifier: MIT
"""Gas-composition string parsing for the SCAM chemistry package.

Parses ``"species:value,species:value,..."`` composition strings (mole
or mass fractions, depending on caller convention) into normalized
``{species: value}`` dicts, resolving a small set of case-insensitive
species-name aliases (e.g. ``"o2"`` -> ``"O2"``).
"""
_ALIASES = {
    "Ar": "AR",
    "ar": "AR",
    "argon": "AR",
    "Argon": "AR",
    "o2": "O2",
    "n2": "N2",
    "co": "CO",
    "co2": "CO2",
    "h2": "H2",
    "h2o": "H2O",
}


def normalize_composition_string(comp: str) -> str:
    pieces = []
    for raw in comp.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if ":" not in raw:
            raise ValueError(f"Composition entries must be species:value; got {raw!r}")
        sp, val = raw.split(":", 1)
        sp = _ALIASES.get(sp.strip(), sp.strip())
        pieces.append(f"{sp}:{val.strip()}")

    if not pieces:
        raise ValueError("Composition string is empty")

    return ",".join(pieces)
