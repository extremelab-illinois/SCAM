# SPDX-License-Identifier: MIT
"""SCAM chemistry package — equilibrium B' chemistry, thermodynamics, and transport.

Provides:
  ThermochemBackend / ThermoBackend  — abstract interface
  CanteraBackend                     — Cantera implementation
  WallState                          — equilibrated wall state dataclass
  compute_bprime_case                — per-grid-point B' computation
  BPrimeCase                         — result dataclass
"""

from .thermo_backend import ThermochemBackend, ThermoBackend, CanteraBackend, WallState
from .thermochemistry import compute_bprime_case
from .models import BPrimeCase

__all__ = [
    "ThermochemBackend",
    "ThermoBackend",
    "CanteraBackend",
    "WallState",
    "compute_bprime_case",
    "BPrimeCase",
]
