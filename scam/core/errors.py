# SPDX-License-Identifier: MIT
"""SCAM exception hierarchy."""


class SCAMError(Exception):
    """Base class for all SCAM errors."""


class SCAMInputError(SCAMError):
    """Invalid user-supplied input (material card, case config, etc.)."""


class SCAMNumericsError(SCAMError):
    """Numerical failure (non-convergence, singular system, etc.)."""


class SCAMMeshError(SCAMError):
    """Mesh construction or remap failure."""


class SCAMIOError(SCAMError):
    """File read/write error."""
