# SPDX-License-Identifier: MIT
"""Results storage for time-history output.

Appended at each output step by the top-level driver.
All lists grow monotonically; convert to numpy arrays after the run.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Results:
    """Time-history output from a SCAM simulation.

    The primary accessor is ``snapshots``: a list of full :class:`SimState`
    objects at each saved output step.  Convenience array accessors
    (``times_array()``, ``T_wall_array()``, etc.) extract scalar quantities.
    """

    # Full SimState at each saved step (primary storage)
    snapshots: list = field(default_factory=list)

    # Thermocouple probe depths [m] from original surface (user-specified)
    tc_positions: list = field(default_factory=list)

    # Temperature at TC probe positions — list of 1-D arrays (n_tc,) per step
    tc_history: list = field(default_factory=list)

    def append(self, state, tc_temps: np.ndarray | None = None) -> None:
        """Append one snapshot from *state* (a SimState)."""
        self.snapshots.append(copy.deepcopy(state))
        if tc_temps is not None:
            self.tc_history.append(tc_temps.copy())

    # ------------------------------------------------------------------
    # Convenience accessors (extract scalars from snapshot list)
    # ------------------------------------------------------------------

    def times_array(self) -> np.ndarray:
        return np.array([s.time for s in self.snapshots])

    def s_array(self) -> np.ndarray:
        return np.array([s.mesh.s_total for s in self.snapshots])

    def s_dot_array(self) -> np.ndarray:
        return np.array([s.s_dot for s in self.snapshots])

    def T_wall_array(self) -> np.ndarray:
        return np.array([s.T_wall for s in self.snapshots])

    def q_cond_array(self) -> np.ndarray:
        return np.array([s.q_cond for s in self.snapshots])

    def tc_array(self) -> np.ndarray | None:
        """Shape (n_tc, n_steps), or None if no TC data was appended."""
        if not self.tc_history:
            return None
        arr = np.array(self.tc_history)   # (n_steps, n_tc)
        if arr.ndim == 2:
            return arr.T                   # (n_tc, n_steps)
        return None
