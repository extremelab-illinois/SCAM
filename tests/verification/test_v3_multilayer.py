# SPDX-License-Identifier: MIT
"""V3 — Multi-material stack (interface conductance + contact resistance).

Adds layer-to-layer coupling (the interface conductance, and optionally a
thermal contact resistance) on top of the single-material conduction verified
in V1/V2. Two inert constant-property layers with different conductivities are
driven to steady state between fixed end temperatures.

Reference: one-dimensional **series thermal resistance**. With layers of
thickness L1, L2 and conductivities k1, k2 and an interface contact resistance
R [m^2 K/W]:

    q = (T_front - T_back) / (L1/k1 + R + L2/k2)              (constant flux)
    dT/dy|_layer = -q / k_layer                                (per-layer slope)
    interface temperature drop = q * R                         (contact jump)

The half-cell boundary treatment (see V1) puts a small first-order artifact on
the two nodes that straddle the interface, so the contact resistance is
verified *incrementally*: the extra interface jump when R is switched on
(R>0 jump minus R=0 jump) equals q*R. Flux and per-layer slopes are checked
directly.
"""
import numpy as np

from scam.config.boundary import (
    BackBCConfig, BackBCType, SurfaceBCConfig, SurfaceBCType,
)
from scam.config.solver import SolverOptions
from scam.config.stack import LayerConfig, StackConfig

from .conftest import make_inert_material, run_case


RHO, CP = 180.0, 710.0
K1, K2 = 2.0, 0.5          # W/m/K (conductive front layer, insulating backup)
L1, L2 = 0.025, 0.025
T_FRONT, T_BACK = 1500.0, 400.0


def _series_flux(R: float) -> float:
    return (T_FRONT - T_BACK) / (L1 / K1 + R + L2 / K2)


def _run(R: float):
    m1 = make_inert_material("M1", k=K1, rho=RHO, cp=CP)
    m2 = make_inert_material("M2", k=K2, rho=RHO, cp=CP)
    stack = StackConfig(layers=[
        LayerConfig("M1", L1, 51, 4),
        LayerConfig("M2", L2, 51, 4, contact_resistance=R),
    ])
    surface_bc = SurfaceBCConfig(
        bc_type=SurfaceBCType.PRESCRIBED_TEMP, T_prescribed=lambda t: T_FRONT,
    )
    back_bc = BackBCConfig(bc_type=BackBCType.PRESCRIBED_TEMP, T_back=T_BACK)
    t_diff = (L1 + L2)**2 / (min(K1, K2) / (RHO * CP))
    options = SolverOptions(
        t_end=40.0 * t_diff, dt_init=0.1, dt_max=t_diff / 20.0, dt_min=1e-3,
        dt_max_dT=1e4, output_dt=40.0 * t_diff,
    )
    return run_case(stack, {"M1": m1, "M2": m2}, surface_bc, back_bc, options,
                    initial_T=T_BACK).snapshots[-1]


def _interface_jump(snap):
    """T(last node of layer 0) - T(first node of layer 1)."""
    lid = snap.mesh.layer_id
    i0 = np.where(lid == 0)[0][-1]
    i1 = np.where(lid == 1)[0][0]
    return snap.T[i0] - snap.T[i1]


def test_v3_series_flux_no_contact():
    snap = _run(0.0)
    q_ref = _series_flux(0.0)
    rel = abs(snap.q_cond - q_ref) / q_ref
    assert rel < 0.01, (
        f"two-layer series flux: q_cond={snap.q_cond:.1f} vs series q={q_ref:.1f} "
        f"W/m^2 ({rel * 100:.2f}% > 1%)."
    )


def test_v3_per_layer_slopes():
    snap = _run(0.0)
    q_ref = _series_flux(0.0)
    y, T, lid = snap.mesh.y_nodes, snap.T, snap.mesh.layer_id
    for layer, k in ((0, K1), (1, K2)):
        idx = np.where(lid == layer)[0][2:-2]   # interior, away from boundaries
        slope = np.polyfit(y[idx], T[idx], 1)[0]
        ref_slope = -q_ref / k
        rel = abs(slope - ref_slope) / abs(ref_slope)
        assert rel < 0.02, (
            f"layer {layer} slope {slope:.1f} vs -q/k {ref_slope:.1f} "
            f"({rel * 100:.2f}% > 2%)."
        )


def test_v3_contact_resistance_jump():
    R = 1.0e-3
    jump_0 = _interface_jump(_run(0.0))     # pure half-cell artifact
    snap_R = _run(R)
    jump_R = _interface_jump(snap_R)
    # The incremental jump attributable to the contact resistance:
    delta_jump = jump_R - jump_0
    expected = snap_R.q_cond * R
    rel = abs(delta_jump - expected) / expected
    assert rel < 0.05, (
        f"contact-resistance jump: measured ΔT={delta_jump:.2f} K vs q*R="
        f"{expected:.2f} K ({rel * 100:.2f}% > 5%). "
        f"(R=0 artifact jump was {jump_0:.2f} K.)"
    )
