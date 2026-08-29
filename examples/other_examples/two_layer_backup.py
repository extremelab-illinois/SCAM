# SPDX-License-Identifier: MIT
"""Two-layer stack: TACOT over FiberForm backup.

Demonstrates:
- Multilayer stack with a contact resistance at the interface
- The front (surface) layer is ablating TACOT
- The back layer is inert FiberForm (carbon preform)
- TC probes in both layers

The temperature gradient and jump at the interface verify the
contact resistance implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scam.config.boundary import BackBCConfig, BackBCType, SurfaceBCConfig, SurfaceBCType
from scam.config.geometry import GeometryConfig, GeometryType
from scam.config.solver import SolverOptions
from scam.config.stack import LayerConfig, StackConfig
from scam.io.material_loader import load_material
from scam.io.writers import write_csv
from scam.solvers.material_response import run


def main() -> None:
    mat_dir = Path(__file__).resolve().parents[2] / "scam/materials"

    tacot_mat, tacot_bpt = load_material(str(mat_dir / "ablative_organic/tacot_v3.0.yaml"))
    fiber_mat, fiber_bpt = load_material(str(mat_dir / "ablative_carbon/fiberform.yaml"))

    mat_cards = {tacot_mat.name: tacot_mat, fiber_mat.name: fiber_mat}
    b_prime_tables = {tacot_mat.name: tacot_bpt, fiber_mat.name: fiber_bpt}

    # Stack: 2.5 cm TACOT + 2.5 cm FiberForm with small contact resistance
    stack = StackConfig(layers=[
        LayerConfig(tacot_mat.name, thickness=0.025, n_nodes=26, n_subcells=4,
                    contact_resistance=0.0),
        LayerConfig(fiber_mat.name,  thickness=0.025, n_nodes=26, n_subcells=4,
                    contact_resistance=1e-4),    # 1e-4 m^2 K/W interface resistance
    ])
    geom = GeometryConfig(geometry_type=GeometryType.SLAB)

    surface_bc = SurfaceBCConfig(
        bc_type=SurfaceBCType.ENERGY_BALANCE,
        alpha_conv=5000.0,
        T_aw=8000.0,
        emissivity=0.85,
        rho_e_u_e=0.1,
        C_M=0.01,
        p_e=10000.0,
    )
    back_bc = BackBCConfig(bc_type=BackBCType.ADIABATIC)

    # TC at 10 mm (TACOT), 20 mm (near interface), 35 mm (FiberForm), 45 mm (back)
    options = SolverOptions(
        t_end=60.0,
        dt_init=0.01,
        dt_max=1.0,
        dt_min=1e-5,
        dt_max_dT=50.0,
        output_dt=5.0,
        tc_positions=[0.010, 0.020, 0.035, 0.045],
    )

    print("Running two-layer TACOT + FiberForm case...")
    results = run(
        stack, mat_cards, b_prime_tables,
        geom, surface_bc, back_bc, options,
        initial_T=300.0,
        verbose=True,
    )

    out_dir = Path(__file__).parent / "results_two_layer"
    write_csv(results, out_dir)

    final = results.snapshots[-1]
    print(f"\n--- Final state at t=60 s ---")
    print(f"  T_wall:  {final.T_wall:.1f} K")
    print(f"  s_total: {final.mesh.s_total*1000:.3f} mm")
    tc = results.tc_array()
    if tc is not None:
        for i, pos in enumerate(results.tc_positions):
            print(f"  TC @ {pos*1000:.0f}mm: {tc[i,-1]:.1f} K")

    # Check interface temperature jump
    # The interface is near node index 25 (global)
    n_tacot = 26  # nodes in TACOT layer (after possible drops)
    n_nodes = final.mesh.n_nodes_total
    if n_nodes > n_tacot:
        T_left  = final.T[min(n_tacot - 1, n_nodes - 1)]
        T_right = final.T[min(n_tacot,     n_nodes - 1)]
        dT_interface = abs(T_left - T_right)
        print(f"\n  Temperature jump at interface: {dT_interface:.1f} K")

    # Optional plot
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Two-layer TACOT/FiberForm — SCAM", fontsize=13)

        # Temperature profiles at several times
        ax = axes[0]
        cmap = plt.get_cmap("plasma")
        n_snaps = len(results.snapshots)
        for idx, snap in enumerate(results.snapshots):
            color = cmap(idx / max(n_snaps - 1, 1))
            y = (snap.mesh.y_nodes + snap.mesh.s_total) * 1000
            ax.plot(y, snap.T, color=color, alpha=0.7)
        # Colorbar proxy
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                    norm=plt.Normalize(0, options.t_end))
        plt.colorbar(sm, ax=ax, label="Time [s]")
        ax.axvline(25.0, color="k", linestyle=":", label="Interface @ 25mm")
        ax.set_xlabel("Original depth [mm]")
        ax.set_ylabel("Temperature [K]")
        ax.set_title("Temperature profiles over time")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # TC time histories
        ax = axes[1]
        times = results.times_array()
        ax.plot(times, results.T_wall_array(), "r-", linewidth=2, label="T_wall")
        if tc is not None:
            for i, pos in enumerate(results.tc_positions):
                ax.plot(times, tc[i], label=f"TC@{pos*1000:.0f}mm")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Temperature [K]")
        ax.set_title("Thermal history")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = out_dir / "two_layer.png"
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=150)
        print(f"\nPlot saved to: {plot_path}")
        plt.show()
    except ImportError:
        pass


if __name__ == "__main__":
    main()
