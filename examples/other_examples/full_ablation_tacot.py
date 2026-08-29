# SPDX-License-Identifier: MIT
"""Full ablation TACOT arcjet case.

Representative TACOT conditions from the Ablation Workshop 2011 benchmark:
  - Stagnation heating: alpha_conv = 5000 W/m^2/K,  T_aw = 8000 K
  - Blowing: rho_e_u_e = 0.1 kg/m^2/s,  C_M = 0.01
  - Edge pressure: p_e = 10000 Pa (low-density arcjet)
  - Slab geometry, 5 cm thick, initially at 300 K
  - Run time: 120 s

Expected behaviour (order of magnitude):
  - T_wall: rises from ~300 K to ~2500-3000 K
  - s_dot: ~0.01-0.05 mm/s at quasi-steady (material dependent)
  - s_total at 120 s: ~1-4 mm (order of magnitude)
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


T_AW         = 8000.0   # [K]  adiabatic wall temperature
ALPHA_CONV   = 5000.0   # [W/m^2/K]  uncorrected Stanton number
RHO_E_UE     = 0.1      # [kg/m^2/s] freestream mass flux
C_M          = 0.01     # [-]  mass-transfer Stanton number
P_E          = 10000.0  # [Pa] edge pressure
T_INIT       = 300.0    # [K]
T_END        = 120.0    # [s]
THICKNESS    = 0.05     # [m]


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    mat_yaml = repo / "scam/materials/ablative_organic/tacot_v3.0.yaml"
    mat, bpt = load_material(str(mat_yaml))
    mat_cards = {mat.name: mat}
    b_prime_tables = {mat.name: bpt}

    stack = StackConfig(layers=[
        LayerConfig(mat.name, thickness=THICKNESS, n_nodes=51, n_subcells=4),
    ])
    geom = GeometryConfig(geometry_type=GeometryType.SLAB)

    surface_bc = SurfaceBCConfig(
        bc_type=SurfaceBCType.ENERGY_BALANCE,
        alpha_conv=ALPHA_CONV,
        T_aw=T_AW,
        emissivity=0.85,
        view_factor=1.0,
        T_rad_in=0.0,
        rho_e_u_e=RHO_E_UE,
        C_M=C_M,
        p_e=P_E,
        lambda_blowing=0.5,
    )
    back_bc = BackBCConfig(bc_type=BackBCType.ADIABATIC)

    options = SolverOptions(
        t_end=T_END,
        dt_init=0.01,
        dt_max=2.0,
        dt_min=1e-5,
        dt_max_dT=50.0,
        dt_max_drho_frac=0.05,
        output_dt=10.0,
        tc_positions=[0.005, 0.010, 0.025],  # 5, 10, 25 mm from original surface
    )

    print("Running full TACOT ablation case...")
    results = run(
        stack, mat_cards, b_prime_tables,
        geom, surface_bc, back_bc, options,
        initial_T=T_INIT,
        verbose=True,
    )

    # Write CSV output
    out_dir = Path(__file__).parent / "results_tacot_arcjet"
    write_csv(results, out_dir)

    # Summary
    times   = results.times_array()
    T_walls = results.T_wall_array()
    s_tots  = results.s_array()
    s_dots  = np.array([s.s_dot for s in results.snapshots])

    print(f"\n--- Summary at t = {T_END:.0f} s ---")
    print(f"  T_wall:  {T_walls[-1]:.1f} K")
    print(f"  s_total: {s_tots[-1]*1000:.3f} mm")
    print(f"  s_dot:   {s_dots[-1]*1000:.4f} mm/s")
    print(f"  n_nodes: {results.snapshots[-1].mesh.n_nodes_total}")

    tc = results.tc_array()
    if tc is not None:
        for i, pos in enumerate(results.tc_positions):
            print(f"  TC @ {pos*1000:.0f}mm: {tc[i, -1]:.1f} K")

    # Optional plot
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("TACOT Full Ablation — SCAM", fontsize=13)

        # T_wall and TCs
        ax = axes[0, 0]
        ax.plot(times, T_walls, "r-", linewidth=2, label="T_wall")
        if tc is not None:
            for i, pos in enumerate(results.tc_positions):
                ax.plot(times, tc[i], label=f"TC@{pos*1000:.0f}mm")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Temperature [K]")
        ax.set_title("Wall & in-depth temperatures")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Recession
        ax = axes[0, 1]
        ax.plot(times, s_tots * 1000, "b-", linewidth=2)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Recession [mm]")
        ax.set_title("Surface recession")
        ax.grid(True, alpha=0.3)

        # Recession rate
        ax = axes[1, 0]
        ax.plot(times, s_dots * 1000, "g-", linewidth=2)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Recession rate [mm/s]")
        ax.set_title("Surface recession rate")
        ax.grid(True, alpha=0.3)

        # Final temperature profile
        ax = axes[1, 1]
        snap = results.snapshots[-1]
        ax.plot(snap.mesh.y_nodes * 1000 + s_tots[-1] * 1000,
                snap.T, "r-", linewidth=2, label=f"t={T_END:.0f}s")
        # Also show mid-run snapshot
        mid_idx = len(results.snapshots) // 2
        snap_mid = results.snapshots[mid_idx]
        s_mid = s_tots[mid_idx]
        ax.plot(snap_mid.mesh.y_nodes * 1000 + s_mid * 1000,
                snap_mid.T, "b--", label=f"t={times[mid_idx]:.0f}s")
        ax.set_xlabel("Original depth [mm]")
        ax.set_ylabel("Temperature [K]")
        ax.set_title("Temperature profiles")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = out_dir / "tacot_arcjet.png"
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=150)
        print(f"\nPlot saved to: {plot_path}")
        plt.show()
    except ImportError:
        pass


if __name__ == "__main__":
    main()
