<!-- SPDX-License-Identifier: MIT -->
# Recession and mesh motion

`SolverOptions.continuous_remap` (see
[`03_solver_options.md`](../reference/03_solver_options.md)) selects between
two ways of representing a receding, ablating surface. Both compute the
same physical recession rate `s_dot` from the surface mass balance
(`m_dot_char = rho_e_u_e · C_M · B'_c`, converted to a recession velocity via
`s_dot = m_dot_char / rho_s`); they differ only in how the mesh represents
that motion.

## Lagrangian fixed grid (default, `continuous_remap: false`)

Node positions (`y_nodes`) are fixed in the material frame. Each step:

1. `s_total` is incremented by `s_dot · dt`.
2. The surface cell's effective thickness (`delta_nodes[0]`) shrinks as
   `s_total` grows.
3. `area_nodes[0]` is updated (matters for the `hollow_cylinder` geometry).
4. Once `delta_nodes[0]` falls below `node_drop_threshold · h_nominal`, the
   thin surface node is dropped and merged into its neighbour
   (`mesh/remap.py::drop_and_merge`), conservatively in both energy
   (volume-weighted `T`) and mass (volume-weighted `ρ` and
   `ρ_components`).

This is simple and cheap, but each discrete node drop discretely exposes the
next, slightly cooler sub-surface node — which shows up as a small
(~±6 K) sawtooth on `T_wall`. Nodes are always dropped from the **back**
face of the ablating layer, deeper into the material, never from the front;
see `mesh/receding.py` and `mesh/remap.py` for the exact mechanism.

## Continuous moving mesh / ALE (`continuous_remap: true`)

Instead of shrinking one cell and dropping it, `apply_recession_ale`
redistributes the ablating (surface) layer's nodes uniformly between the new
surface position and that layer's fixed back face on **every** step,
keeping the node count constant. Fields are conservatively re-interpolated
onto the new node positions:

- `T`, `Z_elem`: linear interpolation at the new node centres.
- `rho_components`: linear interpolation at the new **nodelet** centres
  (preserving the within-cell decomposition sub-structure), from which the
  nodal density is recomposed.

Because the surface moves by only `s_dot·dt` per step, the interpolation
shift is tiny each time and `T_wall` evolves smoothly — there is no discrete
drop event. Only the surface layer (layer 0) moves; deeper layers are left
untouched. If the surface layer is fully consumed in one step
(`new_s >= y_back`), the implementation falls back to the scalar-shrink
Lagrangian step rather than producing a degenerate mesh.

**ALE needs a tighter timestep limit.** Because fields are re-interpolated
every step rather than held fixed until a drop, too large a step relative to
the recession rate causes re-interpolation wobble. `time_integration.py`
enforces a recession-CFL limit automatically when `continuous_remap: true`:
`dt ≤ 0.1 · h / s_dot`, where `h` is the local nominal node spacing.

## Which one to use

ALE matches PATO's own mesh-motion treatment and is what the
`ablation2_template.yaml` deck uses — see
[`04_case_deck_walkthrough.md`](../user/04_case_deck_walkthrough.md). The
Lagrangian scheme remains the default because it is simpler and the
sawtooth is usually immaterial for surface-temperature comparisons at
engineering tolerance; switch to ALE when a smooth `T_wall` time history
matters (e.g. for a Newton-solver-sensitivity study, or when the sawtooth
would be mistaken for a real physical oscillation in a plotted comparison).
