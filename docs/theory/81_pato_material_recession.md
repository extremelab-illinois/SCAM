<!-- SPDX-License-Identifier: MIT -->
# Material Recession Calculation in PATO

In PATO, material recession is computed primarily as a **moving-mesh surface ablation problem**. The code first determines the surface mass-loss rate, converts it into a normal surface velocity, moves the mesh boundary, and then reports recession from the resulting surface displacement.

For the common **B′ ablation boundary condition**, the char mass flux at the wall is computed as

\[
\dot{m}_{c,w} = B'_{c,w}\,\rho_e u_e C_H\,F_b
\]

where \(B'_{c,w}\) is the char blowing parameter, \(\rho_e u_e C_H\) is the edge mass-transfer/heat-transfer quantity, and \(F_b\) is the blowing correction.

The surface recession speed is then obtained by dividing the wall char mass flux by the local solid density:

\[
V_\text{rec} = \frac{\dot{m}_{c,w}}{\rho_s}
\]

This velocity is applied as a normal mesh motion at the ablating boundary. The reported recession is computed geometrically as the distance between the initial and current surface positions:

\[
R = R_0 + |\mathbf{x}_{f,0} - \mathbf{x}_f(t)|
\]

Thus, PATO does not simply integrate a scalar recession rate. Instead, it moves the boundary mesh according to the ablation mass flux and evaluates recession from the resulting surface displacement.

## Role of Pyrolysis Gas Mass Flux

The pyrolysis gas mass flux, \(\dot{m}_{pyro}\), does **not** directly contribute to recession in the standard B′ surface-ablation path. In other words, the recession rate is not computed as

\[
V_\text{rec} \neq \frac{\dot{m}_{c,w} + \dot{m}_{pyro}}{\rho_s}
\]

Instead, recession is driven directly by the **surface char ablation mass flux**:

\[
V_\text{rec} = \frac{\dot{m}_{c,w}}{\rho_s}
\]

However, \(\dot{m}_{pyro}\) can affect recession **indirectly** through the surface balance. The pyrolysis gas flux contributes to the gas blowing parameter,

\[
B'_{g,w}
=
\max\left(
\frac{\dot{m}_{pyro}}
{\rho_e u_e C_H F_b},
0
\right)
\]

which enters the surface mass-balance calculation used to determine \(B'_{c,w}\). Therefore, pyrolysis gas can modify the surface chemistry solution, the blowing correction, and the resulting char ablation rate.

Pyrolysis gas can also affect the surface energy balance through an advective enthalpy term of the form

\[
q_{\text{adv,pyro}}
=
\dot{m}_{pyro}(h_g-h_w)
\]

This energy exchange can influence the wall temperature, which in turn can affect the surface chemistry and the computed value of \(B'_{c,w}\).

Therefore, the indirect coupling can be summarized as

\[
\dot{m}_{pyro}
\rightarrow
B'_{g,w},\ q_{\text{adv,pyro}},\ \text{blowing/surface energy balance}
\rightarrow
B'_{c,w}
\rightarrow
\dot{m}_{c,w}
\rightarrow
V_\text{rec}
\rightarrow
R
\]

PATO also includes a **volume-ablation path** for fibrous materials, where internal solid loss is computed through changes in solid volume fraction and fiber/tow radius. Boundary recession can then occur when material failure or erosion criteria are met. In both the B′ and volume-ablation approaches, the final recession output is based on the geometric movement of the material surface.
