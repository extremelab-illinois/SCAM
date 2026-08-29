<!-- SPDX-License-Identifier: MIT -->
# PATO internal energy balance for a receding pyrolyzing material

**Scope.** This note summarizes the critical implementation-level information needed to understand or reimplement the internal energy balance used by PATO for a pyrolyzing material when the solid mesh can recede. It is focused on the one-temperature `PyrolysisEnergyModel` and the coupled pyrolysis / Darcy mass models in the NASA PATO repository.

## 1. Model selection

PATO selects the internal energy model at runtime from the material-region dictionary. The `Energy` subdictionary provides `EnergyType`; for a receding pyrolyzing material the relevant one-temperature model is typically:

```text
Energy
{
    EnergyType Pyrolysis;
}
```

The model is instantiated through `simpleEnergyModel::New(...)`, which looks up `EnergyType` in the OpenFOAM runtime-selection table.

## 2. Primary internal energy unknown

The solved temperature field is:

```cpp
Ta
```

In the `PyrolysisEnergyModel`, this appears as the volume field `T`, created from `createVolField("Ta")`.

The main material and gas fields coupled into the energy balance are:

| Field | Meaning |
|---|---|
| `rho_s` | bulk solid density field |
| `cp` | solid heat capacity |
| `k` | thermal-conductivity tensor |
| `pyrolysisFlux` | explicit energy/source term associated with solid mass variation by pyrolysis |
| `eps_g` | gas porosity / gas volume fraction |
| `rho_g` | gas density |
| `h_g` | gas enthalpy |
| `p` | pore-gas pressure |
| `M_g` | gas molecular weight |
| `mu_g` | gas viscosity |
| `K` | permeability tensor |
| `g` | gravity vector |

## 3. Receding-mesh internal energy residual

For a dynamic/receding mesh, PATO solves the finite-volume residual:

```cpp
solve
(
    rho_s*cp*(fvm::ddt(T) - fvm::div(mesh_.phi(), T))
  + pyrolysisFlux_
  + fvc::ddt(epsgRhogEg) - fvc::div(mesh_.phi(), epsgRhogEg)
  - fvm::laplacian(k, T)
  - fvc::laplacian(GammaHg, p)
  + fvc::div(phiHg)
);
```

In compact mathematical form, this corresponds to the residual

\[
\rho_s c_p
\left[
\frac{\partial T}{\partial t}
-
\nabla\cdot\left(\mathbf{u}_{mesh} T\right)
\right]
+
\dot{q}_{pyro}
+
\frac{\partial}{\partial t}\left(\epsilon_g \rho_g e_g\right)
-
\nabla\cdot\left(\mathbf{u}_{mesh}\,\epsilon_g \rho_g e_g\right)
-
\nabla\cdot\left(\mathbf{k}\nabla T\right)
-
\nabla\cdot\left(\Gamma_{Hg}\nabla p\right)
+
\nabla\cdot\left(\boldsymbol{\phi}_{Hg}\right)
=0.
\]

Here `mesh_.phi()` is the OpenFOAM mesh-motion flux. Therefore, **surface recession enters the internal energy equation through the dynamic-mesh/ALE correction terms**, not as a standalone recession-rate source in the energy equation.

## 4. Meaning of each energy term

| Code term | Physical role | Numerical treatment |
|---|---|---|
| `rho_s*cp*fvm::ddt(T)` | solid sensible-energy storage | implicit in `T`; `rho_s cp` evaluated from current fields |
| `-rho_s*cp*fvm::div(mesh_.phi(), T)` | ALE correction due to mesh motion/recession | implicit in `T` |
| `+ pyrolysisFlux_` | explicit energy contribution from solid mass variation by pyrolysis | explicit source term |
| `+ fvc::ddt(epsgRhogEg)` | gas internal-energy storage in pores | explicit |
| `- fvc::div(mesh_.phi(), epsgRhogEg)` | ALE correction for gas internal energy | explicit |
| `- fvm::laplacian(k, T)` | conduction through the solid/porous medium | implicit in `T`; `k` explicit |
| `- fvc::laplacian(GammaHg, p)` | pressure-gradient-driven gas enthalpy transport | explicit using current/previous `p` |
| `+ fvc::div(phiHg)` | gravity/buoyancy contribution to gas energy transport | explicit |

The sign convention above is the OpenFOAM residual form: all terms are assembled on the left-hand side and the residual is driven to zero.

## 5. Gas internal energy used by PATO

Before solving the energy equation, PATO updates:

```cpp
epsgRhogEg = eps_g*rho_g*h_g - eps_g*p;
```

That is,

\[
\epsilon_g \rho_g e_g
=
\epsilon_g \rho_g h_g - \epsilon_g p.
\]

This is the pore-gas internal-energy density written from the enthalpy relation \(\rho e = \rho h - p\), multiplied by the gas volume fraction.

## 6. Gas enthalpy transport coefficient

For the Darcy-law pyrolysis model, PATO computes

```cpp
GammaHg = ((h_g*p*M)/(mu*R*T))*K;
```

or

\[
\Gamma_{Hg}
=
\frac{h_g p M_g}{\mu_g R T}\,\mathbf{K}.
\]

This coefficient multiplies the pressure equation contribution in the energy residual as

\[
-\nabla\cdot\left(\Gamma_{Hg}\nabla p\right).
\]

PATO also defines

```cpp
Gamma_GHg = GammaHg*p*M/(R*T);
phiHg = linearInterpolate(Gamma_GHg & g) & mesh.Sf();
```

which provides the gravity/buoyancy gas-energy flux term `+ fvc::div(phiHg)`.

## 7. Coupling to the gas-mass equation

The pressure field used in the energy balance comes from the mass model. For the Darcy-law mass model, the dynamic-mesh gas-mass equation is assembled as

```cpp
solve
(
    fvm::ddt(Eta, p)
  - fvm::div(fvc::interpolate(Eta)*mesh_.phi(), p)
  - fvm::laplacian(Gamma, p)
  - piTotal
  + fvc::div(phiG)
);
```

with

\[
\eta = \frac{\epsilon_g M_g}{RT},
\qquad
\Gamma = \frac{pM_g}{\mu_gRT}\mathbf{K}.
\]

The pyrolysis gas production source is `piTotal`. Thus, pyrolysis affects the internal energy equation both through direct pyrolysis energy/source terms and indirectly through pressure-driven gas enthalpy transport.

## 8. Coupling to the pyrolysis model

For the linear Arrhenius pyrolysis model, each reaction-progress variable `Xsi` is advanced using an Arrhenius source. On a dynamic mesh, the governing update is implemented as

```cpp
solve
(
    fvm::ddt(Xsii)
  - fvm::div(mesh_.phi(), Xsii)
  - pow(mag(1 - Xsii), mp[i])
    * AField[i]
    * pow(Ta_, np[i])
    * exp(-Ep[i] / (R_ * Ta_)),
    "Xsii"
);
```

This corresponds to

\[
\frac{\partial \xi_i}{\partial t}
-
\nabla\cdot\left(\mathbf{u}_{mesh}\xi_i\right)
=
A_i T^{n_i}\exp\left(-\frac{E_i}{RT}\right)\left|1-\xi_i\right|^{m_i}.
\]

The pyrolysis mass-production rate for each reaction is then computed as

```cpp
piPyroReac[i]
= epsI_s[phase]
  * rhoI_s[phase]
  * Fp[i]
  * (fvc::ddt(Xsi[i]) - fvc::div(mesh_.phi(), Xsi[i]));
```

and the total source is

```cpp
piTotal = sum(piPyroReac[i]);
```

`piTotal` feeds the gas-mass equation. The solid phase densities are also updated from the reaction progress variables, and the material-property model subsequently updates the bulk `rho_s`, `cp`, and `k` used by the energy model.

## 9. What recession changes in the internal energy balance

For a receding pyrolyzing material, the critical distinction is that the internal energy equation is solved in a moving control volume. Relative to the fixed-mesh equation, PATO adds ALE correction terms:

```cpp
-rho_s*cp*fvm::div(mesh_.phi(), T)
-fvc::div(mesh_.phi(), epsgRhogEg)
```

The same moving-mesh correction also appears in the pyrolysis reaction-progress equation and in the gas-mass equation.

Therefore, a reimplementation should not simply solve a fixed-grid pyrolysis energy equation and separately subtract recession. Instead, the temperature, gas internal energy, reaction progress, and pore pressure must all be made consistent with the moving mesh.

## 10. Practical calculation sequence

A typical coupled update for a receding pyrolyzing material is conceptually:

1. Update material/gas properties from the current temperature, pressure, density, porosity, and composition fields.
2. Advance pyrolysis reaction progress `Xsi` with the dynamic-mesh correction.
3. Compute reaction gas-production rates `piPyroReac` and `piTotal`.
4. Update solid densities and derived material properties such as `rho_s`, `cp`, and `k`.
5. Solve the gas-mass/pressure equation using `piTotal`, permeability, gas properties, and mesh-motion correction.
6. Recompute gas-energy auxiliary fields:
   - `epsgRhogEg`
   - `GammaHg`
   - `Gamma_GHg`
   - `phiHg`
7. Solve the internal energy equation for `Ta`.
8. Apply boundary-condition coupling, including surface heat flux, ablation/recession, and mesh motion.

The exact ordering may be embedded in PATO's solver/model update loop, but the dependencies above are the important ones.

## 11. Key implementation points for reimplementation

- Use a finite-volume formulation if you want to match PATO closely.
- Treat solid storage and conduction implicitly in temperature.
- Treat gas internal-energy storage and gas enthalpy transport explicitly unless you intentionally redesign the coupling.
- Include mesh-motion/ALE terms for temperature, gas internal energy, reaction progress, and pressure whenever the material recedes.
- Do not treat `piTotal` as a heat source by itself. In PATO, `piTotal` is primarily the gas-mass source from pyrolysis and enters the pressure/mass equation; the energy equation has its own `pyrolysisFlux` field plus gas-energy transport terms.
- Remember that pyrolysis affects the internal energy balance both directly and indirectly:
  - directly through `pyrolysisFlux` and property changes;
  - indirectly through gas generation, pore pressure, gas enthalpy transport, and surface blowing/recession boundary coupling.
- For Darcy-Forchheimer variants, the same energy residual structure is used, but the gas enthalpy transport coefficient is modified by the Forchheimer resistance term.

## 12. Minimal pseudocode

```text
for each time step:
    move/update mesh if recession is active

    update gas and solid material properties

    solve Xsi equation with ALE correction
    compute piPyroReac and piTotal
    update solid densities and pyrolysis progress tau

    solve gas pressure equation with:
        storage
        ALE mesh correction
        Darcy pressure transport
        pyrolysis source piTotal
        gravity flux

    update epsgRhogEg = eps_g*rho_g*h_g - eps_g*p
    update GammaHg, Gamma_GHg, phiHg

    solve internal energy equation:
        solid storage + ALE correction
        pyrolysisFlux
        gas internal-energy storage + ALE correction
        conduction
        pressure-driven gas enthalpy transport
        gravity gas-energy flux

    apply surface boundary condition and recession model
```

## 13. Source files checked

- `src/applications/libraries/libPATOx/MaterialModel/EnergyModel/simple/simpleEnergyModel.C`  
  <https://raw.githubusercontent.com/nasa/pato/main/src/applications/libraries/libPATOx/MaterialModel/EnergyModel/simple/simpleEnergyModel.C>

- `src/applications/libraries/libPATOx/MaterialModel/EnergyModel/types/Pyrolysis/PyrolysisEnergyModel.C`  
  <https://raw.githubusercontent.com/nasa/pato/main/src/applications/libraries/libPATOx/MaterialModel/EnergyModel/types/Pyrolysis/PyrolysisEnergyModel.C>

- `src/applications/libraries/libPATOx/MaterialModel/EnergyModel/types/ForchheimerPyrolysis/ForchheimerPyrolysisEnergyModel.C`  
  <https://raw.githubusercontent.com/nasa/pato/main/src/applications/libraries/libPATOx/MaterialModel/EnergyModel/types/ForchheimerPyrolysis/ForchheimerPyrolysisEnergyModel.C>

- `src/applications/libraries/libPATOx/MaterialModel/MassModel/types/DarcyLaw/DarcyLawMassModel.C`  
  <https://raw.githubusercontent.com/nasa/pato/main/src/applications/libraries/libPATOx/MaterialModel/MassModel/types/DarcyLaw/DarcyLawMassModel.C>

- `src/applications/libraries/libPATOx/MaterialModel/PyrolysisModel/types/LinearArrhenius/LinearArrheniusPyrolysisModel.C`  
  <https://raw.githubusercontent.com/nasa/pato/main/src/applications/libraries/libPATOx/MaterialModel/PyrolysisModel/types/LinearArrhenius/LinearArrheniusPyrolysisModel.C>
