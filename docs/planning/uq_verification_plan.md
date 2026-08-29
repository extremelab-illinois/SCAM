# Verification Plan for SCAM UQ and Sensitivity Analysis Methods

## Purpose

This document defines a verification strategy for the SCAM uncertainty quantification and sensitivity analysis infrastructure.

The goal is to verify that the UQ layer correctly performs:

* distribution sampling;
* sample generation;
* parameter perturbation;
* deterministic ensemble execution;
* quantity extraction;
* uncertainty propagation;
* local sensitivity analysis;
* correlation-based sensitivity analysis;
* Morris screening;
* Sobol sensitivity analysis;
* restart and failure handling.

Verification should proceed from simple cases with exact analytical answers to increasingly realistic SCAM cases. The full charring-ablation solver should not be the first verification target. Instead, the UQ machinery should first be verified using synthetic models and simple thermal cases.

The recommended hierarchy is:

```text
1. Distribution tests
2. Sampling tests
3. Parameter perturbation tests
4. Analytical uncertainty-propagation tests
5. Local-sensitivity tests
6. Correlation-sensitivity tests
7. Morris screening tests
8. Sobol-index tests
9. End-to-end fake-solver tests
10. SCAM thermal integration tests
11. SCAM charring and ablation trend tests
```

## 1. Distribution Tests

The distribution module should be verified before any SCAM solver runs are performed.

### Uniform Distribution

For

[
X \sim U(a,b),
]

the exact mean and variance are

[
\mu_X = \frac{a+b}{2},
]

and

[
\sigma_X^2 = \frac{(b-a)^2}{12}.
]

Verification requirements:

* generate a large sample;
* compute the sample mean;
* compute the sample variance;
* compare against analytical values;
* verify that samples remain inside ([a,b]);
* verify that identical random seeds reproduce identical samples.

### Normal Distribution

For

[
X \sim \mathcal{N}(\mu,\sigma),
]

verify that

[
\bar{X} \rightarrow \mu,
]

and

[
s_X \rightarrow \sigma.
]

Verification requirements:

* sample mean approaches the prescribed mean;
* sample standard deviation approaches the prescribed standard deviation;
* the same random seed gives identical samples;
* different random seeds give different samples.

### Constant Distribution

The constant distribution should always return the nominal value.

Verification requirements:

* all generated samples are equal to the nominal value;
* the shape of the returned sample array is correct;
* the constant distribution works with all samplers that accept fixed parameters.

## 2. Sampling Tests

The sampling module should be verified independently of SCAM.

### Monte Carlo Sampling

Verification requirements:

* sample matrix has shape ((n_{\mathrm{samples}}, n_{\mathrm{parameters}}));
* the same seed gives identical samples;
* different seeds give different samples;
* sampled values respect prescribed bounds;
* sample statistics converge toward the expected distribution statistics.

### Latin Hypercube Sampling

For (N) Latin Hypercube samples, each uncertain variable should occupy each probability interval once:

[
\left[0,\frac{1}{N}\right],
\left[\frac{1}{N},\frac{2}{N}\right],
\dots,
\left[\frac{N-1}{N},1\right].
]

Verification requirements:

* generate (N) samples for each uncertain parameter;
* map samples to CDF space;
* divide CDF space into (N) bins;
* verify that each bin contains exactly one sample for each parameter;
* verify that the same seed gives identical LHS samples;
* verify that samples remain inside prescribed bounds.

This test ensures that the LHS implementation is actually stratified and not simply random Monte Carlo.

### One-at-a-Time Sampling

Verification requirements:

* baseline sample is present;
* each parameter is perturbed independently;
* only one parameter changes per perturbation;
* all other parameters remain at nominal values;
* the number of generated samples is correct.

### Finite-Difference Sampling

For centered finite differences, each uncertain parameter should produce a negative and positive perturbation.

Verification requirements:

* baseline sample is present;
* each parameter has (p_i - \Delta p_i) and (p_i + \Delta p_i) samples;
* perturbations are symmetric around the nominal value;
* the number of generated samples is (1 + 2n_{\mathrm{parameters}});
* bounded parameters do not exceed prescribed limits.

## 3. Parameter Perturbation Tests

The parameter-perturbation module is critical because it maps abstract uncertain variables into SCAM input files and material cards.

### Scalar Replacement

Baseline:

```yaml
surface:
  emissivity: 0.85
```

Sample:

```text
emissivity = 0.90
```

Expected perturbed case:

```yaml
surface:
  emissivity: 0.90
```

Verification requirement:

* the target value is replaced exactly by the sampled value.

### Scalar Scaling

Baseline:

```yaml
boundary:
  front:
    heat_flux: 200000.0
```

Sample:

```text
heat_flux_scale = 1.10
```

Expected perturbed value:

[
q''_{\mathrm{pert}} = 1.10 q''_0 = 220000 \ \mathrm{W/m^2}.
]

Verification requirement:

* the target value is multiplied by the sampled scale factor.

### Scalar Offset

Baseline:

```yaml
surface:
  temperature_initial: 300.0
```

Sample:

```text
temperature_offset = 20.0
```

Expected perturbed value:

[
T_{\mathrm{pert}} = T_0 + 20 = 320 \ \mathrm{K}.
]

Verification requirement:

* the target value is incremented by the sampled offset.

### Curve Scaling

Baseline:

[
k(T) = [0.1, 0.2, 0.4].
]

Sample:

[
\alpha_k = 1.2.
]

Expected perturbed curve:

[
k_{\mathrm{pert}}(T) = [0.12, 0.24, 0.48].
]

Verification requirements:

* all tabulated values are scaled consistently;
* temperature coordinates are unchanged.

### Curve Offset

Baseline:

[
c_p(T) = [800, 1000, 1200].
]

Sample:

[
\Delta c_p = 50.
]

Expected perturbed curve:

[
c_{p,\mathrm{pert}}(T) = [850, 1050, 1250].
]

Verification requirements:

* the offset is added to all tabulated values;
* temperature coordinates are unchanged.

### Linear-Tilt Curve Perturbation

For a baseline curve (y_0(T)), a linear-tilt perturbation may be written as

[
y(T)
====

y_0(T)
\left[
1 + a + b\frac{T-T_{\mathrm{ref}}}{T_{\max}-T_{\min}}
\right].
]

Verification requirements:

* the tilt factor is evaluated correctly at each temperature;
* temperature coordinates are unchanged;
* the curve remains finite;
* optional physical bounds are enforced.

### Baseline Immutability

After applying a perturbation, the original baseline case must remain unchanged.

Verification requirement:

* the perturbed case differs from the baseline, but the baseline object remains exactly equal to its original value.

This is one of the most important UQ infrastructure tests because accidental in-place modification can corrupt an entire ensemble.

## 4. Analytical Uncertainty-Propagation Tests

Use simple synthetic models with exact propagated statistics.

### Additive Linear Model

Let

[
Q = aX_1 + bX_2 + c,
]

where (X_1) and (X_2) are independent.

The exact expected value is

[
\mathbb{E}[Q]
=============

a\mathbb{E}[X_1]
+
b\mathbb{E}[X_2]
+
c.
]

The exact variance is

[
\mathrm{Var}(Q)
===============

a^2\mathrm{Var}(X_1)
+
b^2\mathrm{Var}(X_2).
]

Verification requirements:

* computed mean agrees with the analytical mean;
* computed variance agrees with the analytical variance;
* computed standard deviation agrees with the analytical value;
* sample convergence improves as (n_{\mathrm{samples}}) increases.

This test verifies the propagation module, sample execution, and quantity aggregation.

### Product Model

Let

[
Q = X_1X_2,
]

with independent inputs.

The exact expected value is

[
\mathbb{E}[Q]
=============

\mathbb{E}[X_1]\mathbb{E}[X_2].
]

The exact variance is

[
\mathrm{Var}(Q)
===============

## \mathbb{E}[X_1^2]\mathbb{E}[X_2^2]

\mathbb{E}[X_1]^2\mathbb{E}[X_2]^2.
]

Verification requirements:

* computed mean agrees with the analytical mean;
* computed variance agrees with the analytical variance;
* nonlinear propagation is handled correctly.

### Threshold Exceedance

Let

[
Q = X,
]

where

[
X \sim U(0,1).
]

Then the exact exceedance probability is

[
P(Q > q_{\mathrm{lim}})
=======================

1 - q_{\mathrm{lim}}.
]

For example,

[
P(Q > 0.8) = 0.2.
]

Verification requirements:

* threshold exceedance probability is computed correctly;
* edge cases (q_{\mathrm{lim}} = 0) and (q_{\mathrm{lim}} = 1) are handled correctly.

## 5. Local Sensitivity Tests

Local finite-difference sensitivity should be verified using models with exact derivatives.

### Polynomial Model

Let

[
Q = aX_1 + bX_2^2.
]

The exact derivatives are

[
\frac{\partial Q}{\partial X_1} = a,
]

and

[
\frac{\partial Q}{\partial X_2} = 2bX_2.
]

The normalized sensitivity is

[
S_i =
\frac{\partial Q}{\partial X_i}
\frac{X_i}{Q}.
]

Therefore,

[
S_1
===

a\frac{X_1}{Q},
]

and

[
S_2
===

# 2bX_2\frac{X_2}{Q}

\frac{2bX_2^2}{Q}.
]

Verification requirements:

* finite-difference derivative for (X_1) agrees with (a);
* finite-difference derivative for (X_2) agrees with (2bX_2);
* normalized sensitivities agree with analytical values;
* the sign of each sensitivity is correct.

### Perturbation-Size Convergence

Use relative perturbation sizes such as

[
\frac{\Delta p}{p}
==================

10^{-2},
10^{-3},
10^{-4}.
]

Verification requirements:

* centered finite-difference sensitivity converges as perturbation size decreases;
* roundoff-dominated behavior is identified for excessively small perturbations.

## 6. Correlation Sensitivity Tests

Correlation-based sensitivity metrics should be verified using synthetic models with known qualitative behavior.

### Linear Monotonic Model

Let

[
Q = 3X_1 - 2X_2.
]

Expected behavior:

* Pearson correlation with (X_1) is positive;
* Pearson correlation with (X_2) is negative;
* Spearman correlation with (X_1) is positive;
* Spearman correlation with (X_2) is negative.

### Nonlinear Monotonic Model

Let

[
Q = X_1^3.
]

Expected behavior:

* Spearman correlation between (X_1) and (Q) is high;
* Pearson correlation is positive but may differ from Spearman.

This verifies that Spearman correlation correctly captures monotonic nonlinear relationships.

### Non-Monotonic Model

Let

[
Q = (X_1 - 0.5)^2,
]

with

[
X_1 \sim U(0,1).
]

Expected behavior:

* Pearson correlation may be small;
* Spearman correlation may be small;
* (X_1) still controls (Q).

This test is important because it demonstrates the limitation of correlation-based sensitivity metrics. Correlation metrics are useful screening tools, but they do not detect all forms of dependence.

## 7. Morris Screening Tests

Morris screening should be verified using simple models with known parameter rankings.

### Linear Additive Model

Let

[
Q = a_1X_1 + a_2X_2 + a_3X_3.
]

Expected behavior:

* Morris (\mu^*) ranking follows (|a_1|), (|a_2|), (|a_3|);
* Morris (\sigma) is approximately zero.

Because the model is linear and additive, there are no nonlinear effects or interactions.

### Interaction Model

Let

[
Q = X_1 + X_2 + cX_1X_2.
]

Expected behavior:

* (X_1) and (X_2) have nonzero Morris (\sigma);
* larger (c) produces larger (\sigma).

The nonzero (\sigma) indicates nonlinear behavior or interaction effects.

### Inactive Parameter Test

Let

[
Q = X_1 + 2X_2,
]

while (X_3) is included in the input list but does not appear in (Q).

Expected behavior:

* (X_3) has (\mu^* \approx 0);
* (X_3) has (\sigma \approx 0).

This is a useful regression test for detecting parameter-indexing mistakes.

## 8. Sobol Index Tests

Sobol analysis is optional and should be available only when SALib is installed. If Sobol analysis is implemented, it should be verified using analytical or benchmark cases.

### Additive Linear Model

Let

[
Q = a_1X_1 + a_2X_2 + a_3X_3,
]

with independent inputs.

The analytical first-order Sobol index is

[
S_i
===

\frac{a_i^2\mathrm{Var}(X_i)}
{\sum_j a_j^2\mathrm{Var}(X_j)}.
]

Because the model is additive,

[
S_{T_i} = S_i.
]

Expected behavior:

* first-order Sobol indices match analytical values;
* total-order Sobol indices match first-order indices;
* the sum of first-order indices is approximately one.

### Interaction Model

Let

[
Q = X_1 + X_2 + cX_1X_2.
]

Expected behavior:

* total-order indices are larger than first-order indices;
* (S_T - S_1) indicates interaction contribution;
* interaction contribution increases with (c).

### Ishigami Benchmark

The Ishigami function is a standard nonlinear sensitivity-analysis benchmark:

[
Q
=

\sin X_1
+
a \sin^2 X_2
+
b X_3^4 \sin X_1,
]

with

[
X_i \sim U(-\pi,\pi).
]

Verification requirements:

* Sobol implementation reproduces known Ishigami sensitivity-index trends;
* (X_1) and (X_2) have dominant first-order effects;
* (X_3) has little or no first-order effect but contributes through interactions;
* the total-order index of (X_3) is nonzero.

This benchmark verifies nonlinear and interaction effects.

## 9. End-to-End Fake-Solver Tests

Before connecting to full SCAM simulations, the complete UQ pipeline should be tested with a fake deterministic solver.

Example fake solver:

[
Q = 2X_1 + 5X_2^2.
]

The fake solver should mimic the structure of a SCAM result object or output file, including scalar and time-dependent quantities where possible.

Verification requirements:

* read `uq.yaml`;
* generate samples;
* apply parameter perturbations;
* run the fake solver for each sample;
* extract quantities of interest;
* compute uncertainty statistics;
* compute sensitivity metrics;
* write output files;
* restart without rerunning completed samples;
* record failed samples cleanly.

This test verifies the orchestration infrastructure without involving thermal-solver details.

## 10. SCAM Thermal Integration Tests

After the UQ infrastructure is verified with synthetic models, it should be tested with simple SCAM cases that have analytical or nearly analytical behavior.

### Steady 1-D Conduction

Consider a 1-D slab with constant properties, prescribed heat flux at the front boundary, and fixed temperature at the back boundary.

At steady state,

[
q'' = -k\frac{dT}{dx}.
]

The temperature profile is

[
T(x)
====

T_b
+
\frac{q''}{k}(L-x).
]

The surface-to-back temperature difference is

[
\Delta T
========

# T_s - T_b

\frac{q''L}{k}.
]

Expected normalized sensitivities are

[
S_{q''} = +1,
]

[
S_k = -1,
]

and

[
S_L = +1.
]

Verification requirements:

* SCAM temperature profile matches the analytical steady profile;
* surface temperature scales linearly with heat flux;
* surface temperature varies inversely with thermal conductivity;
* local sensitivities match analytical values.

### Lumped-Capacitance Heating Limit

For a spatially uniform body, or a very high-conductivity limit, the energy balance is

[
\rho c_p L \frac{dT}{dt} = q''.
]

Thus,

[
T(t)
====

T_0
+
\frac{q''}{\rho c_p L}t.
]

Expected normalized sensitivities of the temperature rise are

[
S_{q''} = +1,
]

[
S_{\rho} = -1,
]

[
S_{c_p} = -1,
]

and

[
S_L = -1.
]

Verification requirements:

* SCAM temperature rise matches the analytical lumped solution;
* temperature rise scales linearly with heat flux;
* temperature rise varies inversely with density;
* temperature rise varies inversely with heat capacity;
* temperature rise varies inversely with thickness;
* local sensitivities match analytical values.

## 11. SCAM Charring and Ablation Trend Tests

For full charring and ablation cases, exact analytical solutions are generally unavailable. Verification should therefore use controlled trend tests.

These tests should perturb one parameter at a time and verify that the model response is physically consistent.

Expected trends:

* increasing heat flux increases surface temperature;
* increasing heat flux increases back-wall temperature;
* increasing emissivity reduces surface temperature when reradiation is active;
* increasing thermal conductivity increases back-wall temperature at short and moderate times;
* increasing density reduces temperature rise;
* increasing heat capacity reduces temperature rise;
* increasing activation energy delays decomposition;
* increasing pre-exponential factor accelerates decomposition;
* increasing decomposition enthalpy reduces temperature rise during pyrolysis;
* increasing char thermal conductivity increases heat penetration into the material;
* increasing blowing reduces corrected convective heat flux;
* increasing effective heat of ablation reduces recession rate.

These tests are not validation against experimental data. They are verification checks that the implemented perturbations and solver response are physically consistent.

## 12. Runner Restart and Failure Handling

The UQ runner should be robust to failed samples and interrupted campaigns.

### Restart Test

Verification requirements:

* run an ensemble partially;
* stop after a subset of samples;
* restart the same study;
* completed samples are not rerun;
* missing samples are completed;
* final output contains all completed samples.

### Failed-Run Test

Verification requirements:

* force one sample to fail;
* runner records failure in the status file;
* other samples continue running;
* failed sample is excluded from statistics unless explicitly requested;
* failure summary is written to output.

### Reproducibility Test

Verification requirements:

* the same `uq.yaml` and the same random seed produce identical samples;
* the same deterministic solver produces identical quantities;
* metadata records the random seed, sampler, parameter list, and SCAM version.

## Recommended Test Matrix

The following test matrix should be added under `tests/uq/`.

```text
tests/uq/test_distributions.py
    test_uniform_mean_variance
    test_normal_mean_std
    test_constant_distribution
    test_seed_reproducibility

tests/uq/test_sampling.py
    test_monte_carlo_shape
    test_lhs_stratification
    test_one_at_a_time_structure
    test_finite_difference_symmetry

tests/uq/test_parameters.py
    test_get_by_path_nested_dict
    test_set_by_path_nested_dict
    test_get_by_path_list_entry
    test_set_by_path_list_entry

tests/uq/test_perturb.py
    test_scalar_replace
    test_scalar_scale
    test_scalar_offset
    test_curve_scale
    test_curve_offset
    test_linear_tilt_curve
    test_baseline_immutability

tests/uq/test_propagation.py
    test_linear_model_mean_variance
    test_product_model_mean_variance
    test_threshold_exceedance_uniform
    test_time_dependent_quantiles

tests/uq/test_sensitivity.py
    test_polynomial_local_sensitivity
    test_finite_difference_step_convergence
    test_pearson_linear_model
    test_spearman_monotonic_model
    test_correlation_nonmonotonic_limitation

tests/uq/test_morris_optional.py
    test_morris_linear_ranking
    test_morris_interaction_sigma
    test_morris_inactive_parameter
    test_morris_missing_salib_error

tests/uq/test_sobol_optional.py
    test_sobol_additive_model
    test_sobol_interaction_model
    test_sobol_ishigami_benchmark
    test_sobol_missing_salib_error

tests/uq/test_runner.py
    test_fake_solver_end_to_end
    test_runner_restart
    test_runner_failed_sample
    test_runner_reproducibility

tests/uq/test_scam_integration.py
    test_steady_conduction_sensitivity
    test_lumped_heating_sensitivity
    test_charring_physical_trends
```

## Minimum Verification Set for Initial Implementation

The first implementation does not need every verification case above. The minimum useful verification set should include:

```text
1. Uniform distribution mean and variance
2. Normal distribution mean and standard deviation
3. LHS stratification
4. Scalar replacement perturbation
5. Scalar scaling perturbation
6. Curve scaling perturbation
7. Baseline immutability
8. Additive linear model mean and variance
9. Polynomial local sensitivity
10. Pearson and Spearman tests on simple models
11. Fake-solver end-to-end UQ pipeline
12. Runner restart test
13. Failed-sample handling test
14. Steady 1-D conduction sensitivity test
15. Lumped heating sensitivity test
```

This minimum set verifies the essential UQ infrastructure before applying the method to realistic charring and ablation simulations.

## Summary

The SCAM UQ verification strategy should begin with exact synthetic tests and progress toward physical SCAM tests.

The most important verification principles are:

* verify sampling before solver execution;
* verify perturbations before ensemble runs;
* verify statistics using analytical models;
* verify sensitivities using exact derivatives;
* verify optional SALib methods using known benchmarks;
* verify the runner with a fake deterministic solver;
* verify SCAM integration using simple thermal analytical limits;
* verify full charring and ablation cases through physical trends.

This layered approach gives confidence that any uncertainty or sensitivity result produced by SCAM reflects the behavior of the deterministic solver and the specified uncertain inputs, rather than errors in the UQ infrastructure.