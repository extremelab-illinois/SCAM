# SCAM Sensitivity Analysis and Uncertainty Quantification Solver Development

## Purpose

This document describes the proposed software development needed to add sensitivity analysis and uncertainty quantification capabilities to SCAM.

The goal is not to replace the deterministic SCAM material-response solver, but to add a modular layer that can perturb inputs, execute repeated deterministic runs, extract quantities of interest, and compute sensitivity and uncertainty metrics.

The central design principle is:

```text
SCAM core solver remains deterministic.
The UQ layer manages uncertain inputs, sampling, execution, and post-processing.
```

This separation keeps the governing-equation implementation clean while making uncertainty analysis reproducible, extensible, and testable.

## Development Philosophy

The UQ capability should be implemented as an external orchestration layer around the existing SCAM solver. The core thermal, decomposition, gas transport, surface energy balance, and material-property modules should not contain stochastic logic.

The UQ layer should:

```text
define uncertain parameters
generate samples
perturb SCAM input files or configuration objects
run deterministic SCAM cases
extract scalar and time-dependent quantities of interest
compute sensitivity metrics
compute uncertainty statistics
generate standard output files and plots
```

This approach preserves the existing solver structure and allows the same deterministic case to be run either as a single verification case or as part of a larger uncertainty workflow.

## Recommended Methodological Choices

SCAM should not begin with plain Monte Carlo as the primary method. Plain Monte Carlo is useful as a baseline and for convergence checks, but it is not the most efficient first choice for material-response simulations where each deterministic run may become moderately expensive.

The recommended method hierarchy is:

```text
1. Local finite-difference sensitivity
2. Latin Hypercube Sampling for uncertainty propagation
3. Pearson and Spearman correlation metrics
4. Morris screening for many uncertain parameters
5. Sobol indices for a reduced set of important parameters
```

## Recommended Default Workflow

The recommended default workflow for SCAM UQ is:

```text
Latin Hypercube Sampling
        ↓
deterministic SCAM runs
        ↓
quantities of interest
        ↓
uncertainty statistics
        ↓
Pearson/Spearman sensitivity ranking
```

This should be the first complete UQ workflow implemented because it is practical, relatively inexpensive, and immediately useful for SCAM.

The default UQ workflow should compute:

```text
mean
standard deviation
coefficient of variation
median
5th percentile
95th percentile
threshold exceedance probabilities
Pearson correlation
Spearman rank correlation
```

For example, for a back-wall temperature quantity:

[
\mu_{T_b}(t), \qquad
\sigma_{T_b}(t), \qquad
T_{b,5%}(t), \qquad
T_{b,50%}(t), \qquad
T_{b,95%}(t)
]

## Role of Monte Carlo

Plain Monte Carlo should be supported, but it should not be the default method for the first full SCAM UQ workflow.

Monte Carlo is useful for:

```text
debugging
baseline comparison
simple conduction-only tests
convergence checks
very cheap cases
```

However, for charring, gas generation, surface energy balance, and ablation cases, Latin Hypercube Sampling should generally be preferred because it gives better input-space coverage for a limited number of samples.

## Role of Morris Screening

Morris screening should be used when the number of uncertain parameters is large.

For example, a full charring-ablation case may include uncertainty in:

[
k_v(T), \quad k_c(T), \quad c_{p,v}(T), \quad c_{p,c}(T), \quad
\rho_v, \quad \rho_c, \quad A_i, \quad E_i, \quad
\Delta h_i, \quad \epsilon, \quad q'', \quad B_c', \quad B_g'
]

This can quickly produce 20--50 uncertain inputs. Sobol analysis would be too expensive at that stage. Morris screening should therefore be used to identify the subset of parameters that matter most.

The Morris method should report:

[
\mu_i^*
]

as a measure of overall parameter importance, and

[
\sigma_i
]

as an indicator of nonlinear or interaction effects.

## Role of Sobol Indices

Sobol indices should be treated as an advanced option, not the first method.

Sobol analysis is valuable because it decomposes output variance into first-order and total-effect contributions:

[
S_i, \qquad S_{T_i}
]

where (S_i) measures the direct contribution of input (i), and (S_{T_i}) includes both direct and interaction effects.

However, Sobol analysis can be expensive. It should only be applied after the uncertain parameter set has been reduced using local sensitivity, LHS correlation metrics, or Morris screening.

A practical workflow is:

```text
Start with 20--50 uncertain inputs.
Use Morris screening to identify the dominant 6--10.
Apply Sobol analysis only to the reduced set.
```

## Dependency Strategy

SCAM should not require both SALib and UQpy.

The recommended dependency strategy is:

```text
Core SCAM UQ infrastructure:
    NumPy
    SciPy
    pandas
    matplotlib
    PyYAML

Optional sensitivity-analysis backend:
    SALib

Future advanced UQ backend:
    UQpy
```

For the first implementation, SCAM should provide its own lightweight UQ orchestration layer and use SALib only as an optional dependency for advanced sensitivity-analysis methods such as Morris and Sobol.

UQpy should not be added initially. It is broader than what SCAM needs for the first UQ implementation and would add unnecessary dependency weight. It can be reconsidered later for advanced capabilities such as surrogate modeling, reliability analysis, Bayesian calibration, polynomial chaos, or multi-fidelity UQ.

## Recommended Dependency Policy

The base SCAM installation should not require SALib or UQpy.

The base UQ capability should work with:

```text
finite-difference sensitivity
Monte Carlo sampling
Latin Hypercube Sampling
Pearson correlation
Spearman correlation
mean, standard deviation, and quantiles
```

using only standard scientific Python dependencies.

SALib should be an optional dependency. For example:

```toml
[project.optional-dependencies]
uq = [
    "SALib",
]
```

or, if the project uses `requirements` files:

```text
requirements-uq.txt
```

containing:

```text
SALib
```

The user-facing installation would be:

```bash
pip install -e .[uq]
```

or:

```bash
pip install -r requirements-uq.txt
```

The code should fail gracefully if SALib is not installed and the user requests a SALib-dependent method.

For example:

```text
Morris sensitivity analysis requires SALib.
Install optional UQ dependencies with: pip install -e .[uq]
```

## Proposed Package Structure

A new subpackage should be added:

```text
scam/
    uq/
        __init__.py
        distributions.py
        parameters.py
        perturb.py
        sampling.py
        runner.py
        quantities.py
        sensitivity.py
        propagation.py
        reporting.py
        study.py
```

Associated examples and tests should be added:

```text
examples/
    uq/
        conduction_slab/
        charring_material/
        surface_energy_balance/

tests/
    uq/
        test_distributions.py
        test_parameters.py
        test_perturb.py
        test_sampling.py
        test_quantities.py
        test_runner.py
        test_sensitivity.py
        test_propagation.py
```

## Module Responsibilities

## `scam/uq/distributions.py`

This module defines probability distributions for uncertain inputs.

Required distributions for the first implementation:

```text
constant
uniform
normal
lognormal
triangular
```

The `constant` distribution is useful for debugging and for temporarily deactivating uncertainty in one parameter without changing the input file structure.

A minimal data model could be:

```python
@dataclass
class DistributionSpec:
    kind: str
    nominal: float | None = None
    mean: float | None = None
    std: float | None = None
    lower: float | None = None
    upper: float | None = None
    units: str | None = None
```

The module should expose a common sampling function:

```python
sample_distribution(
    spec: DistributionSpec,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray
```

This keeps the rest of the UQ code independent of the details of individual distributions.

## `scam/uq/parameters.py`

This module defines how uncertain parameters are represented and connected to SCAM input objects.

A parameter should include:

```text
name
target path
nominal value
units
distribution
transform type
bounds
description
```

Example:

```yaml
- name: emissivity
  target: surface.emissivity
  transform: replace
  nominal: 0.85
  units: none
  distribution:
    kind: uniform
    lower: 0.75
    upper: 0.95
```

The target path should refer to a location in a SCAM case dictionary, material card, or solver configuration.

Example target paths:

```text
boundary.front.heat_flux
surface.emissivity
material.phases.virgin.thermal_conductivity
material.phases.char.thermal_conductivity
material.decomposition.reactions[0].pre_exponential_factor
material.decomposition.reactions[0].activation_energy
material.transport.permeability
```

The module should provide utilities such as:

```python
get_by_path(config, path)
set_by_path(config, path, value)
```

These utilities should support nested dictionaries, lists, and dataclass-like objects where practical.

## `scam/uq/perturb.py`

This module applies sampled uncertain inputs to a baseline SCAM case.

The perturbation layer is one of the most important pieces of the UQ infrastructure because it controls how physical uncertainty is mapped into solver inputs.

Supported transforms should include:

```text
replace
scale
offset
scale_curve
offset_curve
linear_tilt_curve
```

For scalar quantities:

```text
replace:     p = sample
scale:       p = p0 * sample
offset:      p = p0 + sample
```

For tabulated temperature-dependent material properties:

```text
scale_curve:
    y(T) = sample * y0(T)

offset_curve:
    y(T) = y0(T) + sample

linear_tilt_curve:
    y(T) = y0(T) * [1 + a + b (T - Tref)/(Tmax - Tmin)]
```

The curve perturbation capability is important because material properties such as ( k(T) ), ( c_p(T) ), permeability, and gas enthalpy tables should not be perturbed point-by-point in an uncorrelated way. That would create unphysical noisy material curves.

Instead, the UQ layer should perturb curves through low-dimensional, physically interpretable modes.

The main function should look like:

```python
perturbed_case = apply_sample(
    baseline_case=case,
    parameters=parameters,
    sample=sample_row,
)
```

The baseline case should not be modified in place.

## `scam/uq/sampling.py`

This module generates samples for uncertain parameters.

The first implementation should include:

```text
one_at_a_time
finite_difference
monte_carlo
latin_hypercube
```

Later extensions should include:

```text
morris
sobol
```

The first four methods should be implemented internally using NumPy/SciPy. Morris and Sobol should use SALib when available.

The module should be deterministic when a random seed is provided.

Example API:

```python
samples = generate_samples(
    parameters=parameters,
    method="latin_hypercube",
    n_samples=500,
    random_seed=42,
)
```

The returned object should preferably be a table-like structure with one row per run and one column per uncertain parameter.

Example:

```text
sample_id, heat_flux_scale, emissivity, k_virgin_scale, activation_energy_scale
0,         0.94,            0.81,       1.07,            0.98
1,         1.03,            0.88,       0.95,            1.04
```

## `scam/uq/runner.py`

This module executes repeated deterministic SCAM runs.

The runner should be responsible for:

```text
creating run directories
writing perturbed input files
calling the deterministic SCAM solver
collecting solver outputs
saving run metadata
handling failed runs
supporting restart
supporting serial and parallel execution
```

The first version can be serial. Parallel execution can be added afterward using Python multiprocessing.

A robust run directory structure would be:

```text
uq_results/
    metadata.yaml
    samples.csv
    runs/
        sample_000000/
            input.yaml
            result.npz
            quantities.yaml
            status.yaml
        sample_000001/
            input.yaml
            result.npz
            quantities.yaml
            status.yaml
```

Each sample should have a status file:

```yaml
sample_id: 12
status: completed
start_time: ...
end_time: ...
error_message: null
```

or:

```yaml
sample_id: 13
status: failed
error_message: "Nonlinear surface energy balance did not converge"
```

This is essential for long UQ runs. Failed samples should not destroy the whole campaign.

## `scam/uq/quantities.py`

This module extracts quantities of interest from SCAM results.

Quantities of interest should be divided into two classes:

```text
scalar quantities
time-dependent quantities
```

Useful scalar quantities include:

```text
max_surface_temperature
max_backwall_temperature
final_surface_temperature
final_backwall_temperature
final_recession
maximum_recession_rate
final_char_depth
maximum_pyrolysis_gas_flux
total_pyrolysis_gas_mass
total_surface_mass_loss
minimum_remaining_thickness
time_to_backwall_temperature_limit
```

Useful time-dependent quantities include:

```text
surface_temperature(t)
backwall_temperature(t)
recession(t)
recession_rate(t)
surface_pyrolysis_gas_flux(t)
surface_mass_loss_rate(t)
char_depth(t)
net_heat_flux(t)
reradiation_flux(t)
chemical_heat_flux(t)
blowing_correction(t)
```

The module should expose a registry of built-in quantities:

```python
BUILTIN_QUANTITIES = {
    "max_backwall_temperature": max_backwall_temperature,
    "final_recession": final_recession,
    "maximum_pyrolysis_gas_flux": maximum_pyrolysis_gas_flux,
}
```

Users should also be able to define custom quantities.

Example:

```python
qoi = Quantity(
    name="max_backwall_temperature",
    kind="scalar",
    function=lambda result: np.max(result.temperature[:, -1]),
)
```

## `scam/uq/sensitivity.py`

This module computes sensitivity metrics from samples and quantities of interest.

The first implementation should include:

```text
finite-difference local sensitivity
normalized local sensitivity
Pearson correlation
Spearman rank correlation
```

A normalized local sensitivity can be defined as:

[
S_i =
\frac{\partial Q}{\partial p_i}
\frac{p_i}{Q}
]

using a centered finite difference:

[
S_i
\approx
\frac{Q(p_i + \Delta p_i) - Q(p_i - \Delta p_i)}
{2 \Delta p_i}
\frac{p_i}{Q}
]

Correlation-based sensitivity metrics are useful for Monte Carlo and Latin Hypercube Sampling runs. They are inexpensive and provide a first ranking of influential parameters.

Later versions should add:

```text
Morris screening
Sobol first-order indices
Sobol total indices
partial rank correlation coefficients
```

Morris and Sobol should be implemented through optional SALib integration rather than as required internal methods.

The module should include a graceful fallback when SALib is unavailable.

Example:

```python
def require_salib(method_name: str) -> None:
    try:
        import SALib  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            f"{method_name} requires SALib. "
            "Install optional UQ dependencies with: pip install -e .[uq]"
        ) from exc
```

## `scam/uq/propagation.py`

This module computes uncertainty propagation statistics.

For scalar quantities:

```text
mean
standard deviation
coefficient of variation
minimum
maximum
median
5th percentile
95th percentile
probability of threshold exceedance
```

For time-dependent quantities:

```text
mean(t)
standard deviation(t)
median(t)
lower_quantile(t)
upper_quantile(t)
```

For example, for back-wall temperature:

[
\mu_{T_b}(t), \quad
\sigma_{T_b}(t), \quad
T_{b,5%}(t), \quad
T_{b,50%}(t), \quad
T_{b,95%}(t)
]

Threshold-exceedance calculations should be supported because they are engineering-relevant:

```python
probability_exceedance(
    quantity="max_backwall_temperature",
    threshold=600.0,
)
```

## `scam/uq/reporting.py`

This module generates standard tables and plots from UQ results.

Useful plots include:

```text
histogram of scalar quantities
uncertainty envelope for time histories
scatter plot of parameter versus quantity
correlation bar plot
local sensitivity tornado plot
Morris ranking plot
Sobol total-index plot
convergence plot of mean and standard deviation
failed-run summary
```

The plotting routines should follow the existing SCAM publication-quality plotting style:

```text
SI units
clear labels
units in square brackets
inward ticks
no grid unless explicitly requested
vector output option
```

The reporting layer should not recompute quantities. It should only read already-computed UQ outputs.

## `scam/uq/study.py`

This module provides the high-level user-facing interface.

The central class should be:

```python
class UQStudy:
    ...
```

The intended user workflow should be:

```python
from scam.uq import UQStudy

study = UQStudy.from_yaml("uq.yaml")
results = study.run()
results.save("uq_results")
```

The `UQStudy` object should coordinate:

```text
input parsing
parameter validation
sample generation
case perturbation
solver execution
quantity extraction
sensitivity analysis
uncertainty propagation
report generation
```

The `study.py` module should be thin. It should coordinate the other modules rather than implementing all logic directly.

## Input File Design

A UQ input file should be separate from the deterministic SCAM case file.

Recommended structure:

```yaml
study:
  name: tacot_uq_example
  baseline_case: case.yaml
  output_directory: uq_results
  sampler: latin_hypercube
  n_samples: 500
  random_seed: 42
  parallel: false

parameters:
  - name: heat_flux_scale
    target: boundary.front.heat_flux
    transform: scale
    nominal: 1.0
    distribution:
      kind: uniform
      lower: 0.9
      upper: 1.1

  - name: emissivity
    target: surface.emissivity
    transform: replace
    nominal: 0.85
    distribution:
      kind: uniform
      lower: 0.75
      upper: 0.95

  - name: virgin_conductivity_scale
    target: material.phases.virgin.thermal_conductivity
    transform: scale_curve
    nominal: 1.0
    distribution:
      kind: normal
      mean: 1.0
      std: 0.1

quantities:
  scalar:
    - max_surface_temperature
    - max_backwall_temperature
    - final_recession
    - final_char_depth
    - maximum_pyrolysis_gas_flux

  time_dependent:
    - surface_temperature
    - backwall_temperature
    - recession
    - surface_pyrolysis_gas_flux

analysis:
  propagation:
    quantiles: [0.05, 0.5, 0.95]
    thresholds:
      - quantity: max_backwall_temperature
        value: 600.0
        units: K

  sensitivity:
    methods:
      - pearson
      - spearman
```

For advanced SALib-enabled analyses:

```yaml
analysis:
  screening:
    method: morris
    trajectories: 15

  variance_decomposition:
    method: sobol
    base_samples: 512
```

## Result Object Design

The UQ result object should contain:

```text
samples
run_status
scalar_quantities
time_dependent_quantities
sensitivity_metrics
uncertainty_statistics
metadata
```

A possible internal structure is:

```python
@dataclass
class UQResults:
    samples: pd.DataFrame
    run_status: pd.DataFrame
    scalar_quantities: pd.DataFrame
    time_histories: dict[str, np.ndarray]
    sensitivity: dict[str, pd.DataFrame]
    statistics: dict[str, pd.DataFrame]
    metadata: dict
```

The result object should be serializable.

Recommended output files:

```text
samples.csv
run_status.csv
scalar_quantities.csv
sensitivity_pearson.csv
sensitivity_spearman.csv
statistics_scalar.csv
statistics_time_histories.npz
metadata.yaml
```

If SALib methods are used, additional files may include:

```text
sensitivity_morris.csv
sensitivity_sobol_first_order.csv
sensitivity_sobol_total_order.csv
```

## Solver Integration Requirements

The UQ layer needs a stable deterministic SCAM runner interface.

A useful internal function would be:

```python
result = run_scam_case(case_config)
```

where `result` contains, at minimum:

```text
time
coordinates
temperature
density
beta
recession
recession_rate
surface_temperature
backwall_temperature
pyrolysis_gas_flux
surface_mass_loss_rate
heat_flux_terms
```

Not every case needs every field. Missing fields should be handled cleanly.

For example, a conduction-only case may not have:

```text
beta
recession
pyrolysis_gas_flux
```

In that case, requesting `final_recession` should produce a clear error message rather than silently returning zero.

## Testing Strategy

The UQ tests should not initially rely on expensive full ablation simulations.

Use three levels of tests:

## Unit Tests

Test each UQ module independently using simple dictionaries and synthetic arrays.

## Synthetic Solver Tests

Use a fake solver:

```python
Q = a * x_1 + b * x_2**2
```

This allows exact verification of sensitivity and uncertainty calculations.

## SCAM Integration Tests

Use a small conduction-only SCAM case with a coarse grid and short final time.

The integration test should verify that:

```text
samples are generated
perturbed cases run
outputs are collected
quantities are extracted
statistics are computed
```

The full surface-energy-balance case should be used as an example, not as a required fast unit test.

## Development Milestones

## Milestone UQ-1: Parameter and Distribution Infrastructure

Add:

```text
distributions.py
parameters.py
perturb.py
```

Required tests:

```text
uniform distribution respects bounds
normal distribution is reproducible with fixed seed
parameter paths can access nested dictionaries
parameter paths can access list entries
scalar replacement works
scalar scaling works
curve scaling works
baseline object is not modified in place
```

This milestone should not require running the SCAM solver.

## Milestone UQ-2: Sampling Infrastructure

Add:

```text
sampling.py
```

Required sampling methods:

```text
one_at_a_time
finite_difference
monte_carlo
latin_hypercube
```

Required tests:

```text
sample matrix has correct shape
same seed gives same samples
different seed gives different samples
samples remain within prescribed bounds
one-at-a-time perturbations have expected values
finite-difference perturbations are symmetric
Latin Hypercube Sampling stratifies each uncertain dimension
```

## Milestone UQ-3: Deterministic Runner Integration

Add:

```text
runner.py
```

Required capabilities:

```text
run a single perturbed case
run multiple samples
save each case input
save each case output
record completed and failed runs
skip completed runs on restart
```

Required tests:

```text
runner completes a small conduction-only ensemble
runner creates one directory per sample
runner writes status files
runner can restart without duplicating completed runs
runner reports failed runs cleanly
```

## Milestone UQ-4: Quantity Extraction

Add:

```text
quantities.py
```

Required scalar quantities:

```text
max_surface_temperature
max_backwall_temperature
final_surface_temperature
final_backwall_temperature
final_recession
maximum_recession_rate
maximum_pyrolysis_gas_flux
total_mass_loss
```

Required tests:

```text
scalar quantities are extracted correctly from synthetic results
time-dependent quantities preserve shape
threshold time is computed correctly
missing quantities give clear error messages
```

## Milestone UQ-5: Uncertainty Propagation

Add:

```text
propagation.py
```

Required capabilities:

```text
mean
standard deviation
coefficient of variation
median
quantiles
threshold exceedance probability
```

Required tests:

```text
statistics are correct for known arrays
quantiles are correct for known arrays
threshold exceedance probability is correct
time-dependent statistics preserve time dimension
```

## Milestone UQ-6: Sensitivity Metrics

Add:

```text
sensitivity.py
```

Required first methods:

```text
normalized finite-difference sensitivity
Pearson correlation
Spearman rank correlation
```

Required tests:

```text
linear synthetic model returns expected sensitivity
monotonic nonlinear synthetic model gives high Spearman correlation
uncorrelated parameter gives near-zero correlation
```

Optional SALib-based methods:

```text
Morris screening
Sobol first-order indices
Sobol total indices
```

Required tests for optional SALib methods:

```text
clear error if SALib is missing
Morris workflow runs on a simple synthetic model when SALib is installed
Sobol workflow runs on a simple synthetic model when SALib is installed
```

## Milestone UQ-7: Reporting

Add:

```text
reporting.py
```

Required plots:

```text
histogram
uncertainty envelope
parameter-versus-quantity scatter
correlation bar plot
local sensitivity tornado plot
convergence plot
```

Optional plots:

```text
Morris ranking plot
Sobol total-index plot
```

Required tests:

```text
plot functions create output files
plot functions work with missing optional metrics
plot labels include units where available
```

## Recommended Initial Pull Request Sequence

## PR 1: UQ data structures

Add:

```text
scam/uq/distributions.py
scam/uq/parameters.py
scam/uq/perturb.py
tests/uq/test_distributions.py
tests/uq/test_parameters.py
tests/uq/test_perturb.py
```

## PR 2: Sampling

Add:

```text
scam/uq/sampling.py
tests/uq/test_sampling.py
```

Include:

```text
finite difference
Monte Carlo
Latin Hypercube Sampling
```

Do not add SALib yet.

## PR 3: Runner and restart logic

Add:

```text
scam/uq/runner.py
tests/uq/test_runner.py
```

## PR 4: Quantities of interest

Add:

```text
scam/uq/quantities.py
tests/uq/test_quantities.py
```

## PR 5: Propagation and basic sensitivity metrics

Add:

```text
scam/uq/propagation.py
scam/uq/sensitivity.py
tests/uq/test_propagation.py
tests/uq/test_sensitivity.py
```

Include:

```text
uncertainty statistics
Pearson correlation
Spearman correlation
normalized local finite-difference sensitivity
```

## PR 6: Optional SALib integration

Add optional support for:

```text
Morris screening
Sobol sensitivity indices
```

Update packaging with:

```toml
[project.optional-dependencies]
uq = [
    "SALib",
]
```

or add:

```text
requirements-uq.txt
```

containing:

```text
SALib
```

## PR 7: Reporting and examples

Add:

```text
scam/uq/reporting.py
examples/uq/
```

## Minimal Working Example

The first complete example should be conduction-only because it is fast and easy to verify.

Example directory:

```text
examples/uq/conduction_slab/
    case.yaml
    uq.yaml
    run_uq.py
    plot_uq.py
```

The example should demonstrate uncertainty in:

```text
heat flux
thermal conductivity
density
specific heat
emissivity
```

and should output:

```text
samples.csv
scalar_quantities.csv
statistics_scalar.csv
surface_temperature_envelope.png
backwall_temperature_envelope.png
sensitivity_bar.png
```

Once this example is stable, the same UQ infrastructure can be applied to charring and ablation cases.

## Recommended Default `uq.yaml`

The recommended default should use Latin Hypercube Sampling with correlation-based sensitivity:

```yaml
study:
  name: default_scam_uq
  baseline_case: case.yaml
  output_directory: uq_results
  sampler: latin_hypercube
  n_samples: 300
  random_seed: 42
  parallel: false

analysis:
  propagation:
    quantiles: [0.05, 0.5, 0.95]

  sensitivity:
    methods:
      - pearson
      - spearman
```

Plain Monte Carlo should remain available:

```yaml
study:
  sampler: monte_carlo
```

but should not be the recommended default for full material-response cases.

## Long-Term Extensions

After the basic UQ infrastructure is stable, future extensions could include:

```text
Morris screening
Sobol sensitivity indices
surrogate modeling
Bayesian calibration
experimental data assimilation
model discrepancy terms
adaptive sampling
multi-fidelity UQ
polynomial chaos
reliability analysis
```

These should not be part of the first implementation. The first goal should be a reliable and tested UQ workflow for repeated deterministic SCAM runs.

## Summary

The proposed development adds a modular UQ layer to SCAM while preserving the deterministic solver.

The key additions are:

```text
uncertain parameter definitions
distribution handling
sample generation
case perturbation
ensemble execution
quantity extraction
uncertainty propagation
sensitivity metrics
standard reporting
```

The recommended practical choices are:

```text
Use Latin Hypercube Sampling as the default UQ propagation method.
Use Pearson and Spearman correlations for the first global sensitivity ranking.
Use finite-difference sensitivity for local interpretation and debugging.
Use Morris screening only when many parameters are included.
Use Sobol indices only after reducing the parameter set.
Use SALib as an optional dependency for Morris and Sobol.
Do not add UQpy initially.
Do not make SALib or UQpy required dependencies.
```

The most important architectural decision is to keep uncertainty handling outside the physics solver. SCAM should remain a deterministic material-response code, while `scam.uq` provides the machinery needed to run sensitivity analysis and uncertainty quantification in a reproducible way.