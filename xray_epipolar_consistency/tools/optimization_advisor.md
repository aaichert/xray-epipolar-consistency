# Optimization Advisor

## Overview

The Optimization Advisor automatically recommends well-conditioned optimization problems based on the results of the Geometric Identifiability Analysis.

Rather than relying on manually selected parameter subsets and heuristic parameter ranges, the advisor uses the local geometry of the parameter space to construct optimization problems that are expected to be numerically stable and physically meaningful.

The advisor performs two tasks:

- selection of compatible optimization parameters,
- automatic estimation of suitable optimization ranges.

The resulting recommendations are intended to provide robust starting points for geometric calibration.

---

# Inputs

The advisor requires

- the Geometric Identifiability Analysis results,
- the parameterization definitions,
- one user-defined detector motion scale

$$
d_{\max}
$$

expressed in detector pixels.

This quantity represents the maximum detector displacement that is considered physically plausible before optimization.

Typical values might be

- 5 px
- 10 px
- 20 px
- 50 px

depending on the expected calibration error.

---

# Local Sensitivity

For every parameter

$$
p_i
$$

the analysis has already estimated the local Jacobian

$$
J_i
=
\frac{\partial x}{\partial p_i}.
$$

Its norm

$$
\|J_i\|
$$

describes how strongly detector coordinates move per unit parameter change.

This allows detector motion to be predicted locally.

---

# Automatic Parameter Scaling

Instead of using manually chosen optimization ranges, every parameter range is computed such that it produces approximately the same expected detector motion.

For parameter

$$
p_i,
$$

the recommended half-range is

$$
\Delta p_i
=
\frac
{
d_{\max}
}
{
\|J_i\|
}.
$$

Consequently,

all parameters are normalized to produce approximately

$$
d_{\max}
$$

pixels of detector motion.

Highly sensitive parameters receive small optimization ranges.

Weak parameters receive larger ranges.

This produces a naturally balanced optimization problem.

---

# Physical Range Limits

Automatically computed ranges are constrained by the parameter's physical limits.

The final range is

$$
\Delta p_i
=
\min
(
\Delta p_{\mathrm{physical}},
\Delta p_{\mathrm{sensitivity}}
).
$$

This prevents unrealistic optimization intervals.

---

# Parameter Compatibility

Two parameters should not be optimized together if they produce nearly identical detector motion.

The advisor therefore constructs the normalized correlation matrix

$$
R.
$$

Parameters satisfying

$$
|\rho| > 0.95
$$

are considered incompatible.

Only one representative should be selected unless there is strong prior knowledge.

---

# Explainability Filtering

Similarly,

if parameter

$$
p_i
$$

can be explained by parameter

$$
p_j
$$

with

more than

$$
95\%
$$

accuracy,

the advisor recommends disabling one of them.

---

# Sloppy Direction Detection

The singular value decomposition

$$
J
=
U
\Sigma
V^T
$$

is inspected.

Whenever a large spectral gap occurs,

the corresponding singular vectors are analyzed.

If the smallest singular vectors consist primarily of a small subset of parameters,

those parameters are reported as mutually coupled.

Example

```text
Weakly observable combination

+0.71 Detector Slant
-0.70 Rotation Axis Lateral Offset
```

The advisor recommends optimizing only one of these parameters.

---

# Automatic Group Construction

A graph is constructed.

Each parameter is represented by one node.

Edges connect parameters that satisfy

- high correlation,
- high explainability,
- participation in the same sloppy direction.

Connected components of this graph represent mutually dependent parameter groups.

The advisor attempts to select one representative parameter from every group.

The resulting parameter subset maximizes observability while minimizing redundancy.

---

# Condition Number Search

The advisor evaluates candidate parameter subsets.

For each subset

- construct the corresponding Jacobian,
- compute

$$
\kappa(J)
$$

the condition number,

- compute the numerical rank,
- compute the smallest singular value.

Subsets with

- low condition number,
- large minimum singular value,
- full numerical rank

are preferred.

---

# Greedy Parameter Selection

Starting from the most observable parameter,

additional parameters are added one by one.

A parameter is accepted only if

- it increases the numerical rank,
- it does not significantly increase the condition number,
- it is not highly correlated with previously selected parameters.

The process stops when no further parameter improves the optimization problem.

---

# Range Refinement

After selecting the final parameter subset,

their ranges are recomputed jointly.

If two parameters remain moderately coupled,

their ranges are reduced until the local linear approximation remains well-conditioned.

This produces conservative optimization intervals for coupled parameters while allowing larger ranges for independent parameters.

---

# Optimization Difficulty

Every recommended parameter subset receives a score based on

- condition number,
- effective rank,
- minimum singular value,
- average parameter correlation,
- average explainability,
- epipolar observability.

The advisor classifies each subset as

- Excellent
- Good
- Acceptable
- Poor
- Ill-conditioned

---

# Suggested Optimization Stages

Instead of recommending one large optimization,

the advisor can automatically construct staged optimization strategies.

Example

Stage 1

- Detector Shift
- Gantry Angle

Stage 2

- Detector Orientation

Stage 3

- Rotation Axis

Stage 4

- Distance Parameters

Each stage is selected to remain well-conditioned while gradually introducing additional degrees of freedom.

---

# Outputs

The advisor produces

- recommended parameter subsets,
- suggested optimization stages,
- automatically scaled optimization ranges,
- condition numbers,
- numerical ranks,
- expected detector motion,
- parameter compatibility graphs,
- detected sloppy parameter combinations,
- parameter exclusion recommendations.

---

# HTML Summary

The HTML report should summarize

- recommended optimization sequence,
- enabled parameters,
- recommended ranges,
- expected detector motion,
- condition numbers,
- numerical rank,
- dominant parameter couplings,
- excluded parameters and the reason for exclusion.

---

# Typical Workflow

1. Perform Geometric Identifiability Analysis.
2. Compute local detector Jacobians.
3. Normalize parameter ranges using the desired detector motion.
4. Detect parameter couplings.
5. Construct the compatibility graph.
6. Search for well-conditioned parameter subsets.
7. Recommend optimization stages.
8. Export the final optimization configuration.

The resulting optimization setup is tailored to the specific acquisition trajectory and parameterization. Rather than relying on manually tuned parameter subsets and optimization ranges, the advisor derives both directly from the local geometry of the inverse problem.