# Geometric Identifiability Analysis

## Overview

The **Geometric Identifiability Analysis** extends the Geometric Sensitivity Analysis by studying not only how individual parameters affect detector measurements, but also how well the entire parameterization can be identified from projection geometry.

Instead of treating each parameter independently, the analysis considers the **geometry of the parameter space itself**. Every parameter perturbation induces a detector motion field, and together these motion fields define a local metric describing which directions in parameter space are easily observable and which directions are fundamentally ambiguous.

This framework naturally captures

- parameter sensitivity,
- parameter coupling,
- gauge freedoms,
- near-gauge transformations,
- optimization conditioning,
- epipolar observability.

Rather than explicitly classifying parameter combinations as "gauge" or "non-gauge", these properties emerge naturally from the local geometry of the parameter manifold.

---

# Motivation

Many geometric parameters produce very similar effects on detector measurements.

Examples include

- detector shifts versus source shifts,
- detector pitch versus object translation,
- detector slant versus rotation-axis offsets.

Some combinations are exactly unobservable because they merely redefine the global coordinate system.

Others are not strictly unobservable but produce only very small changes in the projection geometry. These "sloppy" parameter combinations are difficult to estimate numerically and often dominate optimization failures.

The purpose of this analysis is to identify these parameter directions and quantify their observability.

---

# Inputs

The analysis requires

- a trajectory represented by projection matrices,
- a reconstruction volume,
- one or more parameterizations,
- optimization ranges,
- Monte-Carlo sampling configuration.

---

# Monte-Carlo Sampling

Random 3D points are sampled uniformly throughout the reconstruction volume.

For every optimization parameter

$$
p_i,
$$

a small perturbation

$$
\delta p_i
$$

is applied around the nominal geometry.

Every sampled point is projected before and after the perturbation.

The resulting detector displacement is

$$
\Delta x
=
\begin{bmatrix}
\Delta u\\
\Delta v
\end{bmatrix}.
$$

The detector displacements from all sampled points and all detector images are concatenated into one high-dimensional vector

$$
v_i.
$$

Each optimization parameter is therefore represented by one detector motion vector.

---

# Detector Motion Space

Collecting all detector motion vectors produces the detector Jacobian

$$
J
=
\begin{bmatrix}
v_1 &
v_2 &
\cdots &
v_n
\end{bmatrix}.
$$

Each column corresponds to one optimization parameter.

The Jacobian represents the local linearization of the projection operator.

It completely characterizes the first-order behavior of the parameterization.

---

# The Parameter Metric

The central object of this analysis is the Gram matrix

$$
G
=
J^T J.
$$

This matrix defines a local inner product on parameter space.

Intuitively,

- parameters producing similar detector motion have large inner products,
- orthogonal parameters have small inner products,
- redundant parameters produce nearly linearly dependent columns.

The Gram matrix therefore acts as a local metric on the parameter manifold induced by the projection geometry.

Rather than studying individual parameters, the analysis studies the structure of this metric.

---

# Impact Analysis

Individual parameter influence is still reported.

For every parameter

$$
\mathrm{impact}(p,i)
$$

the detector displacement covariance

$$
\Sigma_{\mathrm{impact}}
=
\operatorname{Cov}(\Delta x)
$$

is computed.

Reported quantities include

- mean displacement,
- RMS displacement,
- covariance,
- anisotropy,
- principal directions,
- maximum displacement.

These describe how strongly each parameter affects detector coordinates.

---

# Epipolar Projection

Detector motion is not equally observable by epipolar consistency.

For every detector pair

$$
(i,j),
$$

the epipolar line

$$
l_j
=
F_{ij}x_i
$$

is computed.

Every detector displacement is decomposed into

- motion parallel to the epipolar line,
- motion orthogonal to the epipolar line.

Only the orthogonal component contributes to epipolar consistency.

Repeating the detector motion analysis with only orthogonal motion produces the epipolar Jacobian

$$
J_{\perp}.
$$

All subsequent analyses are performed both for

- the complete detector Jacobian

$$
J,
$$

and

- the epipolar Jacobian

$$
J_{\perp}.
$$

Comparing both reveals which ambiguities are introduced by the epipolar objective itself.

---

# Parameter Correlation

The normalized correlation between two parameters is

$$
\rho_{ij}
=
\frac
{
v_i^Tv_j
}
{
\|v_i\|
\|v_j\|
}.
$$

Values close to

- 1 indicate identical detector motion,
- -1 indicate identical motion with opposite sign,
- 0 indicate independent detector motion.

The complete parameter correlation matrix is reported.

---

# Pairwise Explainability

The analysis determines how well one parameter can reproduce another.

The optimal scaling factor is

$$
\alpha
=
\frac
{
v_A^Tv_B
}
{
v_B^Tv_B
}.
$$

The residual

$$
\|
v_A
-
\alpha v_B
\|
$$

measures the part of the detector motion that cannot be reproduced.

A normalized explainability score between 0 and 1 is reported.

---

# Multi-Parameter Explainability

Individual parameters may also be reconstructed from combinations of several other parameters.

Given

$$
V
=
(v_1,v_2,\ldots,v_n),
$$

the least-squares problem

$$
\min_{\alpha}
\|
V\alpha
-
v_i
\|^2
$$

is solved.

Parameters with high explainability are likely to be poorly identifiable during optimization.

---

# Singular Value Analysis

The detector Jacobian is factorized

$$
J
=
U
\Sigma
V^T.
$$

The singular values describe the intrinsic observability of the parameterization.

Large singular values correspond to parameter combinations producing large detector motion.

Small singular values correspond to parameter combinations producing only weak detector motion.

Zero singular values correspond to exact gauge freedoms.

The same analysis is repeated for

$$
J_{\perp}.
$$

This reveals the additional degeneracies introduced by epipolar consistency.

---

# Sloppy Parameter Directions

Rather than explicitly defining gauge transformations, the analysis searches for **sloppy directions**.

These correspond to eigenvectors of

$$
G
=
J^TJ
$$

associated with very small eigenvalues.

Each eigenvector represents one linear combination of optimization parameters.

For example,

$$
\begin{bmatrix}
0\\
1\\
0\\
-1\\
0
\end{bmatrix}
$$

may indicate that

- detector slant and
- rotation-axis offset

produce nearly identical detector motion.

Unlike pairwise correlation, this naturally identifies dependencies involving multiple parameters simultaneously.

Gauge freedoms simply appear as sloppy directions with zero eigenvalue.

---

# Automatic Diagnostics

The software should automatically detect important numerical features during the analysis.

## Eigenvalue Spectrum

Plot

- singular values,
- eigenvalues of the Gram matrix,

using logarithmic scaling.

Large drops in the spectrum often indicate transitions between

- well-constrained,
- weakly constrained,
- gauge directions.

---

## Automatic Gap Detection

Whenever two consecutive eigenvalues satisfy

$$
\frac{\lambda_i}
{\lambda_{i+1}}
>
10^2
$$

(or another configurable threshold),

the corresponding eigenvectors should be printed.

These eigenvectors reveal which parameter combinations become poorly observable.

---

## Dominant Parameter Contributions

For every eigenvector

$$
v,
$$

print the largest parameter coefficients.

For example,

```text
Eigenvector 5

+0.71 DetectorOrientation.slant_yaw
-0.70 RotationAxis.offset_lateral
+0.03 RotationAxis.offset_radial
```

This immediately identifies the coupled parameters.

---

## Pairwise Correlation Warnings

Automatically report all parameter pairs satisfying

$$
|\rho| > 0.95.
$$

These pairs are likely to be numerically indistinguishable.

---

## Explainability Warnings

Automatically report parameter pairs with explainability above

$$
95\%.
$$

This identifies parameters that can almost perfectly reproduce one another.

---

## Condition Numbers

Report

- condition number of

$$
J,
$$

- condition number of

$$
J_{\perp}.
$$

Large condition numbers indicate difficult optimization problems.

---

## Effective Rank

Estimate the numerical rank of

$$
J
$$

using a configurable tolerance.

This indicates the effective number of independently observable parameter directions.

---

## Parameter Importance

For every parameter compute

- detector motion magnitude,
- observable detector motion,
- average contribution to dominant eigenvectors.

This distinguishes

- influential parameters,
- poorly observable parameters,
- redundant parameters.

---

## Dominant Motion Visualization

For selected eigenvectors, reconstruct the corresponding detector motion field.

This produces intuitive visualizations of the detector deformation associated with each principal parameter direction.

These visualizations are often significantly easier to interpret than numerical statistics.

---

# Outputs

Each parameterization produces one JSON file containing

- Monte-Carlo configuration,
- detector motion statistics,
- covariance matrices,
- parameter metric,
- correlation matrices,
- explainability matrices,
- singular values,
- eigenvalues,
- eigenvectors,
- detected spectral gaps,
- condition numbers,
- effective rank,
- dominant parameter combinations,
- epipolar analysis,
- derived summary statistics.

---

# HTML Report

A plain HTML report summarizes the analysis.

Suggested sections include

- Trajectory information
- Parameter overview
- Detector impact ranking
- Parameter correlation matrix
- Explainability matrix
- Singular value spectrum
- Eigenvalue spectrum
- Spectral gap diagnostics
- Dominant eigenvectors
- Automatically detected sloppy directions
- Condition numbers
- Effective rank
- Parameter importance ranking
- Links to the corresponding JSON files.

---

# Typical Applications

The Geometric Identifiability Analysis can be used to

- evaluate new parameterizations,
- compare acquisition trajectories,
- identify redundant optimization parameters,
- understand optimization failures,
- quantify parameter observability,
- detect gauge freedoms,
- detect near-gauge ("sloppy") parameter combinations,
- identify poorly conditioned optimization problems,
- determine which parameters can be estimated from epipolar consistency,
- guide the design of more robust calibration parameterizations.

Rather than separating detector motion into predefined categories, the analysis studies the intrinsic geometry of the parameter space induced by the projection operator. Gauge freedoms, parameter coupling, optimization conditioning, and epipolar observability emerge naturally as properties of this geometric structure, providing a comprehensive characterization of the calibration problem.