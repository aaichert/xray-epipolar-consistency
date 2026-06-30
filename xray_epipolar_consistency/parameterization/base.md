← [Back to Parameterization Index](../parameterization.md)

# Parameterization Base

Defines the base classes for projection matrix parameterizations.

## Classes

### `ParameterizationBase`
An abstract base class for geometric parameterizations. It maps parameter names to metadata dictionaries (value, optimize, range, description).

- `get_parameter_vector()` / `set_parameter_vector(values)`: Gets and sets the subset of active parameters (where `"opt": True`).
- `apply_to_trajectory(Ps)`: Applies stationary transformations sequentially to each projection matrix in a trajectory.
- `estimateTrajectoryParameters(Ps)`: Computes the trajectory prior knowledge (iso-center and rotation axis) from a list of projection matrices.

### `ParameterizationChain`
A sequential composition of parameterizations. It delegates calls to its constituent sub-parameterizations.

---

## Math: Prior Knowledge Estimation

The trajectory's prior knowledge consists of the estimated **iso-center** and **rotation axis**, which are used as local coordinate reference systems for other parameterizations.

### 1. Iso-Center Estimation
The iso-center $\mathbf{c}_{\text{iso}}$ is defined as the point that minimizes the sum of squared distances to all principal rays.
For each projection matrix $P_i$, let:
- $\mathbf{C}_i \in \mathbb{R}^3$ be the camera center of projection (dehomogenized).
- $\hat{\mathbf{r}}_i \in \mathbb{R}^3$ be the unit principal ray direction.

The projection matrix of the point onto the orthogonal complement of the ray is:
$$
M_i = I_3 - \hat{\mathbf{r}}_i \hat{\mathbf{r}}_i^T
$$

The sum of squared distances is minimized by solving:
$$
A \mathbf{c}_{\text{iso}} = \mathbf{b}
$$
where:
$$
A = \sum_i M_i, \quad \mathbf{b} = \sum_i M_i \mathbf{C}_i
$$

The iso-center is given by:
$$
\mathbf{c}_{\text{iso}} = A^{\dagger} \mathbf{b}
$$

### 2. Rotation Axis Estimation
The global rotation axis $\mathbf{a}_{\text{rot}}$ is estimated by performing Principal Component Analysis (PCA) on the camera centers $\mathbf{C}_i$.
We center the camera centers:
$$
\mathbf{C}'_i = \mathbf{C}_i - \bar{\mathbf{C}}
$$
and compute their Singular Value Decomposition (SVD):
$$
\text{SVD}(\mathbf{C}') = U \Sigma V^T
$$
The rotation axis is the singular vector corresponding to the smallest singular value (least variance), which is normal to the camera trajectory plane:
$$
\mathbf{a}_{\text{rot}} = \mathbf{v}_3 \cdot \text{sign}(v_{3, z})
$$
*(where the sign is chosen to keep the z-component positive).*

---

## JSON Configuration Example

Here is an example configuring a `ParameterizationChain` composed of `DetectorOrientation` and `RotationAxis`:

```json
{
  "module": "xray_epipolar_consistency.parameterization.base",
  "classname": "ParameterizationChain",
  "parameterizations": [
    {
      "module": "xray_epipolar_consistency.parameterization.detector_orientation",
      "classname": "DetectorOrientation",
      "parameters": {
        "tilt": { "opt": true },
        "slant": { "opt": true },
        "skew": { "opt": true }
      }
    },
    {
      "module": "xray_epipolar_consistency.parameterization.rotation_axis",
      "classname": "RotationAxis",
      "parameters": {
        "tilt_x": { "opt": false },
        "tilt_y": { "opt": false },
        "offset_x": { "opt": true },
        "offset_y": { "opt": true }
      }
    }
  ]
}
```
