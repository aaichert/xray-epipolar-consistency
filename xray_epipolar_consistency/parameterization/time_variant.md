← [Back to Parameterization Index](../parameterization.md)

# Time Variant Parameterization Wrappers

Wraps stationary parameterizations to represent time-varying motion across a trajectory.

## Subclasses

- `LinearDrift`: Linear interpolation between control points.
- `ContinuousMotion`: Cubic spline interpolation between control points.

---

## Mathematical Formulation

Instead of optimizing a single value $p$ for the entire trajectory, parameters are defined at $N$ discrete control points:

$$
\mathbf{p}_{\text{cp}} = [p_0, p_1, \dots, p_{N-1}]
$$

For the $i$-th projection matrix in a scan of $M$ total projections, we compute the normalized trajectory index:

$$
\lambda_i = \frac{i}{M - 1} \in [0, 1]
$$

### 1. Interpolation Schemes

- **`LinearDrift`**:
  We compute the scaled index and fractional component:
  $$
  v = \lambda_i (N - 1), \quad k = \lfloor v \rfloor, \quad t = v - k
  $$
  The interpolated parameter value is:
  $$
  p_i = (1 - t) p_k + t p_{k+1}
  $$

- **`ContinuousMotion`**:
  A cubic spline interpolates the parameters:
  $$
  p_i = \text{CubicSpline}(\lambda_i)
  $$
  using control point locations $x_k = \frac{k}{N-1}$ for $k=0,\dots,N-1$.

### 2. Application
For each projection matrix $P_i$, the interpolated parameter vector $\mathbf{p}_i$ is computed and applied using the referenced stationary parameterization:

$$
P_{i, \text{new}} = \text{apply}(P_i, \mathbf{p}_i)
$$

---

## JSON Configuration Example

Here is an example configuring a cubic spline time-varying `ContinuousMotion` wrapping the stationary `ObjectPose` parameterization with 5 control points:

```json
{
  "module": "xray_epipolar_consistency.parameterization.time_variant",
  "classname": "ContinuousMotion",
  "num_control_points": 5,
  "referenced_module": "xray_epipolar_consistency.parameterization.object_pose",
  "referenced_classname": "ObjectPose",
  "referenced_config": {
    "parameters": {
      "translation_x": { "opt": true },
      "translation_y": { "opt": true },
      "translation_z": { "opt": true },
      "rotation_x": { "opt": true },
      "rotation_y": { "opt": true },
      "rotation_z": { "opt": true }
    }
  }
}
```
