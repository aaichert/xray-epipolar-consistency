← [Back to Parameterization Index](../parameterization.md)

# Gantry Angle Parameterization

Corrects primary and secondary angular misalignment of the gantry.

## Parameters

- `primary_angle` ($\theta_p$): Primary gantry rotation correction [degrees].
- `secondary_angle` ($\theta_s$): Secondary gantry rotation correction [degrees].

---

## Mathematical Formulation

The correction is applied as a 3D coordinate transformation $T$ in the world frame:

$$
P_{\text{new}} = P T
$$

### 1. Defining Plücker Rotation Axes
Using the estimated iso-center $\mathbf{c}_{\text{iso}}$ and global rotation axis $\mathbf{a}_{\text{rot}}$ from prior knowledge, and the projection matrix's principal ray $\hat{\mathbf{r}}$:
- **Primary Axis** ($L_p$): The line passing through the iso-center in the direction of the rotation axis:
  $$
  L_p = \text{join}(\mathbf{c}_{\text{iso}}, \mathbf{a}_{\text{rot}})
  $$
- **Secondary Axis** ($L_s$): The line passing through the iso-center in the direction perpendicular to both the principal ray and the rotation axis:
  $$
  L_s = \text{join}(\mathbf{c}_{\text{iso}}, \hat{\mathbf{r}} \times \mathbf{a}_{\text{rot}})
  $$

### 2. Line Rotation Matrix
A rotation by angle $\alpha$ around a Plücker line $L = (\mathbf{d}, \mathbf{m})$ is calculated using the exponential map:

$$
\text{rotation}(L, \alpha) = \exp(\alpha \Xi)
$$

where the generator matrix $\Xi$ is:

$$
\Xi = \begin{bmatrix} [\hat{\mathbf{d}}]_{\times} & \mathbf{m} \\ \mathbf{0}^T & 0 \end{bmatrix}
$$

with $\hat{\mathbf{d}} = \frac{\mathbf{d}}{\|\mathbf{d}\|}$ being the normalized axis direction.

### 3. Combined Gantry Transformation ($T$)
The final transformation $T$ sequentially rotates around the secondary and primary axes:

$$
T = \text{rotation}(L_s, \theta_s) \cdot \text{rotation}(L_p, \theta_p)
$$

---

## JSON Configuration Example

```json
{
  "module": "xray_epipolar_consistency.parameterization.gantry_angle",
  "classname": "GantryAngle",
  "parameters": {
    "primary_angle": { "opt": true },
    "secondary_angle": { "opt": true }
  }
}
```
