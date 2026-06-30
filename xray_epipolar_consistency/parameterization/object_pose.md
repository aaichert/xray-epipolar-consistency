← [Back to Parameterization Index](../parameterization.md)

# Object Pose Parameterization

Applies a rigid 3D object transformation about the estimated iso-center.

## Parameters

- `translation_x` ($t_x$): Object translation along the X-axis [mm].
- `translation_y` ($t_y$): Object translation along the Y-axis [mm].
- `translation_z` ($t_z$): Object translation along the Z-axis [mm].
- `rotation_x` ($\theta_x$): Object rotation about the X-axis [degrees].
- `rotation_y` ($\theta_y$): Object rotation about the Y-axis [degrees].
- `rotation_z` ($\theta_z$): Object rotation about the Z-axis [degrees].

---

## Mathematical Formulation

The correction is applied to the world coordinates via transformation $T$:

$$
P_{\text{new}} = P T
$$

### Object Transformation ($T$)
The rotation is centered around the estimated iso-center $\mathbf{c}_{\text{iso}}$, and the translation is applied afterwards:

$$
T = T_{\text{3D}}(t_x, t_y, t_z) \cdot T_{\text{3D}}(\mathbf{c}_{\text{iso}}) \cdot R_z(\theta_z) R_y(\theta_y) R_x(\theta_x) \cdot T_{\text{3D}}(-\mathbf{c}_{\text{iso}})
$$

where:
- $T_{\text{3D}}(\mathbf{d})$ is the $4 \times 4$ homogeneous translation matrix by vector $\mathbf{d}$.
- $R_x, R_y, R_z$ are the $4 \times 4$ homogeneous rotation matrices around the respective axes.

---

## JSON Configuration Example

```json
{
  "module": "xray_epipolar_consistency.parameterization.object_pose",
  "classname": "ObjectPose",
  "parameters": {
    "translation_x": { "opt": true },
    "translation_y": { "opt": true },
    "translation_z": { "opt": true },
    "rotation_x": { "opt": true },
    "rotation_y": { "opt": true },
    "rotation_z": { "opt": true }
  }
}
```
