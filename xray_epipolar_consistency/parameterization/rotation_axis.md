← [Back to Parameterization Index](../parameterization.md)

# Rotation Axis Parameterization

Corrects rotation axis misalignment using tilts and offsets defined in a local coordinate frame relative to the turntable.

The parameters describe the physical misalignment (tilt and offset) of the turntable's rotation axis relative to the imaging geometry (principal ray and nominal rotation axis).

## Parameters

- `tilt_pitch` ($\phi_{\text{pitch}}$): Rotation axis tilt in the radial plane (pitching forward/backward relative to the camera) [degrees]. This is a rotation about the local lateral axis $\mathbf{u}$.
- `tilt_roll` ($\phi_{\text{roll}}$): Rotation axis tilt in the lateral plane (rolling left/right relative to the camera) [degrees]. This is a rotation about the local radial axis $\mathbf{v}$.
- `offset_lateral` ($o_{\text{lat}}$): Rotation axis lateral offset [mm] (along the local lateral axis $\mathbf{u}$).
- `offset_radial` ($o_{\text{rad}}$): Rotation axis radial offset [mm] (along the local radial axis $\mathbf{v}$, i.e., in the direction of the principal ray).

---

## Mathematical Formulation

The correction is applied as a 3D coordinate transformation $T$ in the world frame:

$$
P_{\text{new}} = P T
$$

### 1. Defining Local Directions
Given the current projection matrix's principal ray direction $\hat{\mathbf{r}}$ and the global rotation axis $\mathbf{a}_{\text{rot}}$ from prior knowledge, we define a local orthonormal coordinate frame on the rotation plane:
- $\mathbf{d} = \text{normalize}(\mathbf{a}_{\text{rot}})$ (axial direction).
- $\mathbf{u} = \text{normalize}(\hat{\mathbf{r}} \times \mathbf{d})$ (lateral/tangential direction on the rotation plane, perpendicular to both the rotation axis and the principal ray).
- $\mathbf{v} = \mathbf{d} \times \mathbf{u}$ (radial direction on the rotation plane, representing the projection of the principal ray onto the rotation plane).

### 2. Tilt Axes (Plücker lines)
- **Tilt Pitch Axis** ($L_{\text{pitch}}$): Line passing through the iso-center $\mathbf{c}_{\text{iso}}$ in direction $\mathbf{u}$:
  $$
  L_{\text{pitch}} = \text{join}(\mathbf{c}_{\text{iso}}, \mathbf{u})
  $$
- **Tilt Roll Axis** ($L_{\text{roll}}$): Line passing through the iso-center $\mathbf{c}_{\text{iso}}$ in direction $\mathbf{v}$:
  $$
  L_{\text{roll}} = \text{join}(\mathbf{c}_{\text{iso}}, \mathbf{v})
  $$

### 3. Combined Transformation ($T$)
The transformation translates the camera coordinate frame relative to the rotation axis, followed by rotations about the tilt axes:

$$
T = T_{\text{3D}}(o_{\text{lat}} \mathbf{u} + o_{\text{rad}} \mathbf{v}) \cdot \text{rotation}(L_{\text{pitch}}, \phi_{\text{pitch}}) \cdot \text{rotation}(L_{\text{roll}}, \phi_{\text{roll}})
$$

*(where the line rotation is calculated using the exponential map on Plücker coordinates as in the Gantry Angle parameterization).*

---

## JSON Configuration Example

```json
{
  "module": "xray_epipolar_consistency.parameterization.rotation_axis",
  "classname": "RotationAxis",
  "parameters": {
    "tilt_pitch": { "opt": false },
    "tilt_roll": { "opt": false },
    "offset_lateral": { "opt": true },
    "offset_radial": { "opt": true }
  }
}
```
