← [Back to Parameterization Index](../parameterization.md)

# Detector Orientation Parameterization

Corrects global or dynamic detector orientation errors using 2D and 3D coordinate transformations. 

To bridge C-arm/medical, turntable, and general robotics vocabularies, the parameters are named redundantly using both conventions:
* **Roll / Tilt**: Rotation of the detector panel about its normal (optical axis).
* **Yaw / Slant**: Horizontal rotation about the detector's vertical axis.
* **Pitch / Skew**: Vertical rotation about the detector's horizontal axis.

## Parameters

- `tilt_roll` ($\theta_{\text{roll}}$): In-plane rotation (tilt/roll) about the principal point [degrees].
- `slant_yaw` ($\theta_{\text{yaw}}$): Out-of-plane rotation slant/yaw [degrees] (rotation about detector vertical axis).
- `skew_pitch` ($\theta_{\text{pitch}}$): Out-of-plane rotation skew/pitch [degrees] (rotation about detector horizontal axis).

---

## Typical Calibration Examples
* **Detector Panel Mounting Errors (Static)**: When mounting the detector panel, it might have a minor twist (`tilt_roll`) or slant/skew misalignment. Corrected statically using `detector_misalignment_static.json`.
* **Mechanical Flexing / Gantry Sag (Dynamic)**: On C-arms, gravity causes the heavy detector to twist or tilt out-of-plane as the gantry rotates. Corrected by wrapping `DetectorOrientation` parameters in `ContinuousMotion` (e.g., in `detector_misalignment_dynamic.json`).

---

## Mathematical Formulation

For a given projection matrix $P = K R [I_3 \mid -\mathbf{C}]$, the updated matrix $P_{\text{new}}$ is:

$$
P_{\text{new}} = H_{\text{tilt}} H_{\text{slant,skew}} P
$$

### 1. In-Plane Tilt/Roll Homography ($H_{\text{tilt}}$)
Rotates the detector plane coordinates by angle $\theta_{\text{roll}}$ around the principal point $\mathbf{p}_p = (c_x, c_y)^T$:

$$
H_{\text{tilt}} = T_{2D}(\mathbf{p}_p) R_{2D}(\theta_{\text{roll}}) T_{2D}(-\mathbf{p}_p)
$$

where:
- $T_{2D}(\mathbf{d})$ translates 2D coordinates by vector $\mathbf{d}$.
- $R_{2D}(\theta)$ is a standard 2D rotation matrix.

### 2. Out-of-Plane Slant/Yaw and Skew/Pitch Homography ($H_{\text{slant,skew}}$)
Rotates the camera's local 3D coordinates around its $y$-axis (slant/yaw) and $x$-axis (skew/pitch):

$$
R_{\text{slant,skew}} = R_y(\theta_{\text{yaw}}) R_x(\theta_{\text{pitch}})
$$

This 3D camera-space rotation is projected onto the detector plane via the intrinsic matrix $K$:

$$
H_{\text{slant,skew}} = K R_{\text{slant,skew}} K^{-1}
$$

---

## JSON Configuration Example

```json
{
  "module": "xray_epipolar_consistency.parameterization.detector_orientation",
  "classname": "DetectorOrientation",
  "parameters": {
    "tilt_roll": { "opt": true },
    "slant_yaw": { "opt": true },
    "skew_pitch": { "opt": true }
  }
}
```
