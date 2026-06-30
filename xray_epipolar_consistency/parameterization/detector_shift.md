← [Back to Parameterization Index](../parameterization.md)

# Detector Shift Parameterization

Corrects global or time-varying detector translation/shift errors in the 2D detector coordinates.

## Parameters

- `shift_u` ($o_u$): Detector shift in the horizontal $u$ direction [pixels].
- `shift_v` ($o_v$): Detector shift in the vertical $v$ direction [pixels].

---

## Typical Calibration Examples
* **Detector Panel Mounting Errors (Static)**: Minor shift of the detector panel placement relative to the source. Corrected statically using `detector_misalignment_static.json`.
* **C-Arm Detector Sag (Dynamic)**: Due to gravity, the heavy detector panel sags vertically as the C-arm rotates. Corrected by optimizing `shift_v` wrapped in `ContinuousMotion` (e.g., in `detector_misalignment_dynamic.json`).
* **Thermal expansion drift**: Temperature changes cause expansion of the mechanical arm, shifting the detector horizontally or vertically over time. Modeled via `LinearDrift`.

---

## Mathematical Formulation

Applies a 2D translation homography to the projection matrix $P$:

$$
P_{\text{new}} = H P
$$

where $H$ is the 2D homography matrix:

$$
H = \begin{bmatrix} 1 & 0 & o_u \\ 0 & 1 & o_v \\ 0 & 0 & 1 \end{bmatrix}
$$

---

## JSON Configuration Example

```json
{
  "module": "xray_epipolar_consistency.parameterization.detector_shift",
  "classname": "DetectorShift",
  "parameters": {
    "shift_u": { "opt": true },
    "shift_v": { "opt": true }
  }
}
```
