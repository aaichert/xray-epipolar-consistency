← [Back to Parameterization Index](../parameterization.md)

# Distance Parameterization

Corrects source-detector and source-isocenter distance errors (magnification calibration).

## Parameters

- `delta_sid` ($\Delta_{\text{sid}}$): Source-isocenter distance difference [mm].
- `delta_sdd` ($\Delta_{\text{sdd}}$): Source-detector distance difference [mm].

---

## Typical Calibration Examples
* **Manual Laboratory Setups (Static)**: In custom CT or micro-CT setups, geometry distances are often measured manually (introducing 5–20 mm errors). Static calibration corrects these nominal distance offsets (using `distances_static.json`).
* **Thermal Tube Heating (Drift)**: High-power micro-CT systems heat up during long scans, causing linear expansion of the frame and shifting the source/detector further apart. Wrapped in `LinearDrift` (using `distances_dynamic.json`).
* **C-Arm Mechanical Flexing (Dynamic)**: Due to gravity, the heavy tube and detector panel bend the C-arm structure, causing dynamic changes in SDD/SID. Wrapped in `ContinuousMotion`.

---

## Mathematical Formulation

For a given projection matrix $P$, the updated matrix is:

$$
P_{\text{new}} = H P T
$$

### 1. Source Translation ($T$)
Translates the source (camera center) along the principal ray direction $\hat{\mathbf{r}}$ (normalized):

$$
T = \begin{bmatrix} I_3 & -\Delta_{\text{sid}} \hat{\mathbf{r}} \\ \mathbf{0}^T & 1 \end{bmatrix}
$$

This shifts the camera center by $+\Delta_{\text{sid}} \hat{\mathbf{r}}$ in the world frame.

### 2. Detector Scaling ($H$)
Corrects the detector magnification change caused by shifting the detector relative to the source.
Let $d_{\text{sdd}}$ be the original source-detector distance (obtained from `SourceDetectorGeometry`). The new distance is $d_{\text{sdd}} + \Delta_{\text{sdd}}$, yielding the scaling factor:

$$
s = \frac{d_{\text{sdd}} + \Delta_{\text{sdd}}}{d_{\text{sdd}}}
$$

This scaling is applied as a 2D homography $H$ centered at the principal point $\mathbf{p}_p = (c_x, c_y)^T$:

$$
H = T_{2D}(\mathbf{p}_p) S_{2D}(s) T_{2D}(-\mathbf{p}_p)
$$

---

## JSON Configuration Example

```json
{
  "module": "xray_epipolar_consistency.parameterization.distance",
  "classname": "Distance",
  "parameters": {
    "delta_sdd": { "opt": true },
    "delta_sid": { "opt": true }
  }
}
```
