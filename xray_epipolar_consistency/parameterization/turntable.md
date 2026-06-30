← [Back to Parameterization Index](../parameterization.md)

# Turntable Parameterization

Corrects static or time-varying geometric misalignments in a CT system comprised of a rotating turntable and a flat-panel detector. This class parameterizes both the turntable's axis of rotation in 3D world space and the flat-panel detector's extrinsic/intrinsic positioning, including self-contained linear drift parameters for thermal or positional stability calibration.

---

## Parameters

### Turntable Axis Misalignments
- `axis_offset_lateral` ($t_{\text{lat}}$): Translation of the rotation axis along the lateral axis $\mathbf{u}$ [mm].
- `axis_offset_radial` ($t_{\text{rad}}$): Translation of the rotation axis along the radial axis $\mathbf{v}$ [mm].
- `axis_tilt_pitch` ($\theta_{\text{pitch}}$): Tilt of the rotation axis around the lateral axis $\mathbf{u}$ [degrees].
- `axis_tilt_roll` ($\theta_{\text{roll}}$): Tilt of the rotation axis around the radial axis $\mathbf{v}$ [degrees].

### Detector Positioning and Distances
- `detector_shift_u` ($s_u$): Static detector shift in the horizontal $u$ direction [pixels].
- `detector_shift_v` ($s_v$): Static detector shift in the vertical $v$ direction [pixels].
- `delta_sdd` ($\Delta_{\text{sdd}}$): Static source-detector distance offset [mm].
- `delta_sid` ($\Delta_{\text{sid}}$): Static source-isocenter distance offset [mm].

### Linear Drift Parameters (Self-Contained)
- `drift_detector_shift_u` ($d_{s_u}$): Linear drift rate of horizontal detector shift over the trajectory [pixels].
- `drift_detector_shift_v` ($d_{s_v}$): Linear drift rate of vertical detector shift over the trajectory [pixels].
- `drift_sdd` ($d_{\text{sdd}}$): Linear drift rate of source-detector distance offset over the trajectory [mm].
- `drift_sid` ($d_{\text{sid}}$): Linear drift rate of source-isocenter distance offset over the trajectory [mm].

### Detector Orientation
- `detector_roll` ($\phi_{\text{roll}}$): Detector roll (in-plane rotation) [degrees].
- `detector_pitch` ($\phi_{\text{pitch}}$): Detector pitch (out-of-plane rotation) [degrees].
- `detector_yaw` ($\phi_{\text{yaw}}$): Detector yaw (out-of-plane rotation) [degrees].

---

## Typical Calibration Examples
* **Static Axis Misalignment**: The physical turntable rotation axis is offset or tilted relative to the nominal configuration. This is corrected statically by optimizing the axis parameters.
* **Detector Panel Mount Error**: The detector panel has translation offsets or in-plane/out-of-plane orientation offsets relative to the focal spot. Corrected using the detector shift and orientation parameters.
* **Thermal Expansion & Drift**: Heat from the X-ray tube causes linear expansion of the mechanical assembly or focal spot shift over time. This is corrected directly within this class by optimizing the linear drift parameters (`drift_sdd`, `drift_sid`, `drift_detector_shift_u`, `drift_detector_shift_v`).

---

## Mathematical Formulation

The projection matrix $P$ is updated sequentially by applying a 3D transformation to the world coordinate system (corresponding to the rotation axis misalignment and SID translation) and a 2D homography to the detector projection (corresponding to the detector shifts, SDD scaling, and detector orientation).

### Trajectory Progress Interpolation
When applying the parameterization to a trajectory of length $N$, a progress factor $\lambda_i \in [0, 1]$ is computed for each projection index $i \in [0, N-1]$:

$$
\lambda_i = \frac{i}{\max(1, N - 1)}
$$

This progress factor is used to interpolate the time-varying parameters:

*   $s_u(\lambda_i) = s_u + \lambda_i \cdot d_{s_u}$
*   $s_v(\lambda_i) = s_v + \lambda_i \cdot d_{s_v}$
*   $\Delta_{\text{sdd}}(\lambda_i) = \Delta_{\text{sdd}} + \lambda_i \cdot d_{\text{sdd}}$
*   $\Delta_{\text{sid}}(\lambda_i) = \Delta_{\text{sid}} + \lambda_i \cdot d_{\text{sid}}$

### 1. 3D World Space Transformations
The local orthonormal frame at the isocenter is defined by:
*   $\mathbf{d}$: Normalized nominal rotation axis direction.
*   $\mathbf{u}$: Normalized lateral axis (orthogonal to the principal ray $\mathbf{r}$ and axis $\mathbf{d}$).
*   $\mathbf{v} = \mathbf{d} \times \mathbf{u}$: Normalized radial axis.

The 3D rigid transformation matrix representing the rotation axis shift and tilt is:

$$
T_{\text{world}} = \begin{bmatrix} R_{\text{tilt}} & C - R_{\text{tilt}} C + t_{\text{shift}} \\ \mathbf{0}^T & 1 \end{bmatrix}
$$

where $R_{\text{tilt}} = R_{\mathbf{v}}(\theta_{\text{roll}}) R_{\mathbf{u}}(\theta_{\text{pitch}})$, $t_{\text{shift}} = t_{\text{lat}} \mathbf{u} + t_{\text{rad}} \mathbf{v}$, and $C$ is the nominal isocenter.

Applying the axis transformation:

$$
P_{\text{world}} = P \cdot T_{\text{world}}
$$

For the Source-Isocenter Distance (SID) offset (including linear drift), we shift along the principal ray $\mathbf{r}$:

$$
T_{\text{sid}} = \begin{bmatrix} I & -\Delta_{\text{sid}}(\lambda_i) \mathbf{r} \\ \mathbf{0}^T & 1 \end{bmatrix}
$$

$$
P_{\text{geom}} = P_{\text{world}} \cdot T_{\text{sid}}
$$

### 2. 2D Detector Transformations (Homographies)
Let $(c_x, c_y)$ be the principal point and $f$ be the focal length.

*   **SDD Scaling ($H_{\text{scale}}$)**:
    $$H_{\text{scale}} = \begin{bmatrix} s & 0 & (1-s)c_x \\ 0 & s & (1-s)c_y \\ 0 & 0 & 1 \end{bmatrix}$$
    where $s = \frac{\text{sdd} + \Delta_{\text{sdd}}(\lambda_i)}{\text{sdd}}$.

*   **Detector Out-of-Plane Rotations ($H_{\text{out}}$)**:
    $$H_{\text{out}} = K R_y(\phi_{\text{yaw}}) R_x(\phi_{\text{pitch}}) K^{-1}$$

*   **Detector In-Plane Rotation ($H_{\text{roll}}$)**:
    $$H_{\text{roll}} = \begin{bmatrix} \cos\phi & -\sin\phi & c_x(1-\cos\phi) + c_y\sin\phi \\ \sin\phi & \cos\phi & c_y(1-\cos\phi) - c_x\sin\phi \\ 0 & 0 & 1 \end{bmatrix}$$

*   **Detector Shift ($H_{\text{shift}}$)**:
    $$H_{\text{shift}} = \begin{bmatrix} 1 & 0 & s_u(\lambda_i) \\ 0 & 1 & s_v(\lambda_i) \\ 0 & 0 & 1 \end{bmatrix}$$

The final projection matrix is:

$$
P_{\text{final}} = H_{\text{shift}} \cdot H_{\text{roll}} \cdot H_{\text{out}} \cdot H_{\text{scale}} \cdot P_{\text{geom}}
$$

---

## JSON Configuration Example

```json
{
  "module": "xray_epipolar_consistency.parameterization.turntable",
  "classname": "Turntable",
  "parameters": {
    "axis_offset_lateral": { "opt": true },
    "axis_offset_radial": { "opt": true },
    "axis_tilt_pitch": { "opt": true },
    "axis_tilt_roll": { "opt": true },
    "detector_shift_u": { "opt": true },
    "detector_shift_v": { "opt": true },
    "delta_sdd": { "opt": true },
    "delta_sid": { "opt": true },
    "drift_detector_shift_u": { "opt": true },
    "drift_detector_shift_v": { "opt": true },
    "drift_sdd": { "opt": true },
    "drift_sid": { "opt": true },
    "detector_roll": { "opt": true },
    "detector_pitch": { "opt": true },
    "detector_yaw": { "opt": true }
  }
}
```
