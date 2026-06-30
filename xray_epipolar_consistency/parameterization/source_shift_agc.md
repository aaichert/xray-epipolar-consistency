← [Back to Parameterization Index](../parameterization.md)

# Source Shift AGC Parameterization

Corrects translation/shift errors of the X-ray source relative to the detector in the local detector coordinate system.

## Parameters

- `source_shift_x` ($\Delta s_u$): Source shift in the local detector horizontal $u$ direction [mm].
- `source_shift_y` ($\Delta s_v$): Source shift in the local detector vertical $v$ direction [mm].
- `source_shift_z` ($\Delta s_w$): Source shift in the local detector optical axis / principal ray $w$ direction [mm].

---

## Typical Calibration Examples
* **X-ray Tube Alignment Errors**: Static mechanical misalignment of the source position relative to the detector gantry frame.
* **Gantry Sag/Flexing**: Geometric instabilities during rotation that cause the focal spot to shift in the detector coordinate system.

---

## Mathematical Formulation

Translating the source relative to the detector shifts the extrinsic translation vector $\mathbf{t}$ in camera coordinates to $\mathbf{t}' = \mathbf{t} - \Delta \mathbf{s}_{\text{local}}$, where $\Delta \mathbf{s}_{\text{local}} = [\Delta s_u, \Delta s_v, \Delta s_w]^T$.

This is applied to the un-normalized projection matrix $P$ (which maps voxels to pixels, scale factor $s = \| P_{2, 0..2} \|$) by subtracting a scaled intrinsic shift vector from the fourth column:

$$
P'_{\text{3}} = P_{\text{3}} - s K \Delta \mathbf{s}_{\text{local}}
$$

where $K$ is the $3 \times 3$ intrinsic matrix:

$$
K \Delta \mathbf{s}_{\text{local}} = \begin{bmatrix} f_u \Delta s_u + c_u \Delta s_w \\ f_v \Delta s_v + c_v \Delta s_w \\ \Delta s_w \end{bmatrix}
$$

and $f_u, f_v$ are focal lengths (in pixels), and $c_u, c_v$ are principal points (in pixels).

---

## JSON Configuration Example

```json
{
  "module": "xray_epipolar_consistency.parameterization.source_shift_agc",
  "classname": "SourceShiftAGC",
  "parameters": {
    "source_shift_x": { "opt": true },
    "source_shift_y": { "opt": true },
    "source_shift_z": { "opt": true }
  }
}
```
