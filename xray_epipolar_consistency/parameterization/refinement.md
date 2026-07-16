← [Back to Parameterization Index](../parameterization.md)

# Refinement Parameterization

The `Refinement` parameterization is designed for fine-tuning CT geometry calibration. It optimizes coupled geometric parameters along their near-null-space directions, allowing the optimizer to resolve subtle 3D shape, magnification, or perspective errors (which improve reconstruction quality and resolve blurring) without shifting the bulk of projection coordinates in the images.

---

## Parameters

- `refine_slant`: Coupled detector slant (out-of-plane rotation around detector $v$) and compensating lateral shift ($u$) [degrees].
- `refine_skew`: Coupled detector skew (out-of-plane rotation around detector $u$) and compensating vertical shift ($v$) [degrees].
- `refine_source_x`: Coupled source shift (along detector $u$) and compensating detector shift ($u$) [mm].
- `refine_axial_z`: Coupled object translation (along rotation axis $z$) and compensating vertical detector shift ($v$) [mm].

---

## Typical Calibration Examples

* **Fine-Tuning Out-of-Plane Tilts (Static)**: Detector slant and skew can cause severe blurring/axial artifacts in reconstruction. Optimizing them directly is difficult due to strong coupling with simple detector shifts. Using `refine_slant` and `refine_skew` allows resolving these tilts while keeping the projections centered.
* **Axial Divergence / Cone-beam Refinement**: Optimizing `refine_axial_z` resolves the vertical position of the object relative to the cone-beam central plane.
* **Trajectory Refinement**: Optimizing `refine_source_x` resolves minor source trajectory wobble or gantry rotation axis tilt.

---

## Mathematical Formulation

For each coupled parameter, the first-order translation of projection coordinates on the detector is canceled by applying a compensating detector shift homography:

### A. Slant/Skew Compensation
For an out-of-plane slant rotation $\alpha$ and skew rotation $\beta$ on a camera with focal length $f$ (in pixels):
$$
d_u^{\text{comp}} = -\alpha \cdot f, \quad d_v^{\text{comp}} = -\beta \cdot f
$$

### B. Source Shift Compensation
For a source translation $s_x$ along the lateral direction, with source-isocenter distance $\text{SID}$, source-detector distance $\text{SDD}$, and pixel spacing $p_u$:
$$
d_u^{\text{comp}} = -s_x \cdot \frac{\text{SDD} - \text{SID}}{\text{SID} \cdot p_u}
$$

### C. Axial Translation Compensation
For an object translation $z$ along the rotation axis, with magnification $M = \frac{\text{SDD}}{\text{SID}}$ and pixel spacing $p_v$:
$$
d_v^{\text{comp}} = z \cdot \frac{M}{p_v}
$$

The combined translations and rotations are applied sequentially to the projection matrix $P$.

---

## JSON Configuration Example

```json
{
  "module": "xray_epipolar_consistency.parameterization.refinement",
  "classname": "Refinement",
  "parameters": {
    "refine_slant": { "opt": true },
    "refine_skew": { "opt": true },
    "refine_source_x": { "opt": true },
    "refine_axial_z": { "opt": true }
  }
}
```
