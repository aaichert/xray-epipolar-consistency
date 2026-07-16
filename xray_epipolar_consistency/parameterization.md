# Parameterization Index

This document serves as an index for the geometric parameterization modules used in epipolar consistency calibration.

---

## Parameterization Modules

- **[Base Parameterization Class](parameterization/base.md)**
  Defines the abstract base classes for parameter mapping and handles the estimation of the immutable trajectory `prior_knowledge` (iso-center and rotation axis).

- **[Detector Orientation](parameterization/detector_orientation.md)**
  Models global detector misalignment using in-plane tilt and out-of-plane slant/skew rotations.

- **[Detector Shift](parameterization/detector_shift.md)**
  Models global detector misalignment using 2D translations in detector coordinates.

- **[Distance](parameterization/distance.md)**
  Models global source-detector and source-isocenter distance adjustments.

- **[Gantry Angle](parameterization/gantry_angle.md)**
  Models primary and secondary angular gantry misalignments using 3D rotations about Plücker axes.

- **[Object Pose](parameterization/object_pose.md)**
  Models 3D rigid translations and rotations of the scanned object relative to the estimated iso-center.

- **[Rotation Axis](parameterization/rotation_axis.md)**
  Models rotation axis errors using tilt angles and spatial offsets.

- **[Turntable](parameterization/turntable.md)**
  Models the complete set of rotation axis and detector misalignments in a turntable CT setup.

- **[Refinement](parameterization/refinement.md)**
  Models coupled geometric parameters along their near-null-space directions to refine slant/skew, source shifts, and vertical alignment.

- **[Time Variant](parameterization/time_variant.md)**
  Wrappers to model time-varying parameters (like linear drift, continuous spline motion, or jitter) across the trajectory.

---

## Coordinate System Independence

The parameterization framework is fully coordinate-system independent. It makes **no assumptions** about specific axes (e.g., which axis is $X$, $Y$, or $Z$), nor does it assume that the origin is at the isocenter.

To ensure this independence:
* **Nominal Geometry Fitting (`prior_knowledge`)**: When loading a trajectory, the library fits a plane to the estimated source positions (defining the rotation axis $\mathbf{a}_{\text{rot}}$) and finds the isocenter $\mathbf{c}_{\text{iso}}$ as the point closest to all principal rays.
* **Local Reference Frames**: Modality-specific direction vectors (like the lateral axis $\mathbf{u}$ and radial axis $\mathbf{v}$ on the rotation plane) are computed dynamically per-projection, ensuring they rotate correctly with the gantry regardless of the world coordinate system.
* **Isocenter Centering**: 3D rotations (in `GantryAngle`, `RotationAxis`, and `ObjectPose`) are always applied relative to the estimated $\mathbf{c}_{\text{iso}}$, making them invariant to the position of the world origin.

---

## Physical Errors & Optimization Stages

Since geometric inconsistency can often be mathematically explained in multiple ways (e.g., a detector shift vs. a rotation axis offset), **the order of optimization is critical**. We must optimize the most dominant parameters first to anchor the geometry, followed by subtle or time-varying corrections.

The table below maps common scanner/experimental errors to parameter combinations, time-varying wrappers, and the sequential stage JSON configuration files.

| Modality / Error Type | Physical Cause | Primary Parameterization | Time-Varying Wrapper | Suggested Stage Config Filename | Optimization Order & Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Detector Misalignment (Static)** | Static mounting offsets (detector shifts and out-of-plane orientations). | `DetectorShift` + `DetectorOrientation` | *None* | `detector_misalignment_static.json` | **1. Major/Dominant:** Optimize 2D shifts (`shift_u`, `shift_v`) and out-of-plane tilts (`tilt_roll`, `slant_yaw`, `skew_pitch`). Typically run first to establish the baseline detector frame. |
| **Detector Misalignment (Dynamic)** | Angle-dependent panel sag, vibration, or thermal panel movement. | `DetectorShift` + `DetectorOrientation` | `ContinuousMotion` | `detector_misalignment_dynamic.json` | **3. Dynamics:** Optimize time-varying panel movement after static calibration has anchored the gantry. |
| **Distances & Magnification (Static)** | Errors in manually-measured source-detector (SDD) and source-isocenter (SID) distances. | `Distance` | *None* | `distances_static.json` | **2. Scale:** Optimize static `delta_sdd` and `delta_sid` after major shifts are centered. Distances have a subtle effect on epipolar lines and should not absorb lateral offsets. |
| **Distances & Magnification (Dynamic)** | Slow thermal focal spot drift or C-arm gantry flexing under gravity. | `Distance` | `LinearDrift` or `ContinuousMotion` | `distances_dynamic.json` | **3. Dynamics:** Optimize drift parameters together with wobble and jitter in final refinement stages. |
| **Turntable CT (Static)** | Stationary tilt/offset of rotation axis relative to imaging geometry. | `RotationAxis` | *None* | `turntable_stage1_static_axis.json` | **1. Major:** Optimize `offset_lateral` and `tilt_roll` first (largest projection impact), followed by `offset_radial` and `tilt_pitch`. |
| **Turntable CT (Wobble)** | Mechanical runout/wobble of the axis during rotation. | `RotationAxis` | `ContinuousMotion` | `turntable_stage2_axis_wobble.json` | **3. Dynamics:** Optimize time-varying wobble after static calibration has anchored the nominal axis. |
| **Object / Patient Motion** | Gradual/sudden movement of the object during acquisition. | `ObjectPose` | `LinearDrift` / `ContinuousMotion` | `object_motion_linear.json`<br>`object_motion_spline.json` | **1. Motion Tracking:** Optimize time-varying translations and rotations to track patient/object displacement. |
| **Laminography / Planar CT** | Tilt of translation plane; linear trajectory errors (translational drift). | `ObjectPose` | `LinearDrift` or `ContinuousMotion` | `laminography_stage1_plane_tilt.json`<br>`laminography_stage2_trajectory_drift.json` | **1. Major:** Optimize static plane tilt first (`ObjectPose` rotations). <br>**2. Drift:** Optimize linear translational drift (`ObjectPose` translations). |

---

## Modal Adaptations

1. **Tomosynthesis (Dental or Breast Imaging)**:
   * *Typical Errors:* Short-scan angular ranges (e.g. $15^\circ$ to $50^\circ$) amplify out-of-plane detector positioning errors and gantry angular jitter.
2. **Helical / Spiral CT**:
   * *Typical Errors:* Pitch inaccuracies and table translation speed jitter. Corrected by wrapping `ObjectPose` translation along the helix axis in `ContinuousMotion` or `LinearDrift`.
3. **Planar CT / Laminography**:
   * *Typical Errors:* Linear translation plane tilt and linear trajectory drift. Corrected by optimizing trajectory translation drift.
4. **Robotic C-Arm Systems**:
   * *Typical Errors:* Cumulative multi-joint odometry drift. Corrected by wrapping `ObjectPose` in `ContinuousMotion` to track the full 6-DOF trajectory deviation.


