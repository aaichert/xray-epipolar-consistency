import numpy as np
from scipy.spatial.transform import Rotation as Rot
from ProjectiveGeometry23.central_projection import ProjectionMatrix
from ProjectiveGeometry23.source_detector_geometry import SourceDetectorGeometry
from ProjectiveGeometry23.utils import dehomogenize
from xray_epipolar_consistency.parameterization import (
    DetectorShift,
    DetectorOrientation,
    ObjectPose,
    RotationAxis,
    Distance,
    GantryAngle,
    LinearDrift,
    ContinuousMotion,
    Turntable,
    Refinement
)

def get_trajectory():
    P1 = ProjectionMatrix.perspective_look_at(eye=np.array([1000.0, 0.0, 0.0]))
    P2 = ProjectionMatrix.perspective_look_at(eye=np.array([707.1, 0.0, 707.1]))
    P3 = ProjectionMatrix.perspective_look_at(eye=np.array([0.0, 0.0, 1000.0]))
    return [P1, P2, P3]

def test_all_parameterizations():
    # Test DetectorShift
    Ps = get_trajectory()
    ds = DetectorShift(parameters={
        "shift_u": {"value": 5.0, "opt": True},
        "shift_v": {"value": -3.0, "opt": True}
    })
    Ps_ds = ds.apply_to_trajectory(Ps)
    for P, P_new in zip(Ps, Ps_ds):
        pp = P.getPrincipalPoint().flatten()[:2]
        pp_new = P_new.getPrincipalPoint().flatten()[:2]
        np.testing.assert_allclose(pp_new - pp, [5.0, -3.0], atol=1e-7)

    # Test DetectorOrientation
    Ps = get_trajectory()
    do = DetectorOrientation(parameters={
        "tilt_roll": {"value": 5.0, "opt": True}
    })
    Ps_do = do.apply_to_trajectory(Ps)
    for P, P_new in zip(Ps, Ps_do):
        C = dehomogenize(P.getCenterOfProjection()).flatten()
        C_new = dehomogenize(P_new.getCenterOfProjection()).flatten()
        # Camera center shouldn't change for pure detector-plane rotations
        np.testing.assert_allclose(C_new, C, atol=1e-7)

    # Test Distance
    Ps = get_trajectory()
    dist = Distance(parameters={
        "delta_sdd": {"value": 15.0, "opt": True},
        "delta_sid": {"value": 10.0, "opt": True}
    })
    Ps_dist = dist.apply_to_trajectory(Ps)
    for P, P_new in zip(Ps, Ps_dist):
        r = P.getPrincipalRay().flatten()
        sdd = SourceDetectorGeometry(P).source_detector_distance
        sdd_new = SourceDetectorGeometry(P_new).source_detector_distance
        np.testing.assert_allclose(sdd_new - sdd, 15.0, atol=1e-7)

        # Camera center moves by +delta_sid * r (along principal ray)
        C = dehomogenize(P.getCenterOfProjection()).flatten()
        C_new = dehomogenize(P_new.getCenterOfProjection()).flatten()
        np.testing.assert_allclose(C_new - C, 10.0 * r, atol=1e-7)

    # Test ObjectPose
    Ps = get_trajectory()
    op = ObjectPose(parameters={
        "translation_x": {"value": 2.0, "opt": True},
        "translation_y": {"value": -3.0, "opt": True},
        "translation_z": {"value": 4.0, "opt": True}
    })
    Ps_op = op.apply_to_trajectory(Ps)
    for P, P_new in zip(Ps, Ps_op):
        C = dehomogenize(P.getCenterOfProjection()).flatten()
        C_new = dehomogenize(P_new.getCenterOfProjection()).flatten()
        # Since world translated by T = [2, -3, 4], camera center relative to the object translates by T^-1
        np.testing.assert_allclose(C_new - C, [-2.0, 3.0, -4.0], atol=1e-7)

    # Test RotationAxis
    Ps = get_trajectory()
    ra = RotationAxis(parameters={
        "offset_lateral": {"value": 1.5, "opt": True},
        "offset_radial": {"value": -2.5, "opt": True},
        "tilt_pitch": {"value": 0.0, "opt": True},
        "tilt_roll": {"value": 0.0, "opt": True}
    })
    Ps_ra = ra.apply_to_trajectory(Ps)
    pk_ra = ra.prior_knowledge
    rot_axis = np.array(pk_ra["rotation_axis"])
    rot_axis /= np.linalg.norm(rot_axis)
    for P, P_new in zip(Ps, Ps_ra):
        r = P.getPrincipalRay().flatten()
        r /= np.linalg.norm(r)
        u = np.cross(r, rot_axis)
        u /= np.linalg.norm(u)
        v = np.cross(rot_axis, u)
        
        C = dehomogenize(P.getCenterOfProjection()).flatten()
        C_new = dehomogenize(P_new.getCenterOfProjection()).flatten()
        # C_new = C - (offset_lateral * u + offset_radial * v)
        np.testing.assert_allclose(C_new - C, -(1.5 * u - 2.5 * v), atol=1e-7)

    # Test RotationAxis with tilts and verify reference implementation matches efficient implementation
    ra_tilts = RotationAxis(parameters={
        "offset_lateral": {"value": 1.2, "opt": True},
        "offset_radial": {"value": -0.8, "opt": True},
        "tilt_pitch": {"value": 0.5, "opt": True},
        "tilt_roll": {"value": -0.3, "opt": True}
    })
    Ps_ra_tilts = ra_tilts.apply_to_trajectory(Ps)
    assert "first_principal_ray" in ra_tilts.prior_knowledge
    for P in Ps:
        P_eff = ra_tilts.apply_stationary(P)
        P_ref = ra_tilts.apply_stationary_reference_impl(P)
        np.testing.assert_allclose(P_eff.P, P_ref.P, atol=1e-7)


    # Test GantryAngle
    Ps = get_trajectory()
    ga = GantryAngle(parameters={
        "primary_angle": {"value": 1.0, "opt": True},
        "secondary_angle": {"value": 0.0, "opt": True}
    })
    Ps_ga = ga.apply_to_trajectory(Ps)
    pk_ga = ga.prior_knowledge
    iso = np.array(pk_ga["iso_center"])
    rot_axis = np.array(pk_ga["rotation_axis"])
    for P, P_new in zip(Ps, Ps_ga):
        C = dehomogenize(P.getCenterOfProjection()).flatten()
        C_new = dehomogenize(P_new.getCenterOfProjection()).flatten()
        
        # GantryAngle rotates around primary_axis: line passing through iso with direction rot_axis.
        # sa is 0, so only pa is applied. Since P_new = P @ T, C_new = T^-1 @ C.
        # T is a rotation of pa degrees, so T^-1 rotates by -pa degrees.
        angle_rad = np.radians(1.0)
        rot = Rot.from_rotvec(-angle_rad * rot_axis)
        C_expected = rot.apply(C - iso) + iso
        np.testing.assert_allclose(C_new, C_expected, atol=1e-7)

    # Test TimeVariant subclasses wrapping DetectorShift
    Ps = get_trajectory()
    ref_cfg = {
        "parameters": {
            "shift_u": {"opt": True},
            "shift_v": {"opt": True}
        }
    }
    # LinearDrift
    ld = LinearDrift(referenced_class=DetectorShift, num_control_points=3, referenced_config=ref_cfg)
    # Set control points to: cp0=(2.0, 3.0), cp1=(4.0, 5.0), cp2=(6.0, 7.0)
    ld.set_parameter_vector([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    Ps_ld = ld.apply_to_trajectory(Ps)
    
    expected_shifts = [[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]
    for i, (P, P_new) in enumerate(zip(Ps, Ps_ld)):
        pp = P.getPrincipalPoint().flatten()[:2]
        pp_new = P_new.getPrincipalPoint().flatten()[:2]
        np.testing.assert_allclose(pp_new - pp, expected_shifts[i], atol=1e-7)

    # ContinuousMotion
    Ps = get_trajectory()
    cm = ContinuousMotion(referenced_class=DetectorShift, num_control_points=3, referenced_config=ref_cfg)
    cm.set_parameter_vector([0.0, 0.0, 1.0, 2.0, 4.0, 8.0])
    Ps_cm = cm.apply_to_trajectory(Ps)
    assert len(Ps_cm) == 3

    # Turntable
    Ps = get_trajectory()
    tt = Turntable(parameters={
        "axis_offset_lateral": {"value": 1.5, "opt": True},
        "axis_offset_radial": {"value": -2.5, "opt": True},
        "detector_shift_u": {"value": 5.0, "opt": True},
        "detector_shift_v": {"value": -3.0, "opt": True},
        "drift_detector_shift_u": {"value": 2.0, "opt": True},
        "drift_detector_shift_v": {"value": -4.0, "opt": True}
    })
    Ps_tt = tt.apply_to_trajectory(Ps)
    pk_tt = tt.prior_knowledge
    rot_axis = np.array(pk_tt["rotation_axis"])
    rot_axis /= np.linalg.norm(rot_axis)
    
    expected_shifts = [[5.0, -3.0], [6.0, -5.0], [7.0, -7.0]]
    for i, (P, P_new) in enumerate(zip(Ps, Ps_tt)):
        r = P.getPrincipalRay().flatten()
        r /= np.linalg.norm(r)
        u = np.cross(r, rot_axis)
        u /= np.linalg.norm(u)
        v = np.cross(rot_axis, u)

        # 3D axis translation changes camera center: C_new = C - (axis_offset_lateral * u + axis_offset_radial * v)
        C = dehomogenize(P.getCenterOfProjection()).flatten()
        C_new = dehomogenize(P_new.getCenterOfProjection()).flatten()
        np.testing.assert_allclose(C_new - C, -(1.5 * u - 2.5 * v), atol=1e-7)

        # 2D detector translation shifts the principal point (with drift applied)
        pp = P.getPrincipalPoint().flatten()[:2]
        pp_new = P_new.getPrincipalPoint().flatten()[:2]
        np.testing.assert_allclose(pp_new - pp, expected_shifts[i], atol=1e-7)

    # Test Refinement
    Ps = get_trajectory()
    ref = Refinement(parameters={
        "refine_slant": {"value": 0.5, "opt": True},
        "refine_skew": {"value": -0.5, "opt": True},
        "refine_source_x": {"value": 1.0, "opt": True},
        "refine_axial_z": {"value": -1.5, "opt": True}
    })
    Ps_ref = ref.apply_to_trajectory(Ps)
    assert len(Ps_ref) == 3
    for P_new in Ps_ref:
        assert isinstance(P_new, ProjectionMatrix)


def test_thermal_drift_config_loading():
    import json
    import os
    from xray_epipolar_consistency.parameterization import from_dict
    
    config_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(
        config_dir,
        "xray_epipolar_consistency",
        "tools",
        "config",
        "calibration_correction",
        "thermal_drift.json"
    )
    assert os.path.exists(config_path), f"Config file not found at {config_path}"
    
    with open(config_path, "r") as f:
        stage = json.load(f)
        
    assert stage["name"] == "Thermal Drift Calibration"
    assert stage["classname"] == "OptimizerPowell"
    
    param = from_dict(stage["parameterization"])
    assert param.__class__.__name__ == "LinearDrift"
    assert param.num_control_points == 4
    assert param.ref_inst.__class__.__name__ == "DetectorShift"
    assert param.ref_inst.parameters["shift_u"]["opt"] is True
    assert param.ref_inst.parameters["shift_v"]["opt"] is True


