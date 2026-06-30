import numpy as np
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
    Turntable
)

def get_test_trajectory():
    P1 = ProjectionMatrix.perspective_look_at(eye=np.array([1000.0, 50.0, 10.0]), center=np.array([5.0, -5.0, 10.0]))
    P2 = ProjectionMatrix.perspective_look_at(eye=np.array([707.1, 50.0, 707.1]), center=np.array([5.0, -5.0, 10.0]))
    P3 = ProjectionMatrix.perspective_look_at(eye=np.array([-10.0, 50.0, 1000.0]), center=np.array([5.0, -5.0, 10.0]))
    return [P1, P2, P3]

def get_perfect_trajectory():
    # A perfect trajectory where cameras look exactly at the origin and are placed symmetrically, avoiding collinearity with the up vector [0, 1, 0]
    P1 = ProjectionMatrix.perspective_look_at(eye=np.array([1000.0, 0.0, 0.0]), center=np.array([0.0, 0.0, 0.0]))
    P2 = ProjectionMatrix.perspective_look_at(eye=np.array([0.0, 0.0, 1000.0]), center=np.array([0.0, 0.0, 0.0]))
    P3 = ProjectionMatrix.perspective_look_at(eye=np.array([707.1, 0.0, 707.1]), center=np.array([0.0, 0.0, 0.0]))
    return [P1, P2, P3]

def test_optimizations_equivalence_nonzero():
    Ps = get_test_trajectory()
    
    # 1. DetectorShift
    ds = DetectorShift(parameters={
        "shift_u": {"value": 2.5},
        "shift_v": {"value": -3.8}
    })
    ds.estimateTrajectoryParameters(Ps)
    for P in Ps:
        P_opt = ds.apply_stationary(P)
        P_ref = ds.apply_stationary_reference_impl(P)
        np.testing.assert_allclose(P_opt.P, P_ref.P, rtol=1e-12, atol=1e-12)

    # 2. DetectorOrientation
    do = DetectorOrientation(parameters={
        "tilt_roll": {"value": 1.2},
        "slant_yaw": {"value": -0.8},
        "skew_pitch": {"value": 0.4}
    })
    do.estimateTrajectoryParameters(Ps)
    for P in Ps:
        P_opt = do.apply_stationary(P)
        P_ref = do.apply_stationary_reference_impl(P)
        np.testing.assert_allclose(P_opt.P, P_ref.P, rtol=1e-12, atol=1e-12)

    # 3. Distance
    dist = Distance(parameters={
        "delta_sdd": {"value": 12.0},
        "delta_sid": {"value": -8.5},
        "coupled_distance": {"value": 3.0}
    })
    dist.estimateTrajectoryParameters(Ps)
    for P in Ps:
        P_opt = dist.apply_stationary(P)
        P_ref = dist.apply_stationary_reference_impl(P)
        np.testing.assert_allclose(P_opt.P, P_ref.P, rtol=1e-12, atol=1e-12)

    # 4. GantryAngle
    ga = GantryAngle(parameters={
        "primary_angle": {"value": 1.5},
        "secondary_angle": {"value": -2.2}
    })
    ga.estimateTrajectoryParameters(Ps)
    for P in Ps:
        P_opt = ga.apply_stationary(P)
        P_ref = ga.apply_stationary_reference_impl(P)
        np.testing.assert_allclose(P_opt.P, P_ref.P, rtol=1e-12, atol=1e-12)

    # 5. ObjectPose
    op = ObjectPose(parameters={
        "translation_x": {"value": 1.5},
        "translation_y": {"value": -2.0},
        "translation_z": {"value": 0.8},
        "rotation_x": {"value": 1.1},
        "rotation_y": {"value": -0.9},
        "rotation_z": {"value": 1.6}
    })
    op.estimateTrajectoryParameters(Ps)
    for P in Ps:
        P_opt = op.apply_stationary(P)
        P_ref = op.apply_stationary_reference_impl(P)
        np.testing.assert_allclose(P_opt.P, P_ref.P, rtol=1e-10, atol=1e-10)

    # 6. RotationAxis
    ra = RotationAxis(parameters={
        "tilt_pitch": {"value": 0.5},
        "tilt_roll": {"value": -0.6},
        "offset_lateral": {"value": 1.2},
        "offset_radial": {"value": -1.8}
    })
    ra.estimateTrajectoryParameters(Ps)
    for P in Ps:
        P_opt = ra.apply_stationary(P)
        P_ref = ra.apply_stationary_reference_impl(P)
        np.testing.assert_allclose(P_opt.P, P_ref.P, rtol=1e-12, atol=1e-12)

    # 7. Turntable
    tt = Turntable(parameters={
        "axis_offset_lateral": {"value": 1.1},
        "axis_offset_radial": {"value": -1.5},
        "axis_tilt_pitch": {"value": 0.4},
        "axis_tilt_roll": {"value": -0.3},
        "detector_shift_u": {"value": 2.5},
        "detector_shift_v": {"value": -3.1},
        "delta_sdd": {"value": 10.0},
        "delta_sid": {"value": -6.0},
        "detector_roll": {"value": 0.8},
        "detector_pitch": {"value": -0.7},
        "detector_yaw": {"value": 0.5},
        "drift_detector_shift_u": {"value": 0.5},
        "drift_detector_shift_v": {"value": -0.4},
        "drift_sdd": {"value": 1.5},
        "drift_sid": {"value": -1.0}
    })
    Ps_opt = tt.apply_to_trajectory(Ps)
    Ps_ref = tt.apply_to_trajectory_reference_impl(Ps)
    for P_opt, P_ref in zip(Ps_opt, Ps_ref):
        np.testing.assert_allclose(P_opt.P, P_ref.P, rtol=1e-12, atol=1e-12)


def test_optimizations_equivalence_zeros():
    Ps = get_test_trajectory()
    
    classes_to_test = [
        DetectorShift,
        DetectorOrientation,
        Distance,
        GantryAngle,
        ObjectPose,
        RotationAxis,
        Turntable
    ]

    for cls in classes_to_test:
        inst = cls()
        inst.estimateTrajectoryParameters(Ps)
        for P in Ps:
            P_opt = inst.apply_stationary(P)
            P_ref = inst.apply_stationary_reference_impl(P)
            np.testing.assert_allclose(P_opt.P, P_ref.P, rtol=1e-12, atol=1e-12)


def test_distance_coupled_magnification():
    Ps = get_perfect_trajectory()
    
    # Check magnification constancy under coupled_distance change
    dist = Distance(parameters={
        "delta_sdd": {"value": 0.0},
        "delta_sid": {"value": 0.0},
        "coupled_distance": {"value": 50.0} # significant shift
    })
    dist.estimateTrajectoryParameters(Ps)
    
    # Original magnification
    sdd_orig = dist.prior_knowledge["sdd"]
    sid_orig = dist.prior_knowledge["sid"]
    mag_orig = sdd_orig / sid_orig

    Ps_new = dist.apply_to_trajectory(Ps)
    
    for P_orig, P_new in zip(Ps, Ps_new):
        geom_orig = SourceDetectorGeometry(P_orig)
        geom_new = SourceDetectorGeometry(P_new)
        
        # SDD and SID of the new projection matrix
        sdd_new = geom_new.source_detector_distance
        
        iso = np.array(dist.prior_knowledge["iso_center"])
        C_new = dehomogenize(P_new.getCenterOfProjection()).flatten()
        # SID is the distance from source (camera center) to isocenter
        sid_new = np.linalg.norm(C_new - iso)
        
        mag_new = sdd_new / sid_new
        
        # Verify magnification remains constant
        np.testing.assert_allclose(mag_new, mag_orig, rtol=1e-5)
