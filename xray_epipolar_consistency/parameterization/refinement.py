from collections import OrderedDict
import numpy as np
from xray_epipolar_consistency.parameterization.base import ParameterizationBase
from ProjectiveGeometry23.central_projection import ProjectionMatrix

class Refinement(ParameterizationBase):
    """
    Refinement parameterization for coupled parameters.
    Allows optimizing out-of-plane tilts and source shifts
    while cancelling out the dominant first-order image shifts.
    """
    PARAMETERS = OrderedDict({
        "refine_slant": {
            "description": "Coupled detector slant & lateral shift (U) [degrees]",
            "value": 0.0,
            "range": (-2.0, 2.0),
            "opt": True,
        },
        "refine_skew": {
            "description": "Coupled detector skew & vertical shift (V) [degrees]",
            "value": 0.0,
            "range": (-2.0, 2.0),
            "opt": True,
        },
        "refine_source_x": {
            "description": "Coupled source shift & detector shift (U) [mm]",
            "value": 0.0,
            "range": (-3.0, 3.0),
            "opt": True,
        },
        "refine_axial_z": {
            "description": "Coupled object translation & detector shift (V) [mm]",
            "value": 0.0,
            "range": (-2.0, 2.0),
            "opt": True,
        },
    })

    def apply_stationary(self, P: ProjectionMatrix) -> ProjectionMatrix:
        if self.prior_knowledge is None:
            raise RuntimeError("Call estimateTrajectoryParameters() first.")

        # Parameters
        slant = np.radians(self["refine_slant"]["value"])
        skew = np.radians(self["refine_skew"]["value"])
        src_x = self["refine_source_x"]["value"]
        axial_z = self["refine_axial_z"]["value"]

        # Constants from geometry
        sid = self.prior_knowledge["sid"]
        sdd = self.prior_knowledge["sdd"]
        f = P.getFocalLengthPx()
        
        if isinstance(P.pixel_spacing, (list, tuple, np.ndarray)):
            pu, pv = P.pixel_spacing[0], P.pixel_spacing[1]
        else:
            pu = pv = P.pixel_spacing
        mag = sdd / sid

        # Calculate compensating shifts
        # Slant/Skew compensation
        comp_slant_u = -slant * f
        comp_skew_v = -skew * f

        # Source shift compensation
        comp_src_u = -src_x * (sdd - sid) / (sid * pu)

        # Axial translation compensation
        comp_axial_v = axial_z * mag / pv

        # Combined detector shifts
        total_shift_u = comp_slant_u + comp_src_u
        total_shift_v = comp_skew_v + comp_axial_v

        # Import physical modules
        from xray_epipolar_consistency.parameterization.detector_orientation import DetectorOrientation
        from xray_epipolar_consistency.parameterization.detector_shift import DetectorShift
        from xray_epipolar_consistency.parameterization.source_shift_agc import SourceShiftAGC
        from xray_epipolar_consistency.parameterization.object_pose import ObjectPose

        # Apply transformations sequentially to ProjectionMatrix
        # 1. Object Pose (axial translation)
        obj_pose = ObjectPose(
            parameters={
                "translation_x": {"value": 0.0, "opt": False},
                "translation_y": {"value": 0.0, "opt": False},
                "translation_z": {"value": -axial_z, "opt": False},
            },
            prior_knowledge=self.prior_knowledge
        )
        P = obj_pose.apply_stationary(P)

        # 2. Source Shift
        src_shift = SourceShiftAGC(
            parameters={
                "source_shift_x": {"value": src_x, "opt": False},
                "source_shift_y": {"value": 0.0, "opt": False},
                "source_shift_z": {"value": 0.0, "opt": False},
            },
            prior_knowledge=self.prior_knowledge
        )
        P = src_shift.apply_stationary(P)

        # 3. Detector Orientation (slant / skew)
        det_orient = DetectorOrientation(
            parameters={
                "tilt_roll": {"value": 0.0, "opt": False},
                "slant_yaw": {"value": np.degrees(slant), "opt": False},
                "skew_pitch": {"value": np.degrees(skew), "opt": False},
            },
            prior_knowledge=self.prior_knowledge
        )
        P = det_orient.apply_stationary(P)

        # 4. Detector Shifts (sum of all compensations)
        det_shift = DetectorShift(
            parameters={
                "shift_u": {"value": total_shift_u, "opt": False},
                "shift_v": {"value": total_shift_v, "opt": False},
            },
            prior_knowledge=self.prior_knowledge
        )
        P = det_shift.apply_stationary(P)

        return P
