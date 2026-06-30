import importlib
from xray_epipolar_consistency.parameterization.base import ParameterizationBase, ParameterizationChain
from xray_epipolar_consistency.parameterization.detector_shift import DetectorShift
from xray_epipolar_consistency.parameterization.detector_orientation import DetectorOrientation
from xray_epipolar_consistency.parameterization.object_pose import ObjectPose
from xray_epipolar_consistency.parameterization.rotation_axis import RotationAxis
from xray_epipolar_consistency.parameterization.distance import Distance
from xray_epipolar_consistency.parameterization.gantry_angle import GantryAngle
from xray_epipolar_consistency.parameterization.time_variant import TimeVariant, LinearDrift, ContinuousMotion
from xray_epipolar_consistency.parameterization.turntable import Turntable
from xray_epipolar_consistency.parameterization.source_shift_agc import SourceShiftAGC

__all__ = [
    "ParameterizationBase",
    "ParameterizationChain",
    "DetectorShift",
    "DetectorOrientation",
    "ObjectPose",
    "RotationAxis",
    "Distance",
    "GantryAngle",
    "TimeVariant",
    "LinearDrift",
    "ContinuousMotion",
    "Turntable",
    "SourceShiftAGC",
    "from_dict"
]

def from_dict(d: dict) -> ParameterizationBase:
    module_name = d.get("module", "xray_epipolar_consistency.parameterization")
    class_name = d.get("classname")
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls.from_dict(d)
