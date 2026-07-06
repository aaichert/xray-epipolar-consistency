from ._core import (
    RadonIntermediate,
    MetricRadonIntermediate,
    RadonFilter,
    RadonPostProcess,
    compute_for_image_pair,
    computeK01,
    lineToSampleDtr
)

from .scan import Scan
from .volume_renderer import VolumeRenderer
from .parameterization import (
    ParameterizationBase,
    ParameterizationChain,
    DetectorShift,
    DetectorOrientation,
    ObjectPose,
    RotationAxis,
    Distance,
    GantryAngle,
    TimeVariant,
    LinearDrift,
    ContinuousMotion
)
from .optimizer import (
    OptimizationProblem,
    Optimizer,
    OptimizerLBFGS,
    OptimizerPowell
)
from .geometry_correction import CalibrationAndMotionCorrection
from .progress import ProgressBar

import importlib.resources as _pkg_resources
from pathlib import Path as _Path

def get_data_path(*relative_parts: str) -> _Path:
    """Return the absolute path to a file inside the installed package data.
    """
    base = _pkg_resources.files("xray_epipolar_consistency")

    with _pkg_resources.as_file(base) as base_fs:
        base_fs = _Path(base_fs).resolve()

    result = base_fs.joinpath(*relative_parts)

    if not result.exists():
        raise FileNotFoundError(
            f"Package data not found: {'/'.join(relative_parts)}"
        )

    return result

try:
    EXAMPLE_DATA_PATH: _Path | None = get_data_path("example_data")
except FileNotFoundError:
    EXAMPLE_DATA_PATH = None

try:
    TOOLS_CONFIG_PATH: _Path | None = get_data_path("tools", "config")
except FileNotFoundError:
    TOOLS_CONFIG_PATH = None


__all__ = [
    "RadonIntermediate",
    "MetricRadonIntermediate",
    "RadonFilter",
    "RadonPostProcess",
    "compute_for_image_pair",
    "computeK01",
    "lineToSampleDtr",
    "Scan",
    "VolumeRenderer",
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
    "OptimizationProblem",
    "Optimizer",
    "OptimizerLBFGS",
    "OptimizerPowell",
    "CalibrationAndMotionCorrection",
    "get_data_path",
    "EXAMPLE_DATA_PATH",
    "TOOLS_CONFIG_PATH",
    "ProgressBar",
]
