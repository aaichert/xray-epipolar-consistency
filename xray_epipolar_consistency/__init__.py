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
    base = _pkg_resources.files("xray_epipolar_consistency").joinpath(*relative_parts)

    if not isinstance(base, _Path):
        path_str = "/".join(relative_parts)
        print(f"Extracting package data: {path_str} (this may take a few seconds on first run)...")

    with _pkg_resources.as_file(base) as base_fs:
        result = _Path(base_fs).resolve()

    if not result.exists():
        path_str = "/".join(relative_parts)
        raise FileNotFoundError(
            f"Package data not found: {path_str}"
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
