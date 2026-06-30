# Calibration and Motion Correction

The `CalibrationAndMotionCorrection` class orchestrates the optimization of the projection matrix geometry to maximize epipolar consistency.

## Overview of Operation

1. **Initialization**:
   - Takes a CT scan trajectory (projections and projection matrices) and a set of optimization stages.
   - Computes a one-time global trajectory **prior knowledge** (`iso_center` and `rotation_axis`) from the original projection matrices.
   - Initializes all stage parameterizations using this common prior knowledge.

2. **Optimization Stages**:
   - Runs the stages sequentially.
   - For each stage, it instantiates the configured `Optimizer` (e.g., Powell) and sets up the `OptimizationProblem`.
   - The optimizer iteratively adjusts the active parameter vector of the current stage's parameterization to minimize the Epipolar Consistency Cost (ECC) computed by `Scan`.
   - Updates the trajectory projection matrices in the `Scan` object at the end of each stage.

3. **Output Generation**:
   - Generates optimization cost histories, parameter state summaries, and ANSI-colored visual diagnostic matrices.
   - Saves final optimized trajectories and configurations.
