from abc import ABC, abstractmethod
from typing import Sequence
from scipy.optimize import minimize

class OptimizationProblem:
    """
    OptimizationProblem bridges a Scan object and a Parameterization object
    to evaluate the epipolar consistency cost for a parameter vector.
    """
    def __init__(self, scan, parameterization, status_callback=None):
        self.scan = scan
        self.parameterization = parameterization
        self.image_sizes = [P.image_size.copy() for P in self.scan.Ps]
        self.pixel_spacings = [P.pixel_spacing for P in self.scan.Ps]
        self.Ps_original = [P.P.copy() for P in self.scan.Ps]
        self.status_callback = status_callback
        self.iteration = 0
        self.cost_function_values = []
        self.best_cost = float('inf')
        self.best_parameters = None
        self.is_cancelled = False
        
    def cost_function(self, x: Sequence[float]) -> float:
        if self.is_cancelled:
            raise RuntimeError("Optimization cancelled by user")
        self.parameterization.set_parameter_vector(x)
        from ProjectiveGeometry23.central_projection import ProjectionMatrix
        Ps_pm = [
            ProjectionMatrix(P_arr.copy(), image_size, pixel_spacing)
            for P_arr, image_size, pixel_spacing in zip(self.Ps_original, self.image_sizes, self.pixel_spacings)
        ]
        Ps_corrected = self.parameterization.apply_to_trajectory(Ps_pm)
        self.scan.set_projection_matrices(Ps_corrected)
        val, cost_matrix = self.scan.compute_epipolar_consistency()
        self.cost_function_values += [val]
        if val < self.best_cost:
            self.best_cost = val
            self.best_parameters = list(x)
        if self.status_callback is not None:
            self.status_callback(self.iteration, self.parameterization, val, cost_matrix)
        self.iteration = self.iteration + 1
        return val


class Optimizer(ABC):
    """
    Abstract base class for all trajectory calibration optimizers.
    """
    @abstractmethod
    def optimize(self, problem: OptimizationProblem) -> list[float]:
        pass


class OptimizerLBFGS(Optimizer):
    """
    L-BFGS-B optimization algorithm implementation.
    """
    def __init__(self, **kwargs):
        self.options = kwargs.get("options", {"maxiter": 200, "ftol": 1e-12, "gtol": 1e-12})

    def optimize(self, problem: OptimizationProblem) -> list[float]:
        import numpy as np

        # Track originally active parameters for this optimizer run
        original_active = [name for name, p in problem.parameterization.items() if p["opt"]]
        if not original_active:
            return []

        # We will run up to 2 passes
        for pass_idx in range(2):
            print(f"--- L-BFGS-B Pass {pass_idx + 1} ---")
            
            # Re-enable all parameters that were originally active
            for name in original_active:
                problem.parameterization[name]["opt"] = True
                if pass_idx > 0:
                    problem.parameterization[name]["value"] = 0.0

            # Store the starting values for this pass
            pass_start_values = {name: problem.parameterization[name]["value"] for name in original_active}

            # Estimate prior knowledge on the current Ps_original
            problem.parameterization.prior_knowledge = None
            from ProjectiveGeometry23.central_projection import ProjectionMatrix
            Ps_pm = [
                ProjectionMatrix(P_arr.copy(), image_size, pixel_spacing)
                for P_arr, image_size, pixel_spacing in zip(problem.Ps_original, problem.image_sizes, problem.pixel_spacings)
            ]
            problem.parameterization.estimateTrajectoryParameters(Ps_pm)

            out_of_bounds_occurred = False

            # Inner repeat loop for the current pass
            while True:
                active_names = [name for name in original_active if problem.parameterization[name]["opt"]]
                if not active_names:
                    print("No parameters left to optimize in this pass.")
                    break
                
                # Reset remaining active parameters to their starting values for this pass
                for name in active_names:
                    problem.parameterization[name]["value"] = pass_start_values[name]
                
                x0 = np.array([problem.parameterization[name]["value"] for name in active_names])
                bounds = [problem.parameterization[name]["range"] for name in active_names]
                
                print(f"Optimizing parameters: {active_names}")
                res = minimize(
                    problem.cost_function,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options=self.options
                )
                
                # Check which parameters are out of range (at boundaries)
                out_of_bounds = []
                for name, val in zip(active_names, res.x):
                    p = problem.parameterization[name]
                    range_min, range_max = p["range"]
                    # Tolerance to check if parameter is at bounds
                    tol = 1e-6 * (range_max - range_min) if range_max > range_min else 1e-6
                    if val <= range_min + tol or val >= range_max - tol:
                        out_of_bounds.append(name)
                
                if out_of_bounds:
                    print(f"Parameters out of range: {out_of_bounds}. Disabling and repeating optimization.")
                    out_of_bounds_occurred = True
                    for name in out_of_bounds:
                        p = problem.parameterization[name]
                        p["opt"] = False
                        p["value"] = 0.0  # Reset to 0.0 since it is disabled
                    # Repeat loop with remaining parameters
                    continue
                else:
                    # Succeeded! Update active parameters with optimized values
                    for name, val in zip(active_names, res.x):
                        problem.parameterization[name]["value"] = val
                    print("Pass optimization succeeded.")
                    break
            
            # If all parameters got disabled, we cannot proceed to next pass
            active_at_end_of_pass = [name for name in original_active if problem.parameterization[name]["opt"]]
            if not active_at_end_of_pass:
                print("All parameters failed to optimize in this pass.")
                break

            # If no parameters went out of bounds in this pass, and it's the first pass,
            # we don't need to run a second pass.
            if not out_of_bounds_occurred and pass_idx == 0:
                print("No parameters went out of range. Skipping second pass.")
                break

            # If this is the first pass and we are going to run a second pass:
            if pass_idx == 0:
                # Update baseline trajectory using the successfully optimized parameters of Pass 1
                from ProjectiveGeometry23.central_projection import ProjectionMatrix
                Ps_pm = [
                    ProjectionMatrix(P_arr.copy(), image_size, pixel_spacing)
                    for P_arr, image_size, pixel_spacing in zip(problem.Ps_original, problem.image_sizes, problem.pixel_spacings)
                ]
                Ps_new_base = problem.parameterization.apply_to_trajectory(Ps_pm)
                problem.Ps_original = [P.P.copy() for P in Ps_new_base]
                # Reset all parameters to 0.0 before re-enabling them for the second pass
                for name in original_active:
                    problem.parameterization[name]["value"] = 0.0

        # After the second pass, print warnings for parameters that could not be optimized
        failed_parameters = [name for name in original_active if not problem.parameterization[name]["opt"]]
        if failed_parameters:
            print("\n==============================================================")
            print("Warning: The following parameters could not be optimized (went out of bounds):")
            for name in sorted(failed_parameters):
                print(f"  - {name}")
            print("==============================================================\n")

        # Return list of values for the active parameters at the end
        return list(problem.parameterization.get_parameter_vector())


class OptimizerPowell(Optimizer):
    """
    Powell optimization algorithm implementation.
    """
    def __init__(self, **kwargs):
        self.options = kwargs.get("options", {"maxiter": 200, "ftol": 1e-12})

    def optimize(self, problem: OptimizationProblem) -> list[float]:
        x0 = problem.parameterization.get_parameter_vector()
        bounds = [p["range"] for p in problem.parameterization.values() if p["opt"]]
        res = minimize(
            problem.cost_function,
            x0,
            method='Powell',
            bounds=bounds,
            options=self.options
        )
        return list(res.x)

