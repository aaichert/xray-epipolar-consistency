import rich
import builtins
import importlib
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from rich import print

from ProjectiveGeometry23.central_projection import ProjectionMatrix
from xray_epipolar_consistency.optimizer import OptimizationProblem
from xray_epipolar_consistency.parameterization import from_dict
from xray_epipolar_consistency.scan import Scan

class CalibrationAndMotionCorrection:
    """
    CalibrationAndMotionCorrection orchestrates the mathematical optimization of epipolar consistency metric.
    Runs Optimizer over Parameterization to minimize the GPU cost function defined in Scan.
    """

    def __init__(self, Is, Ps, stages, metric_config=None):
        self.stages = stages
        self.scan = Scan(Is, Ps)
        self.image_sizes = [P.image_size.copy() for P in self.scan.get_projection_matrices()]
        self.pixel_spacings = [P.pixel_spacing for P in self.scan.get_projection_matrices()]
        self.Ps_original = [P.P.copy() for P in self.scan.get_projection_matrices()]

        self.ecc_config = self.scan.init_epipolar_consistency(**metric_config)

        self.parameterizations = []
        for stage in self.stages:
            param = from_dict(stage["parameterization"])
            Ps_pm = [
                ProjectionMatrix(P_arr.copy(), image_size, pixel_spacing)
                for P_arr, image_size, pixel_spacing in zip(self.Ps_original, self.image_sizes, self.pixel_spacings)
            ]
            param.estimateTrajectoryParameters(Ps_pm)
            self.parameterizations.append(param)

        cost, self.initial_cost_matrix = self.scan.compute_epipolar_consistency()
        self.initial_cost_matrix = self.initial_cost_matrix.transpose()
        np.fill_diagonal(self.initial_cost_matrix, -1)

        print("ECC configuration:")
        np.set_printoptions(precision=2,suppress=True,floatmode="fixed")
        print(self.ecc_config)

        run_diagnostics = False
        if run_diagnostics:
            self.diagnostics = self.scan.compute_diagnostics()

            if self.initial_cost_matrix.shape[0] <= 80:
                print("\nWeights between 0 (black) and 1 (white) in bottom left diagonal matrixs")
                print("and number epipolar planes sampled in top right diagonal matrix.")
                diagnostics_matrix = self.diagnostics["weight_matrix"].transpose().copy()
                diagnostics_matrix *= np.max(self.diagnostics["sample_count_matrix"][:])
                diagnostics_matrix += self.diagnostics["sample_count_matrix"]
                builtins.print(self._matrix_to_ansi(diagnostics_matrix))

            print("\nDiagnostic result:")
            if "Warning" in self.diagnostics["info"]:
                print(f"[bold red]{self.diagnostics['info']}[/bold red]\n")
            else:
                print(f"[bold green]{self.diagnostics['info']}[/bold green]\n")

        if self.initial_cost_matrix.shape[0] <= 80:
            print("\nInitial cost matrix:")
            builtins.print(self._matrix_to_ansi(self.initial_cost_matrix))
        print(f"\nTest evaluation of cost function: {cost:.6f}")

    @staticmethod
    def _matrix_to_ansi(M):
        M = np.asarray(M)
        finite = np.isfinite(M) & (M >= 0)
        vmax = np.max(M[finite]) if np.any(finite) else 1.0
        vmax = max(vmax, 1e-12)

        def swatch(r, g, b):
            return f"\033[48;2;{r};{g};{b}m  \033[0m"

        try:
            import climage
            H, W = M.shape
            arr = np.zeros((H, W, 3), dtype=np.uint8)
            scaled = np.clip(255.0 * M / vmax, 0, 255).astype(np.uint8)
            for c in range(3):
                arr[..., c] = np.where(finite, scaled, 0)
            arr[..., 0] = np.where(finite, arr[..., 0], 255)
            
            legend = f"Legend: {swatch(0,0,0)}=0, {swatch(255,255,255)}={vmax:.0f} (max), {swatch(255,0,0)}=invalid"
            return legend + "\n" + climage.convert_array(arr, is_unicode=True)
        except ImportError:
            lines = [
                f"Legend: {swatch(0,0,0)}=0, "
                f"{swatch(255,255,255)}={vmax:.0f} (max), "
                f"{swatch(255,0,0)}=invalid"
            ]

            for row in M:
                lines.append(
                    "".join(
                        swatch(255, 0, 0)
                        if not np.isfinite(x) or x < 0
                        else swatch(*(3 * [int(255 * x / vmax)]))
                        for x in row
                    )
                )

            return "\n".join(lines)

    def optimize(self):
        """
        Runs the optimization stages and returns a dictionary of results.
        """
        t_start_opt = time.time()
        cost_history = []
        stage_results = []
        self.iteration_cost_history = []
        
        # Compute initial cost
        initial_cost, _ = self.scan.compute_epipolar_consistency()
        cost_history.append(initial_cost)

        for stage_idx, stage in enumerate(self.stages):
            Ps_initial = [P.P.copy() for P in self.scan.get_projection_matrices()]
            stage_name = stage.get("name", f"Stage {stage_idx + 1}")
            stage_parameterization = self.parameterizations[stage_idx]

            module_name = stage.get("module", "xray_epipolar_consistency.optimizer")
            classname = stage.get("classname")
            kwargs = stage.get("kwargs", {})

            opt_module = importlib.import_module(module_name)
            opt_class = getattr(opt_module, classname)
            optimizer = opt_class(**kwargs)

            

            print("Optimizer: " + classname + " with parameters:")
            for name, doc in stage_parameterization.get_names_and_docstr().items():
                print(f"    {name}: \t{doc}")
            rich.print(kwargs)
            print()
            
            def info_callback(iteration, parameterization, cost, cost_matrix):
                params = ", ".join(f"{x:.2f}" for x in parameterization.get_parameter_vector())
                msg = f"{stage_name} - Iteration {iteration}: ECC = {cost:.6g} [{params}]"
                sys.stdout.write("\r\033[K" + msg)
                sys.stdout.flush()
            problem = OptimizationProblem(self.scan, stage_parameterization, info_callback)
            optimized_vector = optimizer.optimize(problem)
            print()

            stage_parameterization.set_parameter_vector(optimized_vector)
            self.iteration_cost_history.extend(problem.cost_function_values)

            Ps_initial_pm = [
                ProjectionMatrix(P_arr.copy(), image_size, pixel_spacing)
                for P_arr, image_size, pixel_spacing in zip(Ps_initial, self.image_sizes, self.pixel_spacings)
            ]
            Ps_optimized = stage_parameterization.apply_to_trajectory(Ps_initial_pm)
            self.scan.set_projection_matrices(Ps_optimized)  # result will be used in next stage
            self.parameterization = stage_parameterization
                
            stage_cost, stage_cost_matrix = self.scan.compute_epipolar_consistency()
            cost_history.append(stage_cost)
            print(f"{stage_name} completed. Cost: {stage_cost:.6f}")

            stage_results.append({
                "name": stage_name,
                "optimizer": classname,
                "final_cost": stage_cost,
                "parameter_vector": list(optimized_vector),
                "parameters": {k: stage_parameterization[k]["value"] for k in stage_parameterization},
            })
            
        opt_time = time.time() - t_start_opt
        
        return {
            "optimized_parameterization": self.parameterization.to_dict(),
            "stages": stage_results,
            "cost_history": cost_history,
            "optimization_time_sec": opt_time,
            "Ps_optimized": [P.P.tolist() for P in Ps_optimized]
        }
