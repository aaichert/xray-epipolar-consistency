#!/usr/bin/env python3
import json
import sys
import os
import time
import gc
import importlib
from pathlib import Path
from copy import deepcopy

import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from tifffile import imread

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from ProjectiveGeometry23.central_projection import ProjectionMatrix
from ProjectiveGeometry23.utils import dehomogenize
from xray_epipolar_consistency import (
    MetricRadonIntermediate,
    RadonIntermediate,
    RadonFilter,
    RadonPostProcess,
    Scan,
    TimeVariant,
    ParameterizationChain
)
from xray_epipolar_consistency.parameterization import from_dict
from fileformats.ompl import save_ompl, load_ompl


class SingleProjectionOptimizationProblem:
    def __init__(self, parameterization, P_original, dtr_i, Ps_ref_subset, dtrs_ref_subset, T_norm, object_radius_mm, num_planes):
        self.parameterization = parameterization
        self.Ps_current = [ProjectionMatrix(P_original.P.copy(), P_original.image_size, P_original.pixel_spacing)]
        self.dtr_i = dtr_i
        self.Ps_ref_subset = Ps_ref_subset
        self.dtrs_ref_subset = dtrs_ref_subset
        self.T_norm = T_norm
        self.object_radius_mm = object_radius_mm
        self.num_planes = num_planes
        
        # Initialize metric with index 0 being the view i, and 1..n being the reference views in R'
        self.metric = MetricRadonIntermediate()
        self.metric.setRadonIntermediates([dtr_i] + dtrs_ref_subset)
        self.metric.setObjectRadius(object_radius_mm)
        self.metric.setEpipolarPlaneNumber(num_planes)
        
        self.iteration = 0
        self.best_cost = float('inf')
        self.best_parameters = None
        self.is_cancelled = False
        
    def cost_function(self, x) -> float:
        if self.is_cancelled:
            raise RuntimeError("Optimization cancelled by user")
        self.parameterization.set_parameter_vector(x)
        
        P_corrected = self.parameterization.apply_to_trajectory(self.Ps_current)[0]
        
        # Align projection matrices using T_norm
        Ps_aligned = [P_corrected.P @ self.T_norm] + [P.P @ self.T_norm for P in self.Ps_ref_subset]
        self.metric.setProjectionMatrices(Ps_aligned)
        
        # Evaluate consistency on the pairs (0, j) for all j in range(1, len(self.Ps_ref_subset) + 1)
        indices = [[0, j, 0, j] for j in range(1, len(self.Ps_ref_subset) + 1)]
        costs = self.metric.evaluate_indices(indices)
        val = float(np.mean(costs))
        
        if val < self.best_cost:
            self.best_cost = val
            self.best_parameters = list(x)
            
        self.iteration += 1
        return val


def compute_single_dtr(img, convert_to_line_integral, I0, max_val, gaussian_sigma, size_t, size_alpha):
    if convert_to_line_integral:
        img = -np.log(np.clip(img / I0, 1e-6, 1.0)) * 20.0

    img = img / max_val * 10
    img = gaussian_filter(img, sigma=gaussian_sigma)

    return RadonIntermediate(
        img.copy(),
        size_alpha,
        size_t,
        int(RadonFilter.Derivative),
        int(RadonPostProcess.Identity),
    )


def main(config_path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    
    # 1. Load Reference Scan (input_data)
    input_data_path = config["input_data"]
    if not os.path.isabs(input_data_path):
        input_data_path = os.path.normpath(os.path.join(config_path.parent, input_data_path))
    scan_path = Path(input_data_path).resolve()
    scan_ref = json.loads(scan_path.read_text())

    scan_dir_ref = scan_path.parent
    data_dir_path_ref = (scan_dir_ref / scan_ref.get("data_dir", "./")).resolve()

    Is_ref = []
    if len(scan_ref["image_files"]) == 1 and scan_ref["image_files"][0].lower().endswith('.nrrd'):
        image_path_ref = (data_dir_path_ref / scan_ref["image_files"][0]).resolve()
        print(f"Loading reference NRRD stack: {image_path_ref}")
        import nrrd
        data, _ = nrrd.read(str(image_path_ref))
        projs = np.transpose(data, (2, 1, 0))
        Is_ref = [projs[i].astype(np.float32) for i in range(projs.shape[0])]
    else:
        for image_name in tqdm(scan_ref["image_files"], desc="Loading reference images..."):
            image_path_ref = (data_dir_path_ref / image_name).resolve()
            if str(image_path_ref).lower().endswith('.nrrd'):
                import nrrd
                img, _ = nrrd.read(str(image_path_ref))
                img = np.squeeze(img)
                if img.ndim == 2:
                    img = img.T
                Is_ref.append(img.astype(np.float32))
            else:
                Is_ref.append(imread(image_path_ref).astype(np.float32))

    H_img_ref, W_img_ref = Is_ref[0].shape
    image_size_ref = (W_img_ref, H_img_ref)
    if "image_size" in scan_ref and scan_ref["image_size"] is not None:
        image_size_ref = tuple(scan_ref["image_size"])

    ompl_path_ref = (scan_dir_ref / scan_ref["ompl_file"]).resolve()
    Ps_ref_loaded = load_ompl(ompl_path_ref)
    pixel_spacing_ref = float(scan_ref.get("pixel_spacing", Ps_ref_loaded[0].pixel_spacing if Ps_ref_loaded else 1.0))
    Ps_ref = [
        ProjectionMatrix(
            P.P if isinstance(P, ProjectionMatrix) else P,
            image_size=image_size_ref,
            pixel_spacing=pixel_spacing_ref,
        )
        for P in Ps_ref_loaded
    ]

    # Initialize reference scan epipolar consistency (GPU Radon transforms)
    print("Initializing reference scan metrics on GPU...")
    reference_scan = Scan(Is_ref, Ps_ref)
    metric_config = config.get("metric_config", {})
    if "convert_to_line_integral" in scan_ref and "convert_to_line_integral" not in metric_config:
        metric_config["convert_to_line_integral"] = scan_ref["convert_to_line_integral"]
    reference_scan.init_epipolar_consistency(**metric_config)

    # 2. Load Full Scan configurations (reconstruction_config)
    input_data_path_full = config["reconstruction_config"]
    if not os.path.isabs(input_data_path_full):
        input_data_path_full = os.path.normpath(os.path.join(config_path.parent, input_data_path_full))
    scan_path_full = Path(input_data_path_full).resolve()
    scan_full = json.loads(scan_path_full.read_text())

    scan_dir_full = scan_path_full.parent
    data_dir_path_full = (scan_dir_full / scan_full.get("data_dir", "./")).resolve()

    output_dir_path = config["output_dir"]
    if not os.path.isabs(output_dir_path):
        output_dir_path = os.path.normpath(os.path.join(config_path.parent, output_dir_path))
    output_dir = Path(output_dir_path).resolve()
    opt_ompl_path = output_dir / "trajectory_optimized.ompl"

    if opt_ompl_path.exists():
        print(f"Loading optimized starting trajectory for refinement: {opt_ompl_path}")
        Ps_all_loaded = load_ompl(opt_ompl_path)
    else:
        ompl_path_full = (scan_dir_full / scan_full["ompl_file"]).resolve()
        Ps_all_loaded = load_ompl(ompl_path_full)
    
    # Read image sizes for full scan
    if "image_size" in scan_full and scan_full["image_size"] is not None:
        image_size_full = tuple(scan_full["image_size"])
    else:
        # Resolve by loading the first image shape
        if len(scan_full["image_files"]) == 1 and scan_full["image_files"][0].lower().endswith('.nrrd'):
            image_path_full = (data_dir_path_full / scan_full["image_files"][0]).resolve()
            import nrrd
            data, _ = nrrd.read(str(image_path_full))
            H_img_full, W_img_full = data.shape[1], data.shape[0]
            image_size_full = (W_img_full, H_img_full)
            del data
        else:
            image_name = scan_full["image_files"][0]
            image_path_full = (data_dir_path_full / image_name).resolve()
            if image_path_full.suffix.lower() == '.nrrd':
                import nrrd
                img, _ = nrrd.read(str(image_path_full))
                img = np.squeeze(img)
                if img.ndim == 2:
                    img = img.T
                H_img_full, W_img_full = img.shape
            else:
                H_img_full, W_img_full = imread(image_path_full).shape
            image_size_full = (W_img_full, H_img_full)

    pixel_spacing_full = float(scan_full.get("pixel_spacing", Ps_all_loaded[0].pixel_spacing if Ps_all_loaded else 1.0))
    Ps_all = [
        ProjectionMatrix(
            P.P if isinstance(P, ProjectionMatrix) else P,
            image_size=image_size_full,
            pixel_spacing=pixel_spacing_full,
        )
        for P in Ps_all_loaded
    ]

    # Pre-load entire projection stack to CPU memory if it's a single NRRD stack, to avoid multiple disk seeks
    full_stack_data = None
    if len(scan_full["image_files"]) == 1 and scan_full["image_files"][0].lower().endswith('.nrrd'):
        image_path_full = (data_dir_path_full / scan_full["image_files"][0]).resolve()
        print(f"Loading full projection NRRD stack to CPU memory: {image_path_full}")
        import nrrd
        data, _ = nrrd.read(str(image_path_full))
        full_stack_data = np.transpose(data, (2, 1, 0))

    # Pre-calculate source positions for all reference views
    Cs_ref = np.array([dehomogenize(P.getCenterOfProjection()).flatten() for P in Ps_ref])

    stages = [
        json.loads((config_path.parent / stage).read_text())
        for stage in config["geometry_optimization"]["stages"]
    ]
    if len(stages) > 1:
        last_stage_name = stages[-1].get("name", f"Stage {len(stages)}")
        print(f"\nWarning: Multiple stages found in config. Only the last stage ('{last_stage_name}') will be executed for refinement.")
    stages = stages[-1:]

    print(f"Starting geometry refinement for {len(Ps_all)} views...")

    t_start = time.time()

    for stage_idx, stage in enumerate(stages):
        stage_name = stage.get("name", f"Stage {stage_idx + 1}")

        # Check that the parameterization has no time-variant classes (fail fast)
        stage_param_ref = from_dict(stage["parameterization"])

        def check_time_variant(p):
            if isinstance(p, ParameterizationChain):
                for sub_p in p.parameterizations:
                    check_time_variant(sub_p)
            elif isinstance(p, TimeVariant):
                raise ValueError(
                    f"Stage '{stage_name}' uses a time-variant parameterization '{p.__class__.__name__}'. "
                    "Time-variant parameterizations are not supported for view-by-view geometry refinement."
                )

        check_time_variant(stage_param_ref)

        # 1. Initialize prior_knowledge once on the reference projections
        stage_param_ref.apply_to_trajectory(Ps_ref)

        # Print the stage details and parameters pretty representation
        print(f"\nRunning Refinement Stage: {stage_name}")
        print("Refinement stage parameterization:")
        print(stage_param_ref.to_str_table(["opt", "name", "value", "range", "description"]))
        print()

        # 2. Iterate through all views sequentially and optimize each in-place
        pbar = tqdm(range(len(Ps_all)), desc=f"{stage_name} Refinement")
        for i in pbar:
            P_i = Ps_all[i]

            # A. Load image i dynamically
            if full_stack_data is not None:
                img_i = full_stack_data[i].astype(np.float32)
            else:
                image_name = scan_full["image_files"][i]
                image_path_full = (data_dir_path_full / image_name).resolve()
                if image_path_full.suffix.lower() == '.nrrd':
                    import nrrd
                    img_i, _ = nrrd.read(str(image_path_full))
                    img_i = np.squeeze(img_i)
                    if img_i.ndim == 2:
                        img_i = img_i.T
                    img_i = img_i.astype(np.float32)
                else:
                    img_i = imread(image_path_full).astype(np.float32)

            # B. Compute DTR for view i
            dtr_i = compute_single_dtr(
                img_i,
                convert_to_line_integral=metric_config.get("convert_to_line_integral", False),
                I0=reference_scan._prev_I0,
                max_val=reference_scan.max_val,
                gaussian_sigma=metric_config.get("gaussian_sigma", 0.0),
                size_t=reference_scan.size_t,
                size_alpha=reference_scan.size_alpha
            )

            # C. Dynamic Reference Filtering: Find and remove closest source position in reference set R
            C_i = dehomogenize(P_i.getCenterOfProjection()).flatten()
            distances = np.linalg.norm(Cs_ref - C_i, axis=1)
            closest_local_idx = np.argmin(distances)
            
            R_prime_local_indices = [r for r in range(len(Ps_ref)) if r != closest_local_idx]

            Ps_ref_subset = [Ps_ref[r] for r in R_prime_local_indices]
            dtrs_ref_subset = [reference_scan.dtrs[r] for r in R_prime_local_indices]

            # D. Copy the initialized stationary parameterization and reset parameter values to 0.0
            param_i = deepcopy(stage_param_ref)
            for p_name in param_i:
                param_i[p_name]["value"] = 0.0

            # E. Setup SingleProjectionOptimizationProblem
            problem_i = SingleProjectionOptimizationProblem(
                param_i,
                P_i,
                dtr_i,
                Ps_ref_subset,
                dtrs_ref_subset,
                reference_scan.T_norm,
                reference_scan.object_radius_mm,
                reference_scan.num_planes
            )

            # F. Run a simple, single-pass optimization using scipy.optimize.minimize directly
            active_names = [name for name, p in param_i.items() if p["opt"]]
            x0 = np.array([param_i[name]["value"] for name in active_names])
            bounds = [param_i[name]["range"] for name in active_names]

            if len(active_names) > 0:
                from scipy.optimize import minimize
                
                # Determine optimization method
                method = 'L-BFGS-B'
                classname = stage.get("classname")
                if classname == "OptimizerPowell":
                    method = 'Powell'
                    
                kwargs = stage.get("kwargs", {})
                opt_options = kwargs.get("options", {})
                maxiter = opt_options.get("maxiter", 50)
                ftol = opt_options.get("ftol", 1e-5)

                res = minimize(
                    problem_i.cost_function,
                    x0,
                    method=method,
                    bounds=bounds,
                    options={"maxiter": maxiter, "ftol": ftol}
                )
                optimized_vector = res.x
            else:
                optimized_vector = x0

            # H. Save optimized projection matrix to Ps_all[i]
            param_i.set_parameter_vector(optimized_vector)
            Ps_all[i] = param_i.apply_to_trajectory([Ps_all[i]])[0]

            # Update progress bar description with best cost
            pbar.set_description(f"View {i:3d} (ECC: {problem_i.best_cost:.4e})")

            # I. Memory Cleanup: Delete large arrays/DTRs immediately and collect garbage
            del img_i
            del dtr_i
            del problem_i
            gc.collect()

    print(f"\nRefinement optimization completed in {time.time() - t_start:.2f} seconds.")

    # 3. Save Output Directory & Files
    output_dir_path = config["output_dir"]
    if not os.path.isabs(output_dir_path):
        output_dir_path = os.path.normpath(os.path.join(config_path.parent, output_dir_path))
    output_dir = Path(output_dir_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectory_refined_path = output_dir / "trajectory_refined.ompl"
    save_ompl(
        Ps_all,
        trajectory_refined_path,
        first_line_comment="Refined via epipolar consistency",
        spacing=pixel_spacing_full,
        detector_size_px=image_size_full
    )
    print(f"Saved refined trajectory to:\n{trajectory_refined_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <config.json>")
        sys.exit(1)
    main(sys.argv[1])
