#!/usr/bin/env python3
import json
import sys
import os
from pathlib import Path

import numpy as np
from tifffile import imread
from ProjectiveGeometry23.central_projection import ProjectionMatrix

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from xray_epipolar_consistency import CalibrationAndMotionCorrection
from xray_epipolar_consistency.parameterization import from_dict, ParameterizationChain

from tqdm import tqdm
from fileformats.ompl import save_ompl, load_ompl


def safe_relpath(path, start=None):
    if not path:
        return ""
    path_abs = os.path.abspath(path)
    start_abs = os.path.abspath(start) if start else os.getcwd()
    try:
        rel = os.path.relpath(path_abs, start_abs)
        # Count parent directory traversals
        rel_norm = rel.replace('\\', '/')
        parts = rel_norm.split('/')
        if parts.count('..') > 3:
            return path_abs
        return rel
    except ValueError:
        return path_abs


def align_trajectories(Ps_opt, Ps_init):
    """
    Finds a 3D similarity transformation T_align (rotation, translation, scale) such that:
       T_align @ X_opt approx X_init
    and returns aligned optimized projection matrices:
       P_opt_aligned_i = P_opt_i @ T_align_inv
    """
    from ct_recon_fdk_astra.recon_coverage import compute_similarity_transform
    
    # We need to extract the raw 3x4 matrices as numpy arrays for compute_similarity_transform
    mats_ref = [P.P if hasattr(P, "P") else P for P in Ps_init]
    mats_opt = [P.P if hasattr(P, "P") else P for P in Ps_opt]
    
    # Let's get the image size (detector size) from the first matrix
    detector_size = Ps_init[0].image_size if hasattr(Ps_init[0], "image_size") else (600, 400)
    
    T_align = compute_similarity_transform(mats_ref, mats_opt, detector_size)
    T_align_inv = np.linalg.inv(T_align)
    
    Ps_aligned = []
    for P_opt in Ps_opt:
        P_aligned = ProjectionMatrix(
            P_opt.P @ T_align_inv,
            image_size=P_opt.image_size if hasattr(P_opt, "image_size") else (600, 400),
            pixel_spacing=P_opt.pixel_spacing if hasattr(P_opt, "pixel_spacing") else 1.0
        )
        Ps_aligned.append(P_aligned)
        
    return Ps_aligned, T_align


def main(config_path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    
    input_data_path = config["input_data"]
    if not os.path.isabs(input_data_path):
        input_data_path = os.path.normpath(os.path.join(config_path.parent, input_data_path))
    scan_path = Path(input_data_path).resolve()
    scan = json.loads(scan_path.read_text())

    # Copy voxel_dimensions, model_matrix, filter_type, and output_file from input_data config if not present
    if "reconstruction_config" in config:
        for key in ["voxel_dimensions", "model_matrix", "filter_type", "output_file"]:
            if key in scan and key not in config:
                config[key] = scan[key]

    if "convert_to_line_integral" in scan and "convert_to_line_integral" not in config["metric_config"]:
        config["metric_config"]["convert_to_line_integral"] = scan["convert_to_line_integral"]

    scan_dir = scan_path.parent
    data_dir_path = (scan_dir / scan.get("data_dir", "./")).resolve()

    images = []
    # Check if we have a single file containing a stack of projections
    if len(scan["image_files"]) == 1 and scan["image_files"][0].lower().endswith('.nrrd'):
        image_path = (data_dir_path / scan["image_files"][0]).resolve()
        print(f"Loading single NRRD projection stack: {image_path}")
        import nrrd
        data, header = nrrd.read(str(image_path))
        # Transpose from (W, H, num_views) to (num_views, H, W)
        projs = np.transpose(data, (2, 1, 0))
        images = [projs[i].astype(np.float32) for i in range(projs.shape[0])]
    else:
        # Multiple images
        for image_name in tqdm(scan["image_files"], desc="Loading images..."):
            image_path = (data_dir_path / image_name).resolve()
            if str(image_path).lower().endswith('.nrrd'):
                import nrrd
                img, header = nrrd.read(str(image_path))
                img = np.squeeze(img)
                if img.ndim == 2:
                    img = img.T
                images.append(img.astype(np.float32))
            else:
                images.append(imread(image_path).astype(np.float32))

    H_img, W_img = images[0].shape
    image_size = (W_img, H_img)
    if "image_size" in scan and scan["image_size"] is not None:
        image_size = tuple(scan["image_size"])

    ompl_path = (scan_dir / scan["ompl_file"]).resolve()
    Ps_loaded = load_ompl(ompl_path)

    print("Raw CT Scan ", len(images), " image(s) ", len(Ps_loaded), "matrices")

    pixel_spacing = float(scan.get("pixel_spacing", Ps_loaded[0].pixel_spacing if Ps_loaded else 1.0))
    Ps = [
        ProjectionMatrix(
            P.P if isinstance(P, ProjectionMatrix) else P,
            image_size=image_size,
            pixel_spacing=pixel_spacing,
        )
        for P in Ps_loaded
    ]

    stages = [
        json.loads((config_path.parent / stage).read_text())
        for stage in config["geometry_optimization"]["stages"]
    ]

    calib = CalibrationAndMotionCorrection(
        Is=images,
        Ps=Ps,
        stages=stages,
        metric_config=config.get("metric_config", {}),
    )

    output_dir_path = config["output_dir"]
    if not os.path.isabs(output_dir_path):
        output_dir_path = os.path.normpath(os.path.join(config_path.parent, output_dir_path))
    output_dir = Path(output_dir_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run the optimization process
    result = calib.optimize()

    # Save optimized parameterization to JSON file as a chain of all stages
    chain_of_all_stages = ParameterizationChain(calib.parameterizations)
    parameterization_json_path = output_dir / "parameterization.json"
    parameterization_json_path.write_text(json.dumps(chain_of_all_stages.to_dict(), indent=2))
    print(f"Saved optimized parameterization to:\n{parameterization_json_path}")

    # Set up initial and optimized trajectory OMPL paths
    trajectory_initial_path = output_dir / "trajectory_initial.ompl"
    trajectory_optimized_path = output_dir / "trajectory_optimized.ompl"

    # Reconstructions config files setup
    reconstruct_configs = None
    
    reconstruction_config_path_str = config.get("reconstruction_config")
    if reconstruction_config_path_str:
        # Resolve path relative to config_path
        if not os.path.isabs(reconstruction_config_path_str):
            reconstruction_config_path_str = os.path.normpath(os.path.join(config_path.parent, reconstruction_config_path_str))
        recon_config_path_abs = Path(reconstruction_config_path_str).resolve()
        
        # Load the original reconstruction config
        recon_config = json.loads(recon_config_path_abs.read_text())
        
        # 1. Update the data_dir field:
        original_data_dir = recon_config.get("data_dir", "./")
        absolute_data_dir = (recon_config_path_abs.parent / original_data_dir).resolve()
        relative_data_dir = safe_relpath(absolute_data_dir, output_dir)
        recon_config["data_dir"] = relative_data_dir
        
        # Determine the name of the original output file
        orig_output_file = recon_config.get("output_file", "reconstruction.nrrd")
        orig_output_path_abs = (recon_config_path_abs.parent / orig_output_file).resolve()
        orig_dir = orig_output_path_abs.parent
        orig_base = orig_output_path_abs.stem
        orig_ext = orig_output_path_abs.suffix
        
        # Compute relative paths from output_dir to the original reconstructions folder
        rel_to_recon_dir = safe_relpath(orig_dir, output_dir)
        
        # Set output volume paths with YYYYMMDD_HHMM suffix
        recon_config["output_file"] = os.path.join(rel_to_recon_dir, f"{orig_base}_{output_dir.name}{orig_ext}")
        
        # 3. Apply parameterization to all views in original OMPL
        original_ompl_rel = recon_config["ompl_file"]
        original_ompl_path_abs = (recon_config_path_abs.parent / original_ompl_rel).resolve()
        Ps_all = load_ompl(original_ompl_path_abs)
        print(f"Applying optimized parameterizations to all {len(Ps_all)} views from\n{original_ompl_path_abs}")
        
        from copy import deepcopy
        param_list = [deepcopy(p) for p in calib.parameterizations]
        
        # Initialize prior_knowledge exactly once on the original full trajectory Ps_all
        for p in param_list:
            p.prior_knowledge = None
            p.estimateTrajectoryParameters(Ps_all)
            
        Ps_all_opt = Ps_all
        for p in param_list:
            Ps_all_opt = p.apply_to_trajectory(Ps_all_opt)
            
        print("Aligning optimized trajectory to the initial trajectory in voxel space...")
        Ps_all_opt, _ = align_trajectories(Ps_all_opt, Ps_all)
            
        # Save initial trajectory (all views)
        save_ompl(
            Ps_all,
            trajectory_initial_path,
            first_line_comment=f"Initial full trajectory using config: {config_path.name}",
            spacing=pixel_spacing,
            detector_size_px=image_size
        )
        print(f"Saved initial trajectory to:\n{trajectory_initial_path}")

        # Save optimized trajectory (all views) to the output directory
        first_line_comment = f"Optimized via epipolar consistency using config: {config_path.name}"
        save_ompl(
            Ps_all_opt,
            trajectory_optimized_path,
            first_line_comment=first_line_comment,
            spacing=pixel_spacing,
            detector_size_px=image_size
        )
        print(f"Saved optimized trajectory to:\n{trajectory_optimized_path}")
        
        # Set the ompl_file in recon_config relative to the config file (i.e. simply trajectory_optimized.ompl)
        recon_config["ompl_file"] = "trajectory_optimized.ompl"
        
        # Save the adapted config to reconstruction.json in output directory
        reconstruction_json_path = output_dir / "reconstruction.json"
        reconstruction_json_path.write_text(json.dumps(recon_config, indent=2))
        
        reconstruct_configs = (recon_config_path_abs, reconstruction_json_path)
    else:
        # No reconstruction config specified, save initial and optimized subset trajectory
        save_ompl(
            Ps,
            trajectory_initial_path,
            first_line_comment="Initial trajectory",
            spacing=pixel_spacing,
            detector_size_px=image_size
        )
        print(f"Saved initial trajectory to:\n{trajectory_initial_path}")

        print("Aligning optimized trajectory subset to the initial trajectory subset...")
        Ps_opt_aligned, _ = align_trajectories(calib.scan.Ps, Ps)

        save_ompl(
            Ps_opt_aligned,
            trajectory_optimized_path,
            first_line_comment="Optimized via epipolar consistency",
            spacing=pixel_spacing,
            detector_size_px=image_size
        )
        print(f"Saved optimized trajectory to:\n{trajectory_optimized_path}")

    # Verify the saved parameterization.json and trajectory_initial.ompl reproduce trajectory_optimized.ompl
    print("Verifying saved parameterization and trajectories...")
    try:
        saved_chain_dict = json.loads(parameterization_json_path.read_text())
        chain_stages = []
        if "parameterizations" in saved_chain_dict:
            for p_dict in saved_chain_dict["parameterizations"]:
                chain_stages.append(from_dict(p_dict))
        else:
            for st_dict in saved_chain_dict.get("stages", []):
                chain_stages.append(from_dict(st_dict["parameterization"]))
            
        # 2. Load trajectory_initial.ompl
        Ps_initial_loaded = load_ompl(trajectory_initial_path)
        
        # 3. Apply chain
        Ps_test = Ps_initial_loaded
        for p in chain_stages:
            p.prior_knowledge = None
            p.estimateTrajectoryParameters(Ps_initial_loaded)
            Ps_test = p.apply_to_trajectory(Ps_test)
            
        # 4. Load trajectory_optimized.ompl
        Ps_opt_loaded = load_ompl(trajectory_optimized_path)

        # Align Ps_test to Ps_opt_loaded to account for the rigid alignment transformation applied to trajectory_optimized.ompl
        print("Aligning test trajectory to optimized trajectory for verification...")
        Ps_test, _ = align_trajectories(Ps_test, Ps_opt_loaded)

        # 5. Assert equality of projection matrices
        assert len(Ps_test) == len(Ps_opt_loaded), "Verification error: Trajectory lengths do not match!"
        for idx, (P_test, P_opt) in enumerate(zip(Ps_test, Ps_opt_loaded)):
            mat_test = P_test.P if hasattr(P_test, "P") else P_test
            mat_opt = P_opt.P if hasattr(P_opt, "P") else P_opt
            # Use a scale-aware absolute tolerance to prevent failures with large values
            max_val = np.max(np.abs(mat_opt))
            atol = max(1e-5, 1e-5 * max_val)
            np.testing.assert_allclose(mat_test, mat_opt, rtol=1e-5, atol=atol,
                                       err_msg=f"Verification assertion failed: mismatch at matrix index {idx}")
        print("Verification assertion PASSED successfully. The parameterization chain correctly maps the initial trajectory to the optimized trajectory.")
    except Exception as e:
        import traceback
        print(f"Assertion Error/Verification Failed: {e}")
        traceback.print_exc()
        raise e

    # Print final optimized parameters for all stages
    print("\n==============================================================")
    print("Optimization Completed. Final Parameter Values for Each Stage:")
    print("==============================================================")
    for stage_idx, stage_res in enumerate(result["stages"]):
        stage_name = stage_res.get("name", f"Stage {stage_idx + 1}")
        print(f"\n{stage_name}:")
        print(calib.parameterizations[stage_idx])
    print("==============================================================\n")

    # Pipeline Step 1: Geometry Refinement
    if config.get("run_refinement", False):
        print("\n==============================================================")
        print("Pipeline: Triggering Geometry Refinement...")
        print("==============================================================")
        from xray_epipolar_consistency.tools import geometry_refinement
        geometry_refinement.main(str(config_path))

    # Pipeline Step 2: Reconstruction
    if config.get("run_reconstruction", False):
        print("\n==============================================================")
        print("Pipeline: Triggering Direct Reconstruction...")
        print("==============================================================")
        recon_config = config.get("reconstruction_config")
        if not recon_config:
            print("Error: reconstruction_config not specified in configuration. Skipping reconstruction.")
        else:
            if not os.path.isabs(recon_config):
                recon_config = os.path.normpath(os.path.join(os.path.dirname(config_path), recon_config))
            
            # Resolve reconstruct.py path
            reconstruct_file = None
            import ct_recon_fdk_astra.reconstruct as reconstruct
            reconstruct_file = os.path.abspath(reconstruct.__file__)
            if reconstruct_file.endswith('.pyc'):
                reconstruct_file = reconstruct_file[:-1]


            if not reconstruct_file:
                cur_dir = os.path.dirname(os.path.abspath(__file__))
                possible = [
                    os.path.normpath(os.path.join(cur_dir, "..", "..", "ct_recon_fdk_astra", "ct_recon_fdk_astra", "reconstruct.py")),
                    "/home/aaichert/Desktop/install_test/ct_recon_fdk_astra/ct_recon_fdk_astra/reconstruct.py"
                ]
                for p in possible:
                    if os.path.exists(p):
                        reconstruct_file = p
                        break

            if not reconstruct_file:
                print("Error: Could not locate reconstruct.py. Skipping reconstruction.")
            else:
                import subprocess
                suffix = "_refined" if config.get("run_refinement", False) else "_optimized"
                ompl_filename = f"trajectory{suffix}.ompl"
                ompl_path = os.path.join(output_dir_path, ompl_filename)
                
                if not os.path.exists(ompl_path):
                    if os.path.exists(os.path.join(output_dir_path, "trajectory_refined.ompl")):
                        ompl_path = os.path.join(output_dir_path, "trajectory_refined.ompl")
                        suffix = "_refined"
                    elif os.path.exists(os.path.join(output_dir_path, "trajectory_optimized.ompl")):
                        ompl_path = os.path.join(output_dir_path, "trajectory_optimized.ompl")
                        suffix = "_optimized"
                    else:
                        ompl_path = os.path.join(output_dir_path, "trajectory.ompl")
                        suffix = "_optimized"

                with open(recon_config, 'r') as f:
                    recon_dict = json.load(f)

                orig_data_dir = recon_dict.get("data_dir", "./")
                if not os.path.isabs(orig_data_dir):
                    abs_data_dir = os.path.normpath(os.path.join(os.path.dirname(recon_config), orig_data_dir))
                else:
                    abs_data_dir = orig_data_dir
                recon_dict["data_dir"] = abs_data_dir
                recon_dict["ompl_file"] = os.path.abspath(ompl_path)

                base_output = recon_dict.get("output_file", "reconstruction.nrrd")
                orig_output_path_abs = os.path.abspath(os.path.join(os.path.dirname(recon_config), base_output))
                orig_dir = os.path.dirname(orig_output_path_abs)
                orig_name = os.path.basename(orig_output_path_abs)
                stem, ext = os.path.splitext(orig_name)
                out_dir_name = os.path.basename(output_dir_path)

                out_vol_abs = os.path.join(orig_dir, f"{stem}_{out_dir_name}{suffix}{ext}")
                recon_dict["output_file"] = os.path.relpath(out_vol_abs, output_dir_path)

                new_recon_json_path = os.path.join(output_dir_path, f"reconstruction{suffix}.json")
                with open(new_recon_json_path, 'w') as rf:
                    json.dump(recon_dict, rf, indent=2)

                cmd = [sys.executable, "-u", reconstruct_file, new_recon_json_path]
                print(f"\nRunning reconstruction command: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
                print(f"Direct reconstruction completed successfully. Output saved to:\n{out_vol_abs}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <config.json>")
        sys.exit(1)
    main(sys.argv[1])
