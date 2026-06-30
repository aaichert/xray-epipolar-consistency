#!/usr/bin/env python3
import os
import sys
import json
import random
import shutil
from pathlib import Path
from copy import deepcopy
import numpy as np

# Setup paths
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

RECONSTRUCT_DIR = ROOT.parent / "reconstruct"
RECONSTRUCT_EXAMPLE_DATA = RECONSTRUCT_DIR / "example_data"


try:
    from reconstruct import main as reconstruct_main
except ImportError:
    reconstruct_main = None
from xray_epipolar_consistency.tools.geometry_correction import main as geometry_correction_main
from xray_epipolar_consistency.tools.utils import extract_and_save_slices

from fileformats.ompl import load_ompl, save_ompl
from xray_epipolar_consistency.parameterization import (
    DetectorShift,
    DetectorOrientation,
    ObjectPose,
    RotationAxis,
    Distance,
    GantryAngle,
    LinearDrift,
    TimeVariant,
    ParameterizationChain,
    from_dict,
    ContinuousMotion,
    ParameterizationBase
)

CLASSES_TO_TEST = [
    DetectorShift,
    DetectorOrientation,
    RotationAxis,
    Distance,
    GantryAngle,
    LinearDrift,
    ContinuousMotion
]

def make_parameterization_instance(cls, ref_cls=DetectorShift, num_control_points=4):
    if issubclass(cls, TimeVariant):
        return cls(referenced_class=ref_cls, num_control_points=num_control_points)
    return cls()

def set_random_parameter_values(param_inst):
    # Iterate through parameters and set a random value in their range for active parameters
    for name, p_info in param_inst.parameters.items():
        if p_info["opt"]:
            p_min, p_max = p_info["range"]
            while True:
                val = np.random.uniform(p_min, p_max)
                if val >= 0:
                    if val > 0.5 * p_max:
                        break
                else:
                    if val < 0.5 * p_min:
                        break
            p_info["value"] = float(val)

def extract_parameter_comparison(cls, param_inst, optimized_param_dict):
    param_initial = ParameterizationChain([param_inst])
    param_opt = from_dict(optimized_param_dict)
    
    param_stats = {}
    for name in param_initial.keys():
        p_initial = param_initial[name]
        p_opt = param_opt[name]
        
        val_mis = p_initial["value"]
        val_rec = p_opt["value"]
        diff = val_mis + val_rec
        p_min, p_max = p_initial["range"]
        range_span = p_max - p_min
        pct_range = (diff / range_span * 100.0) if range_span > 0 else 0.0
        
        param_stats[name] = {
            "misalignment_value": val_mis,
            "recovered_value": val_rec,
            "difference": diff,
            "pct_range": pct_range
        }
    return param_stats

def generate_html_report(results, output_base):
    import base64
    def get_b64(path):
        p = Path(path)
        if p.exists():
            return f"data:image/png;base64,{base64.b64encode(p.read_bytes()).decode('utf-8')}"
        return ""

    visual_rows = []
    for class_name, res in results.items():
        display_name = res.get("display_name", class_name)
        initial_param = res.get("initial_parameterization", {})
        initial_json_str = json.dumps(initial_param, indent=2)
        preview_dir = output_base / class_name / "preview"
        
        img_tags = []
        has_images = False
        for axis in ["x", "y", "z"]:
            for view in ["misaligned", "optimized", "gt"]:
                img_path = preview_dir / f"{view}_slice_{axis}.png"
                if img_path.exists():
                    has_images = True
                img_tags.append(f'<img class="img-{view}-{axis}" src="{get_b64(img_path)}">')
                
        viewer_content = "".join(img_tags) if has_images else '<div style="color: #888; padding: 20px; text-align: center; line-height: 472px;">Reconstructions / Slices N/A (ASTRA required)</div>'
        row = f"""
        <tr>
            <td class="rotate" style="vertical-align: middle;"><strong>{display_name}</strong></td>
            <td>
                <div class="viewer">
                    {viewer_content}
                </div>
            </td>
            <td style="text-align: left; vertical-align: top;"><pre style="margin: 0; font-size: 11px;">{initial_json_str}</pre></td>
        </tr>
        """
        visual_rows.append(row)

    param_rows = []
    for class_name, res in results.items():
        comp = res["parameter_comparison"]
        num_params = len(comp)
        stats_3d = res.get("image_statistics_3d")
        display_name = res.get("display_name", class_name)
        
        first = True
        for param_name, p_stats in comp.items():
            mis_val = f"{p_stats['misalignment_value']:.4f}"
            rec_val = f"{p_stats['recovered_value']:.4f}"
            diff_val = f"{p_stats['difference']:.4f}"
            pct_val = f"{p_stats['pct_range']:.2f}%"
            
            if first:
                if stats_3d:
                    mae_str = f"{stats_3d['mean_abs_error']:.4e}"
                    rmse_str = f"{stats_3d['root_mean_square_error']:.4e}"
                else:
                    mae_str = "N/A"
                    rmse_str = "N/A"
                
                row = f"""
                <tr>
                    <td rowspan="{num_params}" class="rotate" style="vertical-align: middle;"><strong>{display_name}</strong></td>
                    <td><code>{param_name}</code></td>
                    <td>{mis_val}</td>
                    <td>{rec_val}</td>
                    <td>{diff_val}</td>
                    <td>{pct_val}</td>
                    <td rowspan="{num_params}" style="vertical-align: middle;">{mae_str}</td>
                    <td rowspan="{num_params}" style="vertical-align: middle;">{rmse_str}</td>
                </tr>
                """
                first = False
            else:
                row = f"""
                <tr>
                    <td><code>{param_name}</code></td>
                    <td>{mis_val}</td>
                    <td>{rec_val}</td>
                    <td>{diff_val}</td>
                    <td>{pct_val}</td>
                </tr>
                """
            param_rows.append(row)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Parameterization Validation Report</title>
    <style>
        table {{ border-collapse: collapse; }}
        th, td {{ border: 1px solid #ccc; text-align: center; }}
        .rotate {{ writing-mode: vertical-rl; transform: rotate(180deg); white-space: nowrap; }}
        .viewer {{ width: 512px; height: 512px; background: #000; position: relative; }}
        .viewer img {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: none; object-fit: contain; }}
        .controls {{ position: sticky; top: 0; background: white; padding: 5px 0; z-index: 10; }}
    </style>
    <script>
        function updateView() {{
            var view = document.getElementById('view-select').value;
            var axis = document.getElementById('axis-select').value;
            var imgs = document.querySelectorAll('.viewer img');
            for (var i = 0; i < imgs.length; i++) {{
                var img = imgs[i];
                if (img.classList.contains('img-' + view + '-' + axis)) {{
                    img.style.display = 'block';
                }} else {{
                    img.style.display = 'none';
                }}
            }}
        }}
        window.onload = updateView;
    </script>
</head>
<body>
    <div class="controls">
        <label for="view-select">View:</label>
        <select id="view-select" onchange="updateView()">
            <option value="optimized">Corrected (Optimized)</option>
            <option value="misaligned" selected>Misaligned</option>
            <option value="gt">Ground Truth</option>
        </select>

        <label for="axis-select">Central Plane:</label>
        <select id="axis-select" onchange="updateView()">
            <option value="x">X (fast)</option>
            <option value="y" selected>Y</option>
            <option value="z">Z (slow)</option>
        </select>
    </div>

    <div class="content">
        <h1>Parameterization Validation Report</h1>
        
        <h2>1. Visual Results Table</h2>
        <table>
            <thead>
                <tr>
                    <th>Class</th>
                    <th>Interactive Slice Viewer</th>
                    <th>Random Misalignment</th>
                </tr>
            </thead>
            <tbody>
                {"".join(visual_rows)}
            </tbody>
        </table>

        <h2>2. Parameter Value Differences and Image Statistics Table</h2>
        <table>
            <thead>
                <tr>
                    <th>Class</th>
                    <th>Parameter</th>
                    <th>Misalignment</th>
                    <th>Correction (rec.)</th>
                    <th>Diff (mis + rec)</th>
                    <th>Diff % Range</th>
                    <th>3D MAE</th>
                    <th>3D RMSE</th>
                </tr>
            </thead>
            <tbody>
                {"".join(param_rows)}
            </tbody>
        </table>
    </div>
</body>
</html>
"""
    with open(output_base / "validation.html", "w") as f:
        f.write(html)
        
    old_report = output_base / "index.html"
    if old_report.exists():
        try:
            old_report.unlink()
        except Exception:
            pass

def main():
    if reconstruct_main is None:
        print("Warning: 'reconstruct' package (requires ASTRA) is not available. Skipping 3D reconstructions.")
        
    output_base = ROOT / "output" / "validation"
    output_base.mkdir(parents=True, exist_ok=True)
    gt_val_path = output_base / "ground_truth.nrrd"
    
    # 1. Run reconstruction on original fullscan to generate ground truth 3D image (if ASTRA is available)
    if reconstruct_main is not None:
        if not gt_val_path.exists():
            gt_nrrd_path = RECONSTRUCT_DIR / "output" / "fullscan_180views_600x400.nrrd"
            if not gt_nrrd_path.exists():
                print("Generating ground truth 3D reconstruction...")
                gt_config_path = RECONSTRUCT_EXAMPLE_DATA / "fullscan_180views_600x400.json"
                reconstruct_main(str(gt_config_path))

            
            if not gt_nrrd_path.exists():
                print(f"Error: Ground truth reconstruction not found at {gt_nrrd_path}")
                sys.exit(1)
                
            # Store global GT path
            shutil.copy(str(gt_nrrd_path), str(gt_val_path))
        else:
            print("Ground truth reconstruction already exists, skipping.")
    
    results = {}
    
    for idx, cls in enumerate(CLASSES_TO_TEST):
        class_name = cls.__name__
        print(f"\n===================================================================")
        print(f"[{idx + 1}/{len(CLASSES_TO_TEST)}] Validating parameterization class: {class_name}")
        print(f"===================================================================")
        
        class_dir = output_base / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        preview_dir = class_dir / "preview"
        preview_dir.mkdir(parents=True, exist_ok=True)
        
        # Load GT trajectories
        ompl_18_path = ROOT / "xray_epipolar_consistency" / "example_data" / "synthetic_pumpkin" / "fullscan_18views_600x400.ompl"
        ompl_180_path = RECONSTRUCT_EXAMPLE_DATA / "fullscan_180views_600x400.ompl"

        
        Ps_18 = load_ompl(ompl_18_path)
        Ps_180 = load_ompl(ompl_180_path)
        
        # Assert that 18 matrices are the same as every 10th matrix in the 180 case
        print("Asserting initial 18 views are subset of 180 views...")
        for idx in range(18):
            np.testing.assert_allclose(Ps_18[idx].P, Ps_180[idx * 10].P, rtol=1e-5, atol=1e-5)
            
        # Seed random numbers based on class name for reproducibility
        np.random.seed(42)
        random.seed(42)
        
        # Configure case-specific parameters
        ref_cls = DetectorShift
        num_control_points = 4
        if cls == LinearDrift:
            ref_cls = RotationAxis
            num_control_points = 2
        elif cls == ContinuousMotion:
            ref_cls = ObjectPose
            num_control_points = 5
            
        # Set random values to all parameters
        param_inst = make_parameterization_instance(cls, ref_cls=ref_cls, num_control_points=num_control_points)
        # Estimate prior knowledge on original trajectory
        param_inst.estimateTrajectoryParameters(Ps_180)
        set_random_parameter_values(param_inst)
        
        # Apply misalignment
        print("Applying random misalignment...")
        misaligned_Ps_180 = param_inst.apply_to_trajectory(Ps_180)
        if issubclass(cls, TimeVariant):
            # Evaluate using matching subsampling index mapping
            misaligned_Ps_18 = [misaligned_Ps_180[i * 10] for i in range(18)]
        else:
            param_inst_18 = make_parameterization_instance(cls, ref_cls=ref_cls, num_control_points=num_control_points)
            param_inst_18.parameters = deepcopy(param_inst.parameters)
            param_inst_18.prior_knowledge = deepcopy(param_inst.prior_knowledge)
            misaligned_Ps_18 = param_inst_18.apply_to_trajectory(Ps_18)
            
        # Assert that misaligned 18 matrices are the same as every 10th matrix in the 180 case
        print("Asserting misaligned 18 views are subset of 180 views...")
        for idx in range(18):
            np.testing.assert_allclose(misaligned_Ps_18[idx].P, misaligned_Ps_180[idx * 10].P, rtol=1e-5, atol=1e-5)
            
        # Store parameterization and trajectories
        print("Saving misaligned parameterization and trajectories...")
        with open(class_dir / "random_misalignment.json", "w") as f:
            json.dump(param_inst.to_dict(), f, indent=2)
            
        save_ompl(misaligned_Ps_18, str(class_dir / "misaligned_18views.ompl"))
        save_ompl(misaligned_Ps_180, str(class_dir / "misaligned_180views.ompl"))
        
        # Create fullscan_180views_600x400.json that points to the randomly misaligned ompl
        with open(RECONSTRUCT_EXAMPLE_DATA / "fullscan_180views_600x400.json", "r") as f:
            orig_180_cfg = json.load(f)
            
        new_180_cfg = orig_180_cfg.copy()
        new_180_cfg["data_dir"] = os.path.abspath(RECONSTRUCT_EXAMPLE_DATA)

        new_180_cfg["ompl_file"] = "misaligned_180views.ompl"
        new_180_cfg["output_file"] = "recon_misaligned.nrrd"
        
        with open(class_dir / "fullscan_180views_600x400.json", "w") as f:
            json.dump(new_180_cfg, f, indent=2)
            
        # Generate recon_misaligned.nrrd
        if reconstruct_main is not None:
            print("Reconstructing misaligned 180 views scan...")
            reconstruct_main(str(class_dir / "fullscan_180views_600x400.json"))
        else:
            print("Skipping misalignment reconstruction (ASTRA not available).")
        
        # Create fullscan_18views_600x400.json in class_dir
        with open(ROOT / "xray_epipolar_consistency" / "example_data" / "synthetic_pumpkin" / "fullscan_18views_600x400.json", "r") as f:
            orig_18_cfg = json.load(f)
            
        new_18_cfg = orig_18_cfg.copy()
        new_18_cfg["data_dir"] = os.path.abspath(ROOT / "xray_epipolar_consistency" / "example_data" / "synthetic_pumpkin")
        new_18_cfg["ompl_file"] = "misaligned_18views.ompl"
        new_18_cfg["output_file"] = "reconstruction.nrrd"
        
        with open(class_dir / "fullscan_18views_600x400.json", "w") as f:
            json.dump(new_18_cfg, f, indent=2)
            
        # Create stage.json for geometry_correction.py
        stage_cfg = {
            "name": "Validation Stage",
            "module": "xray_epipolar_consistency.optimizer",
            "classname": "OptimizerLBFGS",
            "parameterization": make_parameterization_instance(cls, ref_cls=ref_cls, num_control_points=num_control_points).to_dict(),
            "kwargs": {
                "options": {
                    "maxiter": 50,
                    "ftol": 1e-5,
                    "gtol": 1e-5,
                    "eps": 1e-3
                }
            }
        }
        with open(class_dir / "stage.json", "w") as f:
            json.dump(stage_cfg, f, indent=2)
            
        # Create config_synthetic_pumpkin.json
        corr_cfg = {
            "input_data": "fullscan_18views_600x400.json",
            "output_dir": "./",
            "reconstruction_config": "fullscan_180views_600x400.json",
            "metric_config": {
                "convert_to_line_integral": False,
                "dtr_size_factor": 1.0,
                "num_planes": 900
            },
            "geometry_optimization": {
                "stages": [
                    "stage.json"
                ]
            }
        }
        with open(class_dir / "config_synthetic_pumpkin.json", "w") as f:
            json.dump(corr_cfg, f, indent=2)
            
        # Run geometry correction pipeline
        print("Running geometry correction optimization...")
        geometry_correction_main(str(class_dir / "config_synthetic_pumpkin.json"))
        
        # Align optimized trajectory back to ground truth trajectory
        optimized_ompl_path = class_dir / "trajectory_optimized.ompl"
        if optimized_ompl_path.exists():
            print("Aligning optimized trajectory to ground truth...")
            optimized_Ps_180 = load_ompl(optimized_ompl_path)
            aligned_Ps_180 = ParameterizationBase.align_trajectories(Ps_180, optimized_Ps_180)
            save_ompl(
                aligned_Ps_180,
                optimized_ompl_path,
                first_line_comment="Aligned to ground-truth trajectory",
                spacing=Ps_180[0].pixel_spacing,
                detector_size_px=Ps_180[0].image_size
            )
            
            # Reconstruct again with the aligned trajectory
            reconstruction_json_path = class_dir / "reconstruction.json"
            if reconstruction_json_path.exists() and reconstruct_main is not None:
                print("Re-reconstructing optimized scan using aligned trajectory...")
                reconstruct_main(str(reconstruction_json_path))
        

        # Load optimized parameterization
        with open(class_dir / "parameterization.json", "r") as f:
            optimized_param_dict = json.load(f)
            
        # Slices extraction & difference computation
        stats_3d = None
        stats_slices = None
        if reconstruct_main is not None:
            print("Extracting central slices and difference images...")
            preview_dir.mkdir(parents=True, exist_ok=True)
            try:
                gt_data, opt_data = extract_and_save_slices(
                    class_dir / "recon_misaligned.nrrd",
                    class_dir / f"recon_misaligned_{class_name}.nrrd",
                    preview_dir,
                    gt_path=gt_val_path
                )
                
                # Compute image stats of difference image
                diff_3d = np.abs(opt_data - gt_data)
                stats_3d = {
                    "mean_abs_error": float(diff_3d.mean()),
                    "root_mean_square_error": float(np.sqrt((diff_3d.astype(np.float64) ** 2).mean())),
                    "max_diff": float(diff_3d.max())
                }
            except Exception as e:
                print(f"Warning: Failed to extract slices or compute image statistics: {e}")
            
        # Parameter value differences
        param_comparison = extract_parameter_comparison(cls, param_inst, optimized_param_dict)
        
        display_name = f"{class_name}[{ref_cls.__name__}]" if issubclass(cls, TimeVariant) else class_name
        results_json = {
            "class_name": class_name,
            "display_name": display_name,
            "initial_parameterization": param_inst.to_dict(),
            "image_statistics_3d": stats_3d,
            "image_statistics_slices": stats_slices,
            "parameter_comparison": param_comparison
        }
        with open(class_dir / "validation_results.json", "w") as f:
            json.dump(results_json, f, indent=2)
            
        results[class_name] = results_json
        print(f"Validation for {class_name} completed successfully.")

    # 2. Write simple HTML file at the end
    print("\nGenerating final HTML report...")
    generate_html_report(results, output_base)

if __name__ == "__main__":
    main()
