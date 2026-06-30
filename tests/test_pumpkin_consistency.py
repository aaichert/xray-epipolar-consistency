import os
import numpy as np
import nrrd
from ProjectiveGeometry23.central_projection import ProjectionMatrix
from fileformats.ompl import load_ompl
from xray_epipolar_consistency.scan import Scan
import xray_epipolar_consistency as ecc

def test_pumpkin_consistency():
    tests_dir = os.path.dirname(__file__)
    workspace_dir = os.path.dirname(tests_dir)
    
    pumpkin_dir = os.path.join(workspace_dir, "xray_epipolar_consistency", "example_data", "synthetic_pumpkin")

    nrrd_path = os.path.join(pumpkin_dir, "fullscan_18views_600x400.nrrd")
    ompl_path = os.path.join(pumpkin_dir, "fullscan_18views_600x400.ompl")
    
    assert os.path.exists(nrrd_path), f"File not found: {nrrd_path}"
    assert os.path.exists(ompl_path), f"File not found: {ompl_path}"
    
    data, header = nrrd.read(nrrd_path)
    # Transpose from (W, H, num_views) to (num_views, H, W)
    projs = np.transpose(data, (2, 1, 0))
    
    images = [projs[i].astype(np.float32) for i in range(projs.shape[0])]
    Ps_loaded = load_ompl(ompl_path)
    
    scan = Scan(Is=images, Ps=Ps_loaded)
    scan.init_epipolar_consistency(
        convert_to_line_integral=False,
        gaussian_sigma=1.2,
        dtr_size_factor=0.5,
        num_planes=180
    )
    
    # Compute GPU cost matrix (fills the lower triangle, i.e., cost_matrix[j, i] for j > i)
    gpu_mean, cost_matrix = scan.compute_epipolar_consistency()
    
    # Get aligned projection matrices used in the GPU metric
    Ps_aligned = [P.P @ scan.T_norm for P in scan.Ps]
    
    # Compute CPU cost matrix and write it to the upper triangle (cost_matrix[j, i] for j < i)
    num_views = len(images)
    rel_diffs = []
    
    for i in range(num_views):
        for j in range(i + 1, num_views):
            # Evaluate the pair (i, j) on the CPU
            cpu_cost, _, _, _, weight = ecc.compute_for_image_pair(
                Ps_aligned[i],
                Ps_aligned[j],
                scan.dtrs[i],
                scan.dtrs[j],
                scan.num_planes,
                scan.object_radius_mm
            )
            
            weighted_cpu_cost = cpu_cost * weight
            
            # Write to the upper triangle: cost_matrix[i, j]
            cost_matrix[i, j] = weighted_cpu_cost
            
            # Read GPU cost from the lower triangle: cost_matrix[j, i]
            gpu_cost = cost_matrix[j, i]
            
            assert cpu_cost > 0, f"CPU cost for pair ({i}, {j}) is zero/negative: {cpu_cost}"
            assert gpu_cost > 0, f"GPU cost for pair ({i}, {j}) is zero/negative: {gpu_cost}"
            
            if weighted_cpu_cost > 10.0:
                rel_diff = abs(gpu_cost - weighted_cpu_cost) / weighted_cpu_cost
                rel_diffs.append(rel_diff)
                
                # Assert relative error of any single pair is within 10%
                assert rel_diff < 0.10, f"CPU and GPU costs for pair ({i}, {j}) differ too much: GPU={gpu_cost:.2f}, CPU={weighted_cpu_cost:.2f}, rel_diff={rel_diff * 100:.2f}%"
            else:
                # For near-zero values, check that absolute difference is small
                abs_diff = abs(gpu_cost - weighted_cpu_cost)
                assert abs_diff < 10.0, f"CPU and GPU absolute difference for near-zero pair ({i}, {j}) too large: GPU={gpu_cost:.4f}, CPU={weighted_cpu_cost:.4f}"

    # Assert average relative error is within 3%
    mean_rel_diff = np.mean(rel_diffs)
    assert mean_rel_diff < 0.03, f"Average relative difference {mean_rel_diff * 100:.2f}% exceeds 3%"
    print(f"Consistency test passed! Mean relative error: {mean_rel_diff * 100:.2f}%")

if __name__ == "__main__":
    test_pumpkin_consistency()
