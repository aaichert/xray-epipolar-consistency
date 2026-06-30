import os
import numpy as np
import nrrd
from ProjectiveGeometry23.central_projection import ProjectionMatrix
import xray_epipolar_consistency as ecc

def test_synthetic_epipolar_consistency():
    tests_dir = os.path.dirname(__file__)
    workspace_dir = os.path.dirname(tests_dir)
    
    image0_path = os.path.join(workspace_dir, "xray_epipolar_consistency", "example_data", "synthetic01.nrrd")
    image1_path = os.path.join(workspace_dir, "xray_epipolar_consistency", "example_data", "synthetic02.nrrd")
    
    assert os.path.exists(image0_path), f"File not found: {image0_path}"
    assert os.path.exists(image1_path), f"File not found: {image1_path}"
    
    raw_I0, header0 = nrrd.read(image0_path)
    raw_I1, header1 = nrrd.read(image1_path)
    
    # Transpose to shape (H, W) and convert to float32
    raw_I0 = raw_I0.T.astype(np.float32)
    raw_I1 = raw_I1.T.astype(np.float32)
    
    P0 = ProjectionMatrix(np.matrix(header0['Projection Matrix']))
    P1 = ProjectionMatrix(np.matrix(header1['Projection Matrix']))
    
    size_t = int(np.ceil(np.hypot(raw_I0.shape[0], raw_I0.shape[1]))) // 2
    size_alpha = int(np.ceil((np.pi / 2.0) * size_t))

    raw_dtr0 = ecc.RadonIntermediate(raw_I0, size_alpha, size_t, int(ecc.RadonFilter.Derivative), int(ecc.RadonPostProcess.Identity))
    raw_dtr1 = ecc.RadonIntermediate(raw_I1, size_alpha, size_t, int(ecc.RadonFilter.Derivative), int(ecc.RadonPostProcess.Identity))

    # Compute using C++ implementation
    cost, v0s, v1s, kappas, _ = ecc.compute_for_image_pair(
        P0.P, P1.P, raw_dtr0, raw_dtr1, 
        1800, 200.0
    )
    
    print(f"Cost: {cost}")
    print(f"v0s shape: {v0s.shape}, range: {v0s.min()} to {v0s.max()}")
    print(f"v1s shape: {v1s.shape}, range: {v1s.min()} to {v1s.max()}")
    
    # Since the images are synthetic and symmetric, the redundancy signals v0s and v1s should be extremely close.
    # We will assert that the mean absolute difference is small.
    # At the moment they seem to differ due to a bug, so this test might fail initially.
    mae = np.mean(np.abs(v0s - v1s))
    max_val = max(np.max(np.abs(v0s)), np.max(np.abs(v1s)))
    rel_error = mae / (max_val + 1e-8)
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Relative Error: {rel_error * 100:.2f}%")
    
    assert rel_error < 0.01, f"Redundancy signals differ significantly: MAE={mae:.4f}, Relative Error={rel_error*100:.2f}%"

if __name__ == "__main__":
    test_synthetic_epipolar_consistency()
