import sys
import os
import json
import numpy as np
import nrrd
import matplotlib.pyplot as plt
import math 
# Remove current directory from sys.path to avoid picking up local uncompiled modules
sys.path = [p for p in sys.path if os.path.abspath(p) != os.path.abspath(os.path.dirname(__file__))]

from ProjectiveGeometry23.central_projection import ProjectionMatrix
from ProjectiveGeometry23.homography import translation2d
import xray_epipolar_consistency as ecc

def main():
    examples_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = sys.argv[1] if len(sys.argv) > 1 else "plot_redundancy.json"
    config_path = os.path.join(examples_dir, config_file) if not os.path.isabs(config_file) else config_file
    with open(config_path, 'r') as f:
        config = json.load(f)

    image0_path = config["image0_path"]
    image1_path = config["image1_path"]
    if not os.path.isabs(image0_path):
        image0_path = os.path.join(examples_dir, image0_path)
    if not os.path.isabs(image1_path):
        image1_path = os.path.join(examples_dir, image1_path)
    object_radius = config["object_radius"]

    print(f"Loading {image0_path} and {image1_path}...")

    raw_I0, header0 = nrrd.read(image0_path)
    raw_I1, header1 = nrrd.read(image1_path)
    
    # Transpose to shape (H, W) and convert to float32
    raw_I0 = raw_I0.T.astype(np.float32)
    raw_I1 = raw_I1.T.astype(np.float32)

    P0 = ProjectionMatrix(np.matrix(header0['Projection Matrix']))
    P1 = ProjectionMatrix(np.matrix(header1['Projection Matrix']))

    # Size parameters for RadonIntermediate
    size_t = int(np.ceil(np.hypot(raw_I0.shape[0], raw_I0.shape[1]))) // 2
    size_alpha = int(np.ceil((np.pi / 2.0) * size_t))

    print("Computing Radon Intermediate functions...")
    # Using Derivative filter and Identity post processing, as in plot_redundancy.py
    dtr0 = ecc.RadonIntermediate(
        raw_I0.copy(), 
        size_alpha, 
        size_t, 
        int(ecc.RadonFilter.Derivative), 
        int(ecc.RadonPostProcess.Identity)
    )
    dtr1 = ecc.RadonIntermediate(
        raw_I1.copy(), 
        size_alpha, 
        size_t, 
        int(ecc.RadonFilter.Derivative), 
        int(ecc.RadonPostProcess.Identity)
    )

    # Initialize MetricRadonIntermediate
    metric = ecc.MetricRadonIntermediate()
    metric.setRadonIntermediates([dtr0, dtr1])
    metric.setObjectRadius(object_radius)
    metric.setEpipolarPlaneNumber(config.get("num_planes", 1800))

    # Prepare translation values from -10 to +10 pixels (201 points)
    v_trans_values = np.linspace(-10.0, 10.0, 201)
    
    # Generate all projection matrices
    # Ps_all[0] corresponds to dtrs[1] (view 40, unmodified)
    # Ps_all[1 + m] corresponds to dtrs[0] (view 0, modified with translation2d)
    Ps_all = [P1.P]
    for v in v_trans_values:
        H = translation2d([0.0, v])
        # P0.P is np.matrix, H is np.ndarray, H @ P0.P returns np.matrix (3, 4)
        P0_modified = H @ P0.P
        Ps_all.append(np.array(P0_modified, dtype=np.float64))

    metric.setProjectionMatrices(Ps_all)

    # Build evaluation indices
    # We evaluate P_view40 (index 0) with dtr_view40 (index 1) 
    # against P_view0_candidate (index 1 + m) with dtr_view0 (index 0)
    # So the indices are (0, 1 + m, 1, 0)
    indices = []
    for m in range(len(v_trans_values)):
        indices.append([0, 1 + m, 1, 0])

    print("Evaluating metric on GPU...")
    # Call the newly exposed evaluate_indices
    costs = metric.evaluate_indices(indices)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(v_trans_values, costs, label='Epipolar Consistency Cost', color='#1f77b4', linewidth=2)
    plt.axvline(0, color='red', linestyle='--', alpha=0.7, label='Ground Truth (v = 0)')
    plt.xlabel('Detector v-translation (pixels)')
    plt.ylabel('[a.u.]')
    plt.gca().yaxis.set_major_formatter(plt.NullFormatter())
    plt.title('Epipolar Consistency Metric vs. Detector v-translation')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    output_dir_name = config.get("output_dir", "output")
    output_dir = os.path.join(examples_dir, output_dir_name) if not os.path.isabs(output_dir_name) else output_dir_name
    os.makedirs(output_dir, exist_ok=True)
    
    output_pdf = os.path.join(output_dir, "plot_metric.pdf")
    plt.savefig(output_pdf, bbox_inches='tight')
    output_png = os.path.join(output_dir, "plot_metric.png")
    plt.savefig(output_png, bbox_inches='tight')
    print(f"Plot saved to: {output_pdf} and {output_png}")

if __name__ == "__main__":
    main()
