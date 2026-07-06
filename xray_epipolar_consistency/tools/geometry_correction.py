#!/usr/bin/env python3
import json
import sys
import io
import os
from pathlib import Path
from itertools import combinations

import numpy as np
from tifffile import imread
from ProjectiveGeometry23.central_projection import ProjectionMatrix
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from xray_epipolar_consistency import CalibrationAndMotionCorrection
from xray_epipolar_consistency.parameterization import from_dict, ParameterizationChain

from tqdm import tqdm

import re
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


def plot_parameter_sweep(scan, parameterization, param_name, Ps_start, number_of_samples=31):
    p_info = parameterization[param_name]
    original_val = p_info["value"]
    range_min, range_max = p_info["range"]
    samples = np.linspace(range_min, range_max, number_of_samples)

    Ps_list = []
    for v in samples:
        p_info["value"] = v
        Ps_list.append(parameterization.apply_to_trajectory(Ps_start))
    p_info["value"] = original_val

    costs = scan.compute_ecc_for_projection_matrices(Ps_list)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(samples, costs, color='#1f77b4', linewidth=2)
    ax.axvline(original_val, color='red', linestyle='--', alpha=0.7, label=f'Current ({original_val:.4f})')
    ax.set_xlabel(param_name)
    ax.set_ylabel('consistency [a.u.]')
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.set_title(f"ECC vs {param_name}")
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    return fig

def plot_2d_sweep(scan, parameterization, param_names, Ps_start, number_of_samples_per_axis=11):
    name1, name2 = param_names
    p_info1 = parameterization[name1]
    p_info2 = parameterization[name2]
    
    original_val1 = p_info1["value"]
    original_val2 = p_info2["value"]

    range_min1, range_max1 = p_info1["range"]
    range_min2, range_max2 = p_info2["range"]

    samples1 = np.linspace(range_min1, range_max1, number_of_samples_per_axis)
    samples2 = np.linspace(range_min2, range_max2, number_of_samples_per_axis)

    X, Y = np.meshgrid(samples1, samples2)

    Ps_list = []
    for v1, v2 in zip(X.ravel(), Y.ravel()):
        p_info1["value"] = v1
        p_info2["value"] = v2
        Ps_list.append(parameterization.apply_to_trajectory(Ps_start))

    p_info1["value"] = original_val1
    p_info2["value"] = original_val2

    costs = scan.compute_ecc_for_projection_matrices(Ps_list)
        
    Z = np.array(costs).reshape(X.shape)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='coolwarm', edgecolor='none', alpha=0.9)

    current_cost = Z[np.abs(samples2 - original_val2).argmin(), np.abs(samples1 - original_val1).argmin()]
    ax.scatter([original_val1], [original_val2], [current_cost], color='red', s=100, marker='*', zorder=10, label='Optimized')

    ax.zaxis.set_major_formatter(plt.NullFormatter())
    ax.set_xlabel(name1)
    ax.set_ylabel(name2)
    ax.set_zlabel('consistency [a.u.]')
    ax.set_title(f"ECC Dependency: {name1} vs {name2}")
    ax.legend()
    return fig

def fig_to_svg_str(fig):
    buf = io.StringIO()
    fig.savefig(buf, format='svg', bbox_inches='tight')
    plt.close(fig)
    content = buf.getvalue()
    svg_start = content.find('<svg')
    if svg_start != -1:
        return content[svg_start:]
    return content

def align_trajectories(Ps_opt, Ps_init):
    """
    Finds a 3D rigid transformation T_align (rotation R, translation t) such that:
       T_align @ C_opt_i approx C_init_i
       R @ v_opt_i approx v_init_i
    and returns aligned optimized projection matrices:
       P_opt_aligned_i = P_opt_i @ T_align_inv
    """
    Cs_opt = []
    Cs_init = []
    vs_opt = []
    vs_init = []
    
    from ProjectiveGeometry23.utils import dehomogenize
    
    for P_opt, P_init in zip(Ps_opt, Ps_init):
        # Camera centers
        C_opt = dehomogenize(P_opt.getCenterOfProjection()).flatten()
        C_init = dehomogenize(P_init.getCenterOfProjection()).flatten()
        Cs_opt.append(C_opt)
        Cs_init.append(C_init)
        
        # Principal rays
        v_opt = P_opt.getPrincipalRay().flatten()
        v_init = P_init.getPrincipalRay().flatten()
        vs_opt.append(v_opt)
        vs_init.append(v_init)
        
    Cs_opt = np.array(Cs_opt)
    Cs_init = np.array(Cs_init)
    vs_opt = np.array(vs_opt)
    vs_init = np.array(vs_init)
    
    # Calculate characteristic length scale for scaling direction vectors
    centroid_opt = np.mean(Cs_opt, axis=0)
    characteristic_length = np.mean(np.linalg.norm(Cs_opt - centroid_opt, axis=1))
    if characteristic_length < 1e-5:
        characteristic_length = 100.0
        
    # Construct 2 * N points: centers, and centers + characteristic_length * principal_rays
    pts_opt = np.vstack([Cs_opt, Cs_opt + characteristic_length * vs_opt])
    pts_init = np.vstack([Cs_init, Cs_init + characteristic_length * vs_init])
    
    # Center the datasets
    mean_opt = np.mean(pts_opt, axis=0)
    mean_init = np.mean(pts_init, axis=0)
    
    pts_opt_centered = pts_opt - mean_opt
    pts_init_centered = pts_init - mean_init
    
    # Compute covariance matrix H
    H = pts_opt_centered.T @ pts_init_centered
    
    # SVD
    U, S, Vt = np.linalg.svd(H)
    
    # Optimal rotation R
    R = Vt.T @ U.T
    
    # Check for reflection
    if np.linalg.det(R) < 0:
        Vt_mod = Vt.copy()
        Vt_mod[2, :] *= -1
        R = Vt_mod.T @ U.T
        
    # Translation t
    t = mean_init - R @ mean_opt
    
    # Construct 4x4 rigid transformation matrix T
    T_align = np.eye(4)
    T_align[:3, :3] = R
    T_align[:3, 3] = t
    
    T_align_inv = np.linalg.inv(T_align)
    
    # Align optimized projection matrices
    Ps_aligned = []
    for P_opt in Ps_opt:
        P_aligned = ProjectionMatrix(
            P_opt.P @ T_align_inv,
            image_size=P_opt.image_size,
            pixel_spacing=P_opt.pixel_spacing
        )
        Ps_aligned.append(P_aligned)
        
    return Ps_aligned, T_align

def compute_reconstruction_metrics(initial_data, opt_data):
    """
    Computes sharpness (mean gradient magnitude), Shannon entropy, and variance of
    both reconstructions (excluding zero/background pixels), and makes a rating.
    """
    initial_data = np.asarray(initial_data, dtype=np.float32)
    opt_data = np.asarray(opt_data, dtype=np.float32)
    
    def compute_sharpness(vol):
        Ny, Nx, Nz = vol.shape
        total_sum = 0.0
        total_count = 0
        for y in range(Ny):
            if y == 0:
                dy = vol[1] - vol[0]
            elif y == Ny - 1:
                dy = vol[Ny - 1] - vol[Ny - 2]
            else:
                dy = (vol[y + 1] - vol[y - 1]) / 2.0
                
            dx, dz = np.gradient(vol[y])
            grad_mag = np.sqrt(dx**2 + dy**2 + dz**2)
            
            non_zero = vol[y] > 1e-4
            total_sum += float(np.sum(grad_mag[non_zero]))
            total_count += int(np.sum(non_zero))
        return total_sum / total_count if total_count > 0 else 0.0
        
    def compute_entropy(vol):
        Ny = vol.shape[0]
        # 1. Get global bounds of active voxels
        g_min, g_max = float('inf'), float('-inf')
        has_active = False
        for y in range(Ny):
            active_mask = vol[y] > 1e-4
            if np.any(active_mask):
                has_active = True
                s_min, s_max = np.min(vol[y][active_mask]), np.max(vol[y][active_mask])
                if s_min < g_min: g_min = s_min
                if s_max > g_max: g_max = s_max
        
        if not has_active or g_max <= g_min:
            return 0.0
            
        # 2. Accumulate histogram counts
        counts = np.zeros(256, dtype=np.int64)
        for y in range(Ny):
            active_mask = vol[y] > 1e-4
            if np.any(active_mask):
                c, _ = np.histogram(vol[y][active_mask], bins=256, range=(g_min, g_max))
                counts += c
                
        probs = counts / np.sum(counts)
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))
        
    def compute_variance(vol):
        Ny = vol.shape[0]
        sum_x = 0.0
        sum_xx = 0.0
        count = 0
        for y in range(Ny):
            active_mask = vol[y] > 1e-4
            if np.any(active_mask):
                active = vol[y][active_mask]
                sum_x += float(np.sum(active))
                sum_xx += float(np.sum(active**2))
                count += len(active)
        if count == 0:
            return 0.0
        mean = sum_x / count
        return float((sum_xx / count) - (mean ** 2))

    sharpness_init = compute_sharpness(initial_data)
    sharpness_opt = compute_sharpness(opt_data)
    
    entropy_init = compute_entropy(initial_data)
    entropy_opt = compute_entropy(opt_data)
    
    var_init = compute_variance(initial_data)
    var_opt = compute_variance(opt_data)
    
    rel_sharp = (sharpness_opt - sharpness_init) / sharpness_init if sharpness_init > 0 else 0.0
    rel_entropy = (entropy_opt - entropy_init) / entropy_init if entropy_init > 0 else 0.0
    rel_var = (var_opt - var_init) / var_init if var_init > 0 else 0.0
    
    if rel_sharp >= 0.01 and rel_entropy <= -0.005:
        rating = "probably better"
        rating_color = "#2ca02c"
    elif rel_sharp < -0.01 or rel_entropy > 0.005:
        rating = "probably worse"
        rating_color = "#d62728"
    else:
        rating = "unclear"
        rating_color = "#ff7f0e"
        
    return {
        "sharpness_init": sharpness_init,
        "sharpness_opt": sharpness_opt,
        "rel_sharpness": rel_sharp,
        "entropy_init": entropy_init,
        "entropy_opt": entropy_opt,
        "rel_entropy": rel_entropy,
        "var_init": var_init,
        "var_opt": var_opt,
        "rel_var": rel_var,
        "rating": rating,
        "rating_color": rating_color
    }

def plot_histograms(initial_data, opt_data):
    """
    Renders SVG histogram comparison slice-by-slice to minimize memory footprint.
    """
    initial_data = np.asarray(initial_data, dtype=np.float32)
    opt_data = np.asarray(opt_data, dtype=np.float32)
    
    def get_hist_counts_edges(vol):
        Ny = vol.shape[0]
        g_min, g_max = float('inf'), float('-inf')
        has_active = False
        for y in range(Ny):
            active_mask = vol[y] > 1e-4
            if np.any(active_mask):
                has_active = True
                s_min, s_max = np.min(vol[y][active_mask]), np.max(vol[y][active_mask])
                if s_min < g_min: g_min = s_min
                if s_max > g_max: g_max = s_max
                
        if not has_active or g_max <= g_min:
            return np.zeros(100, dtype=np.int64), np.linspace(0, 1, 101)
            
        counts = np.zeros(100, dtype=np.int64)
        for y in range(Ny):
            active_mask = vol[y] > 1e-4
            if np.any(active_mask):
                c, edges = np.histogram(vol[y][active_mask], bins=100, range=(g_min, g_max))
                counts += c
        return counts, edges

    init_counts, init_edges = get_hist_counts_edges(initial_data)
    opt_counts, opt_edges = get_hist_counts_edges(opt_data)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Calculate density (normalized probability density)
    init_width = init_edges[1] - init_edges[0]
    init_density = init_counts / (np.sum(init_counts) * init_width) if np.sum(init_counts) > 0 else np.zeros_like(init_counts, dtype=np.float32)
    
    opt_width = opt_edges[1] - opt_edges[0]
    opt_density = opt_counts / (np.sum(opt_counts) * opt_width) if np.sum(opt_counts) > 0 else np.zeros_like(opt_counts, dtype=np.float32)
    
    centers_init = (init_edges[:-1] + init_edges[1:]) / 2.0
    centers_opt = (opt_edges[:-1] + opt_edges[1:]) / 2.0
    
    ax.fill_between(centers_init, init_density, step="mid", alpha=0.5, label='Initial (Misaligned)', color='#d62728')
    ax.fill_between(centers_opt, opt_density, step="mid", alpha=0.5, label='Optimized (Corrected)', color='#2ca02c')
    
    ax.set_xlabel('Voxel Intensity')
    ax.set_ylabel('Density')
    ax.set_title('Voxel Intensity Histogram Comparison')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    
    buf = io.StringIO()
    fig.savefig(buf, format='svg', bbox_inches='tight')
    plt.close(fig)
    content = buf.getvalue()
    svg_start = content.find('<svg')
    if svg_start != -1:
        return content[svg_start:]
    return content

def generate_html_report(calib, filepath, config, result, active_param_sweeps, sweeps_2d_html_list, terminal_log="", b64_slices=None, recon_metrics=None, histogram_svg=None):
    plt.figure(figsize=(8, 3))
    plt.plot(calib.iteration_cost_history, color='#1f77b4', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('ECC Cost')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    buf = io.StringIO()
    plt.savefig(buf, format='svg', bbox_inches='tight')
    plt.close()
    cost_svg = buf.getvalue()
    cost_svg = cost_svg[cost_svg.find('<svg'):]

    viewer_html = ""
    if b64_slices:
        img_tags = []
        for axis in ["x", "y", "z"]:
            for view in ["misaligned", "optimized"]:
                b64_str = b64_slices[view][axis]
                img_tags.append(f'<img class="img-{view}-{axis}" src="{b64_str}">')

        viewer_html = f"""
        <h2>Reconstruction Slices Comparison</h2>
        <div class="viewer">
            {"".join(img_tags)}
        </div>
        """

    grouped_sweeps_html = ""
    for sweep in active_param_sweeps:
        param_name = sweep["name"]
        svg_before = sweep["svg_before"]
        svg_after = sweep["svg_after"]
        
        grouped_sweeps_html += f"""
        <hr style="border: 0; border-top: 1px solid #ccc; margin: 30px 0;">
        <h3 style="text-align: center; font-family: sans-serif; color: #333;">{param_name}</h3>
        <div style="display: flex; flex-direction: row; justify-content: center; align-items: center; gap: 20px; flex-wrap: wrap;">
            <div style="text-align: center;">
                <h4 style="margin-bottom: 5px; color: #666;">Before (Initial)</h4>
                <div class="sweep-card">
                    {svg_before}
                </div>
            </div>
            <div style="text-align: center;">
                <h4 style="margin-bottom: 5px; color: #666;">After (Optimized)</h4>
                <div class="sweep-card">
                    {svg_after}
                </div>
            </div>
        </div>
        """

    sweeps_2d_html = ""
    if sweeps_2d_html_list:
        sweeps_2d_html = "<h2>Parameter Dependency Sweeps (2D)</h2><div class=\"sweeps-grid\">"
        for svg_2d in sweeps_2d_html_list:
            sweeps_2d_html += svg_2d
        sweeps_2d_html += "</div>"

    terminal_html = ""
    if terminal_log:
        terminal_html = f"""
        <h2>Terminal Output</h2>
        <pre>{terminal_log}</pre>
        """

    recon_assessment_html = ""
    if recon_metrics:
        rating_upper = recon_metrics["rating"].upper()
        rating_color = recon_metrics["rating_color"]
        
        recon_assessment_html = f"""
        <h2>Reconstruction Quality Assessment</h2>
        <div style="background-color: #f9f9f9; padding: 20px; border-left: 6px solid {rating_color}; border-radius: 4px; margin-bottom: 20px; font-family: sans-serif;">
            <div style="font-size: 20px; font-weight: bold; margin-bottom: 15px;">
                Overall Rating: <span style="color: {rating_color}; font-weight: 800;">{rating_upper}</span>
            </div>
            <table style="width: 100%; max-width: 600px; text-align: left; margin-bottom: 20px; border-collapse: collapse;">
                <thead>
                    <tr style="background-color: #eee;">
                        <th style="padding: 8px; border: 1px solid #ccc;">Metric</th>
                        <th style="padding: 8px; border: 1px solid #ccc; text-align: right;">Initial (Misaligned)</th>
                        <th style="padding: 8px; border: 1px solid #ccc; text-align: right;">Optimized (Corrected)</th>
                        <th style="padding: 8px; border: 1px solid #ccc; text-align: right;">Relative Change</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #ccc;"><strong>Sharpness (Mean Gradient)</strong></td>
                        <td style="padding: 8px; border: 1px solid #ccc; text-align: right;">{recon_metrics["sharpness_init"]:.6f}</td>
                        <td style="padding: 8px; border: 1px solid #ccc; text-align: right;">{recon_metrics["sharpness_opt"]:.6f}</td>
                        <td style="padding: 8px; border: 1px solid #ccc; text-align: right; color: {"#2ca02c" if recon_metrics["rel_sharpness"] > 0 else "#d62728"}; font-weight: bold;">{recon_metrics["rel_sharpness"]*100:+.2f}%</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #ccc;"><strong>Entropy (Shannon)</strong></td>
                        <td style="padding: 8px; border: 1px solid #ccc; text-align: right;">{recon_metrics["entropy_init"]:.6f}</td>
                        <td style="padding: 8px; border: 1px solid #ccc; text-align: right;">{recon_metrics["entropy_opt"]:.6f}</td>
                        <td style="padding: 8px; border: 1px solid #ccc; text-align: right; color: {"#2ca02c" if recon_metrics["rel_entropy"] < 0 else "#d62728"}; font-weight: bold;">{recon_metrics["rel_entropy"]*100:+.2f}%</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border: 1px solid #ccc;"><strong>Variance</strong></td>
                        <td style="padding: 8px; border: 1px solid #ccc; text-align: right;">{recon_metrics["var_init"]:.6f}</td>
                        <td style="padding: 8px; border: 1px solid #ccc; text-align: right;">{recon_metrics["var_opt"]:.6f}</td>
                        <td style="padding: 8px; border: 1px solid #ccc; text-align: right; color: {"#2ca02c" if recon_metrics["rel_var"] > 0 else "#d62728"}; font-weight: bold;">{recon_metrics["rel_var"]*100:+.2f}%</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """
        
    histogram_html = ""
    if histogram_svg:
        histogram_html = f"""
        <h2>Reconstruction Histogram Comparison</h2>
        <div style="text-align: center; margin-bottom: 30px;">
            {histogram_svg}
        </div>
        """

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Optimization Report</title>
    <style>
        table {{ border-collapse: collapse; }}
        th, td {{ border: 1px solid #ccc; text-align: center; }}
        .viewer {{ width: 512px; height: 512px; background: #000; position: relative; }}
        .viewer img {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: none; object-fit: contain; }}
        .controls {{ position: sticky; top: 0; background: white; padding: 5px 0; z-index: 10; }}
        .sweeps-grid {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px; }}
        .sweep-card {{ border: 1px solid #ccc; padding: 5px; text-align: center; }}
        .sweep-card svg {{ max-width: 360px; height: auto; }}
        pre {{ background: #f4f4f4; padding: 8px; border: 1px solid #ccc; overflow-x: auto; }}
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
            <option value="optimized" selected>Corrected (Optimized)</option>
            <option value="misaligned">Initial (Misaligned)</option>
        </select>

        <label for="axis-select">Axis:</label>
        <select id="axis-select" onchange="updateView()">
            <option value="x" selected>Sagittal (X)</option>
            <option value="y">Coronal (Y)</option>
            <option value="z">Axial (Z)</option>
        </select>
    </div>

    <div class="content">
        <h1>X-Ray Epipolar Consistency Optimization Report</h1>
        
        <h2>Optimization Summary</h2>
        <ul>
            <li><strong>Total Stages:</strong> {len(result.get("stages", []))}</li>
            <li><strong>Initial Cost:</strong> {result.get("cost_history", [0.0])[0]:.4e}</li>
            <li><strong>Final Cost:</strong> {result.get("cost_history", [0.0])[-1]:.4e}</li>
            <li><strong>Execution Time:</strong> {result.get("optimization_time_sec", 0.0):.2f} seconds</li>
        </ul>

        {recon_assessment_html}

        {viewer_html}

        {histogram_html}

        <h2>Cost History</h2>
        {cost_svg}

        <h2>Configuration</h2>
        <pre>{json.dumps(config, indent=2)}</pre>

        <h2>Parameter Sweeps (1D)</h2>
        {grouped_sweeps_html}

        {sweeps_2d_html}

        {terminal_html}
    </div>
</body>
</html>
"""
    Path(filepath).write_text(html, encoding='utf-8')

def main(config_path):
    # Capture standard output for the report
    captured_stdout = io.StringIO()
    original_stdout = sys.stdout
    class Tee:
        def __init__(self, s1, s2):
            self.s1 = s1
            self.s2 = s2
        def write(self, data):
            self.s1.write(data)
            self.s2.write(data)
        def flush(self):
            self.s1.flush()
            self.s2.flush()
        def isatty(self):
            return hasattr(self.s1, "isatty") and self.s1.isatty()
    sys.stdout = Tee(original_stdout, captured_stdout)

    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    
    input_data_path = config["input_data"]
    if not os.path.isabs(input_data_path):
        input_data_path = os.path.normpath(os.path.join(config_path.parent, input_data_path))
    scan_path = Path(input_data_path).resolve()
    scan = json.loads(scan_path.read_text())

    # Copy voxel_dimensions, model_matrix, filter_type, and output_file from input_data config if not present (only if reconstruction_config is specified)
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
    
    # Clean up old preview if it exists
    preview_dir = output_dir / "preview"
    if preview_dir.exists():
        import shutil
        try:
            shutil.rmtree(preview_dir)
        except Exception:
            pass

    active_param_sweeps = []
    last_stage_name = stages[-1].get("name", "Stage")

    # Starting trajectory for the last stage before optimization
    Ps_start_initial = Ps
    for idx in range(len(stages) - 1):
        Ps_start_initial = calib.parameterizations[idx].apply_to_trajectory(Ps_start_initial)

    # We instantiate a clean copy of the initial state of the last stage parameterization from stages[-1]
    last_stage_param_obj_initial = from_dict(stages[-1]["parameterization"])
    last_stage_param_obj_initial.estimateTrajectoryParameters(Ps_start_initial)

    create_report = config.get("create_report", True)

    # Initial sweeps (generated before optimization starts in-memory)
    if create_report:
        for name in tqdm(last_stage_param_obj_initial, desc=f"Initial sweeps for {last_stage_name}"):
            p_info = last_stage_param_obj_initial[name]
            if not p_info["opt"]:
                continue
            fig = plot_parameter_sweep(calib.scan, last_stage_param_obj_initial, name, Ps_start_initial)
            svg_before = fig_to_svg_str(fig)
            active_param_sweeps.append({
                "name": name,
                "svg_before": svg_before,
                "svg_after": None
            })

    # Run the optimization process
    result = calib.optimize()

    # After optimization, the final trajectory (state AFTER optimization) is retrieved from the scan
    Ps_final = calib.scan.get_projection_matrices()

    sweeps_2d_html_list = []
    if create_report:
        # Create a NEW parameterization object for local sweep plots around the optimized state.
        # This sweep parameterization starts at values of 0.0, acting directly on the optimized trajectory.
        sweep_param_obj_optimized = from_dict(stages[-1]["parameterization"])
        sweep_param_obj_optimized.estimateTrajectoryParameters(Ps_final)

        # Optimized sweeps (using the new sweep_param_obj_optimized acting on Ps_final, in-memory)
        for sweep in active_param_sweeps:
            name = sweep["name"]
            fig = plot_parameter_sweep(calib.scan, sweep_param_obj_optimized, name, Ps_final)
            sweep["svg_after"] = fig_to_svg_str(fig)

        # Generate 2D sweeps for pairs of active parameters in the last stage, in-memory
        active_param_names = [
            name for name in sweep_param_obj_optimized
            if sweep_param_obj_optimized[name]["opt"]
        ]
        if config.get("plot_2d_sweeps", False):
            for name1, name2 in tqdm(combinations(active_param_names, 2), desc=f"2D sweeps for {last_stage_name}"):
                short1 = name1.split(".")[-1]
                short2 = name2.split(".")[-1]
                fig = plot_2d_sweep(calib.scan, sweep_param_obj_optimized, (name1, name2), Ps_final)
                svg_2d = fig_to_svg_str(fig)
                sweeps_2d_html_list.append(f"""
                <div class="sweep-card">
                    <h4>{short1} vs {short2}</h4>
                    {svg_2d}
                </div>""")

    # Restore stdout and clean the captured logs for the report
    sys.stdout = original_stdout

    raw_output = captured_stdout.getvalue()
    
    # Escape HTML special characters
    import html
    html_output = html.escape(raw_output)
    
    # Convert ANSI background truecolor codes: \x1b[48;2;R;G;Bm -> <span style="background-color: rgb(R,G,B);">
    html_output = re.sub(
        r'\x1b\[48;2;(\d+);(\d+);(\d+)m',
        r'<span style="background-color: rgb(\1,\2,\3);">',
        html_output
    )
    
    # Convert ANSI reset (\x1b[0m) to </span>
    html_output = html_output.replace('\x1b[0m', '</span>')
    
    # Clean up any remaining ANSI codes
    html_output = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', html_output)
    
    cleaned_lines = []
    for line in html_output.splitlines():
        if '\r' in line:
            line = line.split('\r')[-1]
        cleaned_lines.append(line)
    terminal_log = '\n'.join(cleaned_lines)

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

    # Trigger reconstructions if configured
    run_reconstruction = config.get("run_reconstruction")
    if run_reconstruction is None:
        # Backward compatibility fallback: check if reconstruction_config is specified
        run_reconstruction = "reconstruction_config" in config
        
    if run_reconstruction and reconstruct_configs is not None:
        initial_cfg, optimized_cfg = reconstruct_configs
        import subprocess
        
        # Reconstruct to the relocated paths
        recon_misaligned_path = orig_output_path_abs
        recon_optimized_path = orig_dir / f"{orig_base}_{output_dir.name}{orig_ext}"
        
        def run_subprocess_tee(cmd):
            import subprocess
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
            rc = process.poll()
            if rc != 0:
                raise subprocess.CalledProcessError(rc, cmd)

        if not recon_misaligned_path.exists():
            print(f"Starting initial scan reconstruction (since it does not exist): {recon_misaligned_path}")
            try:
                run_subprocess_tee(["reconstruct", str(initial_cfg)])
            except Exception as e:
                print(f"Warning: Failed to run initial reconstruction: {e}")
            
        print(f"Starting full scan reconstruction: {recon_optimized_path}")
        try:
            run_subprocess_tee(["reconstruct", str(optimized_cfg)])
        except Exception as e:
            print(f"Warning: Failed to run optimized reconstruction: {e}")


    # Extract central slices and compute image quality metrics if both reconstructions exist
    b64_slices = None
    recon_metrics = None
    histogram_svg = None
    if create_report and run_reconstruction and reconstruct_configs is not None:
        recon_misaligned_path = orig_output_path_abs
        recon_optimized_path = orig_dir / f"{orig_base}_{output_dir.name}{orig_ext}"
        
        if recon_misaligned_path.exists() and recon_optimized_path.exists():
            print("Extracting central slices and computing reconstruction quality metrics...")
            try:
                import nrrd
                from PIL import Image
                import base64

                initial_data, _ = nrrd.read(str(recon_misaligned_path))
                opt_data, _ = nrrd.read(str(recon_optimized_path))
                
                if opt_data.nbytes > 1073741824:
                    print("Volume size exceeds 1 GB. Skipping quality metrics and slice extraction to save time.")
                else:
                    recon_metrics = compute_reconstruction_metrics(initial_data, opt_data)
                    histogram_svg = plot_histograms(initial_data, opt_data)
                    
                    Nx, Ny, Nz = opt_data.shape
                    cx, cy, cz = Nx // 2, Ny // 2, Nz // 2
                    
                    slices = {
                        "misaligned": {
                            "x": initial_data[cx, :, :],
                            "y": initial_data[:, cy, :],
                            "z": initial_data[:, :, cz]
                        },
                        "optimized": {
                            "x": opt_data[cx, :, :],
                            "y": opt_data[:, cy, :],
                            "z": opt_data[:, :, cz]
                        }
                    }
                    
                    def to_b64(slice_data):
                        slice_data = np.squeeze(slice_data)
                        s_min, s_max = float(slice_data.min()), float(slice_data.max())
                        if s_max > s_min:
                            norm = (slice_data - s_min) / (s_max - s_min) * 255.0
                        else:
                            norm = np.zeros_like(slice_data)
                        img = Image.fromarray(norm.astype(np.uint8), mode='L')
                        buf = io.BytesIO()
                        img.save(buf, format='PNG', optimize=True, compress_level=9)
                        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

                    b64_slices = {}
                    for cat in ["misaligned", "optimized"]:
                        b64_slices[cat] = {}
                        for axis in ["x", "y", "z"]:
                            b64_slices[cat][axis] = to_b64(slices[cat][axis])
                
                # Free memory immediately
                del initial_data
                del opt_data
                import gc
                gc.collect()
            except Exception as e:
                print(f"Warning: Failed to extract slices and compute metrics: {e}")

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

    # Generate HTML report
    if create_report:
        report_path = (output_dir / "report.html").resolve()
        generate_html_report(
            calib, report_path, config, result, active_param_sweeps, sweeps_2d_html_list, terminal_log,
            b64_slices=b64_slices, recon_metrics=recon_metrics, histogram_svg=histogram_svg
        )
        print("Report saved to:\n", report_path.as_uri())

        if recon_metrics:
            metrics_json_path = output_dir / "metrics.json"
            metrics_json_path.write_text(json.dumps(recon_metrics, indent=2))
            print(f"Saved reconstruction metrics to:\n{metrics_json_path}")



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <config.json>")
        sys.exit(1)
    main(sys.argv[1])
