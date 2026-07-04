#!/usr/bin/env python3
import json
import sys
import os
import argparse
from pathlib import Path

# Set up paths to import local packages
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/home/aaichert/Desktop/install_test/ct_recon_fdk_astra")

import numpy as np
from ProjectiveGeometry23.central_projection import ProjectionMatrix
from ct_recon_fdk_astra.fileformats import load_ompl
from xray_epipolar_consistency.parameterization import from_dict

def make_displacement_field_svg(w, J_cols, coords_nominal, active_names, title, W_img, H_img, view_idx=0, canvas_w=600, canvas_h=400):
    # w: shape (K,)
    # J_cols: shape (2MN, K)
    # coords_nominal: shape (M, N, 2)
    M, N, _ = coords_nominal.shape
    
    # Compute physical displacement field: dx = J * w
    dx_field = J_cols @ w  # Shape: (2MN,)
    dx_view = dx_field.reshape(M, N, 2)[view_idx]  # Shape: (N, 2)
    coords_view = coords_nominal[view_idx]  # Shape: (N, 2)
    
    # Layout dimensions
    padding = max(10, int(min(canvas_w, canvas_h) * 0.1))
    
    # Scale coordinates to fit canvas
    u_min, v_min = np.min(coords_view, axis=0)
    u_max, v_max = np.max(coords_view, axis=0)
    
    u_range = u_max - u_min if u_max > u_min else 1.0
    v_range = v_max - v_min if v_max > v_min else 1.0
    
    scale_x = (canvas_w - 2 * padding) / u_range
    scale_y = (canvas_h - 2 * padding) / v_range
    scale_coords = min(scale_x, scale_y)
    
    # Normalize coords
    x_svg = padding + (coords_view[:, 0] - u_min) * scale_coords
    y_svg = padding + (coords_view[:, 1] - v_min) * scale_coords
    
    # Scale displacements
    dx_mags = np.linalg.norm(dx_view, axis=-1)
    max_dx = np.max(dx_mags)
    scale_dx = (min(canvas_w, canvas_h) * 0.08) / max_dx if max_dx > 1e-6 else 1.0
    
    dx_svg = dx_view[:, 0] * scale_dx
    dy_svg = dx_view[:, 1] * scale_dx
    
    # Sample subset of points to keep SVG size readable
    sample_stride = max(1, N // 80)
    indices = range(0, N, sample_stride)
    
    # Draw elements
    svg_elements = []
    svg_elements.append(f'<rect width="{canvas_w}" height="{canvas_h}" fill="#f9f9f9" stroke="#ccc" />')
    svg_elements.append(f'<text x="15" y="20" font-family="sans-serif" font-size="10" font-weight="bold" fill="#333">{title}</text>')
    
    for idx in indices:
        x1 = x_svg[idx]
        y1 = y_svg[idx]
        x2 = x1 + dx_svg[idx]
        y2 = y1 + dy_svg[idx]
        
        # Circle at nominal
        svg_elements.append(f'<circle cx="{x1:.1f}" cy="{y1:.1f}" r="1.5" fill="#888" />')
        # Arrow line
        svg_elements.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="blue" stroke-width="1.0" />')
        # Simple arrowhead dot
        if dx_mags[idx] > 1e-4:
            svg_elements.append(f'<circle cx="{x2:.1f}" cy="{y2:.1f}" r="1.2" fill="red" />')
            
    return f'<svg width="{canvas_w}" height="{canvas_h}" xmlns="http://www.w3.org/2000/svg">{"".join(svg_elements)}</svg>'

def main():
    parser = argparse.ArgumentParser(description="Geometric Calibration Optimization Advisor")
    parser.add_argument("reconstruction_json", type=str, help="Path to reconstruction.json")
    parser.add_argument("stage_json", type=str, help="Path to stage JSON file")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save output files")
    parser.add_argument("--d_max", type=float, default=10.0, help="Desired maximum detector motion scale in pixels")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of Monte-Carlo points")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--delta_p", type=float, default=1e-5, help="Small perturbation for local derivatives")
    parser.add_argument("--max_epipolar_views", type=int, default=100, help="Max views for epipolar analysis")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Input Files
    recon_path = Path(args.reconstruction_json).resolve()
    recon_config = json.loads(recon_path.read_text())
    
    stage_path = Path(args.stage_json).resolve()
    stage_config = json.loads(stage_path.read_text())

    # Resolve OMPL path relative to reconstruction.json's folder
    ompl_path = recon_config["ompl_file"]
    if not os.path.isabs(ompl_path):
        ompl_path = os.path.normpath(os.path.join(recon_path.parent, ompl_path))
    ompl_path = Path(ompl_path).resolve()

    # Load trajectories
    Ps_nominal = load_ompl(ompl_path)
    M = len(Ps_nominal)
    N = args.num_samples

    # Get detector image size
    W_img = 400
    H_img = 300
    if "image_size" in recon_config and recon_config["image_size"] is not None:
        W_img, H_img = recon_config["image_size"]
    elif len(Ps_nominal) > 0:
        W_img, H_img = Ps_nominal[0].image_size

    # Load and configure parameterization
    parameterization = from_dict(stage_config["parameterization"])
    parameterization.estimateTrajectoryParameters(Ps_nominal)

    # 2. Monte-Carlo Point Sampling
    voxel_dimensions = np.array(recon_config["voxel_dimensions"])
    model_matrix = np.array(recon_config["model_matrix"])

    rng = np.random.default_rng(args.seed)
    voxels = rng.uniform(0.0, 1.0, size=(N, 3)) * voxel_dimensions
    voxels_hom = np.hstack((voxels, np.ones((N, 1))))
    world_points = (model_matrix @ voxels_hom.T).T  # Shape: (N, 4)

    # 3. Nominal Projections
    coords_nominal = np.zeros((M, N, 2))
    for i, P in enumerate(Ps_nominal):
        x_proj = (P.P @ world_points.T).T
        coords_nominal[i, :, 0] = x_proj[:, 0] / x_proj[:, 2]
        coords_nominal[i, :, 1] = x_proj[:, 1] / x_proj[:, 2]

    # 4. Parameter Perturbations and Jacobians
    p_nom = parameterization.get_parameter_vector()
    param_info = parameterization.get_names_and_docstr()
    active_names = list(param_info.keys())
    bounds = parameterization.get_bounds()
    K = len(active_names)

    if K == 0:
        raise ValueError("No active parameters configured in the stage file.")

    delta_x_all = {}
    J_cols = np.zeros((2 * M * N, K))

    for j, name in enumerate(active_names):
        # Local Perturbation
        p_perturbed = p_nom.copy()
        p_perturbed[j] += args.delta_p
        parameterization.set_parameter_vector(p_perturbed)
        Ps_perturbed = parameterization.apply_to_trajectory(Ps_nominal)
        parameterization.set_parameter_vector(p_nom)  # Reset

        # Project perturbed points
        coords_perturbed = np.zeros((M, N, 2))
        for i, P in enumerate(Ps_perturbed):
            x_proj = (P.P @ world_points.T).T
            coords_perturbed[i, :, 0] = x_proj[:, 0] / x_proj[:, 2]
            coords_perturbed[i, :, 1] = x_proj[:, 1] / x_proj[:, 2]

        dx = coords_perturbed - coords_nominal
        delta_x_all[name] = dx
        J_cols[:, j] = (dx / args.delta_p).flatten()

    # 5. Striding for Epipolar Analysis
    if M > args.max_epipolar_views:
        stride = int(np.ceil(M / args.max_epipolar_views))
        selected_views = list(range(0, M, stride))
    else:
        selected_views = list(range(M))

    M_sel = len(selected_views)
    num_pairs = M_sel * (M_sel - 1)

    # Precompute Fundamental Matrices for selected pairs
    Fs = {}
    for i in selected_views:
        for j in selected_views:
            if i != j:
                Fs[(i, j)] = Ps_nominal[i].computeFundamentalMatrix(Ps_nominal[j])

    # Convert nominal coordinates to homogeneous form (M, N, 3)
    coords_nominal_hom = np.zeros((M, N, 3))
    coords_nominal_hom[:, :, :2] = coords_nominal
    coords_nominal_hom[:, :, 2] = 1.0

    J_perp = np.zeros((num_pairs * N, K))
    row_idx = 0
    
    for i in selected_views:
        for j in selected_views:
            if i == j:
                continue

            L = Fs[(i, j)] @ coords_nominal_hom[i].T  # Shape: (3, N)
            a = L[0, :]
            b = L[1, :]
            norm = np.sqrt(a**2 + b**2)
            norm = np.where(norm < 1e-12, 1e-12, norm)

            n_u = a / norm
            n_v = b / norm

            for col_idx, name in enumerate(active_names):
                dx = delta_x_all[name]
                dx_j = dx[j]
                du = dx_j[:, 0]
                dv = dx_j[:, 1]
                scalar_perp = du * n_u + dv * n_v
                J_perp[row_idx:row_idx+N, col_idx] = scalar_perp / args.delta_p

            row_idx += N

    # 6. SVD and Gram Matrix calculations
    U_full, S_full, Vt_full = np.linalg.svd(J_cols, full_matrices=False)
    eigvals_full = S_full**2

    U_perp, S_perp, Vt_perp = np.linalg.svd(J_perp, full_matrices=False)
    eigvals_perp = S_perp**2

    # Pairwise correlation & explainability
    def compute_explainability(V):
        R, K = V.shape
        correlations = np.zeros((K, K))
        explainability = np.zeros((K, K))
        norms = np.linalg.norm(V, axis=0)
        norms_safe = np.where(norms < 1e-12, 1e-12, norms)
        for i in range(K):
            for j in range(K):
                dot_prod = np.dot(V[:, i], V[:, j])
                correlations[i, j] = float(dot_prod / (norms_safe[i] * norms_safe[j]))
                explainability[i, j] = float(correlations[i, j]**2)
        return correlations, explainability

    corr_full, exp_full = compute_explainability(J_cols)
    corr_perp, exp_perp = compute_explainability(J_perp)

    # 7. Local Sensitivity & Recommended Ranges
    # We define sensitivity as the average displacement magnitude over all points and views:
    # RMS(J_i) = sqrt( mean( J_u^2 + J_v^2 ) )
    # Recommended range Delta p_i = d_max / RMS(J_i)
    sensitivity = {}
    range_sensitivity = {}
    range_physical = {}
    recommended_ranges = {}
    
    for idx, name in enumerate(active_names):
        dx_sel = delta_x_all[name][selected_views] / args.delta_p
        rms_val = float(np.sqrt(np.mean(np.sum(dx_sel**2, axis=-1))))
        sensitivity[name] = rms_val
        
        # Range based on sensitivity
        r_sens = args.d_max / rms_val if rms_val > 1e-12 else float('inf')
        range_sensitivity[name] = r_sens
        
        # Range based on physical bounds
        p_min, p_max = bounds[idx]
        r_phys = (p_max - p_min) / 2.0
        range_physical[name] = r_phys
        
        # Recommended range
        recommended_ranges[name] = min(r_phys, r_sens)

    # 8. Sloppy Direction and Coupling Identification
    # A parameter participates in a sloppy direction if its coefficient in any eigenvector 
    # with eigenvalue ratio < 1e-3 is > 0.1 in absolute value.
    max_eig_full = eigvals_full[0]
    sloppy_eigenvectors = []
    param_couplings = []

    for idx, val in enumerate(eigvals_full):
        if val / max_eig_full < 1e-3:
            v = Vt_full[idx]
            coupled = [active_names[i] for i, coef in enumerate(v) if abs(coef) > 0.1]
            if len(coupled) > 1:
                param_couplings.append({
                    "eigenvalue": float(val),
                    "ratio": float(val / max_eig_full),
                    "parameters": coupled,
                    "expression": " ".join([f"{'+' if coef >= 0 else '-'}{abs(coef):.3f}*{active_names[i]}" for i, coef in enumerate(v) if abs(coef) > 0.1])
                })

    # 9. Automatic Group Construction (Compatibility Graph)
    # Connect nodes i, j if:
    # - Correlation > 0.95
    # - Explainability > 0.95
    # - They participate in the same sloppy direction
    adj = {name: set() for name in active_names}
    for i in range(K):
        for j in range(i + 1, K):
            p1 = active_names[i]
            p2 = active_names[j]
            # Correlation / Explainability check
            if abs(corr_full[i, j]) > 0.95 or exp_full[i, j] > 0.95:
                adj[p1].add(p2)
                adj[p2].add(p1)
                
            # Sloppy direction check
            for cp in param_couplings:
                if p1 in cp["parameters"] and p2 in cp["parameters"]:
                    adj[p1].add(p2)
                    adj[p2].add(p1)

    # Connected Components (Groups)
    visited = {name: False for name in active_names}
    groups = []
    for name in active_names:
        if not visited[name]:
            comp = []
            queue = [name]
            visited[name] = True
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for neighbor in adj[curr]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            groups.append(comp)

    # 10. Greedy Parameter Selection
    # Sort active parameters by their sensitivity (or observability ratio if available)
    sorted_candidates = sorted(active_names, key=lambda x: sensitivity[x], reverse=True)
    
    selected_full = []
    for p in sorted_candidates:
        p_idx = active_names.index(p)
        # Check correlation with already selected
        too_correlated = False
        for s in selected_full:
            s_idx = active_names.index(s)
            if abs(corr_full[p_idx, s_idx]) > 0.95:
                too_correlated = True
                break
        if too_correlated:
            continue
            
        test_subset = selected_full + [p]
        test_indices = [active_names.index(x) for x in test_subset]
        J_subset = J_cols[:, test_indices]
        
        _, S_sub, _ = np.linalg.svd(J_subset, full_matrices=False)
        cond = S_sub[0] / S_sub[-1] if S_sub[-1] > 1e-12 else float('inf')
        
        if len(selected_full) == 0 or cond < 1000.0:
            selected_full.append(p)

    # Do the same for epipolar Jacobian
    selected_perp = []
    for p in sorted_candidates:
        p_idx = active_names.index(p)
        too_correlated = False
        for s in selected_perp:
            s_idx = active_names.index(s)
            if abs(corr_perp[p_idx, s_idx]) > 0.95:
                too_correlated = True
                break
        if too_correlated:
            continue
            
        test_subset = selected_perp + [p]
        test_indices = [active_names.index(x) for x in test_subset]
        J_subset = J_perp[:, test_indices]
        
        _, S_sub, _ = np.linalg.svd(J_subset, full_matrices=False)
        cond = S_sub[0] / S_sub[-1] if S_sub[-1] > 1e-12 else float('inf')
        
        if len(selected_perp) == 0 or cond < 1000.0:
            selected_perp.append(p)

    # 11. Range Refinement for selected subset
    refined_ranges = recommended_ranges.copy()
    for i, name1 in enumerate(selected_full):
        idx1 = active_names.index(name1)
        for j, name2 in enumerate(selected_full):
            if i >= j:
                continue
            idx2 = active_names.index(name2)
            c_val = abs(corr_full[idx1, idx2])
            if c_val > 0.5:
                # Moderate coupling: reduce range
                factor = 1.0 - (c_val - 0.5) / 0.5 * 0.5  # Scales range down to 50% when correlation is 1.0
                refined_ranges[name1] *= factor
                refined_ranges[name2] *= factor

    # 12. Evaluate final subsets and compute scoring
    def evaluate_subset(subset, J_mat):
        if not subset:
            return float('inf'), 0, 0.0
        indices = [active_names.index(x) for x in subset]
        J_sub = J_mat[:, indices]
        _, S_sub, _ = np.linalg.svd(J_sub, full_matrices=False)
        cond = S_sub[0] / S_sub[-1] if S_sub[-1] > 1e-12 else float('inf')
        rank = int(np.sum(S_sub >= S_sub[0] * 1e-3))
        return cond, rank, float(S_sub[-1])

    cond_final_full, rank_final_full, min_sv_full = evaluate_subset(selected_full, J_cols)
    cond_final_perp, rank_final_perp, min_sv_perp = evaluate_subset(selected_perp, J_perp)

    def classify_difficulty(cond, rank, K_sub):
        if cond < 10.0 and rank == K_sub:
            return "Excellent"
        elif cond < 100.0 and rank == K_sub:
            return "Good"
        elif cond < 1000.0 and rank == K_sub:
            return "Acceptable"
        elif cond < 1e4:
            return "Poor"
        else:
            return "Ill-conditioned"

    difficulty_full = classify_difficulty(cond_final_full, rank_final_full, len(selected_full))
    difficulty_perp = classify_difficulty(cond_final_perp, rank_final_perp, len(selected_perp))

    # 13. Suggested Optimization Stages
    # Categorize parameters based on name prefixes to build multi-stage strategies
    stage_groups = {
        "DetectorPosition / Shifts": [],
        "DetectorOrientation": [],
        "RotationAxis": [],
        "SourcePosition": []
    }
    
    for name in selected_full:
        if "DetectorPosition" in name or "offset" in name and "Rotation" not in name:
            stage_groups["DetectorPosition / Shifts"].append(name)
        elif "DetectorOrientation" in name:
            stage_groups["DetectorOrientation"].append(name)
        elif "RotationAxis" in name:
            stage_groups["RotationAxis"].append(name)
        else:
            stage_groups["SourcePosition"].append(name)

    suggested_stages = []
    stage_num = 1
    for key, params in stage_groups.items():
        if params:
            suggested_stages.append({
                "stage": stage_num,
                "description": f"Optimize {key}",
                "parameters": params
            })
            stage_num += 1

    # 14. Construct the complete suggested configuration JSON (based on stage_config input)
    # We will modify the active parameterization in stage_config and set recommended ranges.
    suggested_config = json.loads(json.dumps(stage_config))  # Deep copy
    suggested_config["name"] = f"Suggested Geometry Optimization (d_max={args.d_max}px)"
    
    # Traverse through parameterizations and set parameters opt & range
    if "parameterization" in suggested_config and "parameterizations" in suggested_config["parameterization"]:
        for paramz in suggested_config["parameterization"]["parameterizations"]:
            classname = paramz["classname"]
            paramz["parameters"] = {}  # Overwrite with precise recommendations
            
            # Find all nominal parameters of this class
            for p_idx, p_name in enumerate(active_names):
                if classname in p_name:
                    # Strip the classname prefix to get local parameter name
                    local_name = p_name.split(".")[-1]
                    p_val = p_nom[p_idx]
                    
                    if p_name in selected_full:
                        r_val = refined_ranges[p_name]
                        paramz["parameters"][local_name] = {
                            "opt": True,
                            "range": [float(p_val - r_val), float(p_val + r_val)]
                        }
                    else:
                        paramz["parameters"][local_name] = {
                            "opt": False
                        }

    # Save suggested stage configuration JSON
    suggested_json_path = output_dir / "suggested_optimizer_config.json"
    suggested_json_path.write_text(json.dumps(suggested_config, indent=2))
    print(f"Saved suggested optimizer config to {suggested_json_path}")

    # Save detailed advisor output JSON
    advisor_data = {
        "metadata": {
            "d_max": args.d_max,
            "num_views": M,
            "num_samples": N,
        },
        "individual_parameter_analysis": {
            name: {
                "sensitivity_rms": sensitivity[name],
                "range_sensitivity": range_sensitivity[name],
                "range_physical": range_physical[name],
                "range_recommended": recommended_ranges[name],
                "range_refined": refined_ranges[name],
                "selected_in_full": name in selected_full,
                "selected_in_epipolar": name in selected_perp,
            }
            for name in active_names
        },
        "dependency_groups": groups,
        "sloppy_directions_full": param_couplings,
        "recommended_subset_full": {
            "parameters": selected_full,
            "condition_number": cond_final_full,
            "effective_rank": rank_final_full,
            "min_singular_value": min_sv_full,
            "difficulty": difficulty_full,
        },
        "recommended_subset_epipolar": {
            "parameters": selected_perp,
            "condition_number": cond_final_perp,
            "effective_rank": rank_final_perp,
            "min_singular_value": min_sv_perp,
            "difficulty": difficulty_perp,
        },
        "suggested_stages": suggested_stages
    }
    
    advisor_json_path = output_dir / "optimization_advisor_results.json"
    advisor_json_path.write_text(json.dumps(advisor_data, indent=2))
    print(f"Saved advisor results to {advisor_json_path}")

    # 15. Create HTML summary with simple styling and highlighting
    html_output_path = output_dir / "report_optimization_advisor.html"

    # Style colors for difficulty and compatibility status
    def get_difficulty_color(difficulty):
        if difficulty in ["Excellent", "Good"]:
            return "#d4edda", "#155724"  # Light green, dark green text
        elif difficulty == "Acceptable":
            return "#fff3cd", "#856404"  # Light yellow, dark yellow text
        else:
            return "#f8d7da", "#721c24"  # Light red, dark red text

    bg_c, text_c = get_difficulty_color(difficulty_full)
    bg_p, text_p = get_difficulty_color(difficulty_perp)

    # Individual parameters table
    param_table_rows = ""
    for name in active_names:
        p_idx = active_names.index(name)
        p_min, p_max = bounds[p_idx]
        
        is_sel = name in selected_full
        status_text = "Enabled" if is_sel else "Excluded"
        status_color = "#28a745" if is_sel else "#dc3545"
        
        # Color warnings for excluded parameters that were redundant
        reason = "OK"
        if not is_sel:
            # Find if it is highly correlated with selected parameters
            reasons = []
            for s in selected_full:
                s_idx = active_names.index(s)
                if abs(corr_full[p_idx, s_idx]) > 0.95:
                    reasons.append(f"coupled with {s} (&rho;={corr_full[p_idx, s_idx]:.3f})")
            reason = ", ".join(reasons) if reasons else "weak/redundant"

        # Generate per-parameter motion SVGs (View 0 and View M//4)
        w_param = np.zeros(K)
        w_param[p_idx] = 1.0
        v0 = 0
        v90 = M // 4 if M > 4 else min(1, M - 1)
        
        svg_v0 = make_displacement_field_svg(
            w_param, J_cols, coords_nominal, active_names,
            f"View {v0} (0&deg;)", W_img, H_img, view_idx=v0, canvas_w=285, canvas_h=190
        )
        svg_v90 = make_displacement_field_svg(
            w_param, J_cols, coords_nominal, active_names,
            f"View {v90} ({int(v90 * 360 / M)}&deg;)", W_img, H_img, view_idx=v90, canvas_w=285, canvas_h=190
        )

        details_html = f"""
        <details style="margin-top: 5px; cursor: pointer;">
            <summary style="font-size: 11px; color: #2980b9;">Show Motion Field</summary>
            <div style="display: flex; gap: 10px; margin-top: 5px; border: 1px solid #eee; padding: 5px; background: #fff; border-radius: 4px;">
                <div>{svg_v0}</div>
                <div>{svg_v90}</div>
            </div>
        </details>
        """

        param_table_rows += f"""    <tr style="background-color: {'#fff' if is_sel else '#f9f9f9'};">
        <td>
            <b>{name}</b>
            {details_html}
        </td>
        <td style="color: {status_color}; font-weight: bold;">{status_text}</td>
        <td>{sensitivity[name]:.4f}</td>
        <td>{range_sensitivity[name]:.4e}</td>
        <td>{range_physical[name]:.4f}</td>
        <td><b>{refined_ranges[name]:.4e}</b></td>
        <td style="font-size: 12px; color: #555;">{reason}</td>
    </tr>"""

    # Staged optimization strategy table
    stages_rows = ""
    for stage in suggested_stages:
        stages_rows += f"""    <tr>
        <td style="text-align: center; font-weight: bold;">Stage {stage['stage']}</td>
        <td><b>{stage['description']}</b></td>
        <td><code>{', '.join(stage['parameters'])}</code></td>
    </tr>"""

    # Coupled groups list
    groups_html = ""
    for i, gp in enumerate(groups):
        if len(gp) > 1:
            groups_html += f"<li><b>Group {i+1}:</b> <code>{', '.join(gp)}</code></li>"
    if not groups_html:
        groups_html = "<p>No highly dependent parameter groups detected.</p>"
    else:
        groups_html = f"<ul>{groups_html}</ul>"

    # Sloppy direction warnings
    sloppy_html = ""
    for cp in param_couplings:
        sloppy_html += f"<li><b>Eigenvalue {cp['eigenvalue']:.3e} (Ratio {cp['ratio']:.2e}):</b> <code>{cp['expression']}</code></li>"
    if not sloppy_html:
        sloppy_html = "<p>None detected.</p>"
    else:
        sloppy_html = f"<ul>{sloppy_html}</ul>"

    html_content = f"""<!DOCTYPE html>
<html>
<head>
<title>Geometric Calibration Optimization Advisor Report</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; margin: 20px; color: #333; }}
    h1 {{ color: #2c3e50; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }}
    h2 {{ color: #2980b9; margin-top: 30px; }}
    h3 {{ color: #16a085; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
    th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
    th {{ background-color: #f2f2f2; font-weight: bold; }}
    code {{ background-color: #f8f9fa; border: 1px solid #eaecf0; padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: 13px; }}
    pre {{ background-color: #f8f9fa; border: 1px solid #eaecf0; padding: 15px; border-radius: 6px; overflow-x: auto; font-family: monospace; font-size: 13px; }}
    .badge {{ display: inline-block; padding: 6px 12px; font-size: 14px; font-weight: bold; border-radius: 20px; }}
    .card {{ border: 1px solid #ddd; border-radius: 6px; padding: 15px; margin-bottom: 20px; background-color: #fafafa; }}
</style>
</head>
<body>

<h1>Geometric Calibration Optimization Advisor Report</h1>
<p><b>Link to Identifiability Report:</b> <a href="report_identifiability.html">report_identifiability.html</a></p>

<div class="card">
    <b>d_max (Target Detector Motion Scale):</b> {args.d_max} px<br>
    <b>Evaluated Views:</b> {M_sel} / {M} views (stride {int(np.ceil(M / args.max_epipolar_views)) if M > args.max_epipolar_views else 1})<br>
    <b>Monte-Carlo Points:</b> {N} samples
</div>

<h2>Recommended Optimization Difficulty</h2>
<div style="display: flex; gap: 20px;">
    <div style="flex: 1; padding: 20px; border-radius: 6px; background-color: {bg_c}; color: {text_c};">
        <h3>Full Parameter Optimization (Subset)</h3>
        <span class="badge" style="background-color: {text_c}; color: #fff;">{difficulty_full}</span>
        <p style="margin-top: 15px;">
            <b>Condition Number:</b> {cond_final_full:.2f}<br>
            <b>Effective Rank:</b> {rank_final_full} / {len(selected_full)}<br>
            <b>Smallest Singular Value:</b> {min_sv_full:.4f}
        </p>
    </div>
    <div style="flex: 1; padding: 20px; border-radius: 6px; background-color: {bg_p}; color: {text_p};">
        <h3>Epipolar Calibration (Subset)</h3>
        <span class="badge" style="background-color: {text_p}; color: #fff;">{difficulty_perp}</span>
        <p style="margin-top: 15px;">
            <b>Condition Number:</b> {cond_final_perp:.2f}<br>
            <b>Effective Rank:</b> {rank_final_perp} / {len(selected_perp)}<br>
            <b>Smallest Singular Value:</b> {min_sv_perp:.4f}
        </p>
    </div>
</div>

<h2>Suggested Multi-Stage Optimization Strategy</h2>
<table>
    <tr>
        <th style="width: 10%; text-align: center;">Stage</th>
        <th style="width: 30%;">Description</th>
        <th>Recommended Active Parameters</th>
    </tr>
    {stages_rows}
</table>

<h2>Individual Parameter Calibration Guidelines</h2>
<table>
    <tr>
        <th>Parameter Name</th>
        <th>Status</th>
        <th>Sensitivity (RMS)</th>
        <th>Sens-Range</th>
        <th>Phys-Range</th>
        <th>Refined Range (Recommended)</th>
        <th>Exclusion Reason</th>
    </tr>
    {param_table_rows}
</table>

<h2>Detected Redundant Groups & Sloppy Combinations</h2>
<h3>Dependent Parameter Groups</h3>
{groups_html}

<h3>Sloppy Directions</h3>
{sloppy_html}

<h2>Suggested Configuration JSON</h2>
<p>You can directly save this configuration to a JSON file and use it as an optimization stage in the calibration pipeline.</p>
<pre><code>{json.dumps(suggested_config, indent=2)}</code></pre>

<p>For more detailed results, see: <a href="./suggested_optimizer_config.json">suggested_optimizer_config.json</a> and <a href="./optimization_advisor_results.json">optimization_advisor_results.json</a></p>

</body>
</html>
"""

    html_output_path.write_text(html_content)
    print(f"Saved HTML advisor report to {html_output_path}")

if __name__ == "__main__":
    main()
