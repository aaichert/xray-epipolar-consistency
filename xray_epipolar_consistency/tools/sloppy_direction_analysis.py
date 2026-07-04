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

def compute_explainability(V):
    R, K = V.shape
    correlations = np.zeros((K, K))
    explainability = np.zeros((K, K))
    multi_explainability = np.zeros(K)
    
    norms = np.linalg.norm(V, axis=0)
    norms_safe = np.where(norms < 1e-12, 1e-12, norms)
    
    for i in range(K):
        for j in range(K):
            dot_prod = np.dot(V[:, i], V[:, j])
            correlations[i, j] = float(dot_prod / (norms_safe[i] * norms_safe[j]))
            explainability[i, j] = float(correlations[i, j]**2)
            
    for i in range(K):
        if K <= 1:
            multi_explainability[i] = 0.0
            continue
        y = V[:, i]
        X = np.delete(V, i, axis=1)
        alpha, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        if len(residuals) > 0:
            res_norm_sq = float(residuals[0])
        else:
            res_norm_sq = float(np.sum((y - X @ alpha)**2))
        y_norm_sq = float(np.sum(y**2))
        if y_norm_sq > 1e-12:
            multi_explainability[i] = float(1.0 - (res_norm_sq / y_norm_sq))
        else:
            multi_explainability[i] = 0.0
            
    return correlations, explainability, multi_explainability

def make_displacement_field_svg(w, J_cols, coords_nominal, active_names, title, W_img, H_img):
    # w: shape (K,)
    # J_cols: shape (2MN, K)
    # coords_nominal: shape (M, N, 2)
    M, N, _ = coords_nominal.shape
    
    # Compute physical displacement field: dx = J * w
    dx_field = J_cols @ w  # Shape: (2MN,)
    dx_view0 = dx_field.reshape(M, N, 2)[0]  # Shape: (N, 2)
    coords_view0 = coords_nominal[0]  # Shape: (N, 2)
    
    # Layout dimensions
    canvas_w = 600
    canvas_h = 400
    padding = 40
    
    # Scale coordinates to fit canvas
    u_min, v_min = np.min(coords_view0, axis=0)
    u_max, v_max = np.max(coords_view0, axis=0)
    
    u_range = u_max - u_min if u_max > u_min else 1.0
    v_range = v_max - v_min if v_max > v_min else 1.0
    
    scale_x = (canvas_w - 2 * padding) / u_range
    scale_y = (canvas_h - 2 * padding) / v_range
    scale_coords = min(scale_x, scale_y)
    
    # Normalize coords
    x_svg = padding + (coords_view0[:, 0] - u_min) * scale_coords
    y_svg = padding + (coords_view0[:, 1] - v_min) * scale_coords
    
    # Scale displacements
    dx_mags = np.linalg.norm(dx_view0, axis=-1)
    max_dx = np.max(dx_mags)
    scale_dx = 30.0 / max_dx if max_dx > 1e-6 else 1.0
    
    dx_svg = dx_view0[:, 0] * scale_dx
    dy_svg = dx_view0[:, 1] * scale_dx
    
    # Sample subset of points to keep SVG size readable
    sample_stride = max(1, N // 120)
    indices = range(0, N, sample_stride)
    
    # Draw elements
    svg_elements = []
    svg_elements.append(f'<rect width="{canvas_w}" height="{canvas_h}" fill="#f9f9f9" stroke="#ccc" />')
    svg_elements.append(f'<text x="20" y="30" font-family="sans-serif" font-size="14" font-weight="bold" fill="#333">{title}</text>')
    
    for idx in indices:
        x1 = x_svg[idx]
        y1 = y_svg[idx]
        x2 = x1 + dx_svg[idx]
        y2 = y1 + dy_svg[idx]
        
        # Circle at nominal
        svg_elements.append(f'<circle cx="{x1:.1f}" cy="{y1:.1f}" r="2" fill="#888" />')
        # Arrow line
        svg_elements.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="blue" stroke-width="1.2" />')
        # Simple arrowhead dot
        if dx_mags[idx] > 1e-4:
            svg_elements.append(f'<circle cx="{x2:.1f}" cy="{y2:.1f}" r="1.5" fill="red" />')
            
    return f'<svg width="{canvas_w}" height="{canvas_h}" xmlns="http://www.w3.org/2000/svg">{"".join(svg_elements)}</svg>'

def main():
    parser = argparse.ArgumentParser(description="Geometric Identifiability & Sloppy Direction Analysis Tool")
    parser.add_argument("reconstruction_json", type=str, help="Path to reconstruction.json")
    parser.add_argument("stage_json", type=str, help="Path to stage JSON file")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save output files")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of Monte-Carlo points")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--delta_p", type=float, default=1e-5, help="Small perturbation for local derivatives")
    parser.add_argument("--max_epipolar_views", type=int, default=100, help="Max views for epipolar analysis")
    parser.add_argument("--gap_threshold", type=float, default=100.0, help="Eigenvalue ratio threshold for gap detection")
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

    # Get detector image size
    # Check if image size is in reconstruction.json
    W_img = 400
    H_img = 300
    if "image_size" in recon_config and recon_config["image_size"] is not None:
        W_img, H_img = recon_config["image_size"]
    elif len(Ps_nominal) > 0:
        W_img, H_img = Ps_nominal[0].image_size

    # 3. Nominal Projections
    coords_nominal = np.zeros((M, N, 2))
    for i, P in enumerate(Ps_nominal):
        x_proj = (P.P @ world_points.T).T
        coords_nominal[i, :, 0] = x_proj[:, 0] / x_proj[:, 2]
        coords_nominal[i, :, 1] = x_proj[:, 1] / x_proj[:, 2]

    # 4. Parameter Perturbations and Local Jacobians
    p_nom = parameterization.get_parameter_vector()
    active_names = list(parameterization.get_names_and_docstr().keys())
    bounds = parameterization.get_bounds()
    K = len(active_names)

    if K == 0:
        raise ValueError("No active parameters configured (opt: true) in the stage file.")

    delta_x_all = {}
    J_cols = np.zeros((2 * M * N, K))
    sweep_results = {}

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

        # Local Derivative
        J_col = dx / args.delta_p
        J_cols[:, j] = J_col.flatten()

        # Sweep range evaluation
        p_min, p_max = bounds[j]
        
        p_perturbed = p_nom.copy()
        p_perturbed[j] = p_min
        parameterization.set_parameter_vector(p_perturbed)
        Ps_min = parameterization.apply_to_trajectory(Ps_nominal)

        p_perturbed = p_nom.copy()
        p_perturbed[j] = p_max
        parameterization.set_parameter_vector(p_perturbed)
        Ps_max = parameterization.apply_to_trajectory(Ps_nominal)

        parameterization.set_parameter_vector(p_nom)  # Reset

        coords_min = np.zeros((M, N, 2))
        coords_max = np.zeros((M, N, 2))
        for i in range(M):
            x_proj_min = (Ps_min[i].P @ world_points.T).T
            coords_min[i, :, 0] = x_proj_min[:, 0] / x_proj_min[:, 2]
            coords_min[i, :, 1] = x_proj_min[:, 1] / x_proj_min[:, 2]

            x_proj_max = (Ps_max[i].P @ world_points.T).T
            coords_max[i, :, 0] = x_proj_max[:, 0] / x_proj_max[:, 2]
            coords_max[i, :, 1] = x_proj_max[:, 1] / x_proj_max[:, 2]

        dx_min_mag = np.linalg.norm(coords_min - coords_nominal, axis=-1)
        dx_max_mag = np.linalg.norm(coords_max - coords_nominal, axis=-1)

        sweep_results[name] = {
            "max_disp_min_range": float(np.max(dx_min_mag)),
            "mean_disp_min_range": float(np.mean(dx_min_mag)),
            "max_disp_max_range": float(np.max(dx_max_mag)),
            "mean_disp_max_range": float(np.mean(dx_max_mag)),
        }

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

    accumulators = {
        name: {
            "count": 0,
            "sum_perp": np.zeros(2),
            "sum_op_perp": np.zeros((2, 2)),
            "sum_para": np.zeros(2),
            "sum_op_para": np.zeros((2, 2)),
        }
        for name in active_names
    }

    # Stacked scalar projections of derivatives for epipolar Jacobian J_perp
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
            d_u = -n_v
            d_v = n_u

            for col_idx, name in enumerate(active_names):
                dx = delta_x_all[name]
                dx_j = dx[j]
                du = dx_j[:, 0]
                dv = dx_j[:, 1]

                scalar_perp = du * n_u + dv * n_v
                scalar_para = du * d_u + dv * d_v

                J_perp[row_idx:row_idx+N, col_idx] = scalar_perp / args.delta_p

                # Do NOT divide by delta_p twice
                vec_perp = np.stack((scalar_perp * n_u, scalar_perp * n_v), axis=1) / args.delta_p
                vec_para = np.stack((scalar_para * d_u, scalar_para * d_v), axis=1) / args.delta_p

                acc = accumulators[name]
                acc["sum_perp"] += vec_perp.sum(axis=0)
                acc["sum_op_perp"] += vec_perp.T @ vec_perp
                acc["sum_para"] += vec_para.sum(axis=0)
                acc["sum_op_para"] += vec_para.T @ vec_para
                acc["count"] += N

            row_idx += N

    # 6. Metric (Gram Matrix) G = J^T J
    G_full = J_cols.T @ J_cols
    G_perp = J_perp.T @ J_perp

    # 7. SVD of J and Eigendecomposition of G
    U_full, S_full, Vt_full = np.linalg.svd(J_cols, full_matrices=False)
    cond_full = float(S_full[0] / S_full[-1]) if S_full[-1] > 1e-12 else float('inf')
    eigvals_full = S_full**2

    U_perp, S_perp, Vt_perp = np.linalg.svd(J_perp, full_matrices=False)
    cond_perp = float(S_perp[0] / S_perp[-1]) if S_perp[-1] > 1e-12 else float('inf')
    eigvals_perp = S_perp**2

    # 8. Spectral Gap Diagnostics
    def detect_spectral_gaps(eigvals, gap_threshold):
        gaps = []
        for i in range(len(eigvals) - 1):
            ratio = eigvals[i] / eigvals[i+1] if eigvals[i+1] > 1e-12 else float('inf')
            if ratio > gap_threshold:
                gaps.append({
                    "index_before": i,
                    "index_after": i + 1,
                    "ratio": ratio,
                    "value_before": float(eigvals[i]),
                    "value_after": float(eigvals[i+1])
                })
        return gaps

    gaps_full = detect_spectral_gaps(eigvals_full, args.gap_threshold)
    gaps_perp = detect_spectral_gaps(eigvals_perp, args.gap_threshold)

    # 9. Format Sloppy / Unobservable Directions
    def get_eigenvector_expressions(Vt, eigvals, param_names, rel_threshold=1e-3):
        max_val = eigvals[0]
        expressions = []
        for idx, val in enumerate(eigvals):
            v = Vt[idx]
            terms = []
            for coef, name in zip(v, param_names):
                if abs(coef) > 0.01:
                    sign = " + " if coef >= 0 else " - "
                    if len(terms) == 0 and coef >= 0:
                        sign = ""
                    terms.append(f"{sign}{abs(coef):.4f} * {name}")
            expr = "".join(terms) if terms else "0.0"
            is_sloppy = (val / max_val < rel_threshold)
            expressions.append({
                "eigenvalue": float(val),
                "ratio": float(val / max_val),
                "expression": expr,
                "is_sloppy": bool(is_sloppy),
                "coefficients": v.tolist()
            })
        return expressions

    eigen_analysis_full = get_eigenvector_expressions(Vt_full, eigvals_full, active_names)
    eigen_analysis_perp = get_eigenvector_expressions(Vt_perp, eigvals_perp, active_names)

    # 10. Effective Rank Calculation
    def estimate_effective_rank(S, rel_tol=1e-3):
        if len(S) == 0:
            return 0
        return int(np.sum(S >= S[0] * rel_tol))

    eff_rank_full = estimate_effective_rank(S_full)
    eff_rank_perp = estimate_effective_rank(S_perp)

    # 11. Parameter Warnings (High Correlation / High Explainability)
    corr_full, exp_full, multi_exp_full = compute_explainability(J_cols)
    corr_perp, exp_perp, multi_exp_perp = compute_explainability(J_perp)

    def get_warnings(corr, exp, param_names):
        corr_warnings = []
        exp_warnings = []
        K = len(param_names)
        for i in range(K):
            for j in range(i + 1, K):
                if abs(corr[i, j]) > 0.95:
                    corr_warnings.append({
                        "p1": param_names[i],
                        "p2": param_names[j],
                        "correlation": float(corr[i, j])
                    })
                if exp[i, j] > 0.95:
                    exp_warnings.append({
                        "p1": param_names[i],
                        "p2": param_names[j],
                        "explainability": float(exp[i, j])
                    })
        return corr_warnings, exp_warnings

    corr_warn_full, exp_warn_full = get_warnings(corr_full, exp_full, active_names)
    corr_warn_perp, exp_warn_perp = get_warnings(corr_perp, exp_perp, active_names)

    # 12. Parameter Importance Analysis
    # We define dominant eigenvectors as those with eigenvalue ratio >= 1e-3
    dominant_indices_full = [idx for idx, item in enumerate(eigen_analysis_full) if not item["is_sloppy"]]
    dominant_indices_perp = [idx for idx, item in enumerate(eigen_analysis_perp) if not item["is_sloppy"]]

    cov_results = {}
    parameter_importance = []
    
    for idx, name in enumerate(active_names):
        dx = delta_x_all[name]
        
        # Original displacements on selected views
        dx_sel = dx[selected_views] / args.delta_p
        trace_total = float(np.mean(np.sum(dx_sel**2, axis=-1)))

        acc = accumulators[name]
        count = acc["count"]
        Cov_perp = acc["sum_op_perp"] / count - np.outer(acc["sum_perp"] / count, acc["sum_perp"] / count)
        Cov_para = acc["sum_op_para"] / count - np.outer(acc["sum_para"] / count, acc["sum_para"] / count)

        trace_perp = float(np.trace(acc["sum_op_perp"] / count))
        trace_para = float(np.trace(acc["sum_op_para"] / count))
        ratio = trace_perp / trace_total if trace_total > 1e-12 else 0.0

        # Calculate average absolute coefficient in dominant eigenvectors
        if dominant_indices_full:
            contrib_full = float(np.mean([abs(Vt_full[i, idx]) for i in dominant_indices_full]))
        else:
            contrib_full = 0.0

        if dominant_indices_perp:
            contrib_perp = float(np.mean([abs(Vt_perp[i, idx]) for i in dominant_indices_perp]))
        else:
            contrib_perp = 0.0

        cov_results[name] = {
            "trace_impact_total": trace_total,
            "trace_perp": trace_perp,
            "trace_para": trace_para,
            "observability_ratio": ratio,
            "dominant_eigenvector_contribution_full": contrib_full,
            "dominant_eigenvector_contribution_epipolar": contrib_perp,
        }

        parameter_importance.append({
            "name": name,
            "trace_total": trace_total,
            "trace_perp": trace_perp,
            "observability_ratio": ratio,
            "contribution_full": contrib_full,
            "contribution_epipolar": contrib_perp,
            "multi_parameter_explainability_full": multi_exp_full[idx],
            "multi_parameter_explainability_epipolar": multi_exp_perp[idx]
        })

    # Sort parameter importance by observability ratio descending
    parameter_importance.sort(key=lambda x: x["observability_ratio"], reverse=True)

    # 13. Reconstruct motion field visualization SVGs for the most unobservable direction (smallest singular value)
    svgs_html = ""
    if K > 0:
        # Smallest full Jacobian eigenvector
        w_full = Vt_full[-1]
        svg_full = make_displacement_field_svg(w_full, J_cols, coords_nominal, active_names, f"Displacement Field: Smallest Singular Vector (SV={S_full[-1]:.4f})", W_img, H_img)
        
        # Smallest epipolar Jacobian eigenvector
        w_perp = Vt_perp[-1]
        svg_perp = make_displacement_field_svg(w_perp, J_cols, coords_nominal, active_names, f"Displacement Field: Smallest Epipolar Singular Vector (SV={S_perp[-1]:.4f})", W_img, H_img)
        
        svgs_html = f"""
<h2>Dominant Motion Visualization (View 0 Grid)</h2>
<p>The vector fields show the nominal point positions (grey dots) and their scaled perturbations (arrows) for the smallest/redundant singular vectors.</p>
<div>
    <h3>Full Jacobian Redundant Direction</h3>
    {svg_full}
</div>
<div>
    <h3>Epipolar Jacobian Redundant Direction</h3>
    {svg_perp}
</div>
"""

    # 14. Output JSON file
    parameterization_classname = parameterization.__class__.__name__
    json_filename = f"{parameterization_classname}_identifiability.json"
    json_output_path = output_dir / json_filename

    output_data = {
        "metadata": {
            "parameterization_class": parameterization_classname,
            "active_parameters": active_names,
            "num_monte_carlo_samples": N,
            "random_seed": args.seed,
            "delta_p": args.delta_p,
            "num_views_total": M,
            "num_views_epipolar_subset": M_sel,
        },
        "range_sweep_displacements": sweep_results,
        "parameter_metric": {
            "G_full": G_full.tolist(),
            "G_perp": G_perp.tolist(),
        },
        "jacobian_analysis": {
            "full_jacobian": {
                "singular_values": S_full.tolist(),
                "eigenvalues": eigvals_full.tolist(),
                "condition_number": cond_full,
                "effective_rank": eff_rank_full,
                "eigenvectors": Vt_full.tolist(),
                "eigen_analysis": eigen_analysis_full,
                "detected_gaps": gaps_full,
                "correlation_matrix": corr_full.tolist(),
                "explainability_matrix": exp_full.tolist(),
                "multi_explainability": multi_exp_full.tolist(),
            },
            "epipolar_jacobian": {
                "singular_values": S_perp.tolist(),
                "eigenvalues": eigvals_perp.tolist(),
                "condition_number": cond_perp,
                "effective_rank": eff_rank_perp,
                "eigenvectors": Vt_perp.tolist(),
                "eigen_analysis": eigen_analysis_perp,
                "detected_gaps": gaps_perp,
                "correlation_matrix": corr_perp.tolist(),
                "explainability_matrix": exp_perp.tolist(),
                "multi_explainability": multi_exp_perp.tolist(),
            }
        },
        "importance": parameter_importance
    }

    json_output_path.write_text(json.dumps(output_data, indent=2))
    print(f"Saved JSON results to {json_output_path}")

    # 15. Output HTML Report
    html_output_path = output_dir / "report_identifiability.html"

    def build_matrix_table(headers, matrix, row_names=None):
        html = "<table border='1'><tr>"
        if row_names:
            html += "<th></th>"
        for h in headers:
            html += f"<th>{h}</th>"
        html += "</tr>"
        for i, row in enumerate(matrix):
            html += "<tr>"
            if row_names:
                html += f"<td><b>{row_names[i]}</b></td>"
            for val in row:
                html += f"<td>{val:.4f}</td>"
            html += "</tr>"
        html += "</table>"
        return html

    # Warnings blocks
    warnings_html = ""
    
    # Correlation warnings
    if corr_warn_full or corr_warn_perp:
        warnings_html += "<h3>Pairwise Correlation Warnings (|&rho;| > 0.95)</h3><ul>"
        for w in corr_warn_full:
            warnings_html += f"<li><b>Full Jacobian:</b> <code>{w['p1']}</code> and <code>{w['p2']}</code> have correlation <b>{w['correlation']:.4f}</b> (highly coupled)</li>"
        for w in corr_warn_perp:
            warnings_html += f"<li><b>Epipolar Jacobian:</b> <code>{w['p1']}</code> and <code>{w['p2']}</code> have correlation <b>{w['correlation']:.4f}</b> (highly coupled under epipolar objective)</li>"
        warnings_html += "</ul>"

    # Explainability warnings
    if exp_warn_full or exp_warn_perp:
        warnings_html += "<h3>Pairwise Explainability Warnings (Explainability > 95%)</h3><ul>"
        for w in exp_warn_full:
            warnings_html += f"<li><b>Full Jacobian:</b> <code>{w['p1']}</code> can reproduce <code>{w['p2']}</code> with explainability <b>{w['explainability']*100:.2f}%</b></li>"
        for w in exp_warn_perp:
            warnings_html += f"<li><b>Epipolar Jacobian:</b> <code>{w['p1']}</code> can reproduce <code>{w['p2']}</code> with explainability <b>{w['explainability']*100:.2f}%</b></li>"
        warnings_html += "</ul>"

    if not warnings_html:
        warnings_html = "<p>No redundancy warnings triggered.</p>"

    # Gap diagnostics
    gaps_html = ""
    if gaps_full or gaps_perp:
        gaps_html += "<ul>"
        for gap in gaps_full:
            gaps_html += f"<li><b>Full Jacobian Gap:</b> Drop between Eig {gap['index_before']} ({gap['value_before']:.2e}) and Eig {gap['index_after']} ({gap['value_after']:.2e}) by ratio <b>{gap['ratio']:.2e}</b></li>"
        for gap in gaps_perp:
            gaps_html += f"<li><b>Epipolar Jacobian Gap:</b> Drop between Eig {gap['index_before']} ({gap['value_before']:.2e}) and Eig {gap['index_after']} ({gap['value_after']:.2e}) by ratio <b>{gap['ratio']:.2e}</b></li>"
        gaps_html += "</ul>"
    else:
        gaps_html = "<p>No large spectral gaps (ratio > 100) detected.</p>"

    # Sloppy Directions list
    sloppy_full_html = "<ul>"
    has_sloppy_full = False
    for i, item in enumerate(eigen_analysis_full):
        if item["is_sloppy"]:
            has_sloppy_full = True
            sloppy_full_html += f"<li><b>Eigval {item['eigenvalue']:.4e} (Ratio {item['ratio']:.2e}):</b> <code>{item['expression']}</code></li>"
    sloppy_full_html += "</ul>"
    if not has_sloppy_full:
        sloppy_full_html = "<p>None detected.</p>"

    sloppy_perp_html = "<ul>"
    has_sloppy_perp = False
    for i, item in enumerate(eigen_analysis_perp):
        if item["is_sloppy"]:
            has_sloppy_perp = True
            sloppy_perp_html += f"<li><b>Eigval {item['eigenvalue']:.4e} (Ratio {item['ratio']:.2e}):</b> <code>{item['expression']}</code></li>"
    sloppy_perp_html += "</ul>"
    if not has_sloppy_perp:
        sloppy_perp_html = "<p>None detected.</p>"

    # Importance table
    importance_table = """<table border='1'>
    <tr>
        <th>Parameter Name</th>
        <th>Observability Ratio</th>
        <th>Trace Total (Original)</th>
        <th>Trace Perp (Observable)</th>
        <th>Contribution to Dominant Eig (Full)</th>
        <th>Contribution to Dominant Eig (Epipolar)</th>
        <th>Multi-Parameter Explainability (Full)</th>
        <th>Multi-Parameter Explainability (Epipolar)</th>
    </tr>
"""
    for item in parameter_importance:
        importance_table += f"""    <tr>
        <td><b>{item['name']}</b></td>
        <td>{item['observability_ratio']:.4f}</td>
        <td>{item['trace_total']:.4f}</td>
        <td>{item['trace_perp']:.4f}</td>
        <td>{item['contribution_full']:.4f}</td>
        <td>{item['contribution_epipolar']:.4f}</td>
        <td>{item['multi_parameter_explainability_full']:.4f}</td>
        <td>{item['multi_parameter_explainability_epipolar']:.4f}</td>
    </tr>"""
    importance_table += "</table>"

    html_content = f"""<html>
<head>
<title>Geometric Identifiability & Sloppy Direction Report</title>
</head>
<body>
<h1>Geometric Identifiability & Sloppy Direction Analysis Report</h1>
<p><b>Link to Optimization Advisor Report:</b> <a href="report_optimization_advisor.html">report_optimization_advisor.html</a></p>

<h2>Trajectory & Configuration Information</h2>
<ul>
    <li><b>Total Views:</b> {M}</li>
    <li><b>Epipolar Views Evaluated:</b> {M_sel} (Stride: {int(np.ceil(M / args.max_epipolar_views)) if M > args.max_epipolar_views else 1})</li>
    <li><b>Monte-Carlo Point Samples:</b> {N}</li>
    <li><b>Parameterization:</b> {parameterization_classname}</li>
</ul>

<h2>Parameter Importance Ranking (By Observability Ratio)</h2>
{importance_table}

<h2>Condition Numbers & Effective Rank</h2>
<table border='1'>
    <tr>
        <th>Jacobian Type</th>
        <th>Condition Number</th>
        <th>Effective Rank (Tol=1e-3)</th>
        <th>Singular Value Spectrum</th>
        <th>Eigenvalue Spectrum</th>
    </tr>
    <tr>
        <td><b>Full Jacobian (J)</b></td>
        <td>{cond_full:.4f}</td>
        <td>{eff_rank_full} / {K}</td>
        <td>{', '.join([f"{s:.4f}" for s in S_full])}</td>
        <td>{', '.join([f"{e:.4e}" for e in eigvals_full])}</td>
    </tr>
    <tr>
        <td><b>Epipolar Jacobian (J_perp)</b></td>
        <td>{cond_perp:.4f}</td>
        <td>{eff_rank_perp} / {K}</td>
        <td>{', '.join([f"{s:.4f}" for s in S_perp])}</td>
        <td>{', '.join([f"{e:.4e}" for e in eigvals_perp])}</td>
    </tr>
</table>

<h2>Spectral Gap Diagnostics (&lambda;_i / &lambda;_(i+1) > 100)</h2>
{gaps_html}

<h2>Automatically Detected Sloppy Directions (Null Space & Weak Combinations)</h2>
<h3>Full Jacobian (J)</h3>
{sloppy_full_html}

<h3>Epipolar Jacobian (J_perp)</h3>
{sloppy_perp_html}

<h2>Redundancy Warnings</h2>
{warnings_html}

<h2>Pairwise Correlations (Full Jacobian)</h2>
{build_matrix_table(active_names, corr_full, active_names)}

<h2>Pairwise Explainability (Full Jacobian)</h2>
{build_matrix_table(active_names, exp_full, active_names)}

<h2>Pairwise Correlations (Epipolar Jacobian)</h2>
{build_matrix_table(active_names, corr_perp, active_names)}

<h2>Pairwise Explainability (Epipolar Jacobian)</h2>
{build_matrix_table(active_names, exp_perp, active_names)}

{svgs_html}

<p>For more detailed data, see: <a href="./{json_filename}">{json_filename}</a></p>
</body>
</html>
"""

    html_output_path.write_text(html_content)
    print(f"Saved HTML report to {html_output_path}")

if __name__ == "__main__":
    main()
