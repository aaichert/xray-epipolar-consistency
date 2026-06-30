import numpy as np
from scipy.ndimage import gaussian_filter

import ProjectiveGeometry23.utils as pgu
import ProjectiveGeometry23.pluecker as pluecker
from ProjectiveGeometry23.central_projection import ProjectionMatrix
import xray_epipolar_consistency as ecc

from xray_epipolar_consistency.progress import ProgressBar

class Scan:
    """
    Container for a set of x-ray images and projection geometry, e.g. raw data of a CT scan.
    Used to compute the epipolar consistency metric on the GPU for a set of projection matrices.
    """
    def __init__(self, Is: list[np.ndarray], Ps: list[ProjectionMatrix]):
        self.Is = [np.asarray(I, dtype=np.float32) for I in Is]
        self.set_projection_matrices(Ps)

    def get_projection_matrices(self) -> list[ProjectionMatrix]:
        return [ProjectionMatrix(P.P.copy(), P.image_size, P.pixel_spacing) for P in self.Ps]

    def set_projection_matrices(self, Ps: list[ProjectionMatrix]):
        self.Ps = [
            ProjectionMatrix(
                P.P.copy(),
                image_size=P.image_size,
                pixel_spacing=P.pixel_spacing
            )
            for P in Ps
        ]

    def init_epipolar_consistency(
        self,
        convert_to_line_integral=False,
        gaussian_sigma=1.2,
        dtr_size_factor=0.5,
        num_planes=0,
        object_radius_mm=0.0,
        progress_bar=None
    ):
        """
        Initialize epipolar consistency metric for GPU evaluation.

        Args:
            convert_to_line_integral: Apply log attenuation transform to projections.
            gaussian_sigma: Gaussian smoothing of input images before Radon transform.
            dtr_size_factor: Undersampling of Radon space. E.g. gaussian_sigma=2.4, factor=0.25
            num_planes: Number of epipolar planes. Zero for automatic.
            object_radius_mm: Optional override for the estimated object radius in mm.
            progress_bar: Optional custom progress bar container/callback.
        """
        estimated_geometry = self._estimate_iso_center_and_object_radius()
        if object_radius_mm > 0.0:
            self.object_radius_mm = object_radius_mm

        self.num_planes = num_planes

        # Check if we can reuse the existing Radon intermediates (dtrs)
        recompute_dtrs = True
        if hasattr(self, "dtrs") and self.dtrs:
            prev_line_integral = getattr(self, "_prev_line_integral", None)
            prev_sigma = getattr(self, "_prev_sigma", None)
            prev_dtr_size_factor = getattr(self, "_prev_dtr_size_factor", None)
            if (prev_line_integral == convert_to_line_integral and
                prev_sigma == gaussian_sigma and
                prev_dtr_size_factor == dtr_size_factor):
                recompute_dtrs = False

        if recompute_dtrs:
            if convert_to_line_integral:
                I0 = float(self.Is[0].max())
                if I0 <= 0:
                    I0 = 1.0
            else:
                I0 = None

            self._prev_I0 = I0
            self.size_t = int(np.hypot(*self.Is[0].shape[:2]) * dtr_size_factor)
            self.size_alpha = int(np.ceil((np.pi / 2.0) * self.size_t) // 2)

            pbar = progress_bar or ProgressBar
            self.dtrs = []
            for img in pbar(self.Is, desc="Radon transform"):
                if convert_to_line_integral:
                    img = -np.log(np.clip(img / I0, 1e-6, 1.0)) * 20.0

                img = gaussian_filter(img, sigma=gaussian_sigma)

                self.dtrs.append(
                    ecc.RadonIntermediate(
                        img.copy(),
                        self.size_alpha,
                        self.size_t,
                        int(ecc.RadonFilter.Derivative),
                        int(ecc.RadonPostProcess.Identity),
                    )
                )
            self._prev_line_integral = convert_to_line_integral
            self._prev_sigma = gaussian_sigma
            self._prev_dtr_size_factor = dtr_size_factor
        else:
            # We already have self.dtrs and self.size_t / self.size_alpha
            I0 = getattr(self, "_prev_I0", None)

        self.metric = ecc.MetricRadonIntermediate()
        self.metric.setRadonIntermediates(self.dtrs)
        self.metric.setObjectRadius(self.object_radius_mm)
        self.metric.setEpipolarPlaneNumber(num_planes)

        return {
            "I0": I0,
            "gaussian_sigma": gaussian_sigma,
            "object_radius_mm": self.object_radius_mm,
            "Radon_dim_t_alpha": [self.size_t, self.size_alpha],
            "estimated_geometry": estimated_geometry
        }

    def compute_epipolar_consistency(self):
        if not hasattr(self, "metric"):
            raise RuntimeError("lifecycle bug: first call init_epipolar_consistency!")
        Ps_aligned = [P.P @ self.T_norm for P in self.Ps]
        self.metric.setProjectionMatrices(Ps_aligned)
        cost_matrix = self.metric.evaluate_pairwise()
        return float(np.mean(cost_matrix)), cost_matrix

    def compute_ecc_for_projection_matrices(self, Ps_list: list[list[ProjectionMatrix]]) -> list[float]:
        """
        Compute the epipolar consistency cost (ECC) for multiple sets of projection matrices.
        """
        if not hasattr(self, "metric"):
            raise RuntimeError("lifecycle bug: first call init_epipolar_consistency!")
        
        costs = []
        for Ps in Ps_list:
            Ps_aligned = [P.P @ self.T_norm for P in Ps]
            self.metric.setProjectionMatrices(Ps_aligned)
            cost_matrix = self.metric.evaluate_pairwise()
            costs.append(float(np.mean(cost_matrix)))
        return costs

    def _estimate_iso_center_and_object_radius(self):
        Cs = [pgu.dehomogenize(P.getCenterOfProjection()).flatten() for P in self.Ps]
        
        # Estimate iso-center as closest point to backprojection rays of the image center
        A, b = np.zeros((3, 3)), np.zeros(3)
        for C, P_matrix in zip(Cs, self.Ps):
            W, H = P_matrix.image_size
            u0 = (W - 1) / 2.0
            v0 = (H - 1) / 2.0
            x = np.array([u0, v0, 1.0]).reshape(-1, 1)
            X = pgu.dehomogenize(P_matrix.backproject(x)).flatten()
            r = X - C
            r_norm = np.linalg.norm(r)
            if r_norm > 1e-12:
                r /= r_norm
            P_proj = (np.eye(3) - np.outer(r, r))
            A += P_proj
            b += P_proj @ C
        X_iso = np.linalg.pinv(A) @ b

        # Estimate rotation plane by fitting source positions.
        _, _, Vh = np.linalg.svd(Cs - np.mean(Cs, axis=0))
        axis_val = Vh[-1]
        axis_sign = np.sign(axis_val[2]) or np.sign(axis_val[1]) or np.sign(axis_val[0]) or 1.0
        axis = axis_val * axis_sign

        # Construct T_norm that has rotation axis along z and iso-center in origin
        axis = pluecker.join_points(pgu.infinite(axis), pgu.homogenize(X_iso))
        m, d = pluecker.moment(axis).flatten(), pluecker.direction(axis).flatten()
        m, d = m / np.linalg.norm(m), d / np.linalg.norm(d)
        R = np.vstack((np.cross(m, d), m, d))

        self.T_norm = np.eye(4)
        self.T_norm[:3, :3] = R.T
        self.T_norm[:3, 3] = X_iso

        # estimate mean source-iso-center distance
        sid = float(np.mean(np.linalg.norm(np.asarray(Cs) - X_iso, axis=1)))
        # and use the smaller of the fan/cone opening angles of first projection to get a rough object radius
        f = self.Ps[0].getFocalLengthPx()  # square pixels assumed
        cx, cy = self.Ps[0].getPrincipalPoint()[0:2]
        fov = min(np.arctan(abs(cx) / f), np.arctan(abs(cy) / f))
        self.object_radius_mm = float(sid * np.sin(fov))

        return {
            "T_norm": self.T_norm,
            "sid": sid,
            "object_radius_mm": self.object_radius_mm
        }
    
    def compute_diagnostics(self, progress_bar=None):
        if not hasattr(self, "metric"):
            raise RuntimeError("lifecycle bug: first call init_epipolar_consistency!")

        Ps_aligned = [P.P @ self.T_norm for P in self.Ps]

        n = len(self.Ps)

        cost_matrix = np.full((n, n), 0, dtype=np.float32)
        sample_count_matrix = np.full((n, n), 0, dtype=np.int32)
        weight_matrix = np.full((n, n), 0, np.float32)

        self.metric.setProjectionMatrices(Ps_aligned)
        zero_plane_distances = np.asarray(self.metric.compute_zero_plane_distances(), dtype=np.float32)

        def weighting(x):
            if x < -1 or x > 1:
                return 0.0
            xx = x * x
            return 1.0 - 2.0 * xx + xx * xx

        no_contrib = 0.05 * np.pi
        full_contrib = 0.2 * np.pi

        pbar = progress_bar or ProgressBar
        for i in pbar(range(n), desc="Running diagnostics"):
            Pi = ProjectionMatrix(Ps_aligned[i])
            Ci = Pi.getCenterOfProjection()[:3, 0]
            Ci /= np.linalg.norm(Ci)

            for j in range(i + 1, n):
                Pj = ProjectionMatrix(Ps_aligned[j])
                Cj = Pj.getCenterOfProjection()[:3, 0]
                Cj /= np.linalg.norm(Cj)

                theta = np.arccos(
                    np.clip(np.dot(Ci, Cj), -1.0, 1.0)
                )

                x = (abs(theta - np.pi) - no_contrib) / full_contrib
                if x < 0:
                    w = 0
                else:
                    w = 1.0 - weighting(x)

                weight_matrix[i, j] = w

                cost, v0, v1, _, _ = ecc.compute_for_image_pair(
                    Ps_aligned[i],
                    Ps_aligned[j],
                    self.dtrs[i],
                    self.dtrs[j],
                    self.num_planes,
                    self.object_radius_mm,
                )

                cost_matrix[j, i] = cost

                n_samples = len(v0)
                sample_count_matrix[i, j] = n_samples

        np.fill_diagonal(sample_count_matrix, -1)
        np.fill_diagonal(weight_matrix, -1)
        np.fill_diagonal(cost_matrix, -1)
        np.fill_diagonal(zero_plane_distances, -1)

        max_sample_count = np.max(sample_count_matrix[:])

        max_planes_expected = int(self.size_t + self.size_alpha)
        if max_sample_count > max_planes_expected:
            info = f"Performance Warning: more epipolar planes sampled than expected! (N>{max_planes_expected})"
        elif max_sample_count < (self.size_t + self.size_alpha) * 0.5:
            info = f"Accuracy Warning: fewer epipolar planes sampled than expected! (N<{max_planes_expected//2})"
        else:
            info = "Sample count within expected range."
    
        return {
            "info": info,
            "cost_matrix": cost_matrix,
            "weight_matrix": weight_matrix,
            "sample_count_matrix": sample_count_matrix,
            "max_sample_count": max_sample_count,
            "zero_plane_distances": zero_plane_distances
        }
