from collections import OrderedDict
import numpy as np
from ProjectiveGeometry23.source_detector_geometry import SourceDetectorGeometry
from ProjectiveGeometry23.central_projection import ProjectionMatrix
import ProjectiveGeometry23.homography as homography
from xray_epipolar_consistency.parameterization.base import ParameterizationBase

class Distance(ParameterizationBase):
    """
    Source-Detector and Source-Isocenter distances correction.
    """
    
    PARAMETERS = OrderedDict({
        "delta_sdd": {
            "description": "Source detector distance difference [mm]",
            "value": 0.0,
            "range": (-20.0, 20.0),
            "opt": True,
        },
        "delta_sid": {
            "description": "Source iso-center distance difference [mm]",
            "value": 0.0,
            "range": (-20.0, 20.0),
            "opt": True,
        },
        "coupled_distance": {
            "description": "Coupled SID and SDD change (constant magnification) [mm]",
            "value": 0.0,
            "range": (-20.0, 20.0),
            "opt": False,
        },
    })

    def apply_stationary(self, P):
        if self.prior_knowledge is None:
            raise RuntimeError("Call estimateTrajectoryParameters() before using Distance.")

        delta_sid = self["delta_sid"]["value"]
        delta_sdd = self["delta_sdd"]["value"]
        coupled = self["coupled_distance"]["value"]

        sdd = self.prior_knowledge["sdd"]
        sid = self.prior_knowledge["sid"]

        eff_delta_sid = delta_sid + coupled
        eff_delta_sdd = delta_sdd - coupled * (sdd / sid)

        if eff_delta_sid == 0.0 and eff_delta_sdd == 0.0:
            return P

        P_matrix = P.P.copy()

        if eff_delta_sid != 0.0:
            r = P_matrix[2, :3]
            r_norm = np.linalg.norm(r)
            if r_norm > 1e-12:
                r = r / r_norm
            t = -eff_delta_sid * r
            P_matrix[:, 3] += P_matrix[:, :3] @ t

        if eff_delta_sdd != 0.0:
            scale = (sdd + eff_delta_sdd) / sdd
            cx, cy = P.getPrincipalPoint().flatten()[:2]
            P_matrix[0, :] = scale * P_matrix[0, :] + (1.0 - scale) * cx * P_matrix[2, :]
            P_matrix[1, :] = scale * P_matrix[1, :] + (1.0 - scale) * cy * P_matrix[2, :]

        return ProjectionMatrix(
            P_matrix,
            pixel_spacing=P.pixel_spacing,
            image_size=P.image_size,
        )

    def apply_stationary_reference_impl(self, P):
        if self.prior_knowledge is None:
            raise RuntimeError("Call estimateTrajectoryParameters() before using Distance.")

        delta_sid = self["delta_sid"]["value"]
        delta_sdd = self["delta_sdd"]["value"]
        coupled = self["coupled_distance"]["value"]

        sdd = self.prior_knowledge["sdd"]
        sid = self.prior_knowledge["sid"]

        eff_delta_sid = delta_sid + coupled
        eff_delta_sdd = delta_sdd - coupled * (sdd / sid)

        r = P.getPrincipalRay().flatten()
        r /= np.linalg.norm(r)

        T = homography.translation(-eff_delta_sid * r)
        scale = (sdd + eff_delta_sdd) / sdd

        cx, cy = P.getPrincipalPoint().flatten()[:2]

        H = (
            homography.translation2d([cx, cy])
            @ homography.scale2d(scale)
            @ homography.translation2d([-cx, -cy])
        )

        return ProjectionMatrix(
            H @ P.P @ T,
            pixel_spacing=P.pixel_spacing,
            image_size=P.image_size,
        )