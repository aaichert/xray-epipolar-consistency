from collections import OrderedDict
import numpy as np
from ProjectiveGeometry23.central_projection import ProjectionMatrix
import ProjectiveGeometry23.homography as homography
from xray_epipolar_consistency.parameterization.base import ParameterizationBase

class DetectorOrientation(ParameterizationBase):
    """
    Global detector out-of-plane rotation (slant/skew/tilt).
    """

    PARAMETERS = OrderedDict({
        "tilt_roll": {
            "description": "Detector in-plane rotation (tilt/roll) [degrees]",
            "value": 0.0,
            "range": (-2.0, 2.0),
            "opt": True,
        },
        "slant_yaw": {
            "description": "Detector out-of-plane rotation (slant/yaw) [degrees]",
            "value": 0.0,
            "range": (-2.0, 2.0),
            "opt": True,
        },
        "skew_pitch": {
            "description": "Detector out-of-plane rotation (skew/pitch) [degrees]",
            "value": 0.0,
            "range": (-0.5, 0.5),
            "opt": True,
        },
    })

    def apply_stationary(self, P):
        tilt = np.radians(self["tilt_roll"]["value"])
        slant = np.radians(self["slant_yaw"]["value"])
        skew = np.radians(self["skew_pitch"]["value"])

        if tilt == 0.0 and slant == 0.0 and skew == 0.0:
            return P

        P_matrix = P.P.copy()
        cx, cy = P.getPrincipalPoint().flatten()[:2]

        if slant != 0.0 or skew != 0.0:
            f = P.getFocalLengthPx()
            K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
            K_inv = np.array([[1.0/f, 0.0, -cx/f], [0.0, 1.0/f, -cy/f], [0.0, 0.0, 1.0]])
            R_slant_skew = homography.rotation_y(slant)[:3, :3] @ homography.rotation_x(skew)[:3, :3]
            H_slant_skew = K @ R_slant_skew @ K_inv
            P_matrix = H_slant_skew @ P_matrix

        if tilt != 0.0:
            c = np.cos(tilt)
            s = np.sin(tilt)
            row0 = P_matrix[0, :].copy()
            row1 = P_matrix[1, :].copy()
            row2 = P_matrix[2, :]
            P_matrix[0, :] = c * row0 - s * row1 + (cx * (1.0 - c) + cy * s) * row2
            P_matrix[1, :] = s * row0 + c * row1 + (cy * (1.0 - c) - cx * s) * row2

        return ProjectionMatrix(P_matrix, pixel_spacing=P.pixel_spacing, image_size=P.image_size)

    def apply_stationary_reference_impl(self, P):
        tilt = np.radians(self["tilt_roll"]["value"])
        slant = np.radians(self["slant_yaw"]["value"])
        skew = np.radians(self["skew_pitch"]["value"])

        f = P.getFocalLengthPx()
        cx, cy = P.getPrincipalPoint().flatten()[:2]
        K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
        K_inv = np.array([[1.0/f, 0.0, -cx/f], [0.0, 1.0/f, -cy/f], [0.0, 0.0, 1.0]])

        H_tilt = homography.translation2d([cx, cy]) @ homography.rotation2d(tilt) @ homography.translation2d([-cx, -cy])
        R_slant_skew = homography.rotation_y(slant)[:3, :3] @ homography.rotation_x(skew)[:3, :3]
        H_slant_skew = K @ R_slant_skew @ K_inv

        return ProjectionMatrix(H_tilt @ H_slant_skew @ P.P, pixel_spacing=P.pixel_spacing, image_size=P.image_size)
