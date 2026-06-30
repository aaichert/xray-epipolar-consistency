from collections import OrderedDict
from ProjectiveGeometry23.central_projection import ProjectionMatrix
import ProjectiveGeometry23.homography as homography
from xray_epipolar_consistency.parameterization.base import ParameterizationBase

class DetectorShift(ParameterizationBase):
    """
    Global detector translation in detector coordinates.
    """

    PARAMETERS = OrderedDict({
        "shift_u": {
            "description": "Detector shift in u direction [pixels]",
            "value": 0.0,
            "range": (-10.0, 10.0),
            "opt": True,
        },
        "shift_v": {
            "description": "Detector shift in v direction [pixels]",
            "value": 0.0,
            "range": (-10.0, 10.0),
            "opt": True,
        },
    })

    def apply_stationary(self, P):
        su = self["shift_u"]["value"]
        sv = self["shift_v"]["value"]
        if su == 0.0 and sv == 0.0:
            return P
        P_matrix = P.P.copy()
        if su != 0.0:
            P_matrix[0, :] += su * P_matrix[2, :]
        if sv != 0.0:
            P_matrix[1, :] += sv * P_matrix[2, :]
        return ProjectionMatrix(P_matrix, pixel_spacing=P.pixel_spacing, image_size=P.image_size)

    def apply_stationary_reference_impl(self, P):
        H = homography.translation2d([self["shift_u"]["value"], self["shift_v"]["value"]])
        return ProjectionMatrix(H @ P.P, pixel_spacing=P.pixel_spacing, image_size=P.image_size)
