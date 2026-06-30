from collections import OrderedDict
import numpy as np
from ProjectiveGeometry23.central_projection import ProjectionMatrix
from xray_epipolar_consistency.parameterization.base import ParameterizationBase

class SourceShiftAGC(ParameterizationBase):
    """
    Local translation of the X-ray source in the detector coordinate system.
    This parametrization directly translates the source relative to the detector,
    correctly accounting for gantry rotation and detector scaling (voxels vs mm).
    """

    PARAMETERS = OrderedDict({
        "source_shift_x": {
            "description": "Source shift in detector u direction [mm]",
            "value": 0.0,
            "range": (-3.0, 3.0),
            "opt": True,
        },
        "source_shift_y": {
            "description": "Source shift in detector v direction [mm]",
            "value": 0.0,
            "range": (-3.0, 3.0),
            "opt": False,
        },
        "source_shift_z": {
            "description": "Source shift in detector w (optical axis) direction [mm]",
            "value": 0.0,
            "range": (-10.0, 10.0),
            "opt": True,
        },
    })

    def apply_stationary(self, P):
        sx = self["source_shift_x"]["value"]
        sy = self["source_shift_y"]["value"]
        sz = self["source_shift_z"]["value"]
        if sx == 0.0 and sy == 0.0 and sz == 0.0:
            return P
            
        P_matrix = P.P.copy()
        
        M0 = P_matrix[0, :3]
        M1 = P_matrix[1, :3]
        M2 = P_matrix[2, :3]
        
        # Scale factor corresponds to the voxel size in mm/voxel (or 1.0 if in mm coords)
        s = np.linalg.norm(M2)
        c_u = np.dot(M0, M2) / (s**2)
        c_v = np.dot(M1, M2) / (s**2)
        
        f_u = np.linalg.norm(M0 - c_u * M2) / s
        f_v = np.linalg.norm(M1 - c_v * M2) / s
        
        # Shift translation vector t in camera coordinates:
        # t' = t - shift_local
        # The new fourth column is P'_3 = P_3 - s * K * shift_local
        shift_vector = np.array([
            f_u * sx + c_u * sz,
            f_v * sy + c_v * sz,
            sz
        ])
        
        P_matrix[:, 3] -= s * shift_vector
        return ProjectionMatrix(P_matrix, pixel_spacing=P.pixel_spacing, image_size=P.image_size)
