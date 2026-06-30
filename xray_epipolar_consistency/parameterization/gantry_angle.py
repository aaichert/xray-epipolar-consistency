from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
from ProjectiveGeometry23.central_projection import ProjectionMatrix
import ProjectiveGeometry23.homography as homography
import ProjectiveGeometry23.pluecker as pluecker
from ProjectiveGeometry23.utils import homogenize, infinite
from xray_epipolar_consistency.parameterization.base import ParameterizationBase

from scipy.linalg import expm

# TODO move to ProjectiveGeometry23.pluecker
def rotation(L, angle):
    d = pluecker.direction(L).flatten()
    d_norm = np.linalg.norm(d)
    d = d / d_norm
    
    m = pluecker.moment(L).flatten()
    c = np.cross(d, m) / d_norm
    
    c_cos = np.cos(angle)
    c_sin = np.sin(angle)
    
    K = np.array([
        [0.0, -d[2], d[1]],
        [d[2], 0.0, -d[0]],
        [-d[1], d[0], 0.0]
    ])
    
    R = np.eye(3) + c_sin * K + (1.0 - c_cos) * (K @ K)
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = c - R @ c
    return T

def rodrigues_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0]
    ])
    return np.eye(3) + np.sin(angle_rad) * K + (1.0 - np.cos(angle_rad)) * (K @ K)

class GantryAngle(ParameterizationBase):
    """
    Primary and Secondary gantry angular correction.
    """
    
    PARAMETERS = OrderedDict({
        "primary_angle": {
            "description": "Primary rotation angle correction [degrees]",
            "value": 0.0,
            "range": (-2.5, 2.5),
            "opt": True,
        },
        "secondary_angle": {
            "description": "Secondary rotation angle correction [degrees]",
            "value": 0.0,
            "range": (-2.5, 2.5),
            "opt": True,
        },
    })

    def apply_stationary(self, P):
        if self.prior_knowledge is None:
            raise RuntimeError("Call estimateTrajectoryParameters() before using GantryAngle.")

        pa = np.radians(self["primary_angle"]["value"])
        sa = np.radians(self["secondary_angle"]["value"])

        if pa == 0.0 and sa == 0.0:
            return P

        pk = SimpleNamespace(**self.prior_knowledge)
        iso = np.array(pk.iso_center)
        P_matrix = P.P.copy()

        R = np.eye(3)
        t = np.zeros(3)

        if pa != 0.0:
            axis_pri = np.array(pk.rotation_axis)
            R_pri = rodrigues_rotation(axis_pri, pa)
            t = iso - R_pri @ iso
            R = R_pri

        if sa != 0.0:
            r = P.getPrincipalRay().flatten()
            axis_sec = np.cross(r, pk.rotation_axis)
            R_sec = rodrigues_rotation(axis_sec, sa)
            t_sec = iso - R_sec @ iso
            t = R_sec @ t + t_sec
            R = R_sec @ R

        P_matrix[:, 3] += P_matrix[:, :3] @ t
        P_matrix[:, :3] = P_matrix[:, :3] @ R

        return ProjectionMatrix(
            P_matrix,
            pixel_spacing=P.pixel_spacing,
            image_size=P.image_size,
        )

    def apply_stationary_reference_impl(self, P):
        if self.prior_knowledge is None:
            raise RuntimeError("Call estimateTrajectoryParameters() before using GantryAngle.")

        pa = np.radians(self["primary_angle"]["value"])
        sa = np.radians(self["secondary_angle"]["value"])
        pk = SimpleNamespace(**self.prior_knowledge)

        r = P.getPrincipalRay().flatten()

        primary_axis = pluecker.join_points(
            homogenize(pk.iso_center),
            infinite(pk.rotation_axis),
        )

        secondary_axis = pluecker.join_points(
            homogenize(pk.iso_center),
            infinite(np.cross(r, pk.rotation_axis)),
        )

        T = (
            rotation(secondary_axis, sa)
            @ rotation(primary_axis, pa)
        )

        return ProjectionMatrix(
            P.P @ T,
            pixel_spacing=P.pixel_spacing,
            image_size=P.image_size,
        )