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

class RotationAxis(ParameterizationBase):
    """
    Rotation axis tilt and offset.
    """

    PARAMETERS = OrderedDict({
        "tilt_pitch": {
            "description": "Rotation axis pitch (tilt in radial plane) [degrees]",
            "value": 0.0,
            "range": (-1.0, 1.0),
            "opt": True,
        },
        "tilt_roll": {
            "description": "Rotation axis roll (tilt in lateral plane) [degrees]",
            "value": 0.0,
            "range": (-1.0, 1.0),
            "opt": True,
        },
        "offset_lateral": {
            "description": "Rotation axis lateral offset [mm]",
            "value": 0.0,
            "range": (-2.5, 2.5),
            "opt": True,
        },
        "offset_radial": {
            "description": "Rotation axis radial offset [mm]",
            "value": 0.0,
            "range": (-2.5, 2.5),
            "opt": True,
        },
    })

    def _get_axes(self):
        d = np.array(self.prior_knowledge['rotation_axis'])
        r0 = np.array(self.prior_knowledge['first_principal_ray'])

        u0 = np.cross(r0, d)
        u0_norm = np.linalg.norm(u0)
        if u0_norm > 1e-8:
            u0 /= u0_norm
        else:
            u0 = np.cross(d, [1.0, 0.0, 0.0] if abs(d[0]) < 0.9 else [0.0, 1.0, 0.0])
            u0 /= np.linalg.norm(u0)

        v0 = np.cross(d, u0)
        return d, u0, v0

    def apply_stationary(self, P):
        if self.prior_knowledge is None:
            raise RuntimeError("Call estimateTrajectoryParameters() before using RotationAxis.")

        tilt_pitch = np.radians(self["tilt_pitch"]["value"])
        tilt_roll = np.radians(self["tilt_roll"]["value"])
        offset_lat = self["offset_lateral"]["value"]
        offset_rad = self["offset_radial"]["value"]

        if tilt_pitch == 0.0 and tilt_roll == 0.0 and offset_lat == 0.0 and offset_rad == 0.0:
            return P

        d, u0, v0 = self._get_axes()

        r = P.getPrincipalRay().flatten()
        r /= np.linalg.norm(r)

        x = -np.dot(r, u0)
        y = np.dot(r, v0)
        theta = np.arctan2(x, y)

        R_d = rodrigues_rotation(d, theta)
        u = R_d @ u0
        v = R_d @ v0

        iso = np.array(self.prior_knowledge["iso_center"])
        P_matrix = P.P.copy()

        R = np.eye(3)
        if tilt_roll != 0.0:
            R = rodrigues_rotation(v, tilt_roll)

        if tilt_pitch != 0.0:
            R = rodrigues_rotation(u, tilt_pitch) @ R

        t = (np.eye(3) - R) @ iso + offset_lat * u + offset_rad * v

        P_matrix[:, 3] += P_matrix[:, :3] @ t
        P_matrix[:, :3] = P_matrix[:, :3] @ R

        return ProjectionMatrix(
            P_matrix,
            pixel_spacing=P.pixel_spacing,
            image_size=P.image_size,
        )

    def apply_stationary_reference_impl(self, P):
        if self.prior_knowledge is None:
            raise RuntimeError("Call estimateTrajectoryParameters() before using RotationAxis.")

        d, u0, v0 = self._get_axes()

        r = P.getPrincipalRay().flatten()
        r /= np.linalg.norm(r)

        x = -np.dot(r, u0)
        y = np.dot(r, v0)
        theta = np.arctan2(x, y)

        iso = np.array(self.prior_knowledge["iso_center"])
        tilt_pitch_axis_0 = pluecker.join_points(homogenize(iso), infinite(u0))
        tilt_roll_axis_0 = pluecker.join_points(homogenize(iso), infinite(v0))

        T0 = (
            homography.translation(
                self["offset_lateral"]["value"] * u0 +
                self["offset_radial"]["value"] * v0
            )
            @ rotation(
                tilt_pitch_axis_0,
                np.radians(self["tilt_pitch"]["value"]),
            )
            @ rotation(
                tilt_roll_axis_0,
                np.radians(self["tilt_roll"]["value"]),
            )
        )

        rot_axis_line = pluecker.join_points(homogenize(iso), infinite(d))
        R_theta = rotation(rot_axis_line, theta)
        R_minus_theta = rotation(rot_axis_line, -theta)

        T = R_theta @ T0 @ R_minus_theta

        return ProjectionMatrix(
            P.P @ T,
            pixel_spacing=P.pixel_spacing,
            image_size=P.image_size,
        )