from collections import OrderedDict
from types import SimpleNamespace
import numpy as np

from ProjectiveGeometry23.central_projection import ProjectionMatrix
import ProjectiveGeometry23.homography as homography

from xray_epipolar_consistency.parameterization.base import ParameterizationBase


class ObjectPose(ParameterizationBase):
    """
    Rigid 3D object transformation about the estimated iso-center.
    """
        
    PARAMETERS = OrderedDict({
        "translation_x": {
            "description": "Object translation X [mm]",
            "value": 0.0,
            "range": (-2.5, 2.5),
            "opt": True,
        },
        "translation_y": {
            "description": "Object translation Y [mm]",
            "value": 0.0,
            "range": (-2.5, 2.5),
            "opt": True,
        },
        "translation_z": {
            "description": "Object translation Z [mm]",
            "value": 0.0,
            "range": (-2.5, 2.5),
            "opt": True,
        },
        "rotation_x": {
            "description": "Object rotation X [degrees]",
            "value": 0.0,
            "range": (-2.0, 2.0),
            "opt": True,
        },
        "rotation_y": {
            "description": "Object rotation Y [degrees]",
            "value": 0.0,
            "range": (-2.0, 2.0),
            "opt": True,
        },
        "rotation_z": {
            "description": "Object rotation Z [degrees]",
            "value": 0.0,
            "range": (-2.0, 2.0),
            "opt": True,
        },
    })
    

    def apply_stationary(self, P):
        if self.prior_knowledge is None:
            raise RuntimeError("Call estimateTrajectoryParameters() before using ObjectPose.")

        tx = self["translation_x"]["value"]
        ty = self["translation_y"]["value"]
        tz = self["translation_z"]["value"]
        rx = np.radians(self["rotation_x"]["value"])
        ry = np.radians(self["rotation_y"]["value"])
        rz = np.radians(self["rotation_z"]["value"])

        if tx == 0.0 and ty == 0.0 and tz == 0.0 and rx == 0.0 and ry == 0.0 and rz == 0.0:
            return P

        P_matrix = P.P.copy()
        
        R = np.eye(3)
        if rx != 0.0:
            R = homography.rotation_x(rx)[:3, :3] @ R
        if ry != 0.0:
            R = homography.rotation_y(ry)[:3, :3] @ R
        if rz != 0.0:
            R = homography.rotation_z(rz)[:3, :3] @ R

        iso = np.asarray(self.prior_knowledge["iso_center"])
        t_trans = np.array([tx, ty, tz])
        t_total = t_trans + iso - R @ iso

        P_matrix[:, 3] += P_matrix[:, :3] @ t_total
        P_matrix[:, :3] = P_matrix[:, :3] @ R

        return ProjectionMatrix(
            P_matrix,
            pixel_spacing=P.pixel_spacing,
            image_size=P.image_size,
        )

    def apply_stationary_reference_impl(self, P):
        if self.prior_knowledge is None:
            raise RuntimeError("Call estimateTrajectoryParameters() before using ObjectPose.")

        pk = SimpleNamespace(**self.prior_knowledge)

        T = (
            homography.translation([self["translation_x"]["value"], self["translation_y"]["value"], self["translation_z"]["value"] ])
            @ homography.translation(+np.asarray(pk.iso_center))
            @ homography.rotation_z(np.radians(self["rotation_z"]["value"]))
            @ homography.rotation_y(np.radians(self["rotation_y"]["value"]))
            @ homography.rotation_x(np.radians(self["rotation_x"]["value"]))
            @ homography.translation(-np.asarray(pk.iso_center))
        )

        return ProjectionMatrix(
            P.P @ T,
            pixel_spacing=P.pixel_spacing,
            image_size=P.image_size,
        )