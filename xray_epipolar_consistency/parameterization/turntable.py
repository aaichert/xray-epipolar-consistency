from collections import OrderedDict
import numpy as np
from ProjectiveGeometry23.central_projection import ProjectionMatrix
import ProjectiveGeometry23.homography as homography
from xray_epipolar_consistency.parameterization.base import ParameterizationBase

def rodrigues_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0]
    ])
    return np.eye(3) + np.sin(angle_rad) * K + (1.0 - np.cos(angle_rad)) * (K @ K)

class Turntable(ParameterizationBase):
    """
    Parametrization of the turntable and detector misalignments in a CT system,
    including self-contained thermal/positional drift parameters.
    """

    PARAMETERS = OrderedDict({
        # Turntable Axis parameters
        "axis_offset_lateral": {
            "description": "Turntable axis lateral offset [mm]",
            "value": 0.0,
            "range": (-5.0, 5.0),
            "opt": True,
        },
        "axis_offset_radial": {
            "description": "Turntable axis radial offset [mm]",
            "value": 0.0,
            "range": (-5.0, 5.0),
            "opt": True,
        },
        "axis_tilt_pitch": {
            "description": "Turntable axis tilt (pitch) [degrees]",
            "value": 0.0,
            "range": (-2.0, 2.0),
            "opt": True,
        },
        "axis_tilt_roll": {
            "description": "Turntable axis tilt (roll) [degrees]",
            "value": 0.0,
            "range": (-2.0, 2.0),
            "opt": True,
        },
        # Static Detector Offset and Scaling parameters
        "detector_shift_u": {
            "description": "Detector shift in u direction [pixels]",
            "value": 0.0,
            "range": (-20.0, 20.0),
            "opt": True,
        },
        "detector_shift_v": {
            "description": "Detector shift in v direction [pixels]",
            "value": 0.0,
            "range": (-20.0, 20.0),
            "opt": True,
        },
        "delta_sdd": {
            "description": "Source-detector distance offset [mm]",
            "value": 0.0,
            "range": (-30.0, 30.0),
            "opt": True,
        },
        "delta_sid": {
            "description": "Source-isocenter distance offset [mm]",
            "value": 0.0,
            "range": (-30.0, 30.0),
            "opt": True,
        },
        # Linear Drift parameters (total drift from start to end of trajectory)
        "drift_detector_shift_u": {
            "description": "Detector drift in u direction [pixels]",
            "value": 0.0,
            "range": (-10.0, 10.0),
            "opt": False,
        },
        "drift_detector_shift_v": {
            "description": "Detector drift in v direction [pixels]",
            "value": 0.0,
            "range": (-10.0, 10.0),
            "opt": False,
        },
        "drift_sdd": {
            "description": "Source-detector distance drift [mm]",
            "value": 0.0,
            "range": (-15.0, 15.0),
            "opt": False,
        },
        "drift_sid": {
            "description": "Source-isocenter distance drift [mm]",
            "value": 0.0,
            "range": (-15.0, 15.0),
            "opt": False,
        },
        # Detector Orientation parameters
        "detector_roll": {
            "description": "Detector roll (in-plane rotation) [degrees]",
            "value": 0.0,
            "range": (-2.0, 2.0),
            "opt": True,
        },
        "detector_pitch": {
            "description": "Detector pitch (out-of-plane rotation) [degrees]",
            "value": 0.0,
            "range": (-2.0, 2.0),
            "opt": True,
        },
        "detector_yaw": {
            "description": "Detector yaw (out-of-plane rotation) [degrees]",
            "value": 0.0,
            "range": (-2.0, 2.0),
            "opt": True,
        },
    })

    def apply_to_trajectory(self, Ps: list[ProjectionMatrix]) -> list[ProjectionMatrix]:
        if self.prior_knowledge is None:
            self.estimateTrajectoryParameters(Ps)

        # Retrieve prior knowledge
        pk = self.prior_knowledge
        iso = np.array(pk["iso_center"])
        d_axis = np.array(pk["rotation_axis"])
        d_axis /= np.linalg.norm(d_axis)
        sdd = pk["sdd"]

        # Base parameter values
        t_lat = self["axis_offset_lateral"]["value"]
        t_rad = self["axis_offset_radial"]["value"]
        theta_pitch = np.radians(self["axis_tilt_pitch"]["value"])
        theta_roll = np.radians(self["axis_tilt_roll"]["value"])
        det_roll = np.radians(self["detector_roll"]["value"])
        det_pitch = np.radians(self["detector_pitch"]["value"])
        det_yaw = np.radians(self["detector_yaw"]["value"])

        # Base drift and static offsets
        shift_u_base = self["detector_shift_u"]["value"]
        shift_v_base = self["detector_shift_v"]["value"]
        delta_sdd_base = self["delta_sdd"]["value"]
        delta_sid_base = self["delta_sid"]["value"]

        drift_u = self["drift_detector_shift_u"]["value"]
        drift_v = self["drift_detector_shift_v"]["value"]
        drift_sdd = self["drift_sdd"]["value"]
        drift_sid = self["drift_sid"]["value"]

        N = len(Ps)
        Ps_out = []
        for i, P in enumerate(Ps):
            # Compute trajectory progress lambda from 0 to 1
            lmbda = i / max(1, N - 1)

            # Apply drift to parameters
            curr_shift_u = shift_u_base + lmbda * drift_u
            curr_shift_v = shift_v_base + lmbda * drift_v
            curr_delta_sdd = delta_sdd_base + lmbda * drift_sdd
            curr_delta_sid = delta_sid_base + lmbda * drift_sid

            P_corrected = self._apply_geometry(
                P, iso, d_axis, sdd,
                t_lat, t_rad, theta_pitch, theta_roll,
                curr_shift_u, curr_shift_v, curr_delta_sdd, curr_delta_sid,
                det_roll, det_pitch, det_yaw
            )
            Ps_out.append(P_corrected)

        return Ps_out

    def apply_stationary(self, P: ProjectionMatrix) -> ProjectionMatrix:
        if self.prior_knowledge is None:
            raise RuntimeError("Call estimateTrajectoryParameters() before applying Turntable parameterization.")

        pk = self.prior_knowledge
        iso = np.array(pk["iso_center"])
        d_axis = np.array(pk["rotation_axis"])
        d_axis /= np.linalg.norm(d_axis)
        sdd = pk["sdd"]

        # In stationary mode, lambda is 0, so drift is not applied
        return self._apply_geometry(
            P, iso, d_axis, sdd,
            self["axis_offset_lateral"]["value"],
            self["axis_offset_radial"]["value"],
            np.radians(self["axis_tilt_pitch"]["value"]),
            np.radians(self["axis_tilt_roll"]["value"]),
            self["detector_shift_u"]["value"],
            self["detector_shift_v"]["value"],
            self["delta_sdd"]["value"],
            self["delta_sid"]["value"],
            np.radians(self["detector_roll"]["value"]),
            np.radians(self["detector_pitch"]["value"]),
            np.radians(self["detector_yaw"]["value"])
        )

    def _apply_geometry(
        self, P: ProjectionMatrix, iso: np.ndarray, d_axis: np.ndarray, sdd: float,
        t_lat: float, t_rad: float, theta_pitch: float, theta_roll: float,
        shift_u: float, shift_v: float, delta_sdd: float, delta_sid: float,
        det_roll: float, det_pitch: float, det_yaw: float
    ) -> ProjectionMatrix:
        use_turntable = (t_lat != 0.0 or t_rad != 0.0 or theta_pitch != 0.0 or theta_roll != 0.0)
        use_sid = (delta_sid != 0.0)
        use_sdd = (delta_sdd != 0.0)
        use_out_of_plane = (det_pitch != 0.0 or det_yaw != 0.0)
        use_roll = (det_roll != 0.0)
        use_shift = (shift_u != 0.0 or shift_v != 0.0)

        if not (use_turntable or use_sid or use_sdd or use_out_of_plane or use_roll or use_shift):
            return P

        P_matrix = P.P.copy()

        # --- 1. Compute Turntable Axis 3D Transform & SID Translation ---
        if use_turntable or use_sid:
            r_ray = P.getPrincipalRay().flatten()
            r_norm = np.linalg.norm(r_ray)
            if r_norm > 1e-12:
                r_ray /= r_norm

            if use_turntable:
                u_basis = np.cross(r_ray, d_axis)
                u_norm = np.linalg.norm(u_basis)
                if u_norm > 1e-8:
                    u_basis /= u_norm
                else:
                    u_basis = np.cross(d_axis, [1.0, 0.0, 0.0] if abs(d_axis[0]) < 0.9 else [0.0, 1.0, 0.0])
                    u_basis /= np.linalg.norm(u_basis)

                v_basis = np.cross(d_axis, u_basis)

                R_tilt = rodrigues_rotation(v_basis, theta_roll) @ rodrigues_rotation(u_basis, theta_pitch)
                t_shift = t_lat * u_basis + t_rad * v_basis
                t_world = iso - R_tilt @ iso + t_shift

                # Apply world transform
                P_matrix[:, 3] += P_matrix[:, :3] @ t_world
                P_matrix[:, :3] = P_matrix[:, :3] @ R_tilt

            if use_sid:
                t_sid = -delta_sid * r_ray
                P_matrix[:, 3] += P_matrix[:, :3] @ t_sid

        # --- 2. Apply Detector Plane Transformations via 2D Homographies & Row Operations ---
        if use_sdd or use_out_of_plane or use_roll:
            cx, cy = P.getPrincipalPoint().flatten()[:2]

        if use_sdd:
            scale = (sdd + delta_sdd) / sdd
            P_matrix[0, :] = scale * P_matrix[0, :] + (1.0 - scale) * cx * P_matrix[2, :]
            P_matrix[1, :] = scale * P_matrix[1, :] + (1.0 - scale) * cy * P_matrix[2, :]

        if use_out_of_plane:
            f = P.getFocalLengthPx()
            K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
            K_inv = np.array([[1.0/f, 0.0, -cx/f], [0.0, 1.0/f, -cy/f], [0.0, 0.0, 1.0]])
            c_p, s_p = np.cos(det_pitch), np.sin(det_pitch)
            R_x = np.array([
                [1.0, 0.0, 0.0],
                [0.0, c_p, -s_p],
                [0.0, s_p, c_p]
            ])
            c_y, s_y = np.cos(det_yaw), np.sin(det_yaw)
            R_y = np.array([
                [c_y, 0.0, s_y],
                [0.0, 1.0, 0.0],
                [-s_y, 0.0, c_y]
            ])
            H_out = K @ R_y @ R_x @ K_inv
            P_matrix = H_out @ P_matrix

        if use_roll:
            c, s = np.cos(det_roll), np.sin(det_roll)
            row0 = P_matrix[0, :].copy()
            row1 = P_matrix[1, :].copy()
            row2 = P_matrix[2, :]
            P_matrix[0, :] = c * row0 - s * row1 + (cx * (1.0 - c) + cy * s) * row2
            P_matrix[1, :] = s * row0 + c * row1 + (cy * (1.0 - c) - cx * s) * row2

        if use_shift:
            if shift_u != 0.0:
                P_matrix[0, :] += shift_u * P_matrix[2, :]
            if shift_v != 0.0:
                P_matrix[1, :] += shift_v * P_matrix[2, :]

        return ProjectionMatrix(
            P_matrix,
            pixel_spacing=P.pixel_spacing,
            image_size=P.image_size
        )

    def apply_to_trajectory_reference_impl(self, Ps: list[ProjectionMatrix]) -> list[ProjectionMatrix]:
        if self.prior_knowledge is None:
            self.estimateTrajectoryParameters(Ps)

        # Retrieve prior knowledge
        pk = self.prior_knowledge
        iso = np.array(pk["iso_center"])
        d_axis = np.array(pk["rotation_axis"])
        d_axis /= np.linalg.norm(d_axis)
        sdd = pk["sdd"]

        # Base parameter values
        t_lat = self["axis_offset_lateral"]["value"]
        t_rad = self["axis_offset_radial"]["value"]
        theta_pitch = np.radians(self["axis_tilt_pitch"]["value"])
        theta_roll = np.radians(self["axis_tilt_roll"]["value"])
        det_roll = np.radians(self["detector_roll"]["value"])
        det_pitch = np.radians(self["detector_pitch"]["value"])
        det_yaw = np.radians(self["detector_yaw"]["value"])

        # Base drift and static offsets
        shift_u_base = self["detector_shift_u"]["value"]
        shift_v_base = self["detector_shift_v"]["value"]
        delta_sdd_base = self["delta_sdd"]["value"]
        delta_sid_base = self["delta_sid"]["value"]

        drift_u = self["drift_detector_shift_u"]["value"]
        drift_v = self["drift_detector_shift_v"]["value"]
        drift_sdd = self["drift_sdd"]["value"]
        drift_sid = self["drift_sid"]["value"]

        N = len(Ps)
        Ps_out = []
        for i, P in enumerate(Ps):
            # Compute trajectory progress lambda from 0 to 1
            lmbda = i / max(1, N - 1)

            # Apply drift to parameters
            curr_shift_u = shift_u_base + lmbda * drift_u
            curr_shift_v = shift_v_base + lmbda * drift_v
            curr_delta_sdd = delta_sdd_base + lmbda * drift_sdd
            curr_delta_sid = delta_sid_base + lmbda * drift_sid

            P_corrected = self._apply_geometry_reference_impl(
                P, iso, d_axis, sdd,
                t_lat, t_rad, theta_pitch, theta_roll,
                curr_shift_u, curr_shift_v, curr_delta_sdd, curr_delta_sid,
                det_roll, det_pitch, det_yaw
            )
            Ps_out.append(P_corrected)

        return Ps_out

    def apply_stationary_reference_impl(self, P: ProjectionMatrix) -> ProjectionMatrix:
        if self.prior_knowledge is None:
            raise RuntimeError("Call estimateTrajectoryParameters() before applying Turntable parameterization.")

        pk = self.prior_knowledge
        iso = np.array(pk["iso_center"])
        d_axis = np.array(pk["rotation_axis"])
        d_axis /= np.linalg.norm(d_axis)
        sdd = pk["sdd"]

        # In stationary mode, lambda is 0, so drift is not applied
        return self._apply_geometry_reference_impl(
            P, iso, d_axis, sdd,
            self["axis_offset_lateral"]["value"],
            self["axis_offset_radial"]["value"],
            np.radians(self["axis_tilt_pitch"]["value"]),
            np.radians(self["axis_tilt_roll"]["value"]),
            self["detector_shift_u"]["value"],
            self["detector_shift_v"]["value"],
            self["delta_sdd"]["value"],
            self["delta_sid"]["value"],
            np.radians(self["detector_roll"]["value"]),
            np.radians(self["detector_pitch"]["value"]),
            np.radians(self["detector_yaw"]["value"])
        )

    def _apply_geometry_reference_impl(
        self, P: ProjectionMatrix, iso: np.ndarray, d_axis: np.ndarray, sdd: float,
        t_lat: float, t_rad: float, theta_pitch: float, theta_roll: float,
        shift_u: float, shift_v: float, delta_sdd: float, delta_sid: float,
        det_roll: float, det_pitch: float, det_yaw: float
    ) -> ProjectionMatrix:
        # --- 1. Construct Turntable Coordinate System ---
        r_ray = P.getPrincipalRay().flatten()
        r_ray /= np.linalg.norm(r_ray)

        # u_basis is the lateral direction
        u_basis = np.cross(r_ray, d_axis)
        u_norm = np.linalg.norm(u_basis)
        if u_norm > 1e-8:
            u_basis /= u_norm
        else:
            u_basis = np.cross(d_axis, [1.0, 0.0, 0.0] if abs(d_axis[0]) < 0.9 else [0.0, 1.0, 0.0])
            u_basis /= np.linalg.norm(u_basis)

        # v_basis is the radial direction
        v_basis = np.cross(d_axis, u_basis)

        # --- 2. Compute Turntable Axis 3D Transform ---
        R_tilt = rodrigues_rotation(v_basis, theta_roll) @ rodrigues_rotation(u_basis, theta_pitch)
        t_shift = t_lat * u_basis + t_rad * v_basis

        T_world = np.eye(4)
        T_world[:3, :3] = R_tilt
        T_world[:3, 3] = iso - R_tilt @ iso + t_shift

        # Apply world transformation to Projection Matrix
        P_curr = P.P @ T_world

        # --- 3. Apply Source-Isocenter Distance (SID) Offset ---
        T_sid = np.eye(4)
        T_sid[:3, 3] = -delta_sid * r_ray
        P_curr = P_curr @ T_sid

        # --- 4. Apply Detector Plane Transformations via 2D Homographies ---
        cx, cy = P.getPrincipalPoint().flatten()[:2]
        f = P.getFocalLengthPx()

        # A. SDD Scaling
        scale = (sdd + delta_sdd) / sdd
        H_scale = np.array([
            [scale, 0.0, (1.0 - scale) * cx],
            [0.0, scale, (1.0 - scale) * cy],
            [0.0, 0.0, 1.0]
        ])

        # B. Detector Out-of-Plane Rotations (Pitch/Yaw)
        K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
        K_inv = np.array([[1.0/f, 0.0, -cx/f], [0.0, 1.0/f, -cy/f], [0.0, 0.0, 1.0]])
        
        c_p, s_p = np.cos(det_pitch), np.sin(det_pitch)
        R_x = np.array([
            [1.0, 0.0, 0.0],
            [0.0, c_p, -s_p],
            [0.0, s_p, c_p]
        ])
        
        c_y, s_y = np.cos(det_yaw), np.sin(det_yaw)
        R_y = np.array([
            [c_y, 0.0, s_y],
            [0.0, 1.0, 0.0],
            [-s_y, 0.0, c_y]
        ])
        
        H_out = K @ R_y @ R_x @ K_inv

        # C. Detector Roll (In-Plane Rotation)
        c_r, s_r = np.cos(det_roll), np.sin(det_roll)
        H_roll = np.array([
            [c_r, -s_r, cx * (1.0 - c_r) + cy * s_r],
            [s_r, c_r, cy * (1.0 - c_r) - cx * s_r],
            [0.0, 0.0, 1.0]
        ])

        # D. Detector Shifts
        H_shift = np.array([
            [1.0, 0.0, shift_u],
            [0.0, 1.0, shift_v],
            [0.0, 0.0, 1.0]
        ])

        # Combine detector homographies
        H_det = H_shift @ H_roll @ H_out @ H_scale

        return ProjectionMatrix(
            H_det @ P_curr,
            pixel_spacing=P.pixel_spacing,
            image_size=P.image_size
        )
