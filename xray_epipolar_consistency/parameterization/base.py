from __future__ import annotations
from os import name
import numpy as np
from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import deepcopy
from ProjectiveGeometry23.central_projection import ProjectionMatrix
from ProjectiveGeometry23.utils import dehomogenize



from textwrap import indent

np.set_printoptions(suppress=True)

class ParameterizationBase(ABC):
    """
    Base class for geometric parameterizations.

    Subclasses define a class attribute PARAMETERS as an OrderedDict
    mapping parameter names to dictionaries with the schema

        {
            "value": float,
            "optimize": bool,
            "range": (float, float),
            "doc": str,
        }

    Additional keys are allowed and preserved.
    """
    PARAMETERS: OrderedDict[str, dict] = OrderedDict()
    COLUMN_WIDTHS = {
        "opt": 3,
        "name": 26,
        "value": 12,
        "range": 16,
        "description": 40,
    }

    def __init__(self, parameters=None, prior_knowledge=None):
        self.prior_knowledge = prior_knowledge
        self.parameters = deepcopy(self.PARAMETERS)
        for name, cfg in (parameters or {}).items():
            self.parameters.setdefault(name, {}).update(cfg)

    def get_parameter_vector(self) -> np.ndarray:
        return np.array([
            p["value"] for p in self.parameters.values()
            if p["opt"]
        ])

    def get_values_by_name(self) -> dict[str, float]:
        """returns a dict of all parameter values by name."""
        return {k: p["value"] for k, p in self.parameters.items()
        if p["opt"]}

    def set_parameter_vector(self, values: np.ndarray) -> None:
        params = [p for p in self.parameters.values() if p["opt"]]
        for p, value in zip(params, values, strict=True):
            p["value"] = float(value)

    def get_bounds(self) -> list[tuple[float, float]]:
        return [
            tuple(p["range"])
            for p in self.parameters.values()
            if p["opt"]
        ]

    def get_names_and_docstr(self):
        return OrderedDict(
            (name, p["description"])
            for name, p in self.parameters.items()
            if p["opt"]
        )

    def to_str_table(self, keys: list[str] = ["opt", "name", "value", "range"]) -> str:
        headers = [k.upper().ljust(self.COLUMN_WIDTHS.get(k, 12)) for k in keys]
        sep = "-" * len(" ".join(headers))
        rows = []
        for name in self:
            vals = []
            for k in keys:
                v = name if k == "name" else self[name].get(k, "")
                if k == "opt": v = " ✓" if v else " ✗"
                if k == "value" and isinstance(v, float): v = f"{v:.2f}"
                if k == "range" and v: v = f"[{v[0]:.2f}, {v[1]:.2f}]"
                vals.append(str(v).ljust(self.COLUMN_WIDTHS.get(k, 12)))
            rows.append(" ".join(vals))
        return "\n".join([" ".join(headers), sep] + rows)

    def __getitem__(self, key):
        return self.parameters[key]

    def __setitem__(self, key, value):
        self.parameters[key] = value

    def __contains__(self, key):
        return key in self.parameters

    def __iter__(self):
        return iter(self.parameters)

    def keys(self):
        return self.parameters.keys()

    def values(self):
        return list(self.parameters.values())

    def items(self):
        return list(self.parameters.items())

    def to_dict(self) -> dict:
        serialized_params = {}
        for name, p_info in self.parameters.items():
            ref = self.PARAMETERS.get(name, {})
            diff = {k: v for k, v in p_info.items() if v != ref.get(k)}
            if diff:
                serialized_params[name] = diff
        return {
            "module": self.__class__.__module__,
            "classname": self.__class__.__name__,
            "parameters": serialized_params,
            "prior_knowledge": self.prior_knowledge,
        }

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            parameters=d["parameters"],
            prior_knowledge=d.get("prior_knowledge"),
        )

    def apply_to_trajectory(self, Ps: list[ProjectionMatrix]) -> list[ProjectionMatrix]:
        if self.prior_knowledge is None:
            self.estimateTrajectoryParameters(Ps)
        return [self.apply_stationary(ProjectionMatrix(P.P.copy(), P.image_size, P.pixel_spacing)) for P in Ps ]

    @abstractmethod
    def apply_stationary(self, P: ProjectionMatrix) -> ProjectionMatrix:
        pass

    def estimateTrajectoryParameters(self, Ps):
        Cs = np.asarray([
            dehomogenize(P.getCenterOfProjection()).flatten()
            for P in Ps
        ])

        A = np.zeros((3, 3))
        b = np.zeros(3)

        for C, r in zip(Cs, (P.getPrincipalRay().flatten() for P in Ps)):
            M = np.eye(3) - np.outer(r, r)
            A += M
            b += M @ C

        _, _, Vh = np.linalg.svd(Cs - Cs.mean(axis=0))
        iso_center = np.linalg.pinv(A) @ b

        from ProjectiveGeometry23.source_detector_geometry import SourceDetectorGeometry
        sdd = float(np.mean([SourceDetectorGeometry(P).source_detector_distance for P in Ps]))
        sid = float(np.mean(np.linalg.norm(Cs - iso_center, axis=1)))

        r0 = Ps[0].getPrincipalRay().flatten()
        r0 /= np.linalg.norm(r0)

        axis_val = Vh[-1]
        axis_sign = np.sign(axis_val[2]) or np.sign(axis_val[1]) or np.sign(axis_val[0]) or 1.0
        self.prior_knowledge = {
            **(self.prior_knowledge or {}),
            "iso_center": iso_center.tolist(),
            "rotation_axis": (axis_val * axis_sign).tolist(),
            "sdd": sdd,
            "sid": sid,
            "first_principal_ray": r0.tolist(),
        }

    @staticmethod
    def align_trajectories(Ps_before: list[ProjectionMatrix], Ps_after: list[ProjectionMatrix]) -> list[ProjectionMatrix]:
        """
        Aligns the optimized projection matrices (Ps_after) to the reference projection matrices (Ps_before).

        Since the epipolar consistency metric is invariant to a single global rigid transformation of 3D space,
        this function estimates the trajectory parameters (isocenter and rotation axis) for both sets of matrices
        and computes a 3D rigid transformation (rotation R and translation t) to align the optimized trajectory
        with the reference trajectory:
            X_before = R * X_after + t

        It then returns a new list of projection matrices where the camera matrices are transformed by the inverse
        of this alignment homography:
            P_aligned = P_after * T_align_inv

        Args:
            Ps_before: A list of reference/ground-truth ProjectionMatrix objects.
            Ps_after: A list of optimized/estimated ProjectionMatrix objects.

        Returns:
            A list of aligned ProjectionMatrix objects.
        """
        def get_iso_and_axis(Ps):
            Cs = np.asarray([
                dehomogenize(P.getCenterOfProjection()).flatten()
                for P in Ps
            ])
            A = np.zeros((3, 3))
            b = np.zeros(3)
            for P_matrix, C in zip(Ps, Cs):
                W, H = P_matrix.image_size
                u0 = (W - 1) / 2.0
                v0 = (H - 1) / 2.0
                x = np.array([u0, v0, 1.0]).reshape(-1, 1)
                X = dehomogenize(P_matrix.backproject(x)).flatten()
                r = X - C
                r_norm = np.linalg.norm(r)
                if r_norm > 1e-12:
                    r /= r_norm
                M_proj = np.eye(3) - np.outer(r, r)
                A += M_proj
                b += M_proj @ C
            iso = np.linalg.pinv(A) @ b
            _, _, Vh = np.linalg.svd(Cs - Cs.mean(axis=0))
            axis_val = Vh[-1]
            axis_sign = np.sign(axis_val[2]) or np.sign(axis_val[1]) or np.sign(axis_val[0]) or 1.0
            axis = axis_val * axis_sign
            return iso, axis

        iso_before, axis_before = get_iso_and_axis(Ps_before)
        iso_after, axis_after = get_iso_and_axis(Ps_after)

        a = axis_after / np.linalg.norm(axis_after)
        b = axis_before / np.linalg.norm(axis_before)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        if s < 1e-8:
            R = np.eye(3) if c > 0 else -np.eye(3)
        else:
            v_x = np.array([
                [0.0, -v[2], v[1]],
                [v[2], 0.0, -v[0]],
                [-v[1], v[0], 0.0]
            ])
            R = np.eye(3) + v_x + (v_x @ v_x) * (1.0 / (1.0 + c))

        # Compute translation t: R @ iso_after + t = iso_before
        t = iso_before - R @ iso_after

        # T_align_inv maps standard/after coordinates back to world/before coordinates
        T_align_inv = np.eye(4)
        T_align_inv[:3, :3] = R.T
        T_align_inv[:3, 3] = -R.T @ t

        return [
            ProjectionMatrix(P.P @ T_align_inv, pixel_spacing=P.pixel_spacing, image_size=P.image_size)
            for P in Ps_after
        ]

    def __repr__(self):
        body = indent(self.to_str_table(), "  ")
        return f"{self.__class__.__name__}(\n{body}\n)"


class ParameterizationChain(ParameterizationBase):
    """
    Sequential composition of parameterizations.
    """

    def __init__(self, parameterizations: list[ParameterizationBase]):
        super().__init__()
        self.parameterizations = parameterizations

    def get_parameter_vector(self):
        return np.concatenate([p.get_parameter_vector() for p in self.parameterizations])

    def set_parameter_vector(self, values):
        i = 0
        for p in self.parameterizations:
            n = len(p.get_parameter_vector())
            p.set_parameter_vector(values[i:i+n])
            i += n

    def get_bounds(self):
        return [b for p in self.parameterizations for b in p.get_bounds()]

    def get_names_and_docstr(self):
        return OrderedDict(
            (f"{p.__class__.__name__}.{name}", doc)
            for p in self.parameterizations
            for name, doc in p.get_names_and_docstr().items()
        )

    def apply_to_trajectory(self, Ps):
        for p in self.parameterizations:
            Ps = p.apply_to_trajectory(Ps)
        return Ps

    def apply_stationary(self, P):
        raise RuntimeError("Parameterization chain operates on trajectories.")

    def _resolve_key(self, key: str) -> tuple[ParameterizationBase, str]:
        cls_name, param_name = key.split(".", 1)
        for p in self.parameterizations:
            if p.__class__.__name__ == cls_name:
                return p, param_name
        raise KeyError(key)

    def __getitem__(self, key):
        p, param_name = self._resolve_key(key)
        return p.parameters[param_name]

    def __setitem__(self, key, value):
        p, param_name = self._resolve_key(key)
        p.parameters[param_name] = value

    def __contains__(self, key):
        try:
            self._resolve_key(key)
            return True
        except KeyError:
            return False

    def __iter__(self):
        for p in self.parameterizations:
            cls_name = p.__class__.__name__
            for name in p.parameters:
                yield f"{cls_name}.{name}"

    def keys(self):
        return list(self)

    def values(self):
        return [self[k] for k in self]

    def items(self):
        return [(k, self[k]) for k in self]

    def to_dict(self):
        return {
            "module": self.__class__.__module__,
            "classname": self.__class__.__name__,
            "parameterizations": [p.to_dict() for p in self.parameterizations]
        }

    def estimateTrajectoryParameters(self, Ps):
        super().estimateTrajectoryParameters(Ps)
        for p in self.parameterizations:
            p.prior_knowledge = self.prior_knowledge  # potential bug if p overloads estimateTrajectoryParameters

    @classmethod
    def from_dict(cls, d):
        from xray_epipolar_consistency.parameterization import from_dict
        return cls([from_dict(p) for p in d["parameterizations"]])
