from abc import abstractmethod
from collections import OrderedDict
from copy import deepcopy
import numpy as np
from xray_epipolar_consistency.parameterization.base import ParameterizationBase

class TimeVariant(ParameterizationBase):
    """
    Abstract base class for wrapping parameterizations in time-varying motion.
    """
    def __init__(self, parameters: dict | None = None, prior_knowledge: dict | None = None, **kwargs):
        self._prior_knowledge = prior_knowledge
        self.referenced_class = kwargs.get("referenced_class")
        self.num_control_points = kwargs.get("num_control_points", 4)
        ref_cfg = kwargs.get("referenced_config", {})
        
        if self.referenced_class is not None:
            ref_prior = ref_cfg.get("prior_knowledge") or prior_knowledge
            self.ref_inst = self.referenced_class(parameters=ref_cfg.get("parameters"), prior_knowledge=ref_prior)
        else:
            self.ref_inst = None
            
        self.parameters = OrderedDict()
        if self.ref_inst is not None:
            for cp in range(self.num_control_points):
                for name, p_info in self.ref_inst.parameters.items():
                    if p_info["opt"]:
                        key = f"{name}_cp{cp}"
                        val = p_info["value"]
                        if isinstance(val, (list, np.ndarray, tuple)):
                            val_cp = val[cp] if cp < len(val) else val[-1]
                        else:
                            val_cp = val
                        self.parameters[key] = {
                            "description": f"{p_info['description']} (control point {cp})",
                            "value": float(val_cp),
                            "range": p_info["range"],
                            "opt": True,
                        }
            for name, p_info in self.ref_inst.parameters.items():
                if not p_info["opt"]:
                    self.parameters[name] = deepcopy(p_info)
                    
        if parameters is not None:
            for name, cfg in parameters.items():
                if name not in self.parameters:
                    raise KeyError(f"Unknown parameter '{name}'")
                self.parameters[name].update(cfg)

    @abstractmethod
    def get_referenced_parameters(self, lmbda: float) -> list[float]:
        pass

    def get_referenced_bounds(self) -> list[tuple[float, float]]:
        if self.ref_inst is None:
            return []
        return self.ref_inst.get_bounds()

    def estimateTrajectoryParameters(self, Ps):
        super().estimateTrajectoryParameters(Ps)
        if self.ref_inst is not None:
            self.ref_inst.prior_knowledge = self.prior_knowledge

    def _get_control_point_parameters(self) -> np.ndarray:
        """
        Retrieves parameter values for all control points, even if some
        individual control point parameters have been disabled (opt=False)
        during optimization.
        Returns a numpy array of shape (num_control_points, num_ref).
        """
        ref_opt_names = [name for name, p in self.ref_inst.parameters.items() if p["opt"]]
        num_ref = len(ref_opt_names)
        n = self.num_control_points
        y = np.zeros((n, num_ref))
        for cp in range(n):
            for j, name in enumerate(ref_opt_names):
                key = f"{name}_cp{cp}"
                y[cp, j] = self.parameters[key]["value"]
        return y



    def to_dict(self) -> dict:
        if self.ref_inst is None:
            return super().to_dict()
            
        ref_params = deepcopy(self.ref_inst.parameters)
        for name, p_info in ref_params.items():
            cp_key_0 = f"{name}_cp0"
            if cp_key_0 in self.parameters:
                p_info["opt"] = self.parameters[cp_key_0]["opt"]
                p_info["range"] = list(self.parameters[cp_key_0]["range"]) if self.parameters[cp_key_0].get("range") is not None else None
                
                val_list = []
                for cp in range(self.num_control_points):
                    key = f"{name}_cp{cp}"
                    if key in self.parameters:
                        val_list.append(self.parameters[key]["value"])
                p_info["value"] = val_list
            elif name in self.parameters:
                p_info["opt"] = self.parameters[name]["opt"]
                p_info["range"] = list(self.parameters[name]["range"]) if self.parameters[name].get("range") is not None else None
                p_info["value"] = self.parameters[name]["value"]
                
        return {
            "module": self.__class__.__module__,
            "classname": self.__class__.__name__,
            "num_control_points": self.num_control_points,
            "referenced_module": self.ref_inst.__class__.__module__,
            "referenced_classname": self.ref_inst.__class__.__name__,
            "referenced_config": {
                "parameters": ref_params,
                "prior_knowledge": self.ref_inst.prior_knowledge,
            }
        }

    @classmethod
    def from_dict(cls, d: dict):
        import importlib
        ref_mod_name = d.get("referenced_module", "xray_epipolar_consistency.parameterization")
        ref_cls_name = d.get("referenced_classname")
        
        if not ref_cls_name:
            raise ValueError("referenced_classname must be specified for TimeVariant parameterization")
            
        ref_module = importlib.import_module(ref_mod_name)
        ref_class = getattr(ref_module, ref_cls_name)
        
        num_control_points = d.get("num_control_points", 4)
        ref_config = d.get("referenced_config", {})
        
        return cls(
            referenced_class=ref_class,
            num_control_points=num_control_points,
            referenced_config=ref_config
        )

    def __repr__(self):
        if self.ref_inst is None:
            return f"{self.__class__.__name__}(ref=None)"
        stats = []
        ref_opt_names = [name for name, p in self.ref_inst.parameters.items() if p["opt"]]
        for name in ref_opt_names:
            vals = []
            for cp in range(self.num_control_points):
                key = f"{name}_cp{cp}"
                if key in self.parameters:
                    vals.append(self.parameters[key]["value"])
            if vals:
                vals = np.array(vals)
                mean_val = np.mean(vals)
                std_val = np.std(vals)
                min_val = np.min(vals)
                max_val = np.max(vals)
                stats.append(f"{name}(mean={mean_val:.2f}, std={std_val:.2f}, range=[{min_val:.2f}, {max_val:.2f}])")
        
        return f"{self.__class__.__name__}(ref={self.ref_inst.__class__.__name__}, control_points={self.num_control_points}, {', '.join(stats)})"

    @property
    def prior_knowledge(self):
        return getattr(self, "_prior_knowledge", None)

    @prior_knowledge.setter
    def prior_knowledge(self, value):
        self._prior_knowledge = value
        if hasattr(self, "ref_inst") and self.ref_inst is not None:
            self.ref_inst.prior_knowledge = value


class LinearDrift(TimeVariant):
    """
    Linear interpolation of correction parameters between equidistant control points.
    """
    def get_referenced_parameters(self, lmbda: float) -> list[float]:
        n = self.num_control_points
        val = lmbda * (n - 1)
        idx = int(np.floor(val))
        idx = min(idx, n - 2)
        t = val - idx
        
        y = self._get_control_point_parameters()
        p0 = y[idx]
        p1 = y[idx + 1]
        
        return [float((1.0 - t) * a + t * b) for a, b in zip(p0, p1)]

    def apply_to_trajectory(self, Ps, params=None):
        if self.prior_knowledge is None:
            self.estimateTrajectoryParameters(Ps)
        n = self.num_control_points
        y = self._get_control_point_parameters()
        Ps_out = []
        for i, P in enumerate(Ps):
            lmbda = i / max(1, len(Ps) - 1)
            val = lmbda * (n - 1)
            idx = int(np.floor(val))
            idx = min(idx, n - 2)
            t = val - idx
            p0 = y[idx]
            p1 = y[idx + 1]
            ref_params = (1.0 - t) * p0 + t * p1
            self.ref_inst.set_parameter_vector(ref_params)
            
            # Interpolate non-optimized parameters if they are control point lists
            for name, p_info in self.ref_inst.parameters.items():
                if not p_info["opt"]:
                    v_val = self.parameters[name]["value"]
                    if isinstance(v_val, (list, np.ndarray, tuple)):
                        v0 = v_val[idx]
                        v1 = v_val[idx + 1]
                        self.ref_inst.parameters[name]["value"] = float((1.0 - t) * v0 + t * v1)
                    else:
                        self.ref_inst.parameters[name]["value"] = float(v_val)

            Ps_out.append(self.ref_inst.apply_stationary(P))
        return Ps_out

    def apply_stationary(self, P, params=None):
        if params is not None:
            self.ref_inst.set_parameter_vector(params)
        for name, p_info in self.ref_inst.parameters.items():
            val = p_info["value"]
            if isinstance(val, (list, np.ndarray, tuple)):
                p_info["value"] = float(val[0])
        return self.ref_inst.apply_stationary(P)


class ContinuousMotion(TimeVariant):
    """
    Cubic spline interpolation between control points.
    """
    def get_referenced_parameters(self, lmbda: float) -> list[float]:
        from scipy.interpolate import CubicSpline
        n = self.num_control_points
        x = np.linspace(0.0, 1.0, n)
        
        y = self._get_control_point_parameters()
        cs = CubicSpline(x, y, axis=0)
        return [float(val) for val in cs(lmbda)]

    def apply_to_trajectory(self, Ps):
        if self.prior_knowledge is None:
            self.estimateTrajectoryParameters(Ps)
        from scipy.interpolate import CubicSpline
        n = self.num_control_points
        x = np.linspace(0.0, 1.0, n)
        
        y = self._get_control_point_parameters()
        cs = CubicSpline(x, y, axis=0)
        lmbdas = np.linspace(0.0, 1.0, len(Ps))
        all_ref_params = cs(lmbdas)
        
        # Build cubic splines for any non-optimized parameters that are lists/arrays
        non_opt_splines = {}
        for name, p_info in self.ref_inst.parameters.items():
            if not p_info["opt"]:
                v_val = self.parameters[name]["value"]
                if isinstance(v_val, (list, np.ndarray, tuple)):
                    non_opt_splines[name] = CubicSpline(x, v_val)
        
        Ps_out = []
        for i, P in enumerate(Ps):
            self.ref_inst.set_parameter_vector(all_ref_params[i])
            
            # Interpolate non-optimized parameters if they are control point lists
            for name, p_info in self.ref_inst.parameters.items():
                if not p_info["opt"]:
                    if name in non_opt_splines:
                        self.ref_inst.parameters[name]["value"] = float(non_opt_splines[name](lmbdas[i]))
                    else:
                        self.ref_inst.parameters[name]["value"] = float(self.parameters[name]["value"])

            Ps_out.append(self.ref_inst.apply_stationary(P))
        return Ps_out

    def apply_stationary(self, P, params=None):
        if params is not None:
            self.ref_inst.set_parameter_vector(params)
        for name, p_info in self.ref_inst.parameters.items():
            val = p_info["value"]
            if isinstance(val, (list, np.ndarray, tuple)):
                p_info["value"] = float(val[0])
        return self.ref_inst.apply_stationary(P)

