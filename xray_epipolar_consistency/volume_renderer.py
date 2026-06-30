import numpy as np
from ._core import VoxelData, Raycaster

class VolumeRenderer:
    """A high-level Python wrapper for CUDA-based volume rendering.
    
    Exposes volume rendering capabilities including MIP, DRR, Iso-surface,
    and Emission-Absorption (with 1D transfer functions).
    """
    
    def __init__(self, volume_data, model_matrix=None, use_ess=True):
        """Initializes the VolumeRenderer.
        
        Args:
            volume_data (np.ndarray): 3D numpy array of shape (D, H, W) containing float values.
            model_matrix (np.ndarray, optional): 4x4 affine matrix defining how voxels map to world space.
            use_ess (bool): Whether to construct and use Empty Space Skipping geometry.
        """
        # Convert volume data to float32 and ensure C-style memory layout
        volume_data = np.ascontiguousarray(volume_data, dtype=np.float32)
        
        self._voxel_data = VoxelData(volume_data, use_ess)
        
        if model_matrix is not None:
            self.model_matrix = model_matrix
        else:
            self.center_volume()
            
        self._raycaster = Raycaster(self._voxel_data)
        
        # Default rendering parameters
        self._pass_type = "MIP"
        self._raycaster.set_raycast_pass(self._pass_type)
        self._samples_per_voxel = 1.0
        self._raycaster.set_samples_per_voxel(self._samples_per_voxel)
        self._iso_value = 0.5
        self._ray_length_weighted = False
        self._transfer_function = None
        self._tf_min_val = float('nan')
        self._tf_max_val = float('nan')
        self._clip_planes = []

    @property
    def model_matrix(self):
        """Gets or sets the 4x4 model matrix mapping voxels to world (mm) space."""
        return self._voxel_data.get_model_transform()

    @model_matrix.setter
    def model_matrix(self, matrix):
        matrix = np.ascontiguousarray(matrix, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError("Model matrix must be of shape (4, 4)")
        self._voxel_data.set_model_transform(matrix)

    def center_volume(self):
        """Centers the volume at the origin of the world space using voxel spacing."""
        self._voxel_data.center_volume()

    def empty_space_skipping(self, bin_factor=16):
        """Recomputes empty space skipping parameters with the given binning factor.
        
        Args:
            bin_factor (int): Downsampling factor (bin size) for ESS.
        """
        self._voxel_data.empty_space_skipping(bin_factor)

    @property
    def samples_per_voxel(self):
        """Gets or sets the sampling rate (samples per voxel step size)."""
        return self._samples_per_voxel

    @samples_per_voxel.setter
    def samples_per_voxel(self, samples):
        self._samples_per_voxel = float(samples)
        self._raycaster.set_samples_per_voxel(self._samples_per_voxel)

    @property
    def raycast_pass(self):
        """Gets or sets the active raycast pass type.
        
        Options: "Debug", "MIP", "IsoSurface" (or "Iso"), "DRR", "EmissionAbsorption" (or "EA").
        """
        return self._pass_type

    @raycast_pass.setter
    def raycast_pass(self, pass_type):
        self._raycaster.set_raycast_pass(pass_type)
        self._pass_type = pass_type
        # Re-apply relevant parameters for the selected pass
        if pass_type in ("IsoSurface", "Iso"):
            self._raycaster.set_iso_value(self._iso_value)
        elif pass_type in ("DRR", "DigitallyReconstructedRadiograph"):
            self._raycaster.set_ray_length_weighted(self._ray_length_weighted)
        elif pass_type in ("EmissionAbsorption", "EA") and self._transfer_function is not None:
            self._raycaster.set_transfer_function_ea(self._transfer_function, self._tf_min_val, self._tf_max_val)
        elif pass_type in ("EmissionAbsorptionShaded", "EAShaded") and self._transfer_function is not None:
            self._raycaster.set_transfer_function_ea_shaded(self._transfer_function, self._tf_min_val, self._tf_max_val)

    @property
    def iso_value(self):
        """Gets or sets the iso-surface threshold value (used in IsoSurface pass)."""
        return self._iso_value

    @iso_value.setter
    def iso_value(self, val):
        self._iso_value = float(val)
        if self._pass_type in ("IsoSurface", "Iso"):
            self._raycaster.set_iso_value(self._iso_value)

    @property
    def ray_length_weighted(self):
        """Gets or sets whether to divide/weight DRR by ray length (used in DRR pass)."""
        return self._ray_length_weighted

    @ray_length_weighted.setter
    def ray_length_weighted(self, weighted):
        self._ray_length_weighted = bool(weighted)
        if self._pass_type in ("DRR", "DigitallyReconstructedRadiograph"):
            self._raycaster.set_ray_length_weighted(self._ray_length_weighted)

    @property
    def transfer_function(self):
        """Gets or sets the transfer function (used in EmissionAbsorption pass).
        
        Must be a NumPy array of shape (N, 4) containing float RGBA values.
        """
        return self._transfer_function

    @transfer_function.setter
    def transfer_function(self, tf_data):
        if tf_data is not None:
            tf_data = np.ascontiguousarray(tf_data, dtype=np.float32)
            if tf_data.ndim != 2 or tf_data.shape[1] != 4:
                raise ValueError("Transfer function must be a 2D array of shape (N, 4) with RGBA values.")
            self._transfer_function = tf_data
            
            # Extract active range from opacity channel (alpha is channel 3)
            opacities = tf_data[:, 3]
            
            # Left bound: check where the opacity is above 0.01
            idx_left = 0
            for i in range(len(opacities)):
                if opacities[i] > 0.01:
                    idx_left = i
                    break
            
            # Right bound: from the right when it is for the first time below 0.99
            idx_right = len(opacities) - 1
            for i in range(len(opacities) - 1, -1, -1):
                if opacities[i] < 0.99:
                    idx_right = i
                    break
            
            if idx_right < idx_left:
                idx_right = idx_left
                
            self._tf_min_val = float(idx_left)
            self._tf_max_val = float(idx_right)
            
            if self._pass_type in ("EmissionAbsorption", "EA"):
                self._raycaster.set_transfer_function_ea(self._transfer_function, self._tf_min_val, self._tf_max_val)
            elif self._pass_type in ("EmissionAbsorptionShaded", "EAShaded"):
                self._raycaster.set_transfer_function_ea_shaded(self._transfer_function, self._tf_min_val, self._tf_max_val)
        else:
            self._transfer_function = None
            self._tf_min_val = float('nan')
            self._tf_max_val = float('nan')

    @property
    def clip_planes(self):
        """Gets or sets the clip planes.
        
        Must be a list of 4-element vectors/lists representing plane equations.
        """
        return self._clip_planes

    @clip_planes.setter
    def clip_planes(self, planes):
        if planes is None:
            self._clip_planes = []
        else:
            self._clip_planes = [np.array(p, dtype=np.float64) for p in planes]
        self._raycaster.set_clip_planes(self._clip_planes)

    def render(self, projection_matrix, channels=None):
        """Renders a 2D projection of the volume.
        
        Args:
            projection_matrix (ProjectionMatrix): Projection matrix object from ProjectiveGeometry23.
            channels (int, optional): Number of channels for output image. If None, automatically determined.
            
        Returns:
            np.ndarray: Renders the image as a 2D array of shape (height, width) for 1 channel,
                        or 3D array of shape (height, width, channels) for multi-channel.
        """
        # Extract projection matrix 3x4 array P
        if hasattr(projection_matrix, 'P'):
            P = projection_matrix.P
        else:
            P = np.ascontiguousarray(projection_matrix, dtype=np.float64)
            if P.shape != (3, 4):
                raise ValueError("Projection matrix must be of shape (3, 4)")
                
        # Extract image dimensions
        if hasattr(projection_matrix, 'image_size'):
            width = int(projection_matrix.image_size[0])
            height = int(projection_matrix.image_size[1])
        else:
            # Fallback default size if raw 3x4 matrix is passed
            width, height = 600, 400
            
        # Determine number of output channels
        if channels is None:
            if self._pass_type in ("EmissionAbsorption", "EA", "EmissionAbsorptionShaded", "EAShaded"):
                channels = 4
            else:
                channels = 1
                
        P = np.ascontiguousarray(P, dtype=np.float64)
        return self._raycaster.render(P, width, height, channels)
