import numpy as np
import nrrd
from PIL import Image

def save_slice_as_png(slice_data, filepath):
    slice_data = np.nan_to_num(slice_data)
    s_min = slice_data.min()
    s_max = slice_data.max()
    if s_max > s_min:
        norm = (slice_data - s_min) / (s_max - s_min) * 255.0
    else:
        norm = np.zeros_like(slice_data)
    img = Image.fromarray(norm.astype(np.uint8), mode='L')
    img.save(filepath, optimize=True, compress_level=9)

def extract_and_save_slices(recon_initial_path, recon_opt_path, preview_dir, gt_path=None):
    initial_data, _ = nrrd.read(str(recon_initial_path))
    opt_data, _ = nrrd.read(str(recon_opt_path))
    
    Nx, Ny, Nz = opt_data.shape
    cx, cy, cz = Nx // 2, Ny // 2, Nz // 2
    
    slices = {
        "misaligned": {
            "x": initial_data[cx, :, :],
            "y": initial_data[:, cy, :],
            "z": initial_data[:, :, cz]
        },
        "optimized": {
            "x": opt_data[cx, :, :],
            "y": opt_data[:, cy, :],
            "z": opt_data[:, :, cz]
        }
    }
    
    gt_data = None
    if gt_path is not None:
        gt_data, _ = nrrd.read(str(gt_path))
        slices["gt"] = {
            "x": gt_data[cx, :, :],
            "y": gt_data[:, cy, :],
            "z": gt_data[:, :, cz]
        }
        
    categories = ["misaligned", "optimized"]
    if gt_path is not None:
        categories.append("gt")
        
    for category in categories:
        for axis in ["x", "y", "z"]:
            slice_data = slices[category][axis]
            filename = f"{category}_slice_{axis}.png"
            save_slice_as_png(slice_data, preview_dir / filename)
            
    return gt_data if gt_path is not None else initial_data, opt_data
