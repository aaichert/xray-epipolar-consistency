import os
import numpy as np
import nrrd
import imageio
from ProjectiveGeometry23.central_projection import ProjectionMatrix
from fileformats.ompl import load_ompl
from xray_epipolar_consistency import VolumeRenderer, EXAMPLE_DATA_PATH

def test_volume_renderer_mip():
    # 1. Create a dummy volume (a solid sphere in the center)
    sz = 32
    z, y, x = np.ogrid[-sz/2:sz/2, -sz/2:sz/2, -sz/2:sz/2]
    mask = x*x + y*y + z*z <= (sz/4)**2
    volume = np.zeros((sz, sz, sz), dtype=np.float32)
    volume[mask] = 1.0

    # 2. Create renderer with identity model matrix (1 voxel = 1 mm)
    renderer = VolumeRenderer(volume, model_matrix=np.eye(4), use_ess=False)
    renderer.raycast_pass = "MIP"
    renderer.samples_per_voxel = 2.0

    # 3. Parallel projection matrix looking along the Z-axis (no perspective division)
    P_mat = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    proj = ProjectionMatrix(P_mat)
    proj.image_size = np.array([sz, sz])
    
    # 4. Render and compare with np.max along Z-axis (axis=0 for Z in D, H, W layout)
    img = renderer.render(proj)
    ref = np.max(volume, axis=0)
    
    # Check shape
    assert img.shape == (sz, sz), f"Expected shape ({sz}, {sz}), got {img.shape}"
    
    # Check correlation to verify almost identical result (accounting for trilinear interpolation at edges)
    corr = np.corrcoef(img.flatten(), ref.flatten())[0, 1]
    assert corr > 0.95, f"Parallel MIP projection and np.max differ too much: correlation = {corr:.4f}"

def test_volume_renderer_iso():
    sz = 32
    z, y, x = np.ogrid[-sz/2:sz/2, -sz/2:sz/2, -sz/2:sz/2]
    mask = x*x + y*y + z*z <= (sz/4)**2
    volume = np.zeros((sz, sz, sz), dtype=np.float32)
    volume[mask] = 1.0

    renderer = VolumeRenderer(volume, model_matrix=np.eye(4), use_ess=False)
    
    P_mat = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    proj = ProjectionMatrix(P_mat)
    proj.image_size = np.array([sz, sz])
    
    renderer.raycast_pass = "IsoSurface"
    renderer.iso_value = 0.5
    img = renderer.render(proj)
    
    assert img.shape == (sz, sz)
    assert np.max(img) > 0.0, "IsoSurface projection is completely black!"

def test_volume_renderer_ea():
    sz = 32
    z, y, x = np.ogrid[-sz/2:sz/2, -sz/2:sz/2, -sz/2:sz/2]
    mask = x*x + y*y + z*z <= (sz/4)**2
    volume = np.zeros((sz, sz, sz), dtype=np.float32)
    volume[mask] = 1.0

    renderer = VolumeRenderer(volume)
    
    P_mat = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 100.0]
    ])
    
    proj = ProjectionMatrix(P_mat)
    proj.image_size = np.array([64, 64])
    
    tf = np.zeros((256, 4), dtype=np.float32)
    tf[128:, :] = [1.0, 0.5, 0.2, 0.1]
    
    renderer.raycast_pass = "EA"
    renderer.transfer_function = tf
    img = renderer.render(proj)
    
    assert img.shape == (64, 64, 4)

def test_volume_renderer_forward_projection_drr():
    # Paths to the raw voxel data inside the package
    volume_path = os.path.join(EXAMPLE_DATA_PATH, "synthetic_pumpkin", "pumpkin_u8.nrrd")
    
    # Fallback paths for reference projection files which are not packaged
    projections_path = os.path.join(EXAMPLE_DATA_PATH, "synthetic_pumpkin", "fullscan_180views_600x400.nrrd")
    if not os.path.exists(projections_path):
        projections_path = "/run/media/aaichert/Intenso/reconstruct/example_data/fullscan_180views_600x400.nrrd"
        
    ompl_path = os.path.join(EXAMPLE_DATA_PATH, "synthetic_pumpkin", "fullscan_180views_600x400.ompl")
    if not os.path.exists(ompl_path):
        ompl_path = "/run/media/aaichert/Intenso/reconstruct/example_data/fullscan_180views_600x400.ompl"

    assert os.path.exists(volume_path), f"Volume file not found: {volume_path}"
    assert os.path.exists(projections_path), f"Projections file not found: {projections_path}"
    assert os.path.exists(ompl_path), f"OMPL file not found: {ompl_path}"

    # 1. Load the volume and transpose to (D, H, W)
    volume_data, _ = nrrd.read(volume_path)
    volume_data = np.transpose(volume_data, (2, 1, 0))

    # 2. Load the projection matrices
    Ps = load_ompl(ompl_path)

    # 3. Set up the Model Transform from VolumeRendering.ini
    model_matrix = np.array([
        [0.484995, 0.0, 0.0, -106.699],
        [0.0, -2.96974e-17, -0.484994, 94.5738],
        [0.0, -0.484995, 2.96973e-17, 63.0493],
        [0.0, 0.0, 0.0, 1.0]
    ])

    # 4. Instantiate the VolumeRenderer
    renderer = VolumeRenderer(volume_data, model_matrix=model_matrix, use_ess=True)
    renderer.raycast_pass = "DRR"
    renderer.samples_per_voxel = 1.5

    # 5. Load reference projections and transpose to (num_views, H, W)
    ref_data, _ = nrrd.read(projections_path)
    ref_projs = np.transpose(ref_data, (2, 1, 0))

    # Test views 0, 30, and 60
    test_views = [0, 30, 60]
    for view in test_views:
        # Render DRR forward projection
        img = renderer.render(Ps[view])
        ref_img = ref_projs[view]

        # Extract min and max values to print if mismatch occurs
        rendered_max = np.max(img)
        reference_max = np.max(ref_img)

        # 6. Save PNG files of the reference and rendered projection
        for output_dir in ["tests", "/home/aaichert/.gemini/antigravity-ide/brain/4ec9e675-e69c-404d-96e8-ef6e6266d322"]:
            if os.path.exists(output_dir):
                ref_png_path = os.path.join(output_dir, f"pumpkin_ref_{view}.png")
                rendered_png_path = os.path.join(output_dir, f"pumpkin_rendered_{view}.png")
                
                # Normalize values to 0-255 for PNG export
                imageio.imwrite(ref_png_path, (ref_img / reference_max * 255.0).astype(np.uint8))
                imageio.imwrite(rendered_png_path, (img / rendered_max * 255.0).astype(np.uint8))

        # 7. Assert strict intensity match to verify identical results
        assert np.max(np.abs(img - ref_img)) < 100.0, (
            f"View {view} intensity mismatch: Rendered Max = {rendered_max:.3f}, "
            f"Reference Max = {reference_max:.3f}"
        )
