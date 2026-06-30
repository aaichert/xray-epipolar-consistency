import sys, os, numpy as np, nrrd
from PIL import Image
from skimage.transform import radon, resize

# Import the installed package, not local source
sys.path = [p for p in sys.path if os.path.abspath(p) != os.path.abspath(os.path.dirname(__file__))]
import xray_epipolar_consistency._core as ecc

def norm(img):
    return np.clip((img - img.min()) / (img.max() - img.min() + 1e-8) * 255, 0, 255).astype(np.uint8)

def main():
    data, _ = nrrd.read(os.path.join(os.path.dirname(__file__), "../xray_epipolar_consistency/example_data/proj000.nrrd"))
    img = resize(data.astype(np.float32), (300, 300), anti_aliasing=True)
    
    # Add a single high-intensity pixel at exactly the center of the image
    center_y, center_x = img.shape[0] // 2, img.shape[1] // 2
    img[center_y, center_x] = img.max() * 5 
    
    # Add a single high-intensity pixel in the top left corner
    img[10, 10] = img.max() * 5
    
    sk_radon = radon(img, theta=np.linspace(0., 180., 180, endpoint=False), circle=False)
    
    # scikit-image treats the Y axis as going up, while ECC treats it as going down (row index).
    # To match their coordinate systems, we flip the image vertically before passing to ECC.
    img_ecc = np.flipud(img).copy()
    ri = ecc.RadonIntermediate(img_ecc, 180, sk_radon.shape[0], int(getattr(ecc.RadonFilter, "None")), int(getattr(ecc.RadonPostProcess, "Identity")))
    ecc_radon = ri.get_data()
    
    diff = np.abs(sk_radon - ecc_radon)
    print(f"skimage  min/max: {sk_radon.min():.4f} / {sk_radon.max():.4f}")
    print(f"ECC      min/max: {ecc_radon.min():.4f} / {ecc_radon.max():.4f}")
    print(f"Diff min/max/mean: {diff.min():.4f} / {diff.max():.4f} / {diff.mean():.4f}")
    
    for name, arr in [("skimage", sk_radon), ("ecc", ecc_radon), ("diff", diff)]:
        Image.fromarray(norm(arr)).save(f"output/test_radon_transform_{name}.png")

if __name__ == "__main__":
    main()
