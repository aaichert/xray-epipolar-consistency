import sys
import os
import json
import numpy as np
import nrrd
import matplotlib.pyplot as plt
import math
import scipy.ndimage as ndimage
from PIL import Image

from ProjectiveGeometry23.central_projection import ProjectionMatrix
from ProjectiveGeometry23.svg_utils import svg_homogeneous_line
from ProjectiveGeometry23.utils import hessianNormalForm
from ProjectiveGeometry23 import pluecker
from svg_snip.Composer import Composer
from svg_snip.Elements import rect, polyline, text

sys.path = [p for p in sys.path if p != '.']
from xray_epipolar_consistency import _core as ecc

examples_dir = os.path.dirname(os.path.abspath(__file__))
config_file = sys.argv[1] if len(sys.argv) > 1 else "plot_redundancy.json"
config_path = os.path.join(examples_dir, config_file) if not os.path.isabs(config_file) else config_file
with open(config_path, 'r') as f:
    config = json.load(f)

output_dir_name = config.get("output_dir", "output")
output_dir = os.path.join(examples_dir, output_dir_name) if not os.path.isabs(output_dir_name) else output_dir_name
os.makedirs(output_dir, exist_ok=True)

image0_path = config["image0_path"]
image1_path = config["image1_path"]
if not os.path.isabs(image0_path):
    image0_path = os.path.join(examples_dir, image0_path)
if not os.path.isabs(image1_path):
    image1_path = os.path.join(examples_dir, image1_path)
object_radius = config["object_radius"]

print(f"Loading {image0_path} and {image1_path}...")

raw_I0, header0 = nrrd.read(image0_path)
raw_I1, header1 = nrrd.read(image1_path)
raw_I0 = raw_I0.T.astype(np.float32)
raw_I1 = raw_I1.T.astype(np.float32)

P0 = ProjectionMatrix(np.matrix(header0['Projection Matrix']))
P1 = ProjectionMatrix(np.matrix(header1['Projection Matrix']))

C0 = P0.getCenterOfProjection().flatten()
C1 = P1.getCenterOfProjection().flatten()

B = pluecker.join_points(C0, C1)
Bx_dual = pluecker.matrixDual(B)
E0 = pluecker.join(B, np.array([0, 0, 0, 1]))
E90 = Bx_dual @ Bx_dual @ np.array([0, 0, 0, 1])
E0 = hessianNormalForm(E0).flatten()
E90 = hessianNormalForm(E90).flatten()
K = np.column_stack([E0, E90])

size_t = int(np.ceil(np.hypot(raw_I0.shape[0], raw_I0.shape[1]))) // 2
size_alpha = int(np.ceil((np.pi / 2.0) * size_t))

raw_dtr0 = ecc.RadonIntermediate(raw_I0.copy(), size_alpha, size_t, int(getattr(ecc.RadonFilter, "Derivative")), int(getattr(ecc.RadonPostProcess, "Identity")))
raw_dtr1 = ecc.RadonIntermediate(raw_I1.copy(), size_alpha, size_t, int(getattr(ecc.RadonFilter, "Derivative")), int(getattr(ecc.RadonPostProcess, "Identity")))

print("Running C++ Implementation...")
cost, v0s, v1s, kappas, _ = ecc.compute_for_image_pair(
    P0.P, P1.P, raw_dtr0, raw_dtr1, 
    config.get("num_planes", 1800),
    config.get("object_radius", 0.0)
)
print(f"Metric Cost: {cost}")


min_I = min(raw_I0.min(), raw_I1.min())
max_I = max(raw_I0.max(), raw_I1.max())
I0 = Image.fromarray(np.clip((raw_I0 - min_I) / (max_I - min_I + 1e-8) * 255, 0, 255).astype(np.uint8))
I1 = Image.fromarray(np.clip((raw_I1 - min_I) / (max_I - min_I + 1e-8) * 255, 0, 255).astype(np.uint8))

raw_dtr0_data = raw_dtr0.get_data()
raw_dtr1_data = raw_dtr1.get_data()
min_dtr = min(raw_dtr0_data.min(), raw_dtr1_data.min())
max_dtr = max(raw_dtr0_data.max(), raw_dtr1_data.max())
dtr0 = Image.fromarray(np.clip((raw_dtr0_data - min_dtr) / (max_dtr - min_dtr + 1e-8) * 255, 0, 255).astype(np.uint8))
dtr1 = Image.fromarray(np.clip((raw_dtr1_data - min_dtr) / (max_dtr - min_dtr + 1e-8) * 255, 0, 255).astype(np.uint8))

def process_view(color, color_other, I, dtr, raw_dtr_data, P, P_other, kappas, K, out_suffix):
    w, h = I.size
    size_t = int(np.ceil(np.hypot(h, w))) // 2
    size_alpha = int(np.ceil((np.pi / 2.0) * size_t))
    range_t = np.hypot(w, h)

    P_invT = P.pseudoinverse().T

    step = max(1, len(kappas) // 25)
    kappas_py = kappas[::step]

    I_svg = Composer(I)
    I_svg.add(rect, x=0, y=0, width=w, height=h, stroke=color, stroke_width=5, fill="none")

    for idx, k in enumerate(kappas_py):
        E_kappa = K @ [np.cos(k), np.sin(k)]
        l = (P_invT @ E_kappa).flatten()
        l = hessianNormalForm(l).flatten()
        I_svg.add(svg_homogeneous_line, l=l, stroke=color_other, stroke_width=1.5)

        if idx % 4 == 0:
            a_line, b_line, c_line = l
            if np.abs(b_line) > 1e-6:
                x_txt = w * 0.5
                y_txt = -(a_line * x_txt + c_line) / b_line
                if -10 <= y_txt <= h + 10:
                    I_svg.add(text, x=x_txt, y=y_txt, content=f".   {np.degrees(k):.1f}°", fill="white", font_size=14)

    dtr_svg = Composer(dtr)
    pts_list = []
    vals_py = []

    def map_line_to_Radon_space(l):
        T_center_star = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [w * 0.5, h * 0.5, 1.0]
        ])
        l_centered = hessianNormalForm(T_center_star @ l).flatten()

        a_val = np.atan2(l_centered[1], l_centered[0]) / np.pi
        if a_val < 0:
            a_val += 2.0
        t_val = -l_centered[2] / range_t + 0.5
        sign = 1.0
        if a_val > 1.0:
            a_val -= 1.0
            t_val = 1.0 - t_val
            sign = -1.0
        t_idx = t_val * (size_t - 1)
        a_idx = a_val * (size_alpha - 1)
        return a_idx, t_idx, sign

    for idx, k in enumerate(kappas_py):
        E_kappa = K @ [np.cos(k), np.sin(k)]
        l = (P_invT @ E_kappa).flatten()

        a_idx, t_idx, sign = map_line_to_Radon_space(l)
        pts_list.append(f"{a_idx},{t_idx}")

        if idx % 4 == 0:
            dtr_svg.add(text, x=a_idx, y=t_idx, content=f".   {np.degrees(k):.1f}°", fill="black", font_size=14)

    py_coords = [map_line_to_Radon_space((P_invT @ (K @ [np.cos(k), np.sin(k)])).flatten()) for k in kappas]
    a_idxs = [c[0] for c in py_coords]
    t_idxs = [c[1] for c in py_coords]
    signs = [c[2] for c in py_coords]

    vals_py = ndimage.map_coordinates(raw_dtr_data, [t_idxs, a_idxs], order=1, mode='nearest')
    vals_py = list(np.array(signs) * vals_py)

    pts = " ".join(pts_list)
    dtr_svg.add(polyline, points=pts, stroke=color_other, stroke_width=2, fill="none")

    with open(os.path.join(output_dir, f"plot_redundancy_image{out_suffix}.svg"), "w") as f:
        f.write(I_svg.render())
    with open(os.path.join(output_dir, f"plot_redundancy_Radon_space_{out_suffix}.svg"), "w") as f:
        f.write(dtr_svg.render())

    return vals_py

v0s_py = process_view("red", "green", I0, dtr0, raw_dtr0_data, P0, P1, kappas, K, "0")
v1s_py = process_view("green", "red", I1, dtr1, raw_dtr1_data, P1, P0, kappas, K, "1")

plt.figure(figsize=(12, 6))
plt.plot(np.degrees(kappas), v0s[:len(kappas)], label='Redundancies Image 0 (C++)', color='red', alpha=0.7)
plt.plot(np.degrees(kappas), v1s[:len(kappas)], label='Redundancies Image 1 (C++)', color='green', linestyle='dotted', alpha=0.7)

v0s_py_centered = np.array(v0s_py) - np.mean(v0s_py)
v1s_py_centered = np.array(v1s_py) - np.mean(v1s_py)
scale_factor = np.max(np.abs(v0s[:len(kappas)])) / (np.max(np.abs(v0s_py_centered)) + 1e-8)

plt.plot(np.degrees(kappas), v0s_py_centered * scale_factor, label='Redundancies Image 0 (Python)', color='black', alpha=0.7)
plt.plot(np.degrees(kappas), v1s_py_centered * scale_factor, label='Redundancies Image 1 (Python)', color='blue', linestyle='dotted', alpha=0.7)



plt.xlabel('Kappa Angle (degrees)')
plt.ylabel('[a.u.]')
plt.yticks([])
plt.title('Epipolar Consistency Redundancy Signals')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'plot_redundancy.pdf'))
plt.savefig(os.path.join(output_dir, 'plot_redundancy.png'))
