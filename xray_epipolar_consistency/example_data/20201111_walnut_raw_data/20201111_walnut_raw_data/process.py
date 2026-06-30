from pathlib import Path
import numpy as np
import tifffile
import nrrd
from tqdm import tqdm

files = sorted(Path(".").glob("*.tif"))

# Drop the duplicated 360° view
files = files[:-1]

first = tifffile.imread(files[0])
h, w = first.shape

volume_2x2 = np.empty((len(files), h // 2, w // 2), dtype=np.float32)

pbar = tqdm(enumerate(files), total=len(files), unit="file")
for i, f in pbar:
    pbar.set_postfix_str(f.name)

    img = tifffile.imread(f)

    volume_2x2[i] = (
        img[: h // 2 * 2, : w // 2 * 2]
        .reshape(h // 2, 2, w // 2, 2)
        .mean((1, 3), dtype=np.float32)
    )

nrrd.write(
    "../walnut_720_2x2.nrrd",
    np.transpose(volume_2x2, (2, 1, 0)),
    header={"kinds": ["domain", "domain", "list"], "encoding": "raw"},
)

volume_4x4 = (
    volume_2x2[
        :,
        : volume_2x2.shape[1] // 2 * 2,
        : volume_2x2.shape[2] // 2 * 2,
    ]
    .reshape(
        volume_2x2.shape[0],
        volume_2x2.shape[1] // 2,
        2,
        volume_2x2.shape[2] // 2,
        2,
    )
    .mean((2, 4), dtype=np.float32)
)

nrrd.write(
    "../walnut_720_4x4.nrrd",
    np.transpose(volume_4x4, (2, 1, 0)),
    header={"kinds": ["domain", "domain", "list"], "encoding": "raw"},
)

nrrd.write(
    "../walnut_360_4x4.nrrd",
    np.transpose(volume_4x4[::2], (2, 1, 0)),
    header={"kinds": ["domain", "domain", "list"], "encoding": "raw"},
)

print("720 2x2:", volume_2x2.shape)
print("720 4x4:", volume_4x4.shape)
print("360 4x4:", volume_4x4[::2].shape)
