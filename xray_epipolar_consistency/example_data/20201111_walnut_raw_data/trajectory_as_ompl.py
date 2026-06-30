import numpy as np

def walnut_trajectory(
    detector_size_px,
    frameskip=1,
    binning=1,
):
    DSD = 553.74  # mm
    DSO = 110.66  # mm

    pixel_size = 0.050 * binning  # mm

    nx, ny = detector_size_px
    cx = (nx - 1) / 2
    cy = (ny - 1) / 2

    fx = fy = DSD / pixel_size

    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1 ],
    ])

    mats = []

    angles = np.arange(720)[::frameskip] * 0.5

    for a in -np.deg2rad(angles):

        s = np.array([
            DSO * np.cos(a),
            DSO * np.sin(a),
            0.0,
        ])

        z = -s / np.linalg.norm(s)

        up = np.array([0.0, 0.0, 1.0])

        x = np.cross(up, z)
        x /= np.linalg.norm(x)

        y = -np.cross(z, x)

        R = np.vstack([x, y, z])

        t = -R @ s

        P = K @ np.column_stack([R, t])

        mats.append(P)

    return mats



def write_ompl(filename, mats, detector_size_px, binning):

    spacing = 0.050 * binning
    nx, ny = detector_size_px

    with open(filename, "w") as f:

        f.write(
            f'#> spacing="{spacing}" '
            f'detector_size_px="{nx} {ny}"\n'
        )

        for P in mats:

            rows = [
                " ".join(f"{v:.12g}" for v in row)
                for row in P
            ]

            f.write(
                "[" +
                "; ".join(rows) +
                "]\n"
            )


def main():

    configs = [
        (720, 1),
        (720, 2),
        (720, 4),
        (360, 4),
    ]

    ORIG_W = 2240
    ORIG_H = 2368

    for nviews, binning in configs:

        nx = ORIG_W // binning
        ny = ORIG_H // binning

        mats = walnut_trajectory(
            detector_size_px=(nx, ny),
            frameskip=720 // nviews,
            binning=binning,
        )

        write_ompl(
            f"trajectory_{nviews}_{binning}x{binning}.ompl",
            mats,
            detector_size_px=(nx, ny),
            binning=binning,
        )

        print(
            f"Wrote trajectory_{nviews}_{binning}x{binning}.ompl"
        )
    


if __name__ == "__main__":
    main()
