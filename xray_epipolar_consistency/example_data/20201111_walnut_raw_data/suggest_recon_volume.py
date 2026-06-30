import numpy as np
from scipy.optimize import linprog
from itertools import product


def reconstruction_box(projections, detector_size):
    W, H = detector_size

    # detector edge lines
    lines = [
        np.array([ 1,  0,      0]),      # u >= 0
        np.array([-1,  0,  W - 1]),      # u <= W-1
        np.array([ 0,  1,      0]),      # v >= 0
        np.array([ 0, -1,  H - 1]),      # v <= H-1
    ]

    planes = []

    for P in projections:
        for l in lines:
            pi = P.T @ l

            # orient plane so isocenter lies inside
            planes.append(pi)

    # estimate isocenter from source positions
    sources = np.array([
        -np.linalg.inv(P[:, :3]) @ P[:, 3]
        for P in projections
    ])

    center = sources.mean(axis=0)

    # make all planes point inward
    oriented_planes = []

    for pi in planes:
        if pi[:3] @ center + pi[3] < 0:
            pi = -pi
        oriented_planes.append(pi)

    # variables:
    # hx hy hz
    #
    # box:
    # center ± (hx,hy,hz)

    A = []
    b = []

    signs = np.array(list(product([-1, 1], repeat=3)))

    for pi in oriented_planes:

        a = pi[:3]
        d = pi[3]

        margin = a @ center + d

        for s in signs:

            # a·(center+s*h)+d >= 0
            #
            # -(a*s)·h <= margin

            A.append(-(a * s))
            b.append(margin)

    A = np.asarray(A)
    b = np.asarray(b)

    # maximize hx+hy+hz
    c = np.array([-1.0, -1.0, -1.0])

    result = linprog(
        c,
        A_ub=A,
        b_ub=b,
        bounds=[(0, None)] * 3,
        method="highs",
    )

    if not result.success:
        raise RuntimeError(result.message)

    h = result.x

    min_corner = center - h
    physical_size = 2 * h

    return min_corner, physical_size
    
   
import sys


def load_ompl(filename):
    projections = []

    with open(filename) as f:
        header = f.readline()

        detector_size = tuple(
            map(
                int,
                header.split('detector_size_px="')[1]
                      .split('"')[0]
                      .split()
            )
        )

        for line in f:
            line = line.strip()
            if not line:
                continue

            rows = [
                list(map(float, row.split()))
                for row in line[1:-1].split(";")
            ]

            projections.append(np.array(rows))

    return projections, detector_size


def suggest_volume(projections, detector_size):
    min_corner, physical_size = reconstruction_box(
        projections,
        detector_size,
    )

    sources = np.array([
        -np.linalg.inv(P[:, :3]) @ P[:, 3]
        for P in projections
    ])

    isocenter = sources.mean(axis=0)

    W, H = detector_size
    u0 = (W - 1) / 2
    v0 = (H - 1) / 2

    pitches = []

    for P, source in zip(projections, sources):

        P4 = np.vstack([P, [0, 0, 0, 1]])
        Pinv = np.linalg.pinv(P4)

        p0 = Pinv @ [u0,     v0, 1, 1]
        p1 = Pinv @ [u0 + 1, v0, 1, 1]

        p0 = p0[:3] / p0[3]
        p1 = p1[:3] / p1[3]

        d0 = p0 - source
        d1 = p1 - source

        d0 /= np.linalg.norm(d0)
        d1 /= np.linalg.norm(d1)

        n = isocenter - source
        n /= np.linalg.norm(n)

        t0 = np.dot(isocenter - source, n) / np.dot(d0, n)
        t1 = np.dot(isocenter - source, n) / np.dot(d1, n)

        x0 = source + t0 * d0
        x1 = source + t1 * d1

        pitches.append(np.linalg.norm(x1 - x0))

    voxel_spacing = np.mean(pitches)

    volume_shape = np.ceil(
        physical_size / voxel_spacing
    ).astype(int)

    return (
        min_corner,
        physical_size,
        voxel_spacing,
        volume_shape,
    )


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} trajectory.ompl")
        sys.exit(1)

    projections, detector_size = load_ompl(sys.argv[1])

    (
        min_corner,
        physical_size,
        voxel_spacing,
        volume_shape,
    ) = suggest_volume(
        projections,
        detector_size,
    )

    
    # Make numbers behave more nicely.
    voxel_spacing = float(f"{voxel_spacing:.3g}")
    volume_shape = (
        32 * np.ceil(volume_shape / 32)
    ).astype(int)

    max_corner = min_corner + physical_size

    model_matrix = np.eye(4)
    model_matrix[:3, :3] *= voxel_spacing
    model_matrix[:3, 3] = min_corner

    print()
    print("Suggested Reconstruction Box")
    print("------------------")
    print("min_corner    :", min_corner)
    print("max_corner    :", max_corner)
    print("physical_size :", physical_size)
    print()

    print("Suggested volume")
    print("----------------")
    print("voxel_spacing :", voxel_spacing)
    print("volume_shape  :", volume_shape.tolist())
    print()

    print("Model matrix")
    print("------------")
    print(model_matrix)


if __name__ == "__main__":
    main()
    
