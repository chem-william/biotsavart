import argparse
from pathlib import Path

import numpy as np
import numpy.typing as npt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from numba import njit, prange
from numba.typed import Dict
from numba.core import types
from tqdm import tqdm

au2A = 0.529177249
# Convert currents from atomic units (AU/Bohr**2) to ampere/m**2
# Breakdown of factors:
#   6.623618183e-3    : AU => ampere
#   au2A**(-2)        : Bohr**-2 => Å**-2
#   1/(1e-10)**2      : Å**-2 => m**-2  (1 Å = 1e-10 m)
# Combined constant = 6.623618183e-3 * au2A**(-2) / 1e-20
CONVERSION = 6.623618183e-3 * au2A ** (-2) / 1e-20


def export_jmol(bx, by, bz, x_cor, y_cor, z_cor, path: str, with_mol: bool):
    colors = [
        [140 / 255.0, 0, 255 / 255.0],
        [1, 1, 1],
        [255 / 255.0, 165 / 255.0, 0],
    ]  # R -> G -> B
    n_bins = [201]  # Discretizes the interpolation into bins
    cmap_name = "my_list"
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins[0])
    z_colorlist = []
    for n in np.arange(n_bins[0]):
        z_colorlist.append(list(cm(n))[:-1])

    with open(path, "w") as file:
        if with_mol:
            file.write('load "file:$SCRIPT_PATH$/central_region.xyz" \n')
            file.write('write "$SCRIPT_PATH$/central_region2.xyz" \n')
            file.write('load "file:$SCRIPT_PATH$/central_region2.xyz" \n')
        max = np.sqrt(bx[:, :, :] ** 2 + by[:, :, :] ** 2 + bz[:, :, :] ** 2).max()
        print("max b=", max)
        lengths = []
        for ix in range(len(x_cor)):
            for iy in range(len(y_cor)):
                for iz in range(len(z_cor)):
                    vec = np.linalg.norm(
                        [bx[ix, iy, iz], by[ix, iy, iz], bz[ix, iy, iz]]
                    )
                    lengths.append(vec)
                    # if np.round(vec, decimals=14) == np.round(max, decimals=14):
                    #     print("max coord =", [x_cor[ix], y_cor[iy], z_cor[iz]])
        max_length = np.max(lengths)
        min_length = np.min(lengths)
        co = np.max(np.abs(lengths)) * 0.22  # cutoff

        amp = 1 / (
            3 * np.sqrt(bx[:, :, :] ** 2 + by[:, :, :] ** 2 + bz[:, :, :] ** 2).max()
        )

        if max_length == min_length:
            print(f"min and max are equal.\n Max: {max}\n Min: {min}")
        else:
            lengths = (lengths - min_length) / (max_length - min_length)
        z_list = []
        a = 0  # arrow_index
        size = 4
        for ix, x in enumerate(x_cor):
            for iy, y in enumerate(y_cor):
                for iz, z in enumerate(z_cor):
                    norm2 = np.sqrt(
                        bx[ix, iy, iz] ** 2 + by[ix, iy, iz] ** 2 + bz[ix, iy, iz] ** 2
                    )
                    if norm2 > co:
                        rel_z = bz[ix, iy, iz] / norm2
                        z_color = z_colorlist[
                            int(np.round(rel_z, decimals=2) * 100) + 100
                        ]
                        z_list.append(
                            "draw arrow{0} arrow color {8} diameter {7} {{ {1},{2},{3} }} {{ {4},{5},{6} }} \n".format(
                                a,
                                x - bx[ix, iy, iz] / (size * norm2),
                                y - by[ix, iy, iz] / (size * norm2),
                                z - bz[ix, iy, iz] / (size * norm2),
                                (x + bx[ix, iy, iz] / (size * norm2)),
                                (y + by[ix, iy, iz] / (size * norm2)),
                                (z + bz[ix, iy, iz] / (size * norm2)),
                                norm2 * amp,
                                z_color,
                            )
                        )
                    a += 1

        file.writelines(z_list)
        file.write("set defaultdrawarrowscale 0.1 \n")
        file.write("rotate 90 \n")
        file.write("background white \n")


@njit(parallel=True)
def biot_savart_parallel(jx, jy, jz, x_cor, y_cor, z_cor):
    """
    Parallelized calculation of the magnetic field via the Biot-Savart law.
    Uses Numba to parallelize over destination points.

    Parameters
    ----------
    jx, jy, jz : ndarray
        Current density components on a 3D grid of shape (M, N, K).
    x_cor, y_cor, z_cor : 1D array_like
        Coordinates corresponding to each grid axis.

    Returns
    -------
    Bx, By, Bz : ndarray
        Magnetic field components on the grid (each with shape (M, N, K)).
    """
    mu0 = 4 * np.pi * 1e-7  # T * m/A

    x_cor /= 1e10  # Å -> m
    y_cor /= 1e10  # Å -> m
    z_cor /= 1e10  # Å -> m
    dx = np.abs(x_cor[1] - x_cor[0])
    dy = np.abs(y_cor[1] - y_cor[0])
    dz = np.abs(z_cor[1] - z_cor[0])
    dV = dx * dy * dz

    M = len(x_cor)
    N = len(y_cor)
    K = len(z_cor)

    Bx = np.zeros((M, N, K))
    By = np.zeros((M, N, K))
    Bz = np.zeros((M, N, K))

    # Parallel loop over all destination grid points
    for i in prange(M):
        for j in range(N):
            for k in range(K):
                # Coordinates for the destination (field) point.
                source_point = np.array([x_cor[i], y_cor[j], z_cor[k]])

                for m in range(M):
                    for n in range(N):
                        for p in range(K):
                            inner_x = x_cor[m]
                            inner_y = y_cor[n]
                            inner_z = z_cor[p]
                            obs_point = np.array([inner_x, inner_y, inner_z])

                            j_vec = np.array([jx[m, n, p], jy[m, n, p], jz[m, n, p]])
                            r_vec = source_point - obs_point
                            r_norm = np.linalg.norm(r_vec)
                            if r_norm == 0:
                                continue

                            dB = np.cross(j_vec, r_vec) / (r_norm**3)
                            Bx[i, j, k] += dB[0] * dV
                            By[i, j, k] += dB[1] * dV
                            Bz[i, j, k] += dB[2] * dV
    factor = mu0 / (4 * np.pi)
    Bx *= factor
    By *= factor
    Bz *= factor

    return Bx, By, Bz


def calculate_magnetic_field(
    path: Path,
) -> tuple[
    npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray
]:
    jx, jy, jz, x_cor, y_cor, z_cor = np.load(path, allow_pickle=True)
    SKIP = 1

    x_cor *= au2A
    x_cor = x_cor[::SKIP]
    y_cor *= au2A
    y_cor = y_cor[::SKIP]
    z_cor *= au2A
    z_cor = z_cor[::SKIP]
    # j_xyz : atomic unit/Bohr**2 -> ampere/Bohr**2 -> ampere/Å**2 -> ampere/m**2
    jx *= CONVERSION
    jx = jx[::SKIP, ::SKIP, ::SKIP]
    jy *= CONVERSION
    jy = jy[::SKIP, ::SKIP, ::SKIP]
    jz *= CONVERSION
    jz = jz[::SKIP, ::SKIP, ::SKIP]

    Bx, By, Bz = biot_savart_parallel(
        jx, jy, jz, x_cor.copy(), y_cor.copy(), z_cor.copy()
    )

    return Bx, By, Bz, x_cor, y_cor, z_cor


def main():
    parser = argparse.ArgumentParser(
        description="Calculate magnetic field using Biot-Savart law."
    )
    parser.add_argument("--input", default="./", help="Path to current_c_all.npy")
    parser.add_argument("--output", help="Path to output .spt")

    args = parser.parse_args()
    path = Path(args.input)
    output = Path(args.output)

    Bx, By, Bz, x_cor, y_cor, z_cor = calculate_magnetic_field(path)

    print(f"{np.max(Bx) = :.6}")
    print(f"{np.max(By) = :.6}")
    print(f"{np.max(Bz) = :.6}")
    print("=" * 25)

    np.save("./bx.npy", Bx)
    np.save("./by.npy", By)
    np.save("./bz.npy", Bz)

    export_jmol(Bx, By, Bz, x_cor, y_cor, z_cor, path=output, with_mol=True)


if __name__ == "__main__":
    exit(main())
