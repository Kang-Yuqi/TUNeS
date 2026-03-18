import numpy as np
from typing import Optional
import h5py
import argparse
import os


def cic_density(
    pos: np.ndarray,
    box_size: float,
    grid_size: int,
    part_mass: Optional[np.ndarray] = None,
    dtype=np.float32,
):
    """
    Numpy version of CIC density assignment with periodic BCs.
    pos: (N, 3)
    return: (grid_size, grid_size, grid_size)
    """
    x = np.asarray(pos, dtype=dtype)
    x = np.mod(x, box_size)

    N = x.shape[0]

    if part_mass is None:
        w = np.ones(N, dtype=dtype)
    else:
        w = np.asarray(part_mass, dtype=dtype)

    # normalized grid coords: [0, grid_size)
    u = x / box_size * grid_size

    i0 = np.floor(u).astype(np.int64) % grid_size
    du = u - i0
    i1 = (i0 + 1) % grid_size

    # CIC weights
    wx0 = 1.0 - du[:, 0]
    wx1 = du[:, 0]
    wy0 = 1.0 - du[:, 1]
    wy1 = du[:, 1]
    wz0 = 1.0 - du[:, 2]
    wz1 = du[:, 2]

    wt000 = wx0 * wy0 * wz0
    wt100 = wx1 * wy0 * wz0
    wt010 = wx0 * wy1 * wz0
    wt110 = wx1 * wy1 * wz0
    wt001 = wx0 * wy0 * wz1
    wt101 = wx1 * wy0 * wz1
    wt011 = wx0 * wy1 * wz1
    wt111 = wx1 * wy1 * wz1

    def lin_idx(idx):
        return (idx[:, 0] * grid_size + idx[:, 1]) * grid_size + idx[:, 2]

    rho = np.zeros(grid_size * grid_size * grid_size, dtype=dtype)

    for idx, wt in [
        (i0, wt000),
        (np.column_stack([i1[:, 0], i0[:, 1], i0[:, 2]]), wt100),
        (np.column_stack([i0[:, 0], i1[:, 1], i0[:, 2]]), wt010),
        (np.column_stack([i1[:, 0], i1[:, 1], i0[:, 2]]), wt110),
        (np.column_stack([i0[:, 0], i0[:, 1], i1[:, 2]]), wt001),
        (np.column_stack([i1[:, 0], i0[:, 1], i1[:, 2]]), wt101),
        (np.column_stack([i0[:, 0], i1[:, 1], i1[:, 2]]), wt011),
        (np.column_stack([i1[:, 0], i1[:, 1], i1[:, 2]]), wt111),
    ]:
        lin = lin_idx(idx)
        np.add.at(rho, lin, w * wt)

    rho = rho.reshape(grid_size, grid_size, grid_size)

    voxel_vol = (box_size / grid_size) ** 3
    rho = rho / voxel_vol

    return rho


class DataLoader:
    def __init__(self, path):
        self.path = path

    def load_nbody(self):
        with h5py.File(self.path, "r") as file:
            part = file["PartType1"]
            pos = part["Coordinates"][()]
            vel = part["Velocities"][()]
            pid = part["ParticleIDs"][()] - 1

            header = file["Header"]
            box_size = float(header.attrs["BoxSize"])
            redshift = float(header.attrs["Redshift"])
            num_particles = int(header.attrs["NumPart_Total"][1])
            mass = float(header.attrs["MassTable"][1])

        sort = np.argsort(pid)
        return {
            "pos": pos[sort],
            "vel": vel[sort],
            "id": pid[sort],
            "box_size": box_size,
            "redshift": redshift,
            "num_particles": num_particles,
            "mass": mass,
        }


def save_rho(path_out, rho, box_size, redshift):
    outdir = os.path.dirname(path_out)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    with h5py.File(path_out, "w") as f:
        f.create_dataset("rho", data=rho, compression="gzip")
        f.attrs["box_size"] = box_size
        f.attrs["redshift"] = redshift
        f.attrs["grid_size"] = rho.shape[0]


def main():
    parser = argparse.ArgumentParser(description="Convert one N-body snapshot to CIC density grid.")
    parser.add_argument("input", type=str, help="Input snapshot hdf5 file")
    parser.add_argument("--grid", type=int, required=True, help="Grid size, e.g. 256")
    parser.add_argument("--out", type=str, required=True, help="Output h5 file path")
    args = parser.parse_args()

    data = DataLoader(args.input).load_nbody()

    pos = data["pos"]
    box_size = data["box_size"]
    redshift = data["redshift"]
    mass = data["mass"]

    # equal-mass particles
    part_mass = np.full(pos.shape[0], mass, dtype=np.float32)

    rho = cic_density(
        pos=pos,
        box_size=box_size,
        grid_size=args.grid,
        part_mass=part_mass,
        dtype=np.float32,
    )

    save_rho(args.out, rho, box_size, redshift)
    print(f"[OK] saved rho to: {args.out}")


if __name__ == "__main__":
    main()


# ############ bash script #############
# for i in $(seq -f "%03g" 0 1); do
#   python pos2dens.py \
#     "/mnt/f/research/SimML/Gadget-sim/simulations_256_500/nbody2/snapshot_${i}.hdf5" \
#     --grid 256 \
#     --out "/mnt/f/research/SimML/Gadget-sim/simulations_256_500/rho_grid256/nbody2/rho_${i}.h5"
# done