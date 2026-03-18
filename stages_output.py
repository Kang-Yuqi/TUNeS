import os
import glob
import yaml
import h5py
import numpy as np
import torch
from typing import Optional, Tuple, Dict, Any

from utils.Nbody_data_loader import NbodyLoader


# ----------------------------
# YAML helpers
# ----------------------------
def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def resolve_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    project_root = cfg["paths"]["root"]

    def _resolve_dict(d):
        for k, v in d.items():
            if isinstance(v, str) and ("dir" in k or "path" in k):
                if not os.path.isabs(v):
                    d[k] = os.path.join(project_root, v)
            elif isinstance(v, dict):
                _resolve_dict(v)

    _resolve_dict(cfg)
    return cfg


# ----------------------------
# CIC density
# ----------------------------
@torch.no_grad()
def cic_density(pos: np.ndarray,
                box_size: float,
                grid_size: int,
                part_mass: Optional[np.ndarray] = None,
                device: str = "cuda",
                dtype: torch.dtype = torch.float32) -> torch.Tensor:
    x = torch.as_tensor(pos, device=device, dtype=dtype)
    x = torch.remainder(x, box_size)

    if part_mass is None:
        w = torch.ones(x.size(0), device=device, dtype=dtype)
    else:
        w = torch.as_tensor(part_mass, device=device, dtype=dtype)

    u  = x / box_size * float(grid_size)  # [0, grid_size)
    i0 = torch.floor(u).to(torch.long) % grid_size
    du = (u - i0.to(u.dtype))
    i1 = (i0 + 1) % grid_size

    wx0, wy0, wz0 = (1.0 - du[:,0]), (1.0 - du[:,1]), (1.0 - du[:,2])
    wx1, wy1, wz1 = du[:,0], du[:,1], du[:,2]

    idx000 = (i0[:,0], i0[:,1], i0[:,2]); wt000 = wx0 * wy0 * wz0
    idx100 = (i1[:,0], i0[:,1], i0[:,2]); wt100 = wx1 * wy0 * wz0
    idx010 = (i0[:,0], i1[:,1], i0[:,2]); wt010 = wx0 * wy1 * wz0
    idx110 = (i1[:,0], i1[:,1], i0[:,2]); wt110 = wx1 * wy1 * wz0
    idx001 = (i0[:,0], i0[:,1], i1[:,2]); wt001 = wx0 * wy0 * wz1
    idx101 = (i1[:,0], i0[:,1], i1[:,2]); wt101 = wx1 * wy0 * wz1
    idx011 = (i0[:,0], i1[:,1], i1[:,2]); wt011 = wx0 * wy1 * wz1
    idx111 = (i1[:,0], i1[:,1], i1[:,2]); wt111 = wx1 * wy1 * wz1

    def lin_idx(idx):
        return (idx[0] * grid_size + idx[1]) * grid_size + idx[2]

    rho = torch.zeros(grid_size**3, device=device, dtype=dtype)
    for lin, wt in [(lin_idx(idx000), wt000), (lin_idx(idx100), wt100),
                    (lin_idx(idx010), wt010), (lin_idx(idx110), wt110),
                    (lin_idx(idx001), wt001), (lin_idx(idx101), wt101),
                    (lin_idx(idx011), wt011), (lin_idx(idx111), wt111)]:
        rho.scatter_add_(0, lin, w * wt)

    rho = rho.view(grid_size, grid_size, grid_size)
    voxel_vol = (box_size / float(grid_size)) ** 3
    rho = rho / voxel_vol
    return rho


# ----------------------------
# Load stage1
# ----------------------------
def load_pos_and_box(stage1_path: str) -> Tuple[np.ndarray, float]:
    sp = stage1_path.lower()
    if sp.endswith(".h5") or sp.endswith(".hdf5"):
        try:
            with h5py.File(stage1_path, "r") as f:
                if "pos_pred" in f:
                    pos = np.asarray(f["pos_pred"][:], dtype=np.float32)
                elif "pos" in f:
                    pos = np.asarray(f["pos"][:], dtype=np.float32)
                else:
                    raise KeyError("no 'pos_pred' or 'pos' in stage1 file")
                # attrs
                if "box_size" in f.attrs:
                    L = float(f.attrs["box_size"])
                elif "BoxSize" in f.attrs:
                    L = float(f.attrs["BoxSize"])
                else:
                    # fallback: some files store it as dataset
                    if "box_size" in f:
                        L = float(np.asarray(f["box_size"][()]))
                    else:
                        raise KeyError("cannot find box_size in stage1 file attrs/datasets")
            return pos, L
        except OSError:
            pass

    # Gadget-style snapshot
    nbody = NbodyLoader(stage1_path).load_nbody()
    pos = nbody["pos"].astype(np.float32)
    L = float(nbody["box_size"])
    return pos, L


# ----------------------------
# Load stage2 rho robustly
# (handles: .h5/.hdf5, .npy, .npz, .pt, or directory with a matching file)
# ----------------------------
import os
import glob
from typing import Optional

import h5py
import numpy as np
import torch


def _downsample_3d_block_average(arr: np.ndarray, target_grid: int) -> np.ndarray:

    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim < 3:
        raise ValueError(f"arr.ndim must be >= 3, got {arr.ndim}")

    nx, ny, nz = arr.shape[-3:]
    if not (nx == ny == nz):
        raise ValueError(f"last 3 dims must be cubic, got {arr.shape[-3:]}")

    src_grid = nx
    if target_grid is None or target_grid == src_grid:
        return arr

    if src_grid % target_grid != 0:
        raise ValueError(
            f"target_grid={target_grid} must divide source grid={src_grid}"
        )

    fac = src_grid // target_grid

    new_shape = arr.shape[:-3] + (
        target_grid, fac,
        target_grid, fac,
        target_grid, fac
    )
    arr = arr.reshape(new_shape).mean(axis=(-1, -3, -5))
    return arr.astype(np.float32)


def _squeeze_rho_shape(arr: np.ndarray) -> np.ndarray:

    arr = np.asarray(arr, dtype=np.float32)

    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]

    return arr.astype(np.float32)


def load_stage2_rho(stage2_path: str, target_grid: Optional[int] = None) -> np.ndarray:
    def _try_from_h5(p: str) -> Optional[np.ndarray]:
        try:
            with h5py.File(p, "r") as f:
                for k in ["rho_rec", "rho_pred", "rho", "rho_out", "density", "delta"]:
                    if k in f:
                        return np.asarray(f[k][:], dtype=np.float32)
        except Exception:
            return None
        return None

    p = stage2_path
    if os.path.isdir(p):
        cand = []
        cand += glob.glob(os.path.join(p, "*rho*.h5"))
        cand += glob.glob(os.path.join(p, "*rho*.hdf5"))
        cand += glob.glob(os.path.join(p, "*rho*.npy"))
        cand += glob.glob(os.path.join(p, "*rho*.npz"))
        cand += glob.glob(os.path.join(p, "*rho*.pt"))
        cand += glob.glob(os.path.join(p, "*rho*.pth"))
        if len(cand) == 0:
            raise FileNotFoundError(f"[stage2] cannot find rho-like file under dir: {p}")
        cand.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        p = cand[0]

    pl = p.lower()
    arr = None

    if pl.endswith(".h5") or pl.endswith(".hdf5"):
        arr = _try_from_h5(p)
        if arr is None:
            raise KeyError(f"[stage2] no recognized rho key in: {p}")

    elif pl.endswith(".npy"):
        arr = np.load(p).astype(np.float32)

    elif pl.endswith(".npz"):
        z = np.load(p)
        for k in ["rho_rec", "rho_pred", "rho", "arr_0"]:
            if k in z.files:
                arr = np.asarray(z[k], dtype=np.float32)
                break
        if arr is None:
            arr = np.asarray(z[z.files[0]], dtype=np.float32)

    elif pl.endswith(".pt") or pl.endswith(".pth"):
        obj = torch.load(p, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            arr = obj.detach().cpu().numpy().astype(np.float32)
        elif isinstance(obj, dict):
            for k in ["rho_rec", "rho_pred", "rho", "density", "delta"]:
                if k in obj:
                    t = obj[k]
                    if isinstance(t, torch.Tensor):
                        arr = t.detach().cpu().numpy().astype(np.float32)
                    else:
                        arr = np.asarray(t, dtype=np.float32)
                    break
            if arr is None:
                raise KeyError(f"[stage2] unrecognized pt content in: {p}")
        else:
            raise KeyError(f"[stage2] unrecognized pt content in: {p}")

    else:
        raise ValueError(f"[stage2] unsupported stage2 rho file type: {p}")

    arr = _squeeze_rho_shape(arr)

    if target_grid is not None:
        arr = _downsample_3d_block_average(arr, target_grid)

    return arr.astype(np.float32)


# ----------------------------
# Save outputs
# ----------------------------
def save_stage_outputs(save_dir: str,
                       z_fin: float,
                       box_size: float,
                       stage1_pos: np.ndarray,
                       stage1_rho: np.ndarray,
                       stage2_rho: np.ndarray,
                       grid_size: int):
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"stage_outputs_z{z_fin:.4f}.h5")

    with h5py.File(out_path, "w") as f:
        f.attrs["z_fin"] = float(z_fin)
        f.attrs["box_size"] = float(box_size)
        f.attrs["grid_size"] = int(grid_size)

        f.create_dataset("stage1/pos", data=stage1_pos, compression="gzip", compression_opts=4)
        f.create_dataset("stage1/rho", data=stage1_rho, compression="gzip", compression_opts=4)
        f.create_dataset("stage2/rho", data=stage2_rho, compression="gzip", compression_opts=4)

    print(f"[OK] saved: {out_path}")


from typing import Optional, Dict, Any


def save_arrays_to_hdf5(
    out_dir: str,
    arrays: Dict[str, np.ndarray],
    box_size: float,
    redshift: Optional[float] = None,
    prefix: str = "model_pred",
    filename_map: Optional[Dict[str, str]] = None,
    compress_level: int = 4,
):

    os.makedirs(out_dir, exist_ok=True)

    for key, arr in arrays.items():

        if arr is None:
            continue

        arr = np.asarray(arr, dtype=np.float32)

        if filename_map and key in filename_map:
            fname = filename_map[key]
        else:
            fname = f"{prefix}_{key}.hdf5"

        out_path = os.path.join(out_dir, fname)

        with h5py.File(out_path, "w") as f:
            f.attrs["box_size"] = float(box_size)
            if redshift is not None:
                f.attrs["redshift"] = float(redshift)

            if arr.ndim == 3:
                f.attrs["grid_size"] = int(arr.shape[0])
                f.create_dataset("rho", data=arr,
                                 compression="gzip",
                                 compression_opts=compress_level)
            else:
                f.create_dataset(key, data=arr,
                                 compression="gzip",
                                 compression_opts=compress_level)

        print(f"[OK] wrote: {out_path}")

# ============================================================
# main flow
# ============================================================

model_dir = "/mnt/f/research/SimML/tunes_mid/projects/TUNeS_example"
input_initial = "/mnt/f/research/SimML/Gadget-sim/simulations_256_500/nbody1/snapshot_000.hdf5"
target_finals = "/mnt/f/research/SimML/Gadget-sim/simulations_256_500/nbody1/snapshot_004.hdf5"
out_root = "/mnt/f/research/SimML/tunes_mid/projects/eval"

# read z_fin
nb_fin = NbodyLoader(target_finals).load_nbody()
z_fin = float(nb_fin["redshift"])

# load cfg
cfg_path = os.path.join(model_dir, "logs", "config_used.yaml")
cfg = resolve_paths(load_yaml(cfg_path))

# run model
from utils.full_out_zdep import NbodyEmulator
emu = NbodyEmulator(model_dir=model_dir, device="cuda")
stage1_pred_path = emu.run_stage1(z_fin, input_initial, output_dir=out_root)
stage2_rec_path  = emu.run_stage2(z_fin, stage1_output=stage1_pred_path, output_dir=out_root)

# stage2 grid config
stage2_prep = cfg["stage2_prepare"]["params"]
grid_res = int(stage2_prep["grid_res"])

pos1, L = load_pos_and_box(stage1_pred_path)

nb_ini = NbodyLoader(input_initial).load_nbody()
pos_ini = torch.tensor(nb_ini["pos"], dtype=torch.float32)
rho0_t = cic_density(pos_ini, box_size=L, grid_size=grid_res, device="cuda", dtype=torch.float32)
rho0 = rho0_t.detach().cpu().numpy().astype(np.float32)


# stage1: load pos -> compute rho
pos1, L = load_pos_and_box(stage1_pred_path)
rho1_t = cic_density(pos=pos1, box_size=L, grid_size=grid_res, device="cuda", dtype=torch.float32)
rho1 = rho1_t.detach().cpu().numpy().astype(np.float32)

# stage2: load rho (from file or dir)
rho2 = load_stage2_rho(stage2_rec_path)
rho128 = load_stage2_rho(stage2_rec_path, target_grid=128)


export_dir = os.path.join(out_root, f"exports_z{z_fin:.1f}")
save_arrays_to_hdf5(
    out_dir=export_dir,
    box_size=L,
    redshift=z_fin,
    arrays={
        "stage1_pos": pos1,
        "stage1_dens_128": rho1,
        "stage2_dens_128": rho2,
    },
    prefix="model_pred",
)

from utils.particle_position_plotter import field_2d_plotter

field_2d_plotter(
    fields=rho1,
    box_size=500.0,
    grid_size=256,
    redshift=z_fin,
    projection_plane="XY",
    slice_range=(0,64),         # (min,max)
    output_type="rho",          # "delta" or "rho" just for lable
    project_mode="mean",        # "mean" or "sum"
    titles=['Stage1'],            
    ncols=1,
    vrange=(0.0,0.8),           # (vmin, vmax)
    save_path=os.path.join(export_dir, f"stage1_proj"),
    return_fig=True,
    show_3d_view=False,
    cmap="hot",
    interpolation="bicubic",
    )

field_2d_plotter(
    fields=rho2,
    box_size=500.0,
    grid_size=256,
    redshift=z_fin,
    projection_plane="XY",
    slice_range=(0,64),         # (min,max)
    output_type="rho",          # "delta" or "rho" just for lable
    project_mode="mean",        # "mean" or "sum"
    titles=['Stage1'],            
    ncols=1,
    vrange=(0.0,0.8),           # (vmin, vmax)
    save_path=os.path.join(export_dir, f"stage2_proj"),
    return_fig=True,
    show_3d_view=False,
    cmap="hot",
    interpolation="bicubic",
    )