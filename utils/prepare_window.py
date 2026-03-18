import os, h5py, glob
import numpy as np
import torch
from typing import Optional, Dict, List
from utils.Nbody_data_loader import NbodyLoader

# ------------------- CIC -------------------
@torch.no_grad()
def cic_density(pos: torch.Tensor,
                box_size: float,
                grid_size: int,
                part_mass: Optional[torch.Tensor] = None,
                device: Optional[torch.device] = None,
                dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if device is None: device = pos.device
    if dtype  is None: dtype  = torch.float32

    # pos: (N,3) tensor (or array convertible)
    x = torch.as_tensor(pos, device=device, dtype=dtype)
    x = torch.remainder(x, box_size)

    if part_mass is None:
        w = torch.ones(x.size(0), device=device, dtype=dtype)
    else:
        w = torch.as_tensor(part_mass, device=device, dtype=dtype)

    u  = x / box_size * float(grid_size)  # grid coords in [0,grid_size)
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

    rho = torch.zeros(grid_size*grid_size*grid_size, device=device, dtype=dtype)
    for lin, wt in [(lin_idx(idx000), wt000), (lin_idx(idx100), wt100),
                    (lin_idx(idx010), wt010), (lin_idx(idx110), wt110),
                    (lin_idx(idx001), wt001), (lin_idx(idx101), wt101),
                    (lin_idx(idx011), wt011), (lin_idx(idx111), wt111)]:
        rho.scatter_add_(0, lin, w * wt)

    rho = rho.view(grid_size, grid_size, grid_size)
    voxel_vol = (box_size / float(grid_size)) ** 3
    rho = rho / voxel_vol
    return rho  # (N,N,N)

def _periodic_extract_window(vol: np.ndarray, z0: int, y0: int, x0: int, Nw: int) -> np.ndarray:
    N = vol.shape[0]
    z_idx = (np.arange(Nw) + z0) % N
    y_idx = (np.arange(Nw) + y0) % N
    x_idx = (np.arange(Nw) + x0) % N
    return vol[z_idx][:, y_idx][:, :, x_idx]  # (Nw,Nw,Nw)

@torch.no_grad()
def build_windows_from_snapshot(
    snapshot_path: str,
    Nw: int,
    step: int,
    grid_size: int = 256,
    return_delta: bool = True,
    delta_mode: str = "asinh",
    alpha: float = 1.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    recompute: bool = True,
    out_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    save_full: bool = True,
    save_windows: bool = True,):
    # -------------------------
    # 1) Load particle positions and box size
    # -------------------------
    spath_lower = snapshot_path.lower()
    if spath_lower.endswith(".h5"):
        with h5py.File(snapshot_path, "r") as f:
            if "pos_pred" in f:
                pos = np.asarray(f["pos_pred"][:], dtype=np.float32)
            elif "pos" in f:
                pos = np.asarray(f["pos"][:], dtype=np.float32)
            else:
                raise KeyError("no 'pos_pred' or 'pos' in .h5 file")
            # mass = float(f.attrs["mass"])
            L = float(f.attrs["box_size"])
    elif spath_lower.endswith(".hdf5"):
        nbody_final = NbodyLoader(snapshot_path).load_nbody()
        pos = nbody_final["pos"].astype(np.float32)
        # mass = nbody_final["mass"].astype(np.float32)
        L = float(nbody_final["box_size"])
    else:
        raise ValueError("Unsupported file type (expected .h5 or .hdf5).")

    dx = L / float(grid_size)
    Lw = Nw * dx
    stride = step * dx
    N = int(grid_size)

    saved_paths: List[str] = []

    # -------------------------
    # 2) Resolve prefix and out_dir
    # -------------------------
    if out_dir is not None and prefix is None:
        base = os.path.basename(snapshot_path).rsplit(".", 1)[0]
        prefix = base

    # Decide once whether we will save full/windows
    do_save_full = save_full and out_dir is not None
    do_save_windows = save_windows and out_dir is not None

    # Create directory if we are going to save anything
    if out_dir is not None and (do_save_full or do_save_windows):
        os.makedirs(out_dir, exist_ok=True)

    # -------------------------
    # 3) Fast path: skip computation if files already exist and recompute=False
    # -------------------------
    if out_dir is not None and not recompute:
        existing: List[str] = []

        if do_save_full:
            full_path = os.path.join(out_dir, f"{prefix}_full.pt")
            if os.path.exists(full_path):
                existing.append(full_path)

        if do_save_windows:
            pattern = os.path.join(out_dir, f"{prefix}_w*.pt")
            win_paths = sorted(glob.glob(pattern))
            existing.extend(win_paths)

        if len(existing) > 0:
            print(out_dir, "exists, skip compute")
            return {
                "L": float(L),
                "grid_size": int(grid_size),
                "dx": float(dx),
                "Lw": float(Lw),
                "stride": float(stride),
                "Nw": int(Nw),
                "step": int(step),
                "windows": [],         # skip loading windows (avoid heavy I/O)
                "saved_paths": existing,
            }

    # -------------------------
    # 4) Build full rho and delta
    # -------------------------
    rho_t = cic_density(
        pos=pos,
        box_size=L,
        grid_size=grid_size,
        part_mass=None,
        device=torch.device(device),
        dtype=dtype,
    )
    rho = rho_t.detach().cpu().to(torch.float32)  # (N, N, N)

    eps = 1e-8
    rbar = float(rho.mean().item())

    if return_delta:
        delta = (rho - rbar) / (rbar + eps)
        if delta_mode == "none":
            pass
        elif delta_mode == "asinh":
            delta = torch.asinh(delta * float(alpha))
        elif delta_mode == "log1p_delta":
            delta = torch.log1p(torch.clamp(delta, min=-0.999999))
        elif delta_mode == "log_rho":
            delta = torch.log(torch.clamp(rho, min=eps) / (rbar + eps))
        else:
            raise ValueError(f"Unknown delta_mode: {delta_mode}")
    else:
        delta = torch.zeros_like(rho)

    # -------------------------
    # 5) Optionally save full box
    # -------------------------
    if do_save_full:
        full_path = os.path.join(out_dir, f"{prefix}_full.pt")
        torch.save(
            {
                "rho": rho,                               # (N, N, N)
                "delta": delta,                           # (N, N, N)
                "center": np.asarray([L / 2, L / 2, L / 2], dtype=np.float32),
                "Lw": float(L),                           # for full box, window size = box size
                "grid_res": int(grid_size),
            },
            full_path,
        )
        saved_paths.append(full_path)

    # -------------------------
    # 6) Build windows
    # -------------------------
    windows_meta: List[Dict[str, object]] = []
    rho_np = rho.numpy()
    delta_np = delta.numpy()

    win_idx = 0
    for zi, z0 in enumerate(range(0, N, step)):
        for yi, y0 in enumerate(range(0, N, step)):
            for xi, x0 in enumerate(range(0, N, step)):
                rho_w = _periodic_extract_window(rho_np, z0, y0, x0, Nw).astype(np.float32)
                delta_w = _periodic_extract_window(delta_np, z0, y0, x0, Nw).astype(np.float32)

                item = {
                    "rho": torch.from_numpy(rho_w),          # (Nw, Nw, Nw) float32
                    "delta": torch.from_numpy(delta_w),      # (Nw, Nw, Nw) float32
                    "step": int(step),
                    "Nw": int(Nw),
                    "Lw": float(L),
                    "stride": float(stride),
                    "dx": float(dx),
                    "grid_res": int(grid_size),
                    "xyz": [int(xi), int(yi), int(zi)],
                }
                windows_meta.append(item)

                if do_save_windows:
                    win_path = os.path.join(out_dir, f"{prefix}_w{win_idx}.pt")
                    torch.save(item, win_path)
                    saved_paths.append(win_path)

                win_idx += 1

    # -------------------------
    # 7) Return metadata and (optionally) in-memory windows
    # -------------------------
    return {
        "L": float(L),
        "grid_size": int(grid_size),
        "dx": float(dx),
        "Lw": float(Lw),
        "stride": float(stride),
        "Nw": int(Nw),
        "step": int(step),
        "windows": windows_meta,
        "saved_paths": saved_paths,
    }



def build_windows_from_density(
    rho: str,
    box_size: float,
    Nw: int,
    step: int,
    return_delta: bool = True,
    delta_mode: str = "asinh",
    alpha: float = 1.0,
    # saving options
    out_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    save_full: bool = True,
    save_windows: bool = True,
    recompute: bool = True,) -> Dict[str, object]:

    # -----------------------
    # Basic setup
    # -----------------------

    print(rho)

    with h5py.File(rho, "r") as f:
        box_size  = f.attrs["box_size"]
        grid_size = f.attrs["grid_size"]
        rho       = f["rho"][...]


    if rho.ndim != 3:
        raise ValueError("rho must be a 3D array.")

    N = rho.shape[0]
    grid_size = int(N)
    dx = float(box_size / N)
    Lw = float(Nw * dx)
    stride = float(step * dx)

    if prefix is None:
        prefix = "dens"

    do_save_full = save_full and out_dir is not None
    do_save_windows = save_windows and out_dir is not None

    if out_dir and (do_save_full or do_save_windows):
        os.makedirs(out_dir, exist_ok=True)

    # -----------------------
    # Skip computation if files exist
    # -----------------------
    if out_dir and not recompute:
        existing = []
        if do_save_full:
            full_path = os.path.join(out_dir, f"{prefix}_full.pt")
            if os.path.exists(full_path):
                existing.append(full_path)

        if do_save_windows:
            wins = sorted(glob.glob(os.path.join(out_dir, f"{prefix}_w*.pt")))
            existing.extend(wins)

        if len(existing) > 0:
            return {
                "L": box_size,
                "grid_size": N,
                "dx": dx,
                "Lw": Lw,
                "stride": stride,
                "Nw": Nw,
                "step": step,
                "saved_paths": existing,
            }

    # -----------------------
    # Compute delta
    # -----------------------
    rho_t = torch.from_numpy(rho).float()
    eps = 1e-8
    rbar = float(rho_t.mean().item())

    if return_delta:
        delta = (rho_t - rbar) / (rbar + eps)

        if delta_mode == "none":
            pass
        elif delta_mode == "asinh":
            delta = torch.asinh(delta * float(alpha))
        elif delta_mode == "log1p_delta":
            delta = torch.log1p(torch.clamp(delta, min=-0.999999))
        elif delta_mode == "log_rho":
            delta = torch.log(torch.clamp(rho_t, min=eps) / (rbar + eps))
        else:
            raise ValueError(f"Unknown delta_mode: {delta_mode}")
    else:
        delta = torch.zeros_like(rho_t)

    # -----------------------
    # Save full box
    # -----------------------
    saved_paths = []

    if do_save_full:
        full_path = os.path.join(out_dir, f"{prefix}_full.pt")
        torch.save(
            {
                "rho": rho_t,
                "delta": delta,
                "Lw": float(box_size),
                "center": np.array([box_size / 2] * 3, dtype=np.float32),
                "grid_res": grid_size,
            },
            full_path,
        )
        saved_paths.append(full_path)

    # -----------------------
    # Build windows
    # -----------------------
    def extract_periodic(arr, z0, y0, x0, size):
        """Extract periodic 3D cubic window from numpy array."""
        idx = np.arange(size)
        return arr[
            (z0 + idx) % N,
            :, :
        ][:, (y0 + idx) % N][:, :, (x0 + idx) % N]


    rho_np = rho
    delta_np = delta.numpy()

    win_idx = 0

    for z0 in range(0, N, step):
        for y0 in range(0, N, step):
            for x0 in range(0, N, step):
                rho_w = extract_periodic(rho_np, z0, y0, x0, Nw).astype(np.float32)
                delta_w = extract_periodic(delta_np, z0, y0, x0, Nw).astype(np.float32)

                item = {
                    "rho": torch.from_numpy(rho_w),
                    "delta": torch.from_numpy(delta_w),
                    "xyz": [x0 // step, y0 // step, z0 // step],
                    "Nw": Nw,
                    "step": step,
                    "dx": dx,
                    "Lw": Lw,
                    "stride": stride,
                    "grid_res": N,
                }

                if do_save_windows:
                    win_path = os.path.join(out_dir, f"{prefix}_w{win_idx}.pt")
                    torch.save(item, win_path)
                    saved_paths.append(win_path)

                win_idx += 1

    # -----------------------
    # Return metadata
    # -----------------------
    return {
        "L": box_size,
        "grid_size": N,
        "dx": dx,
        "Lw": Lw,
        "stride": stride,
        "Nw": Nw,
        "step": step,
        "saved_paths": saved_paths,
    }


