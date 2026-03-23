import os,glob
import torch
from tqdm import tqdm
import numpy as np
import h5py,yaml,importlib
from utils.Nbody_data_loader import NbodyLoader
from typing import Optional, Literal, Dict

def load_model(model_cfg, checkpoint_path, device):
    module = importlib.import_module(model_cfg["module"])
    Model = getattr(module, model_cfg["class"])
    model = Model(**model_cfg.get("params", {}))
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

import os

def resolve_paths(cfg, cfg_path=None):

    base_root = None

    project = cfg.get("project", {})
    if isinstance(project, dict) and "base_dir" in project:
        name = project.get("name", "unnamed_project")
        base_root = os.path.join(project["base_dir"], name)
    elif cfg_path is not None:
        base_root = os.path.dirname(os.path.abspath(cfg_path))
    else:
        base_root = os.getcwd()

    def _resolve(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, str) and ("dir" in k or "path" in k or k in ["checkpoints"]):
                    if not os.path.isabs(v):
                        obj[k] = os.path.abspath(os.path.join(base_root, v))
                else:
                    _resolve(v)
        elif isinstance(obj, list):
            for item in obj:
                _resolve(item)

    _resolve(cfg)
    return cfg

def _wrap(x: torch.Tensor, L: float) -> torch.Tensor:
    return torch.remainder(x, L)

def _natural_sort(paths):
    import re

    def _alphanum_key(s):
        return [int(t) if t.isdigit() else t for t in re.split("([0-9]+)", s)]
    return sorted(paths, key=_alphanum_key)

def _w1d_tent(Nw: int) -> np.ndarray:
    c = (Nw - 1) * 0.5
    r = np.abs(np.arange(Nw, dtype=np.float32) - c)
    w = 1.0 - r / max(c, 1e-6)
    return np.clip(w, 0.0, 1.0).astype(np.float32)

def _w3d_from_1d(w1d: np.ndarray) -> np.ndarray:
    return (w1d[:, None, None] * w1d[None, :, None] * w1d[None, None, :]).astype(np.float32)

def _periodic_add(
    accum: np.ndarray,
    wacc: np.ndarray,
    tile: np.ndarray,
    wtile: np.ndarray,
    z0: int,
    y0: int,
    x0: int,
):
    N = accum.shape[0]
    Nw = tile.shape[0]
    zz = (np.arange(Nw) + int(z0)) % N
    yy = (np.arange(Nw) + int(y0)) % N
    xx = (np.arange(Nw) + int(x0)) % N
    sl = np.ix_(zz, yy, xx)
    accum[sl] += tile
    wacc[sl] += wtile

def stitch_from_dir_xyz(
    win_dir: str,
    grid_res: int,
    field: Literal["delta", "rho"] = "delta",
    prefix_filter: Optional[str] = None,
    weight_mode: Literal["count", "tent"] = "count",
) -> np.ndarray:
    N = int(grid_res)
    accum = np.zeros((N, N, N), dtype=np.float32)
    waccum = np.zeros((N, N, N), dtype=np.float32)

    paths = sorted(glob.glob(os.path.join(win_dir, "*.pt")))
    if prefix_filter:
        paths = [p for p in paths if os.path.basename(p).startswith(prefix_filter)]
    if not paths:
        raise FileNotFoundError(f"No .pt windows under: {win_dir}")

    first = torch.load(paths[0], map_location="cpu", weights_only=False)
    tile0 = np.asarray(first[field], dtype=np.float32)
    assert tile0.ndim == 3 and tile0.shape[0] == tile0.shape[1] == tile0.shape[2], f"bad tile shape {tile0.shape}"
    Nw0 = int(first.get("Nw", tile0.shape[0]))
    assert Nw0 == tile0.shape[0], f"Nw mismatch: {Nw0} vs {tile0.shape[0]}"

    if weight_mode == "count":
        W3 = np.ones((Nw0, Nw0, Nw0), dtype=np.float32)
    elif weight_mode == "tent":
        W3 = _w3d_from_1d(_w1d_tent(Nw0))
    else:
        raise ValueError(f"Unknown weight_mode: {weight_mode}")

    for p in paths:
        it = torch.load(p, map_location="cpu", weights_only=False)

        tile = np.asarray(it[field], dtype=np.float32)
        assert tile.shape == (Nw0, Nw0, Nw0), f"tile shape mismatch in {p}: {tile.shape} vs {Nw0}"

        xyz = it.get("xyz", None)
        step = int(it.get("step", 0))
        assert xyz is not None and len(xyz) == 3, f"missing xyz in {p}"
        assert step > 0, f"missing/invalid step in {p}: {step}"

        xi, yi, zi = map(int, xyz)
        x0 = xi * step
        y0 = yi * step
        z0 = zi * step

        _periodic_add(accum, waccum, W3 * tile, W3, z0, y0, x0)

    recon = accum / np.maximum(waccum, 1e-12)
    return recon


class NbodyEmulator:
    def __init__(self, model_dir: str, model_ckp_stages: Optional[tuple[str, str]] = None, device: str = "cuda", config: Optional[str] = None):
        
        self.model_dir = os.path.abspath(model_dir)
        self.device = torch.device(device)

        cfg_path = os.path.join(self.model_dir, "logs", "config_used.yaml")
        cfg = load_yaml(cfg_path)
        cfg = resolve_paths(cfg)
        self.cfg = cfg

        if config is not None:
            ckpt_cfg = load_yaml(config)
            ckpt_cfg = resolve_paths(ckpt_cfg)
        else:
            ckpt_cfg = cfg

        ckpt1_dir = ckpt_cfg["stage1_train"]["checkpoint_dir"]
        ckpt2_dir = ckpt_cfg["stage2_train"]["checkpoint_dir"]

        if model_ckp_stages is None:
            model_ckp1 = os.path.join(ckpt1_dir, "best_model.pt")
            model_ckp2 = os.path.join(ckpt2_dir, "best_model.pt")
        else:
            if model_ckp_stages[0] is None:
                model_ckp1 = os.path.join(ckpt1_dir, "best_model.pt")
            else:
                model_ckp1 = os.path.join(
                    ckpt1_dir, f"model_epoch_{model_ckp_stages[0]}.pt"
                )

            if model_ckp_stages[1] is None:
                model_ckp2 = os.path.join(ckpt2_dir, "best_model.pt")
            else:
                model_ckp2 = os.path.join(
                    ckpt2_dir, f"model_epoch_{model_ckp_stages[1]}.pt"
                )

        self.model_ckp1 = model_ckp1
        self.model_ckp2 = model_ckp2
    
        if model_ckp_stages is None:
            model_ckp1 = os.path.join(self.model_dir, ckpt1_dir,'best_model.pt')
            model_ckp2 = os.path.join(self.model_dir, ckpt2_dir,'best_model.pt')
        else:
            if model_ckp_stages[0] is None:
                model_ckp1 = os.path.join(self.model_dir, ckpt1_dir,'best_model.pt')
            else:
                model_ckp1 = os.path.join(self.model_dir, ckpt1_dir,f'model_epoch_{model_ckp_stages[0]}.pt')
            if model_ckp_stages[1] is None:
                model_ckp2 = os.path.join(self.model_dir, ckpt2_dir,'best_model.pt')
            else:
                model_ckp2 = os.path.join(self.model_dir, ckpt2_dir,f'model_epoch_{model_ckp_stages[1]}.pt')
            


        self.model_ckp1 = model_ckp1 
        self.model_ckp2 = model_ckp2

    def run_stage1(self, z_fin, path_ini: str,  output_dir: Optional[str] = None,chunk_size: int = 200000):
        cfg = self.cfg
        device = self.device

        model1 = load_model(cfg["stage1_model"], self.model_ckp1, device)
        if output_dir is None:
            output_dir = os.path.join(self.model_dir, "emul_out")
        os.makedirs(output_dir, exist_ok=True)

        nb_ini = NbodyLoader(path_ini).load_nbody()
        L     = float(nb_ini["box_size"])
        z_ini = float(nb_ini["redshift"])


        ###### stage1 ########

        # ---- keep big arrays on CPU ----
        pos_ini_cpu = torch.as_tensor(nb_ini["pos"], dtype=torch.float32, device="cpu")
        vel_ini_cpu = torch.as_tensor(nb_ini["vel"], dtype=torch.float32, device="cpu")

        pos_ini_cpu = _wrap(pos_ini_cpu, L)
        N = pos_ini_cpu.shape[0]

        # allocate output on CPU
        pos_pred_cpu = torch.empty_like(pos_ini_cpu)

        # move condition scalars once
        z0 = torch.tensor(z_ini, device=device)
        z1 = torch.tensor(z_fin, device=device)

        model1.eval()

        for s in range(0, N, chunk_size):
            e = min(N, s + chunk_size)

            # ---- move chunk to GPU ----
            pos_ini = pos_ini_cpu[s:e].to(device, non_blocking=True)
            vel_ini = vel_ini_cpu[s:e].to(device, non_blocking=True)

            # normalize exactly as training
            pos_in = (pos_ini / L).clamp(0.0, 1.0 - 1e-6)

            # forward (returns (M,3))
            pred_dpos_norm = model1(pos_in, vel_ini, z0, z1)
            pred_dpos = pred_dpos_norm * L

            pos_pred = _wrap(pos_ini + pred_dpos, L)

            # write back to CPU buffer
            pos_pred_cpu[s:e].copy_(pos_pred.detach().to("cpu"))

        pred_path = os.path.join(output_dir, "stage1_rec.h5")
        with h5py.File(pred_path, "w") as f:
            f["pos_pred"] = pos_pred_cpu
            f.attrs["box_size"] = L

        print(f"✅ Saved stage1 pos_pred to {pred_path}")

        return pred_path

    def run_stage2(self, z_fin, z_ini: Optional[int] = None ,stage1_output: Optional[str] = None, output_dir: Optional[str] = None):
        cfg = self.cfg
        device = self.device

        from utils.prepare_window import build_windows_from_snapshot
        fixed_z0 = cfg["stage2_model"]["params"]["fixed_z0"]

        model2 = load_model(cfg["stage2_model"], self.model_ckp2, "cuda")

        if output_dir is None:
            output_dir = os.path.join(self.model_dir, "emul_out")
        os.makedirs(output_dir, exist_ok=True)

        if stage1_output is None:
            stage1_output = os.path.join(output_dir, "stage1_pred.h5")

        stage2_prep = cfg["stage2_prepare"]["params"]
        grid_res = int(stage2_prep["grid_res"])
        Nw = int(stage2_prep["Nw"])
        step = int(stage2_prep["step"])
        data_type = cfg["stage2_dataset"]["params"]["data_type"]

        win_in_dir = os.path.join(output_dir, "wins_in")
        os.makedirs(win_in_dir, exist_ok=True)

        _ = build_windows_from_snapshot(
            snapshot_path=stage1_output,
            grid_size=grid_res,
            Nw=Nw,
            step=step,
            device=device,
            out_dir=win_in_dir,
            prefix="dens",
            recompute=True,
            save_full=False,
            save_windows=True,
        )

        model2.eval()
        pred_paths = _natural_sort(glob.glob(os.path.join(win_in_dir, "dens_w*.pt")))

        win_out_dir = os.path.join(output_dir, "wins_out")
        os.makedirs(win_out_dir, exist_ok=True)

        for win_idx, p in enumerate(pred_paths):
            dp = torch.load(p, map_location="cpu", weights_only=False)
            x = dp[data_type].unsqueeze(0).unsqueeze(0).to("cuda")  # (1,1,Nw,Nw,Nw)

            with torch.no_grad():
                if fixed_z0:
                    yhat = model2(x, z1=z_fin, z0=z_ini).float().cpu()[0, 0]  # (Nw,Nw,Nw)
                else:
                    yhat = model2(x, z1=z_fin).float().cpu()[0, 0]

            item = {
                data_type: yhat.numpy().astype(np.float32),
                "step": int(dp["step"]),
                "Nw": int(dp["Nw"]),
                "grid_res": int(dp["grid_res"]),
                "xyz": dp["xyz"],
            }
            win_path = os.path.join(win_out_dir, f"recon_w{win_idx:05d}.pt")
            torch.save(item, win_path)

        rec = stitch_from_dir_xyz(
            win_dir=win_out_dir,
            grid_res=grid_res,
            field=data_type,
            prefix_filter="recon_w",
            weight_mode="tent")

        rec_path = os.path.join(output_dir, f"stage2_rec.h5")
        with h5py.File(rec_path, "w") as f:
            f[data_type] = rec
            f.attrs["grid_res"] = grid_res

        print(f"✅ Saved stage2 reconstructed {data_type} to {rec_path}")
        return rec_path
