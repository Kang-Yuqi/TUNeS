import os, glob, random, re
import h5py
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any, Optional

def _natural_sort(paths: List[str]) -> List[str]:
    def atoi(t): return int(t) if t.isdigit() else t
    def keys(s): return [atoi(c) for c in re.split(r'(\d+)', s)]
    return sorted(paths, key=keys)

def _rot24(x: torch.Tensor) -> torch.Tensor:
    perms = [(1,2,3), (1,3,2), (2,1,3), (2,3,1), (3,1,2), (3,2,1)]
    px, py, pz = random.choice(perms)
    x = x.permute(0, px, py, pz)
    if random.random() < 0.5: x = torch.flip(x, dims=[1])
    if random.random() < 0.5: x = torch.flip(x, dims=[2])
    if random.random() < 0.5: x = torch.flip(x, dims=[3])
    return x

def _default_transform(x: torch.Tensor) -> torch.Tensor:
    if random.random() < 0.5:
        x = _rot24(x)
    else:
        if random.random() < 0.5: x = torch.flip(x, dims=[1])
        if random.random() < 0.5: x = torch.flip(x, dims=[2])
        if random.random() < 0.5: x = torch.flip(x, dims=[3])
        k = random.randint(0, 3)
        if k: x = torch.rot90(x, k=k, dims=(1, 2))
    return x

def _read_z_from_pred_h5(path_ini: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Read (z_ini, z_fin) from stage1 pred file attributes.
    Your stage1 output uses attrs:
      redshift_ini, redshift_final
    """
    try:
        with h5py.File(path_ini, "r") as f:
            z0 = float(f.attrs.get("redshift_ini",  None))
            z1 = float(f.attrs.get("redshift_final", None))
        return z0, z1
    except Exception:
        return None, None

class WindowDensityZDepDataset(Dataset):
    def __init__(self,
                 out_dir: str,
                 file_pairs: List[Tuple[str, str]],
                 transform: str = "default",
                 data_type: str = "rho",
                 dtype: str = "float32",
                 cache_z: bool = True):

        self.out_dir = out_dir
        self.dtype = getattr(torch, dtype)
        self.transform = transform
        self.data_type = data_type
        self.samples: List[Tuple[str, str, Optional[float], Optional[float]]] = []

        # optional cache to avoid repeatedly opening h5
        z_cache: Dict[str, Tuple[Optional[float], Optional[float]]] = {}

        for path_ini, path_final in file_pairs:
            # ---- tags: must match your stage2_prepare naming ----
            tag_pred = os.path.basename(path_ini).replace(".h5", "")

            sim_name_final = os.path.basename(os.path.dirname(path_final))
            snapshot_tag   = os.path.basename(path_final).replace(".h5", "")
            tag_final      = f"{sim_name_final}_{snapshot_tag}_target"

            pred_dir = os.path.join(out_dir, tag_pred)
            targ_dir = os.path.join(out_dir, tag_final)

            pred_paths = _natural_sort(glob.glob(os.path.join(pred_dir, "dens_w*.pt")))
            if not pred_paths:
                raise RuntimeError(f"No windows under: {pred_dir}")

            # ---- read z from pred h5 once per pair ----
            if cache_z and path_ini in z_cache:
                z0, z1 = z_cache[path_ini]
            else:
                z0, z1 = _read_z_from_pred_h5(path_ini)
                if cache_z:
                    z_cache[path_ini] = (z0, z1)

            for p in pred_paths:
                fname = os.path.basename(p)
                q = os.path.join(targ_dir, fname)
                if not os.path.exists(q):
                    raise FileNotFoundError(f"Target missing: {q}")
                self.samples.append((p, q, z0, z1))

        if not self.samples:
            raise RuntimeError("No window pairs found for given file_pairs.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        p_pred, p_tgt, z0, z1 = self.samples[idx]

        d_pred = torch.load(p_pred, map_location="cpu", weights_only=False)
        d_tgt  = torch.load(p_tgt,  map_location="cpu", weights_only=False)

        x = d_pred[self.data_type].to(self.dtype).unsqueeze(0)  # (1,N,N,N)
        y = d_tgt[self.data_type].to(self.dtype).unsqueeze(0)

        if self.transform is not None and str(self.transform).lower() == "default":
            xy = torch.cat([x, y], dim=0)  # (2,N,N,N)
            xy = _default_transform(xy)
            x, y = xy[:1], xy[1:]

        z0_t = torch.tensor(z0, dtype=torch.float32) if z0 is not None else None
        z1_t = torch.tensor(z1, dtype=torch.float32) if z1 is not None else None

        return {
            "x": x,
            "y": y,
            "z0": z0_t,
            "z1": z1_t,
            "meta": {
                "pred_win": p_pred,
                "targ_win": p_tgt,
                "pair_pred": d_pred.get("pair_tag", None),
                "Lw": d_pred.get("Lw", None),
                "grid_res": d_pred.get("grid_res", None),
            }
        }

def collate_window_zdep_batch(batch):
    x = torch.stack([b["x"] for b in batch], dim=0)
    y = torch.stack([b["y"] for b in batch], dim=0)

    if any(b["z0"] is None or b["z1"] is None for b in batch):
        bad = [i for i,b in enumerate(batch) if (b["z0"] is None or b["z1"] is None)]
        raise RuntimeError(f"Missing z0/z1 in batch indices {bad}. Check stage1 pred attrs.")

    z0 = torch.stack([b["z0"] for b in batch], dim=0).view(-1)
    z1 = torch.stack([b["z1"] for b in batch], dim=0).view(-1)

    meta_keys = batch[0]["meta"].keys()
    meta = {k: [b["meta"][k] for b in batch] for k in meta_keys}
    return {"x": x, "y": y, "z0": z0, "z1": z1, "meta": meta}