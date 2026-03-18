import os
import random
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.Nbody_data_loader import NbodyLoader

class _IniCache:
    """Worker-local LRU cache for ini snapshots (pos, vel)."""
    def __init__(self, max_items: int = 2):
        self.max_items = int(max_items)
        self.cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.order: List[str] = []

    def get(self, key: str):
        if key in self.cache:
            if key in self.order:
                self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: Tuple[torch.Tensor, torch.Tensor]):
        if key in self.cache:
            self.cache[key] = value
            if key in self.order:
                self.order.remove(key)
            self.order.append(key)
            return
        if len(self.order) >= self.max_items:
            old = self.order.pop(0)
            self.cache.pop(old, None)
        self.cache[key] = value
        self.order.append(key)


class DispZdepDataset(Dataset):

    def __init__(
        self,
        file_out,
        data_dir: str,
        target: str = "dpos",
        dtype_str: str = "float32",
        sample_size: int = 65536,
        sample_mode: str = "random",  # "random" | "all"
        ini_cache_items: int = 2,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.target = target
        self.sample_size = int(sample_size)
        self.sample_mode = str(sample_mode)
        self.cache = _IniCache(max_items=ini_cache_items)


        dtype_str = dtype_str.lower()
        if dtype_str == "float16":
            self.dtype = torch.float16
        elif dtype_str == "float32":
            self.dtype = torch.float32
        else:
            raise ValueError("dtype_str must be float16 or float32")

        self.files = file_out

    def __len__(self) -> int:
        return len(self.files)

    def _sample_indices(self, N: int) -> np.ndarray:
        if self.sample_mode == "all":
            return np.arange(N, dtype=np.int64)
        M = min(self.sample_size, N)
        return np.random.randint(0, N, size=(M,), dtype=np.int64)

    def _load_ini_posvel(self, ini_path: str) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Returns pos_ini, vel_ini (CPU tensors float32), and box_size.
        Uses worker-local cache.
        """
        cached = self.cache.get(ini_path)
        if cached is not None:
            pos, vel, box_size = cached
            return pos, vel, box_size


        nbody_ini = NbodyLoader(ini_path).load_nbody()

        pos = torch.as_tensor(nbody_ini["pos"], dtype=torch.float32, device="cpu")
        vel = torch.as_tensor(nbody_ini["vel"], dtype=torch.float32, device="cpu")
        box_size = float(nbody_ini["box_size"])

        self.cache.put(ini_path, (pos, vel, box_size))
        return pos, vel, box_size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pack = torch.load(self.files[idx], map_location="cpu")

        dpos = pack[self.target]
        if not isinstance(dpos, torch.Tensor):
            dpos = torch.as_tensor(dpos)
        dpos = dpos.to(torch.float32)  # load in float32 for sampling

        z_ini = float(pack["z_ini"])
        z_fin = float(pack["z_fin"])
        ini_path = str(pack.get("ini_path", pack.get("path_ini", "")))
        if ini_path == "":
            raise RuntimeError(f"Pair file missing ini_path: {self.files[idx]}")

        pos_ini, vel_ini, box_size = self._load_ini_posvel(ini_path)

        N = dpos.shape[0]
        if pos_ini.shape[0] != N or vel_ini.shape[0] != N:
            raise RuntimeError(
                f"N mismatch: dpos={N}, pos={pos_ini.shape[0]}, vel={vel_ini.shape[0]} | {ini_path}"
            )

        sel = self._sample_indices(N)
        sel_t = torch.as_tensor(sel, dtype=torch.long)

        pos = pos_ini.index_select(0, sel_t).to(self.dtype)
        vel = vel_ini.index_select(0, sel_t).to(self.dtype)
        y   = dpos.index_select(0, sel_t).to(self.dtype)

        out = {
            "pos_ini": pos,  # (M,3)
            "vel_ini": vel,  # (M,3)
            "z_ini": torch.tensor(z_ini, dtype=torch.float32),
            "z_fin": torch.tensor(z_fin, dtype=torch.float32),
            "dpos": y,       # (M,3)
            "box_size": torch.tensor(float(pack.get("box_size", box_size)), dtype=torch.float32),
        }
        return out
