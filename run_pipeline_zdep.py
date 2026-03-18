import math, os, sys, importlib, argparse, yaml, torch
from torch.utils.data import Sampler

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def load_datasets(cfg: dict, stage_id: int, return_pairs: bool = False):
    """
    New logic:
      - pairing is handled by stage{stage_id}_prepare.params_loading
        via expand_global_pairs_with_prev(...)
      - dataset consumes:
          - out_dir (windows root)
          - file_pairs (train/val) from cfg["global_files"]
    """

    dataset_cfg = cfg[f"stage{stage_id}_dataset"]
    dataset_module = importlib.import_module(dataset_cfg["module"])
    DatasetClass = getattr(dataset_module, dataset_cfg["class"])
    base_params = dataset_cfg.get("params", {}).copy()

    # 1) Ensure global_files has file_pairs_train/val
    #    Prefer: if stage_prepare exists, use its params_loading to build pairs.
    prepare_cfg = cfg.get(f"stage{stage_id}_prepare", None)
    if prepare_cfg and "params_loading" in prepare_cfg:
        load_params = prepare_cfg["params_loading"].copy()
        cfg = expand_global_pairs_with_prev(cfg, **load_params)
    else:
        # fallback: if pairs already in cfg.global_files, do nothing
        gf = cfg.get("global_files", {})
        if not gf.get("file_pairs_train", []):
            raise RuntimeError(
                f"No stage{stage_id}_prepare.params_loading and global_files.file_pairs_train is empty."
            )

    gf = cfg.get("global_files", {})
    train_pairs = gf.get("file_pairs_train", [])
    val_pairs   = gf.get("file_pairs_val", [])

    train_out   = gf["file_disp_train"]
    val_out     = gf["file_disp_val"]

    if not train_pairs:
        raise RuntimeError("global_files.file_pairs_train is empty after pairing step.")

    # 2) Build train dataset
    train_params = base_params.copy()
    if stage_id == 1:
        train_params["file_out"] = train_out
    else:
        train_params["file_pairs"] = train_pairs
    train_dataset = DatasetClass(**train_params)

    # 3) Build val dataset (if any)
    val_dataset = None
    if val_pairs and len(val_pairs) > 0:
        val_params = base_params.copy()
        if stage_id == 1:
            val_params["file_out"] = val_out
        else:
            val_params["file_pairs"] = val_pairs
        val_dataset = DatasetClass(**val_params)

    if return_pairs:
        return train_dataset, val_dataset, (train_pairs, val_pairs)

    return train_dataset, val_dataset

def make_loader(dataset, sampler, batch_size, num_workers, collate_fn=None):
    from torch.utils.data import DataLoader

    if dataset is None:
        return None
    kwargs = dict(
        batch_size  = batch_size,
        sampler     = sampler,
        shuffle     = False, # sampler controls order; don't shuffle here
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = False,
    )
    if collate_fn is not None:
        kwargs["collate_fn"] = collate_fn
    return DataLoader(dataset, **kwargs)
    

class EpochSubsetSampler(Sampler[int]):
    """
    Randomly sample a subset of indices per epoch (optionally sharded across ranks).
    - dataset_size: total size of the dataset
    - epoch_size  : how many samples to use per epoch (global, across all ranks)
    - shuffle     : shuffle before taking the head
    - seed        : base seed; effective seed = seed + epoch
    - rank/world_size: DDP sharding (indices[rank::world_size])
    """
    def __init__(self, dataset_size: int, epoch_size: int | None,
                 shuffle: bool = True, seed: int = 42,
                 rank: int = 0, world_size: int = 1):
        self.dataset_size = int(dataset_size)
        self.epoch_size   = int(epoch_size) if epoch_size is not None else self.dataset_size
        self.shuffle      = bool(shuffle)
        self.seed         = int(seed)
        self.rank         = int(rank)
        self.world_size   = int(world_size)
        self.epoch        = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        if self.shuffle:
            perm = torch.randperm(self.dataset_size, generator=g).tolist()
        else:
            perm = list(range(self.dataset_size))

        total = min(self.epoch_size, self.dataset_size)  # global size per epoch
        chosen = perm[:total]

        # shard across ranks
        indices = chosen[self.rank::self.world_size]
        return iter(indices)

    def __len__(self) -> int:
        total = min(self.epoch_size, self.dataset_size)
        # approximate per-rank length for progress bars
        return math.ceil(total / self.world_size)
    
# ============================== Prepare data ==============================
import os

def expand_global_pairs(cfg, out_dir: str, return_pairs: bool = False, fixed_ini=None):
    """
    Expand global_files.file_dir_train/val + file_name into:
      global_files.file_pairs_train / file_pairs_val : list[(ini_path, fin_path)]
      global_files.file_disp_train  / file_disp_val  : list[out_pt_path]

    Parameters
    ----------
    fixed_ini : None | int | str
        - None: build all pairs (i<j) as before.
        - int : use snapshot_names[fixed_ini] as ini for all pairs (ini, j) with j != ini.
               (Recommended: j > ini to keep forward direction.)
        - str : use the snapshot whose basename equals fixed_ini as ini.

    """

    def _flatten_1(x):
        if isinstance(x, (list, tuple)) and len(x) == 1:
            return _flatten_1(x[0])
        return x

    def _flatten_list(lst):
        return [_flatten_1(x) for x in lst]

    def _resolve_fixed_index(names_sorted, fixed_ini):
        if fixed_ini is None:
            return None
        if isinstance(fixed_ini, int):
            if fixed_ini < 0 or fixed_ini >= len(names_sorted):
                raise IndexError(f"fixed_ini index {fixed_ini} out of range [0,{len(names_sorted)-1}]")
            return fixed_ini
        if isinstance(fixed_ini, str):
            if fixed_ini not in names_sorted:
                base = os.path.basename(fixed_ini)
                if base in names_sorted:
                    fixed_ini = base
                else:
                    raise ValueError(f"fixed_ini='{fixed_ini}' not found in file_name list: {names_sorted}")
            return names_sorted.index(fixed_ini)
        raise TypeError("fixed_ini must be None, int, or str")

    def _build_pairs_for_dirs(sim_dirs, snapshot_names, tag, fixed_ini=None):
        pairs = []
        out_paths = []

        names = sorted(snapshot_names)
        ini_idx = _resolve_fixed_index(names, fixed_ini)

        for sim_dir in sim_dirs:
            sim_dir = sim_dir.rstrip("/")
            sim_id = os.path.basename(sim_dir)

            if ini_idx is None:
                # original: all (i<j)
                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        p1 = os.path.join(sim_dir, names[i])
                        p2 = os.path.join(sim_dir, names[j])
                        pairs.append((p1, p2))

                        out_pt = os.path.join(out_dir, tag, sim_id, f"pair_{i:03d}_{j:03d}.pt")
                        out_paths.append(out_pt)
            else:
                # fixed ini: only (ini_idx, j) with j > ini_idx
                i = ini_idx
                for j in range(i + 1, len(names)):
                    p1 = os.path.join(sim_dir, names[i])
                    p2 = os.path.join(sim_dir, names[j])
                    pairs.append((p1, p2))

                    out_pt = os.path.join(out_dir, tag, sim_id, f"pair_{i:03d}_{j:03d}.pt")
                    out_paths.append(out_pt)

        return pairs, out_paths

    gf = cfg.get("global_files", {})
    if "file_dir_train" not in gf or "file_name" not in gf:
        raise RuntimeError("global_files.file_dir_train / global_files.file_name missing")

    train_dirs = _flatten_list(gf["file_dir_train"])
    val_dirs   = _flatten_list(gf.get("file_dir_val", []))
    snap_names = _flatten_list(gf["file_name"])

    train_pairs, train_out = _build_pairs_for_dirs(train_dirs, snap_names, tag="train", fixed_ini=fixed_ini)
    val_pairs, val_out = ([], [])
    if len(val_dirs) > 0:
        val_pairs, val_out = _build_pairs_for_dirs(val_dirs, snap_names, tag="val", fixed_ini=fixed_ini)

    if return_pairs:
        return (train_pairs, train_out), (val_pairs, val_out)

    cfg.setdefault("global_files", {})
    cfg["global_files"]["file_pairs_train"] = train_pairs
    cfg["global_files"]["file_pairs_val"]   = val_pairs
    cfg["global_files"]["file_disp_train"]  = train_out
    cfg["global_files"]["file_disp_val"]    = val_out
    return cfg

import os, glob, re
from typing import Dict, List, Tuple, Optional, Union

_PAIR_RE = re.compile(r"^(?P<sim>.+?)_pair_(?P<i>\d{3})_(?P<j>\d{3})_pred\.h5$")

def expand_global_pairs_with_prev(
    cfg: dict,
    out_dir: str,
    return_pairs: bool = False,
    fixed_ini: Optional[Union[int, str]] = None,
    stage_id: Optional[int] = None,
    use_prev_stage_output: bool = False,
    stage_source_dir: Optional[str] = None,
    prev_glob: str = "*_pred.h5",
    fin_local_key: str = None,
    map_prev_j_to_fin: bool = True):
    """
    Two modes:

    1) legacy (use_prev_stage_output=False):
       Build pairs inside each sim_dir using file_name (i<j or fixed_ini).

    2) prev-stage mode (use_prev_stage_output=True):
       ini:  prev stage outputs in stage_source_dir, filenames like: <sim>_pair_000_014_pred.h5
       fin:  snapshot files from (stage_local_files or cfg['global_files']) with file_dir_train/val + file_name
       pairing:
         - if map_prev_j_to_fin=True: use j index from ini filename to pick fin snapshot name
           (requires that sorted(file_name) indexing matches j)
         - else: full cartesian per sim_id (not recommended; can explode)

    Writes back to cfg['global_files']:
      file_pairs_train/val, file_disp_train/val
    """

    def _flatten_1(x):
        if isinstance(x, (list, tuple)) and len(x) == 1:
            return _flatten_1(x[0])
        return x

    def _flatten_list(lst):
        return [_flatten_1(x) for x in lst]

    def _load_fin_sources(tag: str):

        if fin_local_key is not None:
            if isinstance(fin_local_key, str):
                src = cfg.get(fin_local_key, {})
            elif isinstance(fin_local_key, dict):
                src = fin_local_key
            else:
                raise TypeError(f"fin_local_key must be str or dict, got {type(fin_local_key)}")
        else:
            src = cfg.get("global_files", {})

        if tag == "train":
            sim_dirs = _flatten_list(src.get("file_dir_train", []))
        else:
            sim_dirs = _flatten_list(src.get("file_dir_val", []))

        snap_names = _flatten_list(src.get("file_name", []))
        return sim_dirs, sorted(snap_names)

    def _build_fin_group(sim_dirs, snap_names) -> Dict[str, List[str]]:
        g: Dict[str, List[str]] = {}
        for sim_dir in sim_dirs:
            sim_dir = sim_dir.rstrip("/")
            sim_id = os.path.basename(sim_dir)
            g[sim_id] = [os.path.join(sim_dir, n) for n in snap_names]
        return g

    def _parse_prev_outputs(prev_dir: str) -> Dict[str, List[Tuple[str, int, int]]]:
        """
        return: group[sim_id] = [(path, i, j), ...] sorted by (i,j)
        """
        paths = sorted(glob.glob(os.path.join(prev_dir, prev_glob)))
        g: Dict[str, List[Tuple[str, int, int]]] = {}
        for p in paths:
            base = os.path.basename(p)
            m = _PAIR_RE.match(base)
            if not m:
                continue
            sim_id = m.group("sim")
            i = int(m.group("i"))
            j = int(m.group("j"))
            g.setdefault(sim_id, []).append((p, i, j))
        for sim_id in g:
            g[sim_id].sort(key=lambda x: (x[1], x[2]))
        return g

    def _ensure_stage_source_dir():
        nonlocal stage_source_dir
        if stage_source_dir is not None:
            return
        if stage_id is None:
            raise ValueError("stage_id is required when stage_source_dir is not provided and use_prev_stage_output=True")
        key = f"stage{stage_id-1}_output"
        if key not in cfg or "out_dir" not in cfg[key]:
            raise RuntimeError(f"Missing {key}.out_dir in cfg; cannot locate previous stage outputs")
        stage_source_dir = cfg[key]["out_dir"]

    # ---------------------------
    # mode 2: prev-stage output as ini
    # ---------------------------
    if use_prev_stage_output:
        _ensure_stage_source_dir()

        # fin side
        train_dirs, snap_names = _load_fin_sources("train")
        val_dirs, _snap_val    = _load_fin_sources("val")

        if not train_dirs or not snap_names:
            raise RuntimeError("fin sources missing: need file_dir_train and file_name (stage-local or global_files)")

        fin_train_group = _build_fin_group(train_dirs, snap_names)
        fin_val_group   = _build_fin_group(val_dirs, snap_names) if val_dirs else {}

        # ini side from prev outputs
        ini_group = _parse_prev_outputs(stage_source_dir)

        train_pairs: List[Tuple[str, str]] = []
        train_out:   List[str] = []
        for sim_id, ini_list in ini_group.items():
            if sim_id not in fin_train_group:
                continue
            fin_list = fin_train_group[sim_id]  # indexed by j

            for (ini_path, i, j) in ini_list:
                if map_prev_j_to_fin:
                    if j < 0 or j >= len(fin_list):
                        raise IndexError(
                            f"prev output j={j:03d} out of range for sim '{sim_id}': "
                            f"len(file_name)={len(fin_list)}. Check file_name ordering/mapping."
                        )
                    fin_path = fin_list[j]
                    train_pairs.append((ini_path, fin_path))
                    out_pt = os.path.join(out_dir, "train", sim_id, f"pair_{i:03d}_{j:03d}.pt")
                    train_out.append(out_pt)
                else:
                    # full cartesian: ini_path with all fin snapshots
                    for jj, fin_path in enumerate(fin_list):
                        train_pairs.append((ini_path, fin_path))
                        out_pt = os.path.join(out_dir, "train", sim_id, f"pair_{i:03d}_{j:03d}_to_{jj:03d}.pt")
                        train_out.append(out_pt)

        val_pairs: List[Tuple[str, str]] = []
        val_out:   List[str] = []
        if fin_val_group:
            for sim_id, ini_list in ini_group.items():
                if sim_id not in fin_val_group:
                    continue
                fin_list = fin_val_group[sim_id]
                for (ini_path, i, j) in ini_list:
                    if map_prev_j_to_fin:
                        if j < 0 or j >= len(fin_list):
                            raise IndexError(
                                f"prev output j={j:03d} out of range for val sim '{sim_id}': "
                                f"len(file_name)={len(fin_list)}."
                            )
                        fin_path = fin_list[j]
                        val_pairs.append((ini_path, fin_path))
                        out_pt = os.path.join(out_dir, "val", sim_id, f"pair_{i:03d}_{j:03d}.pt")
                        val_out.append(out_pt)
                    else:
                        for jj, fin_path in enumerate(fin_list):
                            val_pairs.append((ini_path, fin_path))
                            out_pt = os.path.join(out_dir, "val", sim_id, f"pair_{i:03d}_{j:03d}_to_{jj:03d}.pt")
                            val_out.append(out_pt)

        if return_pairs:
            return (train_pairs, train_out), (val_pairs, val_out)

        cfg.setdefault("global_files", {})
        cfg["global_files"]["file_pairs_train"] = train_pairs
        cfg["global_files"]["file_pairs_val"]   = val_pairs
        cfg["global_files"]["file_disp_train"]  = train_out
        cfg["global_files"]["file_disp_val"]    = val_out
        return cfg

    # ---------------------------
    # mode 1: legacy snapshot pairing 
    # ---------------------------
    else:
        def _resolve_fixed_index(names_sorted, fixed_ini):
            if fixed_ini is None:
                return None
            if isinstance(fixed_ini, int):
                if fixed_ini < 0 or fixed_ini >= len(names_sorted):
                    raise IndexError(f"fixed_ini index {fixed_ini} out of range [0,{len(names_sorted)-1}]")
                return fixed_ini
            if isinstance(fixed_ini, str):
                if fixed_ini not in names_sorted:
                    base = os.path.basename(fixed_ini)
                    if base in names_sorted:
                        fixed_ini = base
                    else:
                        raise ValueError(f"fixed_ini='{fixed_ini}' not found in file_name list: {names_sorted}")
                return names_sorted.index(fixed_ini)
            raise TypeError("fixed_ini must be None, int, or str")

        def _build_pairs_for_dirs(sim_dirs, snapshot_names, tag, fixed_ini=None):
            pairs = []
            out_paths = []

            names = sorted(snapshot_names)
            ini_idx = _resolve_fixed_index(names, fixed_ini)

            for sim_dir in sim_dirs:
                sim_dir = sim_dir.rstrip("/")
                sim_id = os.path.basename(sim_dir)

                if ini_idx is None:
                    # original: all (i<j)
                    for i in range(len(names)):
                        for j in range(i + 1, len(names)):
                            p1 = os.path.join(sim_dir, names[i])
                            p2 = os.path.join(sim_dir, names[j])
                            pairs.append((p1, p2))

                            out_pt = os.path.join(out_dir, tag, sim_id, f"pair_{i:03d}_{j:03d}.pt")
                            out_paths.append(out_pt)
                else:
                    # fixed ini: only (ini_idx, j) with j > ini_idx
                    i = ini_idx
                    for j in range(i + 1, len(names)):
                        p1 = os.path.join(sim_dir, names[i])
                        p2 = os.path.join(sim_dir, names[j])
                        pairs.append((p1, p2))

                        out_pt = os.path.join(out_dir, tag, sim_id, f"pair_{i:03d}_{j:03d}.pt")
                        out_paths.append(out_pt)

            return pairs, out_paths

        gf = cfg.get("global_files", {})
        if "file_dir_train" not in gf or "file_name" not in gf:
            raise RuntimeError("global_files.file_dir_train / global_files.file_name missing")

        train_dirs = _flatten_list(gf["file_dir_train"])
        val_dirs   = _flatten_list(gf.get("file_dir_val", []))
        snap_names = _flatten_list(gf["file_name"])

        train_pairs, train_out = _build_pairs_for_dirs(train_dirs, snap_names, tag="train", fixed_ini=fixed_ini)
        val_pairs, val_out = ([], [])
        if len(val_dirs) > 0:
            val_pairs, val_out = _build_pairs_for_dirs(val_dirs, snap_names, tag="val", fixed_ini=fixed_ini)

        if return_pairs:
            return (train_pairs, train_out), (val_pairs, val_out)

        cfg.setdefault("global_files", {})
        cfg["global_files"]["file_pairs_train"] = train_pairs
        cfg["global_files"]["file_pairs_val"]   = val_pairs
        cfg["global_files"]["file_disp_train"]  = train_out
        cfg["global_files"]["file_disp_val"]    = val_out
        return cfg

def run_prepare(cfg, stage_id):
    import importlib, copy

    prepare_cfg = cfg.get(f"stage{stage_id}_prepare", None)
    if not prepare_cfg:
        print(f"[Stage {stage_id}] No prepare step found.")
        return

    mod = importlib.import_module(prepare_cfg["module"])
    func = getattr(mod, prepare_cfg["function"])
    load_params = prepare_cfg.get("params_loading", {}).copy()
    base_params = prepare_cfg.get("params", {}).copy()

    cfg = expand_global_pairs_with_prev(cfg, **load_params)

    gf = cfg.get("global_files", {})
    train_pairs = gf.get("file_pairs_train", [])
    val_pairs   = gf.get("file_pairs_val", [])
    train_out   = gf.get("file_disp_train", None)
    val_out     = gf.get("file_disp_val", None)

    # TRAIN
    if len(train_pairs) > 0:
        params_train = copy.deepcopy(base_params)
        params_train["file_pairs"] = train_pairs
        if train_out is not None: params_train["outnames"] = train_out

        print(f"Preparing TRAIN data ({len(train_pairs)} pairs)")
        func(params_train)
    else:
        print("! No file_pairs_train found, skipped TRAIN prepare.")

    # VAL
    if len(val_pairs) > 0:
        params_val = copy.deepcopy(base_params)
        params_val["file_pairs"] = val_pairs
        if val_out is not None: params_val["outnames"] = val_out
        print(f"Preparing VAL data ({len(val_pairs)} pairs)")
        func(params_val)
    else:
        print("! No file_pairs_val found, skipped VAL prepare.")

# ============================== Train ==============================
def run_train(cfg, stage_id, rank=0, world_size=1):
    
    from torch.utils.data import DistributedSampler

    torch.cuda.empty_cache()
    print(f"[Stage {stage_id}] Cleaned GPU memory before training.")
    torch.cuda.synchronize()
    print(f"[Stage {stage_id}] Training started on rank {rank}/{world_size}...")

    # === Dataset ===
    train_dataset, val_dataset = load_datasets(cfg, stage_id=stage_id)

    # === Sampler ===
    train_cfg = cfg[f"stage{stage_id}_train"]
    epoch_size = train_cfg.get("epoch_size", None)  # e.g. 256
    seed = train_cfg.get("seed", 42)

    if epoch_size is not None:
        # Use epoch subset sampler
        train_sampler = EpochSubsetSampler(
            dataset_size=len(train_dataset),
            epoch_size=epoch_size,
            shuffle=True,
            seed=seed,
            rank=rank,
            world_size=world_size)
        print(f"Epoch subset enabled: epoch_size={epoch_size} / {len(train_dataset)} (world_size={world_size})")
    else:
        # full dataset
        if world_size > 1:
            train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        else:
            train_sampler = None

    val_sampler = None
    if (val_dataset is not None) and (world_size > 1):
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    # === DataLoader ===
    collate_fn = None
    if "collate_fn" in train_cfg:
        collate_cfg = train_cfg["collate_fn"]
        module = importlib.import_module(collate_cfg["module"])
        collate_fn = getattr(module, collate_cfg["function"])
        print(f"Using collate_fn from YAML: {collate_cfg['module']}.{collate_cfg['function']}")

    dataloader = make_loader(
        train_dataset,
        train_sampler,
        batch_size=train_cfg.get("batch_size", 1),
        num_workers=train_cfg.get("num_workers", 0),
        collate_fn=collate_fn,
    )
    print(f"Train loader ready: {len(train_dataset)} samples")

    val_loader = make_loader(
        val_dataset,
        val_sampler,
        batch_size=train_cfg.get("val_batch_size", train_cfg.get("batch_size", 1)),
        num_workers=train_cfg.get("val_num_workers", train_cfg.get("num_workers", 0)),
        collate_fn=collate_fn,
    )
    print(f"Val loader ready: {len(val_dataset)} samples")


    # === Model ===
    model_cfg = cfg[f"stage{stage_id}_model"]
    model_module = importlib.import_module(model_cfg["module"])
    ModelClass = getattr(model_module, model_cfg["class"])
    model = ModelClass(**model_cfg.get("params", {}))

    # === Loss ===
    loss_cfg = cfg[f"stage{stage_id}_loss"]
    loss_module = importlib.import_module(loss_cfg["module"])
    LossClass = getattr(loss_module, loss_cfg["class"])
    loss_fn = LossClass(**loss_cfg.get("params", {}))

    # === Optimizer ===
    opt_cfg = cfg[f"stage{stage_id}_optimizer"]
    opt_class = getattr(torch.optim, opt_cfg["name"])
    optimizer = opt_class(model.parameters(), **opt_cfg["params"])

    # === Scheduler ===
    sched_cfg = cfg[f"stage{stage_id}_scheduler"]
    sched_class = getattr(torch.optim.lr_scheduler, sched_cfg["name"])
    scheduler = sched_class(optimizer, **sched_cfg["params"])

    # === Eval function ===
    eval_fn = None
    if f"stage{stage_id}_eval_fn" in cfg:
        eval_cfg = cfg[f"stage{stage_id}_eval_fn"]
        eval_module = importlib.import_module(eval_cfg["module"])
        eval_fn = getattr(eval_module, eval_cfg["function"])

    # === Trainer ===
    trainer_module_name = train_cfg.get("trainer_module", "trainer")
    trainer_class_name = train_cfg.get("trainer_class", "Trainer")
    trainer_module = importlib.import_module(trainer_module_name)
    TrainerClass = getattr(trainer_module, trainer_class_name)

    trainer_params = train_cfg.get("trainer_params", {}).copy()

    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    trainer = TrainerClass(model, loss_fn, optimizer, scheduler, eval_fn,
                           device, rank, world_size,**trainer_params)

    print(f"Job working on {device}")

    trainer.run(dataloader, train_cfg, val_loader=val_loader)
    print(f"[Stage {stage_id}] Training finished.")

# ============================== Eval ==============================
def run_eval(cfg, stage_id):
    import copy
    eval_cfg = cfg.get(f"stage{stage_id}_eval", None)
    if eval_cfg:
        print(f"[Stage {stage_id}] Running eval...")

        prepare_cfg = cfg.get(f"stage{stage_id}_prepare", None)
        if prepare_cfg and "params_loading" in prepare_cfg:
            load_params = prepare_cfg["params_loading"].copy()
            cfg = expand_global_pairs_with_prev(cfg, **load_params)
        else:
            # fallback: if pairs already in cfg.global_files, do nothing
            gf = cfg.get("global_files", {})
            if not gf.get("file_pairs_train", []):
                raise RuntimeError(
                    f"No stage{stage_id}_prepare.params_loading and global_files.file_pairs_train is empty."
                )

        gf = cfg.get("global_files", {})
        val_pairs   = gf.get("file_pairs_val", [])
        val_out     = gf.get("file_disp_val", None)

        cfg_file_pairs = copy.deepcopy(cfg)
        cfg_file_pairs["file_pairs"] = val_pairs
        cfg_file_pairs["outnames"] = val_out 

        # === Eval class ===
        mod = importlib.import_module(eval_cfg["module"])
        EvalClass = getattr(mod, eval_cfg["class"])
        evaluator = EvalClass(**eval_cfg.get("params", {}))
        evaluator.run(cfg_file_pairs)
    else:
        print(f"[Stage {stage_id}] No eval step found.")

# ============================== Output ==============================
def run_output(cfg, stage_id):
    import copy

    output_cfg = cfg.get(f"stage{stage_id}_output", None)
    if not output_cfg:
        print(f"[Stage {stage_id}] No output step found.")
        return

    print(f"[Stage {stage_id}] Running output...")

    mod  = importlib.import_module(output_cfg["module"])
    func = getattr(mod, output_cfg["function"])

    cfg_expanded = expand_global_pairs(cfg, out_dir=output_cfg["out_dir"], fixed_ini=0)

    gf = cfg_expanded.get("global_files", {})
    train_pairs = gf.get("file_pairs_train", []) or []
    val_pairs   = gf.get("file_pairs_val", []) or []

    train_out   = gf.get("file_disp_train", None)  # 可能是 None / str / list[str]
    val_out     = gf.get("file_disp_val", None)

    out_dir = output_cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    def _run_split(split: str, pairs, outnames):
        if not pairs:
            print(f"[Stage {stage_id}] {split}: 0 pairs, skip.")
            return

        cfg_run = copy.deepcopy(cfg_expanded)
        cfg_run["file_pairs"] = pairs
        cfg_run["outnames"] = outnames

        print(f"[Stage {stage_id}] {split}: {len(pairs)} pairs -> {out_dir}")
        func(cfg_run, stage_id)

    _run_split("train", train_pairs, train_out)
    _run_split("val",   val_pairs,   val_out)



def init_project(cfg):
    from datetime import datetime
    project = cfg.get("project", {})
    name = project.get("name", "unnamed_project")
    base_dir = project.get("base_dir", "./projects")
    use_timestamp = project.get("use_timestamp_log", False)

    root_dir = os.path.join(base_dir, name)
    os.makedirs(root_dir, exist_ok=True)

    # === create timestamped log dir ===
    log_dir = os.path.join(root_dir, "logs")
    if use_timestamp:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_log_dir = os.path.join(log_dir, timestamp)
    else:
        run_log_dir = log_dir

    os.makedirs(run_log_dir, exist_ok=True)

    # === redirect terminal output ===
    if project.get("log_to_file", True):
        log_path = os.path.join(run_log_dir, "terminal.log")
        sys.stdout = open(log_path, "w")
        sys.stderr = sys.stdout
        print(f"Terminal output redirected to {log_path}")


    # === subfolders ===
    ckpt_dir = os.path.join(root_dir, "checkpoints")
    data_dir = os.path.join(root_dir, "data")

    for d in [ckpt_dir, data_dir]:
        os.makedirs(d, exist_ok=True)

    cfg["paths"] = dict(
        root=root_dir,
        logs=run_log_dir,
        checkpoints=ckpt_dir,
        data=data_dir,
    )

    # === save config snapshot ===
    cfg_save_path = os.path.join(run_log_dir, "config_used.yaml")
    with open(cfg_save_path, "w") as f:
        yaml.safe_dump(cfg, f)

    print(f"Project initialized: {name}")
    print(f"Logs: {run_log_dir}")
    print(f"Checkpoints: {ckpt_dir}")
    print("")
    return cfg

def resolve_paths(cfg):
    """Recursively scan all config dicts, prefix relative paths with project root."""
    project_root = cfg["paths"]["root"]
    modified = 0

    def _resolve_dict(d):
        nonlocal modified
        for k, v in d.items():
            if isinstance(v, str) and ("dir" in k or "path" in k):
                if not os.path.isabs(v):
                    new_path = os.path.join(project_root, v)
                    d[k] = new_path
                    modified += 1

            elif isinstance(v, dict):
                _resolve_dict(v)

    _resolve_dict(cfg)
    print(f"resolve_paths executed, updated {modified} relative paths.")
    return cfg


# === Pipeline ===
def main(args):
    import re
    cfg = load_yaml(args.config)
    cfg = init_project(cfg)
    cfg = resolve_paths(cfg)

    print('start')

    # === Stage list ===
    if args.stages:
        stages = [int(s) for s in args.stages]
    else:
        stage_keys = [k for k in cfg.keys() if k.startswith("stage")]
        stages = sorted({int(re.findall(r"stage(\d+)", k)[0]) for k in stage_keys})

    STEP_FUNCS = {
        "prepare": run_prepare,
        "train": run_train,
        "eval": run_eval,
        "output": run_output,
    }

    print("Starting multi-stage pipeline:")
    print("   → Stages:", stages)
    print("   → Steps per stage: prepare → train → eval → output\n")

    for stage_id in stages:
        print(f"\n==============================")
        print(f"Entering Stage {stage_id}")
        print("==============================")

        for step in ["prepare", "train", "eval", "output"]:
            step_key = f"stage{stage_id}_{step}"

            # === only one step ===
            if args.only:
                if step_key == args.only:
                    STEP_FUNCS[step](cfg, stage_id)
                    print(f"Finished only {step_key}")
                    return
                else:
                    continue

            # === run list steps ===
            if step_key in cfg:
                STEP_FUNCS[step](cfg, stage_id)

                if args.until == step_key:
                    print(f"Stopping at {step_key}")
                    return
            else:
                print(f"No config found for {step_key}, skipping.")

        # === link to next stage ===
        next_stage_id = stage_id + 1
        output_key = f"stage{stage_id}_output"
        next_prepare_key = f"stage{next_stage_id}_prepare"

        if (
            output_key in cfg
            and next_prepare_key in cfg
            and cfg[next_prepare_key]["params"].get("use_stage_pred", False)
        ):
            out_dir = cfg[output_key].get("out_dir", f"data/stage{stage_id}_outputs")
            next_params = cfg[next_prepare_key]["params"]
            next_params["stage_source_dir"] = out_dir
            print(f"Linked: Stage{stage_id}_output → Stage{next_stage_id}_prepare")
            print(f"Set stage_source_dir = {out_dir}")

    print("\n Pipeline finished successfully!\n")

if __name__ == "__main__":
    torch.cuda.empty_cache()

    import torch.distributed as dist

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--stages", nargs="+", help="Specify stages to run, e.g. --stages 1 2")
    parser.add_argument("--until", type=str, help="Stop after a specific step, e.g. stage2_train")
    parser.add_argument("--only", type=str, help="Run only a specific step, e.g. stage1_prepare")
    parser.add_argument("--world_size", type=int, default=1)
    args = parser.parse_args()


    if args.world_size > 1:
        rank = int(os.environ["RANK"])
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        main(args)
        dist.destroy_process_group()
    else:
        main(args)
