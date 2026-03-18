import os
import torch
import h5py
import importlib
from utils.Nbody_data_loader import NbodyLoader

def load_model(model_cfg, checkpoint_path, device):
    module = importlib.import_module(model_cfg["module"])
    Model = getattr(module, model_cfg["class"])
    model = Model(**model_cfg.get("params", {}))
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()

def eval_single(model, path_ini, path_fin, device, chunk_size: int = 200000):
    
    nb_ini = NbodyLoader(path_ini).load_nbody()
    nb_fin = NbodyLoader(path_fin).load_nbody()

    L     = float(nb_ini["box_size"])
    z_ini = float(nb_ini["redshift"])
    z_fin = float(nb_fin["redshift"])

    pos_ini_cpu = torch.as_tensor(nb_ini["pos"], dtype=torch.float32, device="cpu")
    vel_ini_cpu = torch.as_tensor(nb_ini["vel"], dtype=torch.float32, device="cpu")

    N = pos_ini_cpu.shape[0]

    pos_pred_cpu = torch.empty_like(pos_ini_cpu)

    z0 = torch.tensor(z_ini, device=device)
    z1 = torch.tensor(z_fin, device=device)

    model.eval()


    for s in range(0, N, chunk_size):
        e = min(N, s + chunk_size)

        # ---- move chunk to GPU ----
        pos_ini = pos_ini_cpu[s:e].to(device, non_blocking=True)
        vel_ini = vel_ini_cpu[s:e].to(device, non_blocking=True)

        # normalize exactly as training
        pos_in = (pos_ini / L).clamp(0.0, 1.0 - 1e-6)
        vel_in = vel_ini

        # forward (returns (M,3))
        pred_dpos_norm = model(pos_in, vel_in, z0, z1)
        pred_dpos = pred_dpos_norm * L

        pos_pred = torch.remainder(pos_ini + pred_dpos, L)
        pos_pred_cpu[s:e].copy_(pos_pred.detach().cpu())

    return pos_pred_cpu.numpy(), L, z_ini, z_fin

def output_stage(cfg, stage_id):

    output_cfg = cfg[f"stage{stage_id}_output"]
    model_cfg  = cfg[f"stage{stage_id}_model"]
    train_cfg  = cfg[f"stage{stage_id}_train"]
    recompute = output_cfg.get("recompute", False)

    # checkpoint
    ckpt_dir = train_cfg["checkpoint_dir"]
    ckpt = os.path.join(ckpt_dir, "best_model.pt")
    if not os.path.exists(ckpt):
        alt = os.path.join(ckpt_dir, "latest_checkpoint.pt")
        if os.path.exists(alt): ckpt = alt
        else: raise FileNotFoundError(f"Missing checkpoint in {ckpt_dir}")

    file_pairs = cfg.get("file_pairs", [])
    if len(file_pairs) == 0:
        raise RuntimeError("prepare_disp_zdep: cfg['file_pairs'] is empty")

    outnames = cfg.get("outnames", None)
    if outnames is not None and len(outnames) != len(file_pairs):
        raise RuntimeError("prepare_disp_zdep: len(outnames) must match len(file_pairs)")

    out_dir = output_cfg.get("out_dir", f"data/stage{stage_id}_outputs")
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"


    if os.path.exists(out_dir) and (not recompute):
        print(f"[Skip] exists: {out_dir}")
    else:
        model = load_model(model_cfg, ckpt, device)

        for i, (path_ini, path_fin) in enumerate(file_pairs, 1):

            pos_pred, L, z_ini, z_fin = eval_single(model, path_ini, path_fin, device, chunk_size = 200000)

            sim_name = os.path.basename(os.path.dirname(path_ini))
            name = os.path.basename(outnames[i-1])
            tag = name.replace(".pt", "")
            out_path = os.path.join(out_dir, f"{sim_name}_{tag}_pred.h5")

            with h5py.File(out_path, "w") as f:
                f["pos_pred"] = pos_pred
                f.attrs["box_size"] = L
                f.attrs["redshift_ini"] = z_ini
                f.attrs["redshift_final"] = z_fin

            print(f"Saved Stage{stage_id} prediction: {out_path}")

        print(f"Output generation finished: {len(file_pairs)} snapshots → {out_dir}")







