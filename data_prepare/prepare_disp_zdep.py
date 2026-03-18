import os
import torch

def periodic_diff(pos, ref, box_size):
    diff = pos - ref
    diff -= box_size * torch.round(diff / box_size)
    return diff


@torch.no_grad()
def prepare_disp_zdep(params):

    from NeuralNbody.data_handler import DataLoader as NbodyLoader

    out_dir   = params["out_dir"]
    recompute = bool(params.get("recompute", False))
    dtype_str = str(params.get("dtype_str", "float16")).lower()

    file_pairs = params.get("file_pairs", [])
    if len(file_pairs) == 0:
        raise RuntimeError("prepare_disp_zdep: params['file_pairs'] is empty")

    outnames = params.get("outnames", None)
    if outnames is not None and len(outnames) != len(file_pairs):
        raise RuntimeError("prepare_disp_zdep: len(outnames) must match len(file_pairs)")

    if dtype_str == "float16":
        save_dtype = torch.float16
    elif dtype_str == "float32":
        save_dtype = torch.float32
    else:
        raise ValueError(f"prepare_disp_zdep: unsupported dtype_str={dtype_str}")

    os.makedirs(out_dir, exist_ok=True)

    n_skip = 0
    n_done = 0

    for idx, (path_ini, path_final) in enumerate(file_pairs):
        snapshot_tag = os.path.basename(path_ini).replace(".hdf5", "")

        # output path
        if outnames is not None:
            out_pt = outnames[idx]
        else:
            out_pt = os.path.join(out_dir, f"pair_{idx:06d}.pt")

        os.makedirs(os.path.dirname(out_pt), exist_ok=True)

        if (not recompute) and os.path.exists(out_pt):
            n_skip += 1
            continue

        print(f"\n=== [Prepare] {idx+1}/{len(file_pairs)} | {snapshot_tag} -> {os.path.basename(path_final)} ===")

        nbody_ini   = NbodyLoader(path_ini).load_nbody()
        nbody_final = NbodyLoader(path_final).load_nbody()

        box_size = float(nbody_ini["box_size"])
        z_ini    = float(nbody_ini["redshift"])
        z_fin    = float(nbody_final["redshift"])

        pos_ini   = torch.as_tensor(nbody_ini["pos"], dtype=torch.float32)
        pos_final = torch.as_tensor(nbody_final["pos"], dtype=torch.float32)

        # (optional) enforce periodic range before diff
        pos_ini   = torch.remainder(pos_ini, box_size)
        pos_final = torch.remainder(pos_final, box_size)

        dpos = periodic_diff(pos_final, pos_ini, box_size).to(save_dtype).cpu()

        torch.save(
            {
                "dpos": dpos,          
                "z_ini": z_ini,
                "z_fin": z_fin,
                "ini_path": path_ini,
                "box_size": box_size,
            },
            out_pt,
        )
        n_done += 1

        if (idx + 1) % 50 == 0:
            print(f"[Prepare] progress: {idx+1}/{len(file_pairs)} | done={n_done} skip={n_skip}")

    print(f"[Prepare] finished: total={len(file_pairs)} done={n_done} skipped={n_skip} out_dir={out_dir}")






