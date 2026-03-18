import torch
from utils.prepare_window import build_windows_from_density,build_windows_from_snapshot

@torch.no_grad()
def prepare_window_3d_dens(params):
    import os
    # ---------------- basic params ----------------
    out_dir       = params.get("out_dir", "data/window_dens")
    os.makedirs(out_dir, exist_ok=True)
    stage1_out    = params["stage_source_dir"]

    Nw            = int(params["Nw"])
    step        = int(params["step"])
    box_size        = int(params["box_size"])
    recompute     = bool(params.get("recompute", False))
    file_pairs    = params.get("file_pairs", [])
    grid_res      = int(params.get("grid_res", 64))
    device        = torch.device(params.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    dtype         = getattr(torch, params.get("dtype", "float32"))
    trans_mode    = params.get("transform", "asinh")
    trans_alpha   = float(params.get("alpha", 1.0))

    # ---------------- main loop ----------------
    for i, (path_ini, path_final) in enumerate(file_pairs, 1):

        tag_pred = os.path.basename(path_ini).replace(".h5", "")
        sim_name_final = os.path.basename(os.path.dirname(path_final))
        snapshot_tag   = os.path.basename(path_final).replace(".h5", "")
        tag_final      = f"{sim_name_final}_{snapshot_tag}_target"

        pred_path = path_ini
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"Stage-1 predicted snapshot not found: {pred_path}")

        pred_out_dir  = os.path.join(out_dir, f"{tag_pred}")
        final_out_dir = os.path.join(out_dir, f"{tag_final}")
        os.makedirs(pred_out_dir, exist_ok=True)
        os.makedirs(final_out_dir, exist_ok=True)

        out = build_windows_from_snapshot(
                snapshot_path=pred_path,
                grid_size=grid_res,
                Nw=Nw, step=step,
                delta_mode=trans_mode,
                alpha=trans_alpha,
                device=device,
                out_dir=pred_out_dir,
                prefix="dens",
                recompute=recompute,
                save_full=True,
                save_windows=True)

        out = build_windows_from_density(
                rho=path_final,
                box_size=box_size,
                Nw=Nw, step=step,
                delta_mode=trans_mode,
                alpha=trans_alpha,
                out_dir=final_out_dir,
                prefix="dens",
                recompute=recompute,
                save_full=True,
                save_windows=True)
        


        