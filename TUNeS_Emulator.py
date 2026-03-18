import time
import argparse

from utils.full_out_zdep import NbodyEmulator
from utils.Nbody_data_loader import NbodyLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model_dir", required=True, help="model directory")
    parser.add_argument("-config", required=True, help="config file used for running the model")
    parser.add_argument("-input", required=True, help="initial snapshot")
    parser.add_argument("-output", required=True, help="output directory")
    parser.add_argument("-z_fin", type=float, default=0.0, help="target redshift (default=0)")
    args = parser.parse_args()

    model_dir = args.model_dir
    config = args.config
    input_initial = args.input
    output_dir = args.output
    z_fin = args.z_fin

    nb_ini = NbodyLoader(input_initial).load_nbody()
    z_ini = float(nb_ini["redshift"])

    print(f"[INFO] z_ini={z_ini:.4f} -> z_fin={z_fin:.4f}")

    t0 = time.time()

    emu = NbodyEmulator(model_dir=model_dir, config=config, device="cuda")

    stage1_pred_path = emu.run_stage1(
        z_fin,
        input_initial,
        output_dir=output_dir
    )

    t1 = time.time() - t0
    print("Stage1:", stage1_pred_path, " Time:", f"{t1:.3f}s")

    stage2_rec_path = emu.run_stage2(
        z_fin,
        stage1_output=stage1_pred_path,
        output_dir=output_dir
    )

    t2 = time.time() - t0

    print("Done.")
    print("Stage2:", stage2_rec_path, " Time:", f"{t2:.3f}s")


if __name__ == "__main__":
    main()
