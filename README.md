# TUNeS (Temporal UNet emulator for Structure formation)

TUNeS is a neural network framework for accelerating N-body simulations by predicting the nonlinear evolution of the matter density field from an initial particle distribution. TUNeS employs a two-stage modeling strategy

- **Stage 1**: Particle-based model that predicts coarse displacements  
- **Stage 2**: 3D UNet that refines the density field on a grid  

This repository provides both:
1. **Quick run using pretrained models**
2. **Full training pipeline for customization**

<img width="673" height="1024" alt="pipeline (1)" src="https://github.com/user-attachments/assets/d79a5a6f-9500-4c7e-82e8-ae3c91b3bfdd" />

---

# 1. Quick Run (Pretrained Model)

This is the **simplest way to use TUNeS**.  
You only need to provide an input snapshot, and the model will generate the evolved structure.

## Requirements

- Input: N-body snapshot (e.g. `snapshot_000.hdf5`)
- Pretrained model: already provided in this repository
- GPU (recommended)

---

## Run

```bash
python TUNeS_Emulator.py \
  -model_dir /path/to/trained_model/model \
  -config /path/to/trained_model/model/config.yaml \
  -input snapshot_000.hdf5 \
  -output out_dir \
  -z_fin 0.0
```

## Arguments

- `-model_dir`  
  Path to the pretrained model directory provided in this repository

- `-config`  
  Path to the configuration file. This is mainly used to locate the checkpoints directory, where the trained model parameters are stored.
  
- `-input`  
  Initial snapshot (e.g. Gadget HDF5 format)

- `-output`  
  Directory to save outputs

- `-z_fin` *(optional, default = 0.0)*  
  Target redshift

## Output

The output directory will contain:

- Stage 1 predictions (particle positions)
- Stage 2 predictions (density field)

---

# 2. Training Your Own Model

If you want to train TUNeS on your own simulations or modify the architecture, use the training pipeline.

## Configuration

An example configuration file is provided:

```text
config/config_example.yaml
```

You only need to modify:

- preject name and base_dir
- global_files, which is the traning set
- stage2_local_files, which are the rho data genrate with pos2dens.py provied in this repository
- optional model hyperparameters

## Run the Training Pipeline

```bash
python run_pipeline_zdep.py --config config/config_example.yaml
```

## Pipeline Control

The pipeline is modular. You can control execution using the --until and --only options.

### Run up to a stage

```bash
python run_pipeline_zdep.py \
  --config config/config_example.yaml \
  --until stage2_train
```

### Run only a specific stage

```bash
python run_pipeline_zdep.py \
  --config config/config_example.yaml \
  --only stage1_prepare
```
