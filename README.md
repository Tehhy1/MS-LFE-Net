# MS-LFE-Net

**Point cloud semantic segmentation for open-pit mines with multi-scale local feature enhancement.**


This repository contains VGS–FPS downsampling and a PointNet++ model with a dual-branch local Transformer (BLT), boundary-aware gating (BGU), boundary-aware dual-stream pooling (BDP), and weighted cross-entropy (WCE). The three classes are **bench (0), slope (1), and benchline (2)**.

## Repository layout

```text
MS-LFE-Net/
  VGS_FPS.py                     # Four-stage downsampling
  train_semseg.py                 # Training, resumption and scene evaluation
  models/
    pointnet2_sem_seg_msg.py      # PointNet++ MSG + MS-LFE
    pointnet2_utils.py            # Grouping, geometry, BDP and feature propagation
    blt.py                       # Dual-branch attention and BGU
  TRAINING.md                    # Detailed experiment and metric conventions
  requirements.txt
  LICENSE
  THIRD_PARTY_NOTICES.md
```

## Environment

The manuscript reports Python 3.10, PyTorch 2.7.1, CUDA 11.8 and an RTX 3090. Install a CUDA-enabled PyTorch build compatible with the experiment workstation, then install the dependencies:


```bash
pip install -r requirements.txt
python VGS_FPS.py --help
python train_semseg.py --help
```

Both entry points require CUDA. VGS, voxel calibration and XY grouping use NumPy on the CPU; FPS uses CUDA. There is no CPU FPS or training fallback in the command-line entry points.


## 1. VGS–FPS downsampling

```bash
python VGS_FPS.py input.txt sampled.txt --sample-ratio 0.025 --block-size 100 100
```

Input text has column names on the first line, the point count on the second line, and comma-separated numeric rows thereafter. The first three columns must be XYZ. All original attribute/label columns are preserved in the output. A companion `sampled.txt.indices.npy` records the selected row indices in the input file. Use a new output filename.


### Four stages

For N input points, the final count is **K = floor(N × sample_ratio)**.

1. **VGS:** target `ceil(0.1N)` candidates when `sample_ratio <= 0.05`, otherwise `ceil(0.2N)`. Estimate the voxel edge from the 3D bounding-box volume, then calibrate using actual occupied voxel counts. Keep the original point nearest each occupied voxel's centroid.
2. **Global skeleton FPS:** select `max(1, ceil(K × skeleton_ratio))` candidates as skeleton points.
3. **XY block FPS:** partition the remaining candidates into fixed-width XY columns spanning the full Z range. Allocate integer local-FPS quotas using weights proportional to `candidate_count ** density_power`.
4. **Merged global FPS:** merge skeleton/local samples and apply global FPS to return exactly K points. The merged target is approximately `merge_factor × K`, capped by available candidates.


| Parameter | Default | Meaning |
|---|---:|---|
| `--sample-ratio` | **0.025 = 1/40** | Final retained fraction of input rows |
| `--skeleton-ratio` | 0.001 | Skeleton fraction of **K**, not N |
| `--block-size BX BY` | Required / 必填 | XY cell widths in coordinate units / XY |
| `--merge-factor` | 2.0 | Merged candidate budget |
| `--density-power` | 0.5 | Density weighting exponent |
| `--seed` | 0 | Initial FPS point seed |
| `--chunk-size` | 100000 | Distance-update batch size, **not spatial block size**  |

The Python function and CLI both default to **1/40**. To retain the earlier 1/80 fraction, explicitly pass `--sample-ratio 0.0125`. The example `100 100` means 100 m × 100 m for metre coordinates and is adjustable, not a validated optimum from the manuscript.


The implemented VGS rule requires `0 < sample_ratio < 0.2` and at least one final point. With volume V and target candidate count T, the initial voxel edge is `(V/T)^(1/3)`. Mine points occupy surfaces rather than the full volume, so occupancy calibration is necessary. It retains at least T candidates; the 10%/20% fractions are approximate, not exact. Inputs are assumed to have positive XYZ extents and enough distinct coordinates.


The final greedy FPS still performs O(MK) distance updates for M merged candidates and K outputs. Million-point outputs can remain very expensive. Voxel calibration also makes multiple full-data passes; this implementation loads data in memory rather than streaming it.


## 2. Prepare model input

**Downsampling output is not yet a training dataset.** This repository does not include the block-generation script. Prepare the arrays below from the annotated, downsampled scene:

```text
data_root/
  raw_points.npy                   # [N,C>=3], XYZ in metres
  raw_labels.npy                   # [N], classes 0/1/2
  train/
    blocks_points_train.npy         # [B,4096,C>=3]
    blocks_labels_train.npy         # [B,4096]
    blocks_indices_train.npy        # [B,4096], rows of raw_points/raw_labels
  test/
    blocks_points_test.npy
    blocks_labels_test.npy
    blocks_indices_test.npy
```

Here `raw_*` means the complete **downsampled scene before block generation**. Split its global X extent into 80% training and 20% test regions first. Generate 120 m × 120 m windows with an 80 m stride separately within each region, then create 4096-point inputs. Training must contain all three classes.


Full-scene testing requires inference blocks covering every original test-region point. The trainer reports coverage and explicitly labels incomplete results as covered-subset metrics. Uncovered predictions are saved as -1; no labels are fabricated.


## 3. Model

- SA point counts: `1024, 256, 64, 16`.
- SA radius pairs: `[2,6], [6,10], [10,20], [20,40]` metres.
- SA output channels: `96, 256, 512, 1024`.
- Geometry: KNN=16; PCA normals, curvature and neighborhood height range.
- BLT: geometric/semantic attention branches, scalar BGU and residual fusion.
- BDP: `(1-theta)*boundary_mean + theta*interior_mean`, initialized with theta=0.5.
- Input: geometry only; XYZ is translated but not rescaled.

The MSG implementation in `pointnet2_utils.py` directly performs BDP. The standalone `BoundaryDualStreamPooling` class in `blt.py` is not called by this model. The active 14-channel geometric descriptor comprises center XYZ, neighbor XYZ, relative XYZ, neighbor normal, curvature and height range; this is an implementation choice for the manuscript's geometric branch.


## 4. Training and evaluation

```bash
# Default: batch 8, epochs 100, Adam, LR 0.0005, WCE.
python train_semseg.py --data_root DATA_ROOT --log_dir log/mine1_seed0 --gpu 0 --seed 0

# Epoch-boundary resume; --epoch is the TOTAL target number of epochs.
python train_semseg.py --data_root DATA_ROOT --log_dir log/mine1_seed0 --resume log/mine1_seed0/checkpoints/last_model.pth --epoch 100 --seed 0

# Evaluate a trained checkpoint without training.
python train_semseg.py --data_root DATA_ROOT --log_dir log/mine1_eval --resume log/mine1_seed0/checkpoints/last_model.pth --eval_only

# Three independent runs and aggregate statistics.
python train_semseg.py --data_root DATA_ROOT --log_dir log/mine1_repeated --seeds 0 1 2

# WCE ablation: replace WCE with ordinary CE.
python train_semseg.py --data_root DATA_ROOT --log_dir log/mine1_ce --loss ce
```

Use `--val_dir VAL_DIRECTORY` for a separately prepared, disjoint validation region within the training 80%. Validation files use the `_val.npy` suffix. With validation, select the best validation mIoU; otherwise use the final epoch. Test data never select the checkpoint. `--skip_test` runs training/validation only.


The script reports OA, mIoU, per-class IoU/Precision/Recall, macro Precision/Recall, confusion matrices, per-epoch training time, peak allocated/reserved GPU memory and test coverage. Aggregate Precision/Recall use macro averaging in the primary output; weighted/micro averages are also saved. Repeated experiments save both population and sample standard deviations. The manuscript does not fully specify these aggregate conventions.

Training scores count block occurrences for monitoring. Scene scores count each original point once after majority voting across overlapping blocks. See **[TRAINING.md](TRAINING.md)** for full metric definitions, data conventions, validation/resumption details and output files.

Current model files do not expose BGU/BDP or radius-ablation switches. `--model` requires a compatible model module; other comparison architectures are not included. Old trainer flags such as `--use_bgu`, `--use_bdp` and `--device` are not supported by this consolidated entry point.

## Verification and scope

Local checks cover VGS ratio thresholds and output indices, XY partitioning, quota budgets, an independent FPS reference, hand-calculated metrics, overlapping-block voting, epoch-boundary resumption, validation selection, repeated-seed statistics and training with the actual model on synthetic data. A 4096-point model forward/backward check is also included in the review.

CUDA calls were intercepted only inside the local checking process because the review computer has no CUDA GPU. The delivered entry points remain CUDA-only. Actual GPU execution, memory, full-mine runtime and manuscript accuracy have not been measured. This repository contains no private mine dataset or pretrained weights and does not claim exact reproduction of the reported results.

## License and acknowledgements

Released under the **[MIT License](LICENSE)**. Original PointNet++ code is derived from [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch). The original `Copyright (c) 2019 benny` notice is retained, alongside the notice for MS-LFE-Net contributions. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
