# SWIN-RTDETR-Implementation

# Swin-RTDETR: Swin Transformer Backbone + RT-DETR-L Head for SIXray Prohibited-Item Detection

A single-cell, reproducible Colab training script that replaces RT-DETR-L's native CNN
backbone with a Swin Transformer, while keeping the official RT-DETR hybrid encoder
(AIFI + FPN/PAN) and decoder. Includes an optional Focal Classification Loss + CIoU
loss variant as an alternative to RT-DETR's native Varifocal + GIoU loss, COCO-pretrained
head-weight transfer, and an extensive self-verifying reproducibility pipeline.

## Key features

- **Swap-in Swin backbone** (`base` / `small` / `tiny` presets) feeding the *unmodified*
  official RT-DETR-L hybrid encoder and decoder, so architectural changes are isolated to
  the backbone alone.
- **Auto-generated model YAML** — every backbone/head layer index is computed
  programmatically (`build_model_yaml`) rather than hand-typed, eliminating the class of
  bug where a manually-edited YAML silently wires the decoder to the wrong feature maps.
- **Wiring self-check** (`check_wiring`) — runs a real forward pass and asserts the decoder
  actually receives `[P3, P4, P5]` at the expected channel count and stride before training
  starts, rather than discovering a mis-wired architecture after a wasted training run.
- **Loss self-test** (`loss_selftest`) — runs one synthetic forward/backward pass before
  training and asserts: the loss is finite, gradients reach the Swin backbone, and (when
  `LOSS = "focal_ciou"`) the Focal and CIoU code paths were actually invoked and not
  silently bypassed.
- **Two selectable loss configurations**:
  - `focal_ciou` — sigmoid Focal Loss (classification) + CIoU + L1 (box regression)
  - `native` — RT-DETR's stock Varifocal + GIoU + L1
  - Implemented as a `RTDETRDetectionLoss` subclass (`FocalCIoURTDETRLoss`), not a global
    monkey-patch, so it can't leak into unrelated model instances in the same process.
  - **Note:** the Hungarian matcher's assignment cost still uses GIoU in both configurations
    — only the final regression loss changes. This is recorded explicitly in the manifest
    (`detector.loss.matcher_cost`) so it's never an implicit, undocumented assumption.
- **COCO-pretrained head transfer** (`transfer_rtdetr_head`) — remaps AIFI/FPN/PAN/decoder
  tensors from `rtdetr-l.pt` onto the Swin-backboned model by head-relative layer position
  (not by absolute index, which differs from the stock checkpoint once the backbone is
  swapped), with a per-layer transferred/total tensor report printed at build time.
- **Separate learning rates** for the Swin backbone vs. the neck/decoder
  (`BACKBONE_LR_MULT`), applied via custom optimizer parameter groups.
- **Automatic input-size fallback with a smoke test** — tries each `(image size, Swin
  window size)` candidate in isolation before committing to a full training run; records
  which candidate actually succeeded in the manifest.
- **Baseline-comparison tooling** — point `BASELINE_ARGS_YAML` at a prior run's
  `args.yaml` to (a) optionally inherit its exact augmentation settings so both runs see
  identical data, and (b) print a diff of every watched hyperparameter that differs between
  this run and the baseline.
- **GPU utilization monitor** — samples `nvidia-smi` during training and prints a verdict
  (GPU-bound / partly data-bound / data-bound) with a concrete suggestion (raise
  `WORKERS`, enable `CACHE`, use a smaller backbone, etc.).
- **Resume support** — set `RESUME_FROM` to a `last.pt` path to continue an interrupted run
  (e.g. after a Colab disconnect); works with `PROJECT_DIR` on Google Drive.
- **Full reproducibility manifest** (`reproducibility_manifest.json`) — every applied
  setting (not just requested ones), including resolved pretrained-weight source, per-layer
  weight-transfer counts, optimizer groups, GPU utilization, and self-test results.
- **Efficiency profiling** — parameter count, GFLOPs, and FP16 batch-1 latency/FPS on the
  final trained checkpoint.
- **Results bundle** — zips the manifest, plots, and metrics (excluding model weights, to
  keep the download small) for easy retrieval from Colab.

## Requirements

- Google Colab (or any environment with a CUDA GPU; CPU training is supported but not
  practical at this model scale)
- Python packages (installed automatically by the script): `ultralytics`, `timm`, `pyyaml`,
  `roboflow` (only if downloading the dataset fresh)

## Quick start

1. Open the script in a Colab cell.
2. Set your Roboflow API key via the `ROBOFLOW_API_KEY` environment variable, a Colab
   secret of the same name, or the interactive prompt.
3. Run the cell. On first run it will:
   - install/verify package versions,
   - download (or reuse) the SIXray dataset,
   - smoke-test the Swin backbone at the configured input size,
   - build and wiring-check the model,
   - run the loss self-test,
   - train, evaluate, and write the reproducibility manifest.

## Configuration

All settings live in the `CFG` class at the top of the script and can be overridden via
environment variables (useful for scripted sweeps without editing the file), e.g.:

```bash
SWIN_VARIANT=small IMG_SIZE=512 LOSS=native SEED=43 EPOCHS=50 python train.py
```

| Setting | Purpose |
|---|---|
| `SWIN_VARIANT` | `base` / `small` / `tiny` — trades accuracy for speed |
| `IMG_SIZE`, `ALLOW_FALLBACK` | Training resolution; whether to silently fall back to a smaller size if the requested one fails the smoke test |
| `LOSS`, `FOCAL_GAMMA`, `FOCAL_ALPHA`, `LOSS_GAIN` | Loss configuration (see "Two selectable loss configurations" above) |
| `TRANSFER_HEAD` | Whether to initialize the neck/decoder from COCO-pretrained `rtdetr-l.pt` |
| `BACKBONE_LR_MULT` | Swin LR as a fraction of the neck/decoder LR |
| `AUG_PRESET` / `BASELINE_ARGS_YAML` / `INHERIT_FROM_BASELINE` | Augmentation policy, and whether to copy it from a baseline run for a fair comparison |
| `SEED` | Set to `42` / `43` / `44` etc. across repeated runs to report mean ± SD, as recommended when comparing closely-scoring configurations |
| `CONF_THRESH_MAP`, `CONF_THRESH_OP` | Confidence threshold used for the standard mAP computation vs. the stated operating-point precision/recall/F1 |
| `RESUME_FROM` | Path to `last.pt` to resume an interrupted run |

## Output

Each run writes to `PROJECT_DIR/<RUN_NAME>/`:

- `weights/best.pt`, `weights/last.pt`
- `swin_rtdetr_l_corrected.yaml`, `swin_backbone.py`, `focal_ciou_loss.py` — copied
  alongside the weights, since both custom modules are required to unpickle the checkpoint
- `reproducibility_manifest.json` — the authoritative record of what was actually run
- Standard Ultralytics training plots/CSVs

`RUN_NAME` is generated automatically from the active configuration (Swin variant, loss
type, head-transfer setting, epoch/time budget, image size, LR, seed), so distinct
configurations never silently overwrite each other's output directory.

## Known limitations / things to check before citing results

- The Hungarian matcher's assignment cost is **not** changed by the `focal_ciou` loss
  setting — only the final regression loss term is. If your write-up states that CIoU
  replaces GIoU "throughout," verify this matches what's actually implemented here.
- `focal_ciou` vs. `native` changes both the loss **and** implicitly compares against a
  backbone that the stock RT-DETR-L checkpoint wasn't trained with — a performance
  difference between the two settings conflates the loss change with the backbone change
  unless you also run a stock-backbone baseline under both losses.
- No multi-seed aggregation is built in. Run the script once per seed (`SEED=42`, `43`,
  `44`, ...) and aggregate `reproducibility_manifest.json["evaluation"]["map_metrics"]`
  across runs yourself (mean ± SD) before reporting a single headline number, especially
  when comparing two configurations with a small score gap.


  ## Results

The proposed Swin-RTDETR model was evaluated on the SIXray test set, achieving a mean
average precision (mAP@50) of **94.37%**, with an average precision of **95.00%** and
recall of **89.70%** across all prohibited-item categories.

<img width="2235" height="1184" alt="results-1" src="https://github.com/user-attachments/assets/ec506986-f610-4ca7-9bc7-2dd06c5fb7ed" />

