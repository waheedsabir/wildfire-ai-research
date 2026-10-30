# Research baseline: reproducible image classification and PyTorch DDP

**Status:** experimental training *code*, not completed wildfire detection research. No metrics or pretrained wildfire detector are shipped. This module is a **binary image-level classifier** (fire/no_fire) and does **not** localize fire/smoke or estimate spread. It provides a stepping stone toward the proposed YOLO-based object detection workstream.

## Architectures

- `resnet18` — CNN baseline
- `mobilenet_v3_small` — lightweight CNN baseline
- `efficientnet_b0` — efficient CNN baseline
- `tiny_cnn` — only for developer smoke tests; **not** a research baseline

Default weights are random and training starts from scratch. `--pretrained` downloads ImageNet weights on demand and should be documented explicitly if used.

## Data layout

Use only genuinely licensed, accurately labeled fire / no-fire **RGB photographs**. This starter expects a local `ImageFolder` layout (not included):

```text
my_dataset/
  train/
    fire/*.jpg
    no_fire/*.jpg
  val/
    fire/*.jpg
    no_fire/*.jpg
```

Build splits by **wildfire event, scene, time and source** rather than random adjacent frames to avoid leakage. Do not treat NASA FIRMS point coordinates as image-level training labels or satellite images. The input module assumes three-channel RGB; multispectral imagery requires explicit preprocessing changes.

## Setup

Install compatible versions of PyTorch and torchvision for your machine from https://pytorch.org/get-started/locally/ and Pillow if needed. For a CPU-only environment:

```bash
python -m pip install torch torchvision pillow
```

## Single process (CPU/GPU)

```bash
python training/train_ddp.py --data-root /path/to/my_dataset --model resnet18 --epochs 5 --out outputs/resnet18
```

## Single-node two-GPU DDP

```bash
torchrun --standalone --nproc_per_node=2 training/train_ddp.py \
  --data-root /path/to/my_dataset --model mobilenet_v3_small --epochs 5 \
  --batch-size 16 --out outputs/mobilenet_ddp
```

`torchrun` requires available GPUs for the GPU example; it can also use CPU/Gloo for debugging. DDP uses one process per GPU, train-data sharding, synchronized gradients, and rank-0 full validation. Input `--batch-size` is **per process**, so global effective batch size is approximately batch-size times world size. Re-evaluate LR when scaling.

## Output and verification

Only rank 0 writes `config.json`, `metrics.jsonl`, and a `last.pt` checkpoint to `--out`. Generated experiment outputs are excluded by `.gitignore` to avoid accidental dataset/checkpoint uploads. Validation metrics are image-level classification accuracy, macro-F1 and confusion matrix, **not** smoke bounding box mAP or real-time wildfire reliability.

The integration test uses **synthetically generated colored images** exclusively to verify code execution. Synthetic test accuracy is meaningless for scientific claims.

```bash
python -m unittest discover -s tests -v
```

## Follow-up research milestones

1. Curate and document a real, licensed fire/smoke dataset, with event-held-out validation.
2. Run reproducible baselines with exact hardware, commands, seeds and dataset checksums.
3. Add proper bounding-box or segmentation annotations, then train YOLO family detectors and investigate cloud-vs-smoke false alarms.
4. Evaluate attention components such as ECA/CBAM only after comparable baselines are established.

This code and initial documentation are AI-assisted scaffolding. Scientific claims should be supported by actual runs and independently inspectable data/results. **Research only; not for emergency decision-making.**
