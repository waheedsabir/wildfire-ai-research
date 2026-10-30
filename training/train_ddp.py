#!/usr/bin/env python3
"""Research-only fire/no-fire image classification baseline with optional PyTorch DDP.

This is an experiment runner, not a trained fire detection system. Train only on
appropriately licensed and labeled datasets; validate on independent wildfire events.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, models, transforms

MODEL_NAMES = ("resnet18", "mobilenet_v3_small", "efficientnet_b0", "tiny_cnn")


def make_model(name: str, num_classes: int, pretrained: bool) -> nn.Module:
    """Create a classifier. tiny_cnn is intended only for integration tests."""
    if name == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
    elif name == "mobilenet_v3_small":
        net = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        )
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
    elif name == "efficientnet_b0":
        net = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
    elif name == "tiny_cnn":
        if pretrained:
            raise ValueError("tiny_cnn has no pretrained weights")
        net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(8, num_classes),
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    return net


def initialize_distributed() -> tuple[bool, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    distributed = world_size > 1
    if distributed:
        dist.init_process_group(backend=backend, init_method="env://")
    return distributed, rank, world_size, device


def make_loaders(root: Path, image_size: int, batch_size: int, workers: int,
                 distributed: bool, rank: int, world_size: int, device: torch.device):
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize,
    ])
    valid_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize,
    ])
    train_dir, valid_dir = root / "train", root / "val"
    if not train_dir.is_dir() or not valid_dir.is_dir():
        raise ValueError("Expected train/ and val/ directories under --data-root")
    training = datasets.ImageFolder(train_dir, transform=train_tf)
    validation = datasets.ImageFolder(valid_dir, transform=valid_tf)
    if len(training.classes) != 2 or training.classes != validation.classes:
        raise ValueError("Expected matching two-class train/ and val/ subdirectories")
    if not training or not validation:
        raise ValueError("Train and validation datasets must both be non-empty")
    sampler = DistributedSampler(training, num_replicas=world_size, rank=rank,
                                 shuffle=True, seed=42) if distributed else None
    loader = DataLoader(training, batch_size=batch_size, sampler=sampler,
                        shuffle=sampler is None, num_workers=workers,
                        pin_memory=device.type == "cuda")
    # Evaluation is done once, on rank 0, without DistributedSampler padding.
    val_loader = DataLoader(validation, batch_size=batch_size, shuffle=False,
                            num_workers=workers, pin_memory=device.type == "cuda") if rank == 0 else None
    return loader, val_loader, sampler, training.classes, len(training), len(validation)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int):
    model.eval()
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    total_loss = 0.0
    n = 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            n += labels.numel()
            preds = logits.argmax(dim=1)
            for true, pred in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                matrix[true, pred] += 1
    accuracy = matrix.diagonal().sum().item() / max(n, 1)
    f1s = []
    for k in range(num_classes):
        tp = matrix[k, k].item()
        fp = matrix[:, k].sum().item() - tp
        fn = matrix[k, :].sum().item() - tp
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return {"val_loss": total_loss / max(n, 1), "val_accuracy": accuracy,
            "val_macro_f1": sum(f1s) / num_classes,
            "val_confusion_matrix": matrix.tolist()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True,
                   help="ImageFolder root with train/{fire,no_fire} and val/{fire,no_fire}")
    p.add_argument("--model", choices=MODEL_NAMES, default="resnet18")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16, help="Per-process batch size")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pretrained", action="store_true", help="Fetch torchvision pretrained weights")
    p.add_argument("--out", type=Path, default=Path("outputs/experiment"))
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 16 or args.lr <= 0:
        p.error("Require epochs>=1, batch-size>=1, image-size>=16, lr>0")

    distributed, rank, world_size, device = initialize_distributed()
    try:
        random.seed(args.seed + rank)
        torch.manual_seed(args.seed)  # all ranks begin from identical parameters
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        train_loader, val_loader, sampler, classes, n_train, n_val = make_loaders(
            args.data_root, args.image_size, args.batch_size, args.num_workers,
            distributed, rank, world_size, device)
        model = make_model(args.model, len(classes), args.pretrained).to(device)
        training_model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None) \
                         if distributed else model
        optimizer = torch.optim.AdamW(training_model.parameters(), lr=args.lr)
        if rank == 0:
            args.out.mkdir(parents=True, exist_ok=True)
            config = {"created_utc": datetime.now(timezone.utc).isoformat(),
                      "framework_torch": torch.__version__, "model": args.model,
                      "pretrained": args.pretrained, "epochs": args.epochs,
                      "batch_size_per_process": args.batch_size, "world_size": world_size,
                      "image_size": args.image_size, "lr": args.lr, "seed": args.seed,
                      "classes": classes, "training_images": n_train,
                      "validation_images": n_val, "device": str(device),
                      "data_root": str(args.data_root),
                      "scientific_status": "experiment_output_not_operational_wildfire_detection"}
            (args.out / "config.json").write_text(json.dumps(config, indent=2, default=str))
        for epoch in range(args.epochs):
            training_model.train()
            if sampler is not None:
                sampler.set_epoch(epoch)
            sums = torch.zeros(3, device=device, dtype=torch.float64)
            for images, labels in train_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = training_model(images)
                loss = nn.functional.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
                sums[0] += float(loss.item()) * labels.numel()
                sums[1] += (logits.argmax(1) == labels).sum().item()
                sums[2] += labels.numel()
            if distributed:
                dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            if rank == 0:
                # Evaluate raw module on rank 0 (not DDP wrapper); other ranks wait.
                report = evaluate(model, val_loader, device, len(classes))
                report.update({"epoch": epoch + 1,
                               "train_loss": float(sums[0] / sums[2]),
                               "train_accuracy": float(sums[1] / sums[2])})
                with (args.out / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps(report) + "\n")
                torch.save({"model_state_dict": model.state_dict(), "classes": classes,
                            "model": args.model, "epoch": epoch + 1,
                            "optimizer_state_dict": optimizer.state_dict()},
                           args.out / "last.pt")
                print(json.dumps(report), flush=True)
            if distributed:
                dist.barrier()
        if rank == 0:
            print("Research-only run complete. No wildfire detection accuracy is claimed outside this dataset.")
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
