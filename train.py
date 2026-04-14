"""
VisionGuard-AD — Training Script
==================================
Unified training pipeline for PatchCore (no gradients) and FastFlow (gradient-based).
Optimized for Apple Silicon M1 with MPS backend.

Usage:
    python train.py --method patchcore --category carpet --backbone wide_resnet50
    python train.py --method fastflow --category carpet --num_epochs 100
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.mvtec_dataset import MVTecDataset, get_dataloader
from data.visa_dataset import VisADataset
from models.backbones.feature_extractor import get_device
from models.patchcore.patchcore import PatchCore
from models.fastflow.fastflow import FastFlow


def setup_logging(output_dir, log_level="INFO"):
    """Configure logging to console and file."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "train.log")

    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="w"),
        ],
    )
    return logging.getLogger("visionguard.train")


def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_patchcore(args, logger):
    """Train PatchCore (gradient-free — build memory bank)."""
    device = get_device(args.device)
    logger.info(f"Training PatchCore on '{args.category}' with device={device}")

    # Load dataset
    DatasetClass = VisADataset if args.dataset == "visa" else MVTecDataset
    train_dataset = DatasetClass(
        root_dir=args.data_root, category=args.category,
        split="train", img_size=args.img_size,
    )
    train_loader = get_dataloader(
        train_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
    )

    stats = train_dataset.get_statistics()
    logger.info(f"Dataset stats: {stats}")

    # Initialize PatchCore
    model = PatchCore(
        backbone=args.backbone,
        coreset_ratio=args.coreset_ratio,
        num_neighbors=args.num_neighbors,
        device=str(device),
        sigma=args.sigma,
    )

    # Fit (build memory bank)
    start = time.time()
    model.fit(train_loader)
    elapsed = time.time() - start
    logger.info(f"PatchCore fit completed in {elapsed:.1f}s")

    # Save
    save_path = os.path.join(args.output_dir, args.category, "patchcore_memory_bank.pt")
    model.save(save_path)

    # Save config
    config = vars(args).copy()
    config["train_time_seconds"] = round(elapsed, 2)
    config["memory_bank_size"] = model.memory_bank.shape[0] if model.memory_bank is not None else 0
    config["device"] = str(device)
    config_path = os.path.join(args.output_dir, args.category, "train_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Training complete. Model saved to {save_path}")


def train_fastflow(args, logger):
    """Train FastFlow (gradient-based normalizing flow)."""
    device = get_device(args.device)
    logger.info(f"Training FastFlow on '{args.category}' with device={device}")

    DatasetClass = VisADataset if args.dataset == "visa" else MVTecDataset
    train_dataset = DatasetClass(
        root_dir=args.data_root, category=args.category,
        split="train", img_size=args.img_size,
    )
    val_dataset = DatasetClass(
        root_dir=args.data_root, category=args.category,
        split="val", img_size=args.img_size,
    )
    train_loader = get_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = get_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Initialize model
    model = FastFlow(
        backbone=args.backbone, flow_steps=args.flow_steps,
        img_size=args.img_size, device=str(device),
    )

    # Optimizer and scheduler (only flow parameters, backbone is frozen)
    flow_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            flow_params.append(param)

    optimizer = torch.optim.Adam(flow_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)

    best_loss = float("inf")
    train_losses = []
    save_dir = os.path.join(args.output_dir, args.category)
    os.makedirs(save_dir, exist_ok=True)

    start = time.time()
    try:
        for epoch in range(1, args.num_epochs + 1):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                images = batch[0]
                optimizer.zero_grad()
                loss, _ = model(images)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(flow_params, max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            train_losses.append(avg_loss)

            # Logging
            lr = scheduler.get_last_lr()[0]
            logger.info(f"Epoch {epoch}/{args.num_epochs} — Loss: {avg_loss:.6f}, LR: {lr:.2e}")

            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                model.save(os.path.join(save_dir, "fastflow_best.pt"))

            # Periodic checkpoint
            if epoch % 10 == 0:
                model.save(os.path.join(save_dir, f"fastflow_epoch{epoch}.pt"))

    except KeyboardInterrupt:
        logger.warning("Training interrupted. Saving checkpoint...")
        model.save(os.path.join(save_dir, "fastflow_interrupted.pt"))

    elapsed = time.time() - start
    logger.info(f"FastFlow training completed in {elapsed:.1f}s, best_loss={best_loss:.6f}")

    # Save training curve
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(train_losses) + 1), train_losses, "b-")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NLL Loss")
    ax.set_title(f"FastFlow Training — {args.category}")
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(save_dir, "training_curve.png"), dpi=150)
    plt.close(fig)

    # Save config
    config = vars(args).copy()
    config["train_time_seconds"] = round(elapsed, 2)
    config["best_loss"] = round(best_loss, 6)
    config["device"] = str(device)
    with open(os.path.join(save_dir, "train_config.json"), "w") as f:
        json.dump(config, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="VisionGuard-AD Training")
    parser.add_argument("--method", type=str, default="patchcore", choices=["patchcore", "fastflow"])
    parser.add_argument("--category", type=str, default="carpet")
    parser.add_argument("--backbone", type=str, default="wide_resnet50")
    parser.add_argument("--dataset", type=str, default="mvtec", choices=["mvtec", "visa"])
    parser.add_argument("--data_root", type=str, default="./data/mvtec")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    # PatchCore args
    parser.add_argument("--coreset_ratio", type=float, default=0.1)
    parser.add_argument("--num_neighbors", type=int, default=9)
    parser.add_argument("--sigma", type=float, default=4.0)
    # FastFlow args
    parser.add_argument("--flow_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=100)
    # Config file (overrides CLI args)
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config file if provided
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        for section in cfg.values():
            if isinstance(section, dict):
                for k, v in section.items():
                    if hasattr(args, k):
                        setattr(args, k, v)

    set_seed(args.seed)
    logger = setup_logging(
        os.path.join(args.output_dir, args.category), "INFO"
    )
    logger.info(f"Arguments: {vars(args)}")

    if args.method == "patchcore":
        train_patchcore(args, logger)
    else:
        train_fastflow(args, logger)


if __name__ == "__main__":
    main()
