from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.crossbar_snn import CrossbarConfig, CrossbarSNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained crossbar-aware SNN.")
    parser.add_argument("--checkpoint", type=str, default="./artifacts/best_model.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=25)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--weight-levels", type=int, default=32)
    parser.add_argument("--crossbar-rows", type=int, default=128)
    parser.add_argument("--crossbar-cols", type=int, default=128)
    parser.add_argument("--data-root", type=str, default="./artifacts/data")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    cfg = CrossbarConfig(
        hidden_dim=args.hidden_dim,
        num_steps=args.num_steps,
        weight_levels=args.weight_levels,
        crossbar_rows=args.crossbar_rows,
        crossbar_cols=args.crossbar_cols,
    )
    model = CrossbarSNN(cfg).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    test_set = datasets.MNIST(root=args.data_root, train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    correct = 0
    total = 0
    spike_l1 = 0.0
    spike_l2 = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits, stats = model(images)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
            spike_l1 += stats["avg_spikes_l1_per_sample"] * labels.numel()
            spike_l2 += stats["avg_spikes_l2_per_sample"] * labels.numel()

    accuracy = correct / max(total, 1)
    summary = {
        "accuracy": accuracy,
        "avg_spikes_l1_per_sample": spike_l1 / max(total, 1),
        "avg_spikes_l2_per_sample": spike_l2 / max(total, 1),
        **model.crossbar_report(),
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
