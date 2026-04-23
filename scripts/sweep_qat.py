"""Sweep weight_levels × num_steps to characterise the accuracy / hardware tradeoff.

Each combination is trained from scratch with QAT (the STE quantisation runs
during every forward pass).  Results are written to a JSON file and printed as
a human-readable table so the hardware team can pick the minimum-cost point
before accuracy degrades.

Example
-------
# Quick exploration (2 epochs):
python scripts/sweep_qat.py --epochs 2

# More accurate results (5 epochs):
python scripts/sweep_qat.py --epochs 5 --weight-levels 4 8 16 32 --num-steps 5 10 25
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.crossbar_snn import CrossbarConfig, CrossbarSNN
from src.train_utils import evaluate, train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep QAT configs over weight_levels × num_steps.")
    parser.add_argument("--weight-levels", type=int, nargs="+", default=[4, 8, 16, 32],
                        help="Quantisation levels to sweep (conductance states in hardware)")
    parser.add_argument("--num-steps", type=int, nargs="+", default=[5, 10, 25],
                        help="Simulation timesteps to sweep (latency proxy)")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3,
                        help="Training epochs per config (2-3 for exploration, 5+ for final)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-root", type=str, default="./artifacts/data")
    parser.add_argument("--out-dir", type=str, default="./artifacts/sweep")
    parser.add_argument("--crossbar-rows", type=int, default=128)
    parser.add_argument("--crossbar-cols", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_config(
    cfg: CrossbarConfig,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> dict:
    """Train one config from scratch and return result dict."""
    model = CrossbarSNN(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0.0
    epoch_log = []
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        test_acc = evaluate(model, test_loader, device)
        best_acc = max(best_acc, test_acc)
        epoch_log.append({"epoch": epoch, "train_loss": round(train_loss, 5),
                           "train_acc": round(train_acc, 4), "test_acc": round(test_acc, 4)})
        print(f"  epoch {epoch}/{epochs}  loss={train_loss:.4f}  "
              f"train={train_acc:.4f}  test={test_acc:.4f}")

    report = model.crossbar_report()
    return {
        "weight_levels": cfg.weight_levels,
        "num_steps": cfg.num_steps,
        "hidden_dim": cfg.hidden_dim,
        "best_test_acc": round(best_acc, 4),
        "final_test_acc": round(epoch_log[-1]["test_acc"], 4),
        "tile_count": report["tile_count"],
        "tile_utilization": round(report["tile_utilization"], 4),
        "bits_per_weight": (cfg.weight_levels - 1).bit_length(),
        "epochs": epoch_log,
    }


def print_table(results: list) -> None:
    header = f"{'wt_levels':>9}  {'num_steps':>9}  {'best_acc':>8}  {'tiles':>5}  {'bits/w':>6}"
    print("\n" + header)
    print("-" * len(header))
    for r in sorted(results, key=lambda x: (-x["best_test_acc"], x["tile_count"])):
        print(f"{r['weight_levels']:>9}  {r['num_steps']:>9}  "
              f"{r['best_test_acc']:>8.4f}  {r['tile_count']:>5}  "
              f"{r['bits_per_weight']:>6}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(root=args.data_root, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=args.data_root, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    configs = list(product(args.weight_levels, args.num_steps))
    print(f"Sweeping {len(configs)} configs × {args.epochs} epochs "
          f"on {device}  (hidden_dim={args.hidden_dim})")

    results = []
    for idx, (wl, ns) in enumerate(configs, 1):
        cfg = CrossbarConfig(
            hidden_dim=args.hidden_dim,
            num_steps=ns,
            weight_levels=wl,
            crossbar_rows=args.crossbar_rows,
            crossbar_cols=args.crossbar_cols,
        )
        print(f"\n[{idx}/{len(configs)}] weight_levels={wl}  num_steps={ns}")
        result = run_config(cfg, train_loader, test_loader, device, args.epochs, args.lr)
        results.append(result)

        # Save incrementally so a partial run is still useful
        out_path = out_dir / "sweep_results.json"
        out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print_table(results)
    print(f"\nFull results: {out_dir / 'sweep_results.json'}")


if __name__ == "__main__":
    main()
