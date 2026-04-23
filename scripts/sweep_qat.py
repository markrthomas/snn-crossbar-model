"""Sweep weight_levels × num_steps × hidden_dim to characterise the accuracy /
hardware tradeoff.

Each combination is trained from scratch with QAT (the STE quantisation runs
during every forward pass).  Results are written to a JSON file and printed as
a human-readable table so the hardware team can pick the minimum-cost point
before accuracy degrades.

Example
-------
# Quick exploration (2 epochs, default grid):
python scripts/sweep_qat.py --epochs 2

# Full sweep with noise-aware training and checkpoint saving:
python scripts/sweep_qat.py --epochs 5 \\
    --weight-levels 4 8 16 32 --num-steps 5 10 25 --hidden-dims 64 128 256 \\
    --noise-sigma 0.05 --save-checkpoints
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.crossbar_snn import CrossbarConfig, CrossbarSNN
from src.train_utils import evaluate, train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep QAT configs over weight_levels × num_steps × hidden_dim."
    )
    parser.add_argument("--weight-levels", type=int, nargs="+", default=[4, 8, 16, 32],
                        help="Quantisation levels to sweep (conductance states in hardware)")
    parser.add_argument("--num-steps", type=int, nargs="+", default=[5, 10, 25],
                        help="Simulation timesteps to sweep (latency proxy)")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 256],
                        help="Hidden layer widths to sweep (tile-count proxy)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Training epochs per config (2-3 for exploration, 5+ for final)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-schedule", choices=["cosine", "none"], default="cosine")
    parser.add_argument("--noise-sigma", type=float, default=0.0,
                        help="Training noise std-dev (0 = standard QAT)")
    parser.add_argument("--data-root", type=str, default="./artifacts/data")
    parser.add_argument("--out-dir", type=str, default="./artifacts/sweep")
    parser.add_argument("--crossbar-rows", type=int, default=128)
    parser.add_argument("--crossbar-cols", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-checkpoints", action="store_true",
                        help="Save best checkpoint + config JSON for each sweep point")
    return parser.parse_args()


def run_config(
    cfg: CrossbarConfig,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    lr_schedule: str,
    noise_sigma: float,
    save_dir: Optional[Path],
) -> dict:
    """Train one config from scratch and return result dict."""
    model = CrossbarSNN(cfg).to(device)
    if noise_sigma > 0.0:
        model.set_training_noise(noise_sigma)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        if lr_schedule == "cosine" else None
    )

    best_acc = 0.0
    best_state: Optional[dict] = None
    epoch_log = []
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        test_acc = evaluate(model, test_loader, device)
        if scheduler is not None:
            scheduler.step()
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        epoch_log.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_acc": round(train_acc, 4),
            "test_acc": round(test_acc, 4),
        })
        print(f"  epoch {epoch}/{epochs}  loss={train_loss:.4f}  "
              f"train={train_acc:.4f}  test={test_acc:.4f}")

    if save_dir is not None and best_state is not None:
        tag = f"wl{cfg.weight_levels}_ns{cfg.num_steps}_hd{cfg.hidden_dim}"
        ckpt_path = save_dir / f"{tag}.pt"
        cfg_path = save_dir / f"{tag}_cfg.json"
        torch.save(best_state, ckpt_path)
        cfg_path.write_text(
            json.dumps({
                "weight_levels": cfg.weight_levels,
                "num_steps": cfg.num_steps,
                "hidden_dim": cfg.hidden_dim,
                "noise_sigma_training": noise_sigma,
            }, indent=2) + "\n",
            encoding="utf-8",
        )

    report = model.crossbar_report()
    return {
        "weight_levels": cfg.weight_levels,
        "num_steps": cfg.num_steps,
        "hidden_dim": cfg.hidden_dim,
        "noise_sigma_training": noise_sigma,
        "best_test_acc": round(best_acc, 4),
        "final_test_acc": round(epoch_log[-1]["test_acc"], 4),
        "tile_count": report["tile_count"],
        "tile_utilization": round(report["tile_utilization"], 4),
        "bits_per_weight": (cfg.weight_levels - 1).bit_length(),
        "epochs": epoch_log,
    }


def print_table(results: list) -> None:
    header = (f"{'hidden':>7}  {'wt_lvl':>7}  {'steps':>6}  "
              f"{'best_acc':>8}  {'tiles':>5}  {'bits/w':>6}")
    print("\n" + header)
    print("-" * len(header))
    for r in sorted(results, key=lambda x: (-x["best_test_acc"], x["tile_count"])):
        print(f"{r['hidden_dim']:>7}  {r['weight_levels']:>7}  {r['num_steps']:>6}  "
              f"{r['best_test_acc']:>8.4f}  {r['tile_count']:>5}  "
              f"{r['bits_per_weight']:>6}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_dir = out_dir if args.save_checkpoints else None

    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(root=args.data_root, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=args.data_root, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    configs = list(product(args.hidden_dims, args.weight_levels, args.num_steps))
    print(f"Sweeping {len(configs)} configs × {args.epochs} epochs on {device}")
    if args.noise_sigma > 0:
        print(f"Noise-aware training: sigma={args.noise_sigma}")
    if args.save_checkpoints:
        print(f"Saving checkpoints to: {out_dir}")

    results = []
    for idx, (hd, wl, ns) in enumerate(configs, 1):
        cfg = CrossbarConfig(
            hidden_dim=hd,
            num_steps=ns,
            weight_levels=wl,
            crossbar_rows=args.crossbar_rows,
            crossbar_cols=args.crossbar_cols,
        )
        print(f"\n[{idx}/{len(configs)}] hidden_dim={hd}  weight_levels={wl}  num_steps={ns}")
        result = run_config(
            cfg, train_loader, test_loader, device,
            args.epochs, args.lr, args.lr_schedule, args.noise_sigma, save_dir,
        )
        results.append(result)

        # Save incrementally so a partial run is still useful.
        out_path = out_dir / "sweep_results.json"
        out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print_table(results)
    print(f"\nFull results: {out_dir / 'sweep_results.json'}")


if __name__ == "__main__":
    main()
