from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.crossbar_snn import CrossbarConfig, CrossbarSNN
from src.train_utils import evaluate, train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an snnTorch model for crossbar-aware SNN design.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-steps", type=int, default=25)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--weight-levels", type=int, default=32)
    parser.add_argument("--crossbar-rows", type=int, default=128)
    parser.add_argument("--crossbar-cols", type=int, default=128)
    parser.add_argument("--data-root", type=str, default="./artifacts/data")
    parser.add_argument("--out-dir", type=str, default="./artifacts")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--noise-sigma", type=float, default=0.0,
                        help="Std-dev of Gaussian noise injected into quantised weights during "
                             "training (0 = standard QAT, >0 = noise-aware training)")
    parser.add_argument("--lr-schedule", choices=["cosine", "none"], default="cosine",
                        help="LR schedule (cosine annealing recommended for quantised models)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(root=args.data_root, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=args.data_root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    cfg = CrossbarConfig(
        hidden_dim=args.hidden_dim,
        num_steps=args.num_steps,
        weight_levels=args.weight_levels,
        crossbar_rows=args.crossbar_rows,
        crossbar_cols=args.crossbar_cols,
    )
    model = CrossbarSNN(cfg).to(device)
    if args.noise_sigma > 0.0:
        model.set_training_noise(args.noise_sigma)
        print(f"Noise-aware training: sigma={args.noise_sigma}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        if args.lr_schedule == "cosine" else None
    )

    history = {"train_loss": [], "train_acc": [], "test_acc": []}
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        test_acc = evaluate(model, test_loader, device)
        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"loss={train_loss:.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), out_dir / "best_model.pt")

    report = model.crossbar_report()
    report["best_test_acc"] = best_acc
    report["device"] = str(device)
    report["epochs"] = args.epochs
    report["batch_size"] = args.batch_size
    report["hidden_dim"] = args.hidden_dim

    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with (out_dir / "crossbar_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Saved:", out_dir / "best_model.pt")
    print("Saved:", out_dir / "history.json")
    print("Saved:", out_dir / "crossbar_report.json")


if __name__ == "__main__":
    main()
