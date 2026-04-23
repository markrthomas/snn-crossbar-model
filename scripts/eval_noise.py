"""Evaluate a trained model under Gaussian weight noise.

Simulates RRAM / analog-crossbar device variability by adding N(0, sigma)
noise to the quantised weights at inference time (weights are not re-trained).
Because noise is random, each sigma level is evaluated over multiple trials
and the mean ± std accuracy is reported.

Sigma is expressed in the same units as the quantised weights (range [-1, 1]).
A useful reference point: for N weight levels the quantisation step size is
2 / (N - 1), so sigma = step gives roughly one LSB of noise per weight.

Example
-------
python scripts/eval_noise.py --checkpoint artifacts/best_model.pt \\
    --weight-levels 32 --sigmas 0.0 0.02 0.05 0.1 0.2 0.3 --trials 5
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.crossbar_snn import CrossbarConfig, CrossbarSNN, quantize_ste
from src.train_utils import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model accuracy under weight noise.")
    parser.add_argument("--checkpoint", type=str, default="./artifacts/best_model.pt")
    parser.add_argument("--sigmas", type=float, nargs="+",
                        default=[0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3],
                        help="Noise std-dev values in weight-space [-1, 1]")
    parser.add_argument("--trials", type=int, default=5,
                        help="Independent noise draws per sigma (for mean±std)")
    parser.add_argument("--weight-levels", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=25)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--data-root", type=str, default="./artifacts/data")
    parser.add_argument("--out", type=str, default=None,
                        help="Optional JSON output path (default: stdout only)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@contextmanager
def noisy_weights(
    model: CrossbarSNN, sigma: float, weight_levels: int, seed: int
) -> Generator[None, None, None]:
    """Context manager: temporarily replace fc1/fc2 weights with quantised + noisy versions.

    The QuantLinear forward path re-quantises weights on every call, which would
    snap the noise back to the grid.  We bypass that by setting levels=1 (the
    identity path in _STEQuantize) while the noisy weights are in place.
    """
    torch.manual_seed(seed)

    # Snapshot original state
    orig_w1 = model.fc1.weight.data.clone()
    orig_w2 = model.fc2.weight.data.clone()
    orig_levels_1 = model.fc1.levels
    orig_levels_2 = model.fc2.levels

    # Quantise then add noise
    q_w1 = quantize_ste(orig_w1, weight_levels).detach()
    q_w2 = quantize_ste(orig_w2, weight_levels).detach()
    model.fc1.weight.data = q_w1 + sigma * torch.randn_like(q_w1)
    model.fc2.weight.data = q_w2 + sigma * torch.randn_like(q_w2)

    # Disable re-quantisation during forward (levels <= 1 is identity in _STEQuantize)
    model.fc1.levels = 1
    model.fc2.levels = 1
    try:
        yield
    finally:
        model.fc1.weight.data = orig_w1
        model.fc2.weight.data = orig_w2
        model.fc1.levels = orig_levels_1
        model.fc2.levels = orig_levels_2


def eval_sigma(
    model: CrossbarSNN,
    loader: DataLoader,
    device: torch.device,
    sigma: float,
    weight_levels: int,
    trials: int,
    base_seed: int,
) -> dict:
    accs: List[float] = []
    for trial in range(trials):
        with noisy_weights(model, sigma, weight_levels, seed=base_seed + trial):
            acc = evaluate(model, loader, device)
        accs.append(acc)

    t = torch.tensor(accs)
    mean = float(t.mean())
    std = float(t.std()) if trials > 1 else 0.0
    return {"sigma": sigma, "mean_acc": round(mean, 4), "std_acc": round(std, 4),
            "trials": trials, "per_trial": [round(a, 4) for a in accs]}


def print_table(results: list, step_size: float) -> None:
    print(f"\n{'sigma':>8}  {'sigma/LSB':>9}  {'mean_acc':>8}  {'std_acc':>7}")
    print("-" * 40)
    for r in results:
        lsb_ratio = r["sigma"] / step_size if step_size > 0 else float("nan")
        print(f"{r['sigma']:>8.4f}  {lsb_ratio:>9.2f}  {r['mean_acc']:>8.4f}  {r['std_acc']:>7.4f}")


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
    )
    model = CrossbarSNN(cfg).to(device)
    state_dict = torch.load(ckpt, map_location=device)
    ck_fc1 = state_dict.get("fc1.weight")
    if ck_fc1 is not None and tuple(ck_fc1.shape) != (cfg.hidden_dim, cfg.input_dim):
        raise RuntimeError(
            f"Checkpoint fc1.weight shape {tuple(ck_fc1.shape)} does not match "
            f"model ({cfg.hidden_dim}, {cfg.input_dim}). Wrong --hidden-dim?"
        )
    model.load_state_dict(state_dict)
    model.eval()

    test_set = datasets.MNIST(root=args.data_root, train=False, download=True,
                               transform=transforms.ToTensor())
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    step_size = 2.0 / (args.weight_levels - 1) if args.weight_levels > 1 else 1.0
    print(f"Checkpoint: {ckpt}")
    print(f"weight_levels={args.weight_levels}  LSB step={step_size:.4f}  "
          f"trials={args.trials}  device={device}")

    results = []
    for sigma in args.sigmas:
        r = eval_sigma(model, test_loader, device, sigma, args.weight_levels, args.trials, args.seed)
        results.append(r)
        print(f"  sigma={sigma:.4f}  acc={r['mean_acc']:.4f} ± {r['std_acc']:.4f}")

    print_table(results, step_size)

    summary = {
        "checkpoint": str(ckpt),
        "weight_levels": args.weight_levels,
        "lsb_step": step_size,
        "hidden_dim": args.hidden_dim,
        "num_steps": args.num_steps,
        "trials": args.trials,
        "results": results,
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"\nResults: {out_path}")
    else:
        print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
