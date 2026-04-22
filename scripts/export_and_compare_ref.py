from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys

import torch
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.crossbar_snn import CrossbarConfig, CrossbarSNN, quantize_ste


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export deterministic vectors and compare against C++ reference model.")
    parser.add_argument("--checkpoint", type=str, default="./artifacts/best_model.pt")
    parser.add_argument("--out-dir", type=str, default="./artifacts/ref_vectors")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--weight-levels", type=int, default=32)
    parser.add_argument("--beta", type=float, default=0.95)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--data-root", type=str, default="./artifacts/data")
    parser.add_argument("--cpp-bin", type=str, default="./artifacts/ref_vectors/crossbar_snn_ref")
    parser.add_argument(
        "--expected-mode",
        type=str,
        choices=["snntorch", "discrete"],
        default="snntorch",
        help="Reference behavior for expected logits before C++ comparison.",
    )
    return parser.parse_args()


def save_tensor_flat(path: Path, tensor: torch.Tensor) -> None:
    flat = tensor.reshape(-1).cpu().numpy()
    with path.open("w", encoding="utf-8") as f:
        for v in flat:
            f.write(f"{float(v):.10f}\n")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = CrossbarConfig(
        hidden_dim=args.hidden_dim,
        num_steps=args.num_steps,
        weight_levels=args.weight_levels,
        beta=args.beta,
        threshold=args.threshold,
    )
    model = CrossbarSNN(cfg)
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    ds = datasets.MNIST(root=args.data_root, train=False, download=True, transform=transforms.ToTensor())
    image, label = ds[args.sample_index]
    x = image.view(-1)

    torch.manual_seed(args.seed)
    spikes = torch.stack([torch.bernoulli(torch.clamp(x, 0.0, 1.0)) for _ in range(cfg.num_steps)], dim=0)
    spikes_batched = spikes.unsqueeze(1)

    q_w1 = quantize_ste(model.fc1.weight.detach().cpu(), cfg.weight_levels)
    q_w2 = quantize_ste(model.fc2.weight.detach().cpu(), cfg.weight_levels)
    expected_logits, _ = model.forward_with_spike_sequence(spikes_batched, mode=args.expected_mode)
    expected_logits = expected_logits.squeeze(0).detach().cpu()

    logits_snntorch, _ = model.forward_with_spike_sequence(spikes_batched, mode="snntorch")
    logits_discrete, _ = model.forward_with_spike_sequence(spikes_batched, mode="discrete")
    snntorch_vs_discrete = torch.max(
        torch.abs(logits_snntorch.squeeze(0).detach().cpu() - logits_discrete.squeeze(0).detach().cpu())
    ).item()

    config_path = out_dir / "config.txt"
    w1_path = out_dir / "w1.txt"
    w2_path = out_dir / "w2.txt"
    spikes_path = out_dir / "spikes.txt"
    expected_path = out_dir / "expected_logits.txt"
    output_path = out_dir / "cpp_logits.txt"

    with config_path.open("w", encoding="utf-8") as f:
        f.write(
            f"{cfg.input_dim} {cfg.hidden_dim} {cfg.output_dim} "
            f"{cfg.num_steps} {cfg.beta} {cfg.threshold} {args.expected_mode}\n"
        )

    save_tensor_flat(w1_path, q_w1)
    save_tensor_flat(w2_path, q_w2)
    save_tensor_flat(spikes_path, spikes)
    save_tensor_flat(expected_path, expected_logits)

    cpp_src = Path("ref/cpp/crossbar_snn_ref.cpp")
    cpp_bin = Path(args.cpp_bin)
    cpp_bin.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        ["g++", "-O2", "-std=c++17", str(cpp_src), "-o", str(cpp_bin)],
        check=True,
    )

    result = subprocess.run(
        [
            str(cpp_bin),
            str(config_path),
            str(w1_path),
            str(w2_path),
            str(spikes_path),
            str(expected_path),
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    print(result.stdout.strip())

    cpp_logits = torch.tensor([float(v.strip()) for v in output_path.read_text(encoding="utf-8").splitlines()])
    max_abs_err = torch.max(torch.abs(cpp_logits - expected_logits)).item()
    print(f"sample_index={args.sample_index} label={label}")
    print(f"expected_mode={args.expected_mode}")
    print(f"snntorch_vs_discrete_max_abs_err={snntorch_vs_discrete:.10f}")
    print(f"python_vs_cpp_max_abs_err={max_abs_err:.10f}")
    print("Wrote vectors to:", out_dir)


if __name__ == "__main__":
    main()
