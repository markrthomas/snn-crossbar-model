from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import List
import sys

import torch
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.asic_spec import AsicFixedPointSpec, default_asic_bundle
from src.crossbar_snn import CrossbarConfig, CrossbarSNN, quantize_ste


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-check Python, C++, and Verilog fixed-point SNN references.")
    parser.add_argument("--checkpoint", type=str, default="./artifacts/best_model.pt")
    parser.add_argument("--out-dir", type=str, default="./artifacts/ref_vectors_fixed")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--weight-levels", type=int, default=32)
    fp = AsicFixedPointSpec()
    parser.add_argument("--beta-num", type=int, default=fp.beta_num)
    parser.add_argument("--beta-den", type=int, default=fp.beta_den)
    parser.add_argument("--scale", type=int, default=fp.weight_scale, help="INT8 weight scale (LSB size); threshold tracks this scale.")
    parser.add_argument("--crossbar-rows", type=int, default=128)
    parser.add_argument("--crossbar-cols", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data-root", type=str, default="./artifacts/data")
    return parser.parse_args()


def to_hex_signed(v: int, nbytes: int) -> str:
    bits = nbytes * 8
    lo = -(2 ** (bits - 1))
    hi = 2 ** (bits - 1) - 1
    v = max(lo, min(hi, int(v)))
    if v < 0:
        v = (1 << bits) + v
    width = nbytes * 2
    return f"{v:0{width}x}"


def write_lines(path: Path, lines: List[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


def run_python_fixed(
    w1_i: torch.Tensor,
    w2_i: torch.Tensor,
    spikes: torch.Tensor,
    beta_num: int,
    beta_den: int,
    threshold: int,
) -> torch.Tensor:
    hidden_dim = w1_i.shape[0]
    output_dim = w2_i.shape[0]
    num_steps = spikes.shape[0]
    input_dim = spikes.shape[1]

    mem1 = torch.zeros(hidden_dim, dtype=torch.int32)
    mem2 = torch.zeros(output_dim, dtype=torch.int32)
    logits = torch.zeros(output_dim, dtype=torch.int32)
    spk1 = torch.zeros(hidden_dim, dtype=torch.int32)

    for t in range(num_steps):
        for h in range(hidden_dim):
            cur1 = 0
            base = h * input_dim
            for i in range(input_dim):
                if spikes[t, i] != 0:
                    cur1 += int(w1_i.view(-1)[base + i].item())
            mem_pre = (beta_num * int(mem1[h].item())) // beta_den + cur1
            if mem_pre >= threshold:
                spk1[h] = 1
                mem1[h] = mem_pre - threshold
            else:
                spk1[h] = 0
                mem1[h] = mem_pre

        for o in range(output_dim):
            cur2 = 0
            base = o * hidden_dim
            for h in range(hidden_dim):
                if spk1[h] != 0:
                    cur2 += int(w2_i.view(-1)[base + h].item())
            mem_pre = (beta_num * int(mem2[o].item())) // beta_den + cur2
            if mem_pre >= threshold:
                mem2[o] = mem_pre - threshold
                logits[o] += 1
            else:
                mem2[o] = mem_pre

    return logits


def run_checked(cmd: List[str]) -> None:
    """Run a subprocess, raising RuntimeError with stderr on non-zero exit."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}): {' '.join(str(c) for c in cmd)}\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    if result.stdout.strip():
        print(result.stdout.strip())


def read_int_vector(path: Path) -> List[int]:
    return [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = CrossbarConfig(
        hidden_dim=args.hidden_dim,
        num_steps=args.num_steps,
        weight_levels=args.weight_levels,
    )
    model = CrossbarSNN(cfg).to(args.device)
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    else:
        print(f"No checkpoint found at {args.checkpoint}; using random weights for cross-check.")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
    model.eval()

    ds = datasets.MNIST(root=args.data_root, train=False, download=True, transform=transforms.ToTensor())
    image, label = ds[args.sample_index]
    x = image.view(-1)

    torch.manual_seed(args.seed)
    spikes = torch.stack([torch.bernoulli(torch.clamp(x, 0.0, 1.0)) for _ in range(cfg.num_steps)], dim=0).to(torch.int32)

    q_w1 = quantize_ste(model.fc1.weight.detach().cpu(), cfg.weight_levels)
    q_w2 = quantize_ste(model.fc2.weight.detach().cpu(), cfg.weight_levels)
    fp_spec = AsicFixedPointSpec()
    wmin = -(2 ** (fp_spec.weight_bits - 1))
    wmax = 2 ** (fp_spec.weight_bits - 1) - 1
    w1_i = torch.round(q_w1 * args.scale).to(torch.int32).clamp(wmin, wmax)
    w2_i = torch.round(q_w2 * args.scale).to(torch.int32).clamp(wmin, wmax)
    threshold = args.scale

    py_logits = run_python_fixed(w1_i, w2_i, spikes, args.beta_num, args.beta_den, threshold)

    nbytes = fp_spec.weight_bits // 8
    write_lines(out_dir / "w1.memh", [to_hex_signed(v, nbytes) for v in w1_i.view(-1).tolist()])
    write_lines(out_dir / "w2.memh", [to_hex_signed(v, nbytes) for v in w2_i.view(-1).tolist()])
    write_lines(out_dir / "spikes.memh", [f"{int(v)}" for v in spikes.view(-1).tolist()])
    write_lines(out_dir / "expected_logits.txt", [str(int(v)) for v in py_logits.tolist()])
    write_lines(
        out_dir / "config_fixed.txt",
        [f"{cfg.input_dim} {cfg.hidden_dim} {cfg.output_dim} {cfg.num_steps} {args.beta_num} {args.beta_den} {threshold}"],
    )

    bundle = default_asic_bundle(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        output_dim=cfg.output_dim,
        crossbar_rows=args.crossbar_rows,
        crossbar_cols=args.crossbar_cols,
        weight_scale=args.scale,
    )
    bundle["runtime"] = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "sample_index": args.sample_index,
        "label": int(label),
        "seed": args.seed,
        "num_steps": cfg.num_steps,
        "hidden_dim": cfg.hidden_dim,
        "weight_levels": cfg.weight_levels,
    }
    (out_dir / "asic_spec.json").write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")

    cpp_bin = out_dir / "crossbar_snn_ref_fixed"
    run_checked(["g++", "-O2", "-std=c++17", "ref/cpp/crossbar_snn_ref_fixed.cpp", "-o", str(cpp_bin)])
    run_checked(
        [
            str(cpp_bin),
            str(out_dir / "config_fixed.txt"),
            str(out_dir / "w1.memh"),
            str(out_dir / "w2.memh"),
            str(out_dir / "spikes.memh"),
            str(out_dir / "expected_logits.txt"),
            str(out_dir / "cpp_logits.txt"),
            str(out_dir / "cpp_summary.txt"),
        ]
    )

    run_checked(
        [
            "iverilog",
            "-g2012",
            f"-DINPUT_DIM={cfg.input_dim}",
            f"-DHIDDEN_DIM={cfg.hidden_dim}",
            f"-DOUTPUT_DIM={cfg.output_dim}",
            f"-DNUM_STEPS={cfg.num_steps}",
            f"-DBETA_NUM={args.beta_num}",
            f"-DBETA_DEN={args.beta_den}",
            f"-DTHRESHOLD={threshold}",
            "-o",
            str(out_dir / "sim_fixed"),
            "test/tb_snn_core_fixed.sv",
            "src/snn_core_fixed.v",
        ]
    )
    run_checked([str(out_dir / "sim_fixed")])

    cpp_logits = read_int_vector(out_dir / "cpp_logits.txt")
    rtl_logits = read_int_vector(out_dir / "verilog_logits.txt")
    expected = [int(v) for v in py_logits.tolist()]

    if cpp_logits != expected:
        raise RuntimeError(f"C++ mismatch.\nexpected={expected}\ncpp={cpp_logits}")
    if rtl_logits != expected:
        raise RuntimeError(f"Verilog mismatch.\nexpected={expected}\nrtl={rtl_logits}")

    print("PASS: Python fixed model == C++ fixed ref == Verilog RTL output")
    print(f"sample_index={args.sample_index} label={label}")
    print("logits:", expected)
    print("vectors:", out_dir)


if __name__ == "__main__":
    main()
