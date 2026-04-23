from __future__ import annotations

import argparse
import json
import shutil
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
    parser = argparse.ArgumentParser(
        description="Cross-check Python, C++, SystemC, and Verilog fixed-point SNN references."
    )
    parser.add_argument("--checkpoint", type=str, default="./artifacts/best_model.pt")
    parser.add_argument("--out-dir", type=str, default="./artifacts/ref_vectors_fixed")
    parser.add_argument(
        "--samples",
        type=int,
        nargs="+",
        default=[0],
        metavar="IDX",
        help="MNIST test-set sample indices to check (default: [0]; use 0..9 for thorough check)",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--weight-levels", type=int, default=32)
    fp = AsicFixedPointSpec()
    parser.add_argument("--beta-num", type=int, default=fp.beta_num)
    parser.add_argument("--beta-den", type=int, default=fp.beta_den)
    parser.add_argument(
        "--scale",
        type=int,
        default=fp.weight_scale,
        help="INT8 weight scale (LSB size); threshold tracks this scale.",
    )
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


def compile_cpp(out_dir: Path) -> Path:
    """Compile C++ reference once; return binary path."""
    cpp_bin = out_dir / "crossbar_snn_ref_fixed"
    run_checked(
        ["g++", "-O2", "-std=c++17", "ref/cpp/crossbar_snn_ref_fixed.cpp", "-o", str(cpp_bin)]
    )
    print(f"  compiled C++  -> {cpp_bin}")
    return cpp_bin


def compile_sc(out_dir: Path) -> Path:
    """Compile SystemC reference once; return binary path."""
    sc_bin = out_dir / "crossbar_snn_ref_fixed_sc"
    run_checked(
        [
            "g++", "-O2", "-std=c++17", "-I/usr/include",
            "ref/systemc/crossbar_snn_ref_fixed_sc.cpp",
            "-lsystemc", "-o", str(sc_bin),
        ]
    )
    print(f"  compiled SC   -> {sc_bin}")
    return sc_bin


def compile_rtl(out_dir: Path, cfg: CrossbarConfig, beta_num: int, beta_den: int, threshold: int) -> Path:
    """Compile Verilog simulation once; return binary path."""
    sim_bin = out_dir / "sim_fixed"
    run_checked(
        [
            "iverilog",
            "-g2012",
            f"-DINPUT_DIM={cfg.input_dim}",
            f"-DHIDDEN_DIM={cfg.hidden_dim}",
            f"-DOUTPUT_DIM={cfg.output_dim}",
            f"-DNUM_STEPS={cfg.num_steps}",
            f"-DBETA_NUM={beta_num}",
            f"-DBETA_DEN={beta_den}",
            f"-DTHRESHOLD={threshold}",
            "-o", str(sim_bin),
            "test/tb_snn_core_fixed.sv",
            "src/snn_core_fixed.v",
        ]
    )
    print(f"  compiled RTL  -> {sim_bin}")
    return sim_bin


def check_sample(
    sample_dir: Path,
    sample_index: int,
    label: int,
    w1_i: torch.Tensor,
    w2_i: torch.Tensor,
    spikes: torch.Tensor,
    beta_num: int,
    beta_den: int,
    threshold: int,
    fp_spec: AsicFixedPointSpec,
    cpp_bin: Path,
    sc_bin: Path,
    sim_bin: Path,
) -> dict:
    """Write vectors for one sample, run all four implementations, assert equality."""
    sample_dir.mkdir(parents=True, exist_ok=True)

    py_logits = run_python_fixed(w1_i, w2_i, spikes, beta_num, beta_den, threshold)
    expected = [int(v) for v in py_logits.tolist()]

    nbytes = fp_spec.weight_bits // 8
    write_lines(sample_dir / "w1.memh", [to_hex_signed(v, nbytes) for v in w1_i.view(-1).tolist()])
    write_lines(sample_dir / "w2.memh", [to_hex_signed(v, nbytes) for v in w2_i.view(-1).tolist()])
    write_lines(sample_dir / "spikes.memh", [f"{int(v)}" for v in spikes.view(-1).tolist()])
    write_lines(sample_dir / "expected_logits.txt", [str(v) for v in expected])

    cfg_vals = w1_i.shape  # (hidden, input)
    input_dim = cfg_vals[1]
    hidden_dim = cfg_vals[0]
    output_dim = w2_i.shape[0]
    num_steps = spikes.shape[0]
    write_lines(
        sample_dir / "config_fixed.txt",
        [f"{input_dim} {hidden_dim} {output_dim} {num_steps} {beta_num} {beta_den} {threshold}"],
    )

    def run_ref(binary: Path, logits_file: str, summary_file: str) -> List[int]:
        run_checked(
            [
                str(binary),
                str(sample_dir / "config_fixed.txt"),
                str(sample_dir / "w1.memh"),
                str(sample_dir / "w2.memh"),
                str(sample_dir / "spikes.memh"),
                str(sample_dir / "expected_logits.txt"),
                str(sample_dir / logits_file),
                str(sample_dir / summary_file),
            ]
        )
        return read_int_vector(sample_dir / logits_file)

    cpp_logits = run_ref(cpp_bin, "cpp_logits.txt", "cpp_summary.txt")
    sc_logits  = run_ref(sc_bin,  "sc_logits.txt",  "sc_summary.txt")

    # RTL sim reads/writes from the fixed path the testbench was compiled with.
    # Swap in this sample's vectors, run, then copy output back.
    fixed_dir = Path("artifacts/ref_vectors_fixed")
    fixed_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("w1.memh", "w2.memh", "spikes.memh", "expected_logits.txt", "config_fixed.txt"):
        shutil.copy2(sample_dir / fname, fixed_dir / fname)
    run_checked([str(sim_bin)])
    rtl_output = fixed_dir / "verilog_logits.txt"
    shutil.copy2(rtl_output, sample_dir / "verilog_logits.txt")
    rtl_logits = read_int_vector(sample_dir / "verilog_logits.txt")

    errors = {}
    if cpp_logits != expected:
        errors["cpp"] = {"expected": expected, "got": cpp_logits}
    if sc_logits != expected:
        errors["sc"] = {"expected": expected, "got": sc_logits}
    if rtl_logits != expected:
        errors["rtl"] = {"expected": expected, "got": rtl_logits}

    return {
        "sample_index": sample_index,
        "label": label,
        "logits": expected,
        "errors": errors,
        "pass": len(errors) == 0,
    }


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

    fp_spec = AsicFixedPointSpec()
    beta_num  = args.beta_num
    beta_den  = args.beta_den
    threshold = args.scale

    q_w1 = quantize_ste(model.fc1.weight.detach().cpu(), cfg.weight_levels)
    q_w2 = quantize_ste(model.fc2.weight.detach().cpu(), cfg.weight_levels)
    wmin = -(2 ** (fp_spec.weight_bits - 1))
    wmax = 2 ** (fp_spec.weight_bits - 1) - 1
    w1_i = torch.round(q_w1 * args.scale).to(torch.int32).clamp(wmin, wmax)
    w2_i = torch.round(q_w2 * args.scale).to(torch.int32).clamp(wmin, wmax)

    ds = datasets.MNIST(
        root=args.data_root, train=False, download=True, transform=transforms.ToTensor()
    )

    # Compile all three non-Python references once upfront.
    print("Compiling references...")
    cpp_bin = compile_cpp(out_dir)
    sc_bin  = compile_sc(out_dir)
    rtl_bin = compile_rtl(out_dir, cfg, beta_num, beta_den, threshold)

    results = []
    failures = []

    print(f"\nChecking {len(args.samples)} sample(s): {args.samples}")
    for sample_index in args.samples:
        image, label = ds[sample_index]
        x = image.view(-1)

        torch.manual_seed(args.seed + sample_index)
        spikes = torch.stack(
            [torch.bernoulli(torch.clamp(x, 0.0, 1.0)) for _ in range(cfg.num_steps)], dim=0
        ).to(torch.int32)

        sample_dir = out_dir / f"sample_{sample_index:04d}"
        print(f"  sample {sample_index:4d}  label={label}", end="  ", flush=True)

        result = check_sample(
            sample_dir=sample_dir,
            sample_index=sample_index,
            label=label,
            w1_i=w1_i,
            w2_i=w2_i,
            spikes=spikes,
            beta_num=beta_num,
            beta_den=beta_den,
            threshold=threshold,
            fp_spec=fp_spec,
            cpp_bin=cpp_bin,
            sc_bin=sc_bin,
            sim_bin=rtl_bin,
        )
        results.append(result)
        status = "PASS" if result["pass"] else "FAIL"
        print(f"logits={result['logits']}  {status}")
        if not result["pass"]:
            failures.append(result)

    # --- Write ASIC bundle for last sample (representative metadata) ---
    bundle = default_asic_bundle(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        output_dim=cfg.output_dim,
        crossbar_rows=args.crossbar_rows,
        crossbar_cols=args.crossbar_cols,
        weight_scale=args.scale,
    )
    bundle["runtime"] = {
        "checkpoint": str(checkpoint_path.resolve()),
        "samples": args.samples,
        "seed": args.seed,
        "num_steps": cfg.num_steps,
        "hidden_dim": cfg.hidden_dim,
        "weight_levels": cfg.weight_levels,
    }
    (out_dir / "asic_spec.json").write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")

    # --- Summary ---
    passed = sum(1 for r in results if r["pass"])
    total  = len(results)
    print(f"\nSummary: {passed}/{total} samples passed.")

    summary_path = out_dir / "cross_check_summary.json"
    summary_path.write_text(
        json.dumps({"passed": passed, "total": total, "results": results}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Summary written to {summary_path}")

    if failures:
        msgs = []
        for r in failures:
            msgs.append(f"  sample {r['sample_index']} (label={r['label']}): {r['errors']}")
        raise SystemExit("FAIL — mismatches in:\n" + "\n".join(msgs))

    print(f"PASS: Python == C++ == SystemC == Verilog RTL ({total} sample(s))")


if __name__ == "__main__":
    main()
