"""Tests for the SystemC fixed-point SNN reference model.

Compiles ref/systemc/crossbar_snn_ref_fixed_sc.cpp, runs it on known vectors,
and verifies its output matches the Python golden and the C++ reference exactly.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers — shared with run_rtl_reference_check.py logic
# ---------------------------------------------------------------------------

def to_hex_signed(v: int, nbytes: int) -> str:
    bits = nbytes * 8
    v = max(-(2 ** (bits - 1)), min(2 ** (bits - 1) - 1, int(v)))
    if v < 0:
        v = (1 << bits) + v
    return f"{v:0{nbytes * 2}x}"


def write_lines(path: Path, lines: list) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def floor_div(a: int, b: int) -> int:
    q, r = divmod(a, b)
    return q - (1 if r != 0 and (a < 0) != (b < 0) else 0)


def run_python_fixed(w1, w2, spikes, beta_num, beta_den, threshold):
    """Pure-Python fixed-point golden — same as run_rtl_reference_check.py."""
    hidden_dim, input_dim = w1.shape
    output_dim = w2.shape[0]
    num_steps = spikes.shape[0]

    mem1 = [0] * hidden_dim
    mem2 = [0] * output_dim
    spk1 = [0] * hidden_dim
    logits = [0] * output_dim

    for t in range(num_steps):
        for h in range(hidden_dim):
            cur1 = sum(int(w1[h, i]) for i in range(input_dim) if spikes[t, i])
            mem_pre = floor_div(beta_num * mem1[h], beta_den) + cur1
            if mem_pre >= threshold:
                spk1[h], mem1[h] = 1, mem_pre - threshold
            else:
                spk1[h], mem1[h] = 0, mem_pre
        for o in range(output_dim):
            cur2 = sum(int(w2[o, h]) for h in range(hidden_dim) if spk1[h])
            mem_pre = floor_div(beta_num * mem2[o], beta_den) + cur2
            if mem_pre >= threshold:
                mem2[o] = mem_pre - threshold
                logits[o] += 1
            else:
                mem2[o] = mem_pre
    return logits


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
SC_SRC = REPO_ROOT / "ref" / "systemc" / "crossbar_snn_ref_fixed_sc.cpp"
CPP_SRC = REPO_ROOT / "ref" / "cpp" / "crossbar_snn_ref_fixed.cpp"


@pytest.fixture(scope="module")
def sc_binary(tmp_path_factory):
    """Compile the SystemC model once per test session."""
    out = tmp_path_factory.mktemp("sc_bin") / "sc_snn_ref"
    result = subprocess.run(
        ["g++", "-O2", "-std=c++17", "-I/usr/include",
         str(SC_SRC), "-lsystemc", "-o", str(out)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"SystemC compile failed:\n{result.stderr}")
    return out


@pytest.fixture(scope="module")
def cpp_binary(tmp_path_factory):
    """Compile the C++ reference model once per test session."""
    out = tmp_path_factory.mktemp("cpp_bin") / "cpp_snn_ref"
    result = subprocess.run(
        ["g++", "-O2", "-std=c++17", str(CPP_SRC), "-o", str(out)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"C++ compile failed:\n{result.stderr}")
    return out


def make_vectors(tmp_dir: Path, hidden_dim=4, input_dim=8, output_dim=3,
                 num_steps=3, beta_num=983, beta_den=1024, threshold=128,
                 seed=0):
    """Write a complete set of test vectors and config to tmp_dir."""
    torch.manual_seed(seed)
    w1 = torch.randint(-128, 128, (hidden_dim, input_dim), dtype=torch.int32)
    w2 = torch.randint(-128, 128, (output_dim, hidden_dim), dtype=torch.int32)
    spikes = torch.randint(0, 2, (num_steps, input_dim), dtype=torch.int32)

    write_lines(tmp_dir / "w1.memh",     [to_hex_signed(v, 1) for v in w1.view(-1).tolist()])
    write_lines(tmp_dir / "w2.memh",     [to_hex_signed(v, 1) for v in w2.view(-1).tolist()])
    write_lines(tmp_dir / "spikes.memh", [str(int(v)) for v in spikes.view(-1).tolist()])

    expected = run_python_fixed(w1.numpy(), w2.numpy(), spikes.numpy(),
                                beta_num, beta_den, threshold)
    write_lines(tmp_dir / "expected_logits.txt", [str(v) for v in expected])
    write_lines(
        tmp_dir / "config_fixed.txt",
        [f"{input_dim} {hidden_dim} {output_dim} {num_steps} {beta_num} {beta_den} {threshold}"],
    )
    return expected


def run_binary(binary: Path, vec_dir: Path, logits_file: str, summary_file: str):
    result = subprocess.run(
        [str(binary),
         str(vec_dir / "config_fixed.txt"),
         str(vec_dir / "w1.memh"),
         str(vec_dir / "w2.memh"),
         str(vec_dir / "spikes.memh"),
         str(vec_dir / "expected_logits.txt"),
         str(vec_dir / logits_file),
         str(vec_dir / summary_file)],
        capture_output=True, text=True,
    )
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_systemc_compiles(sc_binary):
    assert sc_binary.exists()


def test_systemc_matches_python_golden(sc_binary, tmp_path):
    expected = make_vectors(tmp_path)
    result = run_binary(sc_binary, tmp_path, "sc_logits.txt", "sc_summary.txt")
    assert result.returncode == 0, f"SC binary failed:\n{result.stderr}"

    sc_logits = [int(x) for x in (tmp_path / "sc_logits.txt").read_text().split()]
    assert sc_logits == expected, f"SC mismatch: got {sc_logits}, expected {expected}"


def test_systemc_matches_cpp_reference(sc_binary, cpp_binary, tmp_path):
    """SystemC and C++ must produce bit-identical logits."""
    make_vectors(tmp_path, seed=7)

    r_sc  = run_binary(sc_binary,  tmp_path, "sc_logits.txt",  "sc_summary.txt")
    r_cpp = run_binary(cpp_binary, tmp_path, "cpp_logits.txt", "cpp_summary.txt")

    assert r_sc.returncode  == 0, f"SC failed:\n{r_sc.stderr}"
    assert r_cpp.returncode == 0, f"C++ failed:\n{r_cpp.stderr}"

    sc_logits  = [int(x) for x in (tmp_path / "sc_logits.txt").read_text().split()]
    cpp_logits = [int(x) for x in (tmp_path / "cpp_logits.txt").read_text().split()]
    assert sc_logits == cpp_logits, f"SC/C++ diverge: sc={sc_logits} cpp={cpp_logits}"


def test_systemc_zero_spikes(sc_binary, tmp_path):
    """All-zero spike input → all-zero logits."""
    hidden_dim, input_dim, output_dim, num_steps = 4, 8, 3, 3
    w1 = torch.randint(-128, 128, (hidden_dim, input_dim), dtype=torch.int32)
    w2 = torch.randint(-128, 128, (output_dim, hidden_dim), dtype=torch.int32)

    write_lines(tmp_path / "w1.memh",     [to_hex_signed(v, 1) for v in w1.view(-1).tolist()])
    write_lines(tmp_path / "w2.memh",     [to_hex_signed(v, 1) for v in w2.view(-1).tolist()])
    write_lines(tmp_path / "spikes.memh", ["0"] * (num_steps * input_dim))
    write_lines(tmp_path / "expected_logits.txt", ["0"] * output_dim)
    write_lines(tmp_path / "config_fixed.txt",
                [f"{input_dim} {hidden_dim} {output_dim} {num_steps} 983 1024 128"])

    result = run_binary(sc_binary, tmp_path, "sc_logits.txt", "sc_summary.txt")
    assert result.returncode == 0, result.stderr
    logits = [int(x) for x in (tmp_path / "sc_logits.txt").read_text().split()]
    assert logits == [0] * output_dim


def test_systemc_negative_weights(sc_binary, tmp_path):
    """Negative weights should be handled correctly (sign-extend from INT8)."""
    make_vectors(tmp_path, seed=99,
                 hidden_dim=4, input_dim=6, output_dim=2, num_steps=4)
    result = run_binary(sc_binary, tmp_path, "sc_logits.txt", "sc_summary.txt")
    assert result.returncode == 0, result.stderr

    expected = [int(x) for x in (tmp_path / "expected_logits.txt").read_text().split()]
    sc_logits = [int(x) for x in (tmp_path / "sc_logits.txt").read_text().split()]
    assert sc_logits == expected


def test_systemc_summary_zero_error(sc_binary, tmp_path):
    """Summary file must report max_abs_err=0 when logits match expected."""
    make_vectors(tmp_path, seed=5)
    result = run_binary(sc_binary, tmp_path, "sc_logits.txt", "sc_summary.txt")
    assert result.returncode == 0, result.stderr
    summary = (tmp_path / "sc_summary.txt").read_text()
    assert "max_abs_err=0" in summary


def test_systemc_missing_config_exits_nonzero(sc_binary, tmp_path):
    result = subprocess.run(
        [str(sc_binary), str(tmp_path / "no_such_config.txt"),
         "w1.memh", "w2.memh", "spikes.memh", "exp.txt", "out.txt", "sum.txt"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0


def test_systemc_wrong_arg_count_exits_2(sc_binary):
    result = subprocess.run([str(sc_binary)], capture_output=True, text=True)
    assert result.returncode == 2
