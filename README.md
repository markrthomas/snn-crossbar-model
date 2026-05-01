# snn-crossbar-model

`snntorch`-based spiking neural network targeting hardware-oriented crossbar design.
Includes training, quantisation-aware training (QAT), noise robustness evaluation,
and a four-way fixed-point cross-check: **Python == C++ == SystemC == Verilog RTL**.

## Directory layout

```
src/                  Python model, ASIC spec, shared training utilities
  crossbar_snn.py     CrossbarSNN, QuantLinear (with noise-aware training)
  asic_spec.py        Fixed-point / tiling spec (AsicFixedPointSpec)
  train_utils.py      Shared train_one_epoch() / evaluate()
ref/
  cpp/                C++ fixed-point reference (crossbar_snn_ref_fixed.cpp)
  systemc/            SystemC reference (crossbar_snn_ref_fixed_sc.cpp)
src/snn_core_fixed.v  Synthesisable-ready Verilog RTL core
test/                 SystemVerilog testbench + Makefile
scripts/              Sweep, noise-eval, and cross-check runners
tests/                pytest suite (38 tests)
doc/                  Design spec
artifacts/            Generated at runtime (checkpoints, vectors, sweep results)
```

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

System dependencies for RTL and SystemC cross-checks:

```bash
sudo apt-get install -y iverilog g++ libsystemc-dev
```

---

## Train

```bash
python train.py \
  --epochs 5 \
  --hidden-dim 256 \
  --weight-levels 32 \
  --num-steps 25 \
  --noise-sigma 0.05      # optional: noise-aware training
  --lr-schedule cosine    # default
```

Outputs under `artifacts/`: `best_model.pt`, `history.json`, `crossbar_report.json`.

---

## Evaluate

```bash
python evaluate.py --checkpoint artifacts/best_model.pt --hidden-dim 256
```

Prints JSON including accuracy, average spikes/sample, and crossbar tile metrics.

---

## QAT sweep — accuracy vs hardware cost

Sweeps `weight_levels × num_steps × hidden_dim`, training each configuration
from scratch.  Results are ranked and saved to `artifacts/sweep/sweep_results.json`.

```bash
make sweep                         # 3 epochs, default grid
make sweep SWEEP_EPOCHS=5          # longer run
make sweep SWEEP_NOISE=0.05        # with noise-aware training
```

Override any axis:

```bash
python scripts/sweep_qat.py \
  --weight-levels 4 8 16 32 \
  --num-steps 5 10 25 \
  --hidden-dims 64 128 256 \
  --epochs 5 --save-checkpoints
```

Sample output table (sorted by accuracy):

```
 hidden  wt_lvl  steps  best_acc  tiles  bits/w
------------------------------------------------
    256      32     25    0.9612     11       5
    256      16     25    0.9588     11       4
    128      32     10    0.9541      6       5
     64       8      5    0.8901      3       3
```

`--save-checkpoints` writes `artifacts/sweep/wl{N}_ns{N}_hd{N}.pt` and a
matching `_cfg.json` for each point, ready for immediate noise evaluation.

---

## Noise robustness evaluation

Evaluates a trained checkpoint under Gaussian weight noise to simulate RRAM
device variability.  Reports `mean ± std` accuracy across multiple noise trials.

```bash
make noise NOISE_CKPT=artifacts/best_model.pt NOISE_LEVELS=32 NOISE_STEPS=25
```

Or directly:

```bash
python scripts/eval_noise.py \
  --checkpoint artifacts/sweep/wl32_ns25_hd256.pt \
  --weight-levels 32 --num-steps 25 \
  --sigmas 0.0 0.02 0.05 0.1 0.2 \
  --trials 10 \
  --out artifacts/noise_results.json
```

The `sigma/LSB` column gives a hardware-natural variability budget:

```
   sigma  sigma/LSB  mean_acc  std_acc
----------------------------------------
  0.0000       0.00    0.9612   0.0000
  0.0200       0.29    0.9594   0.0008
  0.0500       0.72    0.9511   0.0024
  0.1000       1.45    0.9187   0.0071
```

### Noise-aware training

Pass `--noise-sigma` to `train.py` or `sweep_qat.py` to inject device noise
during QAT.  Gaussian noise (std = sigma) is added to the quantised weights on
every forward pass during training; eval always uses clean quantised weights.

```bash
python train.py --noise-sigma 0.05 --weight-levels 16
```

---

## Visualisation

```bash
make visualize                        # headless, default paths → artifacts/plots/
make visualize VIZ_HIDDEN=128         # match a checkpoint trained with hidden_dim=128
python scripts/visualize.py --show    # open interactive windows (requires a display)
```

Generates PNG plots in `artifacts/plots/`:

| File | Contents | Requires |
|------|----------|---------|
| `training.png` | Loss and accuracy curves per epoch | `artifacts/history.json` |
| `weights.png` | Raw vs quantised weight histograms for fc1 / fc2 | checkpoint |
| `tiles.png` | Crossbar tile fill-fraction grid for fc1 / fc2 | config only |
| `noise.png` | Accuracy ± std vs noise σ (absolute and in LSB units) | `artifacts/noise_results.json` |
| `sweep.png` | Accuracy heatmap (weight_levels × num_steps) + top-config bar chart | `artifacts/sweep/sweep_results.json` |
| `sweep_curves.png` | Per-config test accuracy vs epoch | sweep results with epoch logs |

The tile layout is always generated from the config alone — no training run needed.
The checkpoint `hidden_dim` is auto-detected, so `--hidden-dim` only affects the tile
layout and is not required to match the checkpoint.

---

## Four-way fixed-point cross-check

Verifies that the Python, C++, SystemC, and Verilog RTL implementations produce
**bit-identical logits** on the same fixed-point input vectors.

```bash
make rtl-check
# or
python scripts/run_rtl_reference_check.py
```

The runner:

1. Exports deterministic fixed-point vectors from Python
2. Runs the Python fixed-point golden (`run_python_fixed`)
3. Compiles and runs `ref/cpp/crossbar_snn_ref_fixed.cpp`
4. Compiles and runs `ref/systemc/crossbar_snn_ref_fixed_sc.cpp` (SystemC 2.3)
5. Compiles `src/snn_core_fixed.v` with iverilog and runs the simulation
6. Asserts: **Python == C++ == SystemC == Verilog RTL**

All four implementations use the same floor-division arithmetic (matching
Python `//` semantics) so negative membrane potentials are handled identically.

### Arithmetic model

| Parameter     | Value        | Notes                                   |
|---------------|--------------|-----------------------------------------|
| Weight format | INT8         | scale=128; `round(q_float * scale)`     |
| Membrane      | INT32        |                                         |
| Decay         | 983/1024     | ≈ 0.96; power-of-two denominator        |
| Threshold     | 128          | tracks weight scale (one LSB unit)      |
| Division      | floor (÷)    | matches Python `//`, not C truncation   |

### SystemC module

`SnnCoreFixed` in `ref/systemc/` has the same `clk / rst_n / start / done / busy`
port interface as `snn_core_fixed.v`, and matches its **clock-level** behavior:
one SNN timestep per clock cycle while running, with a sticky `done` handshake
until `start` is released after completion.

New runs arm on a **rising edge** of `start` while idle.  After sticky `done`
completes, `start` must return low before another run can arm (prevents an
accidental immediate re-run if `start` is tied high through the handshake).

---

## Tests

```bash
python -m pytest tests/ -v
```

42 tests covering:

- `asic_spec` — `AsicFixedPointSpec.beta_float` matches `CrossbarConfig` default;
  per-layer tile counts consistent with `default_asic_bundle()`; utilisation
  denominator uses actual tile count
- `train_utils` — `evaluate()` range/edge cases, `train_one_epoch()` finite
  outputs, weight updates, NaN-loss guard
- `eval_noise` — `noisy_weights` context manager (restore on exit, restore on
  exception, sigma=0 equals quantised, re-quantisation bypassed), `eval_sigma`
  determinism and range
- `noise_aware_training` — `QuantLinear` noise inactive in eval, active in
  train, weights not modified; `set_training_noise`; gradient path; finite
  loss/grad throughout; sweep checkpoint saving and config JSON
- `systemc_ref` — compiles, matches Python golden, bit-identical to C++,
  all-zero spikes, negative weights, summary error=0, bad-arg exits

CI runs pytest then the full four-way RTL cross-check on every push.
`make rtl-check` checks 5 samples by default (indices 0–4); override with
`SAMPLES="0 1 2 3 4 5 6 7 8 9"` for one per digit class.
Pass `--skip-compile` to reuse compiled binaries during iterative development.

---

## Hardware design notes

- `weight_levels` approximates resistive states in RRAM/PCM arrays.
- `num_steps` controls temporal resolution; latency ∝ `num_steps × hidden_dim`.
- `crossbar_rows × crossbar_cols` drives tile count and utilisation estimates.
- Spike statistics from `crossbar_report()` are a proxy for switching activity
  and dynamic energy per inference.
- The SystemC module interface (`clk/rst_n/start/done`) is designed for direct
  integration with a TLM stimulus environment when the RTL moves to a pipelined
  clock-accurate implementation.
