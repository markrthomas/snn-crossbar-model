# SNN Fixed-Point RTL Reference Design

This document defines the Verilog reference implementation used for cross-language checking against Python and C++.

## Scope

- One-sample, fixed-point inference for a 2-layer SNN:
  - Input spikes -> FC1 -> LIF -> FC2 -> LIF -> output spike counts (`logits`)
- Uses the same deterministic spike vectors exported by Python.
- Intended as a verification-oriented RTL baseline for future crossbar accelerator work.

## Arithmetic model

- Weight format: signed **INT8** (`w1.memh`, `w2.memh`, one byte per token)
- Spike format: binary (`spikes.memh`)
- Membrane format: signed 32-bit integers
- Decay: integer rational
  - `mem_pre = (BETA_NUM * mem) / BETA_DEN + current`
- Threshold/reset (discrete mode):
  - if `mem_pre >= THRESHOLD`: spike = 1, `mem = mem_pre - THRESHOLD`
  - else: spike = 0, `mem = mem_pre`

Default constants (ASIC-first defaults):

- `WEIGHT_SCALE=128` (float weight -> int8: `round(q_float * scale)` then clamp)
- `THRESHOLD = WEIGHT_SCALE` (one membrane unit == one weight LSB)
- `BETA_NUM=983`, `BETA_DEN=1024` (~0.96; denominator is a power-of-two)

## Generated artifacts

`scripts/run_rtl_reference_check.py` writes `artifacts/ref_vectors_fixed/asic_spec.json`, which includes:

- the fixed-point contract
- a **128x128 crossbar tile** memory map for both weight matrices (logical byte addresses)

## RTL files

- `src/snn_core_fixed.v`:
  - ports: `clk`, `rst_n`, `start`, `done`, `busy` (`busy` high whenever not idle)
  - core fixed-point SNN compute block; no file I/O (memories initialised by the testbench)
  - computes logits across multiple clock cycles (one SNN timestep per cycle)
    after a rising-edge qualified `start` while idle, with a sticky `done`
    handshake until `start` is released; after that handshake, `start` must
    return low before another run can arm
- `test/tb_snn_core_fixed.sv`:
  - loads `w1`, `w2`, and `spikes` memories via `$readmemh` from the directory
    passed as `+data_dir=<absolute-path>` (falls back to
    `artifacts/ref_vectors_fixed` if the plusarg is absent)
  - drives reset/start; waits for `done`
  - writes `verilog_logits.txt` into `data_dir` after scenario 1 completes,
    before any later protocol-verification scenarios run

## Cross-language check flow

Primary runner:

```bash
python3 scripts/run_rtl_reference_check.py
```

This script:

1. exports fixed-point vectors from Python into per-sample directories under `--out-dir`
2. runs Python fixed-point golden (vectorised, floor-division arithmetic)
3. compiles/runs C++ fixed reference (`ref/cpp/crossbar_snn_ref_fixed.cpp`)
4. compiles/runs SystemC reference (`ref/systemc/crossbar_snn_ref_fixed_sc.cpp`)
5. compiles Verilog RTL with `iverilog`, runs each sample with `vvp sim_fixed +data_dir=<sample_dir>`
6. asserts exact equality: Python == C++ == SystemC == Verilog RTL

Pass `--skip-compile` to reuse binaries from a previous run.

Shortcut (5 samples by default):

```bash
make rtl-check
# or: make rtl-check SAMPLES="0 1 2 3 4 5 6 7 8 9"
```
