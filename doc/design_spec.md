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
  - core fixed-point SNN compute block
  - reads vectors with `$readmemh`
  - computes logits across multiple clock cycles (one SNN timestep per cycle)
    after a rising-edge qualified `start` while idle, with a sticky `done`
    handshake until `start` is released; after that handshake, `start` must
    return low before another run can arm
- `test/tb_snn_core_fixed.sv`:
  - drives reset/start
  - waits for `done`
  - writes `artifacts/ref_vectors_fixed/verilog_logits.txt`

## Cross-language check flow

Primary runner:

```bash
python3 scripts/run_rtl_reference_check.py
```

This script:

1. exports fixed-point vectors from Python
2. runs Python fixed-point golden
3. compiles/runs C++ fixed reference (`ref/cpp/crossbar_snn_ref_fixed.cpp`)
4. compiles/runs Verilog RTL (`iverilog` + `vvp`)
5. checks equality: Python == C++ == Verilog

Shortcut:

```bash
make -C test rtl-check
```
