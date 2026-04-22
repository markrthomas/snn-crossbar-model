# SNN Fixed-Point RTL Reference Design

This document defines the Verilog reference implementation used for cross-language checking against Python and C++.

## Scope

- One-sample, fixed-point inference for a 2-layer SNN:
  - Input spikes -> FC1 -> LIF -> FC2 -> LIF -> output spike counts (`logits`)
- Uses the same deterministic spike vectors exported by Python.
- Intended as a verification-oriented RTL baseline for future crossbar accelerator work.

## Arithmetic model

- Weight format: signed 16-bit integers (`w1.memh`, `w2.memh`)
- Spike format: binary (`spikes.memh`)
- Membrane format: signed 32-bit integers
- Decay: integer rational
  - `mem_pre = (BETA_NUM * mem) / BETA_DEN + current`
- Threshold/reset (discrete mode):
  - if `mem_pre >= THRESHOLD`: spike = 1, `mem = mem_pre - THRESHOLD`
  - else: spike = 0, `mem = mem_pre`

Default constants:

- `BETA_NUM=95`
- `BETA_DEN=100`
- `THRESHOLD=256`

## RTL files

- `src/snn_core_fixed.v`:
  - core fixed-point SNN compute block
  - reads vectors with `$readmemh`
  - computes logits for all timesteps on `start`
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
