# snn-crossbar-model

`snntorch`-based spiking neural network project aimed at hardware-oriented crossbar design exploration.

## What this project includes

- Quantized synaptic weights (STE quantization) to emulate limited conductance states.
- Leaky Integrate-and-Fire (LIF) neurons using `snntorch`.
- Time-step based spike simulation suitable for event-driven hardware mapping.
- Crossbar sizing report (`crossbar_report.json`) with tile count/utilization estimates.
- Training + evaluation scripts on MNIST for a concrete baseline.

## Directory layout

- `src/` - SNN model code and Verilog RTL (`snn_core_fixed.v`)
- `test/` - Verilog testbench and test Makefile
- `doc/` - design notes/spec
- `tests/` - optional Python test area
- `artifacts/` - generated checkpoints and reports (created at runtime)

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python train.py \
  --epochs 2 \
  --batch-size 64 \
  --num-steps 25 \
  --hidden-dim 256 \
  --weight-levels 32 \
  --crossbar-rows 128 \
  --crossbar-cols 128
```

Outputs under `artifacts/`:

- `best_model.pt`
- `history.json`
- `crossbar_report.json`

## Evaluate

```bash
python evaluate.py --checkpoint artifacts/best_model.pt
```

The script prints JSON metrics including accuracy, average spikes/sample, and crossbar mapping stats.

## C++ reference model (golden check)

This repo includes a deterministic C++ reference implementation of the forward path:

- `ref/cpp/crossbar_snn_ref.cpp`
- `scripts/export_and_compare_ref.py`

The Python script exports:

- quantized weights (`w1.txt`, `w2.txt`)
- fixed input spike train (`spikes.txt`)
- expected logits from the Python model (`expected_logits.txt`)

Then it compiles and runs the C++ model and reports error:

```bash
python scripts/export_and_compare_ref.py --checkpoint artifacts/best_model.pt
```

Or via Makefile:

```bash
make ref-compare
```

To explicitly match `snntorch` behavior for expected logits:

```bash
python scripts/export_and_compare_ref.py \
  --checkpoint artifacts/best_model.pt \
  --expected-mode snntorch
```

Or use the HW-discrete behavior:

```bash
python scripts/export_and_compare_ref.py \
  --checkpoint artifacts/best_model.pt \
  --expected-mode discrete
```

This gives you a language-independent reference path you can use to validate RTL/HLS models later, with selectable semantics.

Both modes are now supported end-to-end in C++:

- `snntorch`: mirrors `snntorch.Leaky` default reset-delay behavior
- `discrete`: hardware-friendly immediate reset-by-subtraction

## Verilog RTL cross-check (Python + C++ + RTL)

Added fixed-point RTL reference flow:

- RTL core: `src/snn_core_fixed.v`
- Testbench: `test/tb_snn_core_fixed.sv`
- C++ fixed reference: `ref/cpp/crossbar_snn_ref_fixed.cpp`
- Unified runner: `scripts/run_rtl_reference_check.py`

Run the 3-way check:

```bash
python3 scripts/run_rtl_reference_check.py
```

or:

```bash
make rtl-check
```

The runner exports fixed vectors and verifies:

- Python fixed model logits
- C++ fixed reference logits
- Verilog RTL logits

all match exactly.

## Hardware design notes

- `weight_levels` approximates resistive levels in analog/RRAM-like arrays.
- `num_steps` controls temporal resolution and event sparsity.
- `crossbar_rows` and `crossbar_cols` drive tile count and utilization estimates.
- Spike statistics can be used to estimate switching activity and dynamic energy.
