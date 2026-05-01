# Development Roadmap

This document tracks open work items after the initial review and clean-up
pass (commits `4d6dce4`, `a3048b9`, `c4115ba`).  Items are grouped by scope
and ordered roughly by implementation effort within each group.

---

## Near-term cleanups  *(small, self-contained)*

### 1 — Add `CLAUDE.md`

No AI-assistant context file exists.  A `CLAUDE.md` at the repo root should
capture: directory layout, key invariants (floor-division semantics, the
`BYPASS_QUANTIZATION` contract, the beta rational), how to run the four-way
cross-check, and the project's tolerance for coupling between `crossbar_snn.py`
and `asic_spec.py`.  This prevents re-explaining context on every new session.

### 2 — `evaluate.py` validates only the fc1 weight shape

`evaluate.py:44–51` checks `fc1.weight.shape` against `--hidden-dim` but not
`fc2.weight.shape`.  A wrong `--hidden-dim` that happens to make fc1 fit will
still produce a confusing PyTorch `RuntimeError` from `load_state_dict` when
fc2 mismatches.  Fix: add a symmetric check for `fc2.weight` shape
`(cfg.output_dim, cfg.hidden_dim)` and raise the same actionable message.

### 3 — `to_hex_signed()` reinvents two's complement

`run_rtl_reference_check.py:53–61` implements two's complement manually:

```python
if v < 0:
    v = (1 << bits) + v
width = nbytes * 2
return f"{v:0{width}x}"
```

Python 3.2+ supports this directly:

```python
return v.to_bytes(nbytes, byteorder="big", signed=True).hex()
```

The manual version silently produces wrong output for values outside
`[-(2^(bits-1)), 2^(bits-1)-1]` because the clamp is applied before the
sign check but after the call-site clamping — the logic is correct but
unnecessarily fragile.  The stdlib form is shorter and self-documenting.

### 4 — Wire `validate_asic_compat()` into `train.py`

`CrossbarConfig.validate_asic_compat()` was added in the P2 pass but is never
called in the main training entry-point.  `train.py` should call it early
(after building `cfg`, before constructing the model) so a misconfigured
training run fails fast with a clear message rather than silently producing a
model that won't match the RTL.

```python
from src.asic_spec import AsicFixedPointSpec
cfg.validate_asic_compat(AsicFixedPointSpec())
```

---

## Medium-term improvements  *(moderate scope, meaningful impact)*

### 5 — Golden accuracy CI baseline

CI verifies that the code *runs* (pytest + RTL cross-check) but not that a
trained model *achieves acceptable accuracy*.  A lightweight smoke-train step
(2 epochs, tiny hidden_dim) with an accuracy floor assertion would catch
regressions in training dynamics without significantly extending CI wall time.

Suggested form: a pytest mark `@pytest.mark.slow` that trains for 2 epochs on
a 10 % MNIST subset, asserts `test_acc > 0.80`, and is gated behind `CI=true`.

### 6 — RTL parameter sweep in CI

`tb_snn_core_fixed.sv` is only ever compiled and run with one set of
parameters (the defaults or whatever `--hidden-dim` was passed).  The RTL has
meaningful edge-case behaviour at `NUM_STEPS=1` (single timestep: IDLE → RUN →
DONE in back-to-back cycles) and at small dimensions.  Add a second `rtl-check`
invocation with e.g. `--hidden-dim 4 --num-steps 1` to give the FSM edge cases
coverage in CI.

### 7 — Document the training/cross-check arithmetic gap

`forward()` uses snntorch's native `Leaky` (floating-point, PyTorch autograd)
during training.  The cross-check uses `run_python_fixed()` (integer,
floor-division, fixed threshold in membrane units).  These are intentionally
different arithmetic models — training benefits from smooth gradients, the RTL
needs integer determinism — but the gap is not documented anywhere.

The design spec should include a section explaining: why exact integer equality
across the four implementations is achievable (all use the same fixed-point
contract), why that equality does *not* extend to the snntorch training forward
pass, and what the expected accuracy difference between snntorch-mode inference
and fixed-point inference is at the default parameters.

### 8 — `crossbar_report()` delta from `asic_spec.json`

`train.py` writes `artifacts/crossbar_report.json` (flat dict, no memory map)
while `run_rtl_reference_check.py` writes `artifacts/ref_vectors_fixed/asic_spec.json`
(nested, full memory map).  Both now derive from `default_asic_bundle()` but
expose different subsets of its output.  Consider making `crossbar_report()`
optionally emit the full bundle, or at minimum documenting the deliberate split
so future readers don't think the two files are inconsistent.

---

## Long-term / research directions  *(larger scope)*

### 9 — Pipelined / streaming RTL throughput

The current `snn_core_fixed.v` processes one sample serially (one timestep per
cycle, fully sequential across hidden and output neurons within each timestep).
For a real crossbar accelerator the weight matrix-vector product is the
hardware-natural unit of work.  A streaming variant that accepts one input
spike vector per cycle and produces logits after `NUM_STEPS` cycles — without
exposing the internal hidden-layer loop — would be a better model of crossbar
throughput.  This is a non-trivial RTL change that also requires updating the
testbench, the SystemC reference, and the C++ golden.

### 10 — Synthesis and area/power estimates

The RTL is described as "synthesisable" but no synthesis flow exists.  Adding
a Yosys (open-source) or vendor-neutral synthesis script with a target
technology (e.g. the open Sky130 PDK) would enable:

- gate-count and area estimates as a function of `hidden_dim` and `weight_levels`
- switching activity annotation from the simulation for dynamic power estimates
- a closed loop between the tiling/utilisation metrics reported by
  `crossbar_report()` and actual silicon cost

### 11 — Convolutional / recurrent extensions

The two-layer fully-connected topology is convenient for tiling analysis but
limits MNIST accuracy and is not representative of deployable neuromorphic
networks.  Extending to a small convolutional front-end (e.g. 1 conv layer +
pooling + 2 FC layers) would stress the crossbar tiling model more realistically
and bring accuracy above ~98 % without a dramatic increase in parameter count.
This requires updating `asic_spec.py` to handle conv weight tiling and the RTL
to accept pre-computed feature maps.

### 12 — Hardware-in-the-loop verification

Once a synthesis flow exists, closing the loop between the Python golden and
gate-level simulation (using back-annotated timing from synthesis) would verify
that the fixed-point contract holds under realistic timing conditions, not just
in the behavioural RTL simulator.  This is the natural next step after item 10.
