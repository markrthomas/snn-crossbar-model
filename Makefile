PYTHON ?= python3

.PHONY: install train eval ref-compare rtl-check test sweep noise docs clean

install:
	$(PYTHON) -m pip install -r requirements.txt

train:
	$(PYTHON) train.py

eval:
	$(PYTHON) evaluate.py

ref-compare:
	$(PYTHON) scripts/export_and_compare_ref.py

# RTL cross-check: compile once, verify SAMPLES test-set images (default 1).
#   make rtl-check                           # sample 0 only (CI fast path)
#   make rtl-check SAMPLES="0 1 2 3 4 5 6 7 8 9"   # one per digit class
SAMPLES ?= 0
rtl-check:
	$(PYTHON) scripts/run_rtl_reference_check.py --samples $(SAMPLES)

test:
	$(MAKE) -C test rtl-check

# Sweep weight_levels × num_steps.  Override vars to customise, e.g.:
#   make sweep SWEEP_EPOCHS=5 SWEEP_LEVELS="4 8 16 32" SWEEP_STEPS="5 10 25"
SWEEP_EPOCHS  ?= 3
SWEEP_LEVELS  ?= 4 8 16 32
SWEEP_STEPS   ?= 5 10 25
SWEEP_DIMS    ?= 128 256
SWEEP_NOISE   ?= 0.0
sweep:
	$(PYTHON) scripts/sweep_qat.py \
	  --epochs $(SWEEP_EPOCHS) \
	  --weight-levels $(SWEEP_LEVELS) \
	  --num-steps $(SWEEP_STEPS) \
	  --hidden-dims $(SWEEP_DIMS) \
	  --noise-sigma $(SWEEP_NOISE) \
	  --save-checkpoints

# Noise robustness eval against a trained checkpoint.
#   make noise NOISE_CKPT=artifacts/best_model.pt NOISE_LEVELS=32 NOISE_STEPS=25
NOISE_CKPT    ?= ./artifacts/best_model.pt
NOISE_LEVELS  ?= 32
NOISE_STEPS   ?= 25
noise:
	$(PYTHON) scripts/eval_noise.py \
	  --checkpoint $(NOISE_CKPT) \
	  --weight-levels $(NOISE_LEVELS) \
	  --num-steps $(NOISE_STEPS) \
	  --out artifacts/noise_results.json

docs:
	@echo "Documentation available at doc/design_spec.md"

clean:
	$(MAKE) -C test clean
