PYTHON ?= python3

.PHONY: install train eval ref-compare rtl-check test docs clean

install:
	$(PYTHON) -m pip install -r requirements.txt

train:
	$(PYTHON) train.py

eval:
	$(PYTHON) evaluate.py

ref-compare:
	$(PYTHON) scripts/export_and_compare_ref.py

rtl-check:
	$(PYTHON) scripts/run_rtl_reference_check.py

test:
	$(MAKE) -C test rtl-check

docs:
	@echo "Documentation available at doc/design_spec.md"

clean:
	$(MAKE) -C test clean
