HERE := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
REPO := $(abspath $(HERE)../..)
PYTHON := python
SYNTHETIC := $(HERE)build/synthetic
RESULTS := $(HERE)build/results

.PHONY: check test synthetic run figure clean

check: test run figure

test:
	cd $(HERE) && $(PYTHON) -m pytest tests -q

synthetic:
	$(PYTHON) $(HERE)src/synthetic.py --output $(SYNTHETIC) --constructs 200 --seed 7

run: synthetic
	$(PYTHON) $(HERE)src/cli.py validate --input $(SYNTHETIC)/eval
	$(PYTHON) $(HERE)src/cli.py run --input $(SYNTHETIC)/eval --output $(RESULTS) \
		--development $(SYNTHETIC)/dev --workers 4

figure:
	$(PYTHON) $(HERE)src/figure.py --results $(RESULTS) \
		--figure $(REPO)/figures/construct_insert_check_paired_workflows_synthetic

clean:
	rm -rf $(HERE)build
