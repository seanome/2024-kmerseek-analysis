PYTHON ?= python3

.PHONY: test running

test:
	$(PYTHON) -m unittest discover -s tests -v
	$(PYTHON) protocol.py check-rules

running:
	$(MAKE) -C ../.. running
