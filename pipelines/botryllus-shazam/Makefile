.PHONY: running test
running:
	$${SCRATCH:?Run on Sherlock}/nf-running

test:
	python3 -m unittest discover -s ../../scripts/botryllus_shazam -p test_nextflow_stages.py
