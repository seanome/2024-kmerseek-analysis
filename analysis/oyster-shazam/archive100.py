"""Compatibility entrypoint for previously submitted public archive workers."""
import sys
from archive_compare import ROOT, prepare, worker

if __name__ == '__main__':
    ROOT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == 'prepare':
        prepare()
    else:
        worker(sys.argv[1])
