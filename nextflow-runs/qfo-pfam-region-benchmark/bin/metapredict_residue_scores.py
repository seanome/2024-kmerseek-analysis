#!/usr/bin/env python3
"""Per-residue metapredict disorder scores as `accession  position  score` (TSV, no header).

Runs in the metapredict image (metapredict 3.0.1, Dockerfile.metapredict). The same call
predict_disorder_metapredict.py makes for the disorder covariate: predict_disorder_fasta
returns {header: [sequence, per_residue_scores]}, scores 0-1. Pyrrolysine and
selenocysteine are mapped to K and C first, as that script does, because metapredict
refuses them.

The calls are made from this table by `no_search_baselines.py runs`, so IUPred and
metapredict go through one run-finding function.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_disorder_metapredict import accession_of, substitute_noncanonical  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fasta", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()

    import metapredict as mp

    clean = Path("noncanonical_substituted.fasta")
    counts = substitute_noncanonical(a.fasta, clean)
    fasta = clean if counts else a.fasta
    if counts:
        print(f"substituted non-canonical residues: {counts}", file=sys.stderr)

    pred = mp.predict_disorder_fasta(str(fasta), show_progress_bar=False)
    n = 0
    with open(a.out, "w") as fh:
        for header, (seq, scores) in pred.items():
            if len(scores) != len(seq):
                raise SystemExit(f"{header}: {len(scores)} scores for {len(seq)} residues")
            acc = accession_of(header)
            for i, s in enumerate(scores, 1):
                fh.write(f"{acc}\t{i}\t{float(s):.4f}\n")
            n += 1
    print(f"metapredict {mp.__version__}: {n} proteins scored", file=sys.stderr)


if __name__ == "__main__":
    main()
