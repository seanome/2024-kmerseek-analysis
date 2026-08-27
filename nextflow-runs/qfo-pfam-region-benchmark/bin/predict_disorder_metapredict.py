#!/usr/bin/env python3
"""Per-protein disorder from metapredict, as a covariate parquet.

Why this exists alongside the pLDDT proxy: pLDDT below 50 is not a disorder measurement,
it is a confidence measurement that CORRELATES with disorder. It also drops when AlphaFold
simply modelled a protein badly, which usually means a shallow MSA -- and a shallow MSA
independently hurts the profile baselines (jackhmmer, hhblits). So an apparent "accuracy
falls with disorder" effect read off pLDDT could partly be an MSA-depth effect hitting
several arms at once.

metapredict predicts disorder from sequence alone. It needs no structure and no alignment,
so it shares neither confound, and the two axes disagreeing is itself informative.

Verified against metapredict 3.0.1: predict_disorder_fasta returns
{header: [sequence, per_residue_scores]}, scores in 0-1. Sanity check on the way in --
P04637 (p53, a textbook IDP) scores 0.432 disordered and P69905 (haemoglobin alpha, a
compact globin) scores 0.000.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


def accession_of(header: str) -> str:
    """`sp|P04637|P53_HUMAN ...` -> `P04637`, matching every other table's key.

    Falls back to the first whitespace-delimited token for headers that are not
    UniProt-style, rather than dropping the record: an unjoinable accession is a
    left-join miss downstream, which is visible, while a dropped record is not.
    """
    first = header.split()[0]
    parts = first.split("|")
    return parts[1] if len(parts) >= 2 and parts[1] else first


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fasta", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--summary-out", type=Path)
    # metapredict's own default is used when this is not set, rather than a number copied
    # here that would silently diverge from the package's if it ever changed.
    p.add_argument("--threshold", type=float, default=None,
                   help="disorder score at or above which a residue counts as disordered; "
                        "default is whatever metapredict's own default is")
    args = p.parse_args()

    import metapredict as mp

    preds = mp.predict_disorder_fasta(str(args.fasta), show_progress_bar=False)
    thr = args.threshold if args.threshold is not None else 0.5

    rows = []
    for header, value in preds.items():
        # [sequence, scores]. Guarded rather than unpacked blind: this shape is a property
        # of the metapredict version, and a silent misread would produce a plausible-looking
        # column of wrong numbers.
        if not (isinstance(value, (list, tuple)) and len(value) == 2):
            raise SystemExit(
                f"metapredict returned {type(value).__name__} of length "
                f"{len(value) if hasattr(value, '__len__') else '?'} for {header!r}; "
                "expected [sequence, scores]. The API changed -- check the version."
            )
        _seq, scores = value
        n = len(scores)
        if n == 0:
            continue
        rows.append({
            "accession": accession_of(header),
            "n_residues_metapredict": n,
            "disorder_fraction_metapredict": sum(1 for s in scores if s >= thr) / n,
            "mean_disorder_metapredict": sum(scores) / n,
        })

    cov = pl.DataFrame(rows)
    cov.write_parquet(args.out, compression="zstd")

    summary = {
        "n_proteins": cov.height,
        "threshold": thr,
        "metapredict_version": getattr(mp, "__version__", "unknown"),
        "median_disorder_fraction":
            float(cov["disorder_fraction_metapredict"].median()) if cov.height else None,
        "n_mostly_disordered":
            int((cov["disorder_fraction_metapredict"] >= 0.5).sum()) if cov.height else 0,
    }
    if args.summary_out:
        args.summary_out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
