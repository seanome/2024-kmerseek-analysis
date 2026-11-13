#!/usr/bin/env python3
"""Keep only the truth rows on proteins this run searched.

Pfam-N and M-CSA are built once, ahead of every run, by `make pfamn-truth` and
`make mcsa-truth`. Both read the FULL annotations directory, so their human truth file
holds every human protein in Pfam: 44_315 Pfam-N regions on 18_724 proteins. The mini,
midi and midi-plus runs search only a subset (200, 964 and 998 proteins). Scored as it
was, recall on those runs divided the regions a tool found by regions on proteins nobody
searched. On midi-plus that was 2_218 regions that could be found against 44_315 in the
denominator, so every tool's Pfam-N recall came out about 20 times too low, and so did the
recall half of Fmax.

The Pfam-A and Swiss-Prot truth sets do not have this problem because they are built
inside the run, from `--annotations`, and keep only accessions in that directory's
human_pfam_domains.parquet. This script applies the same rule to a truth set built
outside the run: a protein counts as a query exactly when it is in that file.

The output keeps the input's file name and columns, so scoring reads it unchanged.
"""

import argparse
import json
from pathlib import Path

import polars as pl


def query_accessions(annotations: Path) -> set[str]:
    """The human proteins this run searched, by the rule every in-run truth builder uses."""
    path = annotations / "human_pfam_domains.parquet"
    if not path.exists():
        raise SystemExit(f"no {path}; cannot tell which proteins this run searched")
    return set(
        pl.read_parquet(path, columns=["accession"])["accession"].unique().to_list()
    )


def restrict(truth: pl.DataFrame, queries: set[str]) -> pl.DataFrame:
    return truth.filter(pl.col("accession").is_in(queries))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--truth",
        required=True,
        type=Path,
        help="human_<label>_truth.parquet built from the full annotations",
    )
    p.add_argument(
        "--annotations",
        required=True,
        type=Path,
        help="this run's annotations directory (the one --annotations names)",
    )
    p.add_argument("--label", required=True, help="truth set label, for the summary")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--summary-out", required=True, type=Path)
    args = p.parse_args()

    truth = pl.read_parquet(args.truth)
    queries = query_accessions(args.annotations)
    kept = restrict(truth, queries)
    kept.write_parquet(args.out, compression="zstd")

    summary = {
        "truth_set": args.label,
        "n_query_proteins": len(queries),
        "before": {
            "n_regions": truth.height,
            "n_proteins": truth["accession"].n_unique(),
        },
        "after": {"n_regions": kept.height, "n_proteins": kept["accession"].n_unique()},
    }
    args.summary_out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
