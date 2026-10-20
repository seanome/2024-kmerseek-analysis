#!/usr/bin/env python3
"""Notebook 359: keep, per blank exon, the strongest kmerseek call that lands on it.

    kmerseek search ... | reduce_calls.py <blank_exons.parquet> <real|decoy> > best.tsv

Reads kmerseek's CSV on stdin (columns found by header name; regions are 0-based and
end-exclusive on both proteins). A call on query real__<isoform> is checked against every
blank exon whose representative isoform is <isoform>; a call on decoy__<exon_id> only
against that exon. A call lands when it covers at least half of the exon's protein
segment. Because the threshold is on region_evalue alone, the landing call with the
smallest E-value is all the scorer needs for every threshold; it is written with its
target and residues. Ties on E keep the first call kmerseek wrote.

Standard library only, so it runs in any image with Python 3.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks"))
import isoform_blank_exons_359 as ib  # noqa: E402

csv.field_size_limit(sys.maxsize)


def load_segments(parquet, side):
    import polars as pl

    bl = pl.read_parquet(parquet, columns=["exon_id", "rep_isoform", "p_start", "p_end"])
    by_query = {}
    for r in bl.iter_rows(named=True):
        q = f"real__{r['rep_isoform']}" if side == "real" else f"decoy__{r['exon_id']}"
        by_query.setdefault(q, []).append((r["exon_id"], (r["p_start"], r["p_end"])))
    return by_query


def main(parquet, side):
    by_query = load_segments(parquet, side)
    reader = csv.reader(sys.stdin)
    head = next(reader, None)
    if head is None:
        sys.exit("reduce_calls: no header on stdin")
    c = {name: i for i, name in enumerate(head)}
    need = ["query_name", "target_name", "region_start", "region_end", "target_start",
            "target_end", "region_evalue", "region_subseq", "target_subseq"]  # fmt: skip
    missing = [n for n in need if n not in c]
    if missing:
        sys.exit(f"reduce_calls: missing columns {missing}")
    best = {}
    n_calls = 0
    for row in reader:
        n_calls += 1
        segs = by_query.get(row[c["query_name"]])
        if not segs:
            continue
        call = (int(row[c["region_start"]]), int(row[c["region_end"]]))
        e = float(row[c["region_evalue"]])
        for exon_id, seg in segs:
            if ib.lands(call, seg) and (exon_id not in best or e < best[exon_id][0]):
                best[exon_id] = (e, row)
    out = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    out.writerow(["exon_id", "side", "evalue", "query", "qstart", "qend", "target", "tstart", "tend", "qseq", "tseq"])
    for exon_id, (e, row) in sorted(best.items()):
        out.writerow([exon_id, side, repr(e), row[c["query_name"]], row[c["region_start"]], row[c["region_end"]],
                      row[c["target_name"]], row[c["target_start"]], row[c["target_end"]],
                      row[c["region_subseq"]], row[c["target_subseq"]]])  # fmt: skip
    print(f"reduce_calls: {n_calls} calls read, {len(best)} exons with a landing call", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
