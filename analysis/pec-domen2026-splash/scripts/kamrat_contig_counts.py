#!/usr/bin/env python3
"""KaMRaT contig counts turned back into read-count scale, for a DESeq2 fit of the contigs.

Usage: kamrat_contig_counts.py <contigs_all.tsv.gz> <raw-counts.tsv.gz> <design.tsv> <out_dir>

Added 2026-11-02 at Olga's request ("do KAMRAT + DeSeq2 not just a ttest"). KaMRaT has no DESeq2
test, so KaMRaT is used only to merge k-mers into contigs, and the contigs are tested with
PyDESeq2 like the anchor counts (anchor_deseq.py).

contigs_all.tsv.gz (kamrat process, main.nf) gives each contig the mean over its member k-mers of
KaMRaT's normalised counts. KaMRaT index -nfbase 1000000000 scales sample j's counts by
1e9 / (sum of sample j's counts over every k-mer of the input table). DESeq2 needs counts on the
read scale, so each sample's column is multiplied back by (its column sum in DE-kupl's
raw-counts.tsv.gz, the table KaMRaT indexed) / 1e9 and rounded to a whole number.

Check before writing: contigs made of one k-mer must give back that k-mer's raw counts (either
strand, since KaMRaT ran with -unstrand). The script stops if the median relative difference over
those k-mers is above 1 %.

Writes <out_dir>/contig_counts.tsv (no header: contig, nb-merged-kmer, one count per sample, the
format anchor_deseq.py reads), <out_dir>/samples.txt, <out_dir>/unscale.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

NFBASE = 1e9
COMP = str.maketrans("ACGT", "TGCA")


def main() -> None:
    contigs_all, raw_counts, design, out = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
    out.mkdir(parents=True, exist_ok=True)
    samples = pl.read_csv(design, separator="\t")["sample"].to_list()
    d = pl.read_csv(contigs_all, separator="\t")
    missing = set(samples) - set(d.columns)
    assert not missing, f"samples missing from contigs: {missing}"

    # one pass over raw-counts: column sums, and the counts of k-mers that are one-k-mer contigs
    single = d.filter(pl.col("nb-merged-kmer") == 1).head(2000)
    want = {}
    for i, s in enumerate(single["contig"]):
        want[s] = i
        want[s.translate(COMP)[::-1]] = i
    raw = pl.read_csv(raw_counts, separator="\t", schema_overrides={s: pl.Int64 for s in samples})
    tag = raw.columns[0]
    sums = raw.select(samples).sum().to_numpy()[0].astype(np.int64)
    hit = raw.filter(pl.col(tag).is_in(list(want)))
    found = {want[t]: np.array(v, dtype=np.int64) for t, *v in hit.select(tag, *samples).iter_rows()}
    n_kmers = raw.height
    del raw
    factor = sums / NFBASE  # raw = normalised * factor

    norm = d.select(samples).to_numpy().astype(np.float64)
    rows = sorted(found)
    est = single.select(samples).to_numpy().astype(np.float64)[rows] * factor
    obs = np.array([found[r] for r in rows], dtype=np.float64)
    ok = obs >= 10
    rel = np.abs(est[ok] - obs[ok]) / obs[ok]
    check = dict(one_kmer_contigs_checked=len(rows), values_compared=int(ok.sum()),
                 median_relative_difference=float(np.median(rel)) if ok.any() else None,
                 p99_relative_difference=float(np.quantile(rel, 0.99)) if ok.any() else None)
    print(json.dumps(check))
    assert len(rows) > 100 and ok.sum() > 100, "too few one-k-mer contigs found in raw-counts to check"
    assert check["median_relative_difference"] < 0.01, f"unscaling does not give back raw counts: {check}"

    counts = np.rint(norm * factor).astype(np.int64)
    t = d.select("contig", "nb-merged-kmer").with_columns(
        [pl.Series(s, counts[:, j]) for j, s in enumerate(samples)])
    t.write_csv(out / "contig_counts.tsv", separator="\t", include_header=False)
    (out / "samples.txt").write_text("\n".join(samples) + "\n")
    (out / "unscale.json").write_text(json.dumps(dict(
        nfbase=NFBASE, raw_counts_kmers=n_kmers, raw_column_sums=dict(zip(samples, map(int, sums))),
        factor=dict(zip(samples, map(float, factor))), contigs=d.height, check=check), indent=1) + "\n")


if __name__ == "__main__":
    main()
