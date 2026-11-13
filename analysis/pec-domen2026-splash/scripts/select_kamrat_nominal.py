#!/usr/bin/env python3
"""KaMRaT contigs at the same uncorrected cutoff as the anchor-count route.

Usage: select_kamrat_nominal.py <contigs_all.tsv.gz> <design.tsv> <out_dir> [min_abs_log2fc]

KaMRaT 1.2 reports only the BH-corrected t-test p (score -scoreby ttest.padj), which is 1 for
nearly every contig in the real contrasts. This recomputes the uncorrected p with KaMRaT's own
test (Scorer::LogTtestScore in src/data_struct/scorer.cpp, tag v1.2.0): a Welch t-test on
log2(count + 1) between PEC and control, with p = 1 when both groups have zero spread.

The counts are the per-sample columns of contigs_all.tsv.gz: for each contig, the mean over its
member k-mers of KaMRaT's normalised counts (index -nfbase 1000000000, so each sample's k-mer
counts are scaled to sum to 1e9; merge -counts mean:float). KaMRaT itself scores each k-mer and
gives a contig its member k-mers' smallest corrected p; those per-k-mer values were not kept, so
here the test is run once per contig on its mean counts.

log2 fold change = mean of log2(count + 1) in PEC minus the same in control (the difference the
t-test tests). Kept: p <= 0.05 and |log2 fold change| >= 1, the anchor-count route's cutoff.
With min_abs_log2fc = 0 (added 2026-11-02, Olga: "use nominal p < 0.05 for kamrat and de-kupl"):
p < 0.05 and no fold-change cutoff; direction is still the sign of the log2 fold change.

Writes <out_dir>/contigs.tsv (contig, nb-merged-kmer, pvalue, log2fc, direction; up = higher in
PEC) and <out_dir>/summary.json (contigs tested, p <= 0.05, p <= 0.05 and 2-fold, up, down).
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats


def welch_log2(x: np.ndarray, is_pec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """KaMRaT's t-test on log2(x + 1), one row per contig. Returns (p, PEC minus control mean)."""
    y = np.log2(x.astype(np.float64) + 1)
    a, b = y[:, ~is_pec], y[:, is_pec]
    na, nb = a.shape[1], b.shape[1]
    ma, mb = a.mean(1), b.mean(1)
    # equal values can leave a rounding-sized variance; treat it as zero spread, as KaMRaT intends
    va, vb = a.var(1, ddof=1), b.var(1, ddof=1)
    va = np.where(va <= 1e-12 * np.maximum(1, ma**2), 0.0, va)
    vb = np.where(vb <= 1e-12 * np.maximum(1, mb**2), 0.0, vb)
    t1, t2 = va / na, vb / nb
    with np.errstate(divide="ignore", invalid="ignore"):
        df = (t1 + t2) ** 2 / (t1**2 / (na - 1) + t2**2 / (nb - 1))
        t = (ma - mb) / np.sqrt(t1 + t2)
        p = 2 * stats.t.sf(np.abs(t), df)
    p = np.where((va == 0) & (vb == 0), 1.0, p)
    return p, mb - ma


def main() -> None:
    contigs_all, design, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    min_lfc = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    out.mkdir(parents=True, exist_ok=True)
    dsg = pl.read_csv(design, separator="\t")
    d = pl.read_csv(contigs_all, separator="\t", infer_schema_length=None)  # decimals can first appear past row 100
    samples = [s for s in dsg["sample"] if s in d.columns]
    assert len(samples) == dsg.height, f"design samples missing from contigs: {set(dsg['sample']) - set(samples)}"
    cond = dict(zip(dsg["sample"], dsg["condition"]))
    assert set(cond.values()) == {"PEC", "control"}, set(cond.values())
    is_pec = np.array([cond[s] == "PEC" for s in samples])
    p, lfc = welch_log2(d.select(samples).to_numpy(), is_pec)
    assert not np.isnan(p).any(), "NaN p-value"
    d = d.select("contig", "nb-merged-kmer").with_columns(pl.Series("pvalue", p), pl.Series("log2fc", lfc))
    nominal = d.filter(pl.col("pvalue") <= 0.05) if min_lfc > 0 else d.filter(pl.col("pvalue") < 0.05)
    keep = (nominal.filter(pl.col("log2fc").abs() >= min_lfc)
            .with_columns(pl.when(pl.col("log2fc") > 0).then(pl.lit("up")).otherwise(pl.lit("down")).alias("direction")))
    keep.write_csv(out / "contigs.tsv", separator="\t")
    s = {"contigs_tested": d.height, "min_abs_log2fc": min_lfc, "p_le_0.05": nominal.height, "kept": keep.height,
         "p_le_0.05_and_2fold": int((nominal["log2fc"].abs() >= 1).sum()),
         "up": keep.filter(pl.col("direction") == "up").height,
         "down": keep.filter(pl.col("direction") == "down").height,
         "kmers_in_kept_contigs": int(keep["nb-merged-kmer"].sum()),
         "samples_pec": int(is_pec.sum()), "samples_control": int((~is_pec).sum())}
    (out / "summary.json").write_text(json.dumps(s, indent=1) + "\n")
    print(json.dumps(s))


if __name__ == "__main__":
    main()
