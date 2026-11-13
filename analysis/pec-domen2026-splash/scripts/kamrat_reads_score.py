#!/usr/bin/env python3
"""Paper's lists reached by KaMRaT contigs and by the reads holding them (notebook 361).

Usage: kamrat_reads_score.py <pec_root> <manifest.tsv> <out_dir>

Runs paper_recovery.py on the three small contrasts by pointing it at folders that hold only the
searches wanted, so each list is tested against the same 1,000 random gene sets as before. Swaps use
DESeq2 contigs only, scored against the real contrast's list.

Writes to <out_dir>:
  recovery_real.tsv     paper_recovery.py's rows: t-test and DESeq2, contigs and reads
  recovery_swaps.tsv    the same for reads on each swap, plus contrast's flip in `run`
  gene_overlap.tsv      per list and route, list genes reached by contigs only, reads only, both
  inputs.tsv            per label set and route: contigs, contigs with no read, contigs at the limit
                        of 100 reads, distinct reads, read length, peptides, genes reached
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import polars as pl

root, manifest, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
SCRIPT = Path(__file__).resolve().parent / "paper_recovery.py"
CONTRASTS = ["young_13_49d", "young_132d", "old_41_93d"]
ROUTES = {"kamrat_ttest_p05": "kamrat_p05", "kamrat_deseq2_p05": "kamrat_deseq2_p05"}
id_map = root / "db/botznik.id_map.tsv"
bz = (pl.read_csv(id_map, separator="\t").filter(pl.col("original_id").str.contains(r"^g\d+\.t\d+$"))
      .select(pl.col("stable_id").alias("target"), pl.col("original_id").str.replace(r"\.t\d+$", "").alias("gene")))
lists = pl.read_csv(root / "paper/paper_named_lists.tsv", separator="\t")


def recovery(view):
    tsv = view / "recovery.tsv"
    subprocess.run([sys.executable, str(SCRIPT), str(view), str(root / "paper/paper_deseq_all.parquet"),
                    str(root / "paper/paper_named_lists.tsv"), str(id_map), str(tsv)], check=True, stdout=subprocess.DEVNULL)
    return pl.read_csv(tsv, separator="\t")


def reached(search):
    """{direction: genes} from one search folder."""
    hits = pl.read_parquet(search / "align.botznik.parquet", columns=["query", "target", "supported"]).filter(pl.col("supported"))
    pep = pl.read_csv(search / "pep.peptides.tsv", separator="\t", columns=["peptide_id", "direction"])
    h = hits.join(bz, on="target").join(pep, left_on="query", right_on="peptide_id")
    return {d: set(h.filter(pl.col("direction") == d)["gene"]) for d in ("up", "down")}


with tempfile.TemporaryDirectory(dir=out) as tmp:
    view = Path(tmp) / "real"
    for c in CONTRASTS:
        (view / c).mkdir(parents=True)
        for folder in ROUTES.values():
            for s in (f"search_{folder}", f"search_{folder}_reads"):
                (view / c / s).symlink_to(root / "contrasts" / c / s)
    recovery(view).write_csv(out / "recovery_real.tsv", separator="\t")

    swaps = []
    for r in pl.read_csv(manifest, separator="\t").filter(pl.col("contrast").is_in(CONTRASTS)).iter_rows(named=True):
        view = Path(tmp) / f"{r['contrast']}__{r['flip']}"
        (view / r["contrast"]).mkdir(parents=True)
        s = "search_kamrat_deseq2_p05_reads"
        (view / r["contrast"] / s).symlink_to(root / "label_swap" / r["contrast"] / r["flip"] / s)
        swaps.append(recovery(view).filter(pl.col("channel") == "kamrat_deseq2_p05_reads").with_columns(pl.lit(r["flip"]).alias("run")))
    pl.concat(swaps).write_csv(out / "recovery_swaps.tsv", separator="\t")

rows, inputs = [], []
man = pl.read_csv(manifest, separator="\t").filter(pl.col("contrast").is_in(CONTRASTS))
sets = [(c, "real", root / "contrasts" / c, ch, f) for c in CONTRASTS for ch, f in ROUTES.items()]
sets += [(r["contrast"], r["flip"], root / "label_swap" / r["contrast"] / r["flip"], "kamrat_deseq2_p05", "kamrat_deseq2_p05")
         for r in man.iter_rows(named=True)]
for c, run, d, channel, folder in sets:
    a, b = reached(d / f"search_{folder}"), reached(d / f"search_{folder}_reads")
    if run == "real":
        for direction in ("up", "down"):
            g = set(lists.filter((pl.col("contrast") == c) & (pl.col("direction") == direction))["gene_id"]) & set(bz["gene"])
            rows.append(dict(contrast=c, channel=channel, list=direction, list_genes=len(g), contigs_only=len(g & a[direction] - b[direction]),
                             reads_only=len(g & b[direction] - a[direction]), both=len(g & a[direction] & b[direction])))
    s = d / f"search_{folder}_reads"
    held = pl.read_csv(s / "held.contigs.tsv", separator="\t")
    reads = pl.read_csv(s / "held.reads.tsv", separator="\t").unique("read")["read"].str.len_chars()
    pep = pl.read_csv(s / "pep.peptides.tsv", separator="\t", columns=["peptide_id"]).height
    inputs.append(dict(contrast=c, run=run, channel=channel, contigs=held.height, contigs_no_read=int((held["reads_holding"] == 0).sum()),
                       contigs_over_limit=int((held["reads_holding"] > held["reads_kept"]).sum()), read_hits=int(held["reads_holding"].sum()),
                       distinct_reads=reads.len(), mean_read_length=round(float(reads.mean()), 1), peptides=pep,
                       genes_reached_contigs=len(a["up"] | a["down"]), genes_reached_reads=len(b["up"] | b["down"])))
pl.DataFrame(rows).write_csv(out / "gene_overlap.tsv", separator="\t")
pl.DataFrame(inputs).write_csv(out / "inputs.tsv", separator="\t")
print(pl.DataFrame(inputs))
