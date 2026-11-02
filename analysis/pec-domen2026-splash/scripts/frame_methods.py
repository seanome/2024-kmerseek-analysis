#!/usr/bin/env python3
"""Which way of choosing a read's reading frame is right most often, and what each one costs.

Usage: frame_methods.py <real_blastx.tsv> <stops.tsv> <time_dir> <out.tsv> [heldout=<genes_halfB.txt>]
                        blastx.<sens>=<diamond.tsv> ... <db>.<alphabet>.k<k>=<scores.tsv>:<k> ...

Reads and truth are those of frame_shares.py and frame_cutoff_check.py: simulated reads cut from
new-genome CDS (coding sequences; true frame in the read name) and real PRJNA1219309 reads whose
best DIAMOND blastx hit to a new-genome protein has >= 95 % identity and covers >= 90 % of the read.
Frames 0-2 are offsets 0-2 on the read, 3-5 offsets 0-2 on its reverse complement.

Each method gives every read a set of chosen frames (empty when it cannot choose):
  stops.one          the stop-free frames (a frame with no stop codon); right only when there is one
  stops.longest      the frames with the longest stop-free stretch (a stop-free frame counts as
                     longest); differs from stops.one only for reads with no stop-free frame
  <db>.<a>.k<k>      among stop-free frames, the ones with the highest share of k-mers found in the
                     reference (pec-tools frame-scores); empty if that share is 0
  blastx.<sens>      the frame of the read's best DIAMOND hit (highest bit score) to the old gene
                     models; empty when there is no hit
  combinations       the first method in the list that chooses any frame
Methods whose reference is newgenome_halfA (half of the new-genome genes, frame_heldout_split.py)
are scored only on reads whose gene is in the other half (the read's own gene for simulated reads,
the truth blastx hit's gene for real reads), so the read's own gene is not in the reference.
For k-mer methods, no_kmer_match is the share of reads with no reference k-mer in any of the six
frames, stop codons or not.
A read is "right" when the set is the true frame alone, "tied" when the set holds the true frame
and at least one wrong frame, "wrong" when the set has no true frame, and "none" when it is empty.

Resources come from <time_dir>/<method>.time (GNU time -v): wall time, CPU time (user + system)
and peak resident memory, also per million reads in the input. A combination's time is the sum of
its parts and its memory the largest part.
"""
import re
import sys
from pathlib import Path

import polars as pl

blastx_p, stops_p, time_dir, out_p = sys.argv[1:5]
specs = [a for a in sys.argv[5:] if not a.startswith("heldout=")]
heldout_p = next((a.split("=", 1)[1] for a in sys.argv[5:] if a.startswith("heldout=")), None)
BITS = [1 << i for i in range(6)]


def frame_of_hit(qstart, qend, qlen):
    """Frame of a blastx hit: offset of its first codon on the read or on the reverse complement."""
    return (pl.when(qstart < qend).then((qstart - 1) % 3).otherwise(3 + (qlen - qstart) % 3))


def best_hits(path, cols):
    return (pl.read_csv(path, separator="\t", has_header=False, new_columns=cols,
                        schema_overrides={"read_id": pl.Utf8})
            .sort(["read_id", "bits"], descending=[False, True]).unique("read_id", keep="first"))


truth = (best_hits(blastx_p, ["read_id", "g", "pident", "length", "qstart", "qend", "qlen", "bits"])
         .filter(pl.col("pident") >= 95)
         .filter((pl.col("qend") - pl.col("qstart")).abs() + 1 >= 0.9 * pl.col("qlen"))
         .select("read_id", "g", tf_real=frame_of_hit(pl.col("qstart"), pl.col("qend"), pl.col("qlen"))))
# NOTE: frame_cutoff_check.py filters identity before taking the best hit; with --max-target-seqs 1
# and --id 95 in the truth search every row already has >= 95 % identity, so the order is the same.

d = pl.read_csv(stops_p, separator="\t", schema_overrides={"read_id": pl.Utf8})
n_reads = d.height
is_sim = pl.col("read_id").str.starts_with("sim|")
d = (d.join(truth, on="read_id", how="left")
     .with_columns(src=pl.when(is_sim).then(pl.lit("simulated"))
                   .when(pl.col("tf_real").is_not_null()).then(pl.lit("real"))
                   .otherwise(pl.lit("real, frame unknown")),
                   tf=pl.when(is_sim).then(pl.col("read_id").str.split("|").list.get(4, null_on_oob=True).cast(pl.Int64))
                   .otherwise(pl.col("tf_real")))
     .with_columns(gene=pl.when(is_sim).then(pl.col("read_id").str.split("|").list.get(1, null_on_oob=True))
                   .otherwise(pl.col("g")).str.replace(r"#\d+$", "").str.replace(r"-T\d+$", ""))
     .drop("tf_real", "g"))
half_b = set(open(heldout_p).read().split()) if heldout_p else set()
d = d.with_columns(in_half_b=pl.col("gene").is_in(list(half_b)))


def mask_of(conds):
    """Bit mask (int) of the frames whose condition holds."""
    return pl.sum_horizontal([pl.when(c).then(b).otherwise(0) for c, b in zip(conds, BITS)])


nsf = pl.sum_horizontal([(pl.col(f"stops{i}") == 0).cast(pl.Int32) for i in range(6)])
maxrun = pl.max_horizontal([pl.col(f"run{i}") for i in range(6)])
d = d.with_columns(
    n_stopfree=nsf,
    **{"stops.one": mask_of([pl.col(f"stops{i}") == 0 for i in range(6)])},
    **{"stops.longest": pl.when(nsf > 0).then(mask_of([pl.col(f"stops{i}") == 0 for i in range(6)]))
       .otherwise(mask_of([pl.col(f"run{i}") == maxrun for i in range(6)]))})
d = d.with_columns(tstops=pl.coalesce([pl.when(pl.col("tf") == i).then(pl.col(f"stops{i}")) for i in range(6)]))
methods = ["stops.one", "stops.longest"]

for spec in specs:
    name, rest = spec.split("=", 1)
    if name.startswith("blastx."):
        h = (best_hits(rest, ["read_id", "s", "pident", "length", "qstart", "qend", "qlen", "evalue", "bits"])
             .select("read_id", f=frame_of_hit(pl.col("qstart"), pl.col("qend"), pl.col("qlen"))))
        h = h.select("read_id", pl.lit(2).pow(pl.col("f")).cast(pl.Int64).alias(name))
    else:
        path, k = rest.rsplit(":", 1)
        k = int(k)
        s = pl.read_csv(path, separator="\t", schema_overrides={"read_id": pl.Utf8},
                        columns=["read_id", "read_len"] + [f"m{i}" for i in range(6)] + [f"stops{i}" for i in range(6)])
        share = [pl.when(pl.col(f"stops{i}") == 0)
                 .then(pl.col(f"m{i}") / ((pl.col("read_len") - i % 3) // 3 - k + 1)).otherwise(-1.0)
                 for i in range(6)]
        s = s.with_columns([x.alias(f"g{i}") for i, x in enumerate(share)])
        best = pl.max_horizontal([pl.col(f"g{i}") for i in range(6)])
        h = s.select("read_id", pl.when(best > 0).then(mask_of([pl.col(f"g{i}") == best for i in range(6)]))
                     .otherwise(0).alias(name),
                     (pl.sum_horizontal([pl.col(f"m{i}") for i in range(6)]) == 0).alias(f"nohit:{name}"))
    d = d.join(h, on="read_id", how="left").with_columns(pl.col(name).fill_null(0))
    methods.append(name)

# combinations: the k-mer setting from the old gene models that is right in the most real reads
def right_share(m):
    return d.filter(pl.col("src") == "real").select((pl.col(m) == pl.lit(2).pow(pl.col("tf")).cast(pl.Int64)).mean()).item()


bz = [m for m in methods if m.startswith("botznik.")]
best_bz = max(bz, key=right_share) if bz else None
combos = {}
if best_bz and "blastx.default" in methods:
    combos[f"blastx.default+{best_bz}"] = ["blastx.default", best_bz]
    combos[f"blastx.default+{best_bz}+stops.longest"] = ["blastx.default", best_bz, "stops.longest"]
    combos[f"stops.one1+blastx.default+{best_bz}"] = ["stops.one1", "blastx.default", best_bz]
if best_bz and "blastx.fast" in methods:
    combos[f"blastx.fast+{best_bz}"] = ["blastx.fast", best_bz]
d = d.with_columns(**{"stops.one1": pl.when(pl.col("n_stopfree") == 1).then(pl.col("stops.one")).otherwise(0)})
for cname, parts in combos.items():
    e = pl.lit(0, dtype=pl.Int64)
    for p in reversed(parts):
        e = pl.when(pl.col(p) > 0).then(pl.col(p)).otherwise(e)
    d = d.with_columns(e.alias(cname))


def read_time(name):
    """(wall s, cpu s, peak GB) from a GNU time -v file; None if missing."""
    p = Path(time_dir) / f"{name}.time"
    if not p.exists():
        return None
    t = p.read_text()
    wall = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", t).group(1).split(":")
    wall = sum(float(x) * 60 ** i for i, x in enumerate(reversed(wall)))
    cpu = float(re.search(r"User time \(seconds\): (\S+)", t).group(1)) + \
        float(re.search(r"System time \(seconds\): (\S+)", t).group(1))
    rss = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", t).group(1)) / 1024 ** 2
    return wall, cpu, rss


def resources(m):
    parts = combos.get(m, [m])
    parts = ["stops" if p.startswith("stops.") else p for p in parts]
    ts = [read_time(p) for p in dict.fromkeys(parts)]
    if any(t is None for t in ts):
        return dict(wall_s=None, cpu_s=None, peak_gb=None)
    return dict(wall_s=round(sum(t[0] for t in ts), 1), cpu_s=round(sum(t[1] for t in ts), 1),
                peak_gb=round(max(t[2] for t in ts), 2))


rows = []
labelled = d.filter(pl.col("src") != "real, frame unknown")
tbit = pl.lit(2).pow(pl.col("tf")).cast(pl.Int64)
for m in methods + list(combos):
    if m == "stops.one1":
        continue
    has_t = (pl.col(m) // tbit) % 2 == 1
    lab = labelled.filter(pl.col("in_half_b")) if m.startswith("newgenome_halfA.") else labelled
    nohit = (pl.col(f"nohit:{m}").mean() if f"nohit:{m}" in d.columns else pl.lit(None, dtype=pl.Float64))
    stat = (lab.group_by("src").agg(
        no_kmer_match=nohit,
        n=pl.len(),
        right=(pl.col(m) == tbit).mean(),
        tied=(has_t & (pl.col(m) != tbit)).mean(),
        wrong=((pl.col(m) > 0) & ~has_t).mean(),
        none=(pl.col(m) == 0).mean(),
        stopfree_0=(pl.col("n_stopfree") == 0).mean(),
        stopfree_1=(pl.col("n_stopfree") == 1).mean(),
        stopfree_2plus=(pl.col("n_stopfree") >= 2).mean(),
        true_frame_has_stop=(pl.col("tstops") > 0).mean()))
    all_real = d.filter(is_sim.not_())
    res = resources(m)
    for r in stat.iter_rows(named=True):
        r = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
        if not m.startswith("stops."):
            r.update(stopfree_0=None, stopfree_1=None, stopfree_2plus=None)
        with_frame = 1 - r["none"]
        rows.append(dict(method=m, reads=r.pop("src"), **r, with_frame=round(with_frame, 4),
                         right_given_frame=round(r["right"] / with_frame, 4) if with_frame else None,
                         all_real_reads_with_frame=round((all_real[m] > 0).mean(), 4),
                         reads_in_run=n_reads, **res,
                         wall_s_per_M=None if res["wall_s"] is None else round(res["wall_s"] / n_reads * 1e6, 1),
                         cpu_s_per_M=None if res["cpu_s"] is None else round(res["cpu_s"] / n_reads * 1e6, 1)))
t = pl.DataFrame(rows).sort("reads", "method")
t.write_csv(out_p, separator="\t")
mk = read_time("makedb")
if mk:
    print(f"DIAMOND makedb of the old gene models: wall {mk[0]:.1f} s, CPU {mk[1]:.1f} s, peak {mk[2]:.2f} GB")
print("best old-gene-model k-mer setting on real reads:", best_bz)
with pl.Config(tbl_rows=200, tbl_cols=30, tbl_width_chars=300):
    print(t.select("method", "reads", "n", "right", "tied", "wrong", "none", "right_given_frame",
                   "all_real_reads_with_frame", "wall_s_per_M", "cpu_s_per_M", "peak_gb"))
