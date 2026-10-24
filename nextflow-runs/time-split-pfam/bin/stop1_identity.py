#!/usr/bin/env python3
"""Identity of each time-split instance to its closest t0 Swiss-Prot domain of the same
family, on a random 500 instances (notebook 280, Stop 1).

    stop1_identity.py prepare --qs <query-set dir> --work <dir>
        writes <work>/q/<i>.fa (the query's t1 domain) and <work>/lib/<pfam>.fa (every
        Swiss-Prot 2019_08 domain of that family under Pfam 33.1), and <work>/sample.parquet
    (then ssearch36 -m 8CB each query against its family's library; see stop1.sbatch)
    stop1_identity.py collect --qs <dir> --work <dir>
        reads the ssearch36 tables, keeps the top-scoring Swiss-Prot domain per instance,
        and writes identity_sample.csv, identity_bins.csv and five_examples.txt

"Closest" is the Swiss-Prot domain with the highest Smith-Waterman bit score (BLOSUM62,
gap open 11, extend 1). Identity is identical columns / alignment columns, gaps
included, as ssearch36 and BLAST report it. Coordinates are 1-based, end included.
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import polars as pl

SEED = 280
N_SAMPLE = 500
N_EXAMPLES = 5
HP = {"hp_pbotc_1st_ed2": ("ACFILMPVWY", "DEGHKNQRST")}  # kmerseek src/rust/alphabets.rs
M8 = ["query", "target", "pident", "aln_len", "mismatch", "gapopen", "qstart", "qend",
      "sstart", "send", "evalue", "bits", "btop"]
M8_TYPES = {"query": pl.String, "target": pl.String, "pident": pl.Float64,
            "aln_len": pl.Int64, "mismatch": pl.Int64, "gapopen": pl.Int64,
            "qstart": pl.Int64, "qend": pl.Int64, "sstart": pl.Int64, "send": pl.Int64,
            "evalue": pl.Float64, "bits": pl.Float64, "btop": pl.String}
ID_BINS = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100.01]


def hp_string(seq: str, alphabet: str = "hp_pbotc_1st_ed2") -> str:
    h, p = HP[alphabet]
    return "".join("h" if c in h else "p" if c in p else "x" for c in seq)


def prepare(qs: Path, work: Path, n: int = N_SAMPLE, seed: int = SEED) -> None:
    kept = pl.read_parquet(qs / "instances_kept.parquet")
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(kept.height), min(n, kept.height)))
    sample = kept[idx].with_row_index("i")
    seqs = (pl.scan_parquet(qs / "pfamseq_33.1.parquet").select("acc", "seq")
            .join(sample.lazy().select("acc"), on="acc", how="semi").collect())
    sample = sample.join(seqs, on="acc", how="left")
    sample = sample.with_columns(
        query_domain=pl.col("seq").str.slice(pl.col("t1_start") - 1, pl.col("length")))
    bad = sample.filter(pl.col("query_domain").str.len_chars() != pl.col("length"))
    if bad.height:
        raise SystemExit(f"{bad.height} t1 domains run past the end of the t0 sequence")
    sample.drop("seq").write_parquet(work / "sample.parquet")

    sp = pl.read_parquet(qs / "sprot_2019_08.parquet").select(pl.col("acc").alias("sp_acc"), "seq")
    dom = (pl.read_parquet(qs / "sprot_2019_08_pfam33.1_domains.parquet")
           .join(sample.select("pfam_acc").unique(), on="pfam_acc", how="semi")
           .join(sp, on="sp_acc")
           .with_columns(sp_domain=pl.col("seq").str.slice(pl.col("sp_start") - 1,
                                                           pl.col("sp_end") - pl.col("sp_start") + 1)))
    (work / "q").mkdir(parents=True, exist_ok=True)
    (work / "lib").mkdir(parents=True, exist_ok=True)
    for r in sample.iter_rows(named=True):
        (work / "q" / f"{r['i']}.fa").write_text(
            f">{r['acc']}/{r['t1_start']}-{r['t1_end']}\n{r['query_domain']}\n")
    for fam, g in dom.group_by("pfam_acc"):
        (work / "lib" / f"{fam[0]}.fa").write_text("".join(
            f">{a}/{s}-{e}\n{d}\n" for a, s, e, d in
            g.select("sp_acc", "sp_start", "sp_end", "sp_domain").iter_rows()))
    sample.select("i", "pfam_acc").write_csv(work / "jobs.tsv", separator="\t", include_header=False)
    print(f"{sample.height} instances, {dom['pfam_acc'].n_unique()} families, "
          f"{dom.height} Swiss-Prot domains in the libraries")


def btop_alignment(q: str, s: str, qstart: int, sstart: int, btop: str) -> tuple[str, str]:
    """Rebuild the two aligned rows from ssearch36/BLAST BTOP (a number = that many
    identical columns; a letter pair = query letter, subject letter, '-' for a gap)."""
    qa, sa, qi, si = [], [], qstart - 1, sstart - 1
    for num, pair in re.findall(r"(\d+)|([A-Z*\-]{2})", btop):
        if num:
            n = int(num)
            qa.append(q[qi:qi + n]); sa.append(s[si:si + n]); qi += n; si += n
        else:
            a, b = pair
            qa.append(a); sa.append(b)
            qi += a != "-"; si += b != "-"
    return "".join(qa), "".join(sa)


def longest_class_run(qa: str, sa: str) -> tuple[int, int]:
    """Start column and length of the longest gap-free run where both residues fall in
    the same hp_pbotc_1st_ed2 class."""
    best, start, cur = (0, 0), 0, 0
    qh = [hp_string(c) if c != "-" else "-" for c in qa]
    sh = [hp_string(c) if c != "-" else "-" for c in sa]
    for j, (a, b) in enumerate(zip(qh, sh)):
        if a != "-" and b != "-" and a == b and a != "x":
            if cur == 0:
                start = j
            cur += 1
            if cur > best[1]:
                best = (start, cur)
        else:
            cur = 0
    return best


def render(r: dict, qa: str, sa: str) -> str:
    match = "".join("|" if a == b and a != "-" else " " for a, b in zip(qa, sa))
    qh = "".join(hp_string(c) if c != "-" else "-" for c in qa)
    sh = "".join(hp_string(c) if c != "-" else "-" for c in sa)
    hmatch = "".join("|" if a == b and a != "-" else " " for a, b in zip(qh, sh))
    st, ln = longest_class_run(qa, sa)
    mark = " " * st + "^" * ln
    n_id = match.count("|")
    lines = [
        f"{r['acc']} ({r['description']}, {r['kingdom']}) t1 {r['pfam_acc']} {r['pfam_id']} "
        f"[{r['pfam_type']}] residues {r['t1_start']}-{r['t1_end']} ({r['length']} aa)",
        f"closest t0 Swiss-Prot domain: {r['target']}   bits {r['bits']:.1f}",
        f"query  {r['qstart_abs']:>5} {qa} {r['qend_abs']}",
        f"             {match}",
        f"Sprot  {r['sstart_abs']:>5} {sa} {r['send_abs']}",
        f"{n_id} of {len(qa)} aligned columns identical ({100 * n_id / len(qa):.1f}%)",
        "hp_pbotc_1st_ed2 classes (h = ACFILMPVWY, p = DEGHKNQRST):",
        f"query        {qh}",
        f"             {hmatch}",
        f"Sprot        {sh}",
        f"             {mark}",
        f"{hmatch.count('|')} of {len(qa)} columns in the same class; longest gap-free "
        f"same-class run: {ln} columns (^)",
    ]
    return "\n".join(lines)


def collect(qs: Path, work: Path, examples: bool = True) -> None:
    sample = pl.read_parquet(work / "sample.parquet")
    frames = []
    for f in sorted((work / "m8").glob("*.m8")):
        if f.stat().st_size:
            frames.append(pl.read_csv(f, separator="\t", has_header=False, new_columns=M8,
                                      comment_prefix="#", schema_overrides=M8_TYPES)
                          .with_columns(i=pl.lit(int(f.stem), dtype=pl.UInt32)))
    hits = pl.concat(frames)
    best = (hits.sort(["i", "bits", "pident", "target"], descending=[False, True, True, False])
            .unique("i", keep="first", maintain_order=True))
    res = sample.join(best, on="i", how="left")
    n_none = res.filter(pl.col("target").is_null()).height
    # A local alignment can be a short, highly identical patch. Report how much of the
    # query's domain it covers, and identity over the whole domain as a second reading.
    res = res.with_columns(
        query_coverage=(pl.col("qend") - pl.col("qstart") + 1) / pl.col("length"),
        identity_over_domain=100 * (pl.col("pident") * pl.col("aln_len") / 100).round()
        / pl.col("length"))
    res = res.with_columns(
        identity_bin=pl.col("pident").cut(ID_BINS[1:-1], left_closed=True,
                                          labels=[f"{a}-{b if b < 100 else 100}%"
                                                  for a, b in zip(ID_BINS, ID_BINS[1:])]))
    res.drop("query_domain").write_csv(work / "identity_sample.csv")
    bins = (res.group_by("identity_bin").agg(n=pl.len()).sort("identity_bin")
            .with_columns(share=pl.col("n") / pl.col("n").sum()))
    bins.write_csv(work / "identity_bins.csv")
    by_type = (res.group_by("pfam_type").agg(
        n=pl.len(), median_identity=pl.col("pident").median(),
        median_coverage=pl.col("query_coverage").median(),
        under_30=(pl.col("pident") < 30).mean()).sort("pfam_type"))
    by_type.write_csv(work / "identity_by_type.csv")
    print(bins, by_type, f"instances with no ssearch36 hit at all: {n_none}", sep="\n")
    print(f"median query coverage {res['query_coverage'].median():.2f}, median identity over "
          f"the whole query domain {res['identity_over_domain'].median():.1f}%")
    print(f"median identity {res['pident'].median():.1f}%, "
          f"share under 30%: {(res['pident'] < 30).mean():.3f}")

    if not examples:
        return
    # Five examples drawn at random from the 500, rendered as residues.
    rng = random.Random(SEED + 1)
    have = res.filter(pl.col("target").is_not_null())
    ex = have[sorted(rng.sample(range(have.height), N_EXAMPLES))]
    sp = pl.read_parquet(qs / "sprot_2019_08.parquet").select(pl.col("acc").alias("sp_acc"), "seq")
    out = []
    for r in ex.iter_rows(named=True):
        sp_acc, rng_s = r["target"].split("/")
        s0, _ = (int(x) for x in rng_s.split("-"))
        sdom_full = sp.filter(pl.col("sp_acc") == sp_acc)["seq"][0]
        sdom = sdom_full[s0 - 1:]
        qa, sa = btop_alignment(r["query_domain"], sdom, r["qstart"], r["sstart"], r["btop"])
        r = dict(r, qstart_abs=r["t1_start"] + r["qstart"] - 1, qend_abs=r["t1_start"] + r["qend"] - 1,
                 sstart_abs=s0 + r["sstart"] - 1, send_abs=s0 + r["send"] - 1)
        # Check: removing gaps gives back the residues at the stated coordinates.
        assert qa.replace("-", "") == r["query_domain"][r["qstart"] - 1:r["qend"]]
        assert sa.replace("-", "") == sdom_full[r["sstart_abs"] - 1:r["send_abs"]]
        out.append(render(r, qa, sa))
    (work / "five_examples.txt").write_text("\n\n".join(out) + "\n")
    print("\n\n".join(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prepare", "collect"])
    ap.add_argument("--qs", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--n", type=int, default=N_SAMPLE)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-examples", action="store_true")
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    if a.mode == "prepare":
        prepare(a.qs, a.work, a.n, a.seed)
    else:
        collect(a.qs, a.work, examples=not a.no_examples)


if __name__ == "__main__":
    main()
