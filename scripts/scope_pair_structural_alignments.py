"""Structurally align sampled SCOPe40 domain pairs with USalign and keep the alignment strings.

The pairs come in four relationship classes so that HP-class conservation can be read
against the SCOPe hierarchy: same family, same superfamily but different family (the
remote homologs), same fold but different superfamily (analogs or unproven homologs), and
different fold within the same class (the non-homolog control). Every alignment is
structure-based, so the aligned columns do not depend on sequence similarity.

Output is one parquet row per pair with both gapped alignment strings, TM-scores
normalised by each chain, and the identity over aligned columns that USalign reports.

Usage:
    python scripts/scope_pair_structural_alignments.py --out data/230_scope40_pair_alignments.parquet
"""

from __future__ import annotations

import argparse
import random
import subprocess
from collections import defaultdict
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import polars as pl

USALIGN = "/Users/olga/anaconda3/envs/usalign/bin/USalign"
PDBSTYLE = Path("/Users/olga/data/scope/pdbstyle-2.08/pdbstyle-2.08")
DOMAINS = Path("/Users/olga/data/scope/results-foldseek-pdb/scope_domains.tsv")


def ent_path(domain_id: str) -> Path:
    # pdbstyle stores d3p8ta_ under p8/, the two characters after the leading 'd'.
    return PDBSTYLE / domain_id[2:4] / f"{domain_id}.ent"


def sample_pairs(members: list[str], n: int, rng: random.Random) -> list[tuple[str, str]]:
    all_pairs = list(combinations(members, 2))
    if len(all_pairs) <= n:
        return all_pairs
    return rng.sample(all_pairs, n)


def sample_cross_pairs(groups: list[list[str]], n: int, rng: random.Random) -> list[tuple[str, str]]:
    """Pairs whose two members sit in different groups of `groups`."""
    if len(groups) < 2:
        return []
    out = set()
    tries = 0
    while len(out) < n and tries < n * 20:
        tries += 1
        a, b = rng.sample(range(len(groups)), 2)
        pair = (rng.choice(groups[a]), rng.choice(groups[b]))
        out.add(tuple(sorted(pair)))
    return sorted(out)


def build_pairs(per_family: int, per_superfamily: int, per_fold: int, n_diff_fold: int, seed: int):
    rng = random.Random(seed)
    dom = pl.read_csv(DOMAINS, separator="\t")
    dom = dom.filter(pl.col("scop_class").is_in(["a", "b", "c", "d", "e", "f", "g"]))
    dom = dom.filter([ent_path(d).exists() for d in dom["domain_id"]])

    by_family = defaultdict(list)
    by_sf = defaultdict(lambda: defaultdict(list))
    by_fold = defaultdict(lambda: defaultdict(list))
    by_class = defaultdict(lambda: defaultdict(list))
    for row in dom.iter_rows(named=True):
        d = row["domain_id"]
        by_family[row["scop_family"]].append(d)
        by_sf[row["scop_superfamily"]][row["scop_family"]].append(d)
        by_fold[row["scop_fold"]][row["scop_superfamily"]].append(d)
        by_class[row["scop_class"]][row["scop_fold"]].append(d)

    pairs = []
    for fam, members in by_family.items():
        pairs += [(a, b, "same_family") for a, b in sample_pairs(members, per_family, rng)]
    for sf, fams in by_sf.items():
        pairs += [
            (a, b, "same_superfamily_diff_family")
            for a, b in sample_cross_pairs(list(fams.values()), per_superfamily, rng)
        ]
    for fold, sfs in by_fold.items():
        pairs += [
            (a, b, "same_fold_diff_superfamily")
            for a, b in sample_cross_pairs(list(sfs.values()), per_fold, rng)
        ]
    for cls, folds in by_class.items():
        n = int(round(n_diff_fold * len(folds) / sum(len(f) for f in by_class.values())))
        pairs += [(a, b, "diff_fold_same_class") for a, b in sample_cross_pairs(list(folds.values()), n, rng)]
    return pairs


def parse_outfmt1(text: str) -> dict | None:
    """USalign -outfmt 1 prints two FASTA-like records with TM-scores in their headers."""
    lines = [l for l in text.splitlines() if l and not l.startswith("#Total")]
    heads = [l for l in lines if l.startswith(">")]
    if len(heads) != 2:
        return None
    seqs = []
    for h in heads:
        i = lines.index(h)
        seqs.append(lines[i + 1])
    stats = {}
    for h in heads:
        for tok in h.split("\t")[1:]:
            k, v = tok.split("=")
            stats.setdefault(k, []).append(float(v))
    ali = [l for l in lines if l.startswith("# Lali")]
    lali = rmsd = seqid_ali = None
    if ali:
        for tok in ali[0].lstrip("# ").split("\t"):
            k, v = tok.split("=")
            if k == "Lali":
                lali = int(v)
            elif k == "RMSD":
                rmsd = float(v)
            elif k == "seqID_ali":
                seqid_ali = float(v)
    return {
        "qaln": seqs[0],
        "taln": seqs[1],
        "q_len": int(stats["L"][0]),
        "t_len": int(stats["L"][1]),
        "tm_q": stats["TM-score"][0],
        "tm_t": stats["TM-score"][1],
        "lali": lali,
        "rmsd": rmsd,
        "seqid_ali": seqid_ali,
    }


def align_one(job: tuple[str, str, str]) -> dict | None:
    q, t, category = job
    try:
        res = subprocess.run(
            [USALIGN, str(ent_path(q)), str(ent_path(t)), "-outfmt", "1"],
            capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        return None
    parsed = parse_outfmt1(res.stdout)
    if parsed is None:
        return None
    return {"query": q, "target": t, "category": category, **parsed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-family", type=int, default=15)
    ap.add_argument("--per-superfamily", type=int, default=40)
    ap.add_argument("--per-fold", type=int, default=25)
    ap.add_argument("--n-diff-fold", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--procs", type=int, default=14)
    args = ap.parse_args()

    jobs = build_pairs(args.per_family, args.per_superfamily, args.per_fold, args.n_diff_fold, args.seed)
    print(f"{len(jobs)} pairs to align", flush=True)
    rows = []
    with Pool(args.procs) as pool:
        for i, r in enumerate(pool.imap_unordered(align_one, jobs, chunksize=20)):
            if r is not None:
                rows.append(r)
            if i % 5000 == 0:
                print(f"  {i} done, {len(rows)} kept", flush=True)
    df = pl.DataFrame(rows)
    df.write_parquet(args.out)
    print(df.group_by("category").len().sort("category"))


if __name__ == "__main__":
    main()
