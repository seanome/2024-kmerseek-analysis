"""Score notebook 232's Pfam seed pairs with linked HP word pairs (notebook 294).

Takes the same 60,000 Pfam seed pairs notebook 232 scored (same filter, same polars sample
seed, same per-pair rng seed, so the two shuffled targets per pair are 232's), and writes
one row per comparison (real pair or shuffled target) with:

* B1_ext, B2_ext: 232's seed-10 and seed-8 scores with ungapped extension, penalty 2,
  recomputed here (Gate 1 compares B1_ext and B2_ext with 232's stored scores);
* seed<a>_count: shared exact a-mers on the alignment's diagonal (no extension);
* <setting>_ext and <setting>_count for every linked-pair setting in
  notebooks/linked_pairs_utils.py.

With --designs it writes section 4d's extra scores instead (lp.score_pair_designs: one
k-letter seed for k in 8-16, anchor pairs with the gap stored exactly or in 7-residue bins,
and the longest run of matching HP classes), on the same pairs and the same shuffles.

With --unrelated-null it writes section 4f's second null instead: for each pair, two null
targets made from Pfam seed sequences of families in a different Pfam clan (a family with
no clan is its own clan), taken from the same alignment table. Each is cut to the true
target's number of H/P residues at a random start and placed in the true target's gap
pattern; draws come from one default_rng(292) in pair order. Each is scored with
lp.score_pair as the real target would be. A pair whose target is longer than every
other-clan candidate keeps a row with empty scores and the reason in ``skipped``.

Usage:
    python scripts/linked_pairs_on_alignments.py \
        --out /Users/olga/data/pfam/294_linked_pairs_scores.parquet
    python scripts/linked_pairs_on_alignments.py --designs \
        --out /Users/olga/data/pfam/294_gap_designs_scores.parquet
    python scripts/linked_pairs_on_alignments.py --unrelated-null \
        --out /Users/olga/data/pfam/294_unrelated_protein_null_scores.parquet
"""

from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import linked_pairs_utils as lp  # noqa: E402

PFAM = "/Users/olga/data/pfam/230_pfam_seed_pair_alignments.parquet"
PFAM_SEED = "/Users/olga/data/pfam/Pfam-A.seed.gz"  # for the clan of each family
UNRELATED_COLS = ["B1_ext", "B2_ext", "B3_ext", "all_a6_g32_none_ext", "rand_a6_g32_none_ext"]
N_PFAM = 60_000  # notebook 232's sample size
MIN_ALIGNED = 50  # notebook 232's lali filter
N_SHUFFLE = 2  # notebook 232's shuffles per pair


def one(job):
    keys, qaln, taln, seed, designs = job
    rng = np.random.default_rng(seed)
    fn = lp.score_pair_designs if designs else lp.score_pair
    return [{**keys, **r} for r in fn(qaln, taln, rng, N_SHUFFLE)]


def one_unrelated(job):
    keys, qaln, nulls = job
    rows = []
    for kind, info, taln in nulls:
        row = {**keys, "kind": kind, **info}
        if taln is None:
            row.update({c: None for c in UNRELATED_COLS})
        else:
            # n_shuffle=0: score_pair draws nothing from the rng, so its seed is unused
            (r,) = lp.score_pair(qaln, taln, np.random.default_rng(0), 0)
            row.update({c: r[c] for c in UNRELATED_COLS})
        rows.append(row)
    return rows


def unrelated_jobs(pf):
    tab = lp.TABLES[lp.ALPHABET]
    clans = lp.pfam_clans(PFAM_SEED)
    all_aln = pl.read_parquet(PFAM)
    names, seqs, cclans, clens = lp.unrelated_candidates(all_aln, clans)
    print(f"{names.size:_} candidate seed sequences from {all_aln.height:_} pairs; "
          f"{np.unique(cclans).size:_} clans (a family with no clan is its own)", flush=True)
    rng = np.random.default_rng(lp.SPLIT_SEED)
    jobs = []
    for i, r in enumerate(pf.iter_rows(named=True)):
        n_res = int((tab[np.frombuffer(r["taln"].encode(), np.uint8)] != lp.GAP).sum())
        nulls = []
        for k in range(N_SHUFFLE):
            j, off = lp.draw_unrelated(rng, n_res, clans[r["family"]], cclans, clens)
            if j is None:
                nulls.append((f"unrel{k}", {"skipped": off, "source": None, "source_clan": None,
                                            "offset": None, "n_residues": n_res}, None))
                continue
            taln = lp.place_in_gaps(r["taln"], seqs[j][off:off + n_res])
            nulls.append((f"unrel{k}", {"skipped": None, "source": names[j], "source_clan": cclans[j],
                                        "offset": off, "n_residues": n_res}, taln))
        jobs.append(({"pair_index": i, "family": r["family"], "family_clan": clans[r["family"]],
                      "query": r["query"], "target": r["target"]}, r["qaln"], nulls))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument(
        "--limit", type=int, default=None, help="first N pairs only (testing)"
    )
    ap.add_argument(
        "--designs",
        action="store_true",
        help="section 4d's extra scores (lp.score_pair_designs) instead",
    )
    ap.add_argument(
        "--unrelated-null",
        action="store_true",
        help="section 4f's null targets cut from proteins of another Pfam clan instead",
    )
    args = ap.parse_args()

    pf = (
        pl.read_parquet(PFAM)
        .filter(pl.col("lali") >= MIN_ALIGNED)
        .sample(n=N_PFAM, seed=0, shuffle=True)
    )
    if args.unrelated_null:
        jobs = unrelated_jobs(pf)
        if args.limit:
            jobs = jobs[: args.limit]
        t0 = time.time()
        rows = []
        with Pool(args.procs) as pool:
            for r in pool.imap(one_unrelated, jobs, chunksize=20):
                rows += r
        out = pl.DataFrame(rows, infer_schema_length=None)
        out.write_parquet(args.out)
        print(out.shape, f"{time.time() - t0:.0f} s")
        return
    jobs = [
        (
            {
                "pair_index": i,
                "family": r["family"],
                "query": r["query"],
                "target": r["target"],
                "seqid_ali": r["seqid_ali"],
                "q_len": r["q_len"],
                "t_len": r["t_len"],
            },
            r["qaln"],
            r["taln"],
            i,  # 232 seeded each pair's rng with its row index; Pfam rows came first
            args.designs,
        )
        for i, r in enumerate(pf.iter_rows(named=True))
    ]
    if args.limit:
        jobs = jobs[: args.limit]
    t0 = time.time()
    rows = []
    with Pool(args.procs) as pool:
        for i, r in enumerate(pool.imap(one, jobs, chunksize=20)):
            rows += r
            if i % 5_000 == 0:
                print(f"  {i:_} pairs  {time.time() - t0:.0f} s", flush=True)
    out = pl.DataFrame(rows)
    out.write_parquet(args.out)
    print(out.shape, f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
