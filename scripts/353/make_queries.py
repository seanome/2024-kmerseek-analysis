#!/usr/bin/env python3
"""Self-retrieval and shuffle queries for notebook 353 (design in notebooks/353_prereg.md).

For each chosen record: one window of each length in LENGTHS at a random offset; each
window at every substitution level (within-class 0-3 x cross-class 0-2) under the given
alphabet's classes; and shuffles of the unchanged window.

    make_queries.py --records records.tsv --split dev --alphabet dayhoff6 \
        --n-records 1000 --comp-shuffles 0 --dip-shuffles 2 --out-prefix dev.dayhoff6

Writes <prefix>.fasta (id = query_id) and <prefix>.tsv:
  query_id, record_id, kind (real / composition / dipeptide), length, offset (window start
  in the record, 0-based), n_within, n_cross, sub_positions (0-based in the window,
  ';'-joined, within first), shuffle_index, frac_hydrophobic (of the unchanged window),
  window (the unchanged window), sequence (the query).
and <prefix>.counts.tsv: windows and queries made, and what was skipped and why.

Every random draw has its own seed made from the record id, so the offsets are the same
for every alphabet and a rerun gives the same queries. Standard library only (runs in the
kmerseek image); decoys.py comes from nextflow-runs/shared (PR 138).
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "nextflow-runs" / "shared"))

from alphabets import CANONICAL, HYDROPHOBIC, class_of  # noqa: E402
from decoys import dipeptide_shuffle  # noqa: E402

LENGTHS = (15, 30, 53)
N_WITHIN = (0, 1, 2, 3)
N_CROSS = (0, 1, 2)
SEED = 353


def substitute(
    window: str,
    n_within: int,
    n_cross: int,
    classes: dict[str, str],
    rng: random.Random,
):
    """The window with n_within within-class and n_cross cross-class substitutions at
    distinct random positions, or None when the window has too few eligible residues."""
    within_ok = [
        i for i, aa in enumerate(window) if aa in classes and len(classes[aa]) > 1
    ]
    if len(within_ok) < n_within:
        return None
    within_pos = rng.sample(within_ok, n_within)
    cross_ok = [
        i for i, aa in enumerate(window) if aa in classes and i not in within_pos
    ]
    if len(cross_ok) < n_cross:
        return None
    cross_pos = rng.sample(cross_ok, n_cross)
    seq = list(window)
    for i in within_pos:
        seq[i] = rng.choice([aa for aa in classes[window[i]] if aa != window[i]])
    for i in cross_pos:
        seq[i] = rng.choice([aa for aa in CANONICAL if aa not in classes[window[i]]])
    return "".join(seq), sorted(within_pos) + sorted(cross_pos)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--split", choices=("dev", "held_out"), required=True)
    ap.add_argument("--alphabet", required=True)
    ap.add_argument(
        "--n-records", type=int, default=0, help="0 = every record of the split"
    )
    ap.add_argument("--comp-shuffles", type=int, default=0)
    ap.add_argument("--dip-shuffles", type=int, default=0)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    classes = class_of(args.alphabet)
    with open(args.records) as f:
        records = [
            r for r in csv.DictReader(f, delimiter="\t") if r["split"] == args.split
        ]
    records.sort(key=lambda r: r["record_id"])
    if args.n_records and args.n_records < len(records):
        records = sorted(
            random.Random(SEED).sample(records, args.n_records),
            key=lambda r: r["record_id"],
        )

    counts = {
        "records": len(records),
        "windows": 0,
        "record_shorter_than_window": 0,
        "real_queries": 0,
        "real_skipped_too_few_eligible": 0,
        "composition_queries": 0,
        "dipeptide_queries": 0,
        "shuffles_identical_to_window_dropped": 0,
    }
    cols = [
        "query_id",
        "record_id",
        "kind",
        "length",
        "offset",
        "n_within",
        "n_cross",
        "sub_positions",
        "shuffle_index",
        "frac_hydrophobic",
        "window",
        "sequence",
    ]
    n = 0
    with open(f"{args.out_prefix}.fasta", "w") as fa, open(
        f"{args.out_prefix}.tsv", "w", newline=""
    ) as ft:
        w = csv.writer(ft, delimiter="\t", lineterminator="\n")
        w.writerow(cols)

        def emit(rec, kind, L, off, nw, nc, pos, si, frac, window, seq):
            nonlocal n
            qid = f"q{n:07d}"
            n += 1
            fa.write(f">{qid}\n{seq}\n")
            w.writerow(
                [
                    qid,
                    rec["record_id"],
                    kind,
                    L,
                    off,
                    nw,
                    nc,
                    ";".join(map(str, pos)),
                    si,
                    f"{frac:.4f}",
                    window,
                    seq,
                ]
            )

        for rec in records:
            seq = rec["sequence"]
            off_rng = random.Random(f"{SEED}:{rec['record_id']}:offsets")
            for L in LENGTHS:
                if len(seq) < L:
                    counts["record_shorter_than_window"] += 1
                    continue
                off = off_rng.randint(0, len(seq) - L)
                window = seq[off : off + L]
                counts["windows"] += 1
                frac = sum(aa in HYDROPHOBIC for aa in window) / L
                for nw in N_WITHIN:
                    for nc in N_CROSS:
                        rng = random.Random(
                            f"{SEED}:{rec['record_id']}:{L}:{nw}:{nc}:{args.alphabet}"
                        )
                        got = substitute(window, nw, nc, classes, rng)
                        if got is None:
                            counts["real_skipped_too_few_eligible"] += 1
                            continue
                        counts["real_queries"] += 1
                        emit(
                            rec,
                            "real",
                            L,
                            off,
                            nw,
                            nc,
                            got[1],
                            "",
                            frac,
                            window,
                            got[0],
                        )
                for kind, n_shuf in (
                    ("composition", args.comp_shuffles),
                    ("dipeptide", args.dip_shuffles),
                ):
                    for si in range(n_shuf):
                        rng = random.Random(
                            f"{SEED}:{rec['record_id']}:{L}:{kind}:{si}"
                        )
                        if kind == "composition":
                            s = list(window)
                            rng.shuffle(s)
                            s = "".join(s)
                        else:
                            s = dipeptide_shuffle(window, rng)
                        if s == window:
                            counts["shuffles_identical_to_window_dropped"] += 1
                            continue
                        counts[f"{kind}_queries"] += 1
                        emit(rec, kind, L, off, 0, 0, [], si, frac, window, s)
    with open(f"{args.out_prefix}.counts.tsv", "w") as f:
        f.write("what\tn\n")
        for k, v in counts.items():
            f.write(f"{k}\t{v}\n")
    print(counts, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
