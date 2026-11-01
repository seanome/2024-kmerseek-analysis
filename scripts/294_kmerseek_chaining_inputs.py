"""Write the inputs of notebook 294's kmerseek chaining test (section 4f, part 2).

The report half's true pairs at 20-30% identity (the split of section 2, seed 292) become
one query FASTA and one target database. The shuffled database holds every target once,
shuffled with notebook 232's first shuffle of the pair it first appears in (the same
permutation score_pair applies to the HP classes, applied to the residues; X and other
letters outside the two HP classes stay where they are, as in 232). Sequences are the Pfam
seed rows of notebook 230's alignments with the gaps removed.

Writes to --out (default /Users/olga/data/pfam/294_chaining/):

* queries.fa: one record per distinct query, named by its Pfam seed name;
* targets.fa: one record per distinct target;
* targets_shuffled.fa: the same targets, shuffled, named ``shuffled_<name>``;
* pairs.tsv: pair_index, family, query, target, seqid_ali of every pair tested;
* inputs.json: counts, residues, the HP 8-mer load used to size the job, and sha256 of
  each file.

Checks, all on the real data: the shuffled residues rescore to 232's stored null B2 score
of every pair, and a name used twice always carries the same sequence.

Usage:
    python scripts/294_kmerseek_chaining_inputs.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import hp_conservation_utils as hc  # noqa: E402
import linked_pairs_utils as lp  # noqa: E402

PFAM = "/Users/olga/data/pfam/230_pfam_seed_pair_alignments.parquet"
SCORES = "/Users/olga/data/pfam/294_linked_pairs_scores.parquet"
N_PFAM, MIN_ALIGNED = 60_000, 50  # notebook 232's sample
K = 8


def hp_kmer_load(seqs: list[str], k: int = K) -> tuple[float, int]:
    """Spectrum load sum(occ^2) / sum(occ) over distinct HP k-mers (the predictor of
    kmerseek search memory in the project's memory notes), and the number of k-mer
    positions. A k-mer containing a letter outside the two classes is skipped."""
    tab = lp.TABLES[lp.ALPHABET]
    occ: Counter = Counter()
    for s in seqs:
        c = tab[np.frombuffer(s.encode(), np.uint8)]
        for i in range(len(c) - k + 1):
            w = c[i : i + k]
            if (w != lp.GAP).all():
                occ[w.tobytes()] += 1
    v = np.array(list(occ.values()), dtype=float)
    return float((v**2).sum() / v.sum()), int(v.sum())


def write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    with open(path, "w") as fh:
        for name, seq in records:
            fh.write(f">{name}\n")
            for i in range(0, len(seq), 60):
                fh.write(seq[i : i + 60] + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/Users/olga/data/pfam/294_chaining")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pf = (
        pl.read_parquet(PFAM)
        .filter(pl.col("lali") >= MIN_ALIGNED)
        .sample(n=N_PFAM, seed=0, shuffle=True)
        .with_row_index("pair_index")
        .with_columns(pl.col("pair_index").cast(pl.Int64))
    )
    pf = hc.add_identity_bin(pf)
    select = lp.select_half_ids(pf["pair_index"].to_numpy())
    rep = pf.filter(
        ~pl.col("pair_index").is_in(select.tolist())
        & (pl.col("identity_bin") == "20-30%")
    ).sort("pair_index")
    print(f"report half, 20-30% identity: {rep.height:_} pairs")

    stored = (
        pl.read_parquet(SCORES)
        .filter(pl.col("kind") == "null0")
        .select(pl.col("pair_index").cast(pl.Int64), "B2_ext")
    )
    rep = rep.join(stored, on="pair_index", how="left")
    assert rep["B2_ext"].null_count() == 0

    queries: dict[str, str] = {}
    targets: dict[str, str] = {}
    shuffled: dict[str, str] = {}
    n_b2_same = 0
    for r in rep.iter_rows(named=True):
        q = r["qaln"].replace("-", "")
        t = r["taln"].replace("-", "")
        assert queries.setdefault(r["query"], q) == q
        assert targets.setdefault(r["target"], t) == t
        shuf_row = lp.shuffled_target_rows(r["taln"], r["pair_index"], 2)[0]
        (sc,) = lp.score_pair(r["qaln"], shuf_row, np.random.default_rng(0), 0)
        n_b2_same += sc["B2_ext"] == r["B2_ext"]
        shuffled.setdefault(r["target"], shuf_row.replace("-", ""))
    assert n_b2_same == rep.height, f"only {n_b2_same} of {rep.height} shuffles rescore to 232's"
    print(f"shuffled targets rescore to 232's stored null B2 on all {n_b2_same:_} pairs")
    for name, s in shuffled.items():
        assert sorted(s) == sorted(targets[name])

    write_fasta(out / "queries.fa", sorted(queries.items()))
    write_fasta(out / "targets.fa", sorted(targets.items()))
    write_fasta(
        out / "targets_shuffled.fa", sorted((f"shuffled_{n}", s) for n, s in shuffled.items())
    )
    rep.select("pair_index", "family", "query", "target", "seqid_ali").write_csv(
        out / "pairs.tsv", separator="\t"
    )

    load_t, pos_t = hp_kmer_load(list(targets.values()))
    load_s, pos_s = hp_kmer_load(list(shuffled.values()))
    info = {
        "n_pairs": rep.height,
        "n_queries": len(queries),
        "n_targets": len(targets),
        "n_target_residues": sum(map(len, targets.values())),
        "n_query_residues": sum(map(len, queries.values())),
        "hp_8mer_positions_targets": pos_t,
        "hp_8mer_load_targets": load_t,
        "hp_8mer_load_shuffled": load_s,
        "sha256": {
            f: hashlib.sha256((out / f).read_bytes()).hexdigest()
            for f in ["queries.fa", "targets.fa", "targets_shuffled.fa", "pairs.tsv"]
        },
    }
    (out / "inputs.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
