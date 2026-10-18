#!/usr/bin/env python3
"""How many ice-binding proteins of other lineages share linked word pairs with each query.

Notebook 298, stage 1. The same count as notebook 295 section 9
(scripts/convergence_kmer_reach_ceiling.py), with the exact k-mer replaced by a key made of
two short exact words and the spacing between them (notebooks/strobemer_reach_utils.py).
No index is built and kmerseek is not run.

  --gate   run section 9's exact k-mer settings through this code and write the table in
           section 9's format, to compare byte for byte with kmer_reach_ceiling.tsv.
  --nullc  write the null C draws (shuffled within each phase of the repeat period).
  --run    count every linked-pair setting for the 6 real queries and their 60 null draws.

Inputs: data/convergence_afp/{truth_set,queries}.tsv, Swiss-Prot 2026_03, and the
pipeline's own null A and B draws (data/convergence_afp/298_inputs/draws.tsv, copied from
the convergence-afp run's results/inputs on Sherlock).
"""

import os

# One BLAS thread per worker process; set before numpy is imported.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import gzip  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "notebooks"))
import convergence_kmer_reach_ceiling as ceiling  # noqa: E402
import strobemer_reach_utils as sr  # noqa: E402

D = REPO / "data/convergence_afp"
INPUTS = D / "298_inputs"
SEED_C = 20261018
N_DRAWS = 20
MIN_SHARED = [1, 2, 3]

# Section 9's alphabets, read by name from its script, plus threonine against the rest.
ALPHABETS = list(ceiling.ALPHABETS) + ["thr2"]
WORD_LENGTHS = {a: [3, 4, 6, 8, 10] for a in ALPHABETS}
WORD_LENGTHS["dayhoff6"] = [3, 4, 5, 6]
WORD_LENGTHS["protein20"] = [2, 3, 4]


def check_alphabets():
    """Section 9's partitions and notebook 294's must be the same groups of residues."""
    for name, m in ceiling.ALPHABETS.items():
        groups = {}
        for res, cls in m.items():
            groups.setdefault(cls, set()).add(res)
        want = sorted(sorted(g) for g in groups.values())
        got = sorted(sorted(c) for c in sr.CLUSTERS[name])
        assert want == got, name


def load_inputs():
    sprot = ceiling.read_fasta(ceiling.SPROT)
    truth, queries, exclude = ceiling.load_afp(sprot)
    pool = sorted(a for a in sprot if a not in exclude)
    background = random.Random(0).sample(pool, ceiling.N_BACKGROUND)
    return sprot, truth, queries, background


def write_background(background):
    out = D / "298_random_swissprot_5000.txt"
    out.write_text("".join(f"{a}\n" for a in background))
    return out


def gate(sprot, truth, queries, background):
    """Section 9's table, recomputed with sr.all_keys and sr.shared_counts."""
    entries = truth["Entry"].to_list() + background
    seqs = [sprot[e] for e in entries]
    lineage = dict(zip(truth["Entry"], truth["lineage"]))
    n_truth = truth.height
    qrows = list(queries.iter_rows(named=True))
    rows = []
    for alpha in ceiling.ALPHABETS:
        per_k = {}
        for k in ceiling.KS:
            s_ids, keys = sr.all_keys(
                seqs + [q["Sequence"] for q in qrows], alpha, k, "kmer"
            )
            S, n_keys = sr.shared_counts(len(seqs), len(qrows), s_ids, keys)
            per_k[k] = (S, n_keys)
        for qi, q in enumerate(qrows):
            cross = [i for i in range(n_truth) if lineage[entries[i]] != q["lineage"]]
            for k in ceiling.KS:
                S, n_keys = per_k[k]
                if n_keys[len(seqs) + qi] == 0:
                    break
                n_cross = int((S[cross, qi] >= ceiling.MIN_SHARED).sum())
                n_bg = int((S[n_truth:, qi] >= ceiling.MIN_SHARED).sum())
                rows.append(
                    {
                        "query": q["Entry"],
                        "query_lineage": q["lineage"],
                        "alphabet": alpha,
                        "ksize": k,
                        "n_cross_truth": len(cross),
                        "n_cross_reachable": n_cross,
                        "frac_cross_reachable": n_cross / len(cross),
                        "frac_background_reachable": n_bg / len(background),
                    }
                )
    # section 9 loops query, then alphabet, then k
    order = {q["Entry"]: i for i, q in enumerate(qrows)}
    al = {a: i for i, a in enumerate(ceiling.ALPHABETS)}
    rows.sort(key=lambda r: (order[r["query"]], al[r["alphabet"]], r["ksize"]))
    return pl.DataFrame(rows)


def make_null_c(sprot):
    """20 draws per real query, residues shuffled among positions of one repeat phase.

    The period is the pipeline's unit_length for that query (draws.tsv): the published
    repeat length where assets/known_motifs.tsv has one, otherwise the HP-autocorrelation
    peak that bin/prep_inputs.py found when it made null B."""
    draws = pl.read_csv(INPUTS / "draws.tsv", separator="\t")
    real = draws.filter(pl.col("kind") == "real")
    rng = np.random.default_rng(SEED_C)
    rows = []
    for r in real.iter_rows(named=True):
        assert r["sequence"] == sprot[r["accession"]], r["accession"]
        for d in range(1, N_DRAWS + 1):
            s = sr.null_c(r["sequence"], r["unit_length"], rng)
            assert sorted(s) == sorted(r["sequence"])
            rows.append(
                {
                    "query_id": f"{r['accession']}__nullC_{d:02d}",
                    "accession": r["accession"],
                    "lineage": r["lineage"],
                    "kind": "nullC",
                    "draw": d,
                    "unit_length": r["unit_length"],
                    "unit_source": r["unit_source"],
                    "permutation": "",
                    "sequence": s,
                    "frac_positions_changed": float(
                        np.mean(
                            np.frombuffer(s.encode(), np.uint8)
                            != np.frombuffer(r["sequence"].encode(), np.uint8)
                        )
                    ),
                }
            )
    out = pl.DataFrame(rows)
    out.write_csv(INPUTS / "null_c_draws.tsv", separator="\t")
    with open(INPUTS / "null_c_draws.fasta", "w") as fh:
        for r in out.iter_rows(named=True):
            fh.write(f">{r['query_id']}\n{r['sequence']}\n")
    return out


def query_table():
    """366 queries: per real query, itself, then null A, B and C draws 1..20."""
    ab = pl.read_csv(INPUTS / "draws.tsv", separator="\t")
    c = pl.read_csv(INPUTS / "null_c_draws.tsv", separator="\t").drop(
        "frac_positions_changed"
    )
    q = pl.concat([ab, c.select(ab.columns)])
    kind_order = {"real": 0, "nullA": 1, "nullB": 2, "nullC": 3}
    acc_order = {
        a: i for i, a in enumerate(ab["accession"].unique(maintain_order=True))
    }
    return (
        q.with_columns(
            _a=pl.col("accession").replace_strict(acc_order),
            _k=pl.col("kind").replace_strict(kind_order),
        )
        .sort("_a", "_k", "draw")
        .drop("_a", "_k")
    )


_G = {}


def _count_one(task):
    alphabet, a = task
    g = _G
    seqs = g["entry_seqs"] + g["query_seqs"]
    n_e, n_q = len(g["entry_seqs"]), len(g["query_seqs"])
    out = []
    for family in sr.FAMILIES:
        s_ids, keys = sr.all_keys(seqs, alphabet, a, family)
        S, n_keys = sr.shared_counts(n_e, n_q, s_ids, keys)
        rand = S[g["n_truth"] :, :]
        mean_rand = rand.mean(axis=0)
        for qi, q in enumerate(g["queries"]):
            cross = g["cross"][q["lineage"]]
            col = S[cross, qi]
            two_keys = int((n_keys[cross] >= 2).sum())
            for m in MIN_SHARED:
                tr = int((col >= m).sum())
                rr = int((rand[:, qi] >= m).sum())
                out.append(
                    (
                        q["accession"],
                        q["lineage"],
                        q["kind"],
                        q["draw"],
                        alphabet,
                        a,
                        family,
                        m,
                        tr,
                        len(cross),
                        tr / len(cross),
                        two_keys,
                        rr,
                        rr / rand.shape[0],
                        int(n_keys[n_e + qi]),
                        float(mean_rand[qi]),
                    )
                )
    return out


COLUMNS = [
    "query",
    "lineage",
    "kind",
    "draw",
    "alphabet",
    "word_length",
    "family",
    "min_shared",
    "targets_reached",
    "targets_total",
    "reach",
    "targets_with_two_keys",
    "random_reached",
    "random_share",
    "query_keys",
    "mean_shared_keys_per_random_entry",
]


def run(sprot, truth, background, workers):
    q = query_table()
    entries = truth["Entry"].to_list() + background
    lineage = truth["lineage"].to_list()
    _G.update(
        entry_seqs=[sprot[e] for e in entries],
        query_seqs=q["sequence"].to_list(),
        queries=q.select("accession", "lineage", "kind", "draw").to_dicts(),
        n_truth=truth.height,
        cross={
            lin: np.array([i for i, L in enumerate(lineage) if L != lin])
            for lin in set(q["lineage"])
        },
    )
    tasks = [(al, a) for al in ALPHABETS for a in WORD_LENGTHS[al]]
    print(
        f"{len(tasks)} alphabet/word-length pairs x {len(sr.FAMILIES)} families; "
        f"{q.height} queries; {len(entries)} entries",
        file=sys.stderr,
    )
    rows = []
    with ProcessPoolExecutor(workers, mp_context=get_context("fork")) as ex:
        for i, r in enumerate(ex.map(_count_one, tasks)):
            rows += r
            print(
                f"  done {i + 1}/{len(tasks)} {tasks[i]}", file=sys.stderr, flush=True
            )
    out = pl.DataFrame(rows, schema=COLUMNS, orient="row")
    path = D / "298_strobemer_reach.tsv.gz"
    with gzip.open(path, "wb") as fh:
        out.write_csv(fh, separator="\t")
    print(f"wrote {path} ({out.height} rows)", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--nullc", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--gate-out", type=Path, default=D / "298_gate_kmer_reach.tsv")
    args = ap.parse_args()
    check_alphabets()
    sprot, truth, queries, background = load_inputs()
    print(f"background list: {write_background(background)}", file=sys.stderr)
    if args.gate:
        g = gate(sprot, truth, queries, background)
        g.write_csv(args.gate_out, separator="\t")
        same = args.gate_out.read_bytes() == (D / "kmer_reach_ceiling.tsv").read_bytes()
        print(f"gate 1, byte-identical to kmer_reach_ceiling.tsv: {same}")
        if not same:
            sys.exit(1)
    if args.nullc:
        c = make_null_c(sprot)
        print(
            c.group_by("accession", "unit_length", maintain_order=True).agg(
                pl.col("frac_positions_changed").median().alias("median_changed"),
                (pl.col("frac_positions_changed") == 0).sum().alias("n_unchanged"),
            )
        )
    if args.run:
        run(sprot, truth, background, args.workers)


if __name__ == "__main__":
    main()
