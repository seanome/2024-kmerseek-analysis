#!/usr/bin/env python3
"""Write the reference and the query set every method of the convergence test searches.

Reference: reviewed Swiss-Prot, one release, with each header cut to the bare accession.
phmmer, MMseqs2, Foldseek and kmerseek each report a different part of a UniProt
header, so one rewrite here gives every method the same target key.

Queries: one real query per lineage (data/convergence_afp/queries.tsv), plus two kinds
of null draw per real query, made by rearranging the query's own residues:

  null A  the residues shuffled. Keeps the composition, destroys both the order of
          residues and any repeat period.
  null B  keeps composition and period. The query is cut into consecutive units of the
          repeat length; one random reordering of the positions inside a unit is drawn
          and applied to every whole unit, so a repeat stays a repeat but its word
          changes. A trailing part-unit is left in place. The repeat length is the one in
          the published consensus (assets/known_motifs.tsv) when the lineage has one,
          otherwise the lag, from 2 to 30, with the highest autocorrelation of the
          query's two-letter HP string (hp_pbotc_1st_ed2: hydrophobic +1, polar -1).
          A reordering that leaves the sequence unchanged (swapping two equal residues)
          is redrawn.

Query ids are <accession>__real, <accession>__nullA_<draw>, <accession>__nullB_<draw>.

Outputs, in --outdir:
  reference.fasta      >ACCESSION headers, sequence on one line
  reference_lengths.tsv accession, length
  queries.fasta        every real query and null draw
  draws.tsv            one row per query id: accession, lineage, kind, draw, unit_length,
                       unit_source, permutation (null B), sequence
"""

import argparse
import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl

HP_ALPHABET = "hp_pbotc_1st_ed2"
LAG_MIN, LAG_MAX = 2, 30
MAX_TRIES = 1000


def read_fasta(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    name, chunks = None, []
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks)
                name, chunks = line[1:], []
            elif line:
                chunks.append(line.strip())
    if name is not None:
        yield name, "".join(chunks)


def accession(header: str) -> str:
    """sp|P04002|ISPA_PSEAM ... -> P04002."""
    first = header.split()[0]
    parts = first.split("|")
    return parts[1] if len(parts) >= 3 else first


def hp_signs(seq: str, hydrophobic: set[str]) -> np.ndarray:
    return np.array([1.0 if aa in hydrophobic else -1.0 for aa in seq])


def autocorrelation(x: np.ndarray, lags: range) -> np.ndarray:
    n = len(x)
    return np.array(
        [float(np.mean(x[:-lag] * x[lag:])) if lag < n else 0.0 for lag in lags]
    )


def apply_unit_permutation(seq: str, unit: int, perm: list[int]) -> str:
    n_whole = len(seq) // unit
    out = ["".join(seq[u * unit + i] for i in perm) for u in range(n_whole)]
    out.append(seq[n_whole * unit :])
    return "".join(out)


def null_b(seq: str, unit: int, rng: np.random.Generator) -> tuple[str, list[int]]:
    """One within-unit reordering that changes the sequence.

    A reordering that only swaps equal residues (A and A in AAT) returns the query
    itself, so draws are repeated until the sequence changes, up to MAX_TRIES.
    """
    for _ in range(MAX_TRIES):
        perm = [int(i) for i in rng.permutation(unit)]
        out = apply_unit_permutation(seq, unit, perm)
        if out != seq:
            return out, perm
    sys.exit(f"no within-unit reordering of length {unit} changes the sequence")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--swissprot-fasta", type=Path, required=True)
    ap.add_argument(
        "--queries", type=Path, required=True, help="data/convergence_afp/queries.tsv"
    )
    ap.add_argument(
        "--truth", type=Path, required=True, help="data/convergence_afp/truth_set.tsv"
    )
    ap.add_argument(
        "--motifs", type=Path, required=True, help="assets/known_motifs.tsv"
    )
    ap.add_argument("--alphabet-classes", type=Path, required=True)
    ap.add_argument("--n-draws", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument(
        "--subset",
        type=int,
        default=0,
        help="test runs only: keep the truth entries plus this many other entries",
    )
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    truth = set(pl.read_csv(args.truth, separator="\t")["Entry"])
    queries = pl.read_csv(args.queries, separator="\t")
    want = set(queries["Entry"])

    seqs: dict[str, str] = {}
    n_seen = 0
    with open(args.outdir / "reference.fasta", "w") as fa, open(
        args.outdir / "reference_lengths.tsv", "w"
    ) as ln:
        ln.write("accession\tlength\n")
        for header, seq in read_fasta(args.swissprot_fasta):
            acc = accession(header)
            n_seen += 1
            if args.subset and acc not in truth and n_seen > args.subset:
                continue
            if acc in seqs:
                sys.exit(f"duplicate accession {acc} in {args.swissprot_fasta}")
            seqs[acc] = seq
            fa.write(f">{acc}\n{seq}\n")
            ln.write(f"{acc}\t{len(seq)}\n")
    missing = sorted(truth - set(seqs))
    if missing:
        sys.exit(f"truth entries not in the reference (wrong release?): {missing}")

    cls = pl.read_csv(args.alphabet_classes, separator="\t").filter(
        pl.col("alphabet") == HP_ALPHABET
    )
    hydrophobic = set(cls.filter(pl.col("class_index") == 0)["residue"])
    motifs = {
        r["lineage"]: r
        for r in pl.read_csv(args.motifs, separator="\t").iter_rows(named=True)
    }

    rng = np.random.default_rng(args.seed)
    rows = []
    for q in queries.iter_rows(named=True):
        acc, lineage = q["Entry"], q["lineage"]
        seq = seqs[acc]
        if len(seq) != q["Length"]:
            sys.exit(
                f"{acc}: length {len(seq)} in the reference, {q['Length']} in queries.tsv"
            )
        motif = motifs.get(lineage)
        if motif:
            unit, source = (
                int(motif["unit_length"]),
                f"published consensus ({motif['source_doi']})",
            )
            hits = re.findall(motif["regex"], seq)
            if not hits:
                sys.exit(
                    f"{acc}: its own lineage's published motif {motif['regex']} does not match it"
                )
        else:
            lags = range(LAG_MIN, min(LAG_MAX, len(seq) // 2) + 1)
            ac = autocorrelation(hp_signs(seq, hydrophobic), lags)
            unit = lags[int(np.argmax(ac))]
            source = (
                f"HP autocorrelation peak (r={ac.max():.3f}, lags {LAG_MIN}-{lags[-1]})"
            )
        base = {
            "accession": acc,
            "lineage": lineage,
            "unit_length": unit,
            "unit_source": source,
        }
        rows.append(
            {
                **base,
                "query_id": f"{acc}__real",
                "kind": "real",
                "draw": 0,
                "permutation": "",
                "sequence": seq,
            }
        )
        for d in range(1, args.n_draws + 1):
            shuffled = "".join(rng.permutation(list(seq)))
            rows.append(
                {
                    **base,
                    "query_id": f"{acc}__nullA_{d:02d}",
                    "kind": "nullA",
                    "draw": d,
                    "permutation": "",
                    "sequence": shuffled,
                }
            )
            permuted, perm = null_b(seq, unit, rng)
            rows.append(
                {
                    **base,
                    "query_id": f"{acc}__nullB_{d:02d}",
                    "kind": "nullB",
                    "draw": d,
                    "permutation": json.dumps(perm),
                    "sequence": permuted,
                }
            )

    draws = pl.DataFrame(rows).select(
        "query_id",
        "accession",
        "lineage",
        "kind",
        "draw",
        "unit_length",
        "unit_source",
        "permutation",
        "sequence",
    )
    for r in draws.iter_rows(named=True):
        if sorted(r["sequence"]) != sorted(seqs[r["accession"]]):
            sys.exit(f"{r['query_id']}: composition differs from its real query")
    draws.write_csv(args.outdir / "draws.tsv", separator="\t")
    with open(args.outdir / "queries.fasta", "w") as fh:
        for r in draws.iter_rows(named=True):
            fh.write(f">{r['query_id']}\n{r['sequence']}\n")

    print(
        f"reference: {len(seqs)} entries; queries: {draws.height} "
        f"({queries.height} real, {args.n_draws} null A and {args.n_draws} null B draws each)"
    )
    print(
        draws.filter(pl.col("kind") == "real").select(
            "accession", "lineage", "unit_length", "unit_source"
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
