#!/usr/bin/env python3
"""How many truth proteins of other lineages share k-mers with each query, at every k.

kmerseek reports a target only if it shares at least `--min-shared-kmers` (2 in the
convergence-afp run) distinct k-mers with the query in the reduced alphabet. A target
below that is never reported, whatever its score would have been. This script counts,
for each real query, alphabet and k, the cross-lineage truth proteins that clear
that bar, and the share of a random Swiss-Prot sample that clears it too. The pair is
the ceiling on what any setting at that alphabet and k could reach, and the price paid
in unrelated proteins for reaching it.

It reads sequences only; no index is built and kmerseek is not run. The partitions are
copied from kmerseek's src/rust/alphabets.rs (commit d79f863) and sourmash's Dayhoff
encoding, the alphabets whose definitions are two or six groups.

--class afp     ice-binding proteins: the truth set and queries of the convergence-afp run.
--class shell   shell-matrix proteins: the counted targets of
                scripts/convergence_classes_truth.py (reviewed, in a pattern lineage,
                acidic or low complexity) and one query per pattern lineage picked by
                that script's rule. Also writes composition_rule_background.tsv: the share
                of the random sample that passes the class's own composition rule, which
                is how the truth set was defined.

Output: data/convergence_<class>/kmer_reach_ceiling.tsv
"""

import argparse
import gzip
import random
import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parent.parent
SPROT = Path("~/data/uniprot/uniprot_sprot.2026_03.fasta.gz").expanduser()
N_BACKGROUND = 5000
MIN_SHARED = 2  # params.min_shared_kmers in nextflow-runs/convergence-afp/main.nf
KS = list(range(3, 41))


def hp(h, p):
    m = {a: "h" for a in h}
    m.update({a: "p" for a in p})
    return m


ALPHABETS = {
    "protein20": {a: a for a in "ACDEFGHIKLMNPQRSTVWY"},
    "dayhoff6": {
        **{a: "a" for a in "AGPST"},
        "C": "c",
        **{a: "d" for a in "DENQ"},
        **{a: "h" for a in "HKR"},
        **{a: "i" for a in "ILMV"},
        **{a: "f" for a in "FWY"},
    },
    "hp_lehninger2": hp("AFGILMPVWY", "CDEHKNQRST"),
    "hp_thomas_dill2": hp("ACFILMVWY", "DEGHKNPQRST"),
    "hp_kyte_doolittle2": hp("ACFILMV", "DEGHKNPQRSTWY"),
    "hp_thomas_dill_no_c2": hp("AFILMVWY", "CDEGHKNPQRST"),
    "hp_lehninger_c_nonpolar2": hp("ACFGILMPVWY", "DEHKNQRST"),
    "hp_pbotc_1st_ed2": hp("ACFILMPVWY", "DEGHKNQRST"),
}


def encode(seq, table):
    # Residues outside the 20 (X, U, B, Z) break the string, as an unknown letter would
    # break a k-mer in kmerseek.
    return "".join(table.get(a, "*") for a in seq)


def kmers(s, k):
    return {s[i : i + k] for i in range(len(s) - k + 1) if "*" not in s[i : i + k]}


def read_fasta(path):
    seqs, name, buf = {}, None, []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if name:
                    seqs[name] = "".join(buf)
                name, buf = line.split("|")[1], []
            else:
                buf.append(line.strip())
    if name:
        seqs[name] = "".join(buf)
    return seqs


def load_afp(sprot):
    """Truth targets and queries of the convergence-afp run, with sequences."""
    d = REPO / "data/convergence_afp"
    # truth_set.tsv and queries.tsv carry no sequences; they come from the same
    # Swiss-Prot release the searches ran against.
    truth = pl.read_csv(d / "truth_set.tsv", separator="\t").with_columns(
        Sequence=pl.col("Entry").replace_strict(sprot, default=None)
    )
    assert truth["Sequence"].null_count() == 0, "truth entry missing from Swiss-Prot"
    queries = (
        pl.read_csv(d / "queries.tsv", separator="\t")
        .select("Entry", "lineage")
        .join(truth.select("Entry", "Sequence"), on="Entry", how="left")
    )
    return truth.select("Entry", "lineage", "Sequence"), queries, set(truth["Entry"])


def load_shell():
    """Counted shell-matrix targets and one query per pattern lineage.

    The rules are scripts/convergence_classes_truth.py's: a counted target is
    reviewed, in a pattern lineage, and acidic or low complexity; the query is the
    counted entry with the most experimental evidence, ties to the longer entry.
    """
    from convergence_classes_truth import PATTERN_LINEAGES

    t = pl.read_csv(
        REPO / "data/convergence_shell/truth_set.tsv",
        separator="\t",
        infer_schema_length=0,
    ).with_columns(
        pl.col("Length").cast(pl.Int64),
        pl.col("n_experimental_evidence").cast(pl.Int64),
    )
    counted = t.filter(
        (pl.col("tier") == "Swiss-Prot")
        & pl.col("lineage").is_in(PATTERN_LINEAGES["shell"])
        & (pl.col("kind") == "acidic or low complexity")
    )
    queries = (
        counted.sort(
            ["n_experimental_evidence", "Length", "Entry"],
            descending=[True, True, False],
        )
        .group_by("lineage", maintain_order=True)
        .first()
        .sort("lineage")
        .select("Entry", "lineage", "Sequence")
    )
    # Every shell entry, counted or not, stays out of the random sample.
    return counted.select("Entry", "lineage", "Sequence"), queries, set(t["Entry"])


def composition_rule_background(background):
    """Share of the random sample that passes class C's own composition rule."""
    from convergence_classes_truth import (
        DE_MIN,
        LOW_COMPLEXITY_MIN,
        low_complexity_fraction,
    )

    n_de = n_lc = n_either = 0
    for _, seq in background:
        de = (seq.count("D") + seq.count("E")) / len(seq) >= DE_MIN
        lc = low_complexity_fraction(seq) >= LOW_COMPLEXITY_MIN
        n_de += de
        n_lc += lc
        n_either += de or lc
    n = len(background)
    return pl.DataFrame(
        {
            "rule": [
                f"Asp+Glu share >= {DE_MIN}",
                f"low-complexity fraction >= {LOW_COMPLEXITY_MIN}",
                "either (the class C rule)",
            ],
            "n_background": [n] * 3,
            "n_pass": [n_de, n_lc, n_either],
            "frac_pass": [n_de / n, n_lc / n, n_either / n],
        }
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--class", dest="cls", choices=["afp", "shell"], default="afp")
    args = ap.parse_args()
    out_dir = REPO / f"data/convergence_{args.cls}"

    sprot = read_fasta(SPROT)
    if args.cls == "afp":
        truth, queries, exclude = load_afp(sprot)
    else:
        truth, queries, exclude = load_shell()
    print(f"{truth.height} targets, {queries.height} queries", file=sys.stderr)
    pool = sorted(a for a in sprot if a not in exclude)
    background = [(a, sprot[a]) for a in random.Random(0).sample(pool, N_BACKGROUND)]
    print(
        f"background: {len(background)} random entries of {len(sprot)} in Swiss-Prot",
        file=sys.stderr,
    )
    if args.cls == "shell":
        comp = composition_rule_background(background)
        comp.write_csv(out_dir / "composition_rule_background.tsv", separator="\t")
        print(comp, file=sys.stderr)

    rows = []
    for q in queries.iter_rows(named=True):
        # The query's own lineage is left out, as in bin/score.py.
        cross = truth.filter(pl.col("lineage") != q["lineage"])
        for alpha, table in ALPHABETS.items():
            qe = encode(q["Sequence"], table)
            ce = [encode(s, table) for s in cross["Sequence"]]
            be = [encode(s, table) for _, s in background]
            for k in KS:
                qk = kmers(qe, k)
                if not qk:
                    break
                n_cross = sum(len(qk & kmers(t, k)) >= MIN_SHARED for t in ce)
                n_bg = sum(len(qk & kmers(t, k)) >= MIN_SHARED for t in be)
                rows.append(
                    {
                        "query": q["Entry"],
                        "query_lineage": q["lineage"],
                        "alphabet": alpha,
                        "ksize": k,
                        "n_cross_truth": cross.height,
                        "n_cross_reachable": n_cross,
                        "frac_cross_reachable": n_cross / cross.height,
                        "frac_background_reachable": n_bg / len(background),
                    }
                )
    out = pl.DataFrame(rows)
    out.write_csv(out_dir / "kmer_reach_ceiling.tsv", separator="\t")
    print(
        f"wrote {out_dir / 'kmer_reach_ceiling.tsv'} ({out.height} rows)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
