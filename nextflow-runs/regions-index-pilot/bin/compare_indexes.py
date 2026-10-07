#!/usr/bin/env python3
"""Recall at 5% decoy error for every tool, index, decoy window and kmerseek setting, and
the paired tests of PREDICTIONS.md.

Inputs: every <label>.features.parquet and <label>.threshold.json from score_calls.py
(label = tool.index.wNN.setting), the truth table, and the regions index FASTA (for the
identity split).

Outputs:
  thresholds.tsv       one row per label: threshold, target and decoy calls at it
  recall_by_kind.tsv   recall per label x feature kind (and per evidence class, and per
                       chromosome 6 gene group)
  mcnemar.tsv          per tool x setting x decoy window x feature kind: features found
                       with the whole-protein index only, the regions index only, both,
                       neither; recall change; exact two-sided McNemar p
  composition.tsv      kmerseek against the composition-only classifier, regions index,
                       disordered features (prediction 4)
  identity_bins.tsv    recall per label split by the identity of each query feature to
                       its best correct regions-index entry
  feature_calls.parquet every label x feature row, for the notebook

Feature kinds are those of build_query_truth.py; "short" (under 60 aa) overlaps the
others. The exact McNemar p is the two-sided binomial test on the discordant pairs.
"""

import argparse
import json
import math
from pathlib import Path

import polars as pl

KINDS = ["folded_domain", "short", "disordered", "motif", "composition", "other", "all"]

# BLOSUM62, for the identity split's local alignment (gap open 11, extend 1).
_B62 = """
   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4
"""
_rows = [r.split() for r in _B62.strip().splitlines()]
_cols = _rows[0]
B62 = {(r[0], c): int(v) for r in _rows[1:] for c, v in zip(_cols, r[1:])}


def local_identity(a: str, b: str, go: int = 11, ge: int = 1) -> float:
    """Identical positions / aligned columns of the best local alignment (Gotoh)."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0.0
    NEG = -(10**9)
    H = [[0] * (m + 1) for _ in range(n + 1)]
    E = [[NEG] * (m + 1) for _ in range(n + 1)]
    F = [[NEG] * (m + 1) for _ in range(n + 1)]
    best, bi, bj = 0, 0, 0
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            E[i][j] = max(E[i][j - 1] - ge, H[i][j - 1] - go)
            F[i][j] = max(F[i - 1][j] - ge, H[i - 1][j] - go)
            s = H[i - 1][j - 1] + B62.get((ai, b[j - 1]), -4)
            h = max(0, s, E[i][j], F[i][j])
            H[i][j] = h
            if h > best:
                best, bi, bj = h, i, j
    i, j, ident, cols = bi, bj, 0, 0
    while i > 0 and j > 0 and H[i][j] > 0:
        h = H[i][j]
        if h == H[i - 1][j - 1] + B62.get((a[i - 1], b[j - 1]), -4):
            ident += a[i - 1] == b[j - 1]
            cols += 1
            i, j = i - 1, j - 1
        elif h == E[i][j]:
            cols += 1
            j -= 1
        else:
            cols += 1
            i -= 1
    return ident / cols if cols else 0.0


def mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def kind_filter(kind: str) -> pl.Expr:
    if kind == "all":
        return pl.lit(True)
    if kind == "short":
        return pl.col("is_short")
    return pl.col("feature_kind") == kind


def read_fasta(path: Path) -> dict[str, str]:
    out, name, seq = {}, None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name:
                    out[name] = "".join(seq)
                name, seq = line[1:].split()[0], []
            elif line:
                seq.append(line)
    if name:
        out[name] = "".join(seq)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--scored", type=Path, nargs="+", required=True)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--regions-fasta", type=Path, required=True)
    ap.add_argument("--headline-window", default="w10")
    args = ap.parse_args()

    feats = [p for p in args.scored if p.name.endswith(".features.parquet")]
    thrs = [p for p in args.scored if p.name.endswith(".threshold.json")]
    thr = pl.DataFrame([json.loads(p.read_text()) for p in thrs])
    thr = thr.with_columns(
        pl.col("label")
        .str.split(".")
        .list.to_struct(fields=["tool", "index", "window", "setting"])
        .alias("_p")
    ).unnest("_p")
    thr.sort("tool", "setting", "window", "index").write_csv(
        "thresholds.tsv", separator="\t"
    )

    truth = pl.read_parquet(args.truth)
    fc = pl.concat([pl.read_parquet(p) for p in feats], how="diagonal_relaxed")
    fc = fc.with_columns(
        pl.col("label")
        .str.split(".")
        .list.to_struct(fields=["tool", "index", "window", "setting"])
        .alias("_p")
    ).unnest("_p")
    fc = fc.join(
        truth.select(
            "truth_id",
            "accession",
            "gene",
            "feature_type",
            "feature_kind",
            "is_short",
            "experimental",
            "gene_group",
            "start",
            "end",
            "length",
            "description",
        ),
        on="truth_id",
        how="left",
    )

    # Identity split: each feature's best correct regions-index entry, any tool, headline
    # window, aligned locally against the feature's own residues.
    qseq = read_fasta(args.queries)
    rseq = read_fasta(args.regions_fasta)
    best_r = (
        fc.filter(
            (pl.col("index") == "regions")
            & pl.col("found")
            & (pl.col("window") == args.headline_window)
        )
        .select("truth_id", "accession", "start", "end", "best_target")
        .unique(["truth_id", "best_target"])
    )
    ident = {}
    for tid, acc, s, e, tgt in best_r.iter_rows():
        v = local_identity(qseq[acc][s - 1 : e], rseq.get(tgt, ""))
        ident[tid] = max(ident.get(tid, 0.0), v)
    id_df = pl.DataFrame(
        {"truth_id": list(ident), "identity": list(ident.values())},
        schema={"truth_id": pl.UInt32, "identity": pl.Float64},
    ).with_columns(
        pl.when(pl.col("identity") < 0.3)
        .then(pl.lit("<30%"))
        .when(pl.col("identity") < 0.5)
        .then(pl.lit("30-50%"))
        .when(pl.col("identity") < 0.9)
        .then(pl.lit("50-90%"))
        .otherwise(pl.lit(">=90%"))
        .alias("identity_bin")
    )
    fc = fc.join(id_df, on="truth_id", how="left").with_columns(
        pl.col("identity_bin").fill_null("no correct regions-index entry")
    )
    fc.write_parquet("feature_calls.parquet")

    rows = []
    for kind in KINDS:
        for ev, evf in [
            ("all", pl.lit(True)),
            ("experimental", pl.col("experimental")),
        ]:
            for grp in ["all", "MHC", "histone", "olfactory_receptor"]:
                gf = pl.lit(True) if grp == "all" else pl.col("gene_group") == grp
                sub = fc.filter(kind_filter(kind) & evf & gf)
                agg = sub.group_by("tool", "index", "window", "setting").agg(
                    pl.len().alias("features"),
                    pl.col("found").sum().alias("found"),
                    pl.col("found_name").sum().alias("found_name"),
                )
                rows.append(
                    agg.with_columns(
                        pl.lit(kind).alias("feature_kind"),
                        pl.lit(ev).alias("evidence"),
                        pl.lit(grp).alias("gene_group"),
                    )
                )
    rec = (
        pl.concat(rows)
        .with_columns((pl.col("found") / pl.col("features")).alias("recall"))
        .sort(
            "feature_kind",
            "evidence",
            "gene_group",
            "tool",
            "setting",
            "window",
            "index",
        )
    )
    rec.write_csv("recall_by_kind.tsv", separator="\t")

    wide = (
        fc.select("tool", "setting", "window", "index", "truth_id", "found")
        .pivot(
            on="index", index=["tool", "setting", "window", "truth_id"], values="found"
        )
        .join(
            truth.select("truth_id", "feature_kind", "is_short", "experimental"),
            on="truth_id",
        )
    )
    mc = []
    for kind in KINDS:
        sub = wide.filter(kind_filter(kind))
        for (tool, setting, window), g in sub.group_by("tool", "setting", "window"):
            if "whole" not in g.columns or "regions" not in g.columns:
                continue
            w, r = g["whole"].fill_null(False), g["regions"].fill_null(False)
            b = int((w & ~r).sum())
            c = int((~w & r).sum())
            n = g.height
            mc.append(
                {
                    "tool": tool,
                    "setting": setting,
                    "window": window,
                    "feature_kind": kind,
                    "features": n,
                    "whole_only": b,
                    "regions_only": c,
                    "both": int((w & r).sum()),
                    "neither": int((~w & ~r).sum()),
                    "recall_whole": int(w.sum()) / n if n else None,
                    "recall_regions": int(r.sum()) / n if n else None,
                    "recall_change_points": 100 * (c - b) / n if n else None,
                    "mcnemar_p": mcnemar_p(b, c),
                }
            )
    pl.DataFrame(mc).sort("feature_kind", "tool", "setting", "window").write_csv(
        "mcnemar.tsv", separator="\t"
    )

    comp = fc.filter(
        (pl.col("index") == "regions") & (pl.col("feature_kind") == "disordered")
    )
    cl = comp.filter(pl.col("tool") == "composition")
    out = []
    for (tool, setting, window), g in comp.filter(
        pl.col("tool") == "kmerseek"
    ).group_by("tool", "setting", "window"):
        cw = cl.filter(pl.col("window") == window).select(
            "truth_id", pl.col("found").alias("comp")
        )
        j = (
            g.select("truth_id", "found")
            .join(cw, on="truth_id", how="left")
            .with_columns(pl.col("comp").fill_null(False))
        )
        b = int((j["found"] & ~j["comp"]).sum())
        c = int((~j["found"] & j["comp"]).sum())
        out.append(
            {
                "setting": setting,
                "window": window,
                "features": j.height,
                "kmerseek_found": int(j["found"].sum()),
                "composition_found": int(j["comp"].sum()),
                "kmerseek_only": b,
                "composition_only": c,
                "mcnemar_p": mcnemar_p(b, c),
            }
        )
    pl.DataFrame(
        out,
        schema={
            "setting": pl.Utf8,
            "window": pl.Utf8,
            "features": pl.Int64,
            "kmerseek_found": pl.Int64,
            "composition_found": pl.Int64,
            "kmerseek_only": pl.Int64,
            "composition_only": pl.Int64,
            "mcnemar_p": pl.Float64,
        },
    ).write_csv("composition.tsv", separator="\t")

    (
        fc.group_by("tool", "index", "window", "setting", "identity_bin")
        .agg(pl.len().alias("features"), pl.col("found").sum().alias("found"))
        .with_columns((pl.col("found") / pl.col("features")).alias("recall"))
        .sort("identity_bin", "tool", "setting", "window", "index")
        .write_csv("identity_bins.tsv", separator="\t")
    )


if __name__ == "__main__":
    main()
