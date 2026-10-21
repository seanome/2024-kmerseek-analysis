#!/usr/bin/env python3
"""Notebook 357: the count tables, the OASIS test and the occurrences of significant anchors.

These two steps replace `test` and `occurrences` in clade_splash.py (still there, unused),
for the full run's scale: one target bucket of the Dayhoff setting is about 2.3 GB of
parquet, several hundred million occurrence rows, mostly from repeat-rich anchors, and
reading it whole was killed at 16 GB. Here a bucket is read lazily, one gap at a time, and
deduplicated while streaming; classes are small integer codes, not strings per row.

A separate file so that the tasks in clade_splash.py keep their cache key (Nextflow puts the
content of the bin/ scripts a task calls into its hash).

    test         count tables, OASIS and the composition scan, every gap and labelling
    occurrences  rows of the significant anchors, with residues, for Pfam and the figures
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

from clade_splash import (
    LABELLINGS,
    MIN_CLADES,
    MIN_SPECIES_PER_CLADE,
    TARGET_LENGTH,
    composition_table,
    labellings,
    load_oasis,
    parse_setting,
    pearson_chi2,
    read_fasta,
)

TEST_SCHEMA = {
    "setting": pl.String,
    "labelling": pl.String,
    "gap": pl.Int64,
    "anchor": pl.UInt64,
    "n_species": pl.Int64,
    "table_total": pl.Int64,
    "n_clades": pl.Int64,
    "n_targets": pl.Int64,
    "p_oasis": pl.Float64,
    "effect_size": pl.Float64,
    "comp_chi2": pl.Float64,
    "comp_df": pl.Int64,
    "p_comp": pl.Float64,
    "clades": pl.String,
    "clade_weights": pl.String,
}


def label_tables(
    species: pl.DataFrame, seed: int
) -> tuple[dict[str, pl.DataFrame], list[str]]:
    """species_idx -> integer class code, per labelling; and the class names by code."""
    labs = labellings(species, seed)
    names = sorted(set(species["class"].to_list()))
    code = {n: i for i, n in enumerate(names)}
    idx = species["species_idx"].to_numpy()
    out = {}
    for lab_name in LABELLINGS:
        out[lab_name] = pl.DataFrame(
            {
                "species_idx": idx.astype(np.int32),
                "clade": np.array(
                    [code[c] for c in labs[lab_name][idx]], dtype=np.int16
                ),
            }
        )
    return out, names


def test_one_gap(
    pairs: pl.DataFrame,
    lab: pl.DataFrame,
    names: list[str],
    oasis,
    setting: str,
    lab_name: str,
    gap: int,
) -> list[dict]:
    """Tables and tests for every anchor of one gap under one labelling.

    `pairs` has one row per (anchor, species_idx, target): a species counts once per target,
    however many proteins carry it.
    """
    df = pairs.join(lab, on="species_idx")
    carry = df.select("anchor", "species_idx", "clade").unique()
    per_clade = carry.group_by("anchor", "clade").len("n_species")
    good = per_clade.filter(pl.col("n_species") >= MIN_SPECIES_PER_CLADE)
    n_good = (
        good.group_by("anchor").len("n_clades").filter(pl.col("n_clades") >= MIN_CLADES)
    )
    kept = good.join(n_good.select("anchor"), on="anchor").select("anchor", "clade")
    if not kept.height:
        return []
    # Distinct species in the kept classes. The table total is larger whenever a species
    # carries the anchor with more than one target (paralogs, repeats).
    n_carrying = dict(
        carry.join(kept, on=["anchor", "clade"])
        .group_by("anchor")
        .agg(pl.col("species_idx").n_unique())
        .iter_rows()
    )
    tab = (
        df.join(kept, on=["anchor", "clade"])
        .group_by("anchor", "target", "clade")
        .len("n")
    )
    rows = []
    for (anchor,), sub in tab.partition_by(["anchor"], as_dict=True).items():
        wide = (
            sub.pivot(on="clade", index="target", values="n")
            .fill_null(0)
            .sort("target")
        )
        codes = sorted(
            (c for c in wide.columns if c != "target"), key=lambda c: names[int(c)]
        )
        clades = [names[int(c)] for c in codes]
        x = wide.select(codes).to_numpy().astype(np.int64)
        tcodes = wide["target"].to_numpy()
        # return_f_c also gives the row and column vectors the p-value used, which the effect
        # size needs.
        p, f, c = oasis.OASIS_pvalue(x, return_f_c=True)
        eff = float(oasis.effectSize_bin(x, np.asarray(c), np.asarray(f)))
        x2, dfree, pc = pearson_chi2(composition_table(tcodes, x))
        rows.append(
            {
                "setting": setting,
                "labelling": lab_name,
                "gap": int(gap),
                "anchor": int(anchor),
                "n_species": int(n_carrying[anchor]),
                "table_total": int(x.sum()),
                "n_clades": len(clades),
                "n_targets": x.shape[0],
                "p_oasis": float(p),
                "effect_size": eff,
                "comp_chi2": x2,
                "comp_df": dfree,
                "p_comp": pc,
                "clades": ";".join(clades),
                # OASIS's column weights c (one per class, same order as `clades`): the classes at
                # the two ends are the ones the targets split.
                "clade_weights": ";".join(
                    f"{v:.4f}" for v in np.asarray(c, dtype=float)
                ),
            }
        )
    return rows


def cmd_test(a: argparse.Namespace) -> None:
    """Count tables and tests for one anchor bucket of one setting, every gap and labelling."""
    oasis = load_oasis(a.oasis)
    species = pl.read_csv(a.species, separator="\t")
    labs, names = label_tables(species, a.seed)
    files = [str(Path(d) / f"b{a.bucket:02d}.parquet") for d in a.dirs]
    lf = pl.scan_parquet(files)
    gaps = lf.select(pl.col("gap").unique()).collect()["gap"].sort().to_list()
    rows = []
    for gap in gaps:
        pairs = (
            lf.filter(pl.col("gap") == gap)
            .select("anchor", "species_idx", "target")
            .unique()
            .collect(engine="streaming")
        )
        for lab_name in LABELLINGS:
            rows.extend(
                test_one_gap(
                    pairs, labs[lab_name], names, oasis, a.setting, lab_name, gap
                )
            )
        del pairs
    out = (
        pl.DataFrame(rows, schema=TEST_SCHEMA)
        if rows
        else pl.DataFrame(schema=TEST_SCHEMA)
    )
    out.write_parquet(a.out)


def cmd_occurrences(a: argparse.Namespace) -> None:
    """Rows of the significant anchors (real labels) from one target bucket, with residues.

    Writes the occurrences, and a FASTA of every protein that carries one, for hmmsearch.
    """
    sig = (
        pl.read_parquet(a.significant)
        .filter((pl.col("labelling") == "real") & (pl.col("setting") == a.setting))
        .select(pl.col("anchor").cast(pl.UInt64), pl.col("gap").cast(pl.Int16))
        .unique()
    )
    k = parse_setting(a.setting)[1]
    files = [str(Path(d) / f"b{a.bucket:02d}.parquet") for d in a.dirs]
    occ = (
        pl.scan_parquet(files)
        .join(sig.lazy(), on=["anchor", "gap"], how="semi")
        .collect(engine="streaming")
    )
    acc = pl.concat([pl.read_parquet(f) for f in a.accessions])
    occ = occ.join(acc, on=["species_idx", "protein_idx"], how="left").with_columns(
        pl.col("accession").cast(pl.String)
    )
    by_species = defaultdict(set)
    for s, p in occ.select("species_idx", "protein_idx").unique().iter_rows():
        by_species[s].add(p)
    fasta_of = {
        int(p.name.split(".")[0]): p
        for d in a.fasta_dirs
        for p in Path(d).glob("*.fasta.gz")
    }
    seq_of = {}
    for s, prots in by_species.items():
        names, seqs = read_fasta(fasta_of[s])
        for p in prots:
            seq_of[(s, p)] = (names[p], seqs[p])
    window = []
    for s, p, pos0, gap in occ.select(
        "species_idx", "protein_idx", "pos0", "gap"
    ).iter_rows():
        seq = seq_of[(s, p)][1]
        window.append(
            (
                seq[pos0 : pos0 + k],
                seq[pos0 + k : pos0 + k + gap],
                seq[pos0 + k + gap : pos0 + k + gap + TARGET_LENGTH],
                len(seq),
            )
        )
    occ = occ.with_columns(
        pl.Series("anchor_residues", [w[0] for w in window], dtype=pl.String),
        pl.Series("gap_residues", [w[1] for w in window], dtype=pl.String),
        pl.Series("target_residues", [w[2] for w in window], dtype=pl.String),
        pl.Series("protein_length", [w[3] for w in window], dtype=pl.Int32),
        pl.lit(a.setting).alias("setting"),
    )
    occ.write_parquet(a.out)
    with open(a.fasta_out, "w") as fh:
        for (s, p), (name, seq) in sorted(seq_of.items()):
            fh.write(f">s{s}_p{p} {name}\n{seq}\n")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("test")
    s.add_argument("--setting", required=True)
    s.add_argument("--bucket", type=int, required=True)
    s.add_argument("--dirs", nargs="+", required=True)
    s.add_argument("--species", required=True)
    s.add_argument("--oasis", required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_test)

    s = sub.add_parser("occurrences")
    s.add_argument("--setting", required=True)
    s.add_argument("--bucket", type=int, required=True)
    s.add_argument("--dirs", nargs="+", required=True)
    s.add_argument("--accessions", nargs="+", required=True)
    s.add_argument("--significant", required=True)
    s.add_argument("--fasta-dirs", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--fasta-out", required=True)
    s.set_defaults(func=cmd_occurrences)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
