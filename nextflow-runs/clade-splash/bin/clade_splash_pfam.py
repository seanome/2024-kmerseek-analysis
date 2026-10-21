#!/usr/bin/env python3
"""Notebook 357: the Pfam controls of the clade-grouped SPLASH run.

Kept apart from clade_splash.py on purpose: Nextflow puts the content of the bin/ scripts a
task calls into its cache key, so a change here does not make the counting and testing
tasks run again on -resume.

    pfam-overlap   which significant anchors lie inside a Pfam domain (hmmsearch domains)
    align-input    domain envelopes to align, for anchors inside one Pfam family
    align-control  the per-class OASIS test on each aligned column under the target
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

from clade_splash import TARGET_LENGTH, bh, labellings, load_oasis, parse_setting


def read_domtblout(path: str) -> pl.DataFrame:
    """hmmsearch --domtblout: target (the protein, s<species>_p<protein>), query (the Pfam
    family), envelope coordinates on the protein (1-based, inclusive)."""
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = line.split()
            rows.append({"query": f[0], "pfam_name": f[3], "pfam_acc": f[4].split(".")[0],
                         "env_from": int(f[19]), "env_to": int(f[20]), "dom_ievalue": float(f[12])})
    schema = {"query": pl.String, "pfam_name": pl.String, "pfam_acc": pl.String,
              "env_from": pl.Int64, "env_to": pl.Int64, "dom_ievalue": pl.Float64}
    return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)


def read_domains(paths: list[str]) -> pl.DataFrame:
    paths = [p for p in paths if Path(p).exists()]
    if not paths:
        return read_domtblout(os.devnull)
    return pl.concat([read_domtblout(p) for p in paths])


def cmd_pfam_overlap(a: argparse.Namespace) -> None:
    """For each occurrence, the Pfam domains its anchor-to-target window overlaps.

    The window is [pos0, pos0 + k + gap + 10) in 0-based protein coordinates; a domain's
    envelope [env_from, env_to] is 1-based inclusive. An anchor is outside Pfam when no
    occurrence's window overlaps any domain.

    One occurrence file at a time: files are split by anchor hash, so every anchor's
    occurrences are in one file, and memory is that of the largest file.
    """
    dom = read_domains(a.domtblout).select("query", "pfam_acc", "env_from", "env_to")
    sig = pl.read_parquet(a.significant).filter(pl.col("labelling") == "real")
    top = (sig.sort("q_oasis", "p_oasis", "anchor").group_by("setting", "gap", maintain_order=True)
           .head(a.top).select("setting", "gap", "anchor"))
    occ_dir = Path(a.occ_out)
    occ_dir.mkdir(parents=True, exist_ok=True)
    anchors_parts, top_parts = [], []
    keys = ["setting", "gap", "anchor", "species_idx", "protein_idx", "pos0"]
    for i, f in enumerate(a.occurrences):
        occ = pl.read_parquet(f)
        if not occ.height:
            continue
        occ = occ.with_columns(pl.format("s{}_p{}", "species_idx", "protein_idx").alias("query"))
        k_of = {s: parse_setting(s)[1] for s in occ["setting"].unique().to_list()}
        occ = occ.with_columns(pl.col("setting").replace_strict(k_of, return_dtype=pl.Int64).alias("k"))
        occ = occ.with_columns((pl.col("pos0") + 1).alias("win_from"),
                               (pl.col("pos0") + pl.col("k") + pl.col("gap") + TARGET_LENGTH).alias("win_to"))
        j = occ.select(*keys, "query", "win_from", "win_to").join(dom, on="query", how="left")
        j = j.with_columns(((pl.col("env_from") <= pl.col("win_to")) & (pl.col("env_to") >= pl.col("win_from")))
                           .fill_null(False).alias("overlap"))
        per_occ = j.group_by(keys).agg(
            pl.col("overlap").any().alias("in_pfam"),
            pl.col("pfam_acc").filter(pl.col("overlap")).unique().sort().str.join(";").alias("pfam_overlapping"))
        per_occ.write_parquet(occ_dir / f"part{i:04d}.parquet")
        anchors_parts.append(per_occ.group_by("setting", "gap", "anchor").agg(
            pl.len().alias("n_occurrences"), pl.col("in_pfam").sum().alias("n_occ_in_pfam"),
            pl.col("pfam_overlapping").filter(pl.col("pfam_overlapping") != "").str.split(";").flatten()
            .mode().sort().first().alias("main_pfam")))
        # Residues of the top anchors' occurrences, for the per-anchor figures: the a.top
        # anchors with the smallest q (then p) in each (setting, gap), real labels.
        top_parts.append(occ.join(top, on=["setting", "gap", "anchor"]).join(
            per_occ.select(*keys, "in_pfam", "pfam_overlapping"), on=keys, how="left")
            .drop("query", "k", "win_from", "win_to"))
    schema = {"setting": pl.String, "gap": pl.Int16, "anchor": pl.UInt64, "n_occurrences": pl.UInt32,
              "n_occ_in_pfam": pl.UInt32, "main_pfam": pl.String}
    per_anchor = pl.concat(anchors_parts) if anchors_parts else pl.DataFrame(schema=schema)
    per_anchor = per_anchor.with_columns((pl.col("n_occ_in_pfam") == 0).alias("outside_pfam"))
    per_anchor.write_parquet(a.out)
    (pl.concat(top_parts) if top_parts else pl.DataFrame()).write_parquet(a.top_out)


def cmd_align_input(a: argparse.Namespace) -> None:
    """FASTA of the domain envelopes to hmmalign, one per (anchor, family) of the anchors whose
    window is inside one Pfam family in at least half of their occurrences."""
    anchors = pl.read_parquet(a.pfam).filter(~pl.col("outside_pfam") & (pl.col("n_occ_in_pfam") * 2 >= pl.col("n_occurrences")))
    dom = read_domains(a.domtblout).select("query", "pfam_acc", "env_from", "env_to")
    keep = anchors.select("setting", "gap", "anchor", "main_pfam")
    parts = []
    for f in a.occurrences:
        occ = pl.read_parquet(f)
        if not occ.height:
            continue
        occ = occ.select("setting", "gap", "anchor", "species_idx", "protein_idx", "pos0").join(
            keep, on=["setting", "gap", "anchor"])
        occ = occ.with_columns(pl.format("s{}_p{}", "species_idx", "protein_idx").alias("query"))
        parts.append(occ.join(dom, left_on=["query", "main_pfam"], right_on=["query", "pfam_acc"]))
    j = pl.concat(parts) if parts else pl.DataFrame()
    need = set(j["query"].to_list()) if j.height else set()
    seqs, name = {}, None
    for f in a.fastas:
        for line in open(f):
            if line.startswith(">"):
                name = line[1:].split()[0]
                if name in need and name not in seqs:
                    seqs[name] = []
                else:
                    name = None
            elif name is not None:
                seqs[name].append(line.strip())
    seqs = {k: "".join(v) for k, v in seqs.items()}
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    if j.height:
        k_of = {s: parse_setting(s)[1] for s in j["setting"].unique().to_list()}
        for (setting, gap, anchor, fam), sub in j.group_by("setting", "gap", "anchor", "main_pfam"):
            k = k_of[setting]
            tag = f"{setting.replace(':', '_')}_g{gap}_{anchor}_{fam}"
            with open(out / f"{tag}.fa", "w") as fh:
                for r in sub.unique(["species_idx", "protein_idx", "pos0", "env_from"]).sort(
                        "species_idx", "protein_idx", "pos0", "env_from").iter_rows(named=True):
                    ts = r["pos0"] + k + gap  # 0-based target start
                    if not (r["env_from"] - 1 <= ts and ts + TARGET_LENGTH <= r["env_to"]):
                        continue  # the target must lie inside this domain envelope to map to columns
                    dseq = seqs[r["query"]][r["env_from"] - 1:r["env_to"]]
                    fh.write(f">{r['query']}|{r['pos0']}|{r['env_from']}|{r['species_idx']}\n{dseq}\n")
            manifest.append({"setting": setting, "gap": gap, "anchor": anchor, "pfam_acc": fam, "tag": tag})
    man = pl.DataFrame(manifest, schema={"setting": pl.String, "gap": pl.Int64, "anchor": pl.UInt64,
                                         "pfam_acc": pl.String, "tag": pl.String}).sort("tag")
    man.write_csv(a.manifest, separator="\t")
    # Parts of a.part_size rows, so hmmalign and the column tests run in parallel. At least
    # one part, possibly empty, so the rest of the pipeline still runs.
    parts = Path(a.parts_dir)
    parts.mkdir(parents=True, exist_ok=True)
    n_parts = max(1, -(-man.height // a.part_size))
    for i in range(n_parts):
        man.slice(i * a.part_size, a.part_size).write_csv(parts / f"manifest_{i:04d}.tsv", separator="\t")


def read_stockholm_match_columns(path: str) -> dict[str, list[tuple[int, str]]]:
    """hmmalign Stockholm -> for each sequence, (residue index in the input, match column) pairs.

    Match columns are upper-case letters and '-'; insertions are lower-case letters and '.'.
    Returns, per sequence, a list where entry m is (input residue index or -1, residue or '-')
    for match column m.
    """
    aligned = defaultdict(str)
    for line in open(path):
        if not line.strip() or line.startswith("#") or line.startswith("//"):
            continue
        name, s = line.split()
        aligned[name] += s
    out = {}
    for name, s in aligned.items():
        cols, ri = [], 0
        for ch in s:
            if ch == "." :
                continue
            if ch == "-":
                cols.append((-1, "-"))
            elif ch.isupper():
                cols.append((ri, ch))
                ri += 1
            else:  # insertion
                ri += 1
        out[name] = cols
    return out


def cmd_align_control(a: argparse.Namespace) -> None:
    """Per-clade test on each aligned match column that a target residue maps to.

    For each target position t (0-9), each occurrence's residue there maps to a match column of
    the family HMM (or to an insertion, then dropped). For each match column the table is
    residue x class, cell = species with that residue in that column, restricted to the classes
    of the SPLASH table. The same OASIS test gives a p-value per column; BH within the anchor.
    """
    oasis = load_oasis(a.oasis)
    species = pl.read_csv(a.species, separator="\t")
    labs = labellings(species, a.seed)["real"]
    sig = pl.read_parquet(a.significant).filter(pl.col("labelling") == "real")
    manifest = pl.read_csv(a.manifest, separator="\t", schema_overrides={"anchor": pl.UInt64})
    clades_of = {(st, int(g), int(an)): c for st, g, an, c in sig.select("setting", "gap", "anchor", "clades").iter_rows()}
    rows = []
    for r in manifest.iter_rows(named=True):
        sto = Path(a.sto_dir) / f"{r['tag']}.sto"
        if not sto.exists() or sto.stat().st_size == 0:
            continue
        key = (r["setting"], r["gap"], r["anchor"])
        if key not in clades_of:
            continue
        clade_list = clades_of[key].split(";")
        k = parse_setting(r["setting"])[1]
        cols = read_stockholm_match_columns(str(sto))
        per_col = defaultdict(lambda: defaultdict(set))  # column -> (residue, clade) -> species
        for name, mc in cols.items():
            query, pos0, env_from, sidx = name.split("|")
            pos0, env_from, sidx = int(pos0), int(env_from), int(sidx)
            clade = labs[sidx]
            if clade not in clade_list:
                continue
            ts_dom = pos0 + k + r["gap"] - (env_from - 1)  # target start within the domain, 0-based
            want = set(range(ts_dom, ts_dom + TARGET_LENGTH))
            for m, (ri, ch) in enumerate(mc):
                if ri in want:
                    per_col[m][(ch, clade)].add(sidx)
        pvals = []
        for m, cells in sorted(per_col.items()):
            residues = sorted({ch for ch, _ in cells})
            x = np.zeros((len(residues), len(clade_list)), dtype=np.int64)
            for (ch, clade), sp in cells.items():
                x[residues.index(ch), clade_list.index(clade)] = len(sp)
            p = float(oasis.OASIS_pvalue(x)) if x.shape[0] > 1 else 1.0
            pvals.append((m, p, int(x.sum())))
        q = bh(np.array([p for _, p, _ in pvals])) if pvals else np.array([])
        for (m, p, n), qq in zip(pvals, q):
            rows.append({"setting": r["setting"], "gap": r["gap"], "anchor": r["anchor"], "pfam_acc": r["pfam_acc"],
                         "match_column": m, "n_species": n, "p_column": p, "q_column": float(qq)})
    schema = {"setting": pl.String, "gap": pl.Int64, "anchor": pl.UInt64, "pfam_acc": pl.String,
              "match_column": pl.Int64, "n_species": pl.Int64, "p_column": pl.Float64, "q_column": pl.Float64}
    out = pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)
    out.write_parquet(a.out)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("pfam-overlap")
    s.add_argument("--occurrences", nargs="+", required=True)
    s.add_argument("--domtblout", nargs="+", required=True)
    s.add_argument("--occ-out", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--significant", required=True)
    s.add_argument("--top", type=int, default=50)
    s.add_argument("--top-out", required=True)
    s.set_defaults(func=cmd_pfam_overlap)

    s = sub.add_parser("align-input")
    s.add_argument("--pfam", required=True)
    s.add_argument("--occurrences", nargs="+", required=True)
    s.add_argument("--domtblout", nargs="+", required=True)
    s.add_argument("--fastas", nargs="+", required=True)
    s.add_argument("--outdir", required=True)
    s.add_argument("--manifest", required=True)
    s.add_argument("--parts-dir", required=True)
    s.add_argument("--part-size", type=int, default=500)
    s.set_defaults(func=cmd_align_input)

    s = sub.add_parser("align-control")
    s.add_argument("--manifest", required=True)
    s.add_argument("--sto-dir", required=True)
    s.add_argument("--significant", required=True)
    s.add_argument("--species", required=True)
    s.add_argument("--oasis", required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_align_control)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
