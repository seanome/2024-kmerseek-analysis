#!/usr/bin/env python3
"""Write results/leaderboard/alphabets/: the tables behind the leaderboard's alphabet page.

Four tables, one per question a builder of a reduced-alphabet prefilter asks:

* ``classes.parquet``: which residues each of the 19 kmerseek alphabets puts together. One row
  per alphabet and residue (380 rows). Classes from ``notebooks/hp_conservation_utils.py``
  (the copy notebook 230 and 291 use of kmerseek's ``src/rust/alphabets.rs``). For the seven
  H/P alphabets the class letter (h, p or c) is taken from PR 116's
  ``tables/hp_borderline_residues.csv``, and every one of its 140 rows is checked against the
  class table here.
* ``evidence.parquet``: evidence per aligned position as a ratio to the 20 amino acids, with
  its 95% interval, for every alphabet, group and pair selection of notebook 291 (PR 99). The
  numbers are notebook 291's own stored table (``291_evidence_ratio_...parquet``). That table
  has no counts, so this script rebuilds notebook 291's groups from the same alignments with
  the same code, adds the number of pairs (or pair-halves) and clusters per group, and checks
  that its own mean evidence per alphabet equals notebook 291's to 1e-12.
* ``kappa.parquet``: agreement, chance and kappa per alphabet and identity bin for the Pfam
  seed pairs, with the number of pairs: notebook 230 section 2b's ``kappa_parts``, which the
  notebook prints but does not store. Recomputed here with the notebook's code.
* ``usable_k.parquet``: the usable seed lengths k per alphabet and identity bin: notebook 230
  section 2c's ``usable`` table, next to notebook 274's k_min to k_max range (where a seed
  matches no more than 100 human proteins by chance, up to one chance match in Swiss-Prot).
  Also printed but not stored by the notebook, so recomputed here with its code.

``provenance.json`` records the sha256 of every input and output, and the commit of each
notebook and table the numbers come from.

Usage (from the repo root; reads about 100 MB of parquet from ~/data, peak memory about 3 GB):
    python scripts/build_leaderboard_alphabets.py --outdir results/leaderboard/alphabets
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "notebooks"))
import hp_conservation_utils as hc  # noqa: E402

DATA = Path.home() / "data"
PFAM_ALN = DATA / "pfam/230_pfam_seed_pair_alignments.parquet"
SCOPE_ALN = DATA / "scope/230_scope40_pair_alignments.parquet"
PFAM_230 = DATA / "pfam/230_pfam_seed_pair_class_agreement.parquet"
SCOPE_FA = DATA / "scope/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa"
EVIDENCE_291 = DATA / "pfam/291_evidence_ratio_19_alphabets_pfam_seed_scope40.parquet"
K274 = REPO / "tables/274_ksizes_to_test_per_alphabet_human_swissprot.csv"
HP116 = REPO / "tables/hp_borderline_residues.csv"
NB230 = "notebooks/230_hp_class_conservation_in_aligned_homologs.ipynb"
NB291 = "notebooks/291_evidence_per_position_19_alphabets_pfam_seed_scope40.ipynb"

# Same constants as notebooks 230 and 291.
MIN_COLS = 50
MIN_TM = 0.5
LOW = 0.20
P20 = "protein20"
BINS_K = ["20-30%", "30-40%", "40-60%"]
KS_ALL = list(range(3, 41))
MOST_PAIRS = 0.20
FEW_PAIRS = 0.05
CAT_LABEL = {
    "same_family": "SCOPe: same family",
    "same_superfamily_diff_family": "SCOPe: same superfamily, different family",
    "same_fold_diff_superfamily": "SCOPe: same fold, different superfamily",
    "diff_fold_same_class": "SCOPe: different fold, same class",
}
# SCOPe calls two domains homologs only when they share a superfamily.
HOMOLOG_GROUP = {"Pfam seed": True} | {
    v: k.startswith("same_family") or k.startswith("same_super")
    for k, v in CAT_LABEL.items()
}
MODE_LABEL = {
    "low_half": "20% identity or less, grouped on one half of the positions, measured on the other",
    "all": "all identities, whole pairs",
    "low_same": "20% identity or less, grouped and measured on the same positions (biased)",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit(path: str) -> dict:
    out = subprocess.run(
        ["git", "-C", str(REPO), "log", "-1", "--format=%H %cI", "--", path],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return {"path": path, "commit": out[0], "commit_date": out[1]}


# ---------------------------------------------------------------- classes
def build_classes() -> pl.DataFrame:
    clusters = hc.ALPHABET_CLUSTERS
    assert len(clusters) == 19, sorted(clusters)
    hp = pl.read_csv(HP116)
    assert hp.height == 140, hp.height
    letter = {
        (r["alphabet"], r["residue"]): r["class"] for r in hp.iter_rows(named=True)
    }
    rows = []
    for a, cl in clusters.items():
        assert sorted("".join(cl)) == sorted(hc.AA20), a
        for members in cl:
            for r in members:
                rows.append(
                    dict(
                        alphabet=a,
                        n_classes=len(cl),
                        residue=r,
                        class_members="".join(sorted(members)),
                        hp_class=letter.get((a, r)),
                    )
                )
    df = pl.DataFrame(rows, infer_schema_length=None)
    # PR 116 check: the 140 H/P squares put the same residues together as the class table here.
    hp_rows = df.filter(pl.col("hp_class").is_not_null())
    assert hp_rows.height == 140, hp_rows.height
    for (a, members), g in hp_rows.group_by("alphabet", "class_members"):
        assert g["hp_class"].n_unique() == 1, (a, members)
    for a in hp_rows["alphabet"].unique():
        g = hp_rows.filter(pl.col("alphabet") == a)
        assert g["class_members"].n_unique() == g["hp_class"].n_unique(), a
    return df.sort("alphabet", "residue")


# ---------------------------------------------------------------- evidence (notebook 291)
def build_evidence() -> pl.DataFrame:
    alphabets = sorted(
        hc.ALPHABET_CLUSTERS, key=lambda a: (-len(hc.ALPHABET_CLUSTERS[a]), a)
    )
    i20 = alphabets.index(P20)

    def evidence_all(q, t, group, n_groups):
        out = [hc.grouped_evidence(q, t, group, n_groups, a) for a in alphabets]
        return np.stack([o[1] for o in out]), np.stack([o[3] for o in out])

    pf_aln = pl.read_parquet(PFAM_ALN)
    pq, pt, pidx, ppos, pkept = hc.aligned_codes(pf_aln, MIN_COLS)
    pf_pairs = pf_aln[pkept].select("family")
    del pf_aln
    n_pf = pf_pairs.height
    pf_agree, pf_I = evidence_all(pq, pt, pidx, n_pf)
    pf_ah, pf_Ih = evidence_all(pq, pt, pidx * 2 + ppos % 2, 2 * n_pf)
    pf_ah, pf_Ih = pf_ah.reshape(-1, n_pf, 2), pf_Ih.reshape(-1, n_pf, 2)
    del pq, pt, pidx, ppos

    fold_of = {}
    for line in open(SCOPE_FA):
        if line.startswith(">"):
            f = line[1:].split()
            fold_of[f[0]] = ".".join(f[1].split(".")[:2])
    sc_aln = pl.read_parquet(SCOPE_ALN).filter(
        pl.max_horizontal("tm_q", "tm_t") >= MIN_TM
    )
    sq, st, sidx, spos, skept = hc.aligned_codes(sc_aln, MIN_COLS)
    sc_pairs = sc_aln[skept].select("query", "category")
    n_sc = sc_pairs.height
    sc_agree, sc_I = evidence_all(sq, st, sidx, n_sc)
    sc_ah, sc_Ih = evidence_all(sq, st, sidx * 2 + spos % 2, 2 * n_sc)
    sc_ah, sc_Ih = sc_ah.reshape(-1, n_sc, 2), sc_Ih.reshape(-1, n_sc, 2)
    del sq, st, sidx, spos
    sc_fold = np.array([fold_of[q] for q in sc_pairs["query"]])
    sc_cat = sc_pairs["category"].to_numpy()
    pf_fam = pf_pairs["family"].to_numpy()

    def items(I, Ih, a20, ah20, clusters, subset):
        # notebook 291's items(): evidence values and cluster labels per pair selection
        out = {"all": (I[:, subset], clusters[subset])}
        m = subset & (a20 <= LOW)
        out["low_same"] = (I[:, m], clusters[m])
        vals, cl = [], []
        for grp, meas in ((0, 1), (1, 0)):
            m = subset & (ah20[:, grp] <= LOW)
            vals.append(Ih[:, m, meas])
            cl.append(clusters[m])
        out["low_half"] = (np.concatenate(vals, 1), np.concatenate(cl))
        for k, (v, c) in out.items():
            ok = ~np.isnan(v).any(0)
            out[k] = (v[:, ok], c[ok])
        return out

    groups = {
        "Pfam seed": items(
            pf_I, pf_Ih, pf_agree[i20], pf_ah[i20], pf_fam, np.ones(n_pf, bool)
        )
    }
    for c, label in CAT_LABEL.items():
        groups[label] = items(
            sc_I, sc_Ih, sc_agree[i20], sc_ah[i20], sc_fold, sc_cat == c
        )

    rows = []
    for g, modes in groups.items():
        for k, (v, c) in modes.items():
            mean = v.mean(1)
            for i, a in enumerate(alphabets):
                rows.append(
                    dict(
                        alphabet=a,
                        group=g,
                        mode=k,
                        I_bits_here=float(mean[i]),
                        ratio_here=float(mean[i] / mean[i20]),
                        n_items=int(v.shape[1]),
                        n_clusters=int(np.unique(c).size),
                    )
                )
    here = pl.DataFrame(rows)

    stored = pl.read_parquet(EVIDENCE_291)
    j = stored.join(here, on=["alphabet", "group", "mode"], how="full", coalesce=True)
    assert j.height == stored.height == here.height == 19 * 5 * 3, (
        j.height,
        stored.height,
        here.height,
    )
    d_I = (j["I_bits"] - j["I_bits_here"]).abs().max()
    d_r = (j["ratio_to_20aa"] - j["ratio_here"]).abs().max()
    assert d_I < 1e-12 and d_r < 1e-12, (d_I, d_r)
    print(
        f"evidence: notebook 291's stored means reproduced, largest difference {max(d_I, d_r):.1e}"
    )
    return (
        j.drop("I_bits_here", "ratio_here")
        .with_columns(
            pl.col("group")
            .replace_strict(HOMOLOG_GROUP, return_dtype=pl.Boolean)
            .alias("homologs_by_truth"),
            pl.col("mode").replace_strict(MODE_LABEL).alias("pairs_used"),
            pl.when(pl.col("mode") == "low_half")
            .then(pl.lit("pair-halves"))
            .otherwise(pl.lit("pairs"))
            .alias("item_kind"),
            pl.when(pl.col("group") == "Pfam seed")
            .then(pl.lit("Pfam families"))
            .otherwise(pl.lit("SCOPe folds of the query domain"))
            .alias("cluster_kind"),
        )
        .sort("group", "mode", "rank")
    )


# ---------------------------------------------------------------- kappa and usable k (notebook 230)
def build_kappa_and_k() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    # Only the columns used below, so the cross join with k stays small (notebook 230 joins
    # every column; the numbers are the same).
    cols = [
        "alphabet",
        "seqid_ali",
        "n_cols",
        "agree",
        "expected",
        "kappa",
        "longest_run",
    ]
    pfam = hc.add_identity_bin(pl.read_parquet(PFAM_230, columns=cols)).filter(
        pl.col("n_cols") >= MIN_COLS
    )
    kappa = (
        pfam.group_by("alphabet", "identity_bin")
        .agg(
            pl.len().alias("n_pairs"),
            pl.col("agree").mean().alias("agreement"),
            pl.col("expected").mean().alias("chance"),
            pl.col("kappa").mean().alias("kappa"),
        )
        .with_columns(pl.col("identity_bin").cast(pl.String))
        .sort("alphabet", "identity_bin")
    )
    assert kappa.height == 19 * 5, kappa.height
    k274 = pl.read_csv(K274).select("alphabet", "k_min", "k_max")
    # Notebook 230 cross-joins every pair with every k; counting pairs per longest run first
    # gives the same shares (a count over the same total) without the 60-million-row join.
    runs = (
        pfam.filter(pl.col("identity_bin").is_in(BINS_K))
        .group_by("alphabet", "identity_bin", "longest_run")
        .len()
    )
    reach = (
        runs.join(pl.DataFrame({"k": KS_ALL}), how="cross")
        .group_by("alphabet", "identity_bin", "k")
        .agg(
            (
                pl.col("len").filter(pl.col("longest_run") >= pl.col("k")).sum()
                / pl.col("len").sum()
            ).alias("share_reached"),
            pl.col("len").sum().cast(pl.UInt32).alias("n_pairs"),
        )
        .join(k274, on="alphabet")
        .with_columns(
            pl.col("k")
            .is_between(pl.col("k_min"), pl.col("k_max"))
            .alias("in_274_range")
        )
    )
    usable = (
        reach.filter(pl.col("in_274_range") & (pl.col("share_reached") >= FEW_PAIRS))
        .group_by("alphabet", "identity_bin")
        .agg(
            pl.col("n_pairs").first(),
            pl.col("k").min().alias("k_use_lo"),
            pl.col("k").max().alias("k_use_hi"),
            pl.col("k")
            .filter(pl.col("share_reached") >= MOST_PAIRS)
            .max()
            .alias("k_most_hi"),
        )
        .join(k274, on="alphabet")
        .with_columns(pl.col("identity_bin").cast(pl.String))
        .sort("alphabet", "identity_bin")
    )
    assert usable.height == 19 * len(BINS_K), usable.height
    # share of pairs reached at every k in notebook 274's range, for the hover text
    shares = (
        reach.filter(pl.col("in_274_range"))
        .with_columns(pl.col("identity_bin").cast(pl.String))
        .select("alphabet", "identity_bin", "k", "share_reached", "n_pairs")
        .sort("alphabet", "identity_bin", "k")
    )
    return kappa, usable, shares


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--outdir", type=Path, default=REPO / "results/leaderboard/alphabets"
    )
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    tables = {"classes": build_classes(), "evidence": build_evidence()}
    tables["kappa"], tables["usable_k"], tables["k_reach"] = build_kappa_and_k()
    for name, df in tables.items():
        df.write_parquet(args.outdir / f"{name}.parquet")
        print(f"{name}.parquet: {df.height} rows")

    inputs = [PFAM_ALN, SCOPE_ALN, PFAM_230, SCOPE_FA, EVIDENCE_291, K274, HP116]
    prov = {
        "script": "scripts/build_leaderboard_alphabets.py",
        "sources": {
            "evidence": git_commit(NB291)
            | {
                "pr": "https://github.com/seanome/2024-kmerseek-analysis/pull/99",
                "table": str(EVIDENCE_291).replace(str(Path.home()), "~"),
            },
            "kappa": git_commit(NB230)
            | {
                "pr": "https://github.com/seanome/2024-kmerseek-analysis/pull/100",
                "section": "2b, kappa_parts",
            },
            "usable_k": git_commit(NB230)
            | {
                "pr": "https://github.com/seanome/2024-kmerseek-analysis/pull/100",
                "section": "2c, usable",
            },
            "k_range": git_commit(str(K274.relative_to(REPO)))
            | {"pr": "https://github.com/seanome/2024-kmerseek-analysis/pull/101"},
            "hp_classes": git_commit(str(HP116.relative_to(REPO)))
            | {"pr": "https://github.com/seanome/2024-kmerseek-analysis/pull/116"},
            "classes": git_commit("notebooks/hp_conservation_utils.py"),
        },
        "inputs_sha256": {
            (
                str(p.relative_to(REPO))
                if p.is_relative_to(REPO)
                else str(p).replace(str(Path.home()), "~")
            ): sha256(p)
            for p in inputs
        },
        "outputs_sha256": {
            f"{n}.parquet": sha256(args.outdir / f"{n}.parquet") for n in tables
        },
    }
    (args.outdir / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    print("provenance.json written")


if __name__ == "__main__":
    main()
