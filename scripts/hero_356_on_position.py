#!/usr/bin/env python3
"""Notebook 356, Stop 2: are kmerseek's landing calls at the aligned position, and the hero.

Added after Stop 1, not in the pre-registration. The hero rule picked human COL9A1
(P20849) against chicken F1NQ20, where kmerseek's 18-residue call sits exactly on an exon
but pairs it with human residues 65 positions away from where a global alignment of the
two orthologs puts them: a different copy of the Gly-X-Y repeat. So this script asks, for
every verdict exon kmerseek lands on, whether any landing call is ON POSITION:

  global alignment  Biopython PairwiseAligner, global, BLOSUM62, gap open -11, extend -1,
                    of the human query and its 1:1 ortholog (whole proteins)
  on position       at least half of the call's residue pairs (query qstart + i facing
                    target tstart + i; kmerseek calls are gapless) are pairs the global
                    alignment also makes

kmerseek calls come from the cut call tables (every setting), 0-based end-exclusive; a call
lands as in scripts/score_356_exon_landing.py (inside >= 0.8, cover >= 0.3).

Hero rule (pre-registered part): a verdict exon (internal, <= 60 aa, both proteins with an
AlphaFold model), disordered (mean pLDDT < 70), where kmerseek's best of all lands with
p_place < 0.05 and every other tool in the rules (Foldseek, Reseek, ProstT5, phmmer,
MMseqs2, MMseqs2 iterative) spills; highest landing IoU, then smallest p_place, then
target accession. Added here: the landing call must also be on position. The rule's
first pick without that condition is kept as the cautionary case.

Writes:
  tables/356/landing_on_position.parquet  one row per verdict exon kmerseek lands on
  tables/356/hero.json                    both picks, their calls and residues
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import polars as pl
from Bio import Align
from Bio.Align import substitution_matrices

REPO = Path(__file__).resolve().parents[1]
QFO = Path.home() / (
    "data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143/Eukaryota"
)
FASTA = {
    "human": "UP000005640_9606",
    "chicken": "UP000000539_9031",
    "zebrafish": "UP000000437_7955",
}
CALLS = Path.home() / "data/356-exon-landing/out/calls"
TABLES = REPO / "tables" / "356"
INSIDE_MIN, COVER_MIN, P_PLACE_MAX = 0.8, 0.3, 0.05
ON_POSITION_MIN = 0.5
RULE_TOOLS = [
    "foldseek",
    "reseek",
    "prostt5",
    "hmmer3_phmmer",
    "mmseqs2_seqseq",
    "mmseqs2_iterative",
]


def read_fasta(path: Path, keep: set[str]) -> dict[str, str]:
    seqs, acc = {}, None
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                acc = line.split("|")[1]
                if acc in keep:
                    seqs[acc] = []
            elif acc in seqs:
                seqs[acc].append(line.strip())
    return {a: "".join(s) for a, s in seqs.items()}


def aligned_pairs(q: str, t: str, aligner) -> set[tuple[int, int]]:
    """(query index, target index) pairs of the best global alignment, 0-based."""
    a = aligner.align(q, t)[0]
    pairs = set()
    for (qs, qe), (ts, te) in zip(*a.aligned):
        pairs.update((qs + i, ts + i) for i in range(qe - qs))
    return pairs


def main():
    et = pl.read_parquet(TABLES / "exon_tool.parquet")
    verdict = et.filter(
        pl.col("verdict_set") & (pl.col("tool") == "kmerseek_best_of_all")
    )
    landed = verdict.filter("lands").select(
        "species",
        "query_acc",
        "target_acc",
        "exon_id",
        "prot_start",
        "prot_end",
        "target_len",
        "exon_plddt",
    )
    key = ["species", "query_acc", "target_acc"]
    calls = pl.concat(
        [
            pl.read_parquet(p)
            .select("query_acc", "target_acc", "qstart", "qend", "tstart", "tend")
            .with_columns(
                species=pl.lit(Path(p).stem.rsplit(".", 1)[1]),
                arm=pl.lit(Path(p).stem.split(".")[1]),
            )
            for p in sorted(glob.glob(str(CALLS / "kmerseek.*.parquet")))
            # An arm with no call on any pair was written as an empty table.
            if pl.read_parquet_schema(p).get("query_acc") is not None
        ],
        how="vertical_relaxed",
    ).join(landed.select(key).unique(), on=key, how="semi")
    L = pl.col("tend") - pl.col("tstart")
    ov = (
        pl.min_horizontal("tend", "prot_end")
        - pl.max_horizontal("tstart", "prot_start")
    ).clip(lower_bound=0)
    m = (
        calls.join(landed, on=key)
        .with_columns(ov=ov, L=L, E=pl.col("prot_end") - pl.col("prot_start"))
        .filter(
            (pl.col("ov") >= INSIDE_MIN * pl.col("L"))
            & (pl.col("ov") >= COVER_MIN * pl.col("E"))
        )
        .with_columns(iou=pl.col("ov") / (pl.col("L") + pl.col("E") - pl.col("ov")))
    )

    need = {
        "human": set(m["query_acc"]),
        "chicken": set(m.filter(pl.col("species") == "chicken")["target_acc"]),
        "zebrafish": set(m.filter(pl.col("species") == "zebrafish")["target_acc"]),
    }
    seqs = {sp: read_fasta(QFO / f"{FASTA[sp]}.fasta", need[sp]) for sp in FASTA}
    aligner = Align.PairwiseAligner(
        mode="global",
        substitution_matrix=substitution_matrices.load("BLOSUM62"),
        open_gap_score=-11,
        extend_gap_score=-1,
    )
    frac = []
    cache: dict[tuple[str, str], set] = {}
    for r in m.iter_rows(named=True):
        k = (r["query_acc"], r["target_acc"])
        if k not in cache:
            cache[k] = aligned_pairs(
                seqs["human"][r["query_acc"]],
                seqs[r["species"]][r["target_acc"]],
                aligner,
            )
        n = r["tend"] - r["tstart"]
        hit = sum((r["qstart"] + i, r["tstart"] + i) in cache[k] for i in range(n))
        frac.append(hit / n)
    m = m.with_columns(frac_on_position=pl.Series(frac)).with_columns(
        on_position=pl.col("frac_on_position") >= ON_POSITION_MIN
    )
    per_exon = m.group_by(key + ["exon_id"]).agg(
        n_landing_calls=pl.len(),
        n_settings_landing=pl.col("arm").n_unique(),
        any_on_position=pl.col("on_position").any(),
        n_on_position_calls=pl.col("on_position").sum(),
    )
    per_exon.write_parquet(TABLES / "landing_on_position.parquet")

    # Hero.
    t = et.filter(pl.col("verdict_set"))
    ekey = key + ["exon_id"]
    spill_all = (
        t.filter(pl.col("tool").is_in(RULE_TOOLS))
        .group_by(ekey)
        .agg(all_spill=(pl.col("outcome") == "spills").all(), n=pl.len())
        .filter(pl.col("all_spill") & (pl.col("n") == len(RULE_TOOLS)))
    )
    cand = (
        t.filter(
            (pl.col("tool") == "kmerseek_best_of_all")
            & pl.col("place_ok")
            & (pl.col("disorder") == "disordered")
        )
        .join(spill_all.select(ekey), on=ekey)
        .sort(["land_iou", "p_place", "target_acc"], descending=[True, False, False])
    )
    first = cand.row(0, named=True)
    on = cand.join(per_exon.filter("any_on_position").select(ekey), on=ekey, how="semi")
    hero = on.row(0, named=True)

    def describe(row: dict, require_on: bool) -> dict:
        c = m.filter(
            (pl.col("query_acc") == row["query_acc"])
            & (pl.col("target_acc") == row["target_acc"])
            & (pl.col("exon_id") == row["exon_id"])
        )
        if require_on:
            c = c.filter("on_position")
        c = c.sort(
            ["iou", "frac_on_position", "arm"], descending=[True, True, False]
        ).row(0, named=True)
        mapped = [
            ti
            for qi, ti in cache[(row["query_acc"], row["target_acc"])]
            if c["qstart"] <= qi < c["qend"]
        ]
        q = seqs["human"][row["query_acc"]][c["qstart"] : c["qend"]]
        tt = seqs[row["species"]][row["target_acc"]][c["tstart"] : c["tend"]]
        return {
            "species": row["species"],
            "query_acc": row["query_acc"],
            "target_acc": row["target_acc"],
            "exon_id": row["exon_id"],
            "exon_start_0based": row["prot_start"],
            "exon_end_0based_exclusive": row["prot_end"],
            "exon_len": row["exon_len"],
            "exon_plddt": row["exon_plddt"],
            "target_len": row["target_len"],
            "query_len": len(seqs["human"][row["query_acc"]]),
            "arm": c["arm"],
            "qstart": c["qstart"],
            "qend": c["qend"],
            "tstart": c["tstart"],
            "tend": c["tend"],
            "iou": c["iou"],
            "frac_on_position": c["frac_on_position"],
            "query_residues": q,
            "target_residues": tt,
            "n_identical": sum(a == b for a, b in zip(q, tt)),
            # Where the global alignment puts the call's human residues on the target.
            "aligned_target_start_1based": int(min(mapped)) + 1 if mapped else None,
            "aligned_target_end_1based": int(max(mapped)) + 1 if mapped else None,
        }

    out = {
        "n_verdict_exons_landed": landed.height,
        "n_landed_with_a_call_on_position": int(per_exon["any_on_position"].sum()),
        "n_hero_candidates_rule": cand.height,
        "n_hero_candidates_on_position": on.height,
        "rule_first_pick": describe(first, False),
        "hero_on_position": describe(hero, True),
    }
    (TABLES / "hero.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
