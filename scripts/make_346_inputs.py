#!/usr/bin/env python3
"""Inputs of notebook 346: earlier numbers, the six known pairs, sequences and databases.

Notebook 346 quotes numbers from notebooks 241, 245 and 294. This script reads each one
back out of the executed notebook (or the table it wrote), so no number is typed by hand,
and records where it came from. A few numbers in the task brief come from notes of work
whose scripts are not in this repository (the SCOPe rescue test of 2026-10-06, and the
phmmer and jackhmmer E-values); they are kept with kind "from notes, not reproduced here".

It also picks pairs 3 to 6 from notebook 294's report half by the rule written in the
task brief, before any database search, and writes every sequence the searches use.

Outputs, under tables/:

* 346_inputs.tsv: quantity, value, unit, kind, source notebook, cell.
* 346_known_pairs.tsv: the six pairs, which protein is the query, which the partner, and
  for pairs 3 to 6 the Pfam family, identity, alignment and notebook 294's scores.
* 346_sequences.fasta: every query and partner sequence, named as the tables name them.
* 346_databases.tsv: each database searched, its file, number of proteins and residues.

Run from the repository root with the project interpreter:

    /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 scripts/make_346_inputs.py
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
import hp_conservation_utils as hc  # noqa: E402
import linked_pairs_utils as lp  # noqa: E402

TABLES = ROOT / "tables"
NB241 = ROOT / "notebooks/241_alphabet_ranking_BCL2-Ced9_P66-CD47_BHF.ipynb"
NB294 = ROOT / "notebooks/294_linked_hp_word_pairs_kill_test_pfam_seed.ipynb"
SCORES_294 = Path("/Users/olga/data/pfam/294_linked_pairs_scores.parquet")
ALN_230 = Path("/Users/olga/data/pfam/230_pfam_seed_pair_alignments.parquet")
ALIGNERS_233 = Path(
    "/Users/olga/data/pfam/233_aligner_check/233_aligner_pair_evalues.parquet"
)
PFAM_SEED = Path("/Users/olga/data/pfam/Pfam-A.seed.gz")

HUMAN = Path(
    "/Users/olga/data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa"
)
ELEGANS = Path("/Users/olga/data/uniprot/reference_proteomes/UP000001940_6239.fasta.gz")
BORRELIA = Path(
    "/Users/olga/data/uniprot/reference_proteomes/UP000001807_224326.fasta.gz"
)
SWISSPROT = Path("/Users/olga/data/uniprot/uniprot_sprot.fasta.gz")
SCOPE40 = Path("/Users/olga/data/scope/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa")
QUERIES_241 = ROOT / "notebooks/241_queries.fa"

CHOSEN = lp.setting_name("all", lp.DB_ANCHOR, lp.DB_WINDOW, "none")  # all_a6_g32_none
CHOSEN_RAND = lp.setting_name("rand", lp.DB_ANCHOR, lp.DB_WINDOW, "none")
FPR = 1e-3
E_ALIGNER = 0.01  # notebook 294 section 4c: an aligner finds a pair at E <= 0.01
SIRPA_LOOP_MATURE = (
    181,
    187,
)  # P66 residues binding SIRPα, mature numbering (task brief)
SIRPA_LOOP_RESIDUES = "QENDKDT"  # checked against the sequence below
P66_SIGNAL_LENGTH = (
    21  # notebook 241 searched P66 residues 22-618 (241_alphabet_ranking.sbatch)
)


def read_fasta(path: Path) -> dict[str, str]:
    opener = gzip.open if str(path).endswith(".gz") else open
    seqs: dict[str, list[str]] = {}
    name = None
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                name = line[1:].strip()
                seqs[name] = []
            elif name is not None:
                seqs[name].append(line.strip())
    return {k: "".join(v) for k, v in seqs.items()}


def cell_lines(path: Path):
    """(cell index, line) for every line a code cell printed."""
    nb = json.loads(path.read_text())
    for i, cell in enumerate(nb["cells"]):
        for out in cell.get("outputs", []):
            text = out.get("text") or out.get("data", {}).get("text/plain") or ""
            text = "".join(text) if isinstance(text, list) else text
            for line in text.splitlines():
                yield i, line


def one_match(path: Path, pattern: str):
    """The single printed line matching pattern: (cell index, match)."""
    hits = [(i, m) for i, ln in cell_lines(path) if (m := re.search(pattern, ln))]
    if len(hits) != 1:
        raise ValueError(f"{len(hits)} lines match {pattern!r} in {path.name}")
    return hits[0]


def table_cells(line: str) -> list[str]:
    return [c.strip() for c in re.split(r"[│┆]", line)[1:-1]]


def main() -> None:
    rows: list[dict] = []

    def add(quantity, value, unit, kind, source, cell=""):
        rows.append(
            {
                "quantity": quantity,
                "value": float(value),
                "unit": unit,
                "kind": kind,
                "source": source,
                "cell": str(cell),
            }
        )

    # --- notebook 245, from the tables it wrote (each row names its own origin) ---------
    t245 = pl.read_csv(TABLES / "245_inputs.tsv", separator="\t")
    v245 = dict(zip(t245["quantity"], t245["value"]))
    src245 = dict(zip(t245["quantity"], t245["source"]))
    for q, unit in [
        ("kappa_hp_thomas_dill2_pfam_20_30", "none"),
        ("best_rank_BCL2", "rank"),
        ("n_targets_at_best_BCL2", "proteins"),
        ("best_rank_CD47", "rank"),
        ("n_targets_at_best_CD47", "proteins"),
        ("K_proteome", "none"),
        ("n_residues_proteome", "residues"),
    ]:
        add(
            q,
            v245[q],
            unit,
            (
                "measured"
                if "rank" in q or "kappa" in q or "targets" in q
                else "approximate"
            ),
            f"tables/245_inputs.tsv ({src245[q]})",
            "",
        )
    cs = pl.read_csv(TABLES / "245_chance_score_by_alphabet.tsv", separator="\t")
    add(
        "pr_match_unrelated_hp_thomas_dill2",
        cs.filter(pl.col("alphabet") == lp.ALPHABET)["pr_match_unrelated"][0],
        "probability",
        "measured",
        "tables/245_chance_score_by_alphabet.tsv (nb 241)",
        "",
    )

    # --- notebook 241: BCL2's rank by E-value (wwmj5, k = 9) --------------------------
    cell, m = one_match(NB241, r"│ Ced9\s+┆ wwmj5\s+┆ 9\s+┆ (\d+)\s+┆ (\d+)\s+┆")
    add(
        "evalue_rank_BCL2_wwmj5_k9",
        m.group(1),
        "rank",
        "measured",
        "nb 241, Ced9 -> human, wwmj5 k = 9, rank by region_evalue",
        cell,
    )
    add(
        "n_hit_wwmj5_k9",
        m.group(2),
        "proteins",
        "measured",
        "nb 241, Ced9 -> human, wwmj5 k = 9, human proteins hit",
        cell,
    )

    # --- notebook 294 ----------------------------------------------------------------
    for label, q in [
        ("chosen linked pair minus the b", "gain_linked_minus_B2_20_30"),
        ("chosen linked pair minus B3 (s", "gain_linked_minus_B3_20_30"),
        ("B3 minus the better single see", "gain_B3_minus_B2_20_30"),
    ]:
        cell, m = one_match(
            NB294,
            re.escape(label) + r"…?\s*┆\s*(-?[\d.]+)\s*┆\s*(-?[\d.]+)\s*┆\s*(-?[\d.]+)",
        )
        if cell != 11:
            raise ValueError(
                f"{label}: expected cell 11 (shuffled-target null), got {cell}"
            )
        for part, val in zip(["", "_ci_low", "_ci_high"], m.groups()):
            add(
                q + part,
                val,
                "points of recall",
                "measured",
                "nb 294 section 4b, report half, exact FPR 0.001, families resampled",
                cell,
            )
    cell, m = one_match(
        NB294, r"chance key matches per comparison of two (\d+)-residue sequences"
    )
    line = [
        ln for i, ln in cell_lines(NB294) if i == cell and "chance key matches" in ln
    ][0]
    lk = re.search(r"; ([\d.]+) chance key matches", line).group(1)
    b1 = re.search(r"B1's 10-mer [\d.]+ bits, ([\d.]+)", line).group(1)
    b2 = re.search(r"B2's 8-mer [\d.]+ bits, ([\d.]+)", line).group(1)
    src = "nb 294 section 4d, Swiss-Prot word frequencies, one diagonal of two unrelated sequences"
    add(
        "chance_keys_linked_per_131",
        lk,
        "key matches per comparison",
        "measured",
        src,
        cell,
    )
    add(
        "chance_keys_B2_per_131",
        b2,
        "key matches per comparison",
        "measured",
        src,
        cell,
    )
    add(
        "chance_keys_B1_per_131",
        b1,
        "key matches per comparison",
        "measured",
        src,
        cell,
    )
    add("chance_keys_pair_length", m.group(1), "residues", "measured", src, cell)
    cell, m = one_match(
        NB294,
        r"False pairs per query at 0\.001 per comparison against Swiss-Prot: ([\d,_]+)",
    )
    add(
        "false_pairs_per_query_swissprot_at_0.001",
        m.group(1).replace(",", "").replace("_", ""),
        "proteins",
        "computed",
        "nb 294 section 4f",
        cell,
    )

    # --- pairs 3 to 6, picked from notebook 294's report half --------------------------
    d = hc.add_identity_bin(pl.read_parquet(SCORES_294))
    linked = [lp.setting_name(*s) for s in lp.ALL_SETTINGS]
    cols = (
        ["B1_ext", "B2_ext", "B3_ext", "seed10_count", "seed8_count"]
        + [f"{s}_ext" for s in linked]
        + [f"{s}_count" for s in linked]
    )
    real_all, null_all, bin_all, pidx_all = lp.score_arrays(d, cols)
    rep = ~np.isin(pidx_all, lp.select_half_ids(pidx_all))
    # notebook 294 section 4a: thresholds on the report half, 232's rule
    ext = ["B1_ext", "B2_ext"] + [f"{x}_ext" for x in linked] + [f"{CHOSEN}_count"]
    idx = [cols.index(c) for c in ext]
    thr, _, _ = lp.threshold_and_recall(
        real_all[rep][:, idx], null_all[rep][:, :, idx], bin_all[rep], FPR
    )
    thr_b2, thr_linked = thr[ext.index("B2_ext")], thr[ext.index(f"{CHOSEN}_ext")]
    add(
        "threshold_B2_report_half",
        thr_b2,
        "score",
        "recomputed here",
        "nb 294 section 4a rule, on 294_linked_pairs_scores.parquet",
        "",
    )
    add(
        "threshold_linked_report_half",
        thr_linked,
        "score",
        "recomputed here",
        "nb 294 section 4a rule, on 294_linked_pairs_scores.parquet",
        "",
    )
    al = (
        pl.read_parquet(ALIGNERS_233)
        .filter(pl.col("dataset") == "pfam")
        .select("query", "target", "phmmer_evalue", "mmseqs_evalue")
    )
    rep_ids = pidx_all[rep].tolist()
    b23 = (
        d.filter(
            (pl.col("kind") == "real")
            & pl.col("pair_index").is_in(rep_ids)
            & (pl.col("identity_bin") == "20-30%")
        )
        .join(al, on=["query", "target"], how="left")
        .with_columns(
            (pl.col("B2_ext") > thr_b2).alias("B2_hit"),
            (pl.col(f"{CHOSEN}_ext") > thr_linked).alias("linked_hit"),
            (
                (pl.col("phmmer_evalue") <= E_ALIGNER).fill_null(False)
                | (pl.col("mmseqs_evalue") <= E_ALIGNER).fill_null(False)
            ).alias("aligner_hit"),
            (pl.col(f"{CHOSEN}_ext") - pl.col("B2_ext")).alias("gain_linked_minus_B2"),
        )
    )
    only = b23.filter(pl.col("linked_hit") & ~pl.col("aligner_hit"))
    add(
        "n_linked_only_not_aligners_20_30",
        only.height,
        "pairs",
        "recomputed here",
        "nb 294 section 4c rule, report half, 20-30% identity",
        "",
    )
    # Rule (task brief): three pairs with the largest linked - B2 score gain where B2 missed
    # and the linked key hit; ties go to the higher linked score, then the lower pair index.
    gain = (
        b23.filter(pl.col("linked_hit") & ~pl.col("B2_hit"))
        .sort(
            ["gain_linked_minus_B2", f"{CHOSEN}_ext", "pair_index"],
            descending=[True, True, False],
        )
        .head(3)
        .with_columns(
            pl.lit("largest linked minus B2 gain, B2 missed").alias("why_chosen")
        )
    )
    top = (
        only.filter(~pl.col("pair_index").is_in(gain["pair_index"].to_list()))
        .sort([f"{CHOSEN}_ext", "pair_index"], descending=[True, False])
        .head(1)
        .with_columns(
            pl.lit("highest linked score, found by neither phmmer nor MMseqs2").alias(
                "why_chosen"
            )
        )
    )
    pfam = pl.concat([gain, top])
    names = pfam_names(set(pfam["family"].to_list()))
    aln = pl.read_parquet(ALN_230).select("family", "query", "target", "qaln", "taln")
    pfam = (
        pfam.join(aln, on=["family", "query", "target"], how="left")
        .with_columns(pl.col("family").replace_strict(names).alias("family_name"))
        .with_columns(pl.Series("pair", [f"pair {i}" for i in range(3, 7)]))
        .select(
            "pair",
            "pair_index",
            "family",
            "family_name",
            "query",
            "target",
            "seqid_ali",
            "why_chosen",
            "B2_ext",
            "B3_ext",
            f"{CHOSEN}_ext",
            f"{CHOSEN}_count",
            f"{CHOSEN_RAND}_ext",
            f"{CHOSEN_RAND}_count",
            "phmmer_evalue",
            "mmseqs_evalue",
            "qaln",
            "taln",
        )
    )
    assert pfam.height == 4 and pfam["qaln"].null_count() == 0

    # --- sequences ---------------------------------------------------------------------
    human = read_fasta(HUMAN)
    human_by_gene = {h.split("|")[6]: (h, s) for h, s in human.items()}
    elegans = read_fasta(ELEGANS)
    borrelia = read_fasta(BORRELIA)
    q241 = read_fasta(QUERIES_241)
    scope = read_fasta(SCOPE40)

    def by_acc(db, acc):
        hits = [(h, s) for h, s in db.items() if f"|{acc}|" in h]
        assert len(hits) == 1, (acc, len(hits))
        return hits[0]

    ced9_h, ced9 = by_acc(elegans, "P41958")
    p66_h, p66_pre = by_acc(borrelia, "H7C7N8")
    bcl2_h, bcl2 = human_by_gene["BCL2"]
    cd47_h, cd47 = human_by_gene["CD47"]
    sprot = read_fasta(SWISSPROT)
    for acc, seq in [("P10415", bcl2), ("P41958", ced9), ("Q08722", cd47)]:
        assert by_acc(sprot, acc)[1] == seq, f"{acc} differs from Swiss-Prot"
    assert q241["Ced9"] == ced9
    p66 = q241["P66"]
    assert p66_pre[P66_SIGNAL_LENGTH:] == p66, "241's P66 is not H7C7N8 22-618"
    lo, hi = SIRPA_LOOP_MATURE
    assert p66[lo - 1 : hi] == SIRPA_LOOP_RESIDUES
    assert (
        p66_pre[lo - 1 + P66_SIGNAL_LENGTH : hi + P66_SIGNAL_LENGTH]
        == SIRPA_LOOP_RESIDUES
    )
    d1ohu = [(h, s) for h, s in scope.items() if h.startswith("d1ohua_")]
    assert len(d1ohu) == 1
    d1ohu_h, d1ohu_seq = d1ohu[0][0], d1ohu[0][1].upper()
    # The crystal construct differs from CED-9 at a few residues (cysteines changed to
    # serine); find its offset in CED-9 by the best gapless match and count the differences.
    n = len(d1ohu_seq)
    diffs = [
        sum(a != b for a, b in zip(d1ohu_seq, ced9[o : o + n]))
        for o in range(len(ced9) - n + 1)
    ]
    d1ohu_offset = int(np.argmin(diffs))
    d1ohu_diff = [
        f"{a}{d1ohu_offset + i + 1}{b}"
        for i, (a, b) in enumerate(
            zip(ced9[d1ohu_offset : d1ohu_offset + n], d1ohu_seq)
        )
        if a != b
    ]
    assert len(d1ohu_diff) <= 5, d1ohu_diff

    seqs = {
        "CED-9": ced9,
        "d1ohua_": d1ohu_seq,
        "BCL-2": bcl2,
        "P66_mature": p66,
        "P66_precursor": p66_pre,
        "CD47": cd47,
    }
    for r in pfam.iter_rows(named=True):
        seqs[r["query"]] = r["qaln"].replace("-", "")
        seqs[r["target"]] = r["taln"].replace("-", "")
    with open(TABLES / "346_sequences.fasta", "w") as fh:
        for k, s in seqs.items():
            fh.write(
                f">{k}\n"
                + "\n".join(s[i : i + 60] for i in range(0, len(s), 60))
                + "\n"
            )

    for name, header, s, src in [
        ("CED-9", ced9_h, ced9, ELEGANS),
        ("d1ohua_", d1ohu_h, d1ohu_seq, SCOPE40),
        ("BCL-2", bcl2_h, bcl2, HUMAN),
        ("CD47", cd47_h, cd47, HUMAN),
        ("P66_precursor", p66_h, p66_pre, BORRELIA),
    ]:
        add(
            f"length_{name}",
            len(s),
            "residues",
            "counted here",
            f"{src.name} (file date {dt.date.fromtimestamp(src.stat().st_mtime)}): {header[:80]}",
            "",
        )
    add(
        "d1ohua_start_in_CED9",
        d1ohu_offset + 1,
        "residue",
        "counted here",
        f"best gapless match of d1ohua_ to P41958; differences: {', '.join(d1ohu_diff)}",
        "",
    )
    add(
        "d1ohua_n_differences_from_CED9",
        len(d1ohu_diff),
        "residues",
        "counted here",
        f"{', '.join(d1ohu_diff)} (CED-9 residue, position, d1ohua_ residue)",
        "",
    )
    add(
        "length_P66_mature",
        len(p66),
        "residues",
        "counted here",
        "notebooks/241_queries.fa (H7C7N8 residues 22-618)",
        "",
    )
    add(
        "P66_signal_peptide_length",
        P66_SIGNAL_LENGTH,
        "residues",
        "from the 241 search setup",
        "notebooks/241_alphabet_ranking.sbatch comment; checked: 241's P66 = H7C7N8[22:618]",
        "",
    )
    # --- databases ---------------------------------------------------------------------
    dbs = []
    for name, path, n_seqs in [
        ("human GENCODE v49 canonical", HUMAN, human),
        ("C. elegans UniProt UP000001940", ELEGANS, elegans),
        ("B. burgdorferi B31 UniProt UP000001807", BORRELIA, borrelia),
        ("Swiss-Prot", SWISSPROT, sprot),
    ]:
        dbs.append(
            {
                "database": name,
                "file": str(path),
                "n_proteins": len(n_seqs),
                "n_residues": sum(len(s) for s in n_seqs.values()),
                "file_date": dt.date.fromtimestamp(path.stat().st_mtime).isoformat(),
            }
        )
    pl.DataFrame(dbs).write_csv(TABLES / "346_databases.tsv", separator="\t")

    # --- the six pairs, in one table ----------------------------------------------------
    known = pl.DataFrame(
        [
            {
                "pair": "pair 1",
                "query": "CED-9",
                "partner": bcl2_h,
                "query_db": "human GENCODE v49 canonical",
                "expected": "true homologs (SCOP fold f.1); expected negative for an HP key",
            },
            {
                "pair": "pair 1, SCOPe domain",
                "query": "d1ohua_",
                "partner": bcl2_h,
                "query_db": "human GENCODE v49 canonical",
                "expected": "as pair 1, CED-9's SCOPe domain as query",
            },
            {
                "pair": "pair 1, reverse",
                "query": "BCL-2",
                "partner": ced9_h,
                "query_db": "C. elegans UniProt UP000001940",
                "expected": "as pair 1, other direction",
            },
            {
                "pair": "pair 2",
                "query": "P66_mature",
                "partner": cd47_h,
                "query_db": "human GENCODE v49 canonical",
                "expected": "no significant similarity reported; negative control",
            },
            {
                "pair": "pair 2, reverse",
                "query": "CD47",
                "partner": p66_h,
                "query_db": "B. burgdorferi B31 UniProt UP000001807",
                "expected": "as pair 2, other direction",
            },
        ]
        + [
            {
                "pair": r["pair"],
                "query": r["query"],
                "partner": r["target"],
                "query_db": "Swiss-Prot",
                "expected": f"Pfam {r['family']} pair at {100 * r['seqid_ali']:.1f}% identity; {r['why_chosen']}",
            }
            for r in pfam.iter_rows(named=True)
        ]
    )
    known.write_csv(TABLES / "346_known_pairs.tsv", separator="\t")
    pfam.write_csv(TABLES / "346_pfam_pairs.tsv", separator="\t")

    # --- numbers from notes, not reproduced here ---------------------------------------
    notes = "task brief 2026-10-18; from notes, not reproduced here"
    for q, val, unit in [
        ("phmmer_evalue_BCL2_from_CED9_human", 7.9e-5, "E"),
        ("jackhmmer_evalue_BCL2_from_CED9_human", 7.7e-41, "E"),
        ("sw_blosum62_rank_BCL2_scope40", 2, "rank"),
        ("sw_blosum62_n_scope40", 14_280, "domains"),
    ]:
        add(q, val, unit, "from notes, not reproduced here", notes, "")
    pl.DataFrame(rows).write_csv(TABLES / "346_inputs.tsv", separator="\t")
    print(pl.DataFrame(rows).select("quantity", "value", "kind", "cell"))
    print(
        pfam.select(
            "pair",
            "pair_index",
            "family",
            "family_name",
            "seqid_ali",
            "B2_ext",
            "B3_ext",
            f"{CHOSEN}_ext",
        )
    )
    print(pl.DataFrame(dbs))


def pfam_names(families: set[str]) -> dict[str, str]:
    """Pfam accession (no version) -> its description line (#=GF DE) in Pfam-A.seed."""
    out, acc = {}, None
    with gzip.open(PFAM_SEED, "rb") as fh:
        for line in fh:
            if line.startswith(b"#=GF AC"):
                acc = line.split()[2].decode().split(".")[0]
            elif line.startswith(b"#=GF DE") and acc in families:
                out[acc] = line[10:].decode().strip()
            elif line.startswith(b"//"):
                acc = None
    assert set(out) == families, families - set(out)
    return out


if __name__ == "__main__":
    main()
