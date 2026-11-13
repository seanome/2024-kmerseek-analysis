#!/usr/bin/env python3
"""Notebook 285, part 2: Ced9's calls on the BCL-2 family, placed on the Pfam Bcl-2 model's columns.

Part 1 (scripts/285_family_pooled_evidence.py) called two calls "at the same position" when
they cover the same Ced9 residue, whatever residue of the member they land on. Here a
position is a column of the Pfam Bcl-2 profile HMM (PF00452.25, 100 columns), which places
every family member's residues in one alignment.

Rule, fixed before this script was first run (committed before the run):

  1. Columns. Ced9 (notebook 241's queries.fa) and the nine BCL-2 family members (GENCODE
     v49 canonical proteins) are searched against the Bcl-2 model with hmmsearch (HMMER 3.4,
     --domE 10 --incdomE 10 so a weak Ced9 domain is kept). From the domain alignments,
     each residue gets its match column (1-100); insert residues and residues outside the
     domain get none. The model's own E-value for each protein is reported.
  2. Same position. A call is two ungapped stretches, Ced9 residue q + j paired with member
     residue t + j. A pair is in column c when both residues sit in match column c.
  3. Member evidence at column c: the lowest per-protein p (as in part 1:
     p = 1 - exp(-E * L_member / L_database)) over that member's calls (any of the 152
     settings) with a pair in column c; p = 1 if none.
  4. Family score: Fisher's method over all nine members at each column, lowest p over the
     100 columns. Every call kept by this rule is also kept by part 1's rule at the same Ced9
     residue, so this score can only be weaker than part 1's.
  5. Null: the same calls with the member side shifted by d columns (a pair counts at
     column c when the Ced9 residue is in c and the member residue in c + d), for
     3 <= |d| <= 40 (76 shifts). |d| <= 2 is left out because calls one or two columns off
     can still be the true alignment across an insert.
  Pass if the unshifted score is better than at least 95% of the 76 shifted scores, and the
  family still ranks in the top 10 when its score is compared with every group's part 1 score.

Writes notebooks/285_pfam_columns_*.csv.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import chi2

REPO = Path(__file__).resolve().parents[1]
NB = REPO / "notebooks"
DATA = Path("/Users/olga/data/botryllus/alphabet-ranking-three-cases")
FASTA = Path("/Users/olga/data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa")
PFAM = Path("/Users/olga/data/pfam/Pfam-A.hmm")  # Pfam-A, Bcl-2 is PF00452.25
WORK = Path("/Users/olga/data/pfam/285")
HMMER = Path("/Users/olga/anaconda3/envs/hmmer/bin")
MEMBERS = ["BAK1", "BAX", "BCL2", "BCL2A1", "BCL2L1", "BCL2L10", "BCL2L2", "BOK", "MCL1"]
QUERY = "Ced9"
N_COLUMNS = 100
SHIFTS = [d for d in range(-40, 41) if abs(d) >= 3]
NULL_QUANTILE = 0.95
TOP_N = 10


def read_fasta(p: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    name = None
    for line in p.open():
        if line.startswith(">"):
            name = line[1:].strip()
            out[name] = ""
        else:
            out[name] += line.strip()
    return out


def run_hmmsearch(seqs: dict[str, str]) -> tuple[pl.DataFrame, dict[str, np.ndarray]]:
    """Domain hits and, per sequence, the match column of every residue (0 = none)."""
    WORK.mkdir(parents=True, exist_ok=True)
    hmm = WORK / "Bcl-2.hmm"
    if not hmm.exists():
        hmm.write_text(subprocess.run([HMMER / "hmmfetch", PFAM, "Bcl-2"], check=True,
                                      capture_output=True, text=True).stdout)
    fa = WORK / "ced9_bcl2_family.fa"
    fa.write_text("".join(f">{k}\n{v}\n" for k, v in seqs.items()))
    sto, dom = WORK / "ced9_bcl2_family.sto", WORK / "ced9_bcl2_family.domtbl"
    subprocess.run([HMMER / "hmmsearch", "--domE", "10", "--incdomE", "10", "-E", "10",
                    "--incE", "10", "-A", sto, "--domtblout", dom, "-o", "/dev/null", hmm, fa],
                   check=True)
    rows = []
    for line in dom.read_text().splitlines():
        if line.startswith("#"):
            continue
        f = line.split()
        rows.append({"protein": f[0], "full_evalue": float(f[6]), "domain_i_evalue": float(f[12]),
                     "domain_score": float(f[13]), "hmm_from": int(f[15]), "hmm_to": int(f[16]),
                     "ali_from": int(f[17]), "ali_to": int(f[18])})
    hits = pl.DataFrame(rows)
    # Stockholm: one line per domain, "name/start-end  aligned"; uppercase or '-' are match
    # columns, lowercase or '.' are inserts.
    aln: dict[str, str] = {}
    for line in sto.read_text().splitlines():
        if line and not line.startswith(("#", "//")):
            name, s = line.split()
            aln[name] = aln.get(name, "") + s
    cols = {k: np.zeros(len(v), dtype=np.int64) for k, v in seqs.items()}
    for name, s in aln.items():
        prot, start, _ = re.match(r"(.+)/(\d+)-(\d+)$", name).groups()
        # Each row spans all model columns (leading '-' are deletions), so counting starts at
        # column 1 whatever the hit's hmm_from.
        pos, col = int(start) - 1, 0  # 0-based residue index, model column
        for ch in s:
            if ch.isupper() or ch == "-":
                col += 1
            if ch.isalpha():
                if ch.isupper():
                    if cols[prot][pos]:
                        raise ValueError(f"{prot} residue {pos + 1} in two domains")
                    cols[prot][pos] = col
                pos += 1
        assert seqs[prot][int(start) - 1:pos] == re.sub(r"[-.]", "", s).upper()
        assert col == N_COLUMNS, f"{name}: {col} model columns"
    return hits, cols


def main() -> None:
    groups = pl.read_csv(NB / "285_paralog_groups_ensembl115.csv")
    l_db = int(groups["length"].sum())
    fam = groups.filter(pl.col("gene").is_in(MEMBERS))
    assert sorted(fam["gene"]) == MEMBERS
    tseqs = read_fasta(FASTA)
    seqs = {QUERY: read_fasta(DATA / "queries.fa")[QUERY]}
    seqs |= {g: tseqs[t] for g, t in zip(fam["gene"], fam["target_name"])}
    hits, cols = run_hmmsearch(seqs)
    hits.write_csv(NB / "285_pfam_columns_hmmsearch_hits.csv")
    print(hits)

    name_of = dict(zip(fam["target_name"], fam["gene"]))
    length = dict(zip(fam["gene"], fam["length"]))
    calls = (pl.scan_parquet(DATA / "regions.parquet")
             .filter((pl.col("query_name") == QUERY) & pl.col("target_name").is_in(list(name_of))
                     & pl.col("region_evalue").is_finite())
             .collect()
             .with_columns(gene=pl.col("target_name").replace_strict(name_of))
             .sort("gene", "alphabet", "ksize_arm", "region_start", "target_start"))
    qcol = cols[QUERY]

    # W[d][m, c]: strongest evidence (-ln p) member m has at column c+1 under shift d.
    shifts = [0] + SHIFTS
    W = {d: np.zeros((len(MEMBERS), N_COLUMNS)) for d in shifts}
    pair_rows = []
    for r in calls.iter_rows(named=True):
        m = MEMBERS.index(r["gene"])
        x = r["region_evalue"] * length[r["gene"]] / l_db
        w = -np.log(-np.expm1(-x))
        qs, ts, n = int(r["region_start"]), int(r["target_start"]), int(r["region_end"] - r["region_start"])
        qc = qcol[qs:qs + n]
        tc = cols[r["gene"]][ts:ts + n]
        ok = (qc > 0) & (tc > 0)
        for d in shifts:
            hit_cols = qc[ok & (tc == qc + d)]
            if hit_cols.size:
                np.maximum.at(W[d][m], hit_cols - 1, w)
        same = ok & (tc == qc)
        if same.any():
            pair_rows.append({"gene": r["gene"], "alphabet": r["alphabet"], "k": r["ksize_arm"],
                              "query_start0": qs, "query_end0": qs + n, "target_start0": ts,
                              "target_end0": ts + n, "evalue_database": r["region_evalue"],
                              "p_protein": float(-np.expm1(-x)), "neg_ln_p": w,
                              "n_pairs_same_column": int(same.sum()),
                              "first_column": int(qc[same].min()), "last_column": int(qc[same].max())})

    def family_score(Wd: np.ndarray) -> tuple[float, int]:
        lp = chi2.logsf(2 * Wd.sum(axis=0), 2 * len(MEMBERS)) / np.log(10)
        c = int(np.argmin(lp))
        return float(lp[c]), c + 1

    score, best_col = family_score(W[0])
    null = pl.DataFrame([{"shift": d, "log10_p": family_score(W[d])[0],
                          "best_column": family_score(W[d])[1]} for d in shifts])
    shifted = null.filter(pl.col("shift") != 0)["log10_p"].to_numpy()
    frac_shift = float((shifted <= score).mean())
    part1 = pl.read_csv(NB / "285_posthoc_copies_once.csv").filter(pl.col("query") == QUERY)
    others = part1.filter(pl.col("group_id") != fam["group_id"][0])["log10_p_rule"].to_numpy()
    rank = int((others < score).sum()) + 1

    per_col = pl.DataFrame({"column": np.arange(1, N_COLUMNS + 1),
                            "family_log10_p": chi2.logsf(2 * W[0].sum(axis=0), 2 * len(MEMBERS)) / np.log(10),
                            **{g: W[0][i] / np.log(10) for i, g in enumerate(MEMBERS)}})
    per_col.write_csv(NB / "285_pfam_columns_evidence.csv")
    null.write_csv(NB / "285_pfam_columns_shift_null.csv")
    pl.DataFrame(pair_rows).write_csv(NB / "285_pfam_columns_calls.csv")
    # residue -> column map for the ten proteins, so the notebook can draw the alignment
    pl.DataFrame([{"protein": k, "residue1": i + 1, "residue": seqs[k][i], "column": int(c)}
                  for k, v in cols.items() for i, c in enumerate(v)]).write_csv(
        NB / "285_pfam_columns_residue_map.csv")
    summary = {
        "query": QUERY, "model": "PF00452.25 Bcl-2", "n_members": len(MEMBERS),
        "column_rule_log10_p": score, "best_column": best_col,
        "members_with_evidence_at_best_column": int((W[0][:, best_col - 1] > 0).sum()),
        "part1_log10_p": float(part1.filter(pl.col("group_id") == fam["group_id"][0])["log10_p_rule"][0]),
        "rank_vs_part1_groups": rank, "n_groups": part1.height,
        "n_shifts": len(shifted), "shifts_as_good_or_better": int((shifted <= score).sum()),
        "shifted_median_log10_p": float(np.median(shifted)),
        "pass_shift_null": (1 - frac_shift) >= NULL_QUANTILE, "pass_top10": rank <= TOP_N,
    }
    pl.DataFrame([summary]).write_csv(NB / "285_pfam_columns_summary.csv")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
