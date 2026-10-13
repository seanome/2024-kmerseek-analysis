#!/usr/bin/env python3
"""Notebook 285: pool kmerseek calls across each human paralog group, at the same query residue.

Question: does Ced9's evidence on the BCL2 paralog group, pooled, rank that group near the
top of all human groups, and above what dipeptide-shuffled Ced9 gets? Same for P66 -> CD47.

Inputs
  regions   notebook 241's search: Ced9, P66, BHF against the 19_732 GENCODE v49 canonical
            human proteins, 152 alphabet-ksize settings, every region kept (0-based,
            end-exclusive query and target coordinates, ungapped).
  groups    notebooks/285_paralog_groups_ensembl115.csv (scripts/285_paralog_groups.py).

Rule, fixed before any group was scored (this file is committed before it is first run):

  1. Per call: a region with a finite E-value. kmerseek's region E-value is for the whole
     database (its search space is the same for every target), so the chance that one given
     protein t gets a call this good is p = 1 - exp(-E * L_t / L_db), with L_t the protein's
     length and L_db the total length of the 19_732 proteins. A region with no E-value fit
     (E = inf) gives no evidence (p = 1).
  2. Same position: the same query residue. For each protein and each residue i of the
     query, take the lowest p over every call (any of the 152 settings) whose query stretch
     covers i. A protein with no call over i has p = 1 there.
  3. Group score at residue i: Fisher's method over all n members of the group,
     X = -2 * sum(ln p_m), compared with a chi-squared distribution with 2n degrees of
     freedom. Members with no call count, with p = 1, so a large group is not rewarded for
     its size. The group's score is its lowest Fisher p over all query residues.
  4. The same score with each protein as its own group (n = 1) is the single-protein
     baseline: the protein's lowest per-protein p.
  5. Nulls. (a) 1_000 random sets of proteins of the same size as the case group, drawn
     from all 19_732, scored the same way on the real query. (b) The dipeptide-shuffled
     queries of notebooks/284_shuffle_agreement.py, scored the same way on the case group,
     if that run's regions exist.
  Decision: pooling rescues the pair if the case group ranks in the top 10 of the 7_212
  groups AND scores better than 95% of the random same-size sets AND better than 95% of
  the shuffled queries. If the shuffle regions do not exist, that part is reported pending.

Writes notebooks/285_*.csv (small tables the notebook reads and plots).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import chi2

REPO = Path(__file__).resolve().parents[1]
NB = REPO / "notebooks"
DATA = Path("/Users/olga/data/botryllus/alphabet-ranking-three-cases")
FASTA = Path("/Users/olga/data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa")
GROUPS = NB / "285_paralog_groups_ensembl115.csv"
CASES = {"Ced9": "BCL2", "P66": "CD47"}
N_RANDOM = 1_000
SEED = 285
TOP_N = 10  # the decision rule's "top 10 groups"
NULL_QUANTILE = 0.95  # the decision rule's "better than 95% of" each null


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


class Scorer:
    """Per-protein, per-query-residue evidence for one query, and group scores from it."""

    def __init__(self, regions: pl.DataFrame, query_len: int, prot: pl.DataFrame, l_db: int):
        self.prot = prot
        self.query_len = query_len
        row = {t: i for i, t in enumerate(prot["target_name"])}
        length = prot["length"].to_numpy()
        r = regions.filter(pl.col("region_evalue").is_finite())
        t_idx = np.array([row[t] for t in r["target_name"]], dtype=np.int64)
        x = r["region_evalue"].to_numpy() * length[t_idx] / l_db
        # -ln p with p = 1 - exp(-x); written with expm1 so small x keeps its precision.
        w = -np.log(-np.expm1(-x))
        qs = r["region_start"].to_numpy().astype(np.int64)
        qe = r["region_end"].to_numpy().astype(np.int64)
        # W[t, i]: the strongest evidence (-ln p) from any call on protein t covering residue i.
        W = np.zeros((prot.height, query_len), dtype=np.float64)
        for t, a, b, v in zip(t_idx, qs, qe, w):
            seg = W[t, a:b]
            np.maximum(seg, v, out=seg)
        self.W = W
        self.calls = r.with_columns(
            t_idx=pl.Series(t_idx), p_protein=pl.Series(-np.expm1(-x)), neg_ln_p=pl.Series(w)
        )

    def score_rows(self, rows: np.ndarray) -> tuple[float, int]:
        """log10 Fisher p (lowest over residues) and the residue it is at, for a set of rows."""
        s = 2 * self.W[rows].sum(axis=0)
        lp = chi2.logsf(s, 2 * len(rows)) / np.log(10)
        i = int(np.argmin(lp))
        return float(lp[i]), i

    def score_groups(self, group_id: np.ndarray, sizes: dict[int, int]) -> pl.DataFrame:
        """Every group's score, vectorised: sum W rows per group, then Fisher per residue."""
        gids = np.unique(group_id)
        pos = np.searchsorted(gids, group_id)
        S = np.zeros((len(gids), self.query_len))
        np.add.at(S, pos, self.W)
        n = np.array([sizes[g] for g in gids])
        lp = chi2.logsf(2 * S, 2 * n[:, None]) / np.log(10)
        best = lp.argmin(axis=1)
        return pl.DataFrame({
            "group_id": gids,
            "group_size": n,
            "log10_p": lp[np.arange(len(gids)), best],
            "best_residue": best,
        })


def rank_of(values: np.ndarray, v: float) -> tuple[int, int]:
    """Rank 1 = best (lowest). Returns (rank counting only strictly better, n tied with v)."""
    return int((values < v).sum()) + 1, int((values == v).sum())


def ungapped(q: str, t: str, qs: int, ts: int, length: int) -> dict:
    a, b = q[qs:qs + length], t[ts:ts + length]
    match = "".join("|" if x == y else " " for x, y in zip(a, b))
    return {"query_residues": a, "match": match, "target_residues": b,
            "n_identical": match.count("|"), "aligned_length": len(a)}


def in_position_calls(sc: Scorer, members: pl.DataFrame, residue: int, qseq: str,
                      tseqs: dict[str, str]) -> pl.DataFrame:
    """For each member, its lowest-p call covering `residue`, with the real residues."""
    rows = []
    for m in members.iter_rows(named=True):
        c = (sc.calls.filter((pl.col("target_name") == m["target_name"])
                             & (pl.col("region_start") <= residue) & (pl.col("region_end") > residue))
             .sort("neg_ln_p", "alphabet", "ksize_arm", descending=[True, False, False]))
        if c.height == 0:
            rows.append({"gene": m["gene"], "has_call": False})
            continue
        c = c.row(0, named=True)
        qs, qe, ts = int(c["region_start"]), int(c["region_end"]), int(c["target_start"])
        rows.append({"gene": m["gene"], "has_call": True, "alphabet": c["alphabet"],
                     "k": int(c["ksize_arm"]), "query_start0": qs, "query_end0": qe,
                     "target_start0": ts, "target_end0": ts + qe - qs,
                     "evalue_database": c["region_evalue"], "p_protein": c["p_protein"],
                     "neg_ln_p": c["neg_ln_p"],
                     **ungapped(qseq, tseqs[m["target_name"]], qs, ts, qe - qs)})
    return pl.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shuffle-regions", type=Path, default=None,
                    help="folder of notebook 284 shuffle region parquets (regions/*.parquet)")
    args = ap.parse_args()

    prot = pl.read_csv(GROUPS).sort("group_id", "gene")
    l_db = int(prot["length"].sum())
    sizes = dict(zip(*prot.group_by("group_id").agg(n=pl.len()).select("group_id", "n")
                     .to_numpy().T))
    gid = prot["group_id"].to_numpy()
    queries = read_fasta(DATA / "queries.fa")
    tseqs = read_fasta(FASTA)
    rng = np.random.default_rng(SEED)
    lf = pl.scan_parquet(DATA / "regions.parquet")

    summary, tops, members_out, random_out, residue_track = [], [], [], [], []
    for q, partner in CASES.items():
        reg = lf.filter(pl.col("query_name") == q).collect()
        sc = Scorer(reg, len(queries[q]), prot, l_db)
        g = sc.score_groups(gid, sizes).join(
            prot.group_by("group_id").agg(genes=pl.col("gene").sort().str.join(" ")),
            on="group_id")
        single = sc.score_groups(np.arange(prot.height), {i: 1 for i in range(prot.height)})
        single = single.with_columns(gene=prot["gene"], group_id=prot["group_id"])

        case = prot.filter(pl.col("gene") == partner).row(0, named=True)
        cg = g.filter(pl.col("group_id") == case["group_id"]).row(0, named=True)
        g_rank, g_tied = rank_of(g["log10_p"].to_numpy(), cg["log10_p"])
        sp = single.filter(pl.col("gene") == partner).row(0, named=True)
        s_rank, s_tied = rank_of(single["log10_p"].to_numpy(), sp["log10_p"])
        # best protein of the case group alone
        best_member = (single.filter(pl.col("group_id") == case["group_id"])
                       .sort("log10_p", "gene").row(0, named=True))

        rows_case = np.flatnonzero(gid == case["group_id"])
        n = len(rows_case)
        rand = np.array([sc.score_rows(rng.choice(prot.height, n, replace=False))[0]
                         for _ in range(N_RANDOM)])
        frac_rand_better_or_equal = float((rand <= cg["log10_p"]).mean())
        random_out.append(pl.DataFrame({"query": q, "group_size": n, "draw": np.arange(N_RANDOM),
                                        "log10_p": rand}))

        g_sorted = g.sort("log10_p", "group_id").with_row_index("rank", offset=1)
        tops.append(g_sorted.head(25).with_columns(query=pl.lit(q),
                                                   genes=pl.col("genes").str.slice(0, 120)))
        res = int(cg["best_residue"])
        mem = prot.filter(pl.col("group_id") == case["group_id"])
        calls = in_position_calls(sc, mem, res, queries[q], tseqs)
        members_out.append(calls.with_columns(query=pl.lit(q), residue0=pl.lit(res)))
        for r_i in rows_case:
            residue_track.append(pl.DataFrame({
                "query": q, "gene": prot["gene"][int(r_i)], "residue0": np.arange(sc.query_len),
                "neg_ln_p": sc.W[r_i]}))
        # the top group's own in-position calls, to show what the best group looks like
        top = g_sorted.row(0, named=True)
        top_mem = prot.filter(pl.col("group_id") == top["group_id"])
        members_out.append(in_position_calls(sc, top_mem, int(top["best_residue"]), queries[q], tseqs)
                           .filter(pl.col("has_call"))
                           .with_columns(query=pl.lit(q + " top group"),
                                         residue0=pl.lit(int(top["best_residue"]))))

        verdict_rank = g_rank <= TOP_N
        verdict_rand = (1 - frac_rand_better_or_equal) >= NULL_QUANTILE
        summary.append({
            "query": q, "partner": partner, "n_groups": g.height, "case_group_id": case["group_id"],
            "case_group_size": n, "case_group_genes": cg["genes"],
            "case_group_log10_p": cg["log10_p"], "case_group_best_residue0": res,
            "case_group_rank": g_rank, "case_group_n_tied": g_tied,
            "n_groups_with_any_evidence": int((g["log10_p"] < 0).sum()),
            "partner_alone_log10_p": sp["log10_p"], "partner_alone_rank": s_rank,
            "partner_alone_n_tied": s_tied, "n_proteins": prot.height,
            "best_member": best_member["gene"], "best_member_log10_p": best_member["log10_p"],
            "random_sets_frac_better_or_equal": frac_rand_better_or_equal,
            "random_sets_median_log10_p": float(np.median(rand)),
            "top_group_genes": top["genes"][:120], "top_group_log10_p": top["log10_p"],
            "pass_top10": verdict_rank, "pass_random_sets": verdict_rand,
            "shuffle_null": "pending",
        })
        print(json.dumps(summary[-1], indent=1, default=str))

    if args.shuffle_regions is not None:
        files = sorted((args.shuffle_regions / "regions").glob("*.parquet"))
        shuf_rows = []
        for q, partner in CASES.items():
            case_gid = prot.filter(pl.col("gene") == partner)["group_id"][0]
            rows_case = np.flatnonzero(gid == case_gid)
            sl = pl.scan_parquet(files).filter(pl.col("query_name").str.starts_with(q + "_shuf"))
            for name in sl.select("query_name").unique().collect()["query_name"].sort():
                reg = sl.filter(pl.col("query_name") == name).collect()
                sc = Scorer(reg, len(queries[q]), prot, l_db)  # shuffles keep the length
                lp, _ = sc.score_rows(rows_case)
                shuf_rows.append({"query": q, "shuffle": name, "case_group_log10_p": lp})
        shuf = pl.DataFrame(shuf_rows)
        shuf.write_csv(NB / "285_shuffle_null.csv")
        for s in summary:
            v = shuf.filter(pl.col("query") == s["query"])["case_group_log10_p"].to_numpy()
            frac = float((v <= s["case_group_log10_p"]).mean())
            s["shuffle_null"] = f"{len(v)} shuffles, {frac:.2f} as good or better"

    pl.DataFrame(summary).write_csv(NB / "285_family_pooling_summary.csv")
    pl.concat(tops).write_csv(NB / "285_top_groups.csv")
    pl.concat(members_out, how="diagonal_relaxed").write_csv(NB / "285_case_group_calls.csv")
    pl.concat(random_out).write_csv(NB / "285_random_group_null.csv")
    pl.concat(residue_track).write_csv(NB / "285_case_group_residue_evidence.csv")


if __name__ == "__main__":
    main()
