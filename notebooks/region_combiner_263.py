"""Combiner C, consensus vote: a merged region's vote count is the number of alphabet-ksize
pairs that call it at ``region_evalue < E_max``. Regions are ranked by vote count, ties
broken by notebook 262's corrected E-value (pairs tried x lowest E-value), and a region is
called when it has at least ``v_min`` votes. Built for notebook 263.

Pairs that make near-identical calls vote together, so a second count gives one vote per
cluster of pairs: two pairs are in one cluster when the share of merged regions both call,
out of the regions either calls, is above ``AGREE_MIN`` for every two pairs in the cluster
(complete linkage).

Two inputs use the same code:

* notebook 260's region table (human queries against zebrafish, merged per query and
  target species), scored against Pfam and Swiss-Prot with notebook 262's `Calls`;
* notebook 241's search of Ced9, P66 and BHF against the human proteome, merged per query
  and target protein, for the rank check on the known cases.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

import region_combiner_261 as rc
import region_combiner_262 as rb
import region_table_260 as rt

KEY = rc.KEY
EMAX_GRID = rc.EMAX_GRID
#: Two pairs share a cluster when they agree on more than this share of merged regions.
AGREE_MIN = 0.8
#: Shuffled-query calls at most this share of real calls counts as "near zero".
DECOY_SHARE_MAX = 0.05
#: Without shuffled queries, the vote count is frozen at the first one reaching this
#: tune precision (the same 0.5 notebook 262 froze its threshold at).
TARGET_PRECISION = rb.TARGET_PRECISION


# ---------------------------------------------------------------------------
# Votes.
# ---------------------------------------------------------------------------


def per_arm_best(calls: pl.DataFrame, key: list[str] = KEY) -> pl.DataFrame:
    """One row per (merged region, pair): the pair's lowest region_evalue in the region."""
    return calls.group_by(key + ["arm"]).agg(e=pl.col("region_evalue").min())


def votes(per_arm: pl.DataFrame, emax: float, cluster_of: dict[str, int] | None = None,
          key: list[str] = KEY) -> pl.DataFrame:
    """Per merged region with at least one vote: ``n_votes`` (pairs with E < emax),
    ``n_cluster_votes`` (clusters with at least one such pair; equals n_votes when
    `cluster_of` is None) and the voting pairs, sorted."""
    v = per_arm.filter(pl.col("e") < emax)
    if cluster_of is not None:
        v = v.with_columns(cluster=pl.col("arm").replace_strict(cluster_of, return_dtype=pl.Int64))
    else:
        v = v.with_columns(cluster=pl.col("arm"))
    return v.group_by(key).agg(
        n_votes=pl.len(),
        n_cluster_votes=pl.col("cluster").n_unique(),
        voting_arms=pl.col("arm").sort(),
        e_voting_min=pl.col("e").min(),
    )


def agreement(per_arm: pl.DataFrame, emax: float, arms: list[str],
              key: list[str] = KEY) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise agreement between pairs at `emax`: merged regions both call / merged
    regions either calls (Jaccard). Returns the matrix (arms x arms, 1 on the diagonal for
    a pair with any call, 0 for one with none) and each pair's number of calls."""
    v = per_arm.filter(pl.col("e") < emax).select(key + ["arm"])
    rid = v.select(key).unique().sort(key).with_row_index("rid")
    v = v.join(rid, on=key)
    idx = {a: i for i, a in enumerate(arms)}
    m = np.zeros((len(arms), rid.height), dtype=bool)
    for a, r in v.group_by("arm").agg(pl.col("rid")).iter_rows():
        if a in idx:
            m[idx[a], np.asarray(r)] = True
    mi = m.astype(np.int32)
    both = mi @ mi.T
    n = mi.sum(axis=1)
    either = n[:, None] + n[None, :] - both
    with np.errstate(invalid="ignore", divide="ignore"):
        j = np.where(either > 0, both / np.maximum(either, 1), 0.0)
    return j, n


def clusters(j: np.ndarray, arms: list[str], agree_min: float = AGREE_MIN):
    """Complete-linkage clusters on 1 - agreement, cut so every two pairs in a cluster
    agree on more than `agree_min`. Returns ({arm: cluster id}, linkage matrix or None).
    Cluster ids are numbered by the cluster's first pair in `arms` order."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    if len(arms) == 1:
        return {arms[0]: 0}, None
    d = 1.0 - j
    np.fill_diagonal(d, 0.0)
    z = linkage(squareform(d, checks=False), method="complete")
    # fcluster joins at cophenetic distance <= t; agreement > agree_min is distance < 1 - agree_min.
    lab = fcluster(z, t=np.nextafter(1.0 - agree_min, 0.0), criterion="distance")
    first: dict[int, int] = {}
    for i, lab_i in enumerate(lab):
        first.setdefault(int(lab_i), len(first))
    return {a: first[int(lab_i)] for a, lab_i in zip(arms, lab)}, z


def with_votes(c: rb.Calls, v: pl.DataFrame) -> rb.Calls:
    """`c` with n_votes and n_cluster_votes columns (0 for regions with no vote)."""
    r = (c.regions.drop([x for x in ("n_votes", "n_cluster_votes", "voting_arms", "e_voting_min")
                         if x in c.regions.columns])
         .join(v, on=KEY, how="left")
         .with_columns(pl.col("n_votes", "n_cluster_votes").fill_null(0)))
    return rb.Calls(r, c.landed, c.n_features)


def sweep(c: rb.Calls, per_arm: pl.DataFrame, decoy_per_arm: pl.DataFrame | None,
          cluster_by_emax: dict[float, dict[str, int]], n_arms: int,
          emax_grid=EMAX_GRID) -> pl.DataFrame:
    """Calls, precision and recall (both truth sets) and shuffled-query calls for every
    (E_max, v_min, counting), v_min from 1 to the number of pairs or clusters."""
    rows = []
    for emax in emax_grid:
        cl = cluster_by_emax[emax]
        cv = with_votes(c, votes(per_arm, emax, cl))
        dv = votes(decoy_per_arm, emax, cl) if decoy_per_arm is not None else None
        for counting, col, vmax in (("every pair", "n_votes", n_arms),
                                    ("one per cluster", "n_cluster_votes", len(set(cl.values())))):
            for vmin in range(1, vmax + 1):
                row = rb.at_threshold(cv, col, False, vmin)
                row.update(emax=emax, v_min=vmin, counting=counting, n_clusters=len(set(cl.values())))
                row["n_decoy_called"] = (int(dv.filter(pl.col(col) >= vmin).height)
                                         if dv is not None else None)
                rows.append(row)
    return (pl.DataFrame(rows, infer_schema_length=None)
            .drop("threshold")
            .select("emax", "counting", "n_clusters", "v_min", "n_called", "n_decoy_called",
                    pl.exclude("emax", "counting", "n_clusters", "v_min", "n_called",
                               "n_decoy_called")))


def vote_at_target(sw: pl.DataFrame) -> pl.DataFrame:
    """For each (E_max, counting), the smallest v_min meeting the target, and why.

    With shuffled-query counts: shuffled calls <= DECOY_SHARE_MAX x real calls. Without:
    tune Swiss-Prot precision >= TARGET_PRECISION, or Pfam when no setting reaches it on
    Swiss-Prot (notebook 262's fallback)."""
    has_decoy = sw["n_decoy_called"].null_count() < sw.height
    if has_decoy:
        ok = sw.filter(pl.col("n_decoy_called") <= DECOY_SHARE_MAX * pl.col("n_called"))
        rule = f"shuffled-query calls <= {DECOY_SHARE_MAX:g} x real calls"
    else:
        ts = "swissprot" if (sw["precision_swissprot"] >= TARGET_PRECISION).any() else "pfam"
        ok = sw.filter(pl.col(f"precision_{ts}") >= TARGET_PRECISION)
        rule = f"tune {ts} precision >= {TARGET_PRECISION:g} (no shuffled-query calls)"
    return (ok.sort("v_min").group_by("emax", "counting", maintain_order=True).first()
            .with_columns(rule=pl.lit(rule)).sort("emax", "counting"))


def freeze(at_target: pl.DataFrame) -> dict:
    """Among the (E_max, counting) settings at their target vote count, the one with the
    most real calls; ties to one vote per cluster, then the smaller E_max."""
    r = at_target.with_columns(_c=(pl.col("counting") == "one per cluster").cast(pl.Int8))
    best = r.sort(["n_called", "_c", "emax"], descending=[True, True, False]).row(0, named=True)
    best.pop("_c")
    return best


# ---------------------------------------------------------------------------
# Known cases: notebook 241's search against the human proteome.
# ---------------------------------------------------------------------------

#: Residue classes for every alphabet in notebook 241, from kmerseek's src/rust/alphabets.rs
#: (seanome/kmerseek d79f863); the HP and dayhoff6 lists are hp_conservation_utils'.
EXTRA_CLUSTERS = {
    "funcgroups8": ["GVALI", "ST", "CM", "FY", "WHP", "NQ", "DE", "KR"],
    "mmseqs12": ["AST", "LM", "IV", "KR", "EQ", "ND", "FY", "C", "G", "H", "P", "W"],
    "wass14": ["WM", "DI", "P", "C", "AV", "K", "T", "RE", "G", "L", "Y", "SH", "F", "NQ"],
    "hsdm17": ["A", "D", "KE", "R", "N", "T", "S", "Q", "Y", "F", "LIV", "M", "C", "W", "H",
               "G", "P"],
}


def residue_class(alphabet: str) -> dict[str, int]:
    import hp_conservation_utils as hc

    groups = hc.ALPHABET_CLUSTERS.get(alphabet) or EXTRA_CLUSTERS[alphabet]
    return {aa: i for i, g in enumerate(groups) for aa in g}


def case_calls(regions_path: Path, emax_max: float = rt.EVALUE_MAX) -> pl.DataFrame:
    """Notebook 241's calls with region_evalue < emax_max, as notebook 260's merge reads
    them: ``accession`` = the query, ``species`` = the human target protein's header (so
    calls merge per query and target protein), ``arm`` = <alphabet>_k<k>."""
    return (pl.scan_parquet(regions_path)
            .filter(pl.col("region_evalue") < emax_max)
            .with_columns(
                accession=pl.col("query_name"),
                species=pl.col("target_name"),
                arm=pl.col("alphabet") + "_k" + pl.col("ksize_arm").cast(pl.String),
                **{c: pl.col(c).cast(pl.Int64) for c in
                   ("region_start", "region_end", "target_start", "target_end", "region_length")})
            .collect())


def rank_units(per_arm: pl.DataFrame, n_tried: int, emax: float,
               cluster_of: dict[str, int] | None) -> pl.DataFrame:
    """Every merged region with at least one vote at `emax`, ranked within its query by
    vote count (most first), then corrected E-value = n_tried x lowest E-value of any
    pair (lowest first). ``rank`` is the competition rank (ties share the best rank),
    ``n_tied`` how many share it."""
    v = votes(per_arm, emax, cluster_of)
    best = per_arm.group_by(KEY).agg(evalue_min=pl.col("e").min())
    out = []
    for col in ("n_votes", "n_cluster_votes"):
        r = (v.join(best, on=KEY)
             .with_columns(evalue_best_of_n_corrected=n_tried * pl.col("evalue_min"))
             .sort(["accession", col, "evalue_best_of_n_corrected"],
                   descending=[False, True, False])
             .with_columns(_pos=pl.int_range(1, pl.len() + 1).over("accession"))
             .with_columns(rank=pl.col("_pos").min().over("accession", col, "evalue_best_of_n_corrected"),
                           n_ranked=pl.len().over("accession"))
             .with_columns(n_tied=pl.len().over("accession", "rank"))
             .drop("_pos").with_columns(counting=pl.lit(col), n_votes_counted=pl.col(col)))
        out.append(r)
    return pl.concat(out)


def match_line(q: str, t: str, alphabet: str) -> str:
    """'|' same residue, '+' different residue in the same class of `alphabet`, ' ' other."""
    cls = residue_class(alphabet)
    return "".join("|" if a == b else "+" if cls.get(a, -1) == cls.get(b, -2) else " "
                   for a, b in zip(q, t))


def show_call(query_seq: str, target_seq: str, row: dict, query: str, target: str,
              width: int = 60) -> str:
    """The real residues of one kmerseek call, query over target, 1-based coordinates,
    with a match line. kmerseek coordinates are 0-based end-exclusive and gapless."""
    qs, qe, ts, te = row["region_start"], row["region_end"], row["target_start"], row["target_end"]
    q, t = query_seq[qs:qe], target_seq[ts:te]
    m = match_line(q, t, row["alphabet"])
    n_mis = sum(residue_class(row["alphabet"]).get(a, -1) != residue_class(row["alphabet"]).get(b, -2)
                for a, b in zip(q, t))
    lines = [f"{row['arm']}  E = {row['region_evalue']:.3g}  length {len(q)}  "
             f"identical {m.count('|')}  same class {m.count('+')}  "
             f"class mismatches {n_mis} (kmerseek: {row['region_n_mismatches']})"]
    # Encoded strings: one letter per residue class of the alphabet (a = first class, ...).
    cls = residue_class(row["alphabet"])
    qe_ = "".join(chr(ord("a") + cls[a]) if a in cls else "?" for a in q)
    te_ = "".join(chr(ord("a") + cls[b]) if b in cls else "?" for b in t)
    me_ = "".join("|" if a == b else " " for a, b in zip(qe_, te_))
    lab = max(len(query), len(target)) + len(" encoded")
    for i in range(0, len(q), width):
        qq, tt, mm = q[i:i + width], t[i:i + width], m[i:i + width]
        lines += [f"{query:<{lab}} {qs + 1 + i:>5} {qq} {qs + i + len(qq):<5}",
                  f"{'':<{lab}} {'':>5} {mm}",
                  f"{target:<{lab}} {ts + 1 + i:>5} {tt} {ts + i + len(tt):<5}",
                  f"{query + ' encoded':<{lab}} {'':>5} {qe_[i:i + width]}",
                  f"{'':<{lab}} {'':>5} {me_[i:i + width]}",
                  f"{target + ' encoded':<{lab}} {'':>5} {te_[i:i + width]}"]
    return "\n".join(lines)
