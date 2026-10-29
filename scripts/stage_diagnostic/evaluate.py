"""Every metric of the stage diagnostic, per dataset. Datasets are never pooled.

Methods (each a score per query and reference; a reference a method never emitted scores
minus infinity, so it is never found):

  A      full-residue Smith-Waterman, every reference
  B      reduced-alphabet Smith-Waterman, every reference
  C_B    kmerseek candidates at scaled 1, rescored by B
  C_A    the same candidates rescored by A
  D_B    kmerseek candidates at scaled 10, rescored by B
  E1     kmerseek end to end at scaled 1, ranked as the dataset's pipeline ranks
  E10    the same at scaled 10

Found at budget b: the positive scores strictly above the (b+1)-th best negative of that
query under that method. Ties with the threshold count as not found. Self pairs and
same-origin ice-binding pairs are left out of positives and negatives.

Outputs (tables/350_stage_diagnostic/v1/<dataset>/ and the results root):
  positives.parquet      every query x positive, every method: candidate?, score, rank,
                         found at each budget, the stage it was lost at, A/B coordinates,
                         kmerseek's best region, shared seeds at scaled 1 and 10
  query_summary.tsv      per query: candidates examined per stage, seed features
                         available and retained, recall per method and budget
  recall.tsv             macro recall per split, method, budget, with the group bootstrap
  candidate_recall.tsv   the same for candidate recall (C, D)
  loss_by_stage.tsv      positives by the stage they were lost at
  by_group.tsv           recall per origin / framework / class
  pr.parquet             precision-recall points, scores calibrated per query
  collisions.tsv         candidates whose only shared seed is a reduced one
  selection.tsv          the setting chosen on the dev split
  decisions.tsv          the pre-registered rules on the held-out split
  self_hits.tsv          each query against its own entry
  fixtures.tsv           fixture checks, pass / fail
  adapter_check.tsv      our seed counts against kmerseek's n_intersecting_hashes
  cost.tsv               runtime, memory, candidate volume per stage
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl

import fixtures as fx
import seeds
from labels import pairs
from panels import read_fasta
from scoring import Scorer

METHODS = ["A", "B", "C_B", "C_A", "D_B", "E1", "E10"]
NEG = -math.inf


def setting_label(s):
    return f"{s['alphabet']}.k{s['ksize']}" + (".mask" if s["mask"] else "")


# --------------------------------------------------------------------------- tables


def kmerseek_scores(km: pl.DataFrame, ds: str, lab: str, dataset_kind: str) -> pl.DataFrame:
    """Per query x reference: candidate flags and shared seeds at scaled 1 / 10, and the
    end-to-end score at scaled 1 / 10, ranked the way the dataset's pipeline ranks."""
    k = km.filter((pl.col("dataset") == ds) & (pl.col("setting") == lab))
    out = None
    for sc, tag in ((1, "C"), (10, "D")):
        c = k.filter((pl.col("scaled") == sc) & (pl.col("kind") == "candidates")).select(
            "query", "reference",
            pl.lit(True).alias(f"{tag}_cand"), pl.col("n_shared").alias(f"{tag}_shared"),
            pl.col("km_q_start").alias(f"{tag}_q_start"), pl.col("km_q_end").alias(f"{tag}_q_end"),
            pl.col("km_t_start").alias(f"{tag}_t_start"), pl.col("km_t_end").alias(f"{tag}_t_end"),
        )  # fmt: skip
        out = (
            c if out is None else out.join(c, on=["query", "reference"], how="full", coalesce=True)
        )
    modes = {}
    for sc, tag in ((1, "E1"), (10, "E10")):
        e = k.filter((pl.col("scaled") == sc) & (pl.col("kind") == "e2e"))
        mode = "poisson"
        if dataset_kind == "ice_binding":
            finite = e.filter(pl.col("min_region_evalue").is_finite())
            if finite.height > 0:
                mode = "evalue"
            else:
                alt = k.filter((pl.col("scaled") == sc) & (pl.col("kind") == "e2e_noextend"))
                if alt.height > 0:
                    e, mode = alt, "poisson_no_fit"
        if mode == "evalue":
            score = -pl.col("min_region_evalue").clip(lower_bound=1e-300).log10()
        else:  # ties broken by shared k-mers, as the cys-frameworks reduce step does
            score = pl.col("max_region_poisson") + 1e-6 * pl.col("n_shared")
        modes[tag] = mode
        e = e.select("query", "reference", score.alias(tag), pl.lit(True).alias(f"{tag}_reported"))
        out = out.join(e, on=["query", "reference"], how="full", coalesce=True)
    return out, modes


def build_frame(cfg, data_dir, results, ds, s, km) -> tuple[pl.DataFrame, dict]:
    p = pairs(data_dir / ds, ds)
    al = pl.read_parquet(results / ds / "align.parquet")
    a = al.filter(pl.col("scorer") == "full").select(
        "query", "reference", pl.col("score").alias("A"),
        *[pl.col(c).alias(f"A_{c}") for c in ("q_start", "q_end", "t_start", "t_end")],
    )  # fmt: skip
    b = al.filter(pl.col("scorer") == s["alphabet"]).select(
        "query", "reference", pl.col("score").alias("B"),
        *[pl.col(c).alias(f"B_{c}") for c in ("q_start", "q_end", "t_start", "t_end")],
    )  # fmt: skip
    kmf, modes = kmerseek_scores(km, ds, setting_label(s), ds)
    f = p.join(a, on=["query", "reference"], how="left").join(
        b, on=["query", "reference"], how="left"
    )
    f = f.join(kmf, on=["query", "reference"], how="left")
    f = f.with_columns(
        [pl.col(c).fill_null(False) for c in ("C_cand", "D_cand", "E1_reported", "E10_reported")]
    )
    f = f.with_columns(
        C_B=pl.when(pl.col("C_cand")).then(pl.col("B")).otherwise(NEG),
        C_A=pl.when(pl.col("C_cand")).then(pl.col("A")).otherwise(NEG),
        D_B=pl.when(pl.col("D_cand")).then(pl.col("B")).otherwise(NEG),
        E1=pl.col("E1").fill_nan(None).fill_null(NEG),
        E10=pl.col("E10").fill_nan(None).fill_null(NEG),
    )
    for c in ("C_B", "C_A", "D_B", "E1", "E10"):  # NaN would silently zero a query's recall
        assert f[c].is_nan().sum() == 0, f"NaN scores in {c} for {ds}"
    return f, modes


# --------------------------------------------------------------------------- metrics


def per_query(f: pl.DataFrame, budgets: list[int]) -> pl.DataFrame:
    """Recall per query, method and budget; also each positive's rank and found flags."""
    rows = []
    ev = f.filter(~pl.col("relation").is_in(["self", "same_origin"]))
    for (q,), g in ev.group_by(["query"], maintain_order=True):
        is_pos = (g["relation"] == "positive").to_numpy()
        if is_pos.sum() == 0:
            continue
        for m in METHODS:
            sc = g[m].to_numpy().astype(float)
            neg = np.sort(sc[~is_pos])[::-1]
            pos = sc[is_pos]
            for b in budgets:
                thr = neg[b] if len(neg) > b else NEG
                found = (pos > thr) & np.isfinite(pos)
                rows.append(
                    dict(query=q, method=m, budget=b, n_pos=int(is_pos.sum()),
                         n_found=int(found.sum()), recall=float(found.mean()))
                )  # fmt: skip
    return pl.DataFrame(rows)


def positive_detail(f: pl.DataFrame, budgets: list[int], headline: int) -> pl.DataFrame:
    """One row per query x positive: rank and found-at-budget per method, and the stage the
    positive was lost at on kmerseek's path."""
    ev = f.filter(~pl.col("relation").is_in(["self", "same_origin"]))
    out = []
    for (q,), g in ev.group_by(["query"], maintain_order=True):
        is_pos = (g["relation"] == "positive").to_numpy()
        if not is_pos.any():
            continue
        cols = {}
        for m in METHODS:
            sc = g[m].to_numpy().astype(float)
            neg = np.sort(sc[~is_pos])[::-1]
            order = np.argsort(-sc, kind="stable")
            rank = np.empty(len(sc))
            rank[order] = np.arange(1, len(sc) + 1)
            rank[~np.isfinite(sc)] = np.nan
            cols[f"{m}_rank"] = rank[is_pos]
            for b in budgets:
                thr = neg[b] if len(neg) > b else NEG
                cols[f"{m}_found_b{b}"] = ((sc > thr) & np.isfinite(sc))[is_pos]
        sub = g.filter(pl.Series(is_pos)).with_columns([pl.Series(k, v) for k, v in cols.items()])
        out.append(sub)
    d = pl.concat(out)
    h = headline
    lost10 = (
        pl.when(~pl.col("C_cand")).then(pl.lit("seed"))
        .when(~pl.col("D_cand")).then(pl.lit("sketch"))
        .when(~pl.col("E10_reported")).then(pl.lit("filter"))
        .when(~pl.col(f"E10_found_b{h}")).then(pl.lit("rank"))
        .otherwise(pl.lit("found"))
    )  # fmt: skip
    lost1 = (
        pl.when(~pl.col("C_cand")).then(pl.lit("seed"))
        .when(~pl.col("E1_reported")).then(pl.lit("filter"))
        .when(~pl.col(f"E1_found_b{h}")).then(pl.lit("rank"))
        .otherwise(pl.lit("found"))
    )  # fmt: skip
    return d.with_columns(
        lost_at_scaled10=lost10,
        lost_at_scaled1=lost1,
        reduction_loss=pl.col(f"A_found_b{h}") & ~pl.col(f"B_found_b{h}"),
        reduction_gain=~pl.col(f"A_found_b{h}") & pl.col(f"B_found_b{h}"),
    )


def macro(pq: pl.DataFrame, queries: list[str]) -> float:
    s = pq.filter(pl.col("query").is_in(queries))
    return float(s["recall"].mean()) if s.height else float("nan")


def bootstrap(values_by_query: dict, groups: dict, reps: int, seed: int, fn=np.mean):
    """Resample groups (origin / framework / family) with replacement; the statistic is the
    mean over every query of the drawn groups. Returns (2.5%, 97.5%)."""
    gq: dict[str, list[str]] = {}
    for q, g in groups.items():
        if q in values_by_query:
            gq.setdefault(g, []).append(q)
    keys = sorted(gq)
    if len(keys) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(reps):
        draw = rng.choice(len(keys), len(keys), replace=True)
        vals = [values_by_query[q] for i in draw for q in gq[keys[i]]]
        stats.append(fn(vals))
    return tuple(float(x) for x in np.percentile(stats, [2.5, 97.5]))


def pr_points(f: pl.DataFrame, method: str) -> pl.DataFrame:
    """Scores calibrated per query: a pair's value is the share of that query's negatives
    scoring at or above it (an empirical p-value). Pooled over queries, swept from small to
    large. Recall's denominator is every positive, found or not."""
    ev = f.filter(~pl.col("relation").is_in(["self", "same_origin"]))
    vals, lab = [], []
    n_pos_total = 0
    for (q,), g in ev.group_by(["query"], maintain_order=True):
        sc = g[method].to_numpy().astype(float)
        is_pos = (g["relation"] == "positive").to_numpy()
        n_pos_total += int(is_pos.sum())
        neg = np.sort(sc[~is_pos])
        fin = np.isfinite(sc)
        p = 1 - np.searchsorted(neg, sc[fin], side="left") / max(1, len(neg))
        vals += list(p)
        lab += list(is_pos[fin])
    if n_pos_total == 0 or not vals:
        return pl.DataFrame()
    v, y = np.array(vals), np.array(lab)
    order = np.argsort(v, kind="stable")
    v, y = v[order], y[order]
    # one point per distinct calibrated value, so ties never split
    last = np.r_[np.nonzero(np.diff(v))[0], len(v) - 1]
    tp = np.cumsum(y)[last]
    n = last + 1
    return pl.DataFrame(
        dict(method=method, empirical_p=v[last], precision=tp / n, recall=tp / n_pos_total)
    )


# --------------------------------------------------------------------------- checks


def seed_counts(
    data_dir: Path, ds: str, s: dict, f: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Seed features per query (available, retained at 1 and 10) and our shared-seed counts
    on kmerseek's candidate pairs, to check the adapter against n_intersecting_hashes."""
    qs = read_fasta(data_dir / ds / "queries.fasta")
    rs = read_fasta(data_dir / ds / "references.fasta")
    a, k, mask = s["alphabet"], s["ksize"], s["mask"]
    h1 = {r: seeds.hashes(x, a, k, mask, 1) for r, x in rs.items()}
    h10 = {r: seeds.hashes(x, a, k, mask, 10) for r, x in rs.items()}
    full = {r: seeds.reduced_kmers(x, "protein20", k) for r, x in rs.items()}
    qrows, prs = [], []
    for q, x in qs.items():
        q1 = seeds.hashes(x, a, k, mask, 1)
        q10 = seeds.hashes(x, a, k, mask, 10)
        qfull = seeds.reduced_kmers(x, "protein20", k)
        qrows.append(
            dict(query=q, length=len(x), seed_windows=max(0, len(x) - k + 1),
                 seeds_available=len(seeds.reduced_kmers(x, a, k, mask)),
                 seeds_retained_s1=len(q1), seeds_retained_s10=len(q10))
        )  # fmt: skip
        for r in rs:
            n1 = len(q1 & h1[r])
            if n1 == 0:
                continue
            prs.append(
                dict(query=q, reference=r, our_shared_s1=n1, our_shared_s10=len(q10 & h10[r]),
                     shares_full_residue_kmer=len(qfull & full[r]) > 0)
            )  # fmt: skip
    return pl.DataFrame(qrows), pl.DataFrame(prs)


def check_fixtures(cfg, km: pl.DataFrame) -> pl.DataFrame:
    rows = []
    sc10 = cfg["kmerseek"]["scaled_sketch"]
    settings = {
        setting_label(s): s
        for ds in cfg["kmerseek"]["settings"]
        for s in cfg["kmerseek"]["settings"][ds]
    }
    for lab, s in sorted(settings.items()):
        q, r = fx.build(s["alphabet"], s["ksize"], sc10, s["mask"])
        A, B = Scorer("full"), Scorer(s["alphabet"])
        k = km.filter((pl.col("dataset") == "fixtures") & (pl.col("setting") == lab))
        cand = {
            sc: set(
                zip(
                    *k.filter((pl.col("scaled") == sc) & (pl.col("kind") == "candidates")).select(
                        "query", "reference"
                    )
                )
            )
            for sc in (1, sc10)
        }
        n_shared = {
            (row["query"], row["reference"], row["scaled"]): row["n_shared"]
            for row in k.filter(pl.col("kind") == "candidates").iter_rows(named=True)
        }
        for qn, rn, stage, expect in fx.EXPECTED:
            qq, rr = q[qn], r[rn]
            if stage in ("A", "B"):
                S = A if stage == "A" else B
                score = S.score(S.prep(qq), S.prep(rr))
                dup = S.score(S.prep(qq), S.prep(r["fx_exact_duplicate"]))
                best = max(S.score(S.prep(qq), S.prep(x)) for x in r.values())
                ok = {
                    "top": score == best,
                    "hit": score > 0,
                    "ties_duplicate": score == dup,
                    "below_duplicate": score < dup,
                }[expect]
                obs = f"score {score:.1f}, duplicate {dup:.1f}"
            else:
                scaled = 1 if stage == "C" else sc10
                hit = (qn, rn) in cand[scaled]
                ok = hit if expect == "hit" else not hit
                obs = "candidate" if hit else "not a candidate"
            rows.append(
                dict(
                    setting=lab,
                    query=qn,
                    reference=rn,
                    stage=stage,
                    expected=expect,
                    observed=obs,
                    passed=ok,
                )
            )
        # the adapter's seed count against kmerseek's, on the exact duplicate
        for scaled in (1, sc10):
            ours = len(seeds.hashes(q["fx_base"], s["alphabet"], s["ksize"], s["mask"], scaled))
            theirs = n_shared.get(("fx_base", "fx_exact_duplicate", scaled))
            rows.append(
                dict(setting=lab, query="fx_base", reference="fx_exact_duplicate",
                     stage=f"seed count, scaled {scaled}", expected=str(ours),
                     observed=str(theirs), passed=theirs is not None and int(theirs) == ours)
            )  # fmt: skip
    return pl.DataFrame(rows)


def self_hits(
    f: pl.DataFrame, qsum: pl.DataFrame, data_dir: Path, ds: str, s: dict, modes
) -> pl.DataFrame:
    """Each query against its own entry: is it in the database, the same sequence and
    version, and which stage drops it."""
    refs = pl.read_csv(data_dir / ds / "references.tsv", separator="\t", infer_schema_length=0)
    qs = pl.read_csv(data_dir / ds / "queries.tsv", separator="\t", infer_schema_length=0)
    selfp = f.filter((pl.col("relation") == "self") & (pl.col("query") == pl.col("reference")))
    out = (
        qs.select("id", pl.col("sha1").alias("q_sha1"), *([pl.col("seq_version").alias("q_seq_version")] if "seq_version" in qs.columns else []))
        .rename({"id": "query"})
        .join(refs.select(pl.col("id").alias("query"), pl.col("sha1").alias("db_sha1"),
                          *([pl.col("seq_version").alias("db_seq_version")] if "seq_version" in refs.columns else [])),
              on="query", how="left")
        .join(selfp.select("query", "A", "B", "C_cand", "C_shared", "D_cand", "D_shared", "E1_reported", "E10_reported", "E1", "E10"), on="query", how="left")
        .join(qsum, on="query", how="left")
    )  # fmt: skip
    k = s["ksize"]
    reason = (
        pl.when(pl.col("db_sha1").is_null()).then(pl.lit("not in the database"))
        .when(pl.col("db_sha1") != pl.col("q_sha1")).then(pl.lit("database holds another sequence"))
        .when(pl.col("length") < k).then(pl.lit(f"shorter than k={k}"))
        .when(pl.col("seeds_retained_s1") == 0).then(pl.lit("no seed left after the low-complexity mask"))
        .when(~pl.col("C_cand")).then(pl.lit("kmerseek emitted no self candidate at scaled 1"))
        .when(~pl.col("E1_reported")).then(pl.lit("filtered at scaled 1"))
        .when(pl.col("seeds_retained_s10") == 0).then(pl.lit("empty sketch at scaled 10"))
        .when(~pl.col("D_cand")).then(pl.lit("not a candidate at scaled 10"))
        .when(~pl.col("E10_reported")).then(pl.lit("filtered at scaled 10"))
        .otherwise(pl.lit("found at every stage"))
    )  # fmt: skip
    return out.with_columns(
        dataset=pl.lit(ds), setting=pl.lit(setting_label(s)), mask=pl.lit(s["mask"]),
        e2e_ranking=pl.lit(f"{modes['E1']} / {modes['E10']}"), self_status=reason,
    )  # fmt: skip


def ranking_alternatives(f, km, ds, lab, split, h) -> pl.DataFrame:
    """Exploratory, not a pre-registered rule: kmerseek's own scaled-1 end-to-end output
    ranked three ways (as the pipeline ranks it; by the best region Poisson score; by
    rescoring the same reported pairs with B), on the dataset's positives and, for ice
    binding, on same-origin homologs."""
    e = km.filter(
        (pl.col("dataset") == ds)
        & (pl.col("setting") == lab)
        & (pl.col("scaled") == 1)
        & (pl.col("kind") == "e2e")
    ).select(
        "query",
        "reference",
        (pl.col("max_region_poisson") + 1e-6 * pl.col("n_shared")).alias("E1_poisson"),
    )
    g = f.join(e, on=["query", "reference"], how="left").with_columns(
        E1_poisson=pl.col("E1_poisson").fill_null(NEG),
        E1_B=pl.when(pl.col("E1_reported")).then(pl.col("B")).otherwise(NEG),
    )
    targets = [("positive", g)]
    if ds == "ice_binding":
        swap = g.with_columns(
            relation=pl.when(pl.col("relation") == "same_origin").then(pl.lit("positive"))
            .when(pl.col("relation") == "positive").then(pl.lit("same_origin"))
            .otherwise(pl.col("relation"))
        )  # fmt: skip
        targets.append(("same-origin homolog", swap))
    rows = []
    ho = [q for q in split if split[q] == "heldout"]
    for what, frame in targets:
        for col, name in (
            ("E1", "pipeline ranking"),
            ("E1_poisson", "best region Poisson score"),
            ("E1_B", "reported pairs rescored by B"),
            ("C_B", "all scaled-1 candidates rescored by B"),
        ):
            pq = per_query(frame.with_columns(A=pl.col(col)), [h]).filter(
                (pl.col("method") == "A") & pl.col("query").is_in(ho)
            )
            rows.append(
                dict(
                    dataset=ds,
                    positives=what,
                    ranking=name,
                    heldout_recall=float(pq["recall"].mean()),
                )
            )
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------- driver


def run(cfg, data_dir: Path, results: Path) -> None:
    ev = cfg["evaluation"]
    budgets, h = ev["false_hit_budgets"], ev["headline_budget"]
    km = pl.read_parquet(results / "kmerseek_pairs.parquet")
    kcost = pl.read_csv(results / "kmerseek_cost.tsv", separator="\t", infer_schema_length=0)
    all_sel, all_dec, all_self, all_cost, all_alt = [], [], [], [], []
    for ds in ("ice_binding", "toxin_scaffold", "toxin_target"):
        out = results / ds
        qmeta = pl.read_csv(data_dir / ds / "queries.tsv", separator="\t", infer_schema_length=0)
        split = dict(zip(qmeta["id"], qmeta["split"]))
        group = dict(zip(qmeta["id"], qmeta["group"]))
        per_setting = {}
        for s in cfg["kmerseek"]["settings"][ds]:
            lab = setting_label(s)
            f, modes = build_frame(cfg, data_dir, results, ds, s, km)
            pq = per_query(f, budgets)
            per_setting[lab] = (s, f, pq, modes)
            dev = [q for q in split if split[q] == "dev"]
            ev_f = f.filter(pl.col("relation") == "positive")
            cand = ev_f.group_by("query").agg(pl.col("C_cand").mean().alias("c"))
            c_dev = cand.filter(pl.col("query").is_in(dev))["c"].mean()
            n_cand = f.filter(pl.col("C_cand") & pl.col("query").is_in(dev)).height / max(
                1, len(dev)
            )
            all_sel.append(
                dict(
                    dataset=ds,
                    setting=lab,
                    dev_candidate_recall=c_dev,
                    dev_candidates_per_query=n_cand,
                )
            )
        sel = pl.DataFrame([r for r in all_sel if r["dataset"] == ds]).sort(
            ["dev_candidate_recall", "dev_candidates_per_query"], descending=[True, False]
        )
        chosen = sel["setting"][0]
        for r in all_sel:
            if r["dataset"] == ds:
                r["chosen"] = r["setting"] == chosen
        s, f, pq, modes = per_setting[chosen]
        f.write_parquet(out / "pairs_chosen.parquet")

        detail = positive_detail(f, budgets, h)
        qsum, prs = seed_counts(data_dir, ds, s, f)
        detail = detail.join(prs.select("query", "reference", "our_shared_s1", "our_shared_s10", "shares_full_residue_kmer"),
                             on=["query", "reference"], how="left")  # fmt: skip
        detail.write_parquet(out / "positives.parquet")

        # adapter check: our shared-seed count vs kmerseek's, every candidate pair at scaled 1
        chk = (
            f.filter(pl.col("C_cand"))
            .select("query", "reference", "C_shared")
            .join(
                prs.select("query", "reference", "our_shared_s1"),
                on=["query", "reference"],
                how="left",
            )
        )
        agree = chk.filter(
            pl.col("C_shared").cast(pl.Int64) == pl.col("our_shared_s1").fill_null(0)
        ).height
        all_cost.append(
            dict(
                dataset=ds,
                kind="adapter_check",
                setting=chosen,
                n_candidate_pairs=chk.height,
                n_seed_count_agree=agree,
            )
        )
        missed_by_km = prs.join(
            f.filter(pl.col("C_cand")).select("query", "reference"),
            on=["query", "reference"],
            how="anti",
        ).height
        all_cost[-1]["n_pairs_sharing_a_seed_not_emitted"] = missed_by_km
        # the same at scaled 10. Counts at scaled 1 agree under any one-to-one relabelling
        # of the letters; only a scaled sketch shows whether our hash values are kmerseek's.
        d_ours = prs.filter(pl.col("our_shared_s10") > 0).select(
            "query", "reference", "our_shared_s10"
        )
        d_km = f.filter(pl.col("D_cand")).select("query", "reference", "D_shared")
        both = d_km.join(d_ours, on=["query", "reference"], how="full", coalesce=True)
        all_cost[-1]["n_candidate_pairs_s10"] = both.height
        all_cost[-1]["n_seed_count_agree_s10"] = both.filter(
            pl.col("D_shared").cast(pl.Int64).fill_null(0) == pl.col("our_shared_s10").fill_null(0)
        ).height

        # per-query summary
        n_c = f.filter(pl.col("C_cand")).group_by("query").len().rename({"len": "candidates_C"})
        n_d = f.filter(pl.col("D_cand")).group_by("query").len().rename({"len": "candidates_D"})
        n_e1 = (
            f.filter(pl.col("E1_reported")).group_by("query").len().rename({"len": "reported_E1"})
        )
        n_e10 = (
            f.filter(pl.col("E10_reported")).group_by("query").len().rename({"len": "reported_E10"})
        )
        wide = pq.filter(pl.col("budget") == h).pivot(on="method", index="query", values="recall")
        qs = (
            qsum.join(n_c, on="query", how="left").join(n_d, on="query", how="left")
            .join(n_e1, on="query", how="left").join(n_e10, on="query", how="left")
            .join(wide, on="query", how="left")
            .with_columns(split=pl.col("query").replace_strict(split), group=pl.col("query").replace_strict(group),
                          candidates_A_B=pl.lit(f["reference"].n_unique()))
        )  # fmt: skip
        qs.write_csv(out / "query_summary.tsv", separator="\t")

        # recall with bootstrap, per split
        rec = []
        for sp in ("dev", "heldout", "all"):
            qq = [q for q in split if sp == "all" or split[q] == sp]
            for m in METHODS:
                for b in budgets:
                    sub = pq.filter(
                        (pl.col("method") == m)
                        & (pl.col("budget") == b)
                        & pl.col("query").is_in(qq)
                    )
                    vals = dict(zip(sub["query"], sub["recall"]))
                    lo, hi = bootstrap(vals, group, ev["bootstrap_reps"], cfg["seed"])
                    rec.append(dict(split=sp, method=m, budget=b, recall=float(np.mean(list(vals.values()))) if vals else None,
                                    ci_low=lo, ci_high=hi, n_queries=len(vals), n_groups=len({group[q] for q in vals})))  # fmt: skip
        recall = pl.DataFrame(rec)
        recall.write_csv(out / "recall.tsv", separator="\t")

        crec = []
        for sp in ("dev", "heldout", "all"):
            qq = [q for q in split if sp == "all" or split[q] == sp]
            for col in ("C_cand", "D_cand"):
                sub = (
                    detail.filter(pl.col("query").is_in(qq))
                    .group_by("query")
                    .agg(pl.col(col).mean().alias("r"))
                )
                vals = dict(zip(sub["query"], sub["r"]))
                lo, hi = bootstrap(vals, group, ev["bootstrap_reps"], cfg["seed"])
                # the same share among negatives: a seed that admits everything has both near 1
                neg = (
                    f.filter(
                        pl.col("query").is_in(qq)
                        & pl.col("relation").is_in(
                            ["negative_annotated", "negative_hard", "background"]
                        )
                    )
                    .group_by("query")
                    .agg(pl.col(col).mean().alias("r"))
                )
                crec.append(
                    dict(
                        split=sp,
                        stage=col[0],
                        candidate_recall=float(np.mean(list(vals.values()))) if vals else None,
                        ci_low=lo,
                        ci_high=hi,
                        negatives_kept=float(neg["r"].mean()) if neg.height else None,
                        n_queries=len(vals),
                    )
                )
        pl.DataFrame(crec).write_csv(out / "candidate_recall.tsv", separator="\t")

        loss = (
            detail.group_by("split", "lost_at_scaled10").len().rename({"lost_at_scaled10": "stage"}).with_columns(path=pl.lit("scaled 10"))
            .vstack(detail.group_by("split", "lost_at_scaled1").len().rename({"lost_at_scaled1": "stage"}).with_columns(path=pl.lit("scaled 1")))
            .vstack(detail.group_by("split").agg(pl.col("reduction_loss").sum().cast(pl.UInt32).alias("len")).with_columns(stage=pl.lit("A found, B not"), path=pl.lit("scorers")).select("split", "stage", "len", "path"))
            .vstack(detail.group_by("split").agg(pl.col("reduction_gain").sum().cast(pl.UInt32).alias("len")).with_columns(stage=pl.lit("B found, A not"), path=pl.lit("scorers")).select("split", "stage", "len", "path"))
        )  # fmt: skip
        loss.write_csv(out / "loss_by_stage.tsv", separator="\t")

        bg = pq.filter(pl.col("budget") == h).with_columns(
            group=pl.col("query").replace_strict(group), split=pl.col("query").replace_strict(split)
        )
        bg.group_by("split", "group", "method").agg(
            pl.col("recall").mean(), pl.len().alias("n_queries")
        ).sort("split", "group", "method").write_csv(out / "by_group.tsv", separator="\t")
        if ds == "toxin_target":  # homology confound: same toxin family vs not
            fam = pl.read_csv(
                data_dir / ds / "references.tsv", separator="\t", infer_schema_length=0
            ).select(pl.col("id").alias("reference"), pl.col("group").alias("r_family"))
            dd = detail.join(fam, on="reference", how="left").with_columns(
                same_family=pl.col("r_family") == pl.col("q_group")
            )
            strat = []
            for sf in (True, False):
                x = dd.filter(pl.col("same_family") == sf)
                for m in METHODS:
                    strat.append(
                        dict(
                            same_family=sf,
                            method=m,
                            n_positive_pairs=x.height,
                            share_found=float(x[f"{m}_found_b{h}"].mean()) if x.height else None,
                        )
                    )
            pl.DataFrame(strat).write_csv(out / "same_family_vs_not.tsv", separator="\t")

        if ds == "ice_binding":
            # sanity check: the same methods on same-origin homologs (relabelled positive,
            # other-origin proteins left out). A scorer that misses these is broken.
            so = f.with_columns(
                relation=pl.when(pl.col("relation") == "same_origin").then(pl.lit("positive"))
                .when(pl.col("relation") == "positive").then(pl.lit("same_origin"))
                .otherwise(pl.col("relation"))
            )  # fmt: skip
            spq = per_query(so, budgets).filter(pl.col("budget") == h)
            spq.with_columns(split=pl.col("query").replace_strict(split)).group_by(
                "split", "method"
            ).agg(pl.col("recall").mean(), pl.len().alias("n_queries")).sort(
                "split", "method"
            ).write_csv(
                out / "same_origin_sanity.tsv", separator="\t"
            )

        pr = pl.concat(
            [
                pr_points(
                    f.filter(pl.col("query").is_in([q for q in split if split[q] == "heldout"])), m
                )
                for m in METHODS
            ],
            how="diagonal_relaxed",
        )
        pr.write_parquet(out / "pr_heldout.parquet")

        # collisions: candidates whose shared seeds are reduced only
        cp = (
            f.filter(pl.col("C_cand") & ~pl.col("relation").is_in(["self", "same_origin"]))
            .join(
                prs.select("query", "reference", "shares_full_residue_kmer"),
                on=["query", "reference"],
                how="left",
            )
            .with_columns(pl.col("shares_full_residue_kmer").fill_null(False))
        )
        cp.group_by("relation", "shares_full_residue_kmer").len().sort(
            "relation", "shares_full_residue_kmer"
        ).with_columns(dataset=pl.lit(ds), setting=pl.lit(chosen)).write_csv(
            out / "collisions.tsv", separator="\t"
        )

        all_self.append(self_hits(f, qsum, data_dir, ds, s, modes))
        all_alt.append(ranking_alternatives(f, km, ds, chosen, split, h))

        # decisions on held-out at the headline budget
        ho = [q for q in split if split[q] == "heldout"]
        r = {
            m: recall.filter(
                (pl.col("split") == "heldout") & (pl.col("method") == m) & (pl.col("budget") == h)
            ).row(0, named=True)
            for m in METHODS
        }
        pqh = pq.filter((pl.col("budget") == h) & pl.col("query").is_in(ho))

        def diff_ci(m1, m2):
            a = dict(zip(*pqh.filter(pl.col("method") == m1).select("query", "recall")))
            b = dict(zip(*pqh.filter(pl.col("method") == m2).select("query", "recall")))
            d = {q: a[q] - b[q] for q in a if q in b}
            return float(np.mean(list(d.values()))), *bootstrap(
                d, group, ev["bootstrap_reps"], cfg["seed"]
            )

        dh = detail.filter(pl.col("query").is_in(ho))
        cr = dh.group_by("query").agg(
            pl.col("C_cand").mean().alias("c"), pl.col("D_cand").mean().alias("d")
        )
        cd = {q: c - d for q, c, d in cr.select("query", "c", "d").iter_rows()}
        cd_mean = float(np.mean(list(cd.values()))) if cd else float("nan")
        cd_lo, cd_hi = bootstrap(cd, group, ev["bootstrap_reps"], cfg["seed"])
        b_found = dh.filter(pl.col(f"B_found_b{h}"))
        seed_loss = float((~b_found["C_cand"]).mean()) if b_found.height else float("nan")
        ab = diff_ci("A", "B")
        ce = diff_ci("C_B", "E1")
        cacb = diff_ci("C_A", "C_B")
        grp = (
            bg.filter((pl.col("split") == "heldout") & pl.col("method").is_in(["A", "B"]))
            .group_by("group", "method")
            .agg(pl.col("recall").mean(), pl.len().alias("n"))
        )
        sub_ok = grp.filter((pl.col("n") >= 3) & (pl.col("recall") >= 0.5))
        dec = [
            dict(
                rule="no_signal",
                value=r["A"]["recall"],
                ci=f"{r['A']['ci_low']:.2f}-{r['A']['ci_high']:.2f}",
                fires=r["A"]["recall"] < 0.10,
            ),
            dict(
                rule="richer_representation",
                value=ab[0],
                ci=f"{ab[1]:.2f}-{ab[2]:.2f}",
                fires=ab[0] >= 0.10 and ab[1] > 0,
            ),
            dict(
                rule="candidate_generation",
                value=seed_loss,
                ci=f"n={b_found.height} B-found positives",
                fires=seed_loss >= 0.20 if not math.isnan(seed_loss) else False,
            ),
            dict(
                rule="sketch",
                value=cd_mean,
                ci=f"{cd_lo:.2f}-{cd_hi:.2f}",
                fires=cd_mean >= 0.05 and cd_lo > 0,
            ),
            dict(
                rule="ranking",
                value=max(ce[0], cacb[0]),
                ci=f"C_B-E1 {ce[0]:.2f} ({ce[1]:.2f}-{ce[2]:.2f}); C_A-C_B {cacb[0]:.2f} ({cacb[1]:.2f}-{cacb[2]:.2f})",
                fires=ce[0] >= 0.10 or cacb[0] >= 0.10,
            ),
            dict(
                rule="subgroup",
                value=float(sub_ok.height),
                ci="; ".join(f"{g} {m} {v:.2f} (n={n})" for g, m, v, n in sub_ok.iter_rows()),
                fires=sub_ok.height > 0,
            ),
        ]
        for d in dec:
            d.update(dataset=ds, setting=chosen)
        all_dec += dec
        # cost
        al = pl.read_csv(out / "align_cost.tsv", separator="\t")
        for sc_name, g in al.group_by("scorer"):
            if sc_name[0] not in ("full", s["alphabet"]):
                continue
            all_cost.append(dict(dataset=ds, kind="stage", setting=chosen, stage="A" if sc_name[0] == "full" else "B",
                                 cpu_seconds=float(g["seconds"].sum()), peak_mb=float(g["worker_peak_mb"].max()),
                                 candidates_per_query=float(g["n_references"].mean())))  # fmt: skip
        kc = kcost.filter((pl.col("dataset") == ds) & (pl.col("setting") == chosen))
        for row in kc.iter_rows(named=True):
            all_cost.append(dict(dataset=ds, kind="kmerseek", setting=chosen, stage=f"{row['step']} s{row['scaled']}",
                                 cpu_seconds=float(row["wall_seconds"] or "nan"), peak_mb=float(row["peak_rss_mb"] or "nan"),
                                 exit=row["exit"], ka_fit_refused=row["ka_fit_refused"]))  # fmt: skip
        for stage, col in (
            ("C", "candidates_C"),
            ("D", "candidates_D"),
            ("E1", "reported_E1"),
            ("E10", "reported_E10"),
        ):
            all_cost.append(
                dict(
                    dataset=ds,
                    kind="volume",
                    setting=chosen,
                    stage=stage,
                    candidates_per_query=float(qs[col].fill_null(0).mean()),
                )
            )
    res = results
    pl.DataFrame(all_sel).write_csv(res / "selection.tsv", separator="\t")
    pl.DataFrame(all_dec).select("dataset", "setting", "rule", "value", "ci", "fires").write_csv(
        res / "decisions.tsv", separator="\t"
    )
    pl.concat(all_self, how="diagonal_relaxed").write_csv(res / "self_hits.tsv", separator="\t")
    pl.concat(all_alt).write_csv(res / "exploratory_e2e_ranking.tsv", separator="\t")
    pl.DataFrame(all_cost, infer_schema_length=None).write_csv(res / "cost.tsv", separator="\t")
    check_fixtures(cfg, km).write_csv(res / "fixtures.tsv", separator="\t")
