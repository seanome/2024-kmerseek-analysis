"""Rank every human protein across all alphabets at once (notebook 242).

Input: regions.parquet from 241_alphabet_ranking_collect.py, every region of the 152
arms of the three-case sweep (Ced9, P66 and BHF against the 19_732 canonical human
proteins; an arm is one alphabet at one seed length k).

For one query and one metric, each arm ranks the human proteins it hit by their best
region. A protein's normalised rank at an arm is rank / 19_732; a protein the arm did
not hit gets the middle of the tied bottom ranks, (n_hit + 1 + 19_732) / 2 / 19_732.
The ensemble score is the geometric mean of those normalised ranks (a rank product),
lower is better, in two weightings:

  by_arm       every arm counts once;
  by_alphabet  the arms of one alphabet are averaged first, so each of the 19
               alphabets counts once, whatever number of k it was run at.

Top-1% votes count the arms (or alphabets) that put a protein in the top 1% of the
proteome, rank <= 197.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl

DATA = Path("/Users/olga/data/botryllus/alphabet-ranking-three-cases")
HUMAN_FASTA = Path("/Users/olga/data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa")
N = 19_732
TOP1 = math.floor(0.01 * N)  # 197

# label -> (column, lower is better)
METRICS = {
    "E-value": ("region_evalue", True),
    "mean IDF": ("region_mean_idf", False),
    "tf-idf": ("region_tfidf", False),
    "enrichment": ("region_enrichment", False),
    "Poisson p-value": ("region_tail_probability", True),
}
PARTNER = {"Ced9": "BCL2", "P66": "CD47"}
QUERIES = ["Ced9", "P66", "BHF"]


def human_lengths() -> pl.DataFrame:
    """target_name, gene, length (aa) for every canonical human protein."""
    rows, name, n = [], None, 0
    with open(HUMAN_FASTA) as fh:
        for line in fh:
            if line.startswith(">"):
                if name:
                    rows.append((name, n))
                name, n = line[1:].strip(), 0
            else:
                n += len(line.strip())
    rows.append((name, n))
    df = pl.DataFrame(rows, schema=["target_name", "length"], orient="row")
    return df.with_columns(pl.col("target_name").str.split("|").list.get(6).alias("gene"))


def arm_ranks(metric: str, alphabets: list[str] | None = None) -> pl.DataFrame:
    """One row per (query, alphabet, k, target hit): its rank at that arm, average rank
    for ties, plus n_hit, the number of proteins the arm hit for that query. With
    `alphabets`, only those alphabets' arms."""
    col, lower = METRICS[metric]
    lf = (pl.scan_parquet(DATA / "regions.parquet")
          .select("query_name", "alphabet", "ksize_arm", "target_name", "gene", col)
          .filter(pl.col(col).is_not_null() & pl.col(col).is_finite()))
    if alphabets:
        lf = lf.filter(pl.col("alphabet").is_in(alphabets))
    best = (pl.col(col).min() if lower else pl.col(col).max()).alias("v")
    per = (lf.group_by("query_name", "alphabet", "ksize_arm", "target_name", "gene").agg(best)
             .collect())
    arm = ["query_name", "alphabet", "ksize_arm"]
    return per.with_columns(
        pl.col("v").rank("average", descending=not lower).over(arm).alias("rank"),
        pl.len().over(arm).alias("n_hit"),
    )


def ensemble(metric: str, alphabets: list[str] | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Ensemble scores for every (query, human protein), and the per-arm table.

    Returns (scores, arms): scores has query_name, target_name, gene, log_rp_by_arm,
    log_rp_by_alphabet, votes_arm, votes_alphabet, ens_rank_by_arm,
    ens_rank_by_alphabet; arms has one row per (query, alphabet, k) with n_hit."""
    r = arm_ranks(metric, alphabets)
    r = r.with_columns(
        (pl.col("rank") / N).log().alias("log_r"),
        ((pl.col("n_hit") + 1 + N) / 2 / N).log().alias("log_unhit"),
    )
    arms = r.select("query_name", "alphabet", "ksize_arm", "n_hit", "log_unhit").unique()
    n_arms_alpha = arms.group_by("query_name", "alphabet").agg(pl.len().alias("n_k"))
    arms = arms.join(n_arms_alpha, on=["query_name", "alphabet"])
    n_alpha = arms.group_by("query_name").agg(pl.col("alphabet").n_unique().alias("n_alpha"),
                                              pl.len().alias("n_arms"))
    # Baselines: the score of a protein no arm hit.
    base_arm = arms.group_by("query_name").agg(pl.col("log_unhit").mean().alias("base_arm"))
    base_alpha = (arms.group_by("query_name", "alphabet").agg(pl.col("log_unhit").mean().alias("a"))
                  .group_by("query_name").agg(pl.col("a").mean().alias("base_alpha")))
    # Each hit replaces that arm's unhit value.
    r = r.join(arms.select("query_name", "alphabet", "ksize_arm", "n_k"),
               on=["query_name", "alphabet", "ksize_arm"])
    r = r.with_columns((pl.col("log_r") - pl.col("log_unhit")).alias("d"),
                       (pl.col("rank") <= TOP1).alias("top1"))
    by_alpha = (r.group_by("query_name", "target_name", "gene", "alphabet")
                  .agg((pl.col("d").sum() / pl.col("n_k").first()).alias("d_alpha"),
                       pl.col("d").sum().alias("d_arm"),
                       pl.col("top1").sum().alias("votes_arm"),
                       pl.col("top1").any().alias("vote_alpha")))
    t = (by_alpha.group_by("query_name", "target_name", "gene")
           .agg(pl.col("d_arm").sum(), pl.col("d_alpha").sum(),
                pl.col("votes_arm").sum(), pl.col("vote_alpha").sum().alias("votes_alphabet"),
                pl.len().alias("n_alphabets_hit"))
           .join(n_alpha, on="query_name").join(base_arm, on="query_name")
           .join(base_alpha, on="query_name")
           .with_columns((pl.col("base_arm") + pl.col("d_arm") / pl.col("n_arms")).alias("log_rp_by_arm"),
                         (pl.col("base_alpha") + pl.col("d_alpha") / pl.col("n_alpha")).alias("log_rp_by_alphabet")))
    # Proteins no arm hit: all tied at the baseline, below every hit protein.
    lengths = human_lengths()
    full = []
    for q in t["query_name"].unique().to_list():
        tq = t.filter(pl.col("query_name") == q)
        b = tq.select("base_arm", "base_alpha").row(0)
        rest = (lengths.join(tq.select("target_name"), on="target_name", how="anti")
                .with_columns(pl.lit(q).alias("query_name"), pl.lit(b[0]).alias("log_rp_by_arm"),
                              pl.lit(b[1]).alias("log_rp_by_alphabet"), pl.lit(0, dtype=pl.UInt32).alias("votes_arm"),
                              pl.lit(0, dtype=pl.UInt32).alias("votes_alphabet"), pl.lit(0, dtype=pl.UInt32).alias("n_alphabets_hit")))
        keep = ["query_name", "target_name", "gene", "log_rp_by_arm", "log_rp_by_alphabet",
                "votes_arm", "votes_alphabet", "n_alphabets_hit"]
        full.append(pl.concat([tq.join(lengths.select("target_name", "length"), on="target_name")
                                 .select(keep + ["length"]),
                               rest.select(keep + ["length"])], how="vertical_relaxed"))
    scores = pl.concat(full).with_columns(
        pl.col("log_rp_by_arm").rank("min").over("query_name").alias("ens_rank_by_arm"),
        pl.col("log_rp_by_alphabet").rank("min").over("query_name").alias("ens_rank_by_alphabet"),
        pl.col("votes_alphabet").rank("min", descending=True).over("query_name").alias("vote_rank_alphabet"),
    ).with_columns(pl.lit(metric).alias("metric"))
    return scores, arms


def partner_summary(scores: pl.DataFrame, arms: pl.DataFrame, metric: str, rel: float = 0.25,
                    alphabets: list[str] | None = None) -> pl.DataFrame:
    """For each (query, partner) pair and for the partners scored under the other
    queries (a check that the partner is not simply a protein every query ranks high):
    the best single-arm rank, the two ensemble ranks among 19_732, the top-1% votes,
    and the ensemble percentile among human proteins within +/- rel of its length."""
    per_arm = arm_ranks(metric, alphabets)
    rows = []
    for q in QUERIES:
        for partner in PARTNER.values():
            s = scores.filter((pl.col("query_name") == q) & (pl.col("gene") == partner))
            if s.height == 0:
                continue
            s = s.row(0, named=True)
            L = s["length"]
            peers = scores.filter((pl.col("query_name") == q)
                                  & pl.col("length").is_between(L * (1 - rel), L * (1 + rel)))
            pa = per_arm.filter((pl.col("query_name") == q) & (pl.col("gene") == partner))
            best1 = pa.sort("rank").row(0, named=True) if pa.height else None
            rows.append(dict(
                query=q, partner=partner, is_true_pair=PARTNER.get(q) == partner, metric=metric,
                best_single_rank=None if best1 is None else int(best1["rank"]),
                best_single_arm=None if best1 is None else f"{best1['alphabet']} k={best1['ksize_arm']}",
                ens_rank_by_arm=int(s["ens_rank_by_arm"]), ens_rank_by_alphabet=int(s["ens_rank_by_alphabet"]),
                votes_alphabet=int(s["votes_alphabet"]), votes_arm=int(s["votes_arm"]),
                n_alphabets_hit=int(s["n_alphabets_hit"]),
                n_length_peers=peers.height,
                pct_by_alphabet_among_peers=round(100 * (peers["log_rp_by_alphabet"] > s["log_rp_by_alphabet"]).mean(), 1),
            ))
    return pl.DataFrame(rows)


def beat_counts(metric: str, query: str, partner: str,
                alphabets: list[str] | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Is it the same crowd that beats the partner in every alphabet, or a different one?

    For each alphabet where the partner is hit, take the arm where the partner ranks
    best and the set of proteins ranked strictly above it there. Returns
    (per_alphabet, per_protein): per_alphabet has each alphabet's arm, the partner's
    rank and n_beat; per_protein counts, for every protein that beats the partner
    somewhere, the number of alphabets in which it does."""
    r = arm_ranks(metric, alphabets).filter(pl.col("query_name") == query)
    p = (r.filter(pl.col("gene") == partner).sort("rank")
           .group_by("alphabet").first()
           .select("alphabet", "ksize_arm", pl.col("rank").alias("partner_rank"), "n_hit"))
    beats = (r.join(p.select("alphabet", "ksize_arm", "partner_rank"), on=["alphabet", "ksize_arm"])
               .filter((pl.col("rank") < pl.col("partner_rank")) & (pl.col("gene") != partner)))
    per_alpha = p.join(beats.group_by("alphabet").agg(pl.len().alias("n_beat")), on="alphabet", how="left") \
                 .with_columns(pl.col("n_beat").fill_null(0)).sort("partner_rank")
    per_prot = beats.group_by("target_name", "gene").agg(pl.col("alphabet").n_unique().alias("n_alphabets_beating"))
    return per_alpha, per_prot


def independent_expectation(per_alpha: pl.DataFrame) -> np.ndarray:
    """Expected number of proteins beating the partner in exactly j of the alphabets,
    j = 0..A, if each alphabet drew its beating set at random from the 19_731 other
    proteins (Poisson-binomial with p_a = n_beat_a / 19_731)."""
    ps = (per_alpha["n_beat"].to_numpy() / (N - 1)).tolist()
    dist = np.zeros(len(ps) + 1)
    dist[0] = 1.0
    for p in ps:
        dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p
        dist[0] *= (1 - p)
    return dist * (N - 1)


# ---------------------------------------------------------------------------
# The query-side null (242_null_queries.py output)
# ---------------------------------------------------------------------------
NULL = DATA / "null"


HEAVY_BITS = 20.5  # arms at or below this seed information return many regions per query


def chunk_tags(case: str, alphabet: str, k: int, bits: float, n_queries: int) -> list[str]:
    """Output file stems of one arm of the null run. Arms at or below HEAVY_BITS are
    searched 25 queries at a time (cNN); lighter arms all at once (all), because a search
    costs about 4 seconds to start whatever it finds."""
    base = f"{case}.{alphabet}.k{k}"
    if bits <= HEAVY_BITS:
        return [f"{base}.c{i:02d}" for i in range(-(-n_queries // 25))]
    return [f"{base}.all"]


def arm_set(which: str) -> list[dict]:
    """The arms a 242_null_queries.py --arms <which> run searches, from the sweep plan
    (not from the run's own arms.json, which the latest run overwrites)."""
    import json
    plan = pl.DataFrame(json.loads((DATA / "plan.json").read_text()))
    bits = dict(zip(zip(plan["alphabet"], plan["ksize"]), plan["bits"]))
    if which == "all":
        return [{"alphabet": a, "k": k, "bits": b} for (a, k), b in bits.items()]
    chosen = lowest_arms() if which == "lowest" else best_arms()
    return [{"alphabet": a, "k": k, "bits": bits[(a, k)]} for a, k in chosen.items()]


def null_progress(arm_set_name: str = "all") -> tuple[int, int]:
    """(chunks done, chunks expected) for a 242_null_queries.py --arms <name> run."""
    import json
    if not (NULL / "query_sets.json").exists():
        return 0, 0
    arms = arm_set(arm_set_name)
    sets = json.loads((NULL / "query_sets.json").read_text())
    want = {f"{t}.parquet" for a in arms for c, v in sets.items()
            for t in chunk_tags(c, a["alphabet"], a["k"], a["bits"], len(v))}
    have = {p.name for p in (NULL / "ranks").glob("*.parquet")}
    return len(want & have), len(want)


def null_available(arm_set: str = "all") -> bool:
    """True only when every chunk of that null run has been written."""
    done, expected = null_progress(arm_set)
    return expected > 0 and done >= expected


def null_table(case: str, partner: str) -> pl.DataFrame:
    """One row per (query, metric): the partner's best single-arm rank and its ensemble
    (rank product over the 19 arms, one per alphabet), as ranks among 19_732. A query
    that did not hit the partner at an arm gets the middle of the tied bottom ranks."""
    import json
    names = json.loads((NULL / "query_sets.json").read_text())[case]
    arms = arm_set("lowest")
    files = [NULL / "ranks" / f"{t}.parquet" for a in arms
             for t in chunk_tags(case, a["alphabet"], a["k"], a["bits"], len(names))]
    r = pl.read_parquet([f for f in files if f.exists()]).filter(pl.col("gene") == partner)
    grid = (pl.DataFrame({"query_name": names})
              .join(pl.DataFrame({"alphabet": [a["alphabet"] for a in arms], "k": [a["k"] for a in arms]}), how="cross")
              .join(pl.DataFrame({"metric": list(METRICS)}), how="cross"))
    g = grid.join(r.select("query_name", "alphabet", "k", "metric", "rank", "n_hit").with_columns(pl.col("k").cast(pl.Int64)),
                  on=["query_name", "alphabet", "k", "metric"], how="left")
    g = g.with_columns(pl.col("n_hit").fill_null(0),
                       pl.when(pl.col("query_name") == case).then(pl.lit(N)).otherwise(pl.lit(N - 1)).alias("Neff"))
    g = g.with_columns(pl.when(pl.col("rank").is_not_null()).then(pl.col("rank") / pl.col("Neff"))
                         .otherwise((pl.col("n_hit") + 1 + pl.col("Neff")) / 2 / pl.col("Neff")).alias("r"))
    out = (g.group_by("query_name", "metric")
             .agg((pl.col("r").min() * N).alias("best_single_rank"),
                  (pl.col("r").log().mean().exp() * N).alias("ensemble_rank"),
                  pl.col("rank").is_not_null().sum().alias("arms_hit"))
             .with_columns((pl.col("query_name") == case).alias("is_true_query")))
    return out


def null_summary(case: str, partner: str) -> pl.DataFrame:
    """Per metric: the true query's best single-arm rank and ensemble rank, the median
    over the 300 random queries, and the empirical p-value (1 + random queries at least
    as good) / 301."""
    t = null_table(case, partner)
    rows = []
    for m in METRICS:
        s = t.filter(pl.col("metric") == m)
        true = s.filter(pl.col("is_true_query")).row(0, named=True)
        nul = s.filter(~pl.col("is_true_query"))
        row = dict(case=case, partner=partner, metric=m, n_random=nul.height)
        for stat in ("best_single_rank", "ensemble_rank"):
            row[f"{stat}_true"] = round(true[stat], 1)
            row[f"{stat}_random_median"] = round(nul[stat].median(), 1)
            row[f"{stat}_p"] = (1 + (nul[stat] <= true[stat]).sum()) / (1 + nul.height)  # rounded only when shown
        rows.append(row)
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from alphabet_ranking_utils import _figure_with_legend_row, RAMP  # noqa: E402
from hp_conservation_utils import finish_figure  # noqa: E402

FIG = Path(__file__).resolve().parent.parent / "figures"
PURPLE = RAMP(0.85)
GREY = "#9a9a9a"
TOOLS = ("kmerseek 0.4.0 (982a055) search results of notebook 241: Ced9, P66 and BHF against the "
         "19_732 canonical human proteins, 19 alphabets, 152 arms; ensemble computed here, no new search")


def _rank_axis(ax, label):
    ax.set_xscale("log")
    ax.set_xlim(0.8, 3e4)
    ax.set_xticks([1, 10, 100, 1000, 10_000])
    ax.set_xticklabels(["1", "10", "100", "1_000", "10_000"])
    ax.set_xlabel(label, fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def fig_single_vs_ensemble(summary: pl.DataFrame, path: Path, hypothesis: str, conclusion: str):
    """Rows: (query, partner, metric). Open circle: the partner's best rank at any one
    arm. Filled square: its rank when every alphabet's ranks are combined."""
    order = [("Ced9", "BCL2"), ("P66", "CD47"), ("Ced9", "CD47")]
    title = {("Ced9", "BCL2"): "Ced9 → BCL2, known homolog", ("P66", "CD47"): "P66 → CD47, proposed partner",
             ("Ced9", "CD47"): "Ced9 → CD47, no known link (control)"}
    handles = [
        Line2D([], [], marker="o", ls="", ms=8, markerfacecolor="white", markeredgecolor=PURPLE, markeredgewidth=1.6, label="best rank the partner reaches at any single arm (one alphabet, one k)"),
        Line2D([], [], marker="s", ls="", ms=8, color=PURPLE, label="its rank when the ranks of all 19 alphabets are combined (rank product, each alphabet weighted once)"),
        Line2D([], [], marker="s", ls="", ms=8, color=GREY, label="grey: the same two marks for a pair with no known link, the control"),
    ]
    fig, (ax,) = _figure_with_legend_row(1, (11, 7.2), handles)
    y, ticks, labels = 0, [], []
    for pair in order:
        col = GREY if pair == ("Ced9", "CD47") else PURPLE
        s = summary.filter((pl.col("query") == pair[0]) & (pl.col("partner") == pair[1]))
        ax.text(0.85, y - 0.6, title[pair], fontsize=9.5, fontweight="bold", va="bottom")
        for m in METRICS:
            r = s.filter(pl.col("metric") == m).row(0, named=True)
            a, b = r["best_single_rank"], r["ens_rank_by_alphabet"]
            ax.plot([a, b], [y, y], color=col, lw=1, alpha=0.6, zorder=1)
            ax.scatter([a], [y], s=60, facecolors="white", edgecolors=col, linewidths=1.6, zorder=3)
            ax.scatter([b], [y], s=60, marker="s", color=col, zorder=3)
            ticks.append(y); labels.append(m)
            y += 1
        y += 1.4
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_ylim(y - 1, -1.4)
    _rank_axis(ax, "rank among the 19_732 human proteins (1 = best, log scale)")
    finish_figure(fig, path, tools=TOOLS, hypothesis=hypothesis, conclusion=conclusion, header_y=1.01, footer_y=-0.02)


def fig_same_crowd(panels: list[tuple[str, str, str]], path: Path, hypothesis: str, conclusion: str) -> pl.DataFrame:
    """For each (query, partner, metric): how many proteins beat the partner in exactly j
    of the alphabets that see it (bars), against the number expected if each alphabet
    drew the proteins that beat the partner independently of the others (line)."""
    handles = [
        Line2D([], [], marker="s", ls="", ms=9, color=PURPLE, label="observed: proteins that beat the partner in exactly this many alphabets (each alphabet at the k where the partner ranks best)"),
        Line2D([], [], marker="o", ls="-", ms=5, color="black", label="expected if every alphabet picked the proteins that beat the partner independently of the others"),
    ]
    fig, axes = _figure_with_legend_row(len(panels), (5.2 * len(panels), 5.4), handles, sharey=True)
    out = []
    for ax, (q, partner, m) in zip(axes, panels):
        pa, pp = beat_counts(m, q, partner)
        A = pa.height
        obs = np.bincount(pp["n_alphabets_beating"].to_numpy(), minlength=A + 1)[1:]
        exp = independent_expectation(pa)[1:]
        j = np.arange(1, A + 1)
        ax.bar(j, np.maximum(obs, 0.8), color=PURPLE, width=0.7, zorder=2)
        ax.plot(j, np.maximum(exp, 1e-3), "o-", color="black", ms=4, lw=1.2, zorder=3)
        ax.set_yscale("log"); ax.set_ylim(0.8, 3e4)
        ax.set_xticks(j)
        ax.set_xlabel(f"alphabets (of the {A} that see {partner}) in which a protein beats it", fontsize=8.5)
        ax.set_title(f"{q} → {partner}, ranked by {m}", fontsize=10)
        ax.grid(axis="y", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
        half = (A + 1) // 2
        out.append(dict(query=q, partner=partner, metric=m, alphabets=A,
                        beat_in_half_or_more=int(obs[half - 1:].sum()), expected_half=round(float(exp[half - 1:].sum()), 1),
                        beat_in_all=int(obs[-1]), expected_all=round(float(exp[-1]), 2)))
    axes[0].set_ylabel("human proteins (log scale; 0 drawn at the floor)", fontsize=9)
    finish_figure(fig, path, tools=TOOLS, hypothesis=hypothesis, conclusion=conclusion, header_y=1.01, footer_y=-0.02)
    return pl.DataFrame(out)


def fig_length_bias(scores: pl.DataFrame, query: str, path: Path, hypothesis: str, conclusion: str, n_label: int = 8):
    """Every human protein: its length against its ensemble rank for one query."""
    s = scores.filter(pl.col("query_name") == query)
    handles = [
        Line2D([], [], marker="o", ls="", ms=5, color=PURPLE, alpha=0.4, label="one human protein"),
        Line2D([], [], marker="o", ls="", ms=7, markerfacecolor="none", markeredgecolor="black", label=f"the {n_label} best-ranked proteins, numbered; names in the key"),
        Line2D([], [], ls="--", color="#c0392b", label="median length of a human protein"),
    ]
    fig, (ax,) = _figure_with_legend_row(1, (9, 6.2), handles)
    ax.scatter(s["length"], s["ens_rank_by_alphabet"], s=4, color=PURPLE, alpha=0.25, lw=0, rasterized=True)
    top = s.sort("ens_rank_by_alphabet").head(n_label)
    ax.scatter(top["length"], top["ens_rank_by_alphabet"], s=40, facecolors="none", edgecolors="black", lw=1)
    # Numbers beside the circles, names in a key in the empty lower-left corner: the
    # top proteins sit close together and their names would collide.
    key = []
    for i, r in enumerate(top.iter_rows(named=True), start=1):
        ax.annotate(str(i), (r["length"], r["ens_rank_by_alphabet"]), xytext=(6, -3),
                    textcoords="offset points", fontsize=8, fontweight="bold")
        key.append(f"{i}  {r['gene']}, {r['length']:_} aa")
    ax.text(0.02, 0.97, "best-ranked proteins\n" + "\n".join(key), transform=ax.transAxes, va="top",
            fontsize=8, family="monospace", bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#dddddd"))
    med = float(s["length"].median())
    ax.axvline(med, ls="--", color="#c0392b", lw=1.2)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_yaxis()
    ax.set_xlabel("protein length (aa, log scale)", fontsize=9)
    ax.set_ylabel(f"ensemble rank for {query} (1 = best, log scale)", fontsize=9)
    ax.grid(alpha=0.2); ax.spines[["top", "right"]].set_visible(False)
    finish_figure(fig, path, tools=TOOLS, hypothesis=hypothesis, conclusion=conclusion, header_y=1.01, footer_y=-0.02)
    return top.select("gene", "length", "ens_rank_by_alphabet", "votes_alphabet", "n_alphabets_hit")


def fig_null(path: Path, hypothesis: str, conclusion: str) -> pl.DataFrame:
    """Two panels, one per true pair. Rows: metrics, each with two strips: the best
    single-arm rank and the ensemble rank. Grey dots: the 300 random queries of the
    true query's length. Purple diamond: the true query."""
    handles = [
        Line2D([], [], marker="o", ls="", ms=5, color=GREY, alpha=0.6, label="one of 300 random human proteins of the query's length, used as the query"),
        Line2D([], [], marker="D", ls="", ms=8, color=PURPLE, label="the true query (Ced9 or P66); p right of each row = share of random queries at least as good"),
    ]
    pairs = [("Ced9", "BCL2"), ("P66", "CD47")]
    fig, axes = _figure_with_legend_row(2, (14, 6.4), handles, sharey=True)
    fig.subplots_adjust(wspace=0.22)
    rng = np.random.default_rng(0)
    rows = []
    for ax, (case, partner) in zip(axes, pairs):
        t = null_table(case, partner)
        yt, yl, y = [], [], 0
        for m in METRICS:
            for stat, lab in (("best_single_rank", "best single arm"), ("ensemble_rank", "all 19 combined")):
                s = t.filter(pl.col("metric") == m)
                nul = s.filter(~pl.col("is_true_query"))[stat].to_numpy()
                tr = s.filter(pl.col("is_true_query"))[stat][0]
                ax.scatter(nul, y + rng.uniform(-0.25, 0.25, len(nul)), s=6, color=GREY, alpha=0.5, lw=0)
                ax.scatter([tr], [y], marker="D", s=55, color=PURPLE, zorder=4)
                p = (1 + (nul <= tr).sum()) / (1 + len(nul))
                # each panel's own p, just outside its right edge: the panels share row
                # labels, so a p-value in the label would be the last panel's for both
                ax.text(1.01, y, f"p = {p:.3f}", transform=ax.get_yaxis_transform(), va="center",
                        ha="left", fontsize=7.5, clip_on=False)
                yt.append(y); yl.append(f"{m}, {lab}"); y += 1
            y += 0.6
        ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=8); ax.set_ylim(y - 0.4, -0.8)
        _rank_axis(ax, f"rank of {partner} among the 19_732 human proteins (1 = best, log scale)")
        ax.set_title(f"{case} → {partner}", fontsize=10.5)
        rows.append(null_summary(case, partner))
    finish_figure(fig, path, tools=TOOLS.replace("no new search", "plus 300 random queries per pair on 19 arms (242_null_queries.py)"),
                  hypothesis=hypothesis, conclusion=conclusion, header_y=1.01, footer_y=-0.02)
    return pl.concat(rows)


SUBSET3 = ["hp_lehninger2", "polarity4", "funcgroups8"]


def fig_subset(sub: pl.DataFrame, path: Path, subset: list[str], hypothesis: str, conclusion: str):
    """Rows: (query, partner, metric). Open circle: best single arm among the subset's
    alphabets. Filled square: combined over the subset. Filled diamond: combined over
    all 19 alphabets, for comparison. Grey rows: the no-known-link control."""
    order = [("Ced9", "BCL2"), ("P66", "CD47"), ("Ced9", "CD47")]
    title = {("Ced9", "BCL2"): "Ced9 → BCL2, known homolog", ("P66", "CD47"): "P66 → CD47, proposed partner",
             ("Ced9", "CD47"): "Ced9 → CD47, no known link (control)"}
    handles = [
        Line2D([], [], marker="o", ls="", ms=8, markerfacecolor="white", markeredgecolor=PURPLE, markeredgewidth=1.6,
               label=f"best rank at any single arm of {', '.join(subset)}"),
        Line2D([], [], marker="s", ls="", ms=8, color=PURPLE, label=f"combined over those {len(subset)} alphabets"),
        Line2D([], [], marker="D", ls="", ms=7, color=PURPLE, label="combined over all 19 alphabets (section 1), for comparison"),
        Line2D([], [], marker="s", ls="", ms=8, color=GREY, label="grey: the same marks for the pair with no known link, the control"),
    ]
    fig, (ax,) = _figure_with_legend_row(1, (11, 7.6), handles)
    y, ticks, labels = 0, [], []
    for pair in order:
        col = GREY if pair == ("Ced9", "CD47") else PURPLE
        s = sub.filter((pl.col("query") == pair[0]) & (pl.col("partner") == pair[1]))
        ax.text(0.85, y - 0.6, title[pair], fontsize=9.5, fontweight="bold", va="bottom")
        for m in METRICS:
            r = s.filter(pl.col("metric") == m).row(0, named=True)
            a, b, c = r["best_single_rank"], r["ens_rank_by_alphabet"], r["all19"]
            ax.plot([min(a, b, c), max(a, b, c)], [y, y], color=col, lw=1, alpha=0.5, zorder=1)
            ax.scatter([a], [y], s=60, facecolors="white", edgecolors=col, linewidths=1.6, zorder=3)
            ax.scatter([b], [y], s=60, marker="s", color=col, zorder=4)
            ax.scatter([c], [y], s=50, marker="D", color=col, zorder=3)
            ticks.append(y); labels.append(m); y += 1
        y += 1.4
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_ylim(y - 1, -1.4)
    _rank_axis(ax, "rank among the 19_732 human proteins (1 = best, log scale)")
    finish_figure(fig, path, tools=TOOLS, hypothesis=hypothesis, conclusion=conclusion, header_y=1.01, footer_y=-0.02)


# ---------------------------------------------------------------------------
# Every subset of 2 to 4 alphabets
# ---------------------------------------------------------------------------
from itertools import combinations  # noqa: E402

ALPHABETS19 = ["hp_lehninger2", "hp_thomas_dill2", "hp_kyte_doolittle2", "hp_thomas_dill_no_c2",
               "hp_lehninger_c_nonpolar2", "hp_pbotc_1st_ed2", "hp_lehninger_hpc3", "gbmr4", "polarity4",
               "wwmj5", "dayhoff6", "gbmr7", "funcgroups8", "sdm12", "mmseqs12", "wass14", "hsdm17",
               "uniprot18", "protein20"]
# (query, partner): the two proposed pairs, and CD47 for Ced9 as the control with no known link
SUBSET_PAIRS = [("Ced9", "BCL2"), ("P66", "CD47"), ("Ced9", "CD47")]


def lowest_arms() -> dict[str, int]:
    """alphabet -> its lowest-bit k, the arm 242_null_queries.py searches."""
    import json
    plan = pl.DataFrame(json.loads((DATA / "plan.json").read_text()))
    return dict(plan.sort("bits").group_by("alphabet").first().select("alphabet", "ksize").iter_rows())


def best_arms() -> dict[str, int]:
    """alphabet -> the k at which Ced9 ranks BCL2 best under mean IDF; the lowest-bit k
    for an alphabet that never has BCL2 among the hits. Written to
    best_k_by_bcl2_mean_idf.json so 242_null_queries.py --arms best searches the same arms."""
    import json
    r = arm_ranks("mean IDF").filter((pl.col("query_name") == "Ced9") & (pl.col("gene") == "BCL2"))
    b = dict(r.sort("rank", "ksize_arm").group_by("alphabet").first().select("alphabet", "ksize_arm").iter_rows())
    low = lowest_arms()
    best = {a: int(b.get(a, low[a])) for a in ALPHABETS19}
    (DATA / "best_k_by_bcl2_mean_idf.json").write_text(json.dumps(best, indent=1))
    return best


def log_rank_matrix(metric: str, query: str, mode: str = "lowest") -> tuple[np.ndarray, list[str], list[str]]:
    """(M, genes, alphabets): M[t, a] is protein t's log normalised rank in alphabet a for
    this query. mode "lowest": the alphabet's lowest-bit arm only. mode "all": the mean
    over all its arms. A protein an arm did not hit gets the middle of the tied bottom."""
    import json
    plan = pl.DataFrame(json.loads((DATA / "plan.json").read_text()))
    arms = plan.select("alphabet", pl.col("ksize").alias("ksize_arm"))
    if mode in ("lowest", "best"):
        pick = lowest_arms() if mode == "lowest" else best_arms()
        arms = arms.filter(pl.struct("alphabet", "ksize_arm").map_elements(
            lambda s: pick[s["alphabet"]] == s["ksize_arm"], return_dtype=pl.Boolean))
    r = arm_ranks(metric).filter(pl.col("query_name") == query).join(arms, on=["alphabet", "ksize_arm"])
    nhit = arms.join(r.group_by("alphabet", "ksize_arm").agg(pl.len().alias("n_hit")),
                     on=["alphabet", "ksize_arm"], how="left").with_columns(pl.col("n_hit").fill_null(0))
    genes = human_lengths()["target_name"].to_list()
    gi = {g: i for i, g in enumerate(genes)}
    M = np.zeros((len(genes), len(ALPHABETS19)))
    for j, a in enumerate(ALPHABETS19):
        ks = nhit.filter(pl.col("alphabet") == a)
        col = np.zeros(len(genes))
        for k, n in ks.select("ksize_arm", "n_hit").iter_rows():
            v = np.full(len(genes), math.log((n + 1 + N) / 2 / N))
            h = r.filter((pl.col("alphabet") == a) & (pl.col("ksize_arm") == k))
            idx = np.array([gi[t] for t in h["target_name"]], dtype=int)
            if len(idx):
                v[idx] = np.log(h["rank"].to_numpy() / N)
            col += v
        M[:, j] = col / max(ks.height, 1)
    return M, [t.split("|")[6] for t in genes], ALPHABETS19


def subset_scan(metric: str, mode: str = "lowest", sizes=(1, 2, 3, 4)) -> pl.DataFrame:
    """Every subset of the 19 alphabets of the given sizes: the combined rank (rank
    product, each alphabet once) of each pair's partner among the 19_732 proteins.
    Rank = 1 + the number of proteins with a strictly better combined score."""
    mats = {q: log_rank_matrix(metric, q, mode) for q in {q for q, _ in SUBSET_PAIRS}}
    subsets = [c for s in sizes for c in combinations(range(len(ALPHABETS19)), s)]
    out = {"subset": [" + ".join(ALPHABETS19[i] for i in c) for c in subsets],
           "size": [len(c) for c in subsets]}
    for q, p in SUBSET_PAIRS:
        M, genes, _ = mats[q]
        pi = genes.index(p)
        ranks = np.empty(len(subsets), dtype=np.int64)
        for s in sizes:
            idx = [i for i, c in enumerate(subsets) if len(c) == s]
            cols = np.array([subsets[i] for i in idx])          # (n_subsets, s)
            # score of every protein for every subset of this size, in blocks to bound memory
            for b in range(0, len(idx), 256):
                blk = cols[b:b + 256]
                sc = M[:, blk].mean(axis=2)                     # (n_proteins, n_blk)
                # Ties take the middle rank: an arm with no E-value ties every protein,
                # and counting only strictly better proteins would call that rank 1.
                better = (sc < sc[pi]).sum(axis=0)
                tied = (sc == sc[pi]).sum(axis=0) - 1
                ranks[idx[b:b + 256]] = 1 + better + tied // 2
        out[f"{q}->{p}"] = ranks
    return pl.DataFrame(out).with_columns(pl.lit(metric).alias("metric"), pl.lit(mode).alias("arms"))



def null_arm_matrix(case: str, partner: str, metric: str = "mean IDF") -> pl.DataFrame:
    """Every (query, alphabet, k) of the --arms all null run: the partner's log normalised
    rank, with the middle of the tied bottom ranks when the query did not hit it."""
    import json
    names = json.loads((NULL / "query_sets.json").read_text())[case]
    arms = arm_set("all")
    files = [NULL / "ranks" / f"{t}.parquet" for a in arms
             for t in chunk_tags(case, a["alphabet"], a["k"], a["bits"], len(names))]
    r = (pl.read_parquet([f for f in files if f.exists()])
           .filter((pl.col("gene") == partner) & (pl.col("metric") == metric))
           .select("query_name", "alphabet", pl.col("k").cast(pl.Int64), "rank", "n_hit"))
    grid = pl.DataFrame({"query_name": names}).join(
        pl.DataFrame({"alphabet": [a["alphabet"] for a in arms], "k": [a["k"] for a in arms]}), how="cross")
    g = grid.join(r, on=["query_name", "alphabet", "k"], how="left").with_columns(pl.col("n_hit").fill_null(0))
    Neff = pl.when(pl.col("query_name") == case).then(pl.lit(N)).otherwise(pl.lit(N - 1))
    return g.with_columns(pl.when(pl.col("rank").is_not_null()).then((pl.col("rank") / Neff).log())
                            .otherwise(((pl.col("n_hit") + 1 + Neff) / 2 / Neff).log()).alias("log_r"))


def null_subset_test(case: str, partner: str, metric: str = "mean IDF", sizes=(1, 2, 3, 4)) -> tuple[pl.DataFrame, dict]:
    """Does the true query still stand out once every random query gets the same
    advantages? For each query: pick k per alphabet (either the k Ced9 ranks BCL2 best
    at, the same for everyone, or each query's own best k), then the best subset of each
    size (the lowest mean log normalised rank of the partner). The p-value is the share
    of the 300 random queries whose best subset scores at least as well as the true
    query's, (1 + count) / 301. Scores are shown as rank-scale values, exp(score) x 19_732."""
    g = null_arm_matrix(case, partner, metric)
    names = g["query_name"].unique(maintain_order=True).to_list()
    best = best_arms()
    mats = {
        "k picked on BCL2 (same for every query)": g.filter(pl.struct("alphabet", "k").map_elements(
            lambda s: best[s["alphabet"]] == s["k"], return_dtype=pl.Boolean)),
        "each query's own best k": g.group_by("query_name", "alphabet").agg(pl.col("log_r").min()),
    }
    rows, dists = [], {}
    for rule, m in mats.items():
        wide = m.pivot(on="alphabet", index="query_name", values="log_r").select(["query_name"] + ALPHABETS19)
        wide = pl.DataFrame({"query_name": names}).join(wide, on="query_name", how="left")
        X = wide.select(ALPHABETS19).to_numpy()
        ti = names.index(case)
        for sz in sizes:
            combos = np.array(list(combinations(range(len(ALPHABETS19)), sz)))
            sc = X[:, combos].mean(axis=2)          # (queries, subsets)
            bi = sc.argmin(axis=1)
            b = sc[np.arange(len(names)), bi]
            nul = np.delete(b, ti)
            p = (1 + (nul <= b[ti]).sum()) / (1 + len(nul))
            rows.append(dict(case=case, partner=partner, k_rule=rule, size=sz,
                             true_best_subset=" + ".join(ALPHABETS19[i] for i in combos[bi[ti]]),
                             true_score_rank_scale=round(float(np.exp(b[ti]) * N), 1),
                             random_median_rank_scale=round(float(np.exp(np.median(nul)) * N), 1),
                             p=round(float(p), 4)))
            dists[(rule, sz)] = (np.exp(nul) * N, float(np.exp(b[ti]) * N))
    return pl.DataFrame(rows), dists


def fig_subset_scan(scan: pl.DataFrame, path: Path, hypothesis: str, conclusion: str) -> pl.DataFrame:
    """Three panels (BCL2 for Ced9, CD47 for P66, CD47 for Ced9 the control); x: number of
    alphabets combined; y: the partner's combined rank. Light dots: every subset; dark
    diamond: the best subset for BCL2 at that size, so a panel shows whether the subsets
    picked for BCL2 also help the other pairs."""
    cols = [("Ced9->BCL2", "Ced9 → BCL2, known homolog"), ("P66->CD47", "P66 → CD47, proposed partner"),
            ("Ced9->CD47", "Ced9 → CD47, no known link (control)")]
    handles = [
        Line2D([], [], marker="o", ls="", ms=5, color=RAMP(0.25), label="one subset of the 19 alphabets (each alphabet at the k where Ced9 ranks BCL2 best; mean IDF)"),
        Line2D([], [], marker="D", ls="", ms=8, color=PURPLE, label="the subset that ranks BCL2 best at that size, followed into every panel"),
    ]
    fig, axes = _figure_with_legend_row(3, (15, 6.2), handles, sharey=True)
    rng = np.random.default_rng(0)
    best = scan.sort("Ced9->BCL2").group_by("size").first().sort("size")
    for ax, (c, t) in zip(axes, cols):
        for sz in sorted(scan["size"].unique()):
            v = scan.filter(pl.col("size") == sz)[c].to_numpy()
            ax.scatter(sz + rng.uniform(-0.28, 0.28, len(v)), v, s=3, color=RAMP(0.25), alpha=0.5, lw=0, rasterized=True)
        ax.plot(best["size"], best[c], "D-", color=PURPLE, ms=7, lw=1.2, zorder=4)
        for sz, v in zip(best["size"], best[c]):
            ax.annotate(f"{v:_}", (sz, v), xytext=(8, -3), textcoords="offset points", fontsize=8, color=PURPLE)
        ax.set_yscale("log"); ax.set_ylim(2.5e4, 8)  # rank 1 at the top
        ax.set_xticks(sorted(scan["size"].unique())); ax.set_xlabel("alphabets combined", fontsize=9)
        ax.set_title(t, fontsize=10); ax.grid(alpha=0.2); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("rank among the 19_732 human proteins (1 = best, log scale)", fontsize=9)
    finish_figure(fig, path, tools=TOOLS, hypothesis=hypothesis, conclusion=conclusion, header_y=1.01, footer_y=-0.02)
    return best.select("size", "subset", "Ced9->BCL2", "P66->CD47", "Ced9->CD47")


def fig_subset_null(dists: dict, case: str, partner: str, path: Path, hypothesis: str, conclusion: str):
    """Rows: k rule x subset size. Grey dots: the 300 random queries' best-subset score;
    purple diamond: the true query's."""
    handles = [
        Line2D([], [], marker="o", ls="", ms=5, color=GREY, alpha=0.6, label="one random human protein of the query's length: its best subset of that size"),
        Line2D([], [], marker="D", ls="", ms=8, color=PURPLE, label=f"{case}: its best subset of that size; p in each row label = share of random queries at least as good"),
    ]
    fig, (ax,) = _figure_with_legend_row(1, (10.5, 5.6), handles)
    rng = np.random.default_rng(0)
    yt, yl, y = [], [], 0
    for rule in ["k picked on BCL2 (same for every query)", "each query's own best k"]:
        for sz in (1, 2, 3, 4):
            nul, tr = dists[(rule, sz)]
            ax.scatter(nul, y + rng.uniform(-0.25, 0.25, len(nul)), s=6, color=GREY, alpha=0.5, lw=0)
            ax.scatter([tr], [y], marker="D", s=55, color=PURPLE, zorder=4)
            p = (1 + (nul <= tr).sum()) / (1 + len(nul))
            yt.append(y); yl.append(f"{rule}, {sz} alphabet{'s' if sz > 1 else ''} (p = {p:.3f})"); y += 1
        y += 0.7
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=8); ax.set_ylim(y - 0.5, -0.8)
    _rank_axis(ax, f"{partner}: geometric mean of its normalised ranks over the subset, x 19_732 (lower is better)")
    finish_figure(fig, path, tools=TOOLS.replace("no new search", "plus 300 random queries on all 152 arms (242_null_queries.py --arms all)"),
                  hypothesis=hypothesis, conclusion=conclusion, header_y=1.01, footer_y=-0.02)
