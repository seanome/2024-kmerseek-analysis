"""Tables and figures for notebook 256: BHF against 300 dipeptide-shuffled copies of itself.

Input: the top/*.parquet files of 256_bhf_dipeptide_shuffle_null.py, one row per (query,
alphabet-ksize pair, ranking metric), where an alphabet-ksize pair is one alphabet at one seed length k of the notebook
241 sweep. A query that hit no human protein on an alphabet-ksize pair has no row there.

Every comparison is made on the same set of alphabet-ksize pairs for BHF and for the copies: the alphabet-ksize pairs
where BHF itself has a value under that metric (the settings of notebook 241's BHF
figures). On those alphabet-ksize pairs a copy with no row counts as finding nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

DATA = Path("/Users/olga/data/botryllus/alphabet-ranking-three-cases")
RUN = DATA / "null_bhf_dipeptide"
N_HUMAN = 19_732
TOP = 10

# label -> lower is better; order is the order of the figure rows
METRICS = {
    "E-value": True,
    "bit score": False,
    "mean IDF": False,
    "tf-idf": False,
    "enrichment": False,
    "Poisson score": False,
    "Poisson p-value": True,
    "shared k-mers": False,
    "containment": False,
    "protein enrichment": False,
    "protein Poisson p-value": True,
}


def expected_chunks() -> int:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from alphabet_ensemble_utils import chunk_tags
    arms = json.loads((RUN / "arms.json").read_text())
    n = json.loads((RUN / "run.json").read_text())["n"] + 1
    return sum(len(chunk_tags("BHF", a["alphabet"], a["k"], a["bits"], n)) for a in arms)


def progress() -> tuple[int, int]:
    return len(list((RUN / "top").glob("*.parquet"))), expected_chunks()


def load() -> pl.DataFrame:
    df = pl.read_parquet(sorted((RUN / "top").glob("*.parquet")))
    return df.with_columns(
        (pl.col("query_name") == "BHF").alias("is_bhf"),
        pl.col("talk_best_rank").cast(pl.Float64),
    )


def queries() -> list[str]:
    return [l[1:].strip() for l in open(RUN / "queries.fa") if l.startswith(">")]


def bhf_arms(df: pl.DataFrame) -> pl.DataFrame:
    """(metric, alphabet, k) where BHF has a value."""
    return df.filter(pl.col("is_bhf")).select("metric", "alphabet", "k").unique()


def full_grid(df: pl.DataFrame) -> pl.DataFrame:
    """Every query on every BHF alphabet-ksize pair, with its row where it has one (nulls where the query
    found nothing on that alphabet-ksize pair)."""
    q = pl.DataFrame({"query_name": queries()})
    grid = bhf_arms(df).join(q, how="cross")
    return grid.join(df, on=["metric", "alphabet", "k", "query_name"], how="left").with_columns(
        (pl.col("query_name") == "BHF").alias("is_bhf"))


def talk_in_top(df: pl.DataFrame) -> pl.DataFrame:
    """Per query and metric: the share of BHF's alphabet-ksize pairs where one of the 2024 talk's human
    proteins ranks in the top 10. Ties take their best rank (rank 'min'), for BHF and
    copies alike."""
    g = full_grid(df).with_columns((pl.col("talk_best_rank") <= TOP).fill_null(False).alias("hit"))
    return (g.group_by("metric", "query_name", "is_bhf")
             .agg(pl.len().alias("n_alphabet_ksize_pairs"), pl.col("hit").sum().alias("n_top"))
             .with_columns((100 * pl.col("n_top") / pl.col("n_alphabet_ksize_pairs")).alias("pct")))


def talk_summary(t: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for m in METRICS:
        s = t.filter(pl.col("metric") == m)
        if s.height == 0 or s.filter(pl.col("is_bhf")).height == 0:
            continue
        b = s.filter(pl.col("is_bhf"))
        nul = s.filter(~pl.col("is_bhf"))["pct"].to_numpy()
        bp = b["pct"][0]
        rows.append(dict(metric=m, n_alphabet_ksize_pairs=b["n_alphabet_ksize_pairs"][0], bhf_n_top=b["n_top"][0], bhf_pct=round(bp, 1),
                         copies_median_pct=round(float(np.median(nul)), 1),
                         copies_p2_5=round(float(np.percentile(nul, 2.5)), 1),
                         copies_p97_5=round(float(np.percentile(nul, 97.5)), 1),
                         n_copies=len(nul),
                         p=round((1 + (nul >= bp).sum()) / (1 + len(nul)), 3)))
    return pl.DataFrame(rows)


def best_score_p(df: pl.DataFrame) -> pl.DataFrame:
    """Per metric and alphabet-ksize pair: the share of copies whose best score is at least as good as
    BHF's, counting (1 + copies at least as good) / (1 + copies). A copy that found
    nothing on the alphabet-ksize pair is worse than BHF."""
    g = full_grid(df)
    rows = []
    for m, lower in METRICS.items():
        s = g.filter(pl.col("metric") == m)
        b = s.filter(pl.col("is_bhf")).select("alphabet", "k", pl.col("top_value").alias("bhf"),
                                               pl.col("top_gene").alias("bhf_top_gene"),
                                               pl.col("n_tied_top").alias("bhf_n_tied"))
        c = s.filter(~pl.col("is_bhf")).join(b, on=["alphabet", "k"])
        better = (pl.col("top_value") <= pl.col("bhf")) if lower else (pl.col("top_value") >= pl.col("bhf"))
        a = (c.group_by("alphabet", "k", "bhf", "bhf_top_gene", "bhf_n_tied")
              .agg(better.fill_null(False).sum().alias("n_as_good"), pl.len().alias("n_copies"),
                   pl.col("top_value").is_null().sum().alias("n_copies_no_hit"),
                   pl.col("top_value").median().alias("copies_median")))
        rows.append(a.with_columns(pl.lit(m).alias("metric")))
    out = pl.concat(rows, how="diagonal")
    return out.with_columns(((1 + pl.col("n_as_good")) / (1 + pl.col("n_copies"))).alias("p")).sort("metric", "alphabet", "k")


def best_score_summary(p: pl.DataFrame) -> pl.DataFrame:
    return (p.group_by("metric")
             .agg(pl.len().alias("n_alphabet_ksize_pairs"),
                  (pl.col("p") <= 0.05).sum().alias("n_pairs_p_le_0_05"),
                  (0.05 * pl.len()).round(1).alias("expected_by_chance"),
                  pl.col("p").min().alias("smallest_p"),
                  pl.col("p").median().alias("median_p"))
             .with_columns(pl.col("metric").replace_strict({m: i for i, m in enumerate(METRICS)}).alias("_o"))
             .sort("_o").drop("_o"))


def sticky_top_hits(df: pl.DataFrame) -> pl.DataFrame:
    """For BHF's #1 protein at each alphabet-ksize pair: the share of copies that have the same protein in
    their own top 10 on that alphabet-ksize pair. A high share means the protein comes to the top for any
    sequence with BHF's composition."""
    g = full_grid(df)
    b = g.filter(pl.col("is_bhf")).select("metric", "alphabet", "k", pl.col("top_gene").alias("bhf_top_gene"))
    c = (g.filter(~pl.col("is_bhf")).join(b, on=["metric", "alphabet", "k"])
          .with_columns(pl.col("top10").list.contains(pl.col("bhf_top_gene")).fill_null(False).alias("has")))
    per_arm = c.group_by("metric", "alphabet", "k", "bhf_top_gene").agg((100 * pl.col("has").mean()).alias("pct_copies_top10"))
    return per_arm


def sticky_summary(per_arm: pl.DataFrame) -> pl.DataFrame:
    return (per_arm.group_by("metric")
                   .agg(pl.len().alias("n_alphabet_ksize_pairs"),
                        pl.col("pct_copies_top10").median().round(1).alias("median_pct_copies"),
                        (pl.col("pct_copies_top10") >= 10).sum().alias("n_pairs_ge_10pct"))
                   .with_columns(pl.col("metric").replace_strict({m: i for i, m in enumerate(METRICS)}).alias("_o"))
                   .sort("_o").drop("_o"))


def top_gene_recurrence(df: pl.DataFrame, n: int = 8) -> pl.DataFrame:
    """The proteins most often #1 across (copy, alphabet-ksize pair) cells for one metric, next to
    how often each is #1 for BHF."""
    g = full_grid(df).filter(pl.col("top_gene").is_not_null())
    cop = (g.filter(~pl.col("is_bhf")).group_by("metric", "top_gene").len("n_copy_arm")
             .with_columns((pl.col("n_copy_arm") / pl.col("n_copy_arm").sum().over("metric") * 100).round(1).alias("pct_of_copy_arms")))
    bh = g.filter(pl.col("is_bhf")).group_by("metric", "top_gene").len("n_bhf_arms")
    j = cop.join(bh, on=["metric", "top_gene"], how="full", coalesce=True).fill_null(0)
    return j.sort("metric", "n_copy_arm", descending=[False, True]).group_by("metric", maintain_order=True).head(n)


# ---------------------------------------------------------------- figures
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from alphabet_ranking_utils import _figure_with_legend_row, RAMP  # noqa: E402
from hp_conservation_utils import finish_figure  # noqa: E402

FIG = Path(__file__).resolve().parent.parent / "figures"
PURPLE = RAMP(0.85)
GREY = "#9a9a9a"
TOOLS = ("kmerseek 0.4.0 (982a055) search, 256_bhf_dipeptide_shuffle_null.py: BHF and 300 dipeptide-shuffled "
         "copies against the 19_732 GENCODE v49 canonical human proteins, 152 alphabet-ksize pairs of notebook 241")


def fig_talk_top10(t: pl.DataFrame, summ: pl.DataFrame, path: Path, hypothesis: str, conclusion: str):
    """One row per metric: BHF's share of alphabet-ksize pairs with a 2024-talk protein in the top 10
    (purple diamond) over the 300 copies' shares (grey dots, with their middle 95% as a
    pale band drawn first)."""
    handles = [
        Patch(color="#e4e4e4", label="middle 95% of the 300 shuffled copies"),
        Line2D([], [], marker="o", ls="", ms=4, color=GREY, alpha=0.6, label="one shuffled copy of BHF (same length, residue counts and adjacent-pair counts)"),
        Line2D([], [], marker="D", ls="", ms=8, color=PURPLE, label="BHF; p on the right = share of copies at least as high"),
    ]
    ms = [m for m in METRICS if m in summ["metric"].to_list()]
    fig, (ax,) = _figure_with_legend_row(1, (10, 0.5 * len(ms) + 2.2), handles)
    rng = np.random.default_rng(0)
    for y, m in enumerate(ms):
        s = t.filter(pl.col("metric") == m)
        r = summ.filter(pl.col("metric") == m).row(0, named=True)
        ax.barh(y, r["copies_p97_5"] - r["copies_p2_5"], left=r["copies_p2_5"], height=0.7, color="#e4e4e4", zorder=1)
        nul = s.filter(~pl.col("is_bhf"))["pct"].to_numpy()
        ax.scatter(nul, y + rng.uniform(-0.25, 0.25, len(nul)), s=6, color=GREY, alpha=0.5, lw=0, zorder=2)
        ax.scatter([r["bhf_pct"]], [y], marker="D", s=55, color=PURPLE, zorder=4)
        ax.text(1.01, y, f"p = {r['p']:.3f}", transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=8, clip_on=False)
    ax.set_yticks(range(len(ms)))
    ax.set_yticklabels([f"{m} ({summ.filter(pl.col('metric') == m)['n_alphabet_ksize_pairs'][0]} alphabet-ksize pairs)" for m in ms], fontsize=9)
    ax.set_ylim(len(ms) - 0.5, -0.5)
    ax.set_xlim(left=-1.5)  # a diamond at 0% would otherwise sit half outside the axes
    ax.set_xlabel("% of alphabet-ksize pairs where ZNF292, RSF1, TSHZ1-3, RNMT, SFI1, CETN2, TRAPPC10 or NDNF\nranks in the query's top 10 human proteins")
    ax.grid(axis="x", color="#eeeeee", zorder=0)
    finish_figure(fig, path, tools=TOOLS, hypothesis=hypothesis, conclusion=conclusion, header_y=1.01, footer_y=-0.02, tight=False)


def fig_best_score_p(p: pl.DataFrame, path: Path, hypothesis: str, conclusion: str):
    """One row per metric, one dot per alphabet-ksize pair: the share of copies whose best score is at
    least as good as BHF's. Under chance the dots spread evenly from 0 to 1."""
    handles = [
        Line2D([], [], marker="o", ls="", ms=5, color=PURPLE, alpha=0.7, label="one alphabet-ksize pair: share of the 300 copies whose best human hit scores at least as well as BHF's"),
        Line2D([], [], ls="--", color="#555555", label="0.05: BHF better than 95% of its copies"),
    ]
    ms = [m for m in METRICS if m in p["metric"].unique().to_list()]
    fig, (ax,) = _figure_with_legend_row(1, (10, 0.5 * len(ms) + 2.2), handles)
    rng = np.random.default_rng(1)
    for y, m in enumerate(ms):
        v = p.filter(pl.col("metric") == m)["p"].to_numpy()
        ax.scatter(v, y + rng.uniform(-0.25, 0.25, len(v)), s=12, color=PURPLE, alpha=0.6, lw=0, zorder=3)
        k = (v <= 0.05).sum()
        ax.text(1.01, y, f"{k} of {len(v)} alphabet-ksize pairs ≤ 0.05", transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=8, clip_on=False)
    ax.axvline(0.05, ls="--", color="#555555", lw=1, zorder=2)
    ax.set_xscale("log")
    ax.set_xlim(1 / 400, 1.05)
    ax.set_yticks(range(len(ms)))
    ax.set_yticklabels(ms, fontsize=9)
    ax.set_ylim(len(ms) - 0.5, -0.5)
    ax.set_xlabel("share of copies at least as good as BHF (log scale; smallest possible 1/301)")
    ax.grid(axis="x", color="#eeeeee", zorder=0)
    finish_figure(fig, path, tools=TOOLS, hypothesis=hypothesis, conclusion=conclusion, header_y=1.01, footer_y=-0.02, tight=False)


# ---------------------------------------------------------------- where on BHF
REGION_COL = {
    "E-value": "region_evalue", "bit score": "region_ka_bits", "mean IDF": "region_mean_idf",
    "tf-idf": "region_tfidf", "enrichment": "region_enrichment", "Poisson score": "region_poisson_score",
    "Poisson p-value": "region_tail_probability", "shared k-mers": "region_n_shared_kmers",
    "containment": "containment", "protein enrichment": "query_enrichment",
    "protein Poisson p-value": "query_poisson_pvalue",
}


def bhf_sequence() -> str:
    return "".join(open(RUN / "queries.fa").read().split(">")[1].splitlines()[1:])


def _entropy(s: str) -> float:
    from collections import Counter
    n = len(s)
    return -sum(v / n * np.log2(v / n) for v in Counter(s).values())


def bhf_best_regions(p: pl.DataFrame) -> pl.DataFrame:
    """For every (metric, alphabet-ksize pair) cell of best_score_p: the BHF region that gave BHF's best
    value, from notebook 241's regions.parquet (kmerseek coordinates, 0-based, end
    exclusive). Adds the region's residues, its share of charged residues (D, E, K, R)
    and its Shannon entropy in bits per residue."""
    bhf = bhf_sequence()
    r = (pl.scan_parquet(DATA / "regions.parquet")
           .filter(pl.col("query_name").str.strip_chars() == "BHF")
           .select("alphabet", pl.col("ksize_arm").alias("k"), "gene", "region_start", "region_end",
                   *REGION_COL.values())
           .collect())
    rows = []
    for x in p.iter_rows(named=True):
        col, lower = REGION_COL[x["metric"]], METRICS[x["metric"]]
        s = r.filter((pl.col("alphabet") == x["alphabet"]) & (pl.col("k") == x["k"])
                     & (pl.col("gene") == x["bhf_top_gene"]) & pl.col(col).is_finite())
        if s.height == 0:
            continue
        s = s.sort(col, "region_start", descending=[not lower, False]).row(0, named=True)
        a, b = int(s["region_start"]), int(s["region_end"])
        q = bhf[a:b]
        rows.append(dict(metric=x["metric"], alphabet=x["alphabet"], k=x["k"], gene=x["bhf_top_gene"],
                         p=x["p"], start=a, end=b, length=b - a, seq=q,
                         charged=round(sum(c in "DEKR" for c in q) / len(q), 2),
                         entropy=round(_entropy(q), 2)))
    return pl.DataFrame(rows).with_columns((pl.col("p") <= 0.05).alias("beats_95pct"))


def fig_region_coverage(reg: pl.DataFrame, path: Path, hypothesis: str, conclusion: str):
    """BHF drawn as a line (residues 1-252). Below it, for every residue, the number of
    (metric, alphabet-ksize pair) cells whose best BHF region covers it: all cells as a pale band drawn
    first, the cells where BHF beats 95% of its copies as a purple line on top. A strip
    under the axis marks the lysine and arginine residues."""
    bhf = bhf_sequence()
    L = len(bhf)
    x = np.arange(1, L + 1)
    def cover(d):
        c = np.zeros(L)
        for a, b in zip(d["start"], d["end"]):
            c[a:b] += 1
        return c
    call, cwin = cover(reg), cover(reg.filter(pl.col("beats_95pct")))
    nall, nwin = reg.height, reg.filter(pl.col("beats_95pct")).height
    handles = [
        Patch(color="#e4e4e4", label=f"all {nall} (metric, alphabet-ksize pair) cells: BHF's best region covers this residue"),
        Line2D([], [], color=PURPLE, lw=2, label=f"the {nwin} cells where BHF beats at least 95% of its 300 shuffled copies"),
        Line2D([], [], marker="|", ls="", ms=9, color="#b35900", label="K or R (lysine, arginine) in BHF"),
    ]
    fig, (ax,) = _figure_with_legend_row(1, (12, 4.6), handles, sharey=False)
    ax.fill_between(x, call, step="mid", color="#e4e4e4", lw=0, zorder=1)
    ax.step(x, cwin, where="mid", color=PURPLE, lw=2, zorder=3)
    top = max(call.max(), 1)
    ax.plot([1, L], [-0.06 * top] * 2, color="#333333", lw=2, solid_capstyle="butt", clip_on=False)
    kr = [i + 1 for i, c in enumerate(bhf) if c in "KR"]
    ax.scatter(kr, [-0.12 * top] * len(kr), marker="|", s=60, color="#b35900", clip_on=False)
    ax.set_ylim(-0.16 * top, top * 1.05)
    ax.set_xlim(0, L + 1)
    ax.set_xlabel("BHF residue (1-252); black line = BHF")
    ax.set_ylabel("number of (metric, alphabet-ksize pair) cells")
    ax.spines[["top", "right"]].set_visible(False)
    finish_figure(fig, path, tools=TOOLS + "; BHF regions from notebook 241 regions.parquet",
                  hypothesis=hypothesis, conclusion=conclusion, header_y=1.01, footer_y=-0.03, tight=False)
    return pl.DataFrame({"residue": x, "aa": list(bhf), "n_all": call.astype(int), "n_beats_95pct": cwin.astype(int)})


def human_sequences() -> dict[str, str]:
    """Gene symbol -> canonical human protein sequence (GENCODE v49, as searched)."""
    fa = Path("/Users/olga/data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa")
    out, name, buf = {}, None, []
    for line in open(fa):
        if line.startswith(">"):
            if name:
                out[name] = "".join(buf)
            name, buf = line[1:].strip().split("|")[6], []
        else:
            buf.append(line.strip())
    out[name] = "".join(buf)
    return out


def with_human_region(reg: pl.DataFrame) -> pl.DataFrame:
    """Adds the human side of each BHF best region (target_start/end from notebook 241's
    region table, 0-based end exclusive) and the share of K or R on both sides."""
    t = (pl.scan_parquet(DATA / "regions.parquet")
           .filter(pl.col("query_name").str.strip_chars() == "BHF")
           .select("alphabet", pl.col("ksize_arm").alias("k"), "gene",
                   pl.col("region_start").cast(pl.Int64).alias("start"),
                   pl.col("region_end").cast(pl.Int64).alias("end"),
                   pl.col("target_start").cast(pl.Int64), pl.col("target_end").cast(pl.Int64))
           .unique(["alphabet", "k", "gene", "start", "end"], keep="first", maintain_order=True)
           .collect())
    hum = human_sequences()
    kr = lambda q: round(sum(c in "KR" for c in q) / max(len(q), 1), 2)
    j = reg.join(t, on=["alphabet", "k", "gene", "start", "end"], how="left")
    return j.with_columns(
        pl.struct("gene", "target_start", "target_end").map_elements(
            lambda s: hum[s["gene"]][s["target_start"]:s["target_end"]] if s["target_start"] is not None else None,
            return_dtype=pl.Utf8).alias("human_seq"),
        pl.col("seq").map_elements(kr, return_dtype=pl.Float64).alias("KR_bhf"),
    ).with_columns(pl.col("human_seq").map_elements(kr, return_dtype=pl.Float64).alias("KR_human"))
