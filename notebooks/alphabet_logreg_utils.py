"""Learn how to combine kmerseek's values across alphabets (notebook 243).

Training data: the all-against-all search of the 998 Pfam-annotated human proteins with
every arm of the notebook 241 sweep (243_pfam998_search.py). The unit is a matched region,
as in PR #44, but pooled across arms: every region between the same query and target on
the same diagonal (within 10 residues; kmerseek regions are ungapped, so the same aligned
stretch lies on the same diagonal in every alphabet) whose query spans overlap is one
candidate.

Label: the candidate's query span and target span each overlap a domain of the same Pfam
family by at least 20% of the span's length (PR #44's "20% of the region" rule).

Split: the midi-plus truth set's own family split, "selection" (train) and "heldout"
(test); no family is in both. A candidate goes to the split of every family it touches
on either side; one that touches families of both splits is dropped, and one that
touches none goes to its proteins' split when all their domains are in one split.

Features are chosen so they mean the same in a 998-protein and a 19_732-protein
database: mean IDF over ln(number of proteins in the database), the bit score (not the
E-value), seed information in bits, counts and lengths.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import polars as pl

PF998 = Path("/Users/olga/data/alphabet-logreg-pfam998")
TRUTH = Path("/Users/olga/data/qfo-pfam-region-midi-plus/truth/human_domain_truth.parquet")
SWEEP = Path("/Users/olga/data/botryllus/alphabet-ranking-three-cases")
HUMAN = Path("/Users/olga/data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa")
N_HUMAN = 19_732
DIAG_TOL = 10

ALPHABETS19 = ["hp_lehninger2", "hp_thomas_dill2", "hp_kyte_doolittle2", "hp_thomas_dill_no_c2",
               "hp_lehninger_c_nonpolar2", "hp_pbotc_1st_ed2", "hp_lehninger_hpc3", "gbmr4", "polarity4",
               "wwmj5", "dayhoff6", "gbmr7", "funcgroups8", "sdm12", "mmseqs12", "wass14", "hsdm17",
               "uniprot18", "protein20"]
PER_ALPHABET = ["found", "max_seed_bits", "mean_idf_norm", "bit_score", "log_shared_kmers"]
GLOBAL = ["log_span_length", "log_query_length", "log_target_length", "n_alphabets", "n_arms",
          "hydrophobic_windows_query", "hydrophobic_windows_target"]
FEATURES = GLOBAL + [f"{a}__{f}" for a in ALPHABETS19 for f in PER_ALPHABET]


def read_fasta(p: Path, key=lambda h: h) -> dict[str, str]:
    out, name, buf = {}, None, []
    for line in open(p):
        if line.startswith(">"):
            if name is not None:
                out[name] = "".join(buf)
            name, buf = key(line[1:].strip()), []
        else:
            buf.append(line.strip())
    if name is not None:
        out[name] = "".join(buf)
    return out


def hydrophobic_windows(seq: str, start: int, end: int, w: int = 19, need: int = 14) -> int:
    """Windows of 19 residues inside [start, end) with at least 14 of AILMFVW: the length and
    make-up of a membrane-spanning helix. Measures the sticky-target effect of notebook 242."""
    s = seq[max(0, start):max(0, end)]
    if len(s) < w:
        return 0
    h = np.frombuffer(s.encode(), dtype=np.uint8)
    hyd = np.isin(h, np.frombuffer(b"AILMFVW", dtype=np.uint8)).astype(np.int32)
    c = np.convolve(hyd, np.ones(w, dtype=np.int32), mode="valid")
    return int((c >= need).sum())


def load_regions(files: list[Path] | pl.LazyFrame, n_db: int) -> pl.LazyFrame:
    """Regions of every arm, with the columns the candidates are built from."""
    lf = pl.scan_parquet(files) if isinstance(files, list) else files
    return lf.select(
        pl.col("query_name").alias("q"), pl.col("target_name").alias("t"),
        pl.col("region_start").cast(pl.Int64).alias("qs"), pl.col("region_end").cast(pl.Int64).alias("qe"),
        pl.col("target_start").cast(pl.Int64).alias("ts"), pl.col("target_end").cast(pl.Int64).alias("te"),
        "alphabet", pl.col("bits").cast(pl.Float64), pl.col("ksize").cast(pl.Int64),
        (pl.col("region_mean_idf") / math.log(n_db)).alias("mean_idf_norm"),
        pl.when(pl.col("region_ka_lambda") > 0).then(pl.col("region_ka_bits")).otherwise(0.0).fill_null(0.0).alias("bit_score"),
        pl.col("region_n_shared_kmers").fill_null(0).alias("n_shared"),
    )


def cluster_regions(regions: pl.LazyFrame) -> pl.LazyFrame:
    """Each region with the id of the candidate it belongs to (cid)."""
    r = regions.with_columns((pl.col("ts") - pl.col("qs")).alias("diag")).sort("q", "t", "diag")
    # single-linkage on the diagonal within one query-target pair
    # (polars does not nest window expressions, so each step is its own column)
    r = (r.with_columns(pl.col("diag").shift(1).over("q", "t").alias("_dprev"))
          .with_columns(((pl.col("diag") - pl.col("_dprev")) > DIAG_TOL).fill_null(True).cast(pl.Int32).alias("_dnew"))
          .with_columns(pl.col("_dnew").cum_sum().over("q", "t").alias("dg")))
    # then merge overlapping query spans within one diagonal group
    r = (r.sort("q", "t", "dg", "qs")
          .with_columns(pl.col("qe").cum_max().shift(1).over("q", "t", "dg").alias("_prev_end"))
          .with_columns((pl.col("qs") >= pl.col("_prev_end")).fill_null(True).cast(pl.Int32).alias("_new"))
          .with_columns(pl.col("_new").cum_sum().over("q", "t", "dg").alias("cl")))
    return r.with_columns(pl.struct("q", "t", "dg", "cl").hash(seed=0).alias("cid"))


def candidate_spans(clustered: pl.LazyFrame) -> pl.DataFrame:
    return (clustered.group_by("cid").agg(pl.col("q").first(), pl.col("t").first(), pl.col("qs").min(),
                                          pl.col("qe").max(), pl.col("ts").min(), pl.col("te").max())
            .collect())


def candidate_features(clustered: pl.LazyFrame, keep: pl.DataFrame | None = None) -> pl.DataFrame:
    """Per-alphabet and count features for the candidates in `keep` (a frame with cid),
    or for every candidate when keep is None."""
    r = clustered if keep is None else clustered.join(keep.lazy().select("cid"), on="cid", how="semi")
    key = ["cid"]
    per_alpha = (r.group_by(key + ["alphabet"])
                  .agg(pl.col("bits").max().alias("max_seed_bits"), pl.col("mean_idf_norm").max(),
                       pl.col("bit_score").max(), (pl.col("n_shared").max() + 1).log().alias("log_shared_kmers"),
                       pl.col("ksize").n_unique().alias("n_arms_a")))
    wide = (per_alpha.collect()
              .with_columns(pl.lit(1.0).alias("found"))
              .pivot(on="alphabet", index=key, values=PER_ALPHABET + ["n_arms_a"], separator="__"))
    # pivot names columns "<value>__<alphabet>"; rename to "<alphabet>__<value>"
    ren = {}
    for c in wide.columns:
        if "__" in c:
            v, a = c.split("__", 1)
            ren[c] = f"{a}__{v}"
    wide = wide.rename(ren)
    for a in ALPHABETS19:
        for f in PER_ALPHABET + ["n_arms_a"]:
            col = f"{a}__{f}"
            if col not in wide.columns:
                wide = wide.with_columns(pl.lit(0.0).alias(col))
    wide = wide.with_columns([pl.col(f"{a}__{f}").fill_null(0.0) for a in ALPHABETS19 for f in PER_ALPHABET + ["n_arms_a"]])
    wide = wide.with_columns(
        pl.sum_horizontal([pl.col(f"{a}__found") for a in ALPHABETS19]).alias("n_alphabets"),
        pl.sum_horizontal([pl.col(f"{a}__n_arms_a") for a in ALPHABETS19]).alias("n_arms"),
    ).drop([f"{a}__n_arms_a" for a in ALPHABETS19])
    return wide


def build_candidates(regions: pl.LazyFrame) -> pl.DataFrame:
    """Every candidate with its spans and features (used for one query's search)."""
    cl = cluster_regions(regions).collect().lazy()
    return candidate_features(cl).join(candidate_spans(cl), on="cid")


def add_sequence_features(c: pl.DataFrame, qseqs: dict[str, str], tseqs: dict[str, str]) -> pl.DataFrame:
    qlen = np.array([len(qseqs[q]) for q in c["q"]])
    tlen = np.array([len(tseqs[t]) for t in c["t"]])
    hq = [hydrophobic_windows(qseqs[q], s, e) for q, s, e in zip(c["q"], c["qs"], c["qe"])]
    ht = [hydrophobic_windows(tseqs[t], s, e) for t, s, e in zip(c["t"], c["ts"], c["te"])]
    return c.with_columns(
        ((pl.col("qe") - pl.col("qs")).cast(pl.Float64) + 1).log().alias("log_span_length"),
        pl.Series("log_query_length", np.log(qlen)), pl.Series("log_target_length", np.log(tlen)),
        pl.Series("hydrophobic_windows_query", np.log1p(hq)), pl.Series("hydrophobic_windows_target", np.log1p(ht)),
    )


def label_and_split_fast(spans: pl.DataFrame, truth: pl.DataFrame) -> pl.DataFrame:
    """The same rule as label_and_split, with joins instead of a loop, for tens of
    millions of candidates: correct when one Pfam family covers >= 20% of the query span
    and >= 20% of the target span; split from the families the candidate touches."""
    dom = truth.select(pl.col("accession"), pl.col("pfam_id").alias("fam"),
                       (pl.col("domain_start") - 1).alias("d0"), pl.col("domain_end").alias("d1"), "split")
    sides = []
    for acc, s0, s1 in (("q", "qs", "qe"), ("t", "ts", "te")):
        x = (spans.lazy().select("cid", pl.col(acc).alias("accession"), pl.col(s0).alias("s0"), pl.col(s1).alias("s1"))
             .join(dom.lazy(), on="accession")
             .with_columns((pl.min_horizontal("s1", "d1") - pl.max_horizontal("s0", "d0")).alias("o"))
             .filter(pl.col("o") > 0)
             .with_columns((pl.col("o") >= 0.2 * pl.max_horizontal(pl.col("s1") - pl.col("s0"), pl.lit(1))).alias("ok20")))
        sides.append(x)
    correct = (sides[0].filter("ok20").select("cid", "fam").unique()
               .join(sides[1].filter("ok20").select("cid", "fam").unique(), on=["cid", "fam"])
               .select("cid").unique().with_columns(pl.lit(True).alias("correct")))
    touched = (pl.concat([s.select("cid", "split") for s in sides]).unique()
               .group_by("cid").agg(pl.col("split").n_unique().alias("n_split"), pl.col("split").first().alias("tsplit")))
    prot = (truth.group_by("accession").agg(pl.col("split").n_unique().alias("n"), pl.col("split").first().alias("s"))
                 .with_columns(pl.when(pl.col("n") == 1).then(pl.col("s")).alias("psplit")).select("accession", "psplit"))
    out = (spans.lazy()
           .join(correct, on="cid", how="left").with_columns(pl.col("correct").fill_null(False))
           .join(touched, on="cid", how="left")
           .join(prot.lazy().rename({"accession": "q", "psplit": "qsplit"}), on="q", how="left")
           .join(prot.lazy().rename({"accession": "t", "psplit": "tsplit_p"}), on="t", how="left")
           .with_columns(
               pl.when(pl.col("n_split") == 1).then(pl.col("tsplit"))
                 .when(pl.col("n_split").is_null() & (pl.col("qsplit") == pl.col("tsplit_p"))).then(pl.col("qsplit"))
                 .otherwise(None).alias("split"))
           .drop("n_split", "tsplit", "qsplit", "tsplit_p"))
    return out.collect()


def label_and_split(c: pl.DataFrame, truth: pl.DataFrame) -> pl.DataFrame:
    """correct (same family on both sides, >= 20% of each span) and split."""
    from collections import defaultdict
    dom = defaultdict(list)
    for acc, fam, d0, d1, sp in truth.select("accession", "pfam_id", "domain_start", "domain_end", "split").iter_rows():
        dom[acc].append((fam, d0 - 1, d1, sp))
    prot_split = {a: {sp for *_, sp in v} for a, v in dom.items()}
    correct, split = [], []
    for q, t, qs, qe, ts, te in c.select("q", "t", "qs", "qe", "ts", "te").iter_rows():
        fams, touched = [], set()
        for acc, s0, s1 in ((q, qs, qe), (t, ts, te)):
            got = set()
            L = max(1, s1 - s0)
            for fam, d0, d1, sp in dom.get(acc, []):
                o = min(s1, d1) - max(s0, d0)
                if o > 0:
                    touched.add(sp)
                    if o >= 0.2 * L:
                        got.add(fam)
            fams.append(got)
        correct.append(bool(fams[0] & fams[1]))
        if len(touched) == 1:
            split.append(next(iter(touched)))
        elif len(touched) == 0:
            both = prot_split.get(q, set()) | prot_split.get(t, set())
            split.append(next(iter(both)) if len(both) == 1 else None)
        else:
            split.append(None)
    return c.with_columns(pl.Series("correct", correct), pl.Series("split", split, dtype=pl.Utf8))


def family_groups(truth: pl.DataFrame) -> dict[str, int]:
    """protein -> component id, proteins linked when they share a Pfam family, so a
    cross-validation fold never sees a family its training folds have seen."""
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for fam, accs in truth.group_by("pfam_id").agg(pl.col("accession").unique()).iter_rows():
        accs = list(accs)
        for a in accs[1:]:
            ra, rb = find(accs[0]), find(a)
            if ra != rb:
                parent[ra] = rb
    roots = {a: find(a) for a in truth["accession"].unique()}
    # Name each component by its smallest accession, not by its union-find root: the
    # root depends on group_by's row order, which changes between runs, and GroupKFold
    # breaks ties between equal-sized groups by id, so the folds (and the chosen C) moved.
    first: dict[str, str] = {}
    for a, r in roots.items():
        first[r] = min(first.get(r, a), a)
    ids = {m: i for i, m in enumerate(sorted(set(first.values())))}
    return {a: ids[first[r]] for a, r in roots.items()}


def fit_model(train: pl.DataFrame, features: list[str], neg_per_pos: int = 50, seed: int = 0):
    """L2 logistic regression on standardised features. Negatives are subsampled to
    neg_per_pos per positive and reweighted so the fit targets the full prevalence; C is
    chosen by 5-fold cross-validation on average precision, with folds grouped by Pfam
    family component (family_groups on the query protein)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(seed)
    pos = train.filter(pl.col("correct"))
    neg = train.filter(~pl.col("correct"))
    if "weight" in train.columns:
        d = train
        w = d["weight"].to_numpy()
    else:
        n_neg = min(neg.height, neg_per_pos * pos.height)
        d = pl.concat([pos, neg.sample(n_neg, seed=seed)])
        w = np.where(d["correct"].to_numpy(), 1.0, neg.height / max(n_neg, 1))
    X, y = d.select(features).to_numpy(), d["correct"].to_numpy()
    fg = family_groups(pl.read_parquet(TRUTH))
    groups = np.array([fg.get(q, -1) for q in d["q"]])
    best, scores = None, {}
    for C in (0.01, 0.1, 1.0, 10.0):
        aps = []
        for tr, va in GroupKFold(5).split(X, y, groups):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
            m.fit(X[tr], y[tr], logisticregression__sample_weight=w[tr])
            aps.append(average_precision_score(y[va], m.predict_proba(X[va])[:, 1], sample_weight=w[va]))
        scores[C] = float(np.mean(aps))
        if best is None or scores[C] > scores[best]:
            best = C
    model = make_pipeline(StandardScaler(), LogisticRegression(C=best, max_iter=2000))
    model.fit(X, y, logisticregression__sample_weight=w)
    n_neg_total = float(d.filter(~pl.col("correct"))["weight"].sum()) if "weight" in d.columns else neg.height
    return model, {"C": best, "cv_ap": scores, "n_pos": pos.height, "n_neg": int(round(n_neg_total))}


def training_table(files: list[Path] | None = None) -> pl.DataFrame:
    """Candidates of the 998-protein all-against-all, with features, label and split.
    Cached in candidates.parquet, rebuilt when the set of searched arms changes."""
    files = files or sorted((PF998 / "regions").glob("*.parquet"))
    cache, stamp = PF998 / "candidates.parquet", PF998 / "candidates.arms.txt"
    names = "\n".join(f"{f.name} {f.stat().st_mtime_ns}" for f in files)
    if cache.exists() and stamp.exists() and stamp.read_text() == names:
        return pl.read_parquet(cache)
    t = _training_table(files)
    t.write_parquet(cache)
    stamp.write_text(names)
    return t


def _training_table(files: list[Path], neg_per_pos: int = 50, heldout_neg: int = 3_000_000, seed: int = 0) -> pl.DataFrame:
    """Every correct candidate; neg_per_pos incorrect ones per correct one in the training
    families; heldout_neg incorrect ones in the held-out families. Each kept incorrect
    candidate carries weight = (incorrect in its split) / (kept), so weighted fits and
    weighted average precision stand for the whole set."""
    truth = pl.read_parquet(TRUTH)
    seqs = read_fasta(PF998 / "human_pfam_truth.fa")
    cl = cluster_regions(load_regions(files, n_db=998)).collect().lazy()
    lab = label_and_split_fast(candidate_spans(cl), truth).filter(pl.col("split").is_not_null())
    parts = []
    for sp, n_keep in (("selection", None), ("heldout", heldout_neg)):
        x = lab.filter(pl.col("split") == sp)
        pos, neg = x.filter(pl.col("correct")), x.filter(~pl.col("correct"))
        k = min(neg.height, n_keep if n_keep else neg_per_pos * pos.height)
        parts += [pos.with_columns(pl.lit(1.0).alias("weight")),
                  neg.sample(k, seed=seed).with_columns(pl.lit(neg.height / max(k, 1)).alias("weight"))]
    keep = pl.concat(parts)
    feats = candidate_features(cl, keep)
    t = keep.join(feats, on="cid")
    t = add_sequence_features(t, seqs, seqs)
    stats = lab.group_by("split", "correct").agg(pl.len().alias("candidates"))
    stats.write_csv(PF998 / "candidate_counts.csv")
    return t


def _training_table_unsampled(files: list[Path]) -> pl.DataFrame:
    truth = pl.read_parquet(TRUTH)
    seqs = read_fasta(PF998 / "human_pfam_truth.fa")
    c = build_candidates(load_regions(files, n_db=998))
    c = add_sequence_features(c, seqs, seqs)
    return label_and_split(c, truth)


def query_table(query: str, n_db: int = N_HUMAN) -> pl.DataFrame:
    """Candidates for one of notebook 241's queries (Ced9, P66, BHF) against the human
    proteome, with the same features; `gene` is the target's gene symbol."""
    lf = pl.scan_parquet(SWEEP / "regions.parquet").filter(pl.col("query_name") == query).rename({"ksize_arm": "ksize"})
    c = build_candidates(load_regions(lf, n_db=n_db))
    qseqs = read_fasta(SWEEP / "queries.fa")
    tseqs = read_fasta(HUMAN)
    c = add_sequence_features(c, qseqs, tseqs)
    return c.with_columns(pl.col("t").str.split("|").list.get(6).alias("gene"))


# ---------------------------------------------------------------------------
# Feature sets compared in notebook 243
# ---------------------------------------------------------------------------
LENGTH_COMPOSITION = ["log_span_length", "log_query_length", "log_target_length",
                      "hydrophobic_windows_query", "hydrophobic_windows_target"]
FEATURE_SETS = {
    "length and hydrophobic make-up only": LENGTH_COMPOSITION,
    "+ hp_lehninger2's own values": LENGTH_COMPOSITION + [f"hp_lehninger2__{f}" for f in PER_ALPHABET],
    "+ all 19 alphabets": FEATURES,
}


def midrank(score: np.ndarray, i: int, higher_is_better: bool = True) -> float:
    s = score if higher_is_better else -score
    return 1 + float((s > s[i]).sum()) + ((s == s[i]).sum() - 1) / 2


def target_scores(c: pl.DataFrame, prob: np.ndarray) -> pl.DataFrame:
    """One score per target protein: its best candidate's probability, plus the same
    for the single-feature rankers shown beside the model."""
    return (c.with_columns(pl.Series("prob", prob))
              .group_by("t", "gene").agg(pl.col("prob").max(), pl.col("n_alphabets").max(),
                                         pl.col("log_span_length").max(),
                                         pl.col("hp_lehninger2__bit_score").max(),
                                         pl.col("hydrophobic_windows_target").max()))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from alphabet_ranking_utils import _figure_with_legend_row, RAMP  # noqa: E402
from hp_conservation_utils import finish_figure  # noqa: E402

FIG = Path(__file__).resolve().parent.parent / "figures"
PURPLE, GREY, RED = RAMP(0.85), "#9a9a9a", "#c0392b"
TOOLS = ("kmerseek 0.4.0 (982a055): all-against-all of the 998 Pfam-annotated human proteins on every arm of the "
         "notebook 241 sweep (243_pfam998_search.py); logistic regression, scikit-learn, trained on the 'selection' "
         "Pfam families and tested on the 'heldout' families of the midi-plus truth set")


def fig_heldout(curves: dict, prevalence: float, path: Path, hypothesis: str, conclusion: str):
    """Precision-recall on the held-out families, one line per ranker. The three models
    share one hue at three weights; single features are grey, each with its own dash."""
    styles = {
        "+ all 19 alphabets": dict(color=PURPLE, lw=2.4, ls="-"),
        "+ hp_lehninger2's own values": dict(color=RAMP(0.55), lw=2.0, ls="-"),
        "length and hydrophobic make-up only": dict(color=RAMP(0.3), lw=2.0, ls="-"),
        "number of alphabets that find it": dict(color=GREY, lw=1.4, ls="--"),
        "hp_lehninger2 bit score": dict(color=GREY, lw=1.4, ls=":"),
        "candidate length": dict(color=GREY, lw=1.4, ls="-."),
    }
    words = {"+ all 19 alphabets": "model: length, make-up and all 19 alphabets",
             "+ hp_lehninger2's own values": "model: length, make-up and hp_lehninger2 only",
             "length and hydrophobic make-up only": "model: length and hydrophobic make-up only",
             "number of alphabets that find it": "one feature: number of alphabets that find it",
             "hp_lehninger2 bit score": "one feature: hp_lehninger2 bit score",
             "candidate length": "one feature: length of the matched stretch"}
    handles = [Line2D([], [], label=f"{words[k]} (average precision {curves[k][2]:.3f})", **v) for k, v in styles.items() if k in curves]
    handles.append(Line2D([], [], color=RED, ls="--", lw=1.2, label=f"chance: the share of candidate matches that are correct ({prevalence:.4f})"))
    fig, (ax,) = _figure_with_legend_row(1, (9.5, 7.2), handles)
    # Single features first, the full model last: where every curve sits at precision 1
    # the full model must stay visible on top.
    for k, v in reversed(list(styles.items())):
        if k in curves:
            rec, prec, _ = curves[k]
            ax.plot(rec, prec, **v)
    ax.axhline(prevalence, color=RED, ls="--", lw=1.2)
    ax.set_ylabel("precision: share of candidates kept that are correct", fontsize=9)
    # Recall on a log axis: every ranker's precision has fallen to near chance by 30%
    # recall, so on a linear axis the whole comparison sits in the left tenth.
    ax.set_xscale("log"); ax.set_xlim(1e-4, 1); ax.set_ylim(0, 1)
    ax.set_xticks([1e-4, 1e-3, 1e-2, 0.1, 1]); ax.set_xticklabels(["0.01%", "0.1%", "1%", "10%", "100%"])
    ax.set_xlabel("recall: share of the held-out correct candidates found (log scale)", fontsize=9)
    ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    finish_figure(fig, path, tools=TOOLS, hypothesis=hypothesis, conclusion=conclusion, header_y=1.01, footer_y=-0.02)


GLOBAL_LABELS = {
    "log_span_length": "length of the matched stretch",
    "log_query_length": "length of the query protein",
    "log_target_length": "length of the target protein",
    "n_alphabets": "number of alphabets that found it",
    "n_arms": "number of arms that found it",
    "hydrophobic_windows_query": "membrane-helix-like windows, query side",
    "hydrophobic_windows_target": "membrane-helix-like windows, target side",
}
PER_ALPHABET_LABELS = {
    "found": "found at all",
    "max_seed_bits": "highest seed\ninformation (bits)",
    "mean_idf_norm": "best mean IDF\n/ ln(proteins)",
    "bit_score": "best bit\nscore",
    "log_shared_kmers": "most shared\nk-mers",
}


def feature_label(f: str) -> str:
    """A feature column in words, for figure labels and printed tables."""
    if "__" in f:
        a, v = f.split("__")
        return f"{a}, {PER_ALPHABET_LABELS[v].replace(chr(10), ' ')}"
    return GLOBAL_LABELS[f]


def fig_coefficients(model, features: list[str], path: Path, hypothesis: str, conclusion: str) -> pl.DataFrame:
    """Standardised coefficients of the full model. Left: the length and composition
    features. Right: one row per alphabet, one column per value. One colour scale for
    both panels: purple raises the chance a candidate is correct, orange lowers it."""
    from matplotlib.patches import Patch
    coef = dict(zip(features, model.named_steps["logisticregression"].coef_[0]))
    lim = max(abs(v) for v in coef.values())
    cmap = plt.get_cmap("PuOr")          # low (negative) = orange, high (positive) = purple
    col = lambda v: cmap(0.5 + 0.5 * v / lim)
    handles = [Patch(color=cmap(0.9), label="purple: a higher value makes a correct match more likely"),
               Patch(color=cmap(0.1), label="orange: a higher value makes a correct match less likely"),
               Patch(color=cmap(0.5), label="pale: little weight; the deeper the colour, the larger the weight")]
    fig, axes = _figure_with_legend_row(2, (13.5, 7.4), handles, sharey=False, width_ratios=[1, 2.2])
    ax = axes[0]
    g = [f for f in features if "__" not in f]
    ax.barh(range(len(g)), [coef[f] for f in g], color=[col(coef[f]) for f in g], edgecolor="#555555", lw=0.5)
    ax.set_yticks(range(len(g))); ax.set_yticklabels([GLOBAL_LABELS[f] for f in g], fontsize=8.5); ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8); ax.set_xlim(-lim * 1.05, lim * 1.05)
    ax.set_xlabel("weight (change in log-odds per standard deviation)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False); ax.set_title("length, make-up, counts", fontsize=10)
    ax = axes[1]
    M = np.array([[coef[f"{a}__{f}"] for f in PER_ALPHABET] for a in ALPHABETS19])
    im = ax.imshow(M, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(PER_ALPHABET))); ax.set_xticklabels([PER_ALPHABET_LABELS[f] for f in PER_ALPHABET], fontsize=8)
    ax.set_yticks(range(len(ALPHABETS19))); ax.set_yticklabels(ALPHABETS19, fontsize=8)
    ax.set_title("each alphabet's values", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label("weight, same scale as the bars", fontsize=8)
    finish_figure(fig, path, tools=TOOLS, hypothesis=hypothesis, conclusion=conclusion, header_y=1.01, footer_y=-0.02)
    return (pl.DataFrame({"feature": [feature_label(f) for f in coef], "coefficient": [round(float(v), 3) for v in coef.values()]})
              .sort("coefficient", descending=True))


def fig_application(tab: pl.DataFrame, path: Path, hypothesis: str, conclusion: str):
    """Rows: (pair, ranker). x: the partner's rank among the human proteins with any
    candidate (log scale). Grey rows: the control with no known link."""
    handles = [
        Line2D([], [], marker="D", ls="", ms=8, color=PURPLE, label="the trained model (all 19 alphabets), a target scored by its best candidate"),
        Line2D([], [], marker="o", ls="", ms=7, markerfacecolor="white", markeredgecolor=PURPLE, markeredgewidth=1.5, label="one feature alone, for comparison"),
        Line2D([], [], marker="D", ls="", ms=8, color=GREY, label="grey: the same for the pair with no known link, the control"),
    ]
    fig, (ax,) = _figure_with_legend_row(1, (10.5, 6.2), handles)
    y, ticks, labels = 0, [], []
    for (q, p, title) in [("Ced9", "BCL2", "Ced9 → BCL2, known homolog"), ("P66", "CD47", "P66 → CD47, proposed partner"),
                          ("Ced9", "CD47", "Ced9 → CD47, no known link (control)")]:
        col = GREY if (q, p) == ("Ced9", "CD47") else PURPLE
        s = tab.filter((pl.col("query") == q) & (pl.col("partner") == p))
        ax.text(0.85, y - 0.6, title, fontsize=9.5, fontweight="bold", va="bottom")
        for r in s.iter_rows(named=True):
            model = r["ranker"] == "model"
            ax.scatter([r["rank"]], [y], marker="D" if model else "o", s=60,
                       facecolors=col if model else "white", edgecolors=col, linewidths=1.5, zorder=3)
            ax.annotate(f"{r['rank']:_.0f}", (r["rank"], y), xytext=(8, -3), textcoords="offset points", fontsize=7.5)
            ticks.append(y); labels.append(r["ranker"]); y += 1
        y += 1.4
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=8.5); ax.set_ylim(y - 1, -1.4)
    ax.set_xscale("log"); ax.set_xlim(0.8, 3e4)
    ax.set_xticks([1, 10, 100, 1000, 10_000]); ax.set_xticklabels(["1", "10", "100", "1_000", "10_000"])
    ax.set_xlabel("rank of the partner among the human proteins with any candidate (1 = best, log scale)", fontsize=9)
    ax.grid(axis="x", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    finish_figure(fig, path, tools=TOOLS.replace("tested on the 'heldout' families of the midi-plus truth set",
                  "applied to notebook 241's searches of Ced9, P66 and BHF against the 19_732 human proteins"),
                  hypothesis=hypothesis, conclusion=conclusion, header_y=1.01, footer_y=-0.02)
