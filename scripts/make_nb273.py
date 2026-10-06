#!/usr/bin/env python3
"""Generate notebooks/273_hp_agreement_per_100_positions_pfam_seed_scope40.ipynb."""

import json
from pathlib import Path

cells = []


def md(source):
    cells.append({"cell_type": "markdown", "id": f"md-{len(cells):02d}", "metadata": {},
                  "source": source.strip().splitlines(keepends=True)})


def code(source):
    cells.append({"cell_type": "code", "id": f"code-{len(cells):02d}", "execution_count": None,
                  "metadata": {"jupyter": {"source_hidden": True}}, "outputs": [],
                  "source": source.strip("\n").splitlines(keepends=True)})


md(r"""
# 273. H/P agreement between aligned homologs, measured as positions out of 100

No kmerseek in this notebook. It re-reads the aligned pairs of
[notebook 230](https://github.com/seanome/2024-kmerseek-analysis/blob/main/notebooks/230_hp_class_conservation_in_aligned_homologs.ipynb)
and counts, for each pair, how many aligned positions put the two residues in the same
class. Notebook 230 reported this as Cohen's kappa. The numbers "74 of 100 positions agree
in H/P at 20-30% identity, against 51 by chance" were then converted back from kappa using
Swiss-Prot residue shares. Here they are counted directly.

**Data.** The same files notebook 230 reads, with the same filters:

* Pfam-A 38.2 seed alignments: 163,448 sampled pairs from 29,151 families, up to 6 pairs
  per family (`~/data/pfam/230_pfam_seed_pair_alignments.parquet`, built by
  `scripts/pfam_seed_pair_alignments.py`). Pairs with at least 50 aligned positions.
* SCOPe 2.08, 40% identity set: 38,267 domain pairs aligned by structure with USalign
  (`~/data/scope/230_scope40_pair_alignments.parquet`, built by
  `scripts/scope_pair_structural_alignments.py`). Pairs with at least 50 aligned positions
  and TM-score at least 0.5 on one of the two domains.

**Alphabets.** `hp_thomas_dill2`: hydrophobic `ACFILMVWY`, polar `DEGHKNPQRST`.
`protein20`: the 20 amino acids, so "same class" means identical residue.

**Per pair** (an aligned position is a column where neither sequence has a gap):

$$\Pr(\text{agree}) = \frac{\text{aligned positions where both residues are in the same class}}{\text{aligned positions}}$$

$$\Pr(\text{agree by chance}) = h_\text{query}\,h_\text{target} + p_\text{query}\,p_\text{target}$$

where $h_\text{query}$ is the share of the query's aligned residues that are hydrophobic and
$p_\text{query} = 1 - h_\text{query}$ the share that are polar (for protein20, the sum runs
over all 20 amino acids). The shares are the pair's own, not Swiss-Prot's.

**Shuffled null.** The target's residues are shuffled among its own residue positions
(composition and gap pattern kept), and the same aligned columns are scored again. Ten
shuffles per pair, averaged. This measures chance instead of computing it from shares.

**Evidence per aligned position**, in bits: how far the pair's agree/disagree rate is from
the chance rate (the Kullback-Leibler divergence of the two coin flips),

$$I = \Pr(\text{agree})\,\log_2\frac{\Pr(\text{agree})}{\Pr(\text{chance})} + \bigl(1-\Pr(\text{agree})\bigr)\,\log_2\frac{1-\Pr(\text{agree})}{1-\Pr(\text{chance})}$$

computed per pair and averaged over pairs. Zero means the alignment looks like chance;
higher is more evidence of relatedness from each aligned position. The same quantity is
computed for each shuffled pair, against that shuffle's own chance rate. It is above zero
because a finite alignment never lands on its chance rate exactly, so the shuffled value is
the floor to compare against.

**The estimates under test** (from kappa, with Swiss-Prot shares): at 20-30% identity,
H/P 74 agreeing per 100 against 51 by chance; 20 amino acids 25 against 6; I about 0.15 bits
for H/P against 0.26 for 20 amino acids.

**Decision rule, written before running.** If a measured value at 20-30% identity in Pfam is
within 3 positions per 100 (or 0.03 bits for I) of the estimate, the estimate can be quoted.
Otherwise the measured value replaces it in the paper and the SAB material, and this
notebook is the source.
""")

code(r"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path.cwd()))
import hp_conservation_utils as hc

pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(220)

FIG = Path("../figures")
PFAM_ALN = Path("/Users/olga/data/pfam/230_pfam_seed_pair_alignments.parquet")
SCOPE_ALN = Path("/Users/olga/data/scope/230_scope40_pair_alignments.parquet")
PFAM_OUT = Path("/Users/olga/data/pfam/273_pfam_seed_pair_agreement_shuffled.parquet")
SCOPE_OUT = Path("/Users/olga/data/scope/273_scope40_pair_agreement_shuffled.parquet")
PFAM_230 = Path("/Users/olga/data/pfam/230_pfam_seed_pair_class_agreement.parquet")
MIN_COLS = 50  # same filter as notebook 230
MIN_TM = 0.5  # SCOPe only, same as notebook 230
N_SHUFFLE = 10
SEED = 273
HP = "hp_thomas_dill2"
P20 = "protein20"
ALPHABETS = [HP, P20]


def load_or_compute(aln_path, out_path, keys):
    # Per-pair agreement, chance, shuffled agreement and evidence; cached to out_path.
    if out_path.exists():
        return pl.read_parquet(out_path)
    aln = pl.read_parquet(aln_path)
    t = hc.compute_pair_table(aln, keys, ALPHABETS, n_shuffle=N_SHUFFLE, seed=SEED)
    t.write_parquet(out_path)
    return t


pfam_raw = load_or_compute(
    PFAM_ALN, PFAM_OUT, ["family", "family_id", "query", "target", "seqid_ali"]
)
scope_raw = load_or_compute(
    SCOPE_ALN, SCOPE_OUT, ["query", "target", "category", "seqid_ali", "tm_q", "tm_t"]
)

pfam = hc.add_identity_bin(pfam_raw).filter(pl.col("n_cols") >= MIN_COLS)
scope = (
    hc.add_identity_bin(scope_raw)
    .filter(pl.col("n_cols") >= MIN_COLS)
    .filter(pl.max_horizontal("tm_q", "tm_t") >= MIN_TM)
)

CAT_ORDER = [
    "same_family",
    "same_superfamily_diff_family",
    "same_fold_diff_superfamily",
    "diff_fold_same_class",
]
CAT_LABEL = {
    "same_family": "same family",
    "same_superfamily_diff_family": "same superfamily,\ndifferent family",
    "same_fold_diff_superfamily": "same fold,\ndifferent superfamily",
    "diff_fold_same_class": "different fold,\nsame class",
}
COLOR = {HP: "#a50f15", P20: "#444444"}  # notebook 230's colours for these two alphabets

# Check against notebook 230: agreement and chance must be identical for the same pairs.
old = pl.read_parquet(PFAM_230).filter(pl.col("alphabet").is_in(ALPHABETS))
chk = pfam_raw.join(
    old.select("family", "query", "target", "alphabet", "agree", "expected"),
    on=["family", "query", "target", "alphabet"],
    suffix="_230",
)
print(
    f"Pfam pairs x alphabets: {pfam_raw.height:_} computed here, {chk.height:_} matched to nb 230; "
    f"max |agree - agree_230| = {(chk['agree'] - chk['agree_230']).abs().max():.2e}, "
    f"max |chance - chance_230| = {(chk['expected'] - chk['expected_230']).abs().max():.2e}"
)
print(
    f"After filters: Pfam {pfam.filter(pl.col('alphabet') == HP).height:_} pairs, "
    f"SCOPe {scope.filter(pl.col('alphabet') == HP).height:_} pairs"
)
""")

md(r"""
## 1. One pair, position by position

A Bcl-2 family pair from the Pfam seed (PF00452): rat BCL2L10 (`B2L10_RAT`, residues
39-145) against an opossum Bcl-2 family protein (`F6ZMX1_MONDO`, residues 218-317), 28%
identical over aligned positions, to show what is counted. Line
by line: the two aligned sequences; `|` where the residues are identical; the same two
sequences written as H (hydrophobic) and P (polar); `|` where the classes agree; then the
target after one shuffle of its residues, in H/P, with its agreement line. Positions with a
gap on either side (`-` or `.`) are not counted. The figure below draws the same lines,
one row each, with the count for each agreement row at its right.
""")

code(r"""
aln = pl.read_parquet(PFAM_ALN)
ex = (
    aln.filter(
        (pl.col("family") == "PF00452")
        & (pl.col("query") == "B2L10_RAT/39-145")
        & (pl.col("target") == "F6ZMX1_MONDO/218-317")
    ).row(0, named=True)
)
q, t = ex["qaln"], ex["taln"]
cls = {r: "H" for r in hc.ALPHABET_CLUSTERS[HP][0]} | {
    r: "P" for r in hc.ALPHABET_CLUSTERS[HP][1]
}


def hp_letter(c):
    return cls.get(c.upper(), "-")


qh = "".join(hp_letter(c) for c in q)
th = "".join(hp_letter(c) for c in t)
both = [hp_letter(a) != "-" and hp_letter(b) != "-" for a, b in zip(q, t)]
rng = np.random.default_rng(SEED)
t_list = list(t)
res_pos = [i for i, c in enumerate(t) if hp_letter(c) != "-"]
perm = rng.permutation(res_pos)
t_shuf = t_list.copy()
for i, j in zip(res_pos, perm):
    t_shuf[i] = t[j]
tsh = "".join(hp_letter(c) for c in t_shuf)
id_line = "".join(
    "|" if ok and a.upper() == b.upper() else " " for ok, a, b in zip(both, q, t)
)
hp_line = "".join("|" if ok and a == b else " " for ok, a, b in zip(both, qh, th))
sh_line = "".join("|" if ok and a == b else " " for ok, a, b in zip(both, qh, tsh))

n_al = sum(both)
n_id = id_line.count("|")
n_hp = hp_line.count("|")
n_sh = sh_line.count("|")
h_q = sum(1 for ok, a in zip(both, qh) if ok and a == "H") / n_al
h_t = sum(1 for ok, b in zip(both, th) if ok and b == "H") / n_al
chance = h_q * h_t + (1 - h_q) * (1 - h_t)
print(f"query  {ex['query']}\ntarget {ex['target']}\nfamily PF00452 (Bcl-2)")
print(
    f"aligned positions: {n_al}; identical residues: {n_id} ({100 * n_id / n_al:.0f} per 100); "
    f"same H/P class: {n_hp} ({100 * n_hp / n_al:.0f} per 100); "
    f"same H/P class after one shuffle: {n_sh} ({100 * n_sh / n_al:.0f} per 100)"
)
print(
    f"h_query = {h_q:.2f}, h_target = {h_t:.2f}; Pr(agree by chance) = "
    f"{h_q:.2f} x {h_t:.2f} + {1 - h_q:.2f} x {1 - h_t:.2f} = {chance:.2f} ({100 * chance:.0f} per 100)\n"
)
W = 60
for s in range(0, len(q), W):
    e = s + W
    print(f"query            {q[s:e]}")
    print(f"identical        {id_line[s:e]}")
    print(f"target           {t[s:e]}")
    print(f"query H/P        {qh[s:e]}")
    print(f"same H/P class   {hp_line[s:e]}")
    print(f"target H/P       {th[s:e]}")
    print(f"shuffled target  {tsh[s:e]}")
    print(f"same H/P class   {sh_line[s:e]}")
    print(f"columns {s + 1}-{min(e, len(q))}\n")

row = pfam.filter(
    (pl.col("query") == ex["query"])
    & (pl.col("target") == ex["target"])
    & (pl.col("alphabet") == HP)
)
print("The same pair in the computed table (10 shuffles averaged):")
print(
    row.select(
        "n_cols",
        (pl.col("agree") * 100).round(1).alias("agree_per_100"),
        (pl.col("expected") * 100).round(1).alias("chance_per_100"),
        (pl.col("agree_null") * 100).round(1).alias("shuffled_per_100"),
        pl.col("evidence").round(3).alias("I_bits"),
        pl.col("evidence_null").round(3).alias("I_shuffled_bits"),
    )
)
""")

code(r"""
H_COLOR, P_COLOR = "#d08c1e", "#2a9d8f"  # H and P residues; used for nothing else
AGREE_COLOR = "black"
h_res = "".join(sorted(hc.ALPHABET_CLUSTERS[HP][0]))
p_res = "".join(sorted(hc.ALPHABET_CLUSTERS[HP][1]))
cols = np.arange(1, len(q) + 1)
rows = [
    ("identical residue", "agree", id_line, n_id),
    ("query H/P", "class", qh, None),
    ("target H/P", "class", th, None),
    ("same H/P class", "agree", hp_line, n_hp),
    ("shuffled target H/P", "class", tsh, None),
    ("same H/P class, shuffled", "agree", sh_line, n_sh),
]
fig, ax = plt.subplots(figsize=(12, 3.6))
for y, (name, kind, line, n) in enumerate(rows):
    if kind == "class":
        for x, c in zip(cols, line):
            if c in "HP":
                ax.add_patch(plt.Rectangle((x - 0.5, y - 0.35), 1, 0.7, lw=0,
                                           color=H_COLOR if c == "H" else P_COLOR))
    else:
        hit = [x for x, ok, c in zip(cols, both, line) if ok and c == "|"]
        miss = [x for x, ok, c in zip(cols, both, line) if ok and c != "|"]
        ax.scatter(hit, [y] * len(hit), marker="s", s=14, color=AGREE_COLOR, lw=0)
        ax.scatter(miss, [y] * len(miss), marker=".", s=10, color="#9a9a9a", lw=0)
        ax.text(len(q) + 2, y, f"{n} of {n_al} ({100 * n / n_al:.0f} per 100)",
                va="center", ha="left", fontsize=9)
gap = [x for x, ok in zip(cols, both) if not ok]
for y, (_, kind, _, _) in enumerate(rows):
    if kind == "agree":
        ax.scatter(gap, [y] * len(gap), marker="x", s=10, color="#c8c8c8", lw=0.8)
ax.set_yticks(range(len(rows)), [r[0] for r in rows])
ax.set_ylim(len(rows) - 0.5, -0.5)
ax.set_xlim(0.5, len(q) + 0.5)
ax.set_xlabel("alignment column (gaps included)")
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.tick_params(axis="y", length=0)
handles = [
    Patch(color=H_COLOR, label=f"H, hydrophobic: {h_res}"),
    Patch(color=P_COLOR, label=f"P, polar: {p_res}"),
    Line2D([], [], ls="", marker="s", ms=5, color=AGREE_COLOR, label="the two residues agree (counted)"),
    Line2D([], [], ls="", marker=".", ms=6, color="#9a9a9a", label="the two residues differ (counted)"),
    Line2D([], [], ls="", marker="x", ms=4, color="#c8c8c8", mew=0.8,
           label="gap in either sequence (not counted)"),
]
ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3,
          fontsize=9, frameon=False, borderaxespad=0)
print(f"aligned positions counted: {n_al}; gap columns: {len(gap)}; total columns: {len(q)}")
print(f"identical {n_id}, same H/P class {n_hp}, same H/P class after one shuffle {n_sh}")
hc.finish_figure(
    fig,
    FIG / "273_one_pair_position_by_position_bcl2_pfam_seed.png",
    tools=f"no kmerseek; Pfam seed alignment, {HP} classes",
    title=f"One Bcl-2 pair (PF00452), {ex['query']} against {ex['target']}",
    hypothesis=(
        "H/P agreement between homologs is above what the same residues give when shuffled. "
        "More black squares in a row is more agreement."
    ),
    conclusion=(
        f"Of {n_al} aligned positions, {n_id} hold the same residue, {n_hp} the same H/P class, "
        f"and {n_sh} the same H/P class after the target residues are shuffled "
        f"(chance from the two sequences' class shares: {100 * chance:.0f} per 100)."
    ),
)
""")

md(r"""
## 2. Positions agreeing per 100, real pairs against shuffled, and evidence per position

Top row: mean share of aligned positions in the same class, per 100, with a 95% bootstrap
interval of the mean (500 resamples of pairs). Solid bars are the real pairs, hatched bars
the same pairs with one sequence shuffled. The black line across each pair of bars is the
chance rate from the pairs' own class shares. The hatched bar should sit on it, which checks
the formula against the shuffle. It sits up to 0.3 per 100 above the line for H/P, because the
shuffle also draws the target residues that face a gap in the query, and those are more polar
than the aligned ones. Bottom row: mean evidence per aligned position, I, in bits.
Higher is better; the hatched bar is what an unrelated pair of the same composition and
length scores.

Left column: Pfam seed pairs by identity over aligned positions. Right column: SCOPe pairs by
how SCOPe relates the two domains, all identities pooled. The SCOPe categories differ in
identity (median identity per category is printed under the figure), so the right column mixes
relatedness with identity.
""")

code(r"""
METRICS = ["agree", "agree_null", "expected", "evidence", "evidence_null"]


def summary(df, by):
    out = None
    for m in METRICS:
        s = hc.summarise(df, by + ["alphabet"], m)
        out = s if out is None else out.join(s.drop("n"), on=by + ["alphabet"])
    return out


pf_s = summary(pfam, ["identity_bin"]).with_columns(
    pl.col("identity_bin").cast(pl.Enum(hc.IDENTITY_LABELS))
)
sc_s = summary(scope, ["category"]).with_columns(
    pl.col("category").cast(pl.Enum(CAT_ORDER))
)


def per100_table(s, key):
    return (
        s.sort(key, "alphabet")
        .select(
            key,
            "alphabet",
            pl.col("n").alias("pairs"),
            *[
                pl.format(
                    "{} [{}-{}]",
                    (pl.col(f"{m}_mean") * 100).round(1),
                    (pl.col(f"{m}_lo") * 100).round(1),
                    (pl.col(f"{m}_hi") * 100).round(1),
                ).alias(lab)
                for m, lab in [
                    ("agree", "agree per 100"),
                    ("agree_null", "shuffled per 100"),
                    ("expected", "chance per 100"),
                ]
            ],
            *[
                pl.format(
                    "{} [{}-{}]",
                    pl.col(f"{m}_mean").round(3),
                    pl.col(f"{m}_lo").round(3),
                    pl.col(f"{m}_hi").round(3),
                ).alias(lab)
                for m, lab in [("evidence", "I bits"), ("evidence_null", "I shuffled bits")]
            ],
        )
    )


print("Pfam-A 38.2 seed pairs, mean [95% bootstrap interval]")
print(per100_table(pf_s, "identity_bin"))
print("\nSCOPe 2.08 40% set, USalign pairs, mean [95% bootstrap interval]")
print(per100_table(sc_s, "category"))
print("\nSCOPe identity over aligned positions, per category, after this notebook's filters")
print(
    scope.filter(pl.col("alphabet") == HP)
    .group_by("category")
    .agg(
        pl.len().alias("pairs"),
        (pl.col("seqid_ali").median() * 100).round(1).alias("median identity %"),
        ((pl.col("seqid_ali") < 0.2).mean() * 100).round(1).alias("% of pairs under 20%"),
    )
    .with_columns(pl.col("category").cast(pl.Enum(CAT_ORDER)))
    .sort("category")
)

W = 0.2
CHANCE_COLOR = "#2166ac"  # used for nothing else
SLOTS = [(HP, False, -1.5), (HP, True, -0.5), (P20, False, 0.5), (P20, True, 1.5)]


def draw(ax_top, ax_bot, s, key, order, ticklabels):
    x = np.arange(len(order))
    for a, shuffled, off in SLOTS:
        d = s.filter(pl.col("alphabet") == a).sort(key)
        m = "agree_null" if shuffled else "agree"
        e = "evidence_null" if shuffled else "evidence"
        xi = x + off * W
        style = (
            dict(facecolor="white", edgecolor=COLOR[a], hatch="///", lw=1.0)
            if shuffled
            else dict(facecolor=COLOR[a], edgecolor=COLOR[a], lw=1.0)
        )
        for ax, col, scale, fmt in [(ax_top, m, 100, "{:.0f}"), (ax_bot, e, 1, "{:.2f}")]:
            mean = d[f"{col}_mean"].to_numpy() * scale
            lo = d[f"{col}_lo"].to_numpy() * scale
            hi = d[f"{col}_hi"].to_numpy() * scale
            ax.bar(xi, mean, W * 0.92, **style)
            ax.errorbar(xi, mean, yerr=[mean - lo, hi - mean], fmt="none",
                        ecolor="black", lw=0.9, capsize=2)
            for xx, v, h in zip(xi, mean, hi):
                ax.annotate(fmt.format(v), (xx, h), xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7.5)
        if not shuffled:
            ch = d["expected_mean"].to_numpy() * 100
            ax_top.hlines(ch, xi - W * 0.5, xi + W * 1.5, color=CHANCE_COLOR, lw=2.2, zorder=5)
    for ax in (ax_top, ax_bot):
        ax.set_xticks(x, ticklabels)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        for b in x[:-1] + 0.5:
            ax.axvline(b, color="0.85", lw=0.8)


# The legend gets its own grid row above the plots, so tight_layout cannot push it onto them.
fig = plt.figure(figsize=(16, 9.8))
gs = fig.add_gridspec(3, 2, height_ratios=[0.4, 4, 4], width_ratios=[5, 4])
leg_ax = fig.add_subplot(gs[0, :])
leg_ax.axis("off")
axes = np.empty((2, 2), dtype=object)
axes[0, 0] = fig.add_subplot(gs[1, 0])
axes[0, 1] = fig.add_subplot(gs[1, 1], sharey=axes[0, 0])
axes[1, 0] = fig.add_subplot(gs[2, 0])
axes[1, 1] = fig.add_subplot(gs[2, 1], sharey=axes[1, 0])
pf_n = pf_s.filter(pl.col("alphabet") == HP).sort("identity_bin")
sc_n = sc_s.filter(pl.col("alphabet") == HP).sort("category")
draw(axes[0, 0], axes[1, 0], pf_s, "identity_bin", hc.IDENTITY_LABELS,
     [f"{b}\n{n:,} pairs" for b, n in zip(pf_n["identity_bin"], pf_n["n"])])
draw(axes[0, 1], axes[1, 1], sc_s, "category", CAT_ORDER,
     [f"{CAT_LABEL[c]}\n{n:,} pairs" for c, n in zip(sc_n["category"], sc_n["n"])])
axes[0, 0].set_title("Pfam-A 38.2 seed pairs, by identity over aligned positions", fontsize=11)
axes[0, 1].set_title("SCOPe 2.08 40% set, structure-aligned pairs, by SCOPe relationship",
                     fontsize=11)
axes[0, 0].set_ylabel("aligned positions in the same class,\nper 100 positions")
axes[1, 0].set_ylabel("evidence per aligned position, I (bits)\nhigher = more evidence; 0 = chance")
axes[1, 0].set_xlabel("identity over aligned positions")
axes[1, 1].set_xlabel("how SCOPe relates the two domains (all identities)")
axes[0, 0].set_ylim(0, 105)

handles = [
    Patch(facecolor=COLOR[HP], edgecolor=COLOR[HP],
          label="hp_thomas_dill2 (H = ACFILMVWY, P = DEGHKNPQRST), real pairs"),
    Patch(facecolor="white", edgecolor=COLOR[HP], hatch="///",
          label="hp_thomas_dill2, target residues shuffled"),
    Line2D([], [], color=CHANCE_COLOR, lw=2.2,
           label="chance from each pair's own class shares (top row only)"),
    Patch(facecolor=COLOR[P20], edgecolor=COLOR[P20], label="protein20 (20 amino acids), real pairs"),
    Patch(facecolor="white", edgecolor=COLOR[P20], hatch="///",
          label="protein20, target residues shuffled"),
    Line2D([], [], color="black", lw=0.9, marker="|", ms=8,
           label="95% bootstrap interval of the mean over pairs"),
]
leg_ax.legend(handles=handles, loc="lower left", ncol=2, fontsize=9, frameon=False,
              borderaxespad=0, bbox_to_anchor=(0, -0.9))


def val(s, key, k, a, m):
    return s.filter((pl.col(key) == k) & (pl.col("alphabet") == a))[f"{m}_mean"][0]


hp_a = 100 * val(pf_s, "identity_bin", "20-30%", HP, "agree")
hp_c = 100 * val(pf_s, "identity_bin", "20-30%", HP, "expected")
p_a = 100 * val(pf_s, "identity_bin", "20-30%", P20, "agree")
p_c = 100 * val(pf_s, "identity_bin", "20-30%", P20, "expected")
hp_i = val(pf_s, "identity_bin", "20-30%", HP, "evidence")
p_i = val(pf_s, "identity_bin", "20-30%", P20, "evidence")
df_hp = 100 * (val(sc_s, "category", "diff_fold_same_class", HP, "agree")
               - val(sc_s, "category", "diff_fold_same_class", HP, "agree_null"))
sf_hp = 100 * (val(sc_s, "category", "same_superfamily_diff_family", HP, "agree")
               - val(sc_s, "category", "same_superfamily_diff_family", HP, "agree_null"))
hc.finish_figure(
    fig,
    FIG / "273_hp_vs_protein20_agreement_per_100_and_bits_pfam_seed_scope40.png",
    tools=hc.NO_TOOL,
    title="Aligned positions in the same class, counted per pair, real against shuffled",
    hypothesis=(
        "If the H/P pattern is conserved between homologs, real pairs agree at more positions "
        "than shuffled pairs, and the gap shrinks as identity falls and as the SCOPe relationship "
        "gets more distant. "
        "Higher bars and higher I are more conservation; the hatched bar is no signal."
    ),
    conclusion=(
        f"At 20-30% identity in Pfam, {hp_a:.0f} of 100 positions agree in H/P against "
        f"{hp_c:.0f} by chance, and {p_a:.0f} of 100 are identical against {p_c:.0f}, chance taken "
        f"from each pair's own shares (section 3: a random partner gives less). Each "
        f"aligned position carries {hp_i:.2f} bits in H/P and {p_i:.2f} bits as 20 amino acids, "
        f"so a 20-letter position is worth more evidence. In SCOPe, real H/P agreement exceeds "
        f"shuffled by {sf_hp:.0f} per 100 for same superfamily, different family, and by "
        f"{df_hp:.0f} per 100 for different folds, which SCOPe does not call related."
    ),
)
""")

md(r"""
## 3. Measured against the kappa-based estimates

The estimates were made from kappa with Swiss-Prot residue shares, at 20-30% identity. The
measured values are the Pfam 20-30% bin from section 2, and for comparison the SCOPe pairs
in the same superfamily but a different family, also at 20-30% identity.

For I there are two measured versions. "Mean of per-pair I" averages the per-pair values,
which is what section 2 plots. "I of the mean rates" puts the bin's mean Pr(agree) and mean
Pr(chance) into the formula once, which is how the estimate was made. They differ because I
curves upward: pairs far above chance add more than pairs near chance take away, and short
alignments add some I by sampling noise alone (the shuffled I).

The per-pair chance has one more catch. Two homologs at 20-30% identity share a quarter of
their residues, and those identical residues also make the two compositions alike. So
"chance from the pair's own shares" absorbs part of the signal it is compared against. To
see how much, chance is also computed with each query against the target of a different,
randomly chosen pair (Pfam only, triangles), and from the residue shares pooled over all
pairs in the bin.

Each quantity is drawn on the axis of its unit: per 100 positions on the left, bits on the
right. Black horizontal bars are the estimates. A measured point within the grey band
(estimate ± 3 per 100, or ± 0.03 bits) passes the decision rule.
""")

code(r"""
EST = [
    ("H/P agree", HP, "agree", 74),
    ("H/P chance", HP, "expected", 51),
    ("20 aa agree", P20, "agree", 25),
    ("20 aa chance", P20, "expected", 6),
]
EST_I = [("H/P I", HP, 0.15), ("20 aa I", P20, 0.26)]
TOL_100, TOL_BITS = 3, 0.03

sc_bin = summary(
    scope.filter(
        (pl.col("category") == "same_superfamily_diff_family")
        & (pl.col("identity_bin") == "20-30%")
    ),
    [],
)
pf_bin = pf_s.filter(pl.col("identity_bin") == "20-30%")


def g(s, a, m, part="mean"):
    return s.filter(pl.col("alphabet") == a)[f"{m}_{part}"][0]


rows = []
for lab, a, m, est in EST:
    for ds, s in [("Pfam 20-30%", pf_bin), ("SCOPe same superfamily, diff. family, 20-30%", sc_bin)]:
        v = 100 * g(s, a, m)
        rows.append(dict(quantity=lab, unit="per 100", dataset=ds, version="measured mean",
                         estimate=est, measured=round(v, 1),
                         lo=round(100 * g(s, a, m, "lo"), 1), hi=round(100 * g(s, a, m, "hi"), 1),
                         difference=round(v - est, 1), within_rule=bool(abs(v - est) <= TOL_100),
                         pairs=s.filter(pl.col("alphabet") == a)["n"][0]))
for lab, a, est in EST_I:
    for ds, s in [("Pfam 20-30%", pf_bin), ("SCOPe same superfamily, diff. family, 20-30%", sc_bin)]:
        n = s.filter(pl.col("alphabet") == a)["n"][0]
        for version, v, lo, hi in [
            ("mean of per-pair I", g(s, a, "evidence"), g(s, a, "evidence", "lo"), g(s, a, "evidence", "hi")),
            ("mean of per-pair I minus shuffled I", g(s, a, "evidence") - g(s, a, "evidence_null"), None, None),
            ("I of the mean rates", hc.bernoulli_kl_bits(g(s, a, "agree"), g(s, a, "expected")), None, None),
        ]:
            rows.append(dict(quantity=lab, unit="bits", dataset=ds, version=version, estimate=est,
                             measured=round(float(v), 3),
                             lo=None if lo is None else round(lo, 3),
                             hi=None if hi is None else round(hi, 3),
                             difference=round(float(v) - est, 3),
                             within_rule=bool(abs(float(v) - est) <= TOL_BITS), pairs=n))
cmp = pl.DataFrame(rows)
print(cmp)

# Chance with a random partner: each query's aligned class shares against the aligned class
# shares of a different pair's target (a derangement, so no pair meets its own target). Also
# chance from shares pooled over every aligned residue in the bin.
keys_bin = pfam.filter((pl.col("alphabet") == HP) & (pl.col("identity_bin") == "20-30%")).select(
    "family", "query", "target"
)
aln_bin = pl.read_parquet(PFAM_ALN).join(keys_bin, on=["family", "query", "target"], how="semi")


def aligned_class_shares(qaln, taln, alphabet):
    # Class shares of query and target over the columns where both have a residue.
    q = np.frombuffer(qaln.encode(), np.uint8)
    t = np.frombuffer(taln.encode(), np.uint8)
    both = (hc.TABLES[P20][q] != hc.GAP) & (hc.TABLES[P20][t] != hc.GAP)
    tab, k = hc.TABLES[alphabet], hc.SIZES[alphabet]
    n = both.sum()
    return np.bincount(tab[q][both], minlength=k)[:k] / n, np.bincount(tab[t][both], minlength=k)[:k] / n


rng_partner = np.random.default_rng(SEED)
order = rng_partner.permutation(aln_bin.height)
partner = np.empty(aln_bin.height, dtype=int)
partner[order] = np.roll(order, 1)
alt_rows = []
for lab, a, m, est in EST:
    if m != "expected":
        continue
    sh = [aligned_class_shares(qa, ta, a) for qa, ta in zip(aln_bin["qaln"], aln_bin["taln"])]
    fq = np.array([x[0] for x in sh])
    ft = np.array([x[1] for x in sh])
    own = (fq * ft).sum(axis=1).mean()
    rand = (fq * ft[partner]).sum(axis=1).mean()
    pooled = ((fq.mean(axis=0) + ft.mean(axis=0)) / 2) ** 2
    print(
        f"{a}: chance per 100 at Pfam 20-30%, {aln_bin.height:_} pairs: own partner {100 * own:.2f}; "
        f"random other pair's target {100 * rand:.2f}; pooled shares {100 * pooled.sum():.2f}; estimate {est}"
    )
    alt_rows.append(dict(quantity=lab, unit="per 100", dataset="Pfam 20-30%",
                         version="chance with a random partner", estimate=est,
                         measured=round(100 * rand, 1), lo=None, hi=None,
                         difference=round(100 * rand - est, 1),
                         within_rule=bool(abs(100 * rand - est) <= TOL_100), pairs=aln_bin.height))
cmp = pl.concat([cmp, pl.DataFrame(alt_rows)], how="vertical_relaxed")

# x offset of each point inside its group; the I-minus-shuffled version is in the table only.
OFFSET = {
    ("Pfam 20-30%", "measured mean"): -0.22,
    ("Pfam 20-30%", "chance with a random partner"): 0.0,
    ("SCOPe same superfamily, diff. family, 20-30%", "measured mean"): 0.22,
    ("Pfam 20-30%", "mean of per-pair I"): -0.27,
    ("Pfam 20-30%", "I of the mean rates"): -0.09,
    ("SCOPe same superfamily, diff. family, 20-30%", "mean of per-pair I"): 0.09,
    ("SCOPe same superfamily, diff. family, 20-30%", "I of the mean rates"): 0.27,
}
DS_MARK = {"Pfam 20-30%": "o", "SCOPe same superfamily, diff. family, 20-30%": "s"}
RANDOM_MARK = "^"
fig = plt.figure(figsize=(14, 6.0))
gs = fig.add_gridspec(2, 2, height_ratios=[0.75, 4], width_ratios=[4, 2.4])
leg_ax = fig.add_subplot(gs[0, :])
leg_ax.axis("off")
axl, axr = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
for ax, unit, labels, tol in [
    (axl, "per 100", [e[0] for e in EST], TOL_100),
    (axr, "bits", [e[0] for e in EST_I], TOL_BITS),
]:
    sub = cmp.filter(pl.col("unit") == unit)
    for i, lab in enumerate(labels):
        r0 = sub.filter(pl.col("quantity") == lab)
        est = r0["estimate"][0]
        a = HP if lab.startswith("H/P") else P20
        ax.fill_between([i - 0.42, i + 0.42], est - tol, est + tol, color="0.88", lw=0)
        ax.hlines(est, i - 0.42, i + 0.42, color="black", lw=2.5)
        for r in r0.iter_rows(named=True):
            key = (r["dataset"], r["version"])
            if key not in OFFSET:
                continue
            xx = i + OFFSET[key]
            hollow = r["version"] == "I of the mean rates"
            mark = RANDOM_MARK if r["version"] == "chance with a random partner" else DS_MARK[r["dataset"]]
            ax.plot(xx, r["measured"], mark, ms=8, mec=COLOR[a],
                    mfc="white" if hollow else COLOR[a], zorder=4)
            if r["lo"] is not None:
                ax.vlines(xx, r["lo"], r["hi"], color=COLOR[a], lw=1)
            ax.annotate(f"{r['measured']:.2f}" if unit == "bits" else f"{r['measured']:.1f}",
                        (xx, r["measured"]), textcoords="offset points",
                        # label on the side away from the estimate line, so it never sits on it
                        xytext=(0, 7) if r["measured"] >= est else (0, -7),
                        ha="center", va="bottom" if r["measured"] >= est else "top",
                        fontsize=7.5)
    ests = [sub.filter(pl.col("quantity") == lab)["estimate"][0] for lab in labels]
    ax.set_xticks(range(len(labels)), [f"{lab}\nestimate {e:g}" for lab, e in zip(labels, ests)])
    ax.set_xlim(-0.55, len(labels) - 0.45)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
axl.set_ylabel("aligned positions per 100")
axl.set_ylim(0, 85)
axr.set_ylabel("evidence per aligned position, I (bits)")
axr.set_ylim(0.10, 0.31)
axr.set_title("y axis starts at 0.10 bits", fontsize=9)
def pair_of_marks(fill):
    return tuple(
        Line2D([], [], ls="none", marker=mk, mfc=fill, mec="0.3", ms=8) for mk in ("o", "s")
    )


handles = [
    Line2D([], [], color="black", lw=2.5),
    Patch(color="0.88"),
    Line2D([], [], color=COLOR[HP], lw=7),
    Line2D([], [], color=COLOR[P20], lw=7),
    Line2D([], [], ls="none", marker="o", mfc="0.6", mec="0.3", ms=8),
    Line2D([], [], ls="none", marker="s", mfc="0.6", mec="0.3", ms=8),
    pair_of_marks("0.3"),
    pair_of_marks("white"),
    Line2D([], [], ls="none", marker=RANDOM_MARK, mfc="0.3", mec="0.3", ms=8),
]
labels_leg = [
    "estimate from kappa and Swiss-Prot shares",
    "decision-rule band: estimate ± 3 per 100, or ± 0.03 bits",
    "hp_thomas_dill2 (H/P)",
    "protein20 (20 aa)",
    "circle: Pfam seed, 20-30% identity",
    "square: SCOPe same superfamily, different family, 20-30% identity",
    "filled: mean over pairs (bits: mean of per-pair I); vertical line = 95% interval",
    "hollow, bits only: I from the mean Pr(agree) and mean Pr(chance)",
    "triangle: chance with each query against a random other pair's target (Pfam)",
]
from matplotlib.legend_handler import HandlerTuple

leg_ax.legend(handles, labels_leg, loc="lower left", ncol=2, fontsize=8.5, frameon=False,
              borderaxespad=0, bbox_to_anchor=(0, -0.55),
              handler_map={tuple: HandlerTuple(ndivide=None)})

fails = cmp.filter(~pl.col("within_rule") & pl.col("dataset").str.starts_with("Pfam")
                   & pl.col("version").is_in(["measured mean", "mean of per-pair I"]))
fail_txt = "; ".join(
    f"{r['quantity']} {r['measured']} vs {r['estimate']}" for r in fails.iter_rows(named=True)
) or "none"
hc.finish_figure(
    fig,
    FIG / "273_measured_vs_kappa_estimates_pfam_seed_scope40_20_30pct.png",
    tools=hc.NO_TOOL,
    title="Measured agreement and evidence at 20-30% identity, against the kappa-based estimates",
    hypothesis=(
        "Converting kappa back to positions per 100 with Swiss-Prot shares gives the same "
        "numbers as counting them per pair. A point inside the grey band passes; closer to the "
        "black line is better agreement with the estimate."
    ),
    conclusion=(
        f"Pfam values outside the band: {fail_txt}. In both alphabets, chance from each pair's "
        "own shares is 0.7 per 100 above chance with a random partner (circle vs triangle); the "
        "random-partner value is the closer match to the estimate."
    ),
)
""")

md(r"""
## 4. Summary and conclusions

**Question.** Do the kappa-based numbers for H/P agreement (74 vs 51 per 100 at 20-30%
identity; 25 vs 6 for 20 amino acids; I about 0.15 vs 0.26 bits) hold when agreement is
counted per pair and chance is measured by shuffling?

**Answer.** The four per-100 numbers hold. The two I values pass the decision rule but cannot
be rebuilt from those four numbers; quote the measured I instead.

1. At 20-30% identity in Pfam seed pairs (37,085 pairs), 73.9 of 100 aligned positions share an
   H/P class (95% interval 73.8-73.9; estimate 74). Chance from each pair's own class shares is
   51.4 per 100, and shuffling the target gives 51.4 (estimate 51).
2. For the 20 amino acids at the same identity, 25.2 of 100 aligned positions are identical
   (estimate 25). Chance from each pair's own shares is 6.7 per 100. With each query set against
   a random other pair's target, chance is 6.0, the estimate. A random partner removes whatever
   makes two homologs' compositions alike, including the residues they share, so the per-pair
   chance absorbs some of the signal it is compared against. This notebook does not split the
   0.7 into its causes. For H/P the random-partner chance is 50.7
   per 100, also 0.7 below the pair's own, and still rounds to the estimate of 51.
3. Evidence per aligned position at 20-30% identity, mean over pairs: H/P 0.162 bits, 20 amino
   acids 0.251 bits. Shuffled pairs score 0.006 bits in both alphabets. The estimates (0.15,
   0.26) are within 0.03 bits, but the formula applied to the estimated rates gives 0.160 for
   74 vs 51 and 0.270 for 25 vs 6, so the estimates do not follow from their own inputs.
4. In SCOPe pairs from the same superfamily but a different family at 20-30% identity (2,118
   pairs), H/P agreement is 72.1 per 100 and I is 0.144 bits (20 amino acids: 24.4 per 100,
   0.239 bits), also within the band.
5. A result the kappa summary did not show: in Pfam pairs under 20% identity, H/P carries more
   evidence per aligned position than the 20 amino acids (0.103 vs 0.091 bits). From 20%
   identity up, the 20 amino acids carry more: 1.5 times as much at 20-30% and 3.5 times at 60%
   and above. In SCOPe, different-fold pairs, which SCOPe does not call related, still agree on
   56.2 H/P positions per 100 against 50.8 shuffled (186 pairs). They were picked for TM-score
   0.5 or more, so this notebook cannot say whether that excess comes from the structural
   superposition, shared packing, or a distant common ancestor.

**Limits.**
* Aligned positions only. Whether an alignment can be found at all is notebook 230 sections 4-6.
* The SCOPe categories pool all identities; the right column of section 2 mixes relatedness
  with identity.
* The decision rule is loose for small numbers: ± 3 per 100 around a chance of 6 lets 3 to 9
  pass, and ± 0.03 bits is a fifth of 0.15. Read the measured values, not only the pass/fail.
* Identity bins are notebook 230's (`pl.cut`, closed on the right): a pair at exactly 20% counts
  in "<20%", one at exactly 30% in "20-30%".
* Pfam seed pairs are capped at 6 per family, so a large family counts no more than a small one
  with 4 or more seed members.

**Decision.** Quote 74 vs 51 aligned positions per 100 in H/P and 25 vs 6 for the 20 amino
acids, at 20-30% identity in Pfam seed pairs, with chance from a random partner's composition.
For evidence per aligned position, quote the measured 0.16 bits (H/P) and 0.25 bits (20 amino
acids), citing this notebook.
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "2025-kmerseek-analysis", "language": "python", "name": "2025-kmerseek-analysis"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = Path(__file__).resolve().parents[1] / "notebooks" / "273_hp_agreement_per_100_positions_pfam_seed_scope40.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({len(cells)} cells)")
