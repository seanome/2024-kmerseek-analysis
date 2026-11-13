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

**What this decides.** Two choices for the paper and the SAB material.

1. Which numbers to quote for "aligned homologs keep their H/P class": the estimates from
   kappa, or the counts in this notebook. The decision rule above settles this one.
2. Whether the paper can argue that an H/P alphabet keeps more evidence of relatedness per
   aligned position than the 20 amino acids. If H/P has the higher I at the identities where
   homologs are found, that sentence can motivate the H/P alphabet. If not, the case for H/P
   has to rest on other results, and the sentence stays out. This second question was added
   on 2026-10-10, after the run, in answer to review; it is not a test written beforehand.
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
SCOPE_FA = Path("/Users/olga/data/scope/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa")
SCOPE_SPLITS = Path("/Users/olga/data/scope/273_scope40_two_class_split_evidence.parquet")
PFAM_SPLITS = Path("/Users/olga/data/pfam/273_pfam_seed_two_class_split_evidence.parquet")
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
    f"Before filters: Pfam {pfam_raw.filter(pl.col('alphabet') == HP).height:_} pairs from "
    f"{pfam_raw['family'].n_unique():_} families; SCOPe {scope_raw.filter(pl.col('alphabet') == HP).height:_} pairs"
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


L = hc.alignment_lines(q, t, HP)
qh, th, both = L["query_class"], L["target_class"], L["aligned"]
id_line, hp_line = L["identical"], L["same_class"]
rng = np.random.default_rng(SEED)
t_list = list(t)
res_pos = [i for i, c in enumerate(t) if hp_letter(c) != "-"]
perm = rng.permutation(res_pos)
t_shuf = t_list.copy()
for i, j in zip(res_pos, perm):
    t_shuf[i] = t[j]
tsh = "".join(hp_letter(c) for c in t_shuf)
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
hc.print_alignment(
    [
        ("query", q),
        ("identical", id_line),
        ("target", t),
        ("query H/P", qh),
        ("same H/P class", hp_line),
        ("target H/P", th),
        ("shuffled target", tsh),
        ("same H/P class", sh_line),
    ]
)

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
hc.draw_alignment_rows(
    ax, [(name, kind, line) for name, kind, line, _ in rows], both, H_COLOR, P_COLOR, AGREE_COLOR
)
gap = [x for x, ok in zip(cols, both) if not ok]
ax.set_xlabel("alignment column (gaps included)")
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
        f"from each pair's own shares (section 6: a random partner gives less). Each "
        f"aligned position carries {hp_i:.2f} bits in H/P and {p_i:.2f} bits as 20 amino acids, "
        f"so a 20-letter position is worth more evidence. In SCOPe, real H/P agreement exceeds "
        f"shuffled by {sf_hp:.0f} per 100 for same superfamily, different family, and by "
        f"{df_hp:.0f} per 100 for different folds, which SCOPe does not call related."
    ),
)
""")

md(r"""
## 3. Under 20% identity, is the H/P lead a property of distant homologs, or of how the pairs were picked?

Section 2 shows H/P giving more evidence per aligned position than the 20 amino acids in Pfam
pairs under 20% identity, and less at every higher identity. But identity here is the share of
aligned positions with the same residue, which is the 20 amino acid agreement itself. So the
"<20%" bin is the pairs where the 20 amino acids agree least, and their I is held down by the
choice of bin. Three checks:

1. How close identity is to the 20 amino acid agreement, pair by pair.
2. Whether the Pfam lead under 20% survives resampling whole families (pairs from one family
   share sequence, so they are not independent draws). 2,000 resamples of families.
3. The same comparison on a grouping that does not use identity: SCOPe's relationship between
   the two domains. If H/P leads for homologs at all identities, the lead belongs to distant
   homologs. If it leads only after keeping the pairs under 20% identity, it comes from picking
   pairs by low identity.

The figure shows mean I per SCOPe category, both alphabets, for all identities (left) and for
the pairs under 20% identity only (right). Higher is more evidence.
""")

code(r"""
pw = pfam.pivot(
    on="alphabet",
    index=["family", "query", "target", "seqid_ali", "identity_bin"],
    values=["agree", "evidence"],
)
same = ((pw["seqid_ali"] - pw[f"agree_{P20}"]).abs() < 1e-9).mean()
corr = np.corrcoef(pw["seqid_ali"].to_numpy(), pw[f"agree_{P20}"].to_numpy())[0, 1]
print(
    f"1. Pfam, {pw.height:_} pairs: identity equals the 20 amino acid agreement exactly for "
    f"{100 * same:.1f}% of pairs; correlation {corr:.5f}"
)

lo = pw.filter(pl.col("identity_bin") == "<20%").with_columns(
    (pl.col(f"evidence_{HP}") - pl.col(f"evidence_{P20}")).alias("d")
)
fam = lo.group_by("family").agg(pl.col("d").sum(), pl.len().alias("n")).sort("family")
d_sum, d_n = fam["d"].to_numpy(), fam["n"].to_numpy()
rng = np.random.default_rng(SEED)
boot = []
for _ in range(2000):
    i = rng.integers(0, fam.height, fam.height)
    boot.append(d_sum[i].sum() / d_n[i].sum())
print(
    f"2. Pfam under 20% identity, {lo.height:_} pairs from {fam.height:_} families: "
    f"I(H/P) - I(20 aa) = {lo['d'].mean():.4f} bits, 95% interval resampling families "
    f"{np.percentile(boot, 2.5):.4f} to {np.percentile(boot, 97.5):.4f}; "
    f"H/P higher in {100 * (lo['d'] > 0).mean():.1f}% of pairs"
)

sw = scope.pivot(on="alphabet", index=["query", "target", "category", "identity_bin"], values="evidence")


def by_category(df, label):
    return df.group_by("category").agg(
        pl.lit(label).alias("pairs used"),
        pl.len().alias("pairs"),
        pl.col(HP).mean().alias("I H/P"),
        pl.col(P20).mean().alias("I 20 aa"),
    )


sc_tab = (
    pl.concat(
        [
            by_category(sw, "all identities"),
            by_category(sw.filter(pl.col("identity_bin") == "<20%"), "under 20% identity"),
        ]
    )
    .with_columns(
        pl.col("category").cast(pl.Enum(CAT_ORDER)),
        # the ratio uses the unrounded means; the means are rounded for display only
        (pl.col("I H/P") / pl.col("I 20 aa")).round(2).alias("H/P / 20 aa"),
        pl.col("I H/P").round(3),
        pl.col("I 20 aa").round(3),
    )
    .sort("pairs used", "category")
)
print("3. SCOPe 2.08 40% set, mean I per aligned position (bits) by SCOPe relationship")
print(sc_tab)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
x = np.arange(len(CAT_ORDER))
for ax, used in zip(axes, ["all identities", "under 20% identity"]):
    t = sc_tab.filter(pl.col("pairs used") == used).sort("category")
    for j, (a, col) in enumerate([(HP, "I H/P"), (P20, "I 20 aa")]):
        ax.bar(x + (j - 0.5) * 0.38, t[col].to_numpy(), width=0.38, color=COLOR[a])
    ax.set_xticks(
        x, [f"{CAT_LABEL[c]}\n{n:_} pairs" for c, n in zip(CAT_ORDER, t["pairs"].to_list())], fontsize=9
    )
    ax.set_title(f"SCOPe pairs, {used}", loc="left", fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].set_ylabel("mean I per aligned position (bits)")
axes[0].legend(
    handles=[Patch(color=COLOR[HP], label=f"H/P ({HP})"), Patch(color=COLOR[P20], label="20 amino acids")],
    loc="lower left", bbox_to_anchor=(0, 1.08), ncol=2, frameon=False, fontsize=9, borderaxespad=0,
)
ssf = sc_tab.filter(pl.col("category") == "same_superfamily_diff_family")
ssf_all = ssf.filter(pl.col("pairs used") == "all identities").row(0, named=True)
ssf_lo = ssf.filter(pl.col("pairs used") == "under 20% identity").row(0, named=True)
hc.finish_figure(
    fig,
    FIG / "273_scope_relationship_hp_vs_protein20_bits_all_vs_under_20pct.png",
    tools=f"no kmerseek; SCOPe 2.08 40% set, USalign alignments, {HP} and protein20",
    title="Evidence per aligned position by SCOPe relationship: all identities vs under 20% identity",
    hypothesis=(
        "If H/P keeps more evidence for distant homologs, its bar is taller for homologs "
        "(same family, same superfamily) at all identities, not only under 20%."
    ),
    conclusion=(
        f"Same superfamily, different family: 20 amino acids {ssf_all['I 20 aa']} vs H/P "
        f"{ssf_all['I H/P']} bits over all identities; under 20% identity, H/P {ssf_lo['I H/P']} "
        f"vs {ssf_lo['I 20 aa']}."
    ),
)
""")

md(r"""
## 4. Two checks on the H/P lead: grouping on half the positions, and every split into two classes

Section 3 found that the H/P lead under 20% identity may come from how the pairs were picked:
identity is the 20 amino acid agreement, so a low-identity bin holds the pairs where the 20
amino acids happened to agree least. This section tests that directly, then asks whether
ACFILMVWY is a special choice of 9 amino acids or one of many that would do as well. Both
checks were added on 2026-10-12, after the section 2 result had been seen, so neither is a test
written beforehand.

**4a. Group on half of each pair's positions, measure on the other half.** Each pair's aligned
positions are numbered in order and split into odd and even. The pair goes into an identity
bin by the identity of its even positions only. I for both alphabets is then computed on its
odd positions only, with chance from the odd positions' own class shares. The same is done
with odd and even swapped, and both directions are pooled. A pair that lands in a low bin
because its even positions agreed less than usual by chance is measured on positions that
did not have that bad luck.

Hypothesis: if H/P keeps more evidence than the 20 amino acids for distant homologs, the ratio
I(H/P) / I(20 aa) stays above 1 in the lowest bins when grouped this way. If the lead came
from picking, it drops below 1. Two more lines are printed for comparison. The first is the
ratio when pairs are grouped and measured on the same (whole-pair) positions, as in sections 2
and 3. The second is the reverse pick: the 11% of pairs with the lowest H/P agreement, where
the same effect should favour the 20 amino acids. The same even/odd test is also run on the SCOPe pairs at 20%
identity or less within each SCOPe relationship, because those pairs are aligned by structure
rather than from sequence.

**4b. Every split of the 20 amino acids into 9 and 11.** There are 167,960 ways to choose 9 of
the 20 amino acids. For each, mean I over the pairs in a group is computed the same way as for
H/P, with agreement and chance from each pair's own class shares. If ACFILMVWY keeps more evidence
because of what hydrophobicity does in a folded protein, it ranks at or near the top. If any
9 letters would do, it sits in the middle. Groups: the four SCOPe relationships (all pairs,
aligned by structure), and three random samples of 5,000 Pfam seed pairs (aligned from
sequence, not structure): 20% identity or less, 20-30%, and all identities. The results are
cached to `~/data/scope/273_scope40_two_class_split_evidence.parquet` and
`~/data/pfam/273_pfam_seed_two_class_split_evidence.parquet`.

The figure puts both checks on one scale, the ratio I(H/P) / I(20 aa) on a log axis, where 1
means equal evidence per aligned position. Left: SCOPe relationship, all pairs measured whole
(circles), the pairs at 20% identity or less grouped on one half and measured on the other
(diamonds), and the middle 95% of all 167,960 splits for comparison. Right: Pfam identity bins,
grouped and measured on the same positions (open circles) against grouped on one half and
measured on the other (diamonds).
""")

code(r"""
# ---- 4a. even/odd positions, Pfam ----
pf_aln = pl.read_parquet(PFAM_ALN)
pq, pt, pidx, ppos, pkept, pcol = hc.aligned_codes(pf_aln, MIN_COLS, return_columns=True)
pf_pairs = pf_aln[pkept].select("family", "query", "target", "seqid_ali")
n_pf = pf_pairs.height

# whole pairs: must equal the cached per-pair table used in sections 2-3
_, a20_w, _, I20_w = hc.grouped_evidence(pq, pt, pidx, n_pf, P20)
_, ahp_w, _, Ihp_w = hc.grouped_evidence(pq, pt, pidx, n_pf, HP)
cached = pfam.pivot(on="alphabet", index=["family", "query", "target"], values="evidence")
chk = pf_pairs.with_columns(hp_new=pl.Series(Ihp_w), p20_new=pl.Series(I20_w)).join(
    cached, on=["family", "query", "target"]
)
print(
    f"Pfam pairs here {n_pf:_}, matched to the section 2 table {chk.height:_}; max |I difference| "
    f"H/P {(chk['hp_new'] - chk[HP]).abs().max():.1e}, 20 aa {(chk['p20_new'] - chk[P20]).abs().max():.1e}"
)

g_half = pidx * 2 + ppos % 2
_, a20_h, _, I20_h = hc.grouped_evidence(pq, pt, g_half, 2 * n_pf, P20)
_, _, _, Ihp_h = hc.grouped_evidence(pq, pt, g_half, 2 * n_pf, HP)
a20_h, I20_h, Ihp_h = a20_h.reshape(n_pf, 2), I20_h.reshape(n_pf, 2), Ihp_h.reshape(n_pf, 2)
fam_codes = pf_pairs["family"].to_numpy()

EDGES = [0.0, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30, 0.40, 0.60, 1.0]


def in_bin(x, lo, hi):
    return (x <= hi) if lo == 0 else ((x > lo) & (x <= hi))


def other_half(lo, hi):
    # group on one half, measure on the other, both directions pooled
    hp, aa, fam = [], [], []
    for grp, meas in ((0, 1), (1, 0)):
        m = in_bin(a20_h[:, grp], lo, hi) & ~np.isnan(Ihp_h[:, meas]) & ~np.isnan(I20_h[:, meas])
        hp.append(Ihp_h[m, meas]); aa.append(I20_h[m, meas]); fam.append(fam_codes[m])
    return np.concatenate(hp), np.concatenate(aa), np.concatenate(fam)


rows = []
for lo, hi in zip(EDGES[:-1], EDGES[1:]):
    hp, aa, _ = other_half(lo, hi)
    w = in_bin(a20_w, lo, hi)
    rows.append(dict(
        identity=f"{lo:.1%}-{hi:.1%}" if lo else f"<= {hi:.0%}",
        lo=lo, hi=hi,
        pair_halves=hp.size,
        I_hp_other_half=hp.mean(), I_20aa_other_half=aa.mean(),
        ratio_other_half=hp.mean() / aa.mean(),
        whole_pairs=int(w.sum()),
        ratio_same_positions=np.nanmean(Ihp_w[w]) / np.nanmean(I20_w[w]),
    ))
split_tab = pl.DataFrame(rows)
print("\nPfam seed pairs, I(H/P) / I(20 aa) by identity bin")
print(split_tab.drop("lo", "hi").with_columns(pl.col(pl.Float64).round(3)))

hp, aa, fam = other_half(0.0, 0.20)
d_by_fam = pl.DataFrame({"family": fam, "d": hp - aa}).group_by("family").agg(pl.col("d").sum(), pl.len().alias("n"))
d_sum, d_n = d_by_fam["d"].to_numpy(), d_by_fam["n"].to_numpy()
rng = np.random.default_rng(SEED)
boot = []
for _ in range(2000):
    i = rng.integers(0, d_by_fam.height, d_by_fam.height)
    boot.append(d_sum[i].sum() / d_n[i].sum())
pooled_other = hp.mean() / aa.mean()
w20 = a20_w <= 0.20
pooled_same = np.nanmean(Ihp_w[w20]) / np.nanmean(I20_w[w20])
print(
    f"\n20% identity or less, pooled: grouped and measured on the same positions {pooled_same:.2f}; "
    f"grouped on one half, measured on the other {pooled_other:.2f} "
    f"(I H/P {hp.mean():.3f} vs I 20 aa {aa.mean():.3f} bits, {hp.size:_} pair-halves from "
    f"{d_by_fam.height:_} families; I(H/P) - I(20 aa) 95% interval resampling families "
    f"{np.percentile(boot, 2.5):.4f} to {np.percentile(boot, 97.5):.4f} bits)"
)
low_hp = ahp_w <= np.quantile(ahp_w, 0.11)
print(
    f"Reverse pick: the {low_hp.sum():_} pairs with the lowest 11% of H/P agreement give "
    f"I(H/P) / I(20 aa) = {np.nanmean(Ihp_w[low_hp]) / np.nanmean(I20_w[low_hp]):.2f}"
)
""")

code(r"""
# ---- 4b. every split into 9 + 11 amino acids ----
SPLITS = hc.two_class_splits(9)
HP_ROW = int(np.flatnonzero((SPLITS == (hc.class_index(HP) == 0)).all(1))[0])
SPLIT_NAMES = [hc.split_name(s) for s in SPLITS]

sc_aln = pl.read_parquet(SCOPE_ALN).filter(pl.max_horizontal("tm_q", "tm_t") >= MIN_TM)
sq, st_, sidx, spos, skept = hc.aligned_codes(sc_aln, MIN_COLS)
sc_pairs = sc_aln[skept].select("query", "target", "category")
n_sc = sc_pairs.height
_, _, _, I20_sc = hc.grouped_evidence(sq, st_, sidx, n_sc, P20)
_, _, _, Ihp_sc = hc.grouped_evidence(sq, st_, sidx, n_sc, HP)
sc_cat = sc_pairs["category"].to_numpy()
sc_ident = sc_aln[skept]["seqid_ali"].to_numpy()

if SCOPE_SPLITS.exists():
    sc_split = pl.read_parquet(SCOPE_SPLITS)
else:
    J = hc.pair_joint_counts(sq, st_, sidx, n_sc)
    sc_split = pl.DataFrame({"split": SPLIT_NAMES} | {c: hc.mean_split_evidence(J[sc_cat == c], SPLITS) for c in CAT_ORDER})
    sc_split.write_parquet(SCOPE_SPLITS)

if PFAM_SPLITS.exists():
    pf_split = pl.read_parquet(PFAM_SPLITS)
else:
    rng = np.random.default_rng(SEED)
    ident = pf_pairs["seqid_ali"].to_numpy()
    groups = {"<=20%": ident <= 0.2, "20-30%": (ident > 0.2) & (ident <= 0.3), "all identities": np.ones(n_pf, bool)}
    cols = {}
    for name, m in groups.items():
        pick = np.sort(rng.choice(np.flatnonzero(m), 5_000, replace=False))
        sel = np.isin(pidx, pick)
        remap = np.full(n_pf, -1); remap[pick] = np.arange(pick.size)
        J = hc.pair_joint_counts(pq[sel], pt[sel], remap[pidx[sel]], pick.size)
        cols[name] = hc.mean_split_evidence(J, SPLITS)
    pf_split = pl.DataFrame({"split": SPLIT_NAMES} | cols)
    pf_split.write_parquet(PFAM_SPLITS)


def rank_table(tab, groups):
    out = []
    for g in groups:
        v = tab[g].to_numpy()
        top = np.argsort(-v)[:3]
        out.append(dict(
            group=g,
            rank_of_ACFILMVWY=int((v > v[HP_ROW]).sum()) + 1,
            I_ACFILMVWY=v[HP_ROW],
            median_split=float(np.median(v)),
            best_three=", ".join(f"{SPLIT_NAMES[j]} {v[j]:.4f}" for j in top),
        ))
    return pl.DataFrame(out)


print(f"Mean I (bits per aligned position) over all {len(SPLITS):_} splits into 9 + 11 amino acids")
with pl.Config(fmt_str_lengths=100, tbl_width_chars=260):
    print("SCOPe 2.08 40% set, structure-aligned, by relationship")
    print(rank_table(sc_split, CAT_ORDER).with_columns(pl.col(pl.Float64).round(4)))
    print("Pfam-A 38.2 seed, 5,000 sampled pairs per group, aligned from sequence")
    print(rank_table(pf_split, ["<=20%", "20-30%", "all identities"]).with_columns(pl.col(pl.Float64).round(4)))

# ratio to the 20 amino acids per SCOPe relationship, 95% interval resampling the query's fold
fold_of = {}
for line in open(SCOPE_FA):
    if line.startswith(">"):
        f = line[1:].split()
        fold_of[f[0]] = ".".join(f[1].split(".")[:2])
sc_fold = np.array([fold_of[q] for q in sc_pairs["query"]])
rng = np.random.default_rng(SEED)
rel = []
for i, c in enumerate(CAT_ORDER):
    m = sc_cat == c
    u, inv = np.unique(sc_fold[m], return_inverse=True)
    s_hp, s_20 = np.bincount(inv, weights=Ihp_sc[m]), np.bincount(inv, weights=I20_sc[m])
    boot = []
    for _ in range(2000):
        w = np.bincount(rng.integers(0, u.size, u.size), minlength=u.size)
        boot.append((w @ s_hp) / (w @ s_20))
    splits_ratio = sc_split[c].to_numpy() / I20_sc[m].mean()
    rel.append(dict(
        category=c, pairs=int(m.sum()), folds=u.size,
        median_identity=float(np.median(sc_ident[m])),
        ratio=Ihp_sc[m].mean() / I20_sc[m].mean(),
        lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5),
        splits_lo=np.percentile(splits_ratio, 2.5), splits_median=np.median(splits_ratio),
        splits_hi=np.percentile(splits_ratio, 97.5),
        splits_at_or_above=int((splits_ratio >= splits_ratio[HP_ROW] - 1e-12).sum()),
    ))
rel = pl.DataFrame(rel)

# the even/odd test of 4a on SCOPe pairs at 20% identity or less, within each relationship
g_sc = sidx * 2 + spos % 2
_, a20_sh, _, I20_sh = hc.grouped_evidence(sq, st_, g_sc, 2 * n_sc, P20)
_, _, _, Ihp_sh = hc.grouped_evidence(sq, st_, g_sc, 2 * n_sc, HP)
a20_sh, I20_sh, Ihp_sh = a20_sh.reshape(n_sc, 2), I20_sh.reshape(n_sc, 2), Ihp_sh.reshape(n_sc, 2)
low = []
for c in CAT_ORDER:
    hp, aa, fo = [], [], []
    for grp, meas in ((0, 1), (1, 0)):
        m = (sc_cat == c) & (a20_sh[:, grp] <= 0.20)
        hp.append(Ihp_sh[m, meas]); aa.append(I20_sh[m, meas]); fo.append(sc_fold[m])
    hp, aa, fo = map(np.concatenate, (hp, aa, fo))
    u, inv = np.unique(fo, return_inverse=True)
    s_hp, s_20 = np.bincount(inv, weights=hp), np.bincount(inv, weights=aa)
    boot = []
    for _ in range(2000):
        w = np.bincount(rng.integers(0, u.size, u.size), minlength=u.size)
        boot.append((w @ s_hp) / (w @ s_20))
    low.append(dict(category=c, low_pair_halves=hp.size, low_folds=u.size, low_ratio=hp.mean() / aa.mean(),
                    low_lo=np.percentile(boot, 2.5), low_hi=np.percentile(boot, 97.5)))
rel = rel.join(pl.DataFrame(low), on="category", maintain_order="left")
low_cols = [c for c in rel.columns if c.startswith("low")]
print("\nSCOPe: I(H/P) / I(20 aa) with a 95% interval resampling the query's SCOPe fold, "
      "and the same ratio for all splits (middle 95%)")
print(rel.drop(low_cols).with_columns(pl.col(pl.Float64).round(3)))
print("\nSCOPe pairs at 20% identity or less, grouped on one half of the positions and measured on the other "
      "(both directions pooled), 95% interval resampling the query's fold")
print(rel.select("category", *low_cols).with_columns(pl.col(pl.Float64).round(3)))
""")

code(r"""
# ---- figure: both checks on one scale ----
SPLIT_GREY = "#BBBBBB"
fig, (axa, axb) = plt.subplots(1, 2, figsize=(18, 6.6), gridspec_kw=dict(width_ratios=[1.05, 1]))
for ax in (axa, axb):
    ax.set_yscale("log")
    ax.set_ylim(0.07, 9)
    ax.set_yticks([0.1, 0.2, 0.5, 1, 2, 5], ["0.1×", "0.2×", "0.5×", "1×", "2×", "5×"])
    ax.axhline(1, color="black", lw=1, ls=(0, (4, 3)), zorder=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#EEEEEE", zorder=0)
    ax.set_ylabel("evidence per aligned position,\nH/P ÷ 20 amino acids (log scale)", fontsize=11)

BENCH = {"same_family": "homolog", "same_superfamily_diff_family": "homolog",
         "same_fold_diff_superfamily": "not scored", "diff_fold_same_class": "not related"}
for i, r in enumerate(rel.iter_rows(named=True)):
    axa.plot([i + 0.24] * 2, [r["splits_lo"], r["splits_hi"]], color=SPLIT_GREY, lw=7, solid_capstyle="butt", zorder=2)
    axa.plot([i + 0.17, i + 0.31], [r["splits_median"]] * 2, color="white", lw=1.5, zorder=3)
    axa.errorbar(i - 0.16, r["ratio"], yerr=[[r["ratio"] - r["lo"]], [r["hi"] - r["ratio"]]], fmt="o",
                 color=COLOR[HP], ms=9, capsize=4, lw=1.6, zorder=4)
    rank = "best of\n167,960" if r["splits_at_or_above"] == 1 else f"{r['splits_at_or_above']} splits\nat or above"
    axa.annotate(f"{r['ratio']:.2f}×\n{rank}", (i - 0.16, r["ratio"]), xytext=(-12, 0), textcoords="offset points",
                 ha="right", va="center", fontsize=9.5, color=COLOR[HP], fontweight="bold", linespacing=1.1)
    axa.errorbar(i + 0.02, r["low_ratio"], yerr=[[r["low_ratio"] - r["low_lo"]], [r["low_hi"] - r["low_ratio"]]],
                 fmt="D", color=COLOR[HP], ms=7, capsize=3, lw=1.3, zorder=4)
    axa.annotate(f"{r['low_ratio']:.2f}×", (i + 0.02, r["low_lo"]), xytext=(0, -4), textcoords="offset points",
                 ha="center", va="top", fontsize=9, color=COLOR[HP])
axa.set_xticks(
    range(len(CAT_ORDER)),
    [f"{CAT_LABEL[r['category']]}\n{r['pairs']:,} pairs, {r['folds']} folds\nmedian identity "
     f"{r['median_identity']:.0%}\nbenchmark: {BENCH[r['category']]}" for r in rel.iter_rows(named=True)],
    fontsize=9,
)
axa.set_xlim(-0.6, 3.6)
axa.text(-0.55, 1.05, "equal evidence", ha="left", va="bottom", fontsize=9)
axa.set_title("a   SCOPe pairs, aligned by structure, grouped by SCOPe relationship",
              loc="left", fontsize=11.5, pad=48)
axa.legend(handles=[
    Line2D([], [], marker="o", color=COLOR[HP], ms=8, lw=1.6,
           label="H/P (H = ACFILMVWY, P = the other 11): all pairs, measured whole; 95% interval resampling folds"),
    Line2D([], [], marker="D", color=COLOR[HP], ms=6, lw=1.3,
           label="pairs at 20% identity or less, grouped on one half of the positions, measured on the other"),
    Line2D([], [], color=SPLIT_GREY, lw=7,
           label="all 167,960 splits of the 20 amino acids into 9 + 11: middle 95%, white tick at the median"),
], loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=9, frameon=False, borderaxespad=0)

xb = np.arange(split_tab.height)
labels_b = [f"≤{r['hi']:.0%}" if r["lo"] == 0 else f"{r['lo']*100:g}–{r['hi']*100:g}%" for r in split_tab.iter_rows(named=True)]
axb.axvspan(-0.5, 4.5, color="#F3F3F3", zorder=0)
axb.plot(xb, split_tab["ratio_same_positions"], ls=(0, (5, 3)), color=COLOR[HP], lw=1.3, marker="o",
         mfc="white", mec=COLOR[HP], ms=7, zorder=3)
axb.plot(xb, split_tab["ratio_other_half"], ls="-", color=COLOR[HP], lw=1.8, marker="D", ms=6, zorder=4)
for i in (0, 4):
    axb.annotate(f"{split_tab['ratio_same_positions'][i]:.2f}×", (i, split_tab["ratio_same_positions"][i]),
                 xytext=(9, 2), textcoords="offset points", fontsize=9.5, color=COLOR[HP], va="bottom")
    axb.annotate(f"{split_tab['ratio_other_half'][i]:.2f}×", (i, split_tab["ratio_other_half"][i]),
                 xytext=(0, -15), textcoords="offset points", fontsize=9.5, color=COLOR[HP], ha="center",
                 va="top", fontweight="bold")
axb.text(2.0, 0.14, f"all pairs at 20% identity or less, pooled:\n{pooled_same:.2f}× grouped and "
         f"measured on the same positions\n{pooled_other:.2f}× grouped on one half, measured on the other",
         ha="center", va="center", fontsize=9)
axb.set_xticks(xb, labels_b, fontsize=9.5)
axb.set_xlim(-0.5, len(xb) - 0.5)
axb.set_xlabel(f"identity over aligned positions (Pfam-A 38.2 seed pairs, {n_pf:,} pairs)", fontsize=10.5)
axb.set_title("b   Pfam seed pairs, aligned from sequence, grouped by identity", loc="left",
              fontsize=11.5, pad=48)
axb.legend(handles=[
    Line2D([], [], ls=(0, (5, 3)), color=COLOR[HP], marker="o", mfc="white", mec=COLOR[HP],
           label="grouped and measured on the same positions (as in sections 2-3)"),
    Line2D([], [], ls="-", color=COLOR[HP], lw=1.8, marker="D", ms=6,
           label="grouped on one half of the positions, measured on the other (even/odd, both ways)"),
], loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=9, frameon=False, borderaxespad=0)

r_sf = rel.filter(pl.col("category") == "same_fold_diff_superfamily").row(0, named=True)
r_fam = rel.filter(pl.col("category") == "same_family").row(0, named=True)
r_ssf = rel.filter(pl.col("category") == "same_superfamily_diff_family").row(0, named=True)
hc.finish_figure(
    fig,
    FIG / "273_hp_vs_20aa_evidence_ratio_scope_relationship_and_even_odd_positions.png",
    tools=f"no kmerseek; SCOPe 2.08 40% set (USalign pairs, TM-score ≥ {MIN_TM}) and Pfam-A 38.2 seed pairs "
          f"from notebook 230; {HP} and protein20",
    title="Below 20% identity, H/P keeps more evidence per aligned position than identity in structure alignments, not in sequence alignments",
    hypothesis="If H/P keeps more evidence than identity for distant relatives, the red mark sits above 1× without "
               "grouping pairs by identity, and above other two-letter splits.",
    conclusion=(
        f"Over all identities, H/P has {r_fam['ratio']:.2f}× the evidence of identity for SCOPe same-family pairs, "
        f"{r_ssf['ratio']:.2f}× for same-superfamily, different-family pairs and {r_sf['ratio']:.2f}× for same-fold, "
        f"different-superfamily pairs; of all 167,960 splits into 9 + 11, ACFILMVWY scores highest in every SCOPe group. "
        f"At 20% identity or less, grouped on one half and measured on the other, H/P leads for SCOPe same-superfamily, "
        f"different-family pairs ({r_ssf['low_ratio']:.2f}×) but not for Pfam seed pairs ({pooled_other:.2f}×; at most "
        f"{split_tab['ratio_other_half'].max():.2f}× in any bin), where grouping and measuring on the same positions "
        f"gave {pooled_same:.2f}×."
    ),
    footer_y=-0.035,
)
""")

md(r"""
## 5. Kept, worth, evidence, seeds: four questions asked of the same pairs

Figure 1b of the paper draft reports that, at 20% identity or less, the H/P class is kept 2.8
times as well as the residue (Cohen's κ, `hp_lehninger2`). Section 4 found that in the same
Pfam pairs H/P gives less evidence per aligned position than identity. Both can be true,
because they answer different questions. This section asks four of them of the same pairs:

1. **Is the class kept?** κ = (Pr(agree) − Pr(agree by chance)) / (1 − Pr(agree by chance)),
   the share of the room above chance that the pair fills.
2. **What is one agreement worth?** log2(Pr(agree) / Pr(agree by chance)) in bits, from the
   group's mean rates: how much more often an agreeing position turns up in these pairs than
   by chance.
3. **How much evidence per aligned position?** I, as in section 2.
4. **Can an exact seed find the pair?** The share of pair-halves whose longest run of adjacent
   alignment columns in the same class (the longest k-mer they share on the alignment's
   diagonal; a gap breaks the run) reaches k, drawn against the seed's rarity,
   k × log2(1 / Pr(agree by chance)) bits, with chance from the group's mean. The yellow band
   is 23-32 bits, where the winning kmerseek settings sit in the QfO report's winner analysis
   (quoted in notebook 230: median 28 bits per k-mer, interquartile range 23-32). That analysis
   counts k × log2(number of classes), within 4% of this axis for H/P and 11% for the 20 amino
   acids.

Pairs are grouped as in section 4a, by the identity of one half of their aligned positions and
measured on the other half, but the halves here are the first and second half in alignment
order, because a run needs adjacent positions and the even/odd split of section 4a would break
every run. The measured halves at 20% identity or less have a median of 61 aligned positions,
so a run can be at most that long; the code also prints the seed shares for whole pairs.

Hypothesis: if H/P's higher κ means more evidence, H/P also leads on questions 3 and 4.
""")

code(r"""
# ---- 5. kept, worth, evidence, seeds (Pfam, contiguous halves) ----
THREE = [P20, HP, "hp_lehninger2"]
g_c = pidx * 2 + (ppos >= np.bincount(pidx, minlength=n_pf)[pidx] // 2)
half_n = np.bincount(g_c, minlength=2 * n_pf).reshape(n_pf, 2)
Q5 = {}
for a in THREE:
    _, ag, ch, I = hc.grouped_evidence(pq, pt, g_c, 2 * n_pf, a)
    Q5[a] = dict(agree=ag.reshape(n_pf, 2), chance=ch.reshape(n_pf, 2), I=I.reshape(n_pf, 2),
                 run=hc.longest_runs(pq, pt, g_c, 2 * n_pf, pcol, a).reshape(n_pf, 2))
ident_c = Q5[P20]["agree"]


def measured(a, key, lo, hi):
    # group on one half, measure on the other, both directions pooled
    out = []
    for grp, meas in ((0, 1), (1, 0)):
        out.append(Q5[a][key][in_bin(ident_c[:, grp], lo, hi), meas])
    return np.concatenate(out)


BINS5 = [(0.0, 0.20, "<= 20%"), (0.20, 0.30, "20-30%"), (0.30, 0.40, "30-40%")]
rows5 = []
for lo, hi, lab in BINS5:
    for a in THREE:
        ag, ch = measured(a, "agree", lo, hi), measured(a, "chance", lo, hi)
        # a half whose residues all fall in one class has chance 1: no kappa and no I (nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            kap = (ag - ch) / (1 - ch)
        rows5.append(dict(
            identity=lab, alphabet=a, pair_halves=ag.size, without_kappa=int(np.isnan(kap).sum()),
            agree_per_100=100 * ag.mean(), chance_per_100=100 * ch.mean(),
            kappa=np.nanmean(kap),
            bits_per_agreement=np.log2(ag.mean() / ch.mean()),
            I_bits=np.nanmean(measured(a, "I", lo, hi)),
            median_run=float(np.median(measured(a, "run", lo, hi))),
        ))
t5 = pl.DataFrame(rows5)
measured_len = np.concatenate([half_n[in_bin(ident_c[:, grp], 0.0, 0.20), meas] for grp, meas in ((0, 1), (1, 0))])
print(f"Pfam seed pairs, grouped on one contiguous half and measured on the other; measured halves at "
      f"20% identity or less have a median of {np.median(measured_len):.0f} aligned positions")
print(t5.with_columns(pl.col(pl.Float64).round(3)))

# Figure 1b's reading: whole pairs, grouped by their own identity (seqid_ali, notebook 230's bins)
whole = []
for a in THREE:
    _, ag_w, ch_w, _ = hc.grouped_evidence(pq, pt, pidx, n_pf, a)
    whole.append(pl.DataFrame({"alphabet": a, "seqid_ali": pf_pairs["seqid_ali"], "kappa": (ag_w - ch_w) / (1 - ch_w)}))
whole = hc.add_identity_bin(pl.concat(whole)).filter(pl.col("identity_bin") == "<20%")
kw = dict(whole.group_by("alphabet").agg(pl.col("kappa").mean()).iter_rows())
k5 = {a: t5.filter((pl.col("identity") == "<= 20%") & (pl.col("alphabet") == a))["kappa"][0] for a in THREE}
print("\nkappa ratio to the 20 amino acids at 20% identity or less:")
for a in (HP, "hp_lehninger2"):
    print(f"  {a}: whole pairs grouped by their own identity {kw[a] / kw[P20]:.2f} "
          f"({kw[a]:.3f} vs {kw[P20]:.3f}); grouped on one half, measured on the other "
          f"{k5[a] / k5[P20]:.2f} ({k5[a]:.3f} vs {k5[P20]:.3f})")

# seed reach against seed rarity at 20% identity or less
KS = np.arange(1, 61)
reach_rows = []
for a in THREE:
    runs = measured(a, "run", 0.0, 0.20)
    bpl = -np.log2(measured(a, "chance", 0.0, 0.20).mean())
    frac = np.array([(runs >= k).mean() for k in KS])
    Q5[a]["reach"] = (KS * bpl, frac)
    reach_rows.append({"alphabet": a, "bits_per_letter": bpl,
                       **{f"share_at_{b}_bits_pct": 100 * float(np.interp(b, KS * bpl, frac)) for b in (12, 20, 24, 28)}})
print("\nShare of pair-halves at 20% identity or less with an exact same-class run of that rarity (%)")
print(pl.DataFrame(reach_rows).with_columns(pl.col(pl.Float64).round(2)))

# the same on whole pairs grouped by their own identity (seqid_ali <= 0.2), which favours H/P
w20 = (pf_pairs["seqid_ali"] <= 0.20).to_numpy()
whole_rows = []
for a in THREE:
    _, _, ch_w, _ = hc.grouped_evidence(pq, pt, pidx, n_pf, a)
    runs_w = hc.longest_runs(pq, pt, pidx, n_pf, pcol, a)[w20]
    bpl_w = -np.log2(ch_w[w20].mean())
    frac_w = np.array([(runs_w >= k).mean() for k in KS])
    whole_rows.append({"alphabet": a, **{f"share_at_{b}_bits_pct": 100 * float(np.interp(b, KS * bpl_w, frac_w)) for b in (20, 24, 28)}})
print(f"\nWhole pairs grouped by their own identity (seqid_ali <= 20%, {int(w20.sum()):_} pairs), same measure (%)")
print(pl.DataFrame(whole_rows).with_columns(pl.col(pl.Float64).round(2)))
""")

code(r"""
# ---- figure: the four questions, 20% identity or less ----
le = t5.filter(pl.col("identity") == "<= 20%")
S5 = {a: le.filter(pl.col("alphabet") == a).row(0, named=True) for a in (P20, HP)}
order5 = [P20, HP]
LAB5 = {P20: "20 amino acids (identical residue)", HP: f"H/P ({HP}: ACFILMVWY | rest)"}
SHORT5 = {P20: "20 amino\nacids", HP: "H/P"}
fig, axs = plt.subplots(1, 4, figsize=(19, 5.6), gridspec_kw=dict(width_ratios=[1.25, 0.8, 0.8, 1.5]))
fig.legend(handles=[Patch(color=COLOR[a], label=LAB5[a]) for a in order5], loc="upper left",
           bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=10, frameon=False)
ax = axs[0]
for i, a in enumerate(order5):
    s, y = S5[a], 1 - i
    ax.plot([s["chance_per_100"], 100], [y, y], color=COLOR[a], lw=10, alpha=0.18, solid_capstyle="butt")
    ax.plot([s["chance_per_100"], s["agree_per_100"]], [y, y], color=COLOR[a], lw=10, solid_capstyle="butt")
    ax.plot([s["chance_per_100"]] * 2, [y - 0.22, y + 0.22], color="black", lw=1.5)
    ax.text(s["chance_per_100"], y + 0.27, f"chance {s['chance_per_100']:.0f}", ha="center", va="bottom", fontsize=9)
    ax.text(s["agree_per_100"], y - 0.27, f"agree {s['agree_per_100']:.0f}", ha="center", va="top", fontsize=9, color=COLOR[a])
    ax.text(101, y, f"κ = {s['kappa']:.2f}", ha="left", va="center", fontsize=11, color=COLOR[a], fontweight="bold")
ax.set_yticks([1, 0], ["20 amino acids", "H/P"])
ax.set_ylim(-0.7, 1.7)
ax.set_xlim(0, 118)
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.set_xlabel("aligned positions per 100")
ax.set_title("a  Is the class kept?\n    κ = kept above chance ÷ room above chance", loc="left", fontsize=11)
ax.legend(handles=[Patch(color=COLOR[P20], label="kept above chance (dark)"),
                   Patch(color=COLOR[P20], alpha=0.18, label="room above chance (light)"),
                   Line2D([], [], color="black", lw=1.5, label="chance, from each half's own class shares")],
          loc="lower left", bbox_to_anchor=(0, 1.13), fontsize=8.5, frameon=False, borderaxespad=0)
for ax, key, unit, ylab, title, top in [
    (axs[1], "bits_per_agreement", "bits", "bits per agreeing position,\nlog2(agree ÷ chance)", "b  What is one\n    agreement worth?", 2.1),
    (axs[2], "I_bits", "bits", "evidence I per aligned position (bits)", "c  Evidence per\n    aligned position", 0.26),
]:
    vals = [S5[a][key] for a in order5]
    ax.bar([0, 1], vals, color=[COLOR[a] for a in order5], width=0.6)
    for x, v in zip([0, 1], vals):
        ax.text(x, v + top * 0.02, f"{v:.2f} {unit}" if key == "bits_per_agreement" else f"{v:.3f} {unit}",
                ha="center", va="bottom", fontsize=10)
    ax.set_xticks([0, 1], [SHORT5[a] for a in order5])
    ax.set_ylim(0, top)
    ax.set_ylabel(ylab)
    ax.set_title(title, loc="left", fontsize=11)
ax = axs[3]
ax.axvspan(23, 32, color="#FFF3C4", zorder=0)
ax.text(27.5, 0.6, "where the winning\nkmerseek settings sit\n(23-32 bits, notebook 230)", ha="center", va="top", fontsize=8.5)
for a in order5:
    bits, frac = Q5[a]["reach"]
    ax.plot(bits, frac, color=COLOR[a], lw=2, marker="o", ms=3.5)
    kk = 6 if a == P20 else 25
    ax.annotate(f"k = {kk}", (bits[kk - 1], frac[kk - 1]), xytext=(6, 4), textcoords="offset points", fontsize=8.5, color=COLOR[a])
ax.set_yscale("log")
ax.set_xlim(0, 40)
ax.set_ylim(1e-4, 1.05)
ax.set_xlabel("seed rarity: k × log2(1 ÷ chance), bits")
ax.set_ylabel("share of pair-halves with an exact\nsame-class run of k (log scale)")
ax.set_title("d  Can an exact seed of the same rarity find the pair?", loc="left", fontsize=11)
for ax in axs:
    ax.spines[["top", "right"]].set_visible(False)
r24 = {a: 100 * float(np.interp(24, *Q5[a]["reach"])) for a in order5}
hc.finish_figure(
    fig,
    FIG / "273_kept_worth_evidence_seeds_hp_vs_20aa_pfam_seed_le20pct.png",
    tools=f"no kmerseek; Pfam-A 38.2 seed pairs from notebook 230; {HP} and protein20",
    title="At 20% identity or less, the H/P class is kept more reliably than the residue, but each kept class says less",
    hypothesis="If H/P's higher κ meant more evidence, H/P would also lead in c and d.",
    conclusion=(
        f"Pfam seed pairs at 20% identity or less, grouped on one half of their aligned positions and measured on the other "
        f"({S5[HP]['pair_halves']:,} pair-halves). κ: H/P {S5[HP]['kappa']:.2f} vs {S5[P20]['kappa']:.2f}. An H/P agreement "
        f"is worth {S5[HP]['bits_per_agreement']:.2f} bits against {S5[P20]['bits_per_agreement']:.2f}, so I is "
        f"{S5[HP]['I_bits']:.3f} vs {S5[P20]['I_bits']:.3f} bits per position, and at 24 bits of seed rarity "
        f"{r24[HP]:.1f}% of pair-halves have an exact H/P run against {r24[P20]:.1f}% with identical residues."
    ),
    footer_y=-0.06,
)
""")

md(r"""
## 6. Measured against the kappa-based estimates

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
## 7. One typical pair at each identity, position by position

Section 1 showed one pair. Here is one pair from each Pfam identity bin, so the numbers in
sections 2 to 6 can be seen as residues. Each pair is the bin's most typical one, chosen by a
rule fixed before looking at the pairs: among pairs with 80 to 150 gap-free columns (short
enough to read), the pair whose H/P I and 20 amino acid I are both closest to the middle of
the bin (the smallest sum of the two distances from the 50th percentile, percentiles taken
over all pairs in the bin), one Pfam family per bin.

For each pair the printed lines are: the two aligned sequences, with `|` where the residues
are identical (the 20 amino acid count); then the two sequences written as H and P, with `|`
where both residues are H or both are P (the H/P count). Columns with a gap on either side
(`-` or `.`) are not counted. The table first gives each pair's counts beside the bin's
median, so you can see how typical it is. The figure draws all five pairs on one column
scale, highest identity at the top.
""")

code(r"""
wide = (
    pfam.pivot(
        on="alphabet",
        index=["identity_bin", "family", "family_id", "query", "target", "seqid_ali", "n_cols"],
        values=["agree", "expected", "evidence"],
    )
    .with_columns(
        (pl.col(f"evidence_{a}").rank() / pl.len()).over("identity_bin").alias(f"pct_{a}")
        for a in ALPHABETS
    )
    .with_columns(
        ((pl.col(f"pct_{HP}") - 0.5).abs() + (pl.col(f"pct_{P20}") - 0.5).abs()).alias("dist")
    )
)
picked, used = [], set()
for b in hc.IDENTITY_LABELS:
    cand = (
        wide.filter((pl.col("identity_bin") == b) & pl.col("n_cols").is_between(80, 150))
        .filter(~pl.col("family").is_in(list(used)))
        .sort(["dist", "family", "query", "target"])
    )
    row = cand.row(0, named=True)
    used.add(row["family"])
    picked.append(row)
ex_pairs = pl.DataFrame(picked).with_columns(
    pl.col("identity_bin").cast(pl.Enum(hc.IDENTITY_LABELS))
)

med = wide.group_by("identity_bin").agg(
    pl.len().alias("pairs in bin"),
    *[(pl.col(f"agree_{a}").median() * 100).round(1).alias(f"median agree {a}") for a in ALPHABETS],
    *[pl.col(f"evidence_{a}").median().round(3).alias(f"median I {a}") for a in ALPHABETS],
)
show = ex_pairs.join(med, on="identity_bin").sort("identity_bin").select(
    "identity_bin",
    pl.format("{} ({})", "family", "family_id").alias("Pfam family"),
    "query",
    "target",
    (pl.col("seqid_ali") * 100).round(1).alias("identity %"),
    pl.col("n_cols").alias("aligned positions"),
    (pl.col(f"agree_{P20}") * 100).round(1).alias("identical per 100"),
    f"median agree {P20}",
    (pl.col(f"agree_{HP}") * 100).round(1).alias("both H or both P per 100"),
    f"median agree {HP}",
    (pl.col(f"expected_{HP}") * 100).round(1).alias("H/P chance per 100"),
    pl.col(f"evidence_{P20}").round(3).alias(f"I {P20}"),
    f"median I {P20}",
    pl.col(f"evidence_{HP}").round(3).alias(f"I {HP}"),
    f"median I {HP}",
    "pairs in bin",
)
print("One pair per Pfam identity bin; medians are over all pairs in that bin (>= 50 aligned positions)")
print(show)

aln_ex = aln.join(ex_pairs.select("family", "query", "target", "identity_bin"), on=["family", "query", "target"])
ex_lines = {}
for r in aln_ex.sort("identity_bin").iter_rows(named=True):
    L = hc.alignment_lines(r["qaln"], r["taln"], HP)
    ex_lines[r["identity_bin"]] = (r, L)
    n_al = L["n_aligned"]
    n_id, n_hp = L["identical"].count("|"), L["same_class"].count("|")
    print(
        f"\n=== {r['identity_bin']} identity: {r['family']} ({r['family_id']}), "
        f"{r['query']} against {r['target']}"
    )
    print(
        f"gap-free columns {n_al}; identical {n_id} ({100 * n_id / n_al:.0f} per 100); "
        f"both H or both P {n_hp} ({100 * n_hp / n_al:.0f} per 100)\n"
    )
    hc.print_alignment(
        [
            ("query", L["query"]),
            ("identical", L["identical"]),
            ("target", L["target"]),
            ("query H/P", L["query_class"]),
            ("both H or both P", L["same_class"]),
            ("target H/P", L["target_class"]),
        ]
    )
""")

code(r"""
import pubfig as pf
from matplotlib.patches import Rectangle

# Drawn at its printed width (180 mm with padding), in the Nature style; the notebook copy is the same
# figure at 200 dpi, the print copy at 300 dpi. This is the notebook's last figure, so the
# style is left on.
pf.use_style()
GAP_COLOR = "#e4e4e4"  # gap in either sequence; the only grey in the figure
order = list(reversed(hc.IDENTITY_LABELS))  # >=60% at the top, identity falling down the page
x_max = max(len(ex_lines[b][1]["query"]) for b in order)
WIDTH_MM = 179  # plus 1 mm of save padding each side = 180 mm printed
plt.rcParams["savefig.pad_inches"] = 1 / 25.4  # room for the footer box outline

fig = plt.figure(figsize=(WIDTH_MM * pf.MM, 112 * pf.MM))
gs = fig.add_gridspec(
    len(order) + 1, 1, height_ratios=[2.3] + [1] * len(order), hspace=1.0,
    left=0.085, right=0.995, top=0.995, bottom=0.06,
)
leg = fig.add_subplot(gs[0])
axes = [fig.add_subplot(gs[i + 1]) for i in range(len(order))]
for ax in axes[1:]:
    ax.sharex(axes[0])

drawn = {}
for ax, b in zip(axes, order):
    r, L = ex_lines[b]
    c = hc.draw_hp_pair(ax, L, H_COLOR, P_COLOR, GAP_COLOR, x_max=x_max)
    drawn[b] = c
    n = c["gap_free"]
    ax.set_title(
        f"{b.replace('>=', '≥')} identity. {r['family_id']} ({r['family']}), "
        f"{r['query']} against {r['target']}\n"
        f"{c['identical']} of {n} gap-free columns identical ({100 * c['identical'] / n:.0f} per 100), "
        f"{c['same_class']} of {n} both H or both P ({100 * c['same_class'] / n:.0f} per 100)",
        loc="left", pad=2, linespacing=1.3,
    )
    if ax is not axes[-1]:
        ax.tick_params(axis="x", labelbottom=False)
axes[-1].set_xlabel("alignment column (gap columns included)")

# Legend above the first pair: a sample of each mark at the same size as in the pairs (same
# column scale, same row height), three columns of two entries.
leg.set_xlim(0.5, x_max + 0.5)
leg.set_ylim(6.9, -0.5)
leg.axis("off")
def sample(x0, y0, top, bottom, tick=None, gap=False):
    # one alignment column of a pair: query row at y0, match line at y0+1, target at y0+2
    if gap:
        leg.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), 1, 3, lw=0, color=GAP_COLOR, zorder=0))
    for dy, col in ((0, top), (2, bottom)):
        if col:
            leg.add_patch(Rectangle((x0 - 0.5, y0 + dy - 0.35), 1, 0.7, lw=0, color=col, zorder=1))
    if tick:
        height = hc.TICK_TALL if tick == "tall" else hc.TICK_SHORT
        leg.add_patch(Rectangle((x0 - hc.TICK_WIDTH / 2, y0 + hc.TICK_BASE - height), hc.TICK_WIDTH, height,
                                lw=0, color="black", zorder=2))
entries = [
    (dict(top=H_COLOR, bottom=None, single=True),
     f"H residue, hydrophobic: {h_res}.\nCounted unless its column is a gap"),
    (dict(top=P_COLOR, bottom=None, single=True),
     f"P residue, polar: {p_res}.\nCounted unless its column is a gap"),
    (dict(top=H_COLOR, bottom=H_COLOR, tick="tall"),
     "Tall tick: identical residues. Counted\nas identical and as both H or both P"),
    (dict(top=P_COLOR, bottom=P_COLOR, tick="short"),
     "Short tick: different residues, both H or\nboth P. Counted as both H or both P"),
    (dict(top=H_COLOR, bottom=P_COLOR),
     "No tick: one H, one P.\nCounted in the total only"),
    (dict(top=H_COLOR, bottom=None, gap=True),
     "Grey band: gap in either sequence.\nNot counted"),
]
col_x = [2, 2 + x_max / 3, 2 + 2 * x_max / 3]
for k, (kw, text) in enumerate(entries):
    x0, y0 = col_x[k // 2], (k % 2) * 3.7
    if kw.get("single"):
        leg.add_patch(Rectangle((x0 - 0.5, y0 + 1 - 0.35), 1, 0.7, lw=0, color=kw["top"]))
    else:
        sample(x0, y0, kw["top"], kw["bottom"], kw.get("tick"), kw.get("gap", False))
    leg.text(x0 + 2, y0 + 1, text, va="center", ha="left", linespacing=1.15)

print("Counts drawn in each title (gap-free columns are the denominator):")
for b in order:
    c = drawn[b]
    print(
        f"{b}: identical {c['identical']} of {c['gap_free']}, both H or both P "
        f"{c['same_class']} of {c['gap_free']}, gap columns {c['gap_columns']}"
    )
    # the drawn counts must equal the counts the table above was built from
    r, L = ex_lines[b]
    assert c["identical"] == L["identical"].count("|") and c["same_class"] == L["same_class"].count("|")

hi, lo = drawn[">=60%"], drawn["<20%"]
per100 = lambda c, k: round(100 * c[k] / c["gap_free"])
out = FIG / "273_one_pair_per_identity_bin_hp_and_protein20_pfam_seed.png"
hc.finish_figure(
    fig,
    out,
    tools=f"no kmerseek; Pfam-A 38.2 seed alignments, {HP} H/P letters and protein20",
    title="How H/P agreement is counted: one Pfam seed pair per identity bin, position by position",
    subtitle=(
        "Pair shown for each bin: of its pairs with 80 to 150 gap-free columns, the one nearest the "
        "bin median in both H/P and 20 amino acid bits per position, each from a different Pfam family"
    ),
    hypothesis=(
        "As identity falls, identical residues drop faster than residues that are both H or both P. "
        "Five pairs cannot test this; the test over all pairs is "
        "figures/273_hp_vs_protein20_agreement_per_100_and_bits_pfam_seed_scope40.png."
    ),
    conclusion=(
        f"Identity sets the bins, so identical residues fall by construction: from "
        f"{per100(hi, 'identical')} to {per100(lo, 'identical')} per 100 gap-free columns between "
        f"the ≥60% pair and the <20% pair. Residues both H or both P fall less in these five pairs, "
        f"from {per100(hi, 'same_class')} to {per100(lo, 'same_class')} per 100."
    ),
    font_scale=0.62,
    wrap=175,
    footer_y=-0.035,
    footer_box=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="black", lw=0.5),
    tight_layout=False,
)
fig.savefig(out.with_name(out.stem + "_print.png"), dpi=300, bbox_inches="tight")
""")

md(r"""
## 8. Summary and conclusions

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
5. A result the kappa summary did not show, which does not hold up. In Pfam pairs at 20% identity
   or less, H/P gives more evidence per aligned position than the 20 amino acids (0.103 vs 0.091
   bits, 1.13 times), and the lead holds when whole families are resampled (section 3). But
   identity is the 20 amino acid agreement itself (equal for 99.7% of pairs), so that bin holds
   the pairs where the 20 amino acids happened to agree least. Grouped on half of each pair's
   positions and measured on the other half, the lead is gone: 0.79 times at 20% or less, and at
   most 0.91 times in any identity bin (section 4). Picking pairs the other way, by the lowest
   11% of H/P agreement, gives H/P 0.44 times. Whichever alphabet the pairs are picked by looks
   worst.
6. Over all identities, grouped by SCOPe relationship (which does not use identity), homologs
   favour the 20 amino acids: H/P has 0.61 times their evidence for same-family pairs and 0.72
   times for same-superfamily, different-family pairs. H/P leads for the pairs SCOPe does not
   call homologs: same fold, different superfamily, 1.67 times (95% interval 1.50 to 1.88,
   resampling the query's fold; 1,204 pairs, median identity 9%), and different fold, same
   class, 1.52 times (186 pairs). Below 20% identity the SCOPe homologs move toward H/P. Grouped
   on one half and measured on the other, same-superfamily, different-family pairs give H/P 1.07
   times (95% interval 1.03 to 1.10) and same-family pairs 0.90 times. The Pfam pairs at the same
   identity, aligned from sequence, give 0.79 times. Every SCOPe pair here has TM-score 0.5 or
   more and is aligned by structure. So the H/P agreement may come from the superposition placing
   buried positions on buried positions, from shared packing, or from a common ancestor. This
   notebook cannot tell these apart, nor whether the SCOPe-Pfam difference comes from how the
   pairs were aligned or from which pairs are in each set.
7. Of all 167,960 ways to split the 20 amino acids into a class of 9 and a class of 11,
   ACFILMVWY gives the most evidence per aligned position in every SCOPe group (section 4b). In
   three samples of 5,000 Pfam seed pairs, aligned from sequence, it ranks second or third; the
   best there is CFHILMVWY, which has histidine in place of alanine. The splits that keep the
   most evidence are the hydrophobic residues, give or take one letter.
8. The H/P class is kept more reliably than the residue, but each kept class says less
   (section 5). At 20% identity or less, grouped on one half of the positions and measured on
   the other, κ is 0.42 for H/P against 0.17 for the 20 amino acids (2.5 times; hp_lehninger2
   0.36, 2.1 times). One H/P agreement is worth 0.48 bits and one identical residue 1.74 bits,
   and I is 0.148 against 0.218 bits per position. Exact runs are longer in H/P (median 9 against
   2 adjacent columns, in measured halves of median 61 positions). But at 24 bits of seed rarity,
   0.9% of pair-halves have an H/P run that long and 4.5% have an identical-residue run. On whole
   pairs grouped by their own identity, which favours H/P, the shares are 0.4% and 1.3%.

**Limits.**
* Aligned positions only, with the alignment given. Whether an exact k-mer index can reach a pair
  at all is notebook 230 sections 4-6.
* The SCOPe categories pool all identities; the right column of section 2 mixes relatedness
  with identity.
* The decision rule is loose for small numbers: ± 3 per 100 around a chance of 6 lets 3 to 9
  pass, and ± 0.03 bits is a fifth of 0.15. Read the measured values, not only the pass/fail.
* Identity bins are notebook 230's (`pl.cut`, closed on the right): a pair at exactly 20% counts
  in "<20%", one at exactly 30% in "20-30%".
* Pfam seed pairs are capped at 6 per family, so a large family counts no more than a small one
  with 4 or more seed members.
* Sections 4a and 5 measure each pair on half its positions, which raises the shuffled floor of
  I in both alphabets by about the same amount. An equal amount added to both pulls the ratio toward
  1, so removing the extra floor would lower the 0.79 slightly, not raise it.
* The SCOPe intervals resample the query's fold only. A different-fold pair has two folds, so
  the interval for that group is too narrow by an unknown amount.

**Decision.**

1. Which numbers to quote. Quote 74 vs 51 aligned positions per 100 in H/P and 25 vs 6 for the
   20 amino acids, at 20-30% identity in Pfam seed pairs, with chance from a random partner's
   composition. For evidence per aligned position, quote the measured 0.16 bits (H/P) and 0.25
   bits (20 amino acids), citing this notebook.
2. Whether to argue that H/P keeps more evidence per aligned position. Not from the "<20%" bin:
   do not quote "1.13 times under 20% identity". It comes from how the pairs were picked; Pfam
   pairs grouped on the other half of their positions give 0.79 times, and at most 0.91 times in
   any bin. With a structural alignment the lead is real below 20% identity, even within a
   superfamily (SCOPe same superfamily, different family: 1.07 times), and largest at the fold
   level (1.67 times over all identities). The paper can use two sentences. First, "H/P class is
   kept: 74 of 100 aligned positions against 51 by chance". Second, "of all 167,960 ways to split
   the amino acids into 9 and 11, the hydrophobic set keeps the most evidence per aligned
   position between structure-aligned domains at every SCOPe level". Each needs its condition
   beside it.
   The lead over identity holds for pairs aligned by structure, not for Pfam's sequence
   alignments. At the fold level it is between domains that a homology benchmark either does not
   score (same fold) or counts as a false positive (different fold).
   All of these are per aligned position, for a method that is given the correct alignment.
   An exact k-mer sees less: notebook 230 found that an H/P 26-mer is shared by 1.7% of 20-30% pairs,
   against 11% for a protein 6-mer of about the same bits.
3. Figure 1b. Its 2.8 times at 20% identity or less groups pairs by their own identity, so it
   carries the picking effect of section 3. Grouped that way, this notebook gets 2.76 times for
   hp_lehninger2; grouped on one half and measured on the other, 2.09 times. Quote the second,
   or say how the pairs were grouped. "The H/P class is kept" stands; "H/P gives more evidence"
   does not.
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
