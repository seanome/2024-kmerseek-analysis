#!/usr/bin/env python3
"""Generate notebooks/294_linked_hp_word_pairs_kill_test_pfam_seed.ipynb."""

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
# 294. Two short linked HP words instead of one long one: a kill test on Pfam seed pairs

**Result:** no. By the rule set before the run the answer was "neither PASS nor KILL",
because its KILL condition can be read three ways (section 4a). By the sharper rule set
after that run (section 4b), it is a KILL. At 20–30% identity and exactly FPR 0.001, the
best linked setting finds 6.7 percentage points more true pairs than a single 8-letter
seed (95% interval 3.2 to 8.9, families resampled), all of it below the 10 points asked
for. The same seed plus a second extended segment, with no second exact word (B3), does as
well (linked minus B3: −1.7, interval −3.0 to 0.5). The gain comes from scoring a second
segment across an insertion, not from the second word in the key. The kmerseek index
stays as it is.

**Question.** [Notebook 232](232_mismatch_tolerant_seed_extend.ipynb) found that the longest
exact run of matching hydrophobic/polar (HP) classes between true homologs at 20–30%
identity is about 12 residues, so one exact HP word of 20 or more letters rarely survives
between them. Does a key made of two short exact words, each of which only has to survive
6–10 residues, find more true pairs than 232's single 10-letter seed at the same
false-positive rate?

**Words used here**

- *HP alphabet*: each residue is written as H (hydrophobic: `ACFILMVWY`) or P (polar:
  `DEGHKNPQRST`), the 2-letter Thomas–Dill split used in notebooks 230–232
  (`hp_thomas_dill2`). X ends a word.
- *Seed*: a run of aligned columns whose HP classes all match, as a BLAST word hit.
  *Extension*: from the seed, walk along the alignment without gaps, scoring +1 per column
  whose HP classes match and −2 per column where they differ; a gap column stops the walk.
  The seed's score is the best total reached (232's scoring).
- *Anchor*: one short exact HP word (6, 7, 8 or 10 letters) shared by both sequences.
  *Gap*: the number of residues between the end of the first anchor and the start of the
  second, counted in each sequence. *Window*: the largest gap allowed (8, 16 or 32).
- *Key* (a linked word pair; Sahlin 2021 calls it a strobemer, see
  [`294_prior_art.md`](294_prior_art.md)): two anchors plus a gap rule. The gap is stored
  exactly (`exact`: query gap = target gap), stored in 3-residue bins (`binned3`), or not
  stored (`none`: both gaps only have to fit the window, so the key can step over an
  insertion or deletion).
- *How keys are picked*: every anchor pair inside the window (`all`, many keys per
  position), or one partner per position (`rand`): for each first anchor, the second is the
  word in the window with the smallest hash of the two words combined (Sahlin's randstrobe
  rule). Each sequence picks on its own, so the two pick the same partner only when their
  windows agree. One key per position is what an index can afford to store.
- *Comparison*: one query–target pair, real or shuffled, scored only on its own alignment.
  A seed or anchor counts only where it sits on aligned columns, as in 232.
- *Null*: the true pair with the target's residues shuffled along its alignment row (gaps
  stay put, so composition and gap pattern are kept). *Threshold*: the score that 0.1% of
  null comparisons exceed. *False-positive rate (FPR)*: the share of null comparisons above
  the threshold; it is at or below 0.001 here and is printed for every setting.
  *Recall*: the share of true pairs above the threshold. Higher recall is better.
- *B1*: 232's 10-letter seed with extension. *B2*: an 8-letter seed with the same
  extension, so linking is charged against a shorter single seed as well as a longer one.
  *B3* (added 2026-10-15): B2's seed plus the best second gap-free segment containing any
  matching column within 32 residues, scored like a linked key, to tell a second exact word
  apart from a second extended segment.
- *Linked-pair score*: each shared key is extended with the same step from both anchors.
  A key's score is the best total of at most two non-overlapping gap-free segments that
  together contain both anchors. The pair's score is its best key. The no-extension score
  is the number of shared keys.
- *Bootstrap interval*: resample with replacement 1,000 times, each true pair with its own
  shuffled targets, recompute threshold and recall; the middle 95% of values. Section 4a
  resamples pairs one by one; section 4b resamples whole Pfam families.
- *Gate 1*: a check, before anything new, that this code reproduces notebook 232's numbers.

**Data, all from notebook 232**

- *True pairs*: 60,000 pairs of Pfam 38.2 seed-alignment members with at least 50 aligned
  columns, sampled from `~/data/pfam/230_pfam_seed_pair_alignments.parquet` (polars sample,
  seed 0). Per identity bin: <20% 6,655; 20–30% 15,130; 30–40% 13,872; 40–60% 16,862;
  ≥60% 7,481. Identity is identical residues over aligned columns; as in 232, a bin
  includes its upper edge (exactly 20% falls in <20%, exactly 60% in 40–60%).
- *Null*: two shuffles per pair, 120,000 null comparisons, 60,000 in each half. That is
  above the 20,000 the 0.001 threshold needs to rest on at least 20 events, so the null was
  not enlarged.
- *Threshold rule*: 232's: the 99.9th percentile of the pooled null scores, taking the
  higher of two neighbouring values; recall counts true pairs scoring strictly above it.

**Rule, fixed before the run** (task brief, 2026-10-14). The setting is chosen on a random
half of the pairs (by recall at 20–30% identity); every reported number comes from the
other half.

- PASS: in the 20–30% identity bin, the chosen linked-pair setting recovers at least 10
  percentage points more true pairs than the better of B1 and B2 at FPR 0.001, and the two
  bootstrap 95% intervals do not overlap.
- KILL: no linked-pair setting beats the better of B1 and B2 in any identity bin outside
  the bootstrap interval.
- Anything else: report, and do not touch the index.

**Rule set after the first run** (2026-10-15, after section 4a's numbers were seen, because
the KILL condition above can be read three ways): one paired gain of the chosen setting over
the better single seed at 20–30% identity, at exactly FPR 0.001, with families resampled.
PASS if its whole 95% interval is at or above 10 points; KILL if the whole interval is below
10; unclear otherwise. Section 4b gives it, and says it is not a pre-registered test.

**Decision this informs.** Whether kmerseek's index should store linked HP word pairs
instead of single k-mers. On a pass the next step would be a one-page design note, not
code.

Code: [`linked_pairs_utils.py`](linked_pairs_utils.py) (scoring, tested against brute force
in `tests/test_linked_pairs_utils.py`); scores from
`scripts/linked_pairs_on_alignments.py`; Swiss-Prot key rates from
`scripts/linked_pairs_key_spectrum.py`. No kmerseek run; everything is Python on the
laptop.
""")

code(r"""
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path.cwd()))
import hp_conservation_utils as hc
import linked_pairs_utils as lp
import pubfig as pf
import seed_extend_utils as se

pf.use_style()
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(220)

FIG = Path("../figures")
SCORES = Path("/Users/olga/data/pfam/294_linked_pairs_scores.parquet")
SPECTRUM = Path("/Users/olga/data/pfam/294_linked_pairs_key_spectrum_swissprot.parquet")
NB232 = Path("/Users/olga/data/pfam/232_seed_extend_scores.parquet")
ALN = Path("/Users/olga/data/pfam/230_pfam_seed_pair_alignments.parquet")
FPR = 1e-3
N_BOOT = 1000
SPLIT_SEED = 292  # kept from the first run, when this notebook was numbered 292; it fixes the halves
SPROT_FASTA = Path("/Users/olga/data/uniprot/uniprot_sprot.fasta.gz")  # the FASTA the spectrum script read
BINS = lp.BINS

# One colour per meaning, kept across all three figures.
COLOR = {
    "B1": pf.OKABE_ITO["blue"],  # 232's 10-residue seed
    "B2": pf.OKABE_ITO["sky_blue"],  # 8-residue seed; B3 (same seed + second segment) shares it, open marker
    "linked": pf.OKABE_ITO["vermillion"],  # chosen linked pair, with extension
    "linked_count": pf.OKABE_ITO["orange"],  # chosen linked pair, key count only
    "exact": pf.OKABE_ITO["bluish_green"],  # Fig. 3 gap rules
    "binned3": pf.OKABE_ITO["reddish_purple"],
    "none": pf.OKABE_ITO["black"],
}
TREAT_LABEL = {"exact": "gap stored exactly", "binned3": "gap in 3-residue bins",
               "none": "gap not stored"}
def plain(setting: str) -> str:
    m, a, g, t = next(x for x in lp.ALL_SETTINGS if lp.setting_name(*x) == setting)
    pick = "every pair of" if m == "all" else "randstrobe pairs of"
    return f"{pick} {a}-letter anchors, gap up to {g}, {TREAT_LABEL[t]}"


MODE_LABEL = {"all": "every anchor pair in the window", "rand": "one partner per position (randstrobe)"}

d = hc.add_identity_bin(pl.read_parquet(SCORES))
n_pairs = d.filter(pl.col("kind") == "real").height
print(f"{n_pairs:_} true pairs, {d.height - n_pairs:_} null comparisons")
print(d.filter(pl.col("kind") == "real").group_by("identity_bin").len().sort("identity_bin"))

LINKED = [lp.setting_name(*s) for s in lp.ALL_SETTINGS]
COLS = ["B1_ext", "B2_ext", "B3_ext", "seed10_count", "seed8_count"] + [f"{s}_ext" for s in LINKED] + [f"{s}_count" for s in LINKED]
""")

md(r"""
## 1. Gate 1: notebook 232's baseline, reproduced

B1 rescored from the alignments with this notebook's code, next to the scores 232 stored.
The rule for going on: every per-bin recall equal to 232's.
""")

code(r"""
real_all, null_all, bin_all, pidx_all = lp.score_arrays(d, COLS)
thr_all, rate_all, rec_all = lp.threshold_and_recall(real_all, null_all, bin_all, FPR)

s232 = hc.add_identity_bin(
    pl.read_parquet(NB232).filter(
        (pl.col("dataset") == "pfam") & (pl.col("alphabet") == lp.ALPHABET) & (pl.col("penalty") == 2.0)
        & pl.col("seed").is_in([8, 10])
    )
)
r232 = se.recall_at_null_quantile(s232, ["seed"], FPR)
same_scores = (
    d.join(s232.pivot(on="seed", index=["query", "target", "kind"], values="score"), on=["query", "target", "kind"])
    .select((pl.col("B1_ext") == pl.col("10")).mean().alias("B1 equal"), (pl.col("B2_ext") == pl.col("8")).mean().alias("B2 equal"), pl.len())
)
print("share of the 180,000 comparisons whose score equals 232's stored score:")
print(same_scores)
gate1 = pl.DataFrame({
    "identity_bin": BINS,
    "B1 here": rec_all[:, COLS.index("B1_ext")].round(3),
    "B1 in 232": [r232.filter((pl.col("seed") == 10) & (pl.col("identity_bin") == b))["recall"][0] for b in BINS],
    "B2 here": rec_all[:, COLS.index("B2_ext")].round(3),
    "B2 in 232": [r232.filter((pl.col("seed") == 8) & (pl.col("identity_bin") == b))["recall"][0] for b in BINS],
}).with_columns(pl.col("B1 in 232").round(3), pl.col("B2 in 232").round(3))
print(gate1)
print(f"thresholds: B1 {thr_all[0]:.0f}, B2 {thr_all[1]:.0f}; achieved null rate B1 {rate_all[0]:.5f}, B2 {rate_all[1]:.5f}")
gate1_ok = (gate1["B1 here"] == gate1["B1 in 232"]).all() and (gate1["B2 here"] == gate1["B2 in 232"]).all()
print("Gate 1 reproduced to the digit:", gate1_ok)
assert gate1_ok

fig, ax = pf.figure(pf.ONE_COLUMN_MM, 55, layout=None)
x = np.arange(len(BINS))
for name, col232, colhere, dx in [("B1", "B1 in 232", "B1 here", -0.12), ("B2", "B2 in 232", "B2 here", 0.12)]:
    ax.plot(x + dx, 100 * gate1[col232], color=COLOR[name], marker="o", ms=7, mfc="none", mew=1, lw=0, label=f"{name}, notebook 232 (open circle)")
    ax.plot(x + dx, 100 * gate1[colhere], color=COLOR[name], marker="o", ms=2.5, lw=0, label=f"{name}, rescored here (dot)")
ax.set_xticks(x, BINS)
ax.set_xlabel("Identity of the aligned pair")
ax.set_ylabel("Recall at FPR 0.001 (%)")
ax.set_ylim(0, 105)
ax.legend(loc="lower right", fontsize=5.5)
hc.finish_figure(
    fig, FIG / "294_gate1_reproduce_nb232_pfam_seed.png", tools=hc.NO_TOOL,
    hypothesis="This notebook's code gives notebook 232's seed-10 and seed-8 recall at a null false-positive rate of 0.001.",
    conclusion=(f"All 180,000 scores equal, so every dot sits in its circle. B1 (10-letter seed): " + ", ".join(f"{b} {100 * v:.1f}%" for b, v in zip(BINS, gate1['B1 here']))
                + ". B2 (8-letter seed): " + ", ".join(f"{b} {100 * v:.1f}%" for b, v in zip(BINS, gate1['B2 here'])) + ". Higher is better."),
    title="Gate 1: notebook 232's seeds rescored, HP alphabet, Pfam seed pairs", wrap=70,
)
""")

md(r"""
## 2. The sweep, on the selection half

72 linked-pair settings (2 ways to pick keys × 4 anchor lengths × 3 window sizes × 3 gap
rules), each scored with extension. Recall at 20–30% identity at the 0.001 threshold, on a
random half of the true pairs and their own shuffled nulls. The setting with the highest
value is chosen here and only evaluated in section 4. Every square is one setting; the
chosen one is outlined. Settings whose threshold is 0 (the table's `threshold` column) never
reach an FPR of 0.001: almost no null comparison has a shared key, so any shared key counts.
Their FPR is lower than B1's and B2's, and their recall is not comparable.
""")

code(r"""
rng = np.random.default_rng(SPLIT_SEED)
perm = rng.permutation(np.unique(pidx_all))
select_ids = np.sort(perm[: perm.size // 2])
is_select = np.isin(pidx_all, select_ids)
print(f"selection half {is_select.sum():_} pairs, report half {(~is_select).sum():_} pairs")

real_s, null_s, bin_s = real_all[is_select], null_all[is_select], bin_all[is_select]
thr_s, rate_s, rec_s = lp.threshold_and_recall(real_s, null_s, bin_s, FPR)
b2030 = BINS.index("20-30%")
sweep = pl.DataFrame({
    "setting": LINKED,
    "mode": [s[0] for s in lp.ALL_SETTINGS], "anchor": [s[1] for s in lp.ALL_SETTINGS],
    "gap_max": [s[2] for s in lp.ALL_SETTINGS], "gap_rule": [s[3] for s in lp.ALL_SETTINGS],
    "threshold": [thr_s[COLS.index(f"{s}_ext")] for s in LINKED],
    "null_rate": [rate_s[COLS.index(f"{s}_ext")] for s in LINKED],
    "recall_20_30": [rec_s[b2030, COLS.index(f"{s}_ext")] for s in LINKED],
}).sort("recall_20_30", descending=True)
CHOSEN = sweep["setting"][0]
CHOSEN_RAND = sweep.filter(pl.col("mode") == "rand")["setting"][0]
base_sel = {n: rec_s[b2030, COLS.index(f"{n}_ext")] for n in ["B1", "B2"]}
print(f"selection half, 20-30% bin: B1 {base_sel['B1']:.3f}, B2 {base_sel['B2']:.3f}")
print(f"chosen: {CHOSEN}; best one-partner-per-position setting: {CHOSEN_RAND}")
print(sweep.with_columns(pl.col("recall_20_30").round(3), pl.col("null_rate").round(5)))

# heatmap: rows = anchor length x gap rule, columns = key choice x window
rows_ = [(a, t) for a in lp.ANCHOR_LENGTHS for t in lp.TREATMENTS]
cols_ = [(m, g) for m in lp.MODES for g in lp.GAP_MAXES]
H = np.array([[sweep.filter(pl.col("setting") == lp.setting_name(m, a, g, t))["recall_20_30"][0] for (m, g) in cols_] for (a, t) in rows_])
fig, ax = pf.figure(pf.ONE_COLUMN_MM, 105, layout=None)
cmap = mpl.colors.ListedColormap(plt.cm.viridis(np.linspace(0.15, 1, 256)))
im = ax.imshow(H * 100, cmap=cmap, aspect="auto", vmin=H.min() * 100, vmax=H.max() * 100)
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        v = 100 * H[i, j]
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=5, color="white" if v < (H.min() + H.max()) * 50 else "black")
for (m, a, g, t) in [s for s in lp.ALL_SETTINGS if lp.setting_name(*s) == CHOSEN]:
    ax.add_patch(Rectangle((cols_.index((m, g)) - 0.5, rows_.index((a, t)) - 0.5), 1, 1, fill=False, ec=COLOR["linked"], lw=1.6, clip_on=False, zorder=10))
ax.set_xticks(range(len(cols_)), [f"{'every pair' if m == 'all' else 'randstrobe'}\ngap ≤{g}" for m, g in cols_], fontsize=5)
ax.set_xlabel("Every anchor pair, or one partner per position (randstrobe); largest gap allowed (residues)")
ax.set_yticks(range(len(rows_)), [f"anchor {a}, {TREAT_LABEL[t]}" for a, t in rows_], fontsize=5)
for k in range(1, len(lp.ANCHOR_LENGTHS)):
    ax.axhline(k * len(lp.TREATMENTS) - 0.5, color="white", lw=1.5)
ax.axvline(len(lp.GAP_MAXES) - 0.5, color="white", lw=1.5)
cb = fig.colorbar(im, ax=ax, location="top", shrink=0.8, pad=0.02)
cb.set_label("Recall at 20–30% identity, FPR 0.001 (%), selection half")
for n, ls_ in [("B1", "-"), ("B2", "--")]:
    cb.ax.axvline(100 * base_sel[n], color="black", lw=0.8, ls=ls_)
hc.finish_figure(
    fig, FIG / "294_sweep_recall_20_30_selection_half_pfam_seed.png", tools=hc.NO_TOOL,
    hypothesis="Some linked HP word pair recovers more 20-30% identity Pfam seed pairs than a single seed at null FPR 0.001.",
    conclusion=(f"Best (outlined): {plain(CHOSEN)}, {100 * sweep['recall_20_30'][0]:.1f}%, vs B1 {100 * base_sel['B1']:.1f}% (solid line on the colour bar) and B2 {100 * base_sel['B2']:.1f}% (dashed) on this half. "
                f"Best with one key per position: {plain(CHOSEN_RAND)}, {100 * sweep.filter(pl.col('setting') == CHOSEN_RAND)['recall_20_30'][0]:.1f}%. Higher is better."),
    title="Selection half: every linked-pair setting, HP alphabet", wrap=75, header_y=0.955,
)
""")

md(r"""
## 3. What one key looks like on a real pair

One true pair from the report half at 20–30% identity, picked by a rule written before
looking: among pairs where the chosen setting scores above its threshold and B1 and B2 both
do not, keep those whose best key spans an insertion or deletion (query gap ≠ target gap),
then take the shortest alignment. Rows per block: query residues, query HP classes (H =
`ACFILMVWY`, P = `DEGHKNPQRST`), `|` where the two classes match, target HP classes, target
residues. The two boxes are the anchors of the best-scoring key; the line under them is
the gap; the black bar marks the longest run of matching HP classes, the longest exact HP
word the two share; the arrow marks the alignment column with an insertion.

The cell also lists which Pfam families the pairs found only by the linked key come from,
and how hydrophobic they are, because a key built from 6-letter HP words could favour
hydrophobic, membrane-spanning stretches.
""")

code(r"""
m_c, a_c, g_c, t_c = next(s for s in lp.ALL_SETTINGS if lp.setting_name(*s) == CHOSEN)
c_ext = COLS.index(f"{CHOSEN}_ext")
rep = ~is_select
thr_rep, _, _ = lp.threshold_and_recall(real_all[rep], null_all[rep], bin_all[rep], FPR)
cand = (rep & (bin_all == b2030) & (real_all[:, c_ext] > thr_rep[c_ext])
        & (real_all[:, COLS.index("B1_ext")] <= thr_rep[COLS.index("B1_ext")])
        & (real_all[:, COLS.index("B2_ext")] <= thr_rep[COLS.index("B2_ext")]))
print(f"report-half 20-30% pairs found by the chosen setting but by neither B1 nor B2: {cand.sum():_} "
      f"of {(rep & (bin_all == b2030)).sum():_}")
aln = (pl.read_parquet(ALN).filter(pl.col("lali") >= 50).sample(n=60_000, seed=0, shuffle=True)
       .with_row_index("pair_index").filter(pl.col("pair_index").is_in(pidx_all[cand].tolist())))
ex_rows = []
for r in aln.iter_rows(named=True):
    k = lp.best_key(r["qaln"], r["taln"], m_c, a_c, g_c, t_c)
    if k and k["gap_query"] != k["gap_target"]:
        ex_rows.append({**r, **k})
ex = (pl.DataFrame(ex_rows).with_columns(pl.col("qaln").str.len_chars().alias("aln_len"))
      .sort("aln_len", "pair_index"))
print(f"{ex.height} of them have a best key across an insertion or deletion")
all_2030 = (pl.read_parquet(ALN).filter(pl.col("lali") >= 50).sample(n=60_000, seed=0, shuffle=True)
            .with_row_index("pair_index").filter(pl.col("pair_index").is_in(pidx_all[rep & (bin_all == b2030)].tolist())))
H_SET = set("ACFILMVWY")
def h_share(aln_str):
    res = [c for c in aln_str.upper() if c != "-"]
    return sum(c in H_SET for c in res) / len(res)
all_2030 = all_2030.with_columns(
    pl.col("qaln").map_elements(h_share, return_dtype=pl.Float64).alias("query_h_share"),
    pl.col("pair_index").is_in(pidx_all[cand].tolist()).alias("found_only_by_linked"),
)
print("hydrophobic share of the query (ACFILMVWY), report-half 20-30% pairs:")
print(all_2030.group_by("found_only_by_linked").agg(pl.len().alias("n_pairs"), pl.col("query_h_share").median().round(3).alias("median"),
      (pl.col("query_h_share") >= 0.6).mean().round(3).alias("share_with_60pct_or_more_H")).sort("found_only_by_linked"))
print("Pfam families with the most pairs found only by the linked key:")
print(all_2030.filter(pl.col("found_only_by_linked")).group_by("family", "family_id").len().sort(["len", "family"], descending=[True, False]).head(10))
E = ex.row(0, named=True)
row_scores = real_all[np.flatnonzero(pidx_all == E["pair_index"])[0]]
assert E["score"] == row_scores[c_ext], "best_key disagrees with the stored score"

tab = lp.TABLES[lp.ALPHABET]
q, t = E["qaln"], E["taln"]
qc, tc = tab[np.frombuffer(q.encode(), np.uint8)], tab[np.frombuffer(t.encode(), np.uint8)]
match = (qc == tc) & (qc != lp.GAP) & (tc != lp.GAP)
run = np.diff(np.concatenate(([0], match.astype(int), [0])))
starts, ends = np.flatnonzero(run == 1), np.flatnonzero(run == -1)
lr = int(np.argmax(ends - starts)); run_s, run_e = starts[lr], ends[lr]
aligned = (np.frombuffer(q.encode(), np.uint8) != ord("-")) & (np.frombuffer(t.encode(), np.uint8) != ord("-"))
n_ident = sum(1 for x, y in zip(q, t) if x == y and x != "-")
hp = lambda c: "".join("-" if ch == "-" else ("H" if v == 0 else ("P" if v == 1 else "X")) for ch, v in zip(c, tab[np.frombuffer(c.encode(), np.uint8)]))
q_hp, t_hp = hp(q), hp(t)
mline = "".join("|" if mm else " " for mm in match)
q0 = int(E["query"].split("/")[1].split("-")[0]); t0 = int(E["target"].split("/")[1].split("-")[0])
print(f"{E['family']}: {E['query']} vs {E['target']}; {n_ident} of {int(aligned.sum())} aligned columns identical "
      f"({100 * E['seqid_ali']:.0f}%), {int(match.sum())} of {int(aligned.sum())} with matching HP classes")
print(f"best key: anchors of {a_c} at alignment columns {E['c1']} and {E['c2']}; gap {E['gap_query']} residues in the query, "
      f"{E['gap_target']} in the target; key score {E['score']:.0f} (threshold {thr_rep[c_ext]:.0f}); {E['n_keys']} shared keys")
print(f"B1 score {row_scores[COLS.index('B1_ext')]:.0f}, B2 score {row_scores[COLS.index('B2_ext')]:.0f} (threshold {thr_rep[COLS.index('B1_ext')]:.0f}); "
      f"longest shared HP run {run_e - run_s} residues")
print("Rows: query residues, query HP, | where the HP classes match, target HP, target residues; numbers are residue positions")
qi, ti = q0, t0
for s_ in range(0, len(q), 70):
    qs, ts = q[s_:s_ + 70], t[s_:s_ + 70]
    print(f"query  {qi:>4} {qs}")
    print(f"            {q_hp[s_:s_ + 70]}")
    print(f"            {mline[s_:s_ + 70]}")
    print(f"            {t_hp[s_:s_ + 70]}")
    print(f"target {ti:>4} {ts}")
    print()
    qi += sum(ch != "-" for ch in qs)
    ti += sum(ch != "-" for ch in ts)

W = 70
nblk = int(np.ceil(len(q) / W))
fig, ax = pf.figure(pf.TWO_COLUMN_MM, 16 + 22 * nblk, layout=None)
ax.set_axis_off()
ax.set_xlim(-12, W + 1); ax.set_ylim(-nblk * 7.2 + 1.4, 3.2)
mono = dict(family="monospace", fontsize=6, va="center")
def resnum(s, start, upto):
    return start + sum(1 for ch in s[:upto] if ch != "-")
ax.add_patch(Rectangle((0, 2.4), 1.2, 0.8, fill=False, ec=COLOR["linked"], lw=1.2))
ax.text(1.6, 2.8, f"anchor ({a_c} matching HP classes)", fontsize=6, va="center")
ax.plot([22, 25.5], [2.8, 2.8], color=COLOR["linked"], lw=1.2)
ax.text(26, 2.8, "gap between anchors", fontsize=6, va="center")
ax.plot([40, 43.5], [2.8, 2.8], color="black", lw=2.5)
ax.text(44, 2.8, f"longest shared HP run ({run_e - run_s})", fontsize=6, va="center")
for b in range(nblk):
    s, e = b * W, min((b + 1) * W, len(q))
    y0 = -b * 7.2
    for k_, (lab, txt) in enumerate([(f"{resnum(q, q0, s)}", q[s:e]), ("", q_hp[s:e]), ("", mline[s:e]), ("", t_hp[s:e]), (f"{resnum(t, t0, s)}", t[s:e])]):
        y = y0 - k_
        ax.text(-1, y, lab, ha="right", **{**mono, "fontsize": 5.5})
        for i, ch in enumerate(txt):
            ax.text(i + 0.5, y, ch, ha="center", **mono)
    ax.text(-12, y0, "query", fontsize=5.5, va="center"); ax.text(-12, y0 - 4, "target", fontsize=5.5, va="center")
    for cstart in (E["c1"], E["c2"]):
        lo, hi = max(cstart, s), min(cstart + a_c, e)
        if lo < hi:
            ax.add_patch(Rectangle((lo - s, y0 - 3.5), hi - lo, 3.0, fill=False, ec=COLOR["linked"], lw=1.2))
    lo, hi = max(E["c1"] + a_c, s), min(E["c2"], e)
    if lo < hi:
        ax.plot([lo - s, hi - s], [y0 - 4.9, y0 - 4.9], color=COLOR["linked"], lw=1.2)
    lo, hi = max(run_s, s), min(run_e, e)
    if lo < hi:
        ax.plot([lo - s, hi - s], [y0 - 5.4, y0 - 5.4], color="black", lw=2.5)
    for col_ in range(max(E["c1"] + a_c, s), min(E["c2"], e)):
        if q[col_] == "-" or t[col_] == "-":
            ax.annotate("insertion", xy=(col_ - s + 0.5, y0 + 0.5), xytext=(col_ - s + 0.5, y0 + 1.3), ha="center", fontsize=5.5,
                        arrowprops=dict(arrowstyle="->", lw=0.6, color="black"))
hc.finish_figure(
    fig, FIG / "294_example_linked_key_pfam_seed_pair.png", tools=hc.NO_TOOL,
    hypothesis="A key of two short anchors can add a second extended segment across an insertion, which a single seed cannot.",
    conclusion=(f"{E['family']}, {n_ident} of {int(aligned.sum())} aligned residues identical, {int(match.sum())} of {int(aligned.sum())} with the same HP class. "
                f"B1 finds a seed here (longest shared HP run {run_e - run_s}), but its extension stops at the insertion with score {row_scores[COLS.index('B1_ext')]:.0f}, below the threshold of {thr_rep[COLS.index('B1_ext')]:.0f}. "
                f"The key's anchors sit {E['gap_query']} and {E['gap_target']} residues apart, its two segments add up to {E['score']:.0f}, above its threshold of {thr_rep[c_ext]:.0f}."),
    title=f"Best key on one Pfam seed pair: {plain(CHOSEN)}", wrap=110,
)
""")

md(r"""
## 4. The verdict, on the report half

### 4a. The rule set before the run

Recall at the 0.001 threshold (232's rule: strictly above the threshold, so the null rate
is at or below 0.001), with bootstrap intervals that resample pairs one by one. The KILL
condition, "no setting beats the better of B1 and B2 outside the bootstrap interval", is
checked for all 72 settings in three readings: (1) the setting's interval lies wholly above
both baselines' intervals; (2) the setting's own value lies above them; (3) a paired
bootstrap of the setting minus the better baseline excludes 0. The readings disagree, which
is why section 4b restates the rule.
""")

code(r"""
real_r, null_r, bin_r = real_all[rep], null_all[rep], bin_all[rep]
# every linked setting is bootstrapped too: the KILL rule is about any setting, in any bin
EXT = ["B1_ext", "B2_ext"] + [f"{x}_ext" for x in LINKED] + [f"{CHOSEN}_count"]
idx = [COLS.index(c) for c in EXT]
thr_r, rate_r, rec_r = lp.threshold_and_recall(real_r[:, idx], null_r[:, :, idx], bin_r, FPR)
boot = lp.bootstrap_recall(real_r[:, idx], null_r[:, :, idx], bin_r, FPR, N_BOOT, seed=1)
lo, hi = np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)
SHOW = {"B1": "B1_ext", "B2": "B2_ext", "linked": f"{CHOSEN}_ext", "linked_count": f"{CHOSEN}_count",
        "randstrobe": f"{CHOSEN_RAND}_ext"}
J = {n: EXT.index(c) for n, c in SHOW.items()}
res = pl.DataFrame([
    {"score": n, "column": SHOW[n], "threshold": thr_r[j], "null_rate": rate_r[j], "identity_bin": b,
     "n_true_pairs": int((bin_r == i).sum()), "recall": rec_r[i, j], "ci_low": lo[i, j], "ci_high": hi[i, j]}
    for n, j in J.items() for i, b in enumerate(BINS)
])
print(res.with_columns(pl.col("recall", "ci_low", "ci_high").round(3), pl.col("null_rate").round(5)))

jB1, jB2, jL = J["B1"], J["B2"], J["linked"]
best_base = jB1 if rec_r[b2030, jB1] >= rec_r[b2030, jB2] else jB2
base_name = "B1" if best_base == jB1 else "B2"
gain = rec_r[b2030, jL] - rec_r[b2030, best_base]
pass_rule = gain >= 0.10 and lo[b2030, jL] > hi[b2030, best_base]
base_hi = np.maximum(hi[:, jB1], hi[:, jB2])  # upper end of the better baseline's interval, per bin
lin = [EXT.index(f"{x}_ext") for x in LINKED]
# reading 1 (the one PASS uses): the setting's interval lies wholly above the baselines' intervals
beats_strict = [(LINKED[k], BINS[i]) for k, j in enumerate(lin) for i in range(len(BINS)) if lo[i, j] > base_hi[i]]
# reading 2: the setting's own value lies above the baselines' intervals
beats_loose = [(LINKED[k], BINS[i]) for k, j in enumerate(lin) for i in range(len(BINS)) if rec_r[i, j] > base_hi[i]]
# reading 3: a paired bootstrap of the setting minus the better baseline (same resamples) excludes 0
better = [jB1 if rec_r[i, jB1] >= rec_r[i, jB2] else jB2 for i in range(len(BINS))]
beats_paired = [(LINKED[k], BINS[i]) for k, j in enumerate(lin) for i in range(len(BINS))
                if np.percentile(boot[:, i, j] - boot[:, i, better[i]], 2.5) > 0]
print("All 72 linked settings, report half, against the better of B1 and B2 in each identity bin:")
print(f"  setting-bin cases whose interval lies wholly above both baselines' intervals: {len(beats_strict)}")
print(f"  setting-bin cases whose value lies above both baselines' intervals: {len(beats_loose)}")
print(f"  setting-bin cases whose paired gain over the better baseline excludes 0: {len(beats_paired)}")
cases = (pl.DataFrame(beats_loose, schema=["setting", "identity_bin"], orient="row").with_columns(pl.lit("value above both intervals").alias("reading"))
         .vstack(pl.DataFrame(beats_paired, schema=["setting", "identity_bin"], orient="row").with_columns(pl.lit("paired gain excludes 0").alias("reading"))))
print(cases.sort("reading", "setting", "identity_bin"))
if pass_rule:
    VERDICT_PRE = "PASS"
elif not (beats_strict or beats_loose or beats_paired):
    VERDICT_PRE = "KILL"
else:
    VERDICT_PRE = "NEITHER PASS NOR KILL"
diff_boot = boot[:, b2030, jL] - boot[:, b2030, best_base]
d_lo, d_hi = np.percentile(diff_boot, [2.5, 97.5])
verdict_pre_line = (f"Rule set before the run: {VERDICT_PRE}. At 20-30% identity the chosen setting finds {100 * rec_r[b2030, jL]:.1f}% "
                f"[{100 * lo[b2030, jL]:.1f}-{100 * hi[b2030, jL]:.1f}] of true pairs, {base_name} {100 * rec_r[b2030, best_base]:.1f}% "
                f"[{100 * lo[b2030, best_base]:.1f}-{100 * hi[b2030, best_base]:.1f}]: +{100 * gain:.1f} points, +10 needed. "
                f"KILL needs no setting to beat the baselines in any bin; {len(beats_paired)} setting-bin cases do by a paired bootstrap. Report, and leave the index as it is.")
print(verdict_pre_line)
print(f"Only the strictest reading (one interval wholly above another) would call it KILL; the other two do not. Not PASS under any reading.")
print(f"Gain of the chosen setting over {base_name} at 20-30%, paired bootstrap (same resamples): {100 * gain:.1f} points, 95% interval {100 * d_lo:.1f} to {100 * d_hi:.1f}")
names = list(J)
jR = J["randstrobe"]
r_lo, r_hi = np.percentile(boot[:, b2030, jR] - boot[:, b2030, best_base], [2.5, 97.5])
print(f"Randstrobe setting over {base_name} at 20-30%, paired: {100 * (rec_r[b2030, jR] - rec_r[b2030, best_base]):.1f} points, 95% interval {100 * r_lo:.1f} to {100 * r_hi:.1f}")
# why the single-setting intervals are wide: recall at 20-30% one score step either side of the threshold
for n in ["B2", "linked"]:
    j = J[n]
    col = real_r[:, idx[j]][bin_r == b2030]
    nul = null_r[:, :, idx[j]].ravel()
    print(f"{n}: " + "; ".join(f"threshold {t:.0f}: recall {100 * (col > t).mean():.1f}%, null rate {(nul > t).mean():.5f}"
                               for t in (thr_r[j] - 1, thr_r[j], thr_r[j] + 1)))

# the three readings at 20-30% identity: each score's interval, and the paired gain
fig, axes = pf.figure(pf.ONE_COLUMN_MM, 50, ncols=2, layout=None, gridspec_kw={"width_ratios": [1.3, 1]})
ax = axes[0]
for y, (n, mk, filled) in enumerate([("linked", "D", True), ("B2", "s", True), ("B1", "o", True)]):
    j = J[n]
    ax.errorbar(100 * rec_r[b2030, j], y, xerr=[[100 * (rec_r[b2030, j] - lo[b2030, j])], [100 * (hi[b2030, j] - rec_r[b2030, j])]],
                color=COLOR[n], marker=mk, ms=4, lw=1.2, capsize=2)
ax.set_yticks(range(3), ["chosen linked pair", "B2: 8-letter seed", "B1: 10-letter seed"], fontsize=5.5)
ax.set_xlabel("Recall at 20–30% identity (%)\n232's threshold; pairs resampled")
ax.set_title("Readings 1-2: the intervals overlap", fontsize=6)
ax = axes[1]
ax.errorbar(100 * gain, 0, xerr=[[100 * (gain - d_lo)], [100 * (d_hi - gain)]], color=COLOR["linked"], marker="D", ms=4, lw=1.2, capsize=2)
ax.axvline(0, color="0.6", lw=0.5)
ax.set_yticks([])
ax.set_xlim(-2, 12)
ax.set_xlabel("Paired gain over B2 (points)")
ax.set_title("Reading 3: the gain excludes 0", fontsize=6)
hc.finish_figure(
    fig, FIG / "294_rule_before_run_three_readings_pfam_seed.png", tools=hc.NO_TOOL,
    hypothesis="The rule set before the run gives one answer.",
    conclusion=verdict_pre_line,
    title="Rule set before the run: three readings at 20-30% identity", wrap=70,
)
""")

md(r"""
### 4b. The rule set after the first run (2026-10-15)

This rule was written on 2026-10-15, after the numbers in 4a had been seen, because the
rule above could be read three ways. Its result is not a pre-registered test; it says
what the data already collected can and cannot rule out.

- *One number*: the gain in recall at 20–30% identity of the chosen setting over the better
  single seed (B1 or B2, whichever is higher on this half), from the same resamples
  (a paired bootstrap).
- *At exactly FPR 0.001*: scores are whole numbers, so 232's threshold calls fewer than
  0.1% of null comparisons, by a different amount for each score. Here recall is read off
  the line between the two thresholds either side of 0.001, which is what breaking ties at
  random to call exactly 0.1% of the null gives.
- *Resampled by family*: pairs from one Pfam family share sequence, so the bootstrap draws
  whole families (with all their pairs and shuffled targets), 1,000 times.
- *Cut-off*: 10 percentage points, the gain the original rule asked for.
  PASS if the whole 95% interval is at or above 10. KILL if the whole interval is below
  10: a gain worth an index change is ruled out. Unclear if it crosses 10: more pairs are
  needed.
- *B3 control*: B2's 8-letter seed plus the best second gap-free segment that contains any
  matching column within 32 residues of the seed, scored as up to two segments exactly like
  a linked key. If the linked key does no better than B3, the gain comes from scoring a
  second segment, not from the second exact word.

The recall figure shows every score with family-resampled intervals; the decision figure
shows the paired gains against the 10-point line.
""")

code(r"""
fam_codes = (d.filter(pl.col("kind") == "real").sort("pair_index")
             .select(pl.col("family").cast(pl.Categorical).to_physical()).to_series().to_numpy())
assert (np.sort(pidx_all) == pidx_all).all()  # score_arrays sorts by pair_index too
fam_r = fam_codes[rep]
NEW = {"B1": "B1_ext", "B2": "B2_ext", "B3": "B3_ext", "linked": f"{CHOSEN}_ext",
       "linked_count": f"{CHOSEN}_count", "randstrobe": f"{CHOSEN_RAND}_ext"}
JN = {n: i for i, n in enumerate(NEW)}
idx_n = [COLS.index(c) for c in NEW.values()]
rr, nr = real_r[:, idx_n], null_r[:, :, idx_n]
thr_n, w_n, rec_n = lp.recall_at_exact_fpr(rr, nr, bin_r, FPR)
boot_n = lp.bootstrap_recall_by_group(rr, nr, bin_r, fam_r, FPR, N_BOOT, seed=2)
lo_n, hi_n = np.percentile(boot_n, 2.5, axis=0), np.percentile(boot_n, 97.5, axis=0)
print(f"report half: {rr.shape[0]:_} true pairs in {np.unique(fam_r).size:_} Pfam families "
      f"(largest family {np.bincount(fam_r).max()} pairs)")
res_n = pl.DataFrame([
    {"score": n, "column": NEW[n], "threshold": thr_n[j], "share_of_ties_called": w_n[j], "identity_bin": b,
     "n_true_pairs": int((bin_r == i).sum()), "recall_exact_fpr": rec_n[i, j], "ci_low": lo_n[i, j], "ci_high": hi_n[i, j]}
    for n, j in JN.items() for i, b in enumerate(BINS)
])
print("Recall at exactly FPR 0.001, family-resampled 95% intervals:")
print(res_n.with_columns(pl.col("recall_exact_fpr", "ci_low", "ci_high", "share_of_ties_called").round(3)))

DELTA = 0.10
best_n = "B1" if rec_n[b2030, JN["B1"]] >= rec_n[b2030, JN["B2"]] else "B2"
def paired(a, b):
    g = rec_n[b2030, JN[a]] - rec_n[b2030, JN[b]]
    lo_, hi_ = np.percentile(boot_n[:, b2030, JN[a]] - boot_n[:, b2030, JN[b]], [2.5, 97.5])
    return g, lo_, hi_
def call(lo_, hi_):
    return "PASS" if lo_ >= DELTA else ("KILL" if hi_ < DELTA else "UNCLEAR")
COMPARE = [("linked", best_n, "chosen linked pair minus the better single seed", True),
           ("randstrobe", best_n, "randstrobe (one key per position) minus the better single seed", True),
           ("linked", "B3", "chosen linked pair minus B3 (seed + second segment)", False),
           ("B3", best_n, "B3 minus the better single seed", False)]
gains = pl.DataFrame([{"comparison": lab, "gain": g, "ci_low": l, "ci_high": h,
                       "call_vs_10_points": call(l, h) if decides else ("control: interval excludes 0" if l > 0 or h < 0 else "control: interval includes 0")}
                      for a_, b_, lab, decides in COMPARE for g, l, h in [paired(a_, b_)]])
print(f"Paired gains at 20-30% identity, exact FPR 0.001, family-resampled (better single seed here: {best_n}):")
print(gains.with_columns((pl.col("gain", "ci_low", "ci_high") * 100).round(1)))
VERDICT = gains["call_vs_10_points"][0]
g0, l0, h0 = gains.row(0)[1:4]
gb3, lb3, hb3 = gains.row(2)[1:4]
verdict_line = (f"Verdict (rule set after the first run): {VERDICT}. At 20-30% identity and exactly FPR 0.001 the chosen setting "
                f"finds {100 * g0:+.1f} points more true pairs than {best_n} (95% interval {100 * l0:+.1f} to {100 * h0:+.1f}, families resampled); "
                f"the whole interval is {'below' if h0 < DELTA else 'not below'} the 10-point cut-off. "
                f"Against B3, the same 8-letter seed plus a second segment with no second exact word, the gain is {100 * gb3:+.1f} ({100 * lb3:+.1f} to {100 * hb3:+.1f}).")
print(verdict_line)

# recall by identity bin
fig, ax = pf.figure(pf.ONE_COLUMN_MM, 76, layout=None)
x = np.arange(len(BINS))
style = {"B1": ("B1: one 10-letter seed + extension", "o", True, "B1"),
         "B2": ("B2: one 8-letter seed + extension", "s", True, "B2"),
         "B3": ("B3: 8-letter seed + a second segment", "s", False, "B2"),
         "linked": ("chosen linked pair + extension", "D", True, "linked"),
         "linked_count": ("chosen linked pair, keys counted, no extension", "D", False, "linked_count"),
         "randstrobe": ("one partner per position (randstrobe) + extension", "^", False, "linked")}
off = dict(zip(style, np.linspace(-0.25, 0.25, len(style))))
for n, (lab, mk, filled, ckey) in style.items():
    j = JN[n]
    ax.errorbar(x + off[n], 100 * rec_n[:, j], yerr=[100 * (rec_n[:, j] - lo_n[:, j]), 100 * (hi_n[:, j] - rec_n[:, j])],
                color=COLOR[ckey], marker=mk, ms=3, mfc=COLOR[ckey] if filled else "white", lw=0.8, capsize=1.5, label=lab)
ax.set_xticks(x, BINS)
ax.set_xlabel("Identity of the aligned pair")
ax.set_ylabel("Recall at exactly FPR 0.001 (%)\n(bars: 95% interval, families resampled)")
ax.set_ylim(0, 102)
ax.legend(loc="upper left", fontsize=4.8, title=f"Chosen: {plain(CHOSEN)}.\nTriangles: the same anchors and gap, one partner per position.", title_fontsize=4.8, alignment="left")
hc.finish_figure(
    fig, FIG / "294_recall_by_identity_report_half_pfam_seed.png", tools=hc.NO_TOOL,
    hypothesis="The chosen linked HP word pair finds more 20-30% identity Pfam seed pairs than a single seed, at exactly FPR 0.001 per comparison.",
    conclusion=(f"20-30% identity: chosen {100 * rec_n[b2030, JN['linked']]:.1f}%, B3 {100 * rec_n[b2030, JN['B3']]:.1f}%, "
                f"B2 {100 * rec_n[b2030, JN['B2']]:.1f}%, B1 {100 * rec_n[b2030, JN['B1']]:.1f}%, randstrobe {100 * rec_n[b2030, JN['randstrobe']]:.1f}%. Higher is better."),
    title="Report half: recall by identity, HP alphabet, exactly FPR 0.001", wrap=70,
)
""")

code(r"""
# decision figure: paired gains against the 10-point line
fig, ax = pf.figure(pf.ONE_COLUMN_MM, 58, layout=None)
ys = np.arange(len(COMPARE))[::-1]
for y, (row, (a_, b_, lab, decides)) in zip(ys, zip(gains.iter_rows(named=True), COMPARE)):
    col = COLOR["linked"] if a_ != "B3" else COLOR["B2"]
    mk = {"linked": "D", "randstrobe": "^", "B3": "s"}[a_]
    filled = a_ == "linked"
    ax.errorbar(100 * row["gain"], y, xerr=[[100 * (row["gain"] - row["ci_low"])], [100 * (row["ci_high"] - row["gain"])]],
                color=col, marker=mk, ms=4, mfc=col if filled else "white", lw=1.2, capsize=2)
    ax.text(-14.5, y + 0.28, f"{lab}: {row['call_vs_10_points']}", fontsize=5, va="bottom")
ax.axvline(100 * DELTA, color="black", ls="--", lw=0.8)
ax.axvline(0, color="0.6", lw=0.5)
ax.text(100 * DELTA + 0.3, ys[0] + 0.75, "10-point cut-off", fontsize=5, va="center")
ax.set_xlim(-15, 20)
ax.set_ylim(-0.6, len(COMPARE) - 0.1)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("Paired gain in recall at 20–30% identity, exactly FPR 0.001 (points)\nbars: 95% interval, families resampled")
hc.finish_figure(
    fig, FIG / "294_decision_paired_gain_vs_10_points_pfam_seed.png", tools=hc.NO_TOOL,
    hypothesis="A linked HP key gains at least 10 points of recall over a single seed (rule set after the first run).",
    conclusion=verdict_line + " An interval wholly left of the dashed line is a KILL; wholly right, a PASS; across it, unclear.",
    title="Decision: paired gains against the 10-point cut-off", wrap=58,
)
""")

md(r"""
### 4c. Against phmmer and MMseqs2 on the same pairs

[Notebook 233](233_extended_kmerseek_vs_aligners_on_pairs.ipynb) ran phmmer 3.4 (each pair
on its own, `-Z 20000`, so the E-value is what a proteome-sized database would give) and
MMseqs2 18 (`-s 7.5`, all 116,176 domain sequences against themselves) on these same 60,000
Pfam pairs. A pair is found by an aligner at E ≤ 0.01. The HP scores here use 232's threshold
(null rate at or below 0.001 per comparison), not the exact-FPR read-out of section 4b, so
they sit up to 2.4 points below the values there. The comparison favours the HP scores twice.
First, the HP scores are computed on Pfam's own curated alignment of each pair, while
phmmer and MMseqs2 have to find the alignment themselves. Second, the aligners' E-values
account for a whole database, the HP threshold only for one pair's own alignment, so the
HP threshold is the more lenient one. The two aligners are not held to the same cut-off
either: phmmer's E-values assume 20,000 sequences, MMseqs2's are computed against 116,176,
so MMseqs2's E ≤ 0.01 is about 5.8 times stricter. FoldSeek needs structures and was not
run on these Pfam sequences, so this notebook has no FoldSeek comparison.
""")

code(r"""
AL = Path("/Users/olga/data/pfam/233_aligner_check/233_aligner_pair_evalues.parquet")
E_CUT = 0.01
al = pl.read_parquet(AL).filter(pl.col("dataset") == "pfam").select("query", "target", "phmmer_evalue", "mmseqs_evalue")
cmp_cols = {"B2": "B2_ext", "B3": "B3_ext", "linked": f"{CHOSEN}_ext", "randstrobe": f"{CHOSEN_RAND}_ext"}
rep_pairs = d.filter((pl.col("kind") == "real") & pl.col("pair_index").is_in(pidx_all[rep].tolist()))
thr_c = {n: thr_r[EXT.index(c)] if c in EXT else None for n, c in cmp_cols.items()}
thr_c["B3"] = thr_n[JN["B3"]]
found = rep_pairs.join(al, on=["query", "target"], how="left").with_columns(
    *[(pl.col(c) > thr_c[n]).alias(n) for n, c in cmp_cols.items()],
    (pl.col("phmmer_evalue") <= E_CUT).fill_null(False).alias("phmmer"),
    (pl.col("mmseqs_evalue") <= E_CUT).fill_null(False).alias("MMseqs2"),
).with_columns((pl.col("phmmer") | pl.col("MMseqs2")).alias("phmmer or MMseqs2"))
SHOWN = ["B2", "B3", "linked", "randstrobe", "phmmer", "MMseqs2"]
tab_c = found.group_by("identity_bin").agg(pl.len().alias("n_pairs"), *[(100 * pl.col(c).mean()).round(1).alias(c) for c in SHOWN]).sort("identity_bin")
print(f"report half: {found.height:_} true pairs; aligner rows missing (no hit at E <= 10, counted as not found): "
      f"phmmer {found['phmmer_evalue'].is_null().sum():_}, MMseqs2 {found['mmseqs_evalue'].is_null().sum():_}")
print("Share of true pairs found (%), HP scores at 232's threshold, aligners at E <= 0.01:")
print(tab_c)
b23 = found.filter(pl.col("identity_bin") == "20-30%")
only = pl.DataFrame([{"HP score": n, "found": int(b23[n].sum()),
                      "found by neither phmmer nor MMseqs2": int((b23[n] & ~b23["phmmer or MMseqs2"]).sum()),
                      "found, MMseqs2 not": int((b23[n] & ~b23["MMseqs2"]).sum())} for n in ["B2", "B3", "linked", "randstrobe"]])
print(f"20-30% identity, {b23.height:_} pairs:")
print(only)

fig, ax = pf.figure(pf.ONE_COLUMN_MM, 70, layout=None)
x = np.arange(len(BINS))
sty = {"B2": (COLOR["B2"], "s", True), "B3": (COLOR["B2"], "s", False), "linked": (COLOR["linked"], "D", True),
       "randstrobe": (COLOR["linked"], "^", False), "phmmer": ("#555555", "v", True), "MMseqs2": ("#555555", "v", False)}
lab = {"B2": "B2: 8-letter HP seed", "B3": "B3: HP seed + second segment", "linked": "chosen linked HP pair",
       "randstrobe": "randstrobe HP pair", "phmmer": "phmmer (E <= 0.01)", "MMseqs2": "MMseqs2 (E <= 0.01)"}
for n, (col, mk, filled) in sty.items():
    ax.plot(x, tab_c[n], color=col, marker=mk, ms=3.5, mfc=col if filled else "white", lw=0.8, label=lab[n])
ax.set_xticks(x, BINS)
ax.set_xlabel("Identity of the aligned pair")
ax.set_ylabel("True pairs found (%)")
ax.set_ylim(0, 102)
ax.legend(loc="upper left", fontsize=5)
lk = only.filter(pl.col("HP score") == "linked").row(0, named=True)
hc.finish_figure(
    fig, FIG / "294_hp_keys_vs_phmmer_mmseqs2_pfam_seed.png",
    tools="phmmer 3.4, MMseqs2 18 (notebook 233); HP scores computed from alignments, no search",
    hypothesis="HP keys find remote Pfam pairs the sequence aligners miss.",
    conclusion=(f"At 20-30% identity: phmmer {tab_c.filter(pl.col('identity_bin') == '20-30%')['phmmer'][0]:.1f}%, MMseqs2 {tab_c.filter(pl.col('identity_bin') == '20-30%')['MMseqs2'][0]:.1f}%, "
                f"linked HP pair {tab_c.filter(pl.col('identity_bin') == '20-30%')['linked'][0]:.1f}%, B3 {tab_c.filter(pl.col('identity_bin') == '20-30%')['B3'][0]:.1f}%. "
                f"Of the {lk['found']:,} pairs the linked key finds there, {lk['found by neither phmmer nor MMseqs2']} are found by neither aligner. HP scores at 232's threshold, on Pfam's own alignment of each pair; both make the HP side the more lenient. Higher is better."),
    title="HP keys against phmmer and MMseqs2, report half, Pfam seed pairs", wrap=75,
)
""")

md(r"""
## 5. Bits, measured on Swiss-Prot

How often does a key match by chance? For each setting, every key in Swiss-Prot
(`uniprot_sprot.fasta.gz`, 572,970 proteins, encoded in `hp_thomas_dill2`) is counted, and
the chance that two keys drawn at random are equal is the sum over keys of p(key)². Its
−log2 is the measured number of bits. The prediction drawn as a line assumes one bit per HP
letter, plus log2 of the number of gap values the key stores. The y-axis is the expected
number of chance key matches between two unrelated sequences of 131 residues (the median
domain length of the pairs above), over every pair of positions, not one diagonal.

Key frequencies are counted directly from Swiss-Prot. The project's stored Swiss-Prot
spectra (`~/data/kmerseek-kmer-spectra/sprot.hp_thomas_dill.k*.spectrum.csv.gz`) start at
15 letters and list 72,533 distinct HP 15-mers, more than the 32,768 two letters allow, so
they use a different encoding. Only keys that fit inside 131 residues are counted.
""")

code(r"""
spec = pl.read_parquet(SPECTRUM)
check = spec.filter(pl.col("mode") == "check")
spec = spec.filter(pl.col("mode") != "check").with_columns(
    (pl.col("keys_per_sequence") ** 2 * (2.0 ** -pl.col("predicted_bits"))).alias("predicted_chance_matches"),
    (pl.col("predicted_bits") - pl.col("measured_bits")).alias("bits_short_of_prediction"),
)
L_MED = spec["median_length"][0]
print(f"valid HP 15-mers counted in Swiss-Prot: {int(check['rate'][0]):_}; median pair length {L_MED:.0f}")
words = spec.filter(pl.col("mode") == "word").select("anchor", "predicted_bits", "measured_bits", "expected_chance_matches")
print("single words:"); print(words.with_columns(pl.col("measured_bits", "expected_chance_matches").round(2)))
tabl = spec.filter(pl.col("mode") != "word").select("mode", "anchor", "gap_max", "treatment", "keys_per_sequence", "predicted_bits", "measured_bits",
                                                    "bits_short_of_prediction", "expected_chance_matches").sort("mode", "gap_max", "treatment", "anchor")
print(tabl.with_columns(pl.col("keys_per_sequence", "predicted_bits", "measured_bits", "bits_short_of_prediction").round(2), pl.col("expected_chance_matches").round(3)))
ch = spec.filter((pl.col("mode") == m_c) & (pl.col("anchor") == a_c) & (pl.col("gap_max") == g_c) & (pl.col("treatment") == t_c)).row(0, named=True)
w10 = words.filter(pl.col("anchor") == 10).row(0, named=True)
w8 = words.filter(pl.col("anchor") == 8).row(0, named=True)
print(f"chosen {CHOSEN}: {ch['measured_bits']:.1f} bits per key measured (prediction {ch['predicted_bits']:.1f}); "
      f"{ch['expected_chance_matches']:.1f} chance key matches per comparison of two {L_MED:.0f}-residue sequences; "
      f"B1's 10-mer {w10['measured_bits']:.1f} bits, {w10['expected_chance_matches']:.1f}; B2's 8-mer {w8['measured_bits']:.1f} bits, {w8['expected_chance_matches']:.1f}")

fig, axd = plt.subplot_mosaic([["legend", "legend"], ["all", "rand"]], figsize=(pf.TWO_COLUMN_MM * pf.MM, 85 * pf.MM),
                              height_ratios=[0.28, 1], sharey=False)
from matplotlib.lines import Line2D
handles = ([Line2D([], [], color=COLOR[tr], marker="o", lw=0, ms=4, label=TREAT_LABEL[tr]) for tr in lp.TREATMENTS]
           + [Line2D([], [], color="0.4", marker=mk, lw=0, ms=4, label=f"gap up to {g} residues") for g, mk in zip(lp.GAP_MAXES, ["o", "s", "^"])]
           + [Line2D([], [], color="0.4", lw=0.6, label="prediction for gap up to 32 (same colour as its triangles):\n1 bit per HP letter, plus the stored gap")]
           + [Line2D([], [], color=COLOR[n], marker="*", ms=7, lw=0, label=f"{n}: one {a_}-letter word (single seed)") for n, a_ in [("B1", 10), ("B2", 8)]]
           + [Line2D([], [], marker="o", ms=8, mfc="none", mec=COLOR["linked"], mew=1.2, lw=0, label="chosen linked pair")])
axd["legend"].set_axis_off()
axd["legend"].legend(handles=handles, loc="center", ncol=4, fontsize=5.5, frameon=False)
for mode in lp.MODES:
    ax = axd[mode]
    for tr in lp.TREATMENTS:
        for g, mk in zip(lp.GAP_MAXES, ["o", "s", "^"]):
            s_ = spec.filter((pl.col("mode") == mode) & (pl.col("treatment") == tr) & (pl.col("gap_max") == g)).sort("anchor")
            if g == max(lp.GAP_MAXES):
                ax.plot(s_["anchor"], s_["predicted_chance_matches"], color=COLOR[tr], lw=0.6, alpha=0.6)
            ax.plot(s_["anchor"], s_["expected_chance_matches"], color=COLOR[tr], marker=mk, ms=3, lw=0)
    for name, a_ in [("B1", 10), ("B2", 8)]:
        r_ = words.filter(pl.col("anchor") == a_)
        ax.plot(r_["anchor"], r_["expected_chance_matches"], marker="*", ms=6, lw=0, color=COLOR[name])
    if mode == m_c:
        ax.plot([a_c], [ch["expected_chance_matches"]], marker="o", ms=8, mfc="none", mec=COLOR["linked"], mew=1.2, lw=0)
    ax.set_yscale("log")
    ax.set_ylim(2e-4, 2e4)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xticks(lp.ANCHOR_LENGTHS)
    ax.set_xlabel("Anchor length (HP letters)")
    ax.set_title(MODE_LABEL[mode], fontsize=6.5)
axd["all"].set_ylabel("Expected chance key matches between two\nunrelated 131-residue proteins\n(key frequencies from Swiss-Prot)")
axd["rand"].set_yticklabels([])
fig.text(0.5, -0.02, "Points: measured on Swiss-Prot. Lines: prediction. Each step down the y-axis is ten times fewer chance matches.", ha="center", fontsize=5.5)
hc.finish_figure(
    fig, FIG / "294_chance_key_matches_swissprot.png", tools=hc.NO_TOOL,
    title="Chance key matches: HP keys counted on Swiss-Prot",
    hypothesis="A key of two HP anchors of a letters each carries about 2a bits, plus the bits of any stored gap, so it matches by chance about as rarely as one word of that many letters.",
    conclusion=(f"Every key carries fewer bits than predicted (points above their lines); randstrobe keys lose 3 to 5 bits, because the hash rule ties the second anchor to the first. "
                f"Chosen setting: {ch['measured_bits']:.1f} bits per key ({ch['predicted_bits']:.0f} predicted) and {ch['expected_chance_matches']:,.0f} chance key matches per pair of {L_MED:.0f}-residue sequences, "
                f"against {w10['expected_chance_matches']:.0f} for B1's 10-mer. Lower is better."),
    footer_y=-0.06, wrap=140,
)
""")

md(r"""
## 6. Summary and what this does not show
""")

code(r"""
print(verdict_pre_line)
print(verdict_line)
print()
print("Report half, recall at exactly FPR 0.001 (95% interval, families resampled):")
for n in NEW:
    j = JN[n]
    print(f"  {n:13s} {NEW[n]:28s} " + "  ".join(f"{b} {100 * rec_n[i, j]:.1f} [{100 * lo_n[i, j]:.1f}-{100 * hi_n[i, j]:.1f}]" for i, b in enumerate(BINS)))
print()
import gzip
with gzip.open(SPROT_FASTA, "rb") as fh:
    n_sprot = sum(1 for ln in fh if ln.startswith(b">"))
print(f"Swiss-Prot proteins in {SPROT_FASTA.name}: {n_sprot:_}")
print(f"False pairs per query at 0.001 per comparison against Swiss-Prot: {FPR * n_sprot:.0f}")
""")

md(r"""
Every number below is printed by a cell above. Recall: higher is better. Chance key
matches: lower is better.

1. **Gate 1 passed.** This notebook's code gives notebook 232's score on all 180,000
   comparisons and its seed-10 recall in every identity bin: 11.9%, 35.5%, 63.5%, 88.2% and
   98.6%.
2. **By the rule set before the run: neither PASS nor KILL** (section 4a). The chosen
   setting (every pair of 6-letter HP anchors up to 32 residues apart, gap not stored)
   gains 5.6 points over the 8-letter seed at 20–30% identity, short of 10, but the KILL
   condition holds under only one of its three readings.
3. **By the rule set after the run: KILL** (section 4b; set on 2026-10-15 after seeing
   section 4a, so not a pre-registered test). At exactly FPR 0.001 with families
   resampled, the gain is 6.7 points (95% interval 3.2 to 8.9): a gain of 10 points is
   ruled out. The one-key-per-position version an index can afford gains 2.4 (−0.7 to 6.4).
4. **The second exact word adds nothing.** B3, the same 8-letter seed plus the best second
   segment within 32 residues, finds 47.6% at 20–30% identity against the linked key's
   45.9% (linked minus B3: −1.7, interval −3.0 to 0.5). B3 gains 8.4 points over the plain
   8-letter seed (4.0 to 10.9); that interval crosses 10, so by the same rule B3 is
   unclear, not shown to be worth 10 points either. So what helps is extending into a
   second segment across an insertion, which needs no change to the index. kmerseek already chains matches across
   small shifts (`--chain-max-shift`); whether that gives the same gain was not tested here.
5. **Against phmmer and MMseqs2** (section 4c, notebook 233's runs on the same pairs). At
   20–30% identity phmmer finds 75.4% and MMseqs2 55.2%, against 43.5% for the linked HP key
   and 46.2% for B3. These HP values are at 232's threshold, so they are lower than the
   exact-FPR values in item 4. The HP scores also get Pfam's alignment for free and the more
   lenient threshold, so the comparison favours them. Of the 3,266 pairs the linked key
   finds there, 139 are found by neither aligner. Under 20% identity the
   HP scores (15.3% linked, 17.1% B3) sit above MMseqs2 (8.4%) but below phmmer (18.8%);
   that bin is defined by identical residues, which by construction favours an alphabet that
   ignores them, and the HP threshold is the lenient one. phmmer's E-values assume a
   database of 20,000 sequences and MMseqs2's one of 116,176, so part of the gap between
   the two aligners may come from that. FoldSeek was not run on these sequences.
6. **The cost.** The chosen key gives about 3,896 chance key matches between two unrelated
   131-residue proteins, against 70 for B2's 8-letter word and 18 for B1's 10-letter word
   (section 5), 55 times B2's.
7. **Bits, measured on Swiss-Prot.** Every HP key carries fewer bits than one per letter.
   Keys built from every anchor pair fall 0.5 to 1.6 bits short of that prediction.
   Randstrobe keys fall 2.8 to 5.2 bits short, because the hash rule ties the second anchor
   to the first.
8. **Not a membrane effect.** The 709 report-half pairs at 20–30% identity found only by the
   linked key are no more hydrophobic than the other 6,803 (median 43% of query residues in
   `ACFILMVWY` in both), and no family has more than 2 of them.
9. **Not shown here.** All numbers are per comparison, on each pair's own alignment, against
   a null that shuffles residues. That null breaks local HP patterns such as amphipathic
   helices and membrane runs, which unrelated real proteins keep, so it may be too easy for
   an HP key. A false-positive rate of 0.001 per comparison is about 573 false pairs per
   query against Swiss-Prot's 572,970 proteins (counted in section 6). Not tested:
   database false positives, chance key matches off the alignment's diagonal, SCOPe pairs, other alphabets and mismatch
   scores, a full-alphabet BLAST ceiling on the same pairs, and three anchors.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parents[1] / "notebooks" / "294_linked_hp_word_pairs_kill_test_pfam_seed.ipynb"
out.write_text(json.dumps(nb, indent=1) + "\n")
print(out)
