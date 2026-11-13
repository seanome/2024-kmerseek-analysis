#!/usr/bin/env python3
"""Generate notebooks/354_kill_test_fragment_source_massspec18_PXD005280.ipynb.

The first cell is the pre-registration in scripts/make_nb354_prereg.py, unchanged. The
notebook reads only the tables `make pull` copies into notebooks/354_data/ and the frozen
inputs in data/kill_tests/fragment_source/.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_nb354_prereg import PREREGISTRATION  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "354_kill_test_fragment_source_massspec18_PXD005280.ipynb"
SUMMARY = (ROOT / "scripts" / "nb354_summary.md").read_text() if (ROOT / "scripts" / "nb354_summary.md").exists() else "## Summary\n\nNot written yet.\n"
REVIEW = (ROOT / "scripts" / "nb354_review.md").read_text() if (ROOT / "scripts" / "nb354_review.md").exists() else "## Second reader\n\nNot run yet.\n"

cells = []


def md(source):
    cells.append({"cell_type": "markdown", "id": f"md-{len(cells):02d}", "metadata": {},
                  "source": source.strip().splitlines(keepends=True)})


def code(source):
    cells.append({"cell_type": "code", "id": f"code-{len(cells):02d}", "execution_count": None,
                  "metadata": {"jupyter": {"source_hidden": True}}, "outputs": [],
                  "source": source.strip("\n").splitlines(keepends=True)})


md(SUMMARY)
md(PREREGISTRATION)

code(r'''
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import pubfig as pf

pf.use_style()
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_width_chars(200)
pl.Config.set_fmt_str_lengths(80)

FROZEN = Path("../data/kill_tests/fragment_source")
DATA = Path("354_data")
RES = DATA / "results"
FIG = Path("../figures")

calls = pl.read_csv(FROZEN / "calls.tsv", separator="\t", infer_schema_length=0).with_columns(
    pl.col("length").cast(pl.Int64), pl.col("alc_percent").cast(pl.Float64),
    (pl.col("source_absent") == "True").alias("source_absent"),
    (pl.col("denovo_equals_db_search") == "True").alias("denovo_equals_db_search"))
freeze = json.loads((FROZEN / "freeze_summary.json").read_text())
choice = json.loads((DATA / "dev_choice.json").read_text())
dev = pl.read_csv(DATA / "dev_table.tsv", separator="\t")
m = json.loads((RES / "metrics.json").read_text())
curve = pl.read_csv(RES / "curve.tsv", separator="\t")
per_call = pl.read_csv(RES / "per_call.tsv", separator="\t", infer_schema_length=10_000)
runtime = pl.read_csv(RES / "runtime.tsv", separator="\t")
aln = pl.read_csv(RES / "alignments.tsv", separator="\t", infer_schema_length=10_000)
compute = pl.read_csv(RES / "compute.tsv", separator="\t")

KMERSEEK = [f"{s['alphabet']}_k{s['k']}" for s in choice["kmerseek_settings"]]
METHODS = KMERSEEK + ["comparison"]
LABEL = {"comparison": "C: exact index + blastp-short"}
for s in choice["kmerseek_settings"]:
    LABEL[f"{s['alphabet']}_k{s['k']}"] = f"K: kmerseek {s['alphabet']}, k = {s['k']}"
COLOUR = {"comparison": pf.OKABE_ITO["vermillion"]}
for name, c in zip(KMERSEEK, [pf.OKABE_ITO["blue"], pf.OKABE_ITO["bluish_green"]]):
    COLOUR[name] = c
TOOLS = ("Queries: 250 PEAKS de novo calls, PXD005280 HeLa run 1 (test split). Target: UniProt "
         "human reference proteome 2026_03 + cRAP, 20,734 proteins. Check: Smith-Waterman, PAM30.")
print(json.dumps(freeze, indent=1))
''')

md(r"""
## 1. The frozen calls

What the 500 calls look like before any method sees them: length, PEAKS's average local
confidence (ALC), and whether the de novo sequence equals the database-search peptide of
the same spectrum (I read as L, Q as K). A call that differs is a disagreement between the two sequence assignments; this alone
does not identify which assignment is wrong.
""")

code(r'''
tab = (calls.group_by("split", "source_absent")
       .agg(pl.len().alias("n_calls"),
            pl.col("length").median().alias("median_length_aa"),
            pl.col("alc_percent").median().alias("median_alc_percent"),
            pl.col("denovo_equals_db_search").mean().round(3).alias("fraction_denovo_equals_db_search"))
       .sort("split", "source_absent"))
print(tab)

fig, axs = pf.figure(pf.TWO_COLUMN_MM, 55, ncols=2)
same = calls.filter(pl.col("denovo_equals_db_search"))
diff = calls.filter(~pl.col("denovo_equals_db_search"))
bins = np.arange(3.5, 41.5, 1)
axs[0].hist([same["length"].to_list(), diff["length"].to_list()], bins=bins, stacked=True,
            color=[pf.GREY, pf.OKABE_ITO["orange"]],
            label=[f"de novo call = database-search peptide (n = {same.height})",
                   f"de novo call differs (n = {diff.height})"])
axs[0].set_xlabel("Call length (aa)")
axs[0].set_ylabel("Calls (n)")
bins = np.arange(50, 101, 2.5)
axs[1].hist([same["alc_percent"].to_list(), diff["alc_percent"].to_list()], bins=bins, stacked=True,
            color=[pf.GREY, pf.OKABE_ITO["orange"]])
axs[1].set_xlabel("PEAKS average local confidence, ALC (%)")
axs[1].set_ylabel("Calls (n)")
pf.shared_legend(fig, ncol=2)
for ax, letter in zip(axs, "ab"):
    pf.panel_label(ax, letter)
pf.save(fig, FIG / "354_frozen_calls_length_alc_PXD005280")
''')

md(r"""
## 2. How often one k-mer matches by chance

Notebook 300's counting estimate, applied to this database: the expected number of places
in the 20,734 proteins where a k-mer of the database's own composition appears by chance is
(start positions) x Pr(same)^k, where Pr(same) is the chance two random database residues
fall in the same letter of the alphabet (the sum of squared letter frequencies). Fewer is
better for telling sources apart; one or more means a single k-mer seed cannot by itself
point to one protein.
""")

code(r'''
import gzip
import sys
sys.path.insert(0, "../nextflow-runs/fragment-source/bin")
from alphabets import CLUSTERS, class_of

seqs = []
with gzip.open(FROZEN / "db_full.fasta.gz", "rt") as fh:
    cur = []
    for line in fh:
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur))
            cur = []
        else:
            cur.append(line.strip())
    seqs.append("".join(cur))
letters = Counter("".join(seqs))
rows = []
for alphabet in ["protein20", "massspec18", "hsdm17", "sdm12"]:
    cls = class_of(alphabet)
    freq = Counter()
    for aa, n in letters.items():
        if aa in cls:
            freq[cls[aa]] += n
    tot = sum(freq.values())
    pr_same = sum((n / tot) ** 2 for n in freq.values())
    for k in range(4, 10):
        starts = sum(max(len(s) - k + 1, 0) for s in seqs)
        rows.append({"alphabet": alphabet, "letters": len(CLUSTERS[alphabet]), "k": k,
                     "pr_same": round(pr_same, 4), "expected_chance_places": starts * pr_same ** k})
chance = pl.DataFrame(rows)
print(chance.pivot(on="k", index=["alphabet", "letters", "pr_same"], values="expected_chance_places"))
fig, ax = pf.figure(pf.ONE_COLUMN_MM, 55)
style = {"protein20": ("D", pf.OKABE_ITO["vermillion"]), "massspec18": ("o", pf.OKABE_ITO["blue"]),
         "hsdm17": ("s", pf.OKABE_ITO["bluish_green"]), "sdm12": ("^", pf.OKABE_ITO["reddish_purple"])}
for alphabet, (marker, colour) in style.items():
    d = chance.filter(pl.col("alphabet") == alphabet)
    ax.plot(d["k"], d["expected_chance_places"], marker=marker, color=colour, label=alphabet)
ax.axhline(1, color=pf.GREY, lw=0.5, ls="--")
ax.set_yscale("log")
ax.set_yticks([0.01, 0.1, 1, 10, 100, 1000, 10000])
ax.set_yticklabels(["0.01", "0.1", "1", "10", "100", "1,000", "10,000"])
ax.set_xlabel("k-mer length k (residues)")
ax.set_ylabel("Expected chance places per k-mer (n); each step is tenfold")
pf.shared_legend(fig, ncol=4)
pf.save(fig, FIG / "354_expected_chance_kmer_places_human_crap")
''')

md(r"""
## 3. Development split: the k and alphabet that go to test

Each kmerseek setting was run on the 250 development calls and on 2,500 composition
shuffles of them. The threshold is the smallest check score at most 1% of shuffles reach;
recovery is the share of development calls with an accepted source in the top five at or
above it. Higher is better. The settings frozen for test are ringed. Development numbers
choose settings; they are not the result.
""")

code(r'''
print(dev.sort("alphabet", "k"))
print(json.dumps(choice, indent=1))
fig, ax = pf.figure(pf.ONE_COLUMN_MM, 62)
marks = {"massspec18": ("o", pf.OKABE_ITO["blue"]), "hsdm17": ("s", pf.OKABE_ITO["bluish_green"]),
         "sdm12": ("^", pf.OKABE_ITO["reddish_purple"])}
# The three alphabets give the same recovery at most k, so each is drawn 0.1 to one side of
# its k to keep all three visible; massspec18 sits at the true k and is drawn last.
offset = {"massspec18": 0.0, "hsdm17": -0.1, "sdm12": 0.1}
for alphabet in ["hsdm17", "sdm12", "massspec18"]:
    marker, colour = marks[alphabet]
    d = dev.filter(pl.col("alphabet") == alphabet).sort("k")
    ax.plot(d["k"] + offset[alphabet], d["dev_recovery"] * 100, marker=marker, color=colour,
            label=f"kmerseek {alphabet}" + ("" if alphabet == "massspec18" else f" (drawn {offset[alphabet]:+.1f} in k)"),
            zorder=3 if alphabet == "massspec18" else 2)
comp = dev.filter(pl.col("setting") == "comparison")
ax.axhline(comp["dev_recovery"][0] * 100, color=pf.OKABE_ITO["vermillion"], lw=1,
           label="C: exact index + blastp-short (no k)")
for s in choice["kmerseek_settings"]:
    r = dev.filter((pl.col("alphabet") == s["alphabet"]) & (pl.col("k") == s["k"]))
    ax.scatter(r["k"], r["dev_recovery"] * 100, s=60, facecolor="none", edgecolor="black", lw=0.8,
               label="frozen for test" if s is choice["kmerseek_settings"][0] else None, zorder=5)
ax.set_xlabel("k-mer length k (residues)")
ax.set_ylabel("Development recovery, top five (%)")
ax.set_xticks(sorted(dev.filter(pl.col("k") > 0)["k"].unique()))
pf.shared_legend(fig, ncol=2)
pf.save(fig, FIG / "354_dev_recovery_by_k_PXD005280")
''')

md(r"""
## 4. Recovery against false assignment, test split

Each line is one workflow with its threshold swept from low to high. The x axis is the
share of the 50 source-absent calls that still get a confident source (lower is better);
the y axis is the share of the 200 source-present calls with an accepted source in the top
five (higher is better). The filled point is the threshold fixed on development. The pass
region is the top-left corner: recovery at least 95% (rule 1) and false assignment at most
1% (rule 2).
""")

code(r'''
fig, ax = pf.figure(pf.ONE_COLUMN_MM, 75)
ax.axhspan(95, 101, xmin=0, xmax=1, color="#e8e8e8", zorder=0)
ax.axhline(95, color=pf.GREY, lw=0.5, ls="--")
ax.axvline(1, color=pf.GREY, lw=0.5, ls="--")
ax.text(1.5, 96, "rule 1: recovery >= 95%", fontsize=5, color="#555555", va="bottom")
for name in METHODS:
    d = curve.filter(pl.col("method") == name).sort("false_assignment", "recovery")
    ax.step(d["false_assignment"] * 100, d["recovery"] * 100, where="post", color=COLOUR[name], label=LABEL[name])
    mm = m["metrics"][name]
    ax.scatter([mm["false_assignment"] * 100], [mm["recovery_top5"] * 100], color=COLOUR[name], s=18, zorder=5)
ax.scatter([], [], color="black", s=18, label="threshold fixed on development")
ax.set_xlabel("Source-absent calls given a confident source (%)")
ax.set_ylabel("Source-present calls with the source in the top five (%)")
ax.set_xlim(-2, 102)
ax.set_ylim(0, 101)
pf.shared_legend(fig, ncol=1)
pf.save(fig, FIG / "354_recovery_vs_false_assignment_PXD005280")
rows = [{"method": LABEL[n], "threshold": m["metrics"][n]["tau"],
         "recovery_top5": f"{m['metrics'][n]['n_recovered']} of {m['metrics'][n]['n_present']}",
         "false_assignment": f"{m['metrics'][n]['n_false']} of {m['metrics'][n]['n_absent']}"} for n in METHODS]
print(pl.DataFrame(rows))
''')

md(r"""
### Where kmerseek's extra recoveries come from

Both workflows score candidates with the same check, so a call kmerseek recovers and C does
not must differ in which candidates were sent or in the threshold. The table lists the
source-present calls recovered by one workflow only, with the check score of the accepted
source under each (empty: not among the candidates).
""")

code(r'''
wide = (per_call.filter(pl.col("case") == "present")
        .pivot(on="method", index="call_id", values=["recovered", "accepted_score", "list_length"], aggregate_function="first"))
k0 = KMERSEEK[0]
only = wide.filter(pl.col(f"recovered_{k0}") != pl.col("recovered_comparison")).join(
    calls.select("call_id", "peptide", "db_search_peptide", "accepted_in_db"), on="call_id")
print(f"thresholds: {LABEL[k0]} {m['metrics'][k0]['tau']}, {LABEL['comparison']} {m['metrics']['comparison']['tau']}")
print(only.select("call_id", "peptide", "db_search_peptide", "accepted_in_db", f"recovered_{k0}", "recovered_comparison",
                  f"accepted_score_{k0}", "accepted_score_comparison").sort(f"recovered_{k0}", f"accepted_score_{k0}"))
fig, ax = pf.figure(pf.ONE_COLUMN_MM, 55)
both = wide.with_columns(pl.col(f"accepted_score_{k0}").fill_null(0), pl.col("accepted_score_comparison").fill_null(0))
ax.scatter(both["accepted_score_comparison"], both[f"accepted_score_{k0}"], s=6, color=pf.GREY, label="source-present test call")
ax.axvline(m["metrics"]["comparison"]["tau"], color=COLOUR["comparison"], lw=0.8, label=f"C threshold ({m['metrics']['comparison']['tau']})")
ax.axhline(m["metrics"][k0]["tau"], color=COLOUR[k0], lw=0.8, label=f"kmerseek threshold ({m['metrics'][k0]['tau']})")
ax.set_xlabel("Accepted source's check score, C\n(0 = not among C's candidates)")
ax.set_ylabel("Accepted source's check score, kmerseek\n(0 = not among its candidates)")
pf.shared_legend(fig, ncol=1)
pf.save(fig, FIG / "354_accepted_source_score_kmerseek_vs_comparison_PXD005280")
''')

md(r"""
## 5. Candidate-list length

The candidate list is every candidate whose check score reaches the method's threshold. A
shorter list is better (rule 3: median at most five). The dashed lines are the candidates
each workflow sent to the check, before the threshold.
""")

code(r'''
pc = per_call.filter(pl.col("case") == "present")
fig, ax = pf.figure(pf.ONE_COLUMN_MM, 62)
for name in METHODS:
    d = pc.filter(pl.col("method") == name)
    for col, ls, suffix in (("list_length", "-", "after the threshold"),
                            ("n_candidates_before_check", "--", "sent to the check")):
        x = np.sort(np.asarray(d[col].to_list(), dtype=float))
        y = np.arange(1, len(x) + 1) / len(x) * 100
        ax.step(np.maximum(x, 0.5), y, where="post", color=COLOUR[name], ls=ls, label=f"{LABEL[name]}, {suffix}")
ax.axvline(5, color=pf.GREY, lw=0.5)
ax.set_xscale("log")
ax.set_xticks([0.5, 1, 5, 10, 100, 1000])
ax.set_xticklabels(["0", "1", "5", "10", "100", "1,000"])
ax.set_xlabel("Candidates per source-present call (n)")
ax.set_ylabel("Calls with at most this many (%)")
pf.shared_legend(fig, ncol=1)
pf.save(fig, FIG / "354_candidate_list_length_PXD005280")
print(pc.group_by("method").agg(pl.col("list_length").median().alias("median_list_length"),
                                pl.col("list_length").max().alias("max_list_length"),
                                pl.col("n_candidates_before_check").median().alias("median_sent_to_check")).sort("method"))
''')

md(r"""
## 6. Runtime, end to end

One CPU, the 250 test calls against the full database, from a fresh index build. Build is
kmerseek's index (K) or the exact 5-mer index plus `makeblastdb` (C); search is kmerseek's
search (K) or the exact lookup plus blastp-short (C); verify is the shared full-alphabet
check. Shorter is better (rule 4b asks for C's total to be at least five times K's).
""")

code(r'''
print(runtime.sort("method", "step"))
fig, axs = pf.figure(pf.TWO_COLUMN_MM, 45, ncols=2)
steps = ["build", "search", "verify"]
shade = {"build": 1.0, "search": 0.6, "verify": 0.3}
for ax, col, xlabel in ((axs[0], "wall_s", "Wall time (s)"), (axs[1], "cpu_s", "CPU time (s)")):
    for i, name in enumerate(METHODS):
        left = 0.0
        for st in steps:
            v = runtime.filter((pl.col("method") == name) & (pl.col("step") == st))[col]
            v = float(v[0]) if len(v) else 0.0
            ax.barh(i, v, left=left, color=COLOUR[name], alpha=shade[st], edgecolor="black", lw=0.3)
            left += v
        ax.text(left, i, f" {left:,.0f} s", va="center", fontsize=5)
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels([LABEL[n].replace(": ", ":\n") for n in METHODS])
    ax.set_xlabel(xlabel)
    ax.invert_yaxis()
axs[1].set_yticklabels([])
for st in steps:  # legend proxies in grey: the shade is the step, the colour the workflow
    axs[0].barh([], [], color="black", alpha=shade[st], edgecolor="black", lw=0.3, label=st)
pf.shared_legend(fig, ncol=3, title="Step (shade; the colour is the workflow)")
for ax, letter in zip(axs, "ab"):
    pf.panel_label(ax, letter)
pf.save(fig, FIG / "354_runtime_build_search_verify_PXD005280")
''')

md(r"""
## 7. Shuffled controls

Five composition shuffles and five shuffles keeping adjacent residue pairs of each test
call, searched against the full database. Each bar is the share given a confident source at
the method's fixed threshold. A shuffle has no source, so lower is better; the thresholds
were set to allow 1% on development composition shuffles.
""")

code(r'''
rows = []
for name in METHODS:
    mm = m["metrics"][name]
    rows += [{"method": LABEL[name], "control": "composition shuffle", "assigned_percent": mm["control_assigned_composition"] * 100},
             {"method": LABEL[name], "control": "adjacent-pair shuffle", "assigned_percent": mm["control_assigned_dipeptide"] * 100}]
ctrl = pl.DataFrame(rows)
print(ctrl)
# A short peptide has few orders that keep every adjacent pair, often only its own.
def read_fa(path):
    out, name = {}, None
    for line in open(path):
        if line.startswith(">"):
            name = line[1:].strip()
        else:
            out[name] = line.strip()
    return out
ctrl_seqs = read_fa(FROZEN / "queries_test_controls.fasta")
orig = dict(zip(calls["call_id"], calls["peptide"]))
same = Counter()
for q, s in ctrl_seqs.items():
    cid, kind = q.rsplit("_", 1)
    same[(kind.rstrip("0123456789"), s == orig[cid])] += 1
for kind in ("comp", "dipep"):
    n_same, n = same[(kind, True)], same[(kind, True)] + same[(kind, False)]
    print(f"{'composition' if kind == 'comp' else 'adjacent-pair'} shuffles identical to the call itself: {n_same} of {n}")
fig, ax = pf.figure(pf.ONE_COLUMN_MM, 45)
for j, kind in enumerate(["composition shuffle", "adjacent-pair shuffle"]):
    for i, name in enumerate(METHODS):
        v = ctrl.filter((pl.col("method") == LABEL[name]) & (pl.col("control") == kind))["assigned_percent"][0]
        ax.scatter(v, i + (0.15 if j else -0.15), marker="o" if j == 0 else "s", color=COLOUR[name], s=14)
    ax.scatter([], [], marker="o" if j == 0 else "s", color="black", s=14, label=kind)
ax.axvline(1, color=pf.GREY, lw=0.5, ls="--")
ax.set_yticks(range(len(METHODS)))
ax.set_yticklabels([LABEL[n].replace(": ", ":\n") for n in METHODS])
ax.invert_yaxis()
ax.set_xlabel("Shuffled test calls given a confident source (%)")
pf.shared_legend(fig, ncol=2, title="Marker = control (circle composition, square adjacent pairs); colour = workflow")
pf.save(fig, FIG / "354_shuffled_controls_PXD005280")
''')

md(r"""
## 8. Examples, residue by residue

One call per outcome, picked by a fixed rule: the lowest scan number in each group.
Each pair shows the call, the matched stretch of the protein with 1-based coordinates, the
match line (`|` same residue, `:` I against L or Q against K, blank otherwise, `-` a gap) and
how many aligned columns hold the same residue.
""")

code(r'''
def show(call_id, target, pair, method=""):
    a = aln.filter((pl.col("call_id") == call_id) & (pl.col("target") == target) & (pl.col("pair") == pair)
                   & (pl.col("method").fill_null("") == method)).row(0, named=True)
    c = calls.filter(pl.col("call_id") == call_id).row(0, named=True)
    w = max(len(target), 8)
    print(f"  {'call':<{w}} {a['query_start']:>5} {a['query_line']} {a['query_end']}")
    print(f"  {'':<{w}} {'':>5} {a['match_line']}")
    print(f"  {target:<{w}} {a['target_start']:>5} {a['target_line']} {a['target_end']}")
    print(f"  {a['n_identical']} of {a['n_columns']} identical; check score {a['score']}")

def describe(call_id):
    c = calls.filter(pl.col("call_id") == call_id).row(0, named=True)
    print(f"{call_id}: de novo call {c['peptide']} ({c['length']} aa, ALC {c['alc_percent']:.0f}%, "
          f"local confidence {c['local_confidence_percent']}); database search read {c['db_search_peptide']}; "
          f"accepted source {c['accepted_in_db']}")

pcw = per_call.pivot(on="method", index=["call_id", "case"], values="recovered", aggregate_function="first")
k0 = KMERSEEK[0]
groups = {
    "found by kmerseek and by C": (pl.col(k0) == True) & (pl.col("comparison") == True),
    "found by kmerseek only": (pl.col(k0) == True) & (pl.col("comparison") == False),
    "found by C only": (pl.col(k0) == False) & (pl.col("comparison") == True),
    "found by neither": (pl.col(k0) == False) & (pl.col("comparison") == False),
}
present_ids = pcw.filter(pl.col("case") == "present")
for title, cond in groups.items():
    sel = present_ids.filter(cond).with_columns(pl.col("call_id").str.replace("scan", "").cast(pl.Int64).alias("scan")).sort("scan")
    print(f"\n=== {title}: {sel.height} of {present_ids.height} source-present calls")
    if sel.height == 0:
        continue
    cid = sel["call_id"][0]
    describe(cid)
    acc = sorted(calls.filter(pl.col("call_id") == cid)["accepted_in_db"][0].split(";"))[0]
    print(" against its accepted source:")
    show(cid, acc, "accepted source")
    for name in METHODS:
        top = aln.filter((pl.col("call_id") == cid) & (pl.col("method") == name) & (pl.col("pair") == "top candidate"))
        if top.height:
            print(f" {LABEL[name]}, top candidate:")
            show(cid, top["target"][0], "top candidate", name)
        else:
            print(f" {LABEL[name]}: no candidate")

pa = per_call.filter((pl.col("case") == "absent"))
for name in METHODS:
    sel = pa.filter((pl.col("method") == name) & (pl.col("confident") == True)).with_columns(
        pl.col("call_id").str.replace("scan", "").cast(pl.Int64).alias("scan")).sort("scan")
    print(f"\n=== source-absent calls given a confident source by {LABEL[name]}: {sel.height} of {pa.filter(pl.col('method') == name).height}")
    if sel.height:
        cid = sel["call_id"][0]
        describe(cid)
        show(cid, sel["top_target"][0], "top candidate", name)
''')

md(r"""
## 9. Compute used

Every SLURM job of this notebook (`make cost`), including the download, the kmerseek build
and the trial runs. CPU-hours are allocated cores x elapsed time; the budget was two
compute-days.
""")

code(r'''
cost = compute.with_columns(pl.col("ElapsedRaw").cast(pl.Float64).alias("elapsed_s"), pl.col("NCPUS").cast(pl.Int64))
cost = cost.with_columns((pl.col("elapsed_s") * pl.col("NCPUS") / 3600).alias("allocated_cpu_hours"),
                         (pl.col("used_cpu_s") / 3600).alias("used_cpu_hours"))
print(cost.select("JobID", "JobName", "State", "NCPUS", "Elapsed", "allocated_cpu_hours", "used_cpu_hours"))
tot = cost.select(pl.col("allocated_cpu_hours").sum(), pl.col("used_cpu_hours").sum(),
                  (pl.col("elapsed_s").sum() / 3600).alias("elapsed_hours_summed_over_jobs"))
print(tot)
fig, ax = pf.figure(pf.ONE_COLUMN_MM, 50)
by = cost.group_by("JobName").agg(pl.col("allocated_cpu_hours").sum(), pl.col("used_cpu_hours").sum()).sort("allocated_cpu_hours")
ax.barh(by["JobName"], by["allocated_cpu_hours"], color=pf.GREY, label="allocated (cores x elapsed)")
ax.barh(by["JobName"], by["used_cpu_hours"], color=pf.OKABE_ITO["blue"], height=0.4, label="used (TotalCPU)")
ax.set_xlabel("CPU-hours")
pf.shared_legend(fig, ncol=2)
pf.save(fig, FIG / "354_compute_cpu_hours")
''')

md(r"""
## 10. Pass rule, applied
""")

code(r'''
for name, v in m["per_setting"].items():
    print(f"{LABEL[name]}: {'PASS' if v['pass'] else 'FAIL'}")
    for rule, r in v["rules"].items():
        print(f"  {rule}: value {r['value']}, needed {r['needed']}, holds {r['holds']}"
              + (f" (C's recovery at false rate <= {r['false_rate']}: {r['comparison_recovery_at_same_false_rate']})" if "false_rate" in r else "")
              + (f" (matched recovery: {r['matched_recovery']})" if "matched_recovery" in r else ""))
print("\nVerdict:", m["verdict"])
print("Diagnostics:", json.dumps(m["diagnostics"], indent=1))
''')

md(REVIEW)

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "2025-kmerseek-analysis", "language": "python", "name": "2025-kmerseek-analysis"},
                                    "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1) + "\n")
print("wrote", OUT)
