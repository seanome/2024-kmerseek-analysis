#!/usr/bin/env python3
"""Generate notebooks/355_construct_check_kill_test_not_run_missing_submissions.ipynb."""

import json
from pathlib import Path

import black

# The Lint workflow runs black on the committed notebook, so each cell is formatted here.
# Formatting at build time keeps this script and the notebook it writes from drifting apart,
# which is what happens when black is run on the notebook alone.
BLACK_MODE = black.Mode(is_pyi=False, line_length=88)

cells = []


def md(source):
    cells.append(
        {
            "cell_type": "markdown",
            "id": f"md-{len(cells):02d}",
            "metadata": {},
            "source": source.strip().splitlines(keepends=True),
        }
    )


def code(source):
    formatted = black.format_str(source.strip("\n"), mode=BLACK_MODE).rstrip("\n")
    cells.append(
        {
            "cell_type": "code",
            "id": f"code-{len(cells):02d}",
            "execution_count": None,
            "metadata": {"jupyter": {"source_hidden": True}},
            "outputs": [],
            "source": formatted.splitlines(keepends=True),
        }
    )


md(r"""
# 355. Does a reduced-alphabet construct check catch tag and linker mistakes? NOT RUN: the submissions do not exist here

## Verdict: NOT RUN, missing data

The kill test asks whether writing a protein in a reduced alphabet (grouping the 20 amino
acids into a handful of letters, which is how kmerseek finds distant relatives) catches
documented construct mistakes that exact matching and local alignment miss. It needs 200
historical construct submissions from a protein-production facility: what was ordered, what
came back, and what the discrepancy turned out to be.

Those 200 submissions are not in this repository, are not in any other repository or data
directory on this machine, and cannot be obtained from a public database. They are a
facility's own records. Nothing stands in for them: a published construct has no discrepancy
record, and a construct this notebook builds by putting a tag into a clean protein contains
only the mistakes that were put there on purpose, which is the one thing the test must not
assume. So the four thresholds below have no numbers.

| Threshold, fixed before any run | Required | Measured |
| --- | --- | --- |
| 1. Documented unexpected inserts recovered | at least 95% | not measurable, no submissions |
| 2. False flags per 100 clean constructs | at most 1 | not measurable, no clean submissions |
| 3. Documented errors the comparison workflow missed | at least 5 | not measurable, no discrepancy records |
| 4. Median review time per batch | no increase | not measurable, no recorded review times |

Compute used: well under one compute-hour, all on the laptop, no index build, no GPU, no
Sherlock job. The budget was two compute-days. Section 8 gives the numbers.

## What is in this notebook instead

Everything needed so the test runs on the day the submissions arrive, plus one mechanics
check that already says something about the prior.

1. The pass rule, committed before any result existed
   (`data/kill_tests/construct_check/pass_rule.json`).
2. The exact format the 200 submissions must arrive in, enforced by the loader that reads
   them, so a partial export fails loudly instead of quietly shrinking the test.
3. The tag and linker list, frozen with a source for every sequence and a checksum for every
   file.
4. The three methods, written and unit-checked: exact tag matching, local alignment, and the
   reduced-alphabet check. The facility's DNA-level check is part of the comparison and is
   read from the facility's own call file, not left out.
5. A mechanics check on constructs this notebook builds. **It does not count toward the
   verdict**, and it cannot: there is no documented mistake in it. What it does show is that
   the representation is the wrong size for the job, which is what the stated prior expected.

## The one thing the mechanics check already shows

Eight of the twelve frozen tag and linker sequences are shorter than 19 amino acids. kmerseek
runs a hydrophobic/polar alphabet at k of 19 or more, and a sequence shorter than k has no
k-mers at all, so those eight cannot be found by a reduced-alphabet k-mer check at any
setting kmerseek is run at. That is a property of the tag list and the method, not of any
data that is missing. The two that are long enough are the GST and MBP fusion partners, at
218 and 370 amino acids, and exact matching already finds both.

In the other direction, at hydrophobic/polar k=19 the check flagged a clean Ced9 protein as
containing MBP. The 19 residues in Ced9 and the 19 in MBP share 3 amino acids out of 19, and
encode to the same 19-letter hydrophobic/polar string. Section 6 prints them.
""")

md(r"""
## 1. The question, the decision, and the rule, all fixed before any result

**The question.** When a facility's sequenced and translated construct is not what was
ordered, does a reduced-alphabet kmerseek check find the unexpected piece of sequence (a tag,
a linker, another known insert) in cases where exact matching of the tag list and local
alignment of received against intended do not?

**The decision this would change.** Whether kmerseek's reduced alphabet gets a second
application in the paper, beyond finding distant homologs: a quality-control check a facility
would run on every construct. That is a different argument from the homology one, and it
would need its own figure and its own section. A fail means the paper does not make that
claim and nobody builds the tool.

**The prior, stated before running anything: strongly unfavorable.** The workflow is real.
The representation is probably wrong for it. A reduced alphabet is built to find similarity
that is too distant for the amino acids to show. A construct mistake is the opposite problem:
the inserted sequence is usually known exactly, so an exact search is already the right tool,
and the inserted piece is often shorter than one k-mer.

**The pass rule. All four must hold, on the test split, with the alphabet and k frozen on the
development split first.**

1. Recover at least 95% of documented unexpected inserts.
2. At most one false flag per 100 clean constructs.
3. Find at least five documented errors the comparison workflow missed.
4. Median review time per batch does not increase.

**Drop it if** exact searches already catch the discrepancies, or flags cannot tell an insert
the customer ordered from one nobody ordered, or the useful detections depend on DNA
information the protein method does not have.

Threshold 4 needs the facility's own recorded review minutes. A count of flagged regions is
not a substitute for it. If those minutes do not arrive, threshold 4 is reported as not
assessable, which under this rule is not a pass.

The rule is in `data/kill_tests/construct_check/pass_rule.json`, written by
`notebooks/construct_check_utils.py` before any result existed, so what any later run used can
be compared against what was promised.
""")

code(r"""
import json
import sys
from pathlib import Path

import polars as pl

REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO / "notebooks"))

import construct_check_utils as cc
import pubfig as pf

FROZEN = REPO / "data" / "kill_tests" / "construct_check"
TABLES = REPO / "tables"
FIGURES = REPO / "figures"
pf.use_style()

pass_rule = json.loads((FROZEN / "pass_rule.json").read_text())
for name, rule in pass_rule["pass_rule"].items():
    print(f"{name}: {rule['direction']} {rule['threshold']}")
    print(f"    {rule['what']}")
print()
for name, clause in pass_rule["fail_clauses"].items():
    print(f"fail if {name}: {clause}")
""")

md(r"""
## 2. The data that is missing, and the format it must arrive in

Searched before concluding it does not exist: every file in this repository, every open and
closed pull request in `seanome/2024-kmerseek-analysis` (174 of them), and the data
directories on this machine. The word "construct" appears in the repository only as the
ordinary verb, in code that builds a table or a set of pairs. There is no facility submission
record anywhere.

The table below is the contract. It is printed from the same module that validates an
arriving file, `construct_check_utils.load_submissions`, so the documented columns and the
enforced ones cannot drift apart. The rules the loader checks, beyond the columns being
present:

1. `submission_id` is unique.
2. `intended_protein` and `received_protein` are one-letter amino-acid code, no gaps or
   spaces. `X` for an unknown residue and `*` for a stop are allowed, nothing else.
3. For a row whose `discrepancy_class` is `unexpected_insert`, all four `resolved_insert_*`
   columns are filled, and the residues at `resolved_insert_start` to `resolved_insert_end`
   of `received_protein` are exactly `resolved_insert_sequence`, counting from 1 and including
   both ends. This is checked on load, because a coordinate convention off by one would move
   every number in the result.
4. `insert_was_intended` is `true` or `false` for every insert, never blank. The test is
   partly about telling a tag the customer ordered from one nobody ordered, and a blank here
   removes the row from both sides of that comparison.
5. `split` is `dev` or `test`, assigned so that every submission sharing an intended protein
   falls in the same split. Otherwise the same protein appears on both sides and the choice of
   alphabet and k is no longer made on data held out from the test.

Also needed, beyond the submissions file: the recorded review minutes per batch before and
after (threshold 4 cannot be measured without them); the DNA where the facility kept it,
plus its own DNA-level call file, because the DNA check is part of the comparison the protein
method is measured against; and any insert sequence the discrepancy records resolved that is
not a standard tag, with its own source.
""")

code(r"""
contract = pl.DataFrame(
    {
        "column": list(cc.SUBMISSION_COLUMNS) + list(cc.OPTIONAL_SUBMISSION_COLUMNS),
        "required": ["required"] * len(cc.SUBMISSION_COLUMNS)
        + ["optional"] * len(cc.OPTIONAL_SUBMISSION_COLUMNS),
        "what goes in it": list(cc.SUBMISSION_COLUMNS.values())
        + list(cc.OPTIONAL_SUBMISSION_COLUMNS.values()),
    }
)
with pl.Config(fmt_str_lengths=120, tbl_rows=30, tbl_width_chars=200):
    print(contract)
print()
print("discrepancy_class is one of:", ", ".join(sorted(cc.DISCREPANCY_CLASSES)))
""")

md(r"""
**Figure 1. What the test needs, what it does with it, and what comes out.** Solid boxes are
in this repository. Dashed boxes are the parts that are not available, and every one of them
is an input: the submissions and the facility's DNA calls and review times. The four
thresholds on the right all depend on an input that is missing, which is why none of them has
a number.
""")

code(r"""
from matplotlib.patches import FancyArrowPatch, Rectangle

PRESENT = pf.OKABE_ITO["blue"]
MISSING = pf.OKABE_ITO["vermillion"]

fig, ax = pf.figure(pf.TWO_COLUMN_MM, 78)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

def box(x, y, width, height, text, present):
    # Solid edge for what is here, dashed for what is not. The colour repeats the same
    # meaning, so the figure survives being printed in grey.
    ax.add_patch(
        Rectangle(
            (x, y), width, height,
            facecolor="white",
            edgecolor=PRESENT if present else MISSING,
            linewidth=1.0 if present else 1.2,
            linestyle="-" if present else (0, (3, 2)),
            zorder=2,
        )
    )
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", zorder=3)

def arrow(x0, y0, x1, y1):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6,
            color=pf.OKABE_ITO["black"], linewidth=0.5, zorder=1,
        )
    )

ax.text(16, 97, "Input", ha="center", fontweight="bold")
ax.text(50, 97, "What is run on it", ha="center", fontweight="bold")
ax.text(85, 97, "What comes out", ha="center", fontweight="bold")

# Laid out in four rows so that no arrow crosses a box. The middle column is left empty
# across the review-minutes row, because no method is run on review times: they feed
# threshold 4 directly.
box(1, 72, 30, 20, "200 historical construct\nsubmissions:\nintended protein, received\nprotein, resolved\ndiscrepancy record", False)
box(1, 56, 30, 12, "Facility DNA calls\n(VecScreen or its own)", False)
box(1, 42, 30, 10, "Recorded review\nminutes per batch", False)
box(1, 18, 30, 20, "Frozen tag and linker list:\n12 sequences, with a source\nand a checksum for each\n+ 6 clean Swiss-Prot\nhost proteins", True)

box(36, 80, 28, 10, "Exact tag match\n(protein20)", True)
box(36, 68, 28, 10, "Local alignment\n(BLOSUM62)", True)
box(36, 56, 28, 10, "Facility DNA check", True)
box(36, 26, 28, 12, "kmerseek reduced alphabet\n(the piece under test):\nshared reduced k-mers", True)
box(36, 12, 28, 12, "Mechanics check on\nconstructs built here\n(does not count)", True)

box(69, 66, 30, 24, "Thresholds 1-3:\ninserts recovered,\nfalse flags per 100 clean,\nerrors the comparison\nworkflow missed\nNOT MEASURABLE", False)
box(69, 42, 30, 10, "Threshold 4:\nreview time per batch\nNOT MEASURABLE", False)
box(69, 12, 30, 24, "What this notebook does\ndeliver: the pass rule, the\ndata contract, the frozen\ntag list, working code,\nand the mechanics check", True)

arrow(31, 86, 36, 85)
arrow(31, 80, 36, 73)
arrow(31, 74, 36, 34)
arrow(31, 62, 36, 61)
arrow(31, 47, 69, 47)
arrow(31, 30, 36, 32)
arrow(31, 24, 36, 18)
arrow(64, 85, 69, 80)
arrow(64, 73, 69, 76)
arrow(64, 61, 69, 70)
arrow(64, 34, 69, 67)
arrow(64, 18, 69, 24)

ax.plot([], [], color=PRESENT, linewidth=1.0, label="in this repository")
ax.plot([], [], color=MISSING, linewidth=1.2, linestyle=(0, (3, 2)),
        label="not available, so nothing downstream can be measured")
pf.shared_legend(fig, ncol=2)
pf.save(fig, FIGURES / "355-what_the_kill_test_needs_and_what_is_missing")
print("saved figures/355-what_the_kill_test_needs_and_what_is_missing.png")
""")

md(r"""
## 3. The frozen tag and linker list

Twelve sequences: the tags and linkers the task named (His, FLAG, HA, Myc, Strep, GST, MBP,
common glycine-serine linkers) and nothing else. No collection of arbitrary translated vector
fragments.

Where each sequence came from is recorded in the file. Two kinds of source:

- A subsequence of a named UniProt entry, with the coordinates it was taken from. The HA tag
  is influenza A hemagglutinin 114-122 (UniProt P03437); the Myc tag is human MYC 425-434
  (P01106); GST is the whole Schistosoma japonicum sj26 protein (P08515); MBP is the mature
  chain of E. coli MalE, residues 27-396 of P0AEX9, because a fusion carries the mature chain
  and not the signal peptide. The build script fails if a sequence claimed to come from a
  UniProt entry is not in it at those coordinates, so a typo cannot enter the frozen file.
- Synthetic, with the publication that introduced it. Every citation was resolved against the
  Crossref API, not written from memory; they are in
  `data/kill_tests/construct_check/references.md`. Two short glycine-serine linkers have no
  citation, because no single paper introduced them, and the file says so rather than
  attaching a paper that does not describe that sequence.

The list, the host proteins, the UniProt records they were built from, the pass rule and the
manifest are checksummed in `data/kill_tests/construct_check/SHA256SUMS`, frozen 2026-10-20.
""")

code(r"""
tags = cc.load_tag_list()
tag_table = pl.DataFrame(
    [
        {
            "name": tag.name,
            "category": tag.category,
            "length_aa": len(tag.sequence),
            "sequence": tag.sequence if len(tag.sequence) <= 24 else tag.sequence[:21] + "...",
            "source": tag.source_kind
            + (
                f" {tag.source_accession} {tag.source_coords_1based}"
                if tag.source_accession
                else ""
            ),
            "citation": tag.source_id or "none, see references.md",
        }
        for tag in tags
    ]
).sort("length_aa")
with pl.Config(fmt_str_lengths=60, tbl_rows=20, tbl_width_chars=200):
    print(tag_table)

checksums = [
    line
    for line in (FROZEN / "SHA256SUMS").read_text().split("\n")
    if line and not line.startswith("#")
]
print(f"\n{len(checksums)} files checksummed in SHA256SUMS")
print(f"{sum(1 for tag in tags if len(tag.sequence) < 19)} of {len(tags)} sequences are "
      "shorter than 19 aa, the smallest k a hydrophobic/polar alphabet is run at")
""")

md(r"""
**Figure 2. Most of the tag list is shorter than one k-mer.**

A sequence shorter than k has no k-mers, so a k-mer method cannot find it at that k, however
good the alphabet is. Panel **a** counts how many of the twelve frozen sequences still have at
least one k-mer as k grows. Panel **b** puts each sequence at its own length, so the whole list
is visible rather than a summary.

Higher is better in panel **a**: it is the number of tags the method could possibly find. The
shaded band is k of 19 to 30, the range a hydrophobic/polar alphabet is run at in this project.
Nothing about this figure depends on the missing submissions.
""")

code(r"""
kmers = pl.read_csv(TABLES / "355-mechanics_tag_kmers_by_ksize.tsv", separator="\t")
reachable = (
    kmers.filter(pl.col("has_any_kmer") == "true")
    .group_by("ksize")
    .agg(pl.len().alias("n_tags"))
    .sort("ksize")
)
print(reachable.to_pandas().to_string(index=False))

lengths = (
    kmers.select("tag_name", "category", "tag_length_aa").unique().sort("tag_length_aa")
)
print()
print(lengths.to_pandas().to_string(index=False))

HP_LOW, HP_HIGH = 19, 30
CATEGORY_COLOUR = {
    "affinity tag": pf.OKABE_ITO["blue"],
    "epitope tag": pf.OKABE_ITO["vermillion"],
    "linker": pf.OKABE_ITO["bluish_green"],
    "fusion partner": pf.OKABE_ITO["reddish_purple"],
}

fig, axs = pf.figure(pf.TWO_COLUMN_MM, 62, ncols=2)

ax = axs[0]
ax.axvspan(HP_LOW, HP_HIGH, color=pf.GREY, alpha=0.25,
           label="k a hydrophobic/polar alphabet is run at (19-30)")
ax.plot(reachable["ksize"], reachable["n_tags"], color=pf.OKABE_ITO["black"],
        marker="o", markersize=3, label="frozen tags with at least one k-mer")
ax.set_xlabel("k (amino acids per k-mer)")
ax.set_ylabel("Tags with at least one k-mer (n of 12)")
ax.set_ylim(-0.5, 12.5)
ax.set_yticks(range(0, 13, 2))
ax.grid(axis="y", which="major")

# Panel b: one row per sequence, so every tag can be read back to its name.
ordered = lengths.sort("tag_length_aa")
positions = {}
for index, row in enumerate(ordered.iter_rows(named=True)):
    positions[row["tag_name"]] = index
for category, colour in CATEGORY_COLOUR.items():
    rows = ordered.filter(pl.col("category") == category)
    if rows.is_empty():
        continue
    axs[1].plot(
        rows["tag_length_aa"],
        [positions[name] for name in rows["tag_name"]],
        "o", color=colour, markersize=4, linestyle="none", label=category,
    )
# Panel b's x axis is a length, not k, so the grey band of panel a would mean something
# different here. A dashed line, its own mark with its own legend entry, instead.
axs[1].axvline(HP_LOW, color=pf.OKABE_ITO["black"], linestyle="--",
               label="19 aa, the smallest k this alphabet is run at")
axs[1].set_xscale("log")
axs[1].set_xticks([3, 5, 10, 19, 30, 100, 400])
axs[1].set_xticklabels(["3", "5", "10", "19", "30", "100", "400"])
axs[1].set_yticks(list(positions.values()))
axs[1].set_yticklabels(list(positions.keys()))
axs[1].set_xlabel("Sequence length (aa)")
axs[1].grid(axis="y", which="major")
axs[1].set_ylim(-0.7, len(positions) - 0.3)

pf.shared_legend(fig, ncol=3)
for ax, letter in zip(axs, "ab"):
    pf.panel_label(ax, letter)
pf.save(fig, FIGURES / "355-frozen_tag_lengths_vs_ksize_reachability")
print()
print("the step from 3 to 5 aa takes as much room on panel b's x axis as the step from "
      "30 to 50, because the spacing is by ratio")
print("saved figures/355-frozen_tag_lengths_vs_ksize_reachability.png")
""")

md(r"""
## 4. The three methods, and the DNA check

All three return the same kind of record, so they can be compared: a submission, a method, the
first and last residue of the flagged region counted along the received protein from 1, the
name of the insert if the method can name one, and the evidence.

- **Exact tag match (protein20).** Each of the twelve frozen sequences looked for as an exact
  substring of the received protein, in the full 20-letter alphabet. What a facility does now.
- **Local alignment (BLOSUM62).** Received aligned locally against intended; any run of at
  least three residues of received that the alignment does not cover is reported as an insert.
  Also what a facility does now. It finds that something is there but cannot say what.
- **kmerseek reduced alphabet.** Received and every tag written in a reduced alphabet, and a
  flag raised where they share a run of identical reduced k-mers. This is kmerseek's seeding
  step, run here in Python so the mechanics check needs no index and no Sherlock job. The
  alphabets come from `ortholog_analysis_utils.ALPHABET_AA_GROUPS`, which holds kmerseek's own
  encoding tables.
- **The facility's DNA check.** `dna_check_flags` reads the facility's own call file (VecScreen
  or equivalent) and turns it into the same records. It is part of the comparison the protein
  method is measured against. Where the facility kept no DNA for a submission, that submission
  counts as having no DNA check, and that count is reported rather than hidden.

The alphabet and k are not chosen here. `choose_alphabet_and_k` implements the choice, by a
rule fixed in advance: among the alphabet and k pairs whose false flags on the development
split's clean constructs stay at or under one per 100, take the highest recovery of documented
unexpected inserts; ties break toward the larger k and then the alphabet with more letters,
since both make a chance match less likely. If no pair stays inside the budget, the function
raises instead of widening the budget, because under the fixed rule that is a fail.

Twenty-one unit checks run on the frozen tag list and on constructs built from the frozen host
proteins: that each method finds an insert it is given, that the coordinates it reports hold
the residues it claims, that the loader refuses a file breaking any rule of the contract, and
that the scorer returns FAIL on a case built to fail and PASS on one built to pass, so the
verdict is not hardwired. None of them is evidence that the method works on real submissions.
""")

code(r"""
import subprocess

result = subprocess.run(
    [sys.executable, "-m", "pytest", str(REPO / "tests" / "test_construct_check_utils.py"), "-q"],
    capture_output=True,
    text=True,
    cwd=REPO,
)
print(result.stdout.strip().split("\n")[-1])
print()
print("methods available:")
for name in ("exact_tag_flags", "alignment_flags", "reduced_kmer_flags", "dna_check_flags",
             "choose_alphabet_and_k", "evaluate_pass_rule"):
    print(f"  construct_check_utils.{name}")
""")

md(r"""
## 5. Mechanics check: does each method work on a construct built on purpose?

**This section does not count toward the verdict, and cannot.** Every construct in it was built
by this analysis: a frozen tag put at a random position, at least 20 residues from either end,
into one of six clean Swiss-Prot proteins (BCL2, Ced9, MCL1, BAX, p53, EGFR). The only mistakes
in it are the ones put there on purpose. What it checks is the mechanics: that the code finds
what it is given, and what the representation can reach at all.

Twelve tags x six hosts x two kinds of insert = 144 built constructs, each run through three
methods. The two kinds of insert are the tag exactly as frozen, and the tag with its middle
residue changed, which is the case an exact search is expected to lose.

Two ways of counting a recovery, both reported, because they differ:

- **flagged**: a flag touches the inserted residues, whatever it names.
- **named right**: the flag also names the tag that was put in. A flag that lands on the insert
  while naming a different tag does not tell a reviewer what is there. Local alignment names no
  tag at all, so it is counted the loose way only.

The reduced-alphabet check runs at hydrophobic/polar `hp_thomas_dill` with k=19. That is the
smallest k in the range this project runs a hydrophobic/polar alphabet at, which gives it the
best chance of reaching a short tag; a larger k can only reach fewer of them.
""")

code(r"""
recovery = pl.read_csv(TABLES / "355-mechanics_recovery_by_method.tsv", separator="\t")
summary = (
    recovery.group_by(["insert_kind", "method"])
    .agg(
        pl.col("flagged").eq("true").sum().alias("flagged"),
        pl.col("flagged_named_right").eq("true").sum().alias("named_right"),
        pl.len().alias("n_constructs"),
    )
    .sort(["insert_kind", "method"])
)
with pl.Config(fmt_str_lengths=60, tbl_width_chars=200):
    print(summary)

per_tag = (
    recovery.filter(pl.col("insert_kind") == "exact tag")
    .group_by(["tag_name", "tag_length_aa", "method"])
    .agg(pl.col("flagged_named_right").eq("true").sum().alias("named_right_of_6_hosts"))
    .sort(["tag_length_aa", "tag_name", "method"])
)
print()
with pl.Config(fmt_str_lengths=60, tbl_rows=40, tbl_width_chars=200):
    print(per_tag)
""")

md(r"""
**Figure 3. The reduced alphabet reaches only the two sequences longer than k.**

Each point is one of the twelve frozen sequences, at its real length, and the share of the six
host proteins in which the method named the insert correctly. Higher is better. Panel **a** is
the tag exactly as frozen, panel **b** the same tag with its middle residue changed.

The two methods land on the same point wherever they agree, so exact matching is drawn as the
large pale circle and the reduced-alphabet check as the smaller solid one on top of it. A
length with only a large pale circle at 1.0 is a tag exact matching named and the reduced
alphabet did not.

Local alignment is on neither panel, because it names no tag at all. It flagged the inserted
region in all 144 built constructs and said nothing about what was there. That is its row in
the table above, and it is the gap a naming method would fill.
""")

code(r"""
# One colour per method, the same colour in every figure of this notebook. Grey is kept for
# the shaded band only, so it never also means a method.
METHOD_COLOUR = {
    "exact tag match (protein20)": pf.OKABE_ITO["blue"],
    "kmerseek reduced alphabet (hp_thomas_dill, k=19)": pf.OKABE_ITO["vermillion"],
    "local alignment (BLOSUM62)": pf.OKABE_ITO["bluish_green"],
}
NAMING_METHODS = [
    "exact tag match (protein20)",
    "kmerseek reduced alphabet (hp_thomas_dill, k=19)",
]
N_HOSTS = recovery["host"].n_unique()

plotted = (
    recovery.filter(pl.col("method").is_in(NAMING_METHODS))
    .group_by(["method", "insert_kind", "tag_name", "tag_length_aa"])
    .agg(pl.col("flagged_named_right").eq("true").sum().alias("named_right"))
    .with_columns((pl.col("named_right") / N_HOSTS).alias("fraction"))
    .sort("tag_length_aa")
)

# Two panels, one per kind of insert, because the two kinds overplot each other badly in one.
# Within a panel the two methods often land on the same point, so exact matching is drawn
# first as a large pale circle and the reduced alphabet on top of it as a smaller solid one:
# where they agree, both are still visible.
fig, axs = pf.figure(pf.TWO_COLUMN_MM, 62, ncols=2, sharey=True)
for ax, insert_kind in zip(axs, ("exact tag", "tag with one substitution")):
    ax.axvspan(19, 500, color=pf.GREY, alpha=0.25,
               label="length at least k=19, so the tag has a k-mer")
    for method, markersize, alpha in (
        (NAMING_METHODS[0], 9, 0.55),
        (NAMING_METHODS[1], 4.5, 1.0),
    ):
        rows = plotted.filter(
            (pl.col("method") == method) & (pl.col("insert_kind") == insert_kind)
        )
        ax.plot(
            rows["tag_length_aa"],
            rows["fraction"],
            "o",
            color=METHOD_COLOUR[method],
            markersize=markersize,
            alpha=alpha,
            linestyle="none",
            label=method,
        )
    ax.set_xscale("log")
    ax.set_xticks([3, 5, 10, 19, 30, 100, 400])
    ax.set_xticklabels(["3", "5", "10", "19", "30", "100", "400"])
    ax.set_xlim(2.4, 500)
    ax.set_xlabel("Inserted sequence length (aa)")
    ax.set_title(insert_kind)
    ax.grid(axis="y", which="major")
axs[0].set_ylabel(f"Named correctly (fraction of {N_HOSTS} host proteins)")
axs[0].set_ylim(-0.08, 1.12)
pf.shared_legend(fig, ncol=1)
# Just above the axes corner, not out to the left, where panel a's letter would sit on top of
# the y-axis label.
for ax, letter in zip(axs, "ab"):
    pf.panel_label(ax, letter, dx_pt=-2, dy_pt=6)
pf.save(fig, FIGURES / "355-mechanics_named_right_by_length_built_constructs")
print(plotted.to_pandas().to_string(index=False))
print()
print("the step from 3 to 5 aa takes as much room as the step from 30 to 50, because the "
      "x spacing is by ratio")
print("saved figures/355-mechanics_named_right_by_length_built_constructs.png")
""")

md(r"""
## 6. Controls, and the false flag on a clean protein

Three controls, all on the same six clean host proteins:

- **Clean, nothing inserted.** Every flag here is a false one.
- **Shuffled insert.** Each long-enough tag shuffled, keeping its composition and length, then
  inserted. A flag on that region means the method is answering to composition, not to the
  tag's sequence.
- **Composition-matched natural stretch.** A real stretch of EGFR, chosen as the stretch whose
  amino-acid composition is closest to the tag it stands in for, inserted at a random position.

The second and third diagnose mechanics only and do not count toward the primary result.

Lower is better everywhere in this section. Threshold 2 of the pass rule allows at most one
false flag per 100 clean constructs.
""")

code(r"""
controls = pl.read_csv(TABLES / "355-mechanics_false_flags_on_controls.tsv", separator="\t")
kinds = (
    controls.with_columns(
        pl.when(pl.col("construct_kind") == "clean")
        .then(pl.lit("clean, nothing inserted"))
        .when(pl.col("construct_kind").str.starts_with("shuffled"))
        .then(pl.lit("shuffled insert"))
        .otherwise(pl.lit("composition-matched natural stretch"))
        .alias("control")
    )
    .group_by(["control", "method"])
    .agg(
        pl.col("n_flags_total").sum().alias("flags"),
        pl.col("n_flags_on_inserted_region").sum().alias("flags_on_inserted_region"),
        pl.len().alias("n_constructs"),
    )
    .with_columns(
        (100 * pl.col("flags") / pl.col("n_constructs")).round(1).alias("flags_per_100")
    )
    .sort(["control", "method"])
)
with pl.Config(fmt_str_lengths=60, tbl_rows=20, tbl_width_chars=200):
    print(kinds)

clean_only = controls.filter(pl.col("construct_kind") == "clean")
print()
print("every flag raised on a clean host protein, with what it named:")
print(
    clean_only.filter(pl.col("n_flags_total") > 0)
    .select("host", "method", "n_flags_total", "flag_names")
    .to_pandas()
    .to_string(index=False)
)
""")

md(r"""
**The false flag, residue by residue.**

A flag without its residues is not a result, so here are the residues. At hydrophobic/polar
`hp_thomas_dill` with k=19, the check flags residues 216-234 of Ced9, a clean protein with
nothing inserted, as containing the MBP fusion partner. The matching MBP window is residues
12-30 of the mature chain.

The two 19-residue stretches share 3 amino acids out of 19. They encode to the same 19-letter
hydrophobic/polar string, which is why the k-mer matched. This is the representation doing
exactly what it was built to do (ignore which hydrophobic residue it is) in a job where that is
the wrong thing to do (the residues are what identifies a tag).
""")

code(r"""
def load_hosts():
    # The frozen clean host proteins, as {short name: sequence}.
    hosts, name = {}, None
    for line in (FROZEN / "clean_host_proteins_swissprot.fasta").read_text().split("\n"):
        if line.startswith(">"):
            name = line[1:].split("|")[2].split(" ")[0]
            hosts[name] = []
        elif line.strip() and name:
            hosts[name].append(line.strip())
    return {key: "".join(parts) for key, parts in hosts.items()}


hosts = load_hosts()
ced9 = hosts["CED9_CAEEL"]
mbp = next(tag for tag in tags if tag.name.startswith("MBP"))

false_flags = cc.reduced_kmer_flags("CED9_CAEEL", ced9, tags, "hp_thomas_dill", 19)
for flag in false_flags:
    print(flag.evidence, f"-> received {flag.received_start}-{flag.received_end}")
    segment = ced9[flag.received_start - 1 : flag.received_end]
    encoded = cc.encode(segment, "hp_thomas_dill")
    start_in_mbp = cc.encode(mbp.sequence, "hp_thomas_dill").find(encoded)
    mbp_segment = mbp.sequence[start_in_mbp : start_in_mbp + len(encoded)]
    identical = sum(1 for a, b in zip(segment, mbp_segment) if a == b)
    print(f"  Ced9 {flag.received_start}-{flag.received_end}          {segment}")
    print(f"  {'':>24}{cc.match_line(segment, mbp_segment)}")
    print(f"  MBP  {start_in_mbp + 1}-{start_in_mbp + len(encoded)}             {mbp_segment}")
    print(f"  {identical} of {len(encoded)} identical amino acids")
    print(f"  Ced9 in hp_thomas_dill    {encoded}")
    print(f"  {'':>24}{cc.match_line(encoded, cc.encode(mbp_segment, 'hp_thomas_dill'))}")
    print(f"  MBP  in hp_thomas_dill    {cc.encode(mbp_segment, 'hp_thomas_dill')}")
    print(f"  {len(encoded)} of {len(encoded)} identical hydrophobic/polar letters")
    print()
print("hp_thomas_dill: h = ACFILMVWY, p = DEGHKNPQRST")
""")

md(r"""
**Figure 4. False flags per 100 clean constructs, and the one case in residues.**

Panel **a**: flags raised per 100 clean constructs, by method. Lower is better. The dashed line
is threshold 2 of the pass rule, at most one per 100. Both searching methods are far over it on
these six clean proteins: exact matching at 50 per 100 and the reduced-alphabet check at 17 per
100, against local alignment at 0.

The two come from different causes, and only one of them is a problem with the method. All
three of exact matching's flags are the three-residue glycine-serine linker `GSG`, which occurs
naturally in MCL1, BAX and EGFR; a facility would drop a three-residue entry from its list and
the rate would go to 0. The reduced-alphabet flag has no such fix: it is a 19-letter
hydrophobic/polar string shared between clean Ced9 and the MBP fusion partner, at the shortest
k this alphabet is run at, and raising k only loses the short tags faster.

Six clean proteins cannot measure a rate per 100. What the panel shows is which flags have a
fix and which do not, not a rate to quote.

Panel **b**: the reduced-alphabet false flag, residue by residue. Amino acids on top, the
hydrophobic/polar encoding underneath, with the match lines between them.
""")

code(r"""
clean_rate = (
    clean_only.group_by("method")
    .agg(pl.col("n_flags_total").sum().alias("flags"), pl.len().alias("n_constructs"))
    .with_columns((100 * pl.col("flags") / pl.col("n_constructs")).alias("per_100"))
    .sort("per_100")
)
print(clean_rate.to_pandas().to_string(index=False))

SHORT_NAME = {
    "exact tag match (protein20)": "Exact tag match\n(protein20)",
    "local alignment (BLOSUM62)": "Local alignment\n(BLOSUM62)",
    "kmerseek reduced alphabet (hp_thomas_dill, k=19)":
        "kmerseek reduced alphabet\n(hp_thomas_dill, k=19)",
}

fig, axs = pf.figure(pf.TWO_COLUMN_MM, 50, ncols=2, width_ratios=[1, 1.6])

ax = axs[0]
labels = [SHORT_NAME[method] for method in clean_rate["method"]]
ax.axvline(1.0, color=pf.OKABE_ITO["black"], linestyle="--",
           label="threshold 2: at most 1 per 100")
for index, (method, value) in enumerate(
    zip(clean_rate["method"], clean_rate["per_100"])
):
    ax.plot([value], [index], "o", color=METHOD_COLOUR[method], markersize=5,
            linestyle="none")
    # The value on the mark, so no point has to be read off the axis.
    ax.annotate(f"{value:.0f}", (value, index), textcoords="offset points",
                xytext=(0, 7), ha="center")
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)
ax.set_xlabel("False flags per 100 clean proteins (n = 6 proteins)")
# The x limit holds every value, including exact matching's 50: a point outside the view is
# a point the reader cannot see.
ax.set_xlim(-3, max(55, float(clean_rate["per_100"].max()) + 5))
ax.set_ylim(-0.6, len(labels) - 0.4)
ax.grid(axis="y", which="major")
ax.legend(loc="lower right", frameon=False)

ax = axs[1]
ax.axis("off")
segment = ced9[215:234]
mbp_encoded = cc.encode(mbp.sequence, "hp_thomas_dill")
start_in_mbp = mbp_encoded.find(cc.encode(segment, "hp_thomas_dill"))
mbp_segment = mbp.sequence[start_in_mbp : start_in_mbp + 19]
identical = sum(1 for a, b in zip(segment, mbp_segment) if a == b)
text_lines = [
    "Clean Ced9 flagged as containing MBP",
    f"Ced9 216-234      {segment}",
    f"                  {cc.match_line(segment, mbp_segment)}",
    f"MBP  {start_in_mbp + 1}-{start_in_mbp + 19}         {mbp_segment}",
    f"                  {identical} of 19 identical amino acids",
    "",
    f"Ced9 hp           {cc.encode(segment, 'hp_thomas_dill')}",
    f"                  {cc.match_line(cc.encode(segment, 'hp_thomas_dill'), cc.encode(mbp_segment, 'hp_thomas_dill'))}",
    f"MBP  hp           {cc.encode(mbp_segment, 'hp_thomas_dill')}",
    "                  19 of 19 identical h/p letters",
    "",
    "h = ACFILMVWY    p = DEGHKNPQRST",
]
ax.text(0, 1, "\n".join(text_lines), family="monospace", fontsize=5.5,
        va="top", ha="left", transform=ax.transAxes)

for ax, letter in zip(axs, "ab"):
    pf.panel_label(ax, letter)
pf.save(fig, FIGURES / "355-mechanics_false_flags_per_100_clean_and_ced9_mbp_residues")
print()
print("saved figures/355-mechanics_false_flags_per_100_clean_and_ced9_mbp_residues.png")
""")

md(r"""
## 7. The five comparison-missed cases: none, and why there cannot be any here

The task asks for the five documented errors the comparison workflow missed, shown as intended
against received with the insert highlighted. There are none to show, and the reason is not that
the method failed to find them: there are no documented errors at all, because there are no
submissions. A built construct cannot supply one. The mistake in it was put there by this
analysis, so "the comparison workflow missed it" would only mean "this analysis chose an insert
exact matching cannot see", which is a statement about the choice, not about the method.

What can be said from the mechanics check, on built constructs only:

- Of the 144 built constructs, the reduced-alphabet check at hydrophobic/polar k=19 named the
  inserted sequence correctly in 24, all of them GST or MBP, the only two frozen sequences at
  least 19 residues long. Exact matching found both as well, in every case.
- There is no built construct the reduced-alphabet check named and exact matching did not.
  On the mechanics evidence the count that threshold 3 needs to be at least five is zero.
- Local alignment flagged the inserted region in all 144, and in all 144 said nothing about
  what the insert was. That is the gap a naming method could fill. Exact matching fills it for
  every tag in the frozen list that is present exactly.

Section 8's verdict does not rest on these numbers. They are mechanics.
""")

code(r"""
reduced_method = "kmerseek reduced alphabet (hp_thomas_dill, k=19)"
exact_method = "exact tag match (protein20)"
named = recovery.filter(pl.col("flagged_named_right") == "true")
print(f"built constructs where the insert was named correctly, by method:")
print(named.group_by("method").agg(pl.len().alias("n")).sort("method").to_pandas().to_string(index=False))

wide = recovery.pivot(
    values="flagged_named_right", index=["tag_name", "host", "insert_kind"], on="method"
)
only_reduced = wide.filter(
    (pl.col(reduced_method) == "true") & (pl.col(exact_method) != "true")
)
print()
print(f"built constructs the reduced alphabet named and exact matching did not: "
      f"{only_reduced.height}")
if only_reduced.height:
    print(only_reduced.to_pandas().to_string(index=False))
""")

md(r"""
## 8. Summary and conclusions

**Verdict: NOT RUN, missing data.** The kill test was not run and no part of its pass rule was
measured, because the 200 historical facility construct submissions do not exist in this
repository and cannot be obtained anywhere else. Nothing was substituted for them.

### Each threshold, with its number

| Threshold | Required | Result |
| --- | --- | --- |
| 1. Documented unexpected inserts recovered | at least 95% | not measurable: no submissions, so no documented insert |
| 2. False flags per 100 clean constructs | at most 1 | not measurable: no clean submissions. On six clean Swiss-Prot proteins, which cannot measure a rate per 100, the reduced-alphabet check raised 1 flag and exact matching raised 3 |
| 3. Documented errors the comparison workflow missed | at least 5 | not measurable: no discrepancy records. On built constructs the count is 0 |
| 4. Median review time per batch | no increase | not measurable: no recorded review times |

### Claims, each with the cell that produced it

1. No facility construct submission record exists in this repository or in any of its 174 pull
   requests. Searched before concluding it; section 2.
2. Eight of the twelve frozen tag and linker sequences are shorter than 19 amino acids, so at
   the k a hydrophobic/polar alphabet is run at they have no k-mer and a k-mer method cannot
   find them at all. Section 3 table and figure 2.
3. On 144 constructs built by inserting a frozen tag into a clean protein, the reduced-alphabet
   check at hydrophobic/polar k=19 named the insert correctly in 24, all of them the GST or MBP
   fusion partner. Exact matching named all 72 exact-tag inserts and none of the 72 with one
   residue changed. Section 5 table and figure 3.
4. There is no built construct the reduced-alphabet check named and exact matching did not.
   Section 7.
5. Local alignment flagged the inserted region in all 144 built constructs and named the insert
   in none. Section 5 table.
6. At hydrophobic/polar k=19 the check flags residues 216-234 of clean Ced9 as containing MBP.
   The two 19-residue stretches share 3 amino acids out of 19 and encode to the same 19-letter
   hydrophobic/polar string. Section 6 printout and figure 4b.

Claims 2 to 6 are about the frozen tag list and sequences this analysis built. None of them is
a result of the kill test, and none of them can turn its verdict into a PASS or a FAIL.

### What this does say about the prior

The prior, written before running anything, was strongly unfavorable, for the reason that a
construct insert is usually known exactly while a reduced alphabet exists to find sequence that
is not. Two things found here point the same way, and neither needs the missing data:

- Most of the tag list is shorter than one k-mer at the k this alphabet is run at. That is
  arithmetic, not a measurement, and no amount of data changes it. Anything short enough to
  need a different method (His6, FLAG, HA, Myc, Strep, the short linkers) is also short enough
  for an exact search to find reliably.
- Where the reduced alphabet does reach, it matched a clean protein to a fusion partner on 3
  identical amino acids out of 19. On real submissions that is the behaviour threshold 2 is
  there to catch.

This is not the verdict. The verdict is NOT RUN. If the submissions arrive, the test runs
against the rule committed here, and these two observations are predictions it can falsify.

### What Olga must supply for the test to run

1. The 200 submissions as one tab-separated file in the format of section 2, including clean
   constructs, and with every unexpected insert's resolved sequence and its coordinates in the
   received protein.
2. The recorded review minutes per batch, before and after. Without them threshold 4 is not
   assessable, and not assessable is not a pass.
3. The DNA where the facility kept it, plus its own DNA-level call file, so the DNA check is in
   the comparison.
4. Any resolved insert sequence that is not one of the twelve standard tags and linkers frozen
   here, with its source.

### Compute

| | |
| --- | --- |
| CPU-hours | about 0.05 (3 minutes of one core: the mechanics check runs in 7 seconds, the unit tests in 0.3 seconds, the rest is downloading 10 UniProt records) |
| Elapsed hours | about 2.5, one session, all on the laptop |
| Budget | 2 compute-days. Used well under 1% of it |
| GPU | none |
| kmerseek index builds | none. No index was built and no `nextflow run` was started, so nothing ran on the laptop that the house rule forbids |
| Sherlock jobs | none needed |

### Limits of what is here

- The reduced-alphabet check is kmerseek's seeding step reimplemented in Python, not the
  kmerseek binary. It is faithful for the question of which sequences share an identical
  reduced k-mer, which is what decides whether a tag can be found at all, and it needs no
  index. A real run would use kmerseek and would also apply its scoring and E-value, which
  can only remove flags, not add them.
- Six clean host proteins cannot measure a false-flag rate per 100.
- The single false flag in section 6 is one case, not a rate.
- `choose_alphabet_and_k` has never been run on real data, so the alphabet and k are not
  chosen. Only the rule for choosing them is fixed.

## 9. Second agent's attempt to disprove these claims

A second agent, in a fresh session, went through the claims above against the code and data.
Its held and failed list is in the pull request body and is repeated here.
""")

md(r"""
_The held/failed list is filled in below once the second agent has reported._
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "2025-kmerseek-analysis",
            "language": "python",
            "name": "2025-kmerseek-analysis",
        },
        "language_info": {"name": "python", "version": "3.13"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "355_construct_check_kill_test_not_run_missing_submissions.ipynb"
)
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({len(cells)} cells)")
