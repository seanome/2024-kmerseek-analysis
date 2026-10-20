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

Ten of the twelve frozen tag and linker sequences are shorter than 19 amino acids, and a
sequence shorter than k has no k-mers at all, so a k-mer method cannot find it at that k
however good the alphabet is. The pipelines in this repository run a hydrophobic/polar
alphabet from k=15 to k=30, and across that whole range the number of the twelve sequences
with even one k-mer is 3 at k=15 and 2 from k=16 up. The two that are always long enough are
the GST and MBP fusion partners, at 218 and 370 amino acids. This is arithmetic about the tag
list, not a measurement, and no amount of missing data changes it.

Where the reduced alphabet does reach, it adds one thing over exact matching and costs
another. The thing it adds: on constructs built here, it named GST and MBP even when one
residue had been changed, which exact matching cannot do, in 12 of 144 cases. The cost: at
hydrophobic/polar k=19 it flagged a clean Ced9 protein as containing MBP. Those 19 residues of
Ced9 and 19 of MBP share 3 amino acids out of 19 and encode to the same 19-letter
hydrophobic/polar string. Sections 5 to 7 give both numbers and section 6 prints the
residues.
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

Searched before concluding it does not exist: every file in this repository, the pull requests
in `seanome/2024-kmerseek-analysis` open or closed at the time of the search on 2026-10-20
(174 then, 179 by the end of that day), and the data directories on this machine. Outside this
notebook's own files, "construct" appears in the repository only as the ordinary verb, in code
that builds a table or a set of pairs. A sibling branch opened later the same day,
`olgabot/construct-insert-check`, asks the same question with its own scaffolding; its
submission templates are headers with no rows, so it has no submissions either. There is no
facility submission record anywhere.

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
**Figure 1. What the test needs, what it does with it, and what comes out.** Solid black boxes are
in this repository. Dashed purple boxes are the parts that are not available, and every one of
them is an input: the submissions and the facility's DNA calls and review times. The four
thresholds on the right all depend on an input that is missing, which is why none of them has
a number.
""")

code(r"""
from matplotlib.patches import FancyArrowPatch, Rectangle

# Blue, vermillion and bluish green are reserved for the three methods in figures 3 and 4,
# so this figure marks present and missing with black and reddish purple instead.
PRESENT = pf.OKABE_ITO["black"]
MISSING = pf.OKABE_ITO["reddish_purple"]

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
is visible rather than a summary; the marker shape is the tag's category.

Higher is better in panel **a**: it is the number of tags the method could possibly find. The
shaded band is k=15 to k=30, the range the pipelines in this repository run a hydrophobic/polar
alphabet at (`nextflow-runs/kmer-spectra/main.nf` starts at 15, the Pfam region benchmark at
16). The dashed line in panel **b** is k=19, the value the mechanics check uses. Across the
whole band the count in panel **a** is 3 at k=15 and 2 from k=16 up. Nothing about this figure
depends on the missing submissions.
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

# The range this repository's own pipelines run a hydrophobic/polar alphabet at, and the k
# the mechanics check uses inside it.
HP_LOW, HP_HIGH = 15, 30
MECHANICS_K = 19
# Category on marker shape, in one colour: hue is spoken for by the methods in figures 3 and
# 4, and a tag category is not an amount, so shape is the right channel for it.
CATEGORY_MARKER = {
    "affinity tag": "o",
    "epitope tag": "s",
    "linker": "^",
    "fusion partner": "D",
}
CATEGORY_COLOUR_SINGLE = pf.OKABE_ITO["black"]

fig, axs = pf.figure(pf.TWO_COLUMN_MM, 62, ncols=2)

ax = axs[0]
ax.axvspan(
    HP_LOW,
    HP_HIGH,
    color=pf.GREY,
    alpha=0.25,
    label=f"k this repository runs a hydrophobic/polar alphabet at ({HP_LOW}-{HP_HIGH})",
)
# No marker on this line: a circle is the affinity-tag shape in panel b, and the two panels
# share one legend.
ax.plot(reachable["ksize"], reachable["n_tags"], color=pf.OKABE_ITO["black"],
        label="frozen tags with at least one k-mer")
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
for category, marker in CATEGORY_MARKER.items():
    rows = ordered.filter(pl.col("category") == category)
    if rows.is_empty():
        continue
    axs[1].plot(
        rows["tag_length_aa"],
        [positions[name] for name in rows["tag_name"]],
        marker=marker,
        color=CATEGORY_COLOUR_SINGLE,
        markersize=4,
        linestyle="none",
        label=category,
    )
# Panel b's x axis is a length, not k, so the grey band of panel a would mean something
# different here. A dashed line, its own mark with its own legend entry, instead.
axs[1].axvline(
    MECHANICS_K,
    color=pf.OKABE_ITO["black"],
    linestyle="--",
    label=f"{MECHANICS_K} aa, the k the mechanics check uses",
)
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

Thirty-one unit checks run on the frozen tag list and on constructs built from the frozen host
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

The reduced-alphabet check runs at hydrophobic/polar `hp_thomas_dill` with k=19. The pipelines
in this repository run this alphabet from k=15 to k=30, so 19 is inside that range and not the
lowest value available. Lowering it to 15 would add exactly one more reachable sequence, the
15-residue glycine-serine linker, and nothing else; raising it reaches no more. So the choice
of 19 is not what decides the result.
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
alphabet did not. In panel **b**, the two orange points at 1.0 with no blue under them are the
12 constructs the reduced alphabet named and exact matching missed: GST and MBP with one
residue changed, in all six hosts.

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
    ax.axvline(
        19,
        color=pf.OKABE_ITO["black"],
        linestyle="--",
        label="19 aa: to the right of this, a tag has a k-mer at k=19",
    )
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
ax.axvline(
    1.0,
    color=pf.GREY,
    linestyle=":",
    label="threshold 2: at most 1 per 100",
)
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
## 7. The cases only the reduced alphabet named, and why none of them counts

Threshold 3 asks for at least five documented errors that the comparison workflow missed. No
documented error exists here, so the threshold has no number. A built construct cannot supply
one: the mistake in it was put there by this analysis, so "the comparison workflow missed it"
would only say "this analysis chose an insert exact matching cannot see", which is a fact about
the choice and not about the method.

What the mechanics check does show, on built constructs only:

- Of the 144 built constructs, the reduced-alphabet check at hydrophobic/polar k=19 named the
  inserted sequence correctly in 24: GST and MBP, the only two frozen sequences long enough to
  have a k-mer, in all six hosts, both as the exact sequence and with one residue changed.
- In 12 of those 144, the reduced alphabet named the insert and exact matching did not. All 12
  are GST or MBP with one residue changed, which is exactly the case an exact substring search
  must lose. This is the one thing in the whole notebook pointing the favourable way, and it is
  narrow: it needs an insert longer than k, and local alignment already flagged every one of
  those 12 regions, just without naming what was in them.
- Local alignment flagged the inserted region in all 144 and named the insert in none.

The cell below prints both counts. The 12 is not a count of documented errors and cannot be
read as progress toward the five that threshold 3 asks for: these are inserts this analysis
planted, not mistakes a facility made and resolved. Section 8's verdict does not rest on any of
these numbers.
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
2. Ten of the twelve frozen tag and linker sequences are shorter than 19 amino acids. Across
   k=15 to k=30, the range this repository runs a hydrophobic/polar alphabet at, the number of
   the twelve with even one k-mer is 3 at k=15 and 2 from k=16 up. Section 3 table and
   figure 2.
3. On 144 constructs built by inserting a frozen tag into a clean protein, the reduced-alphabet
   check at hydrophobic/polar k=19 named the insert correctly in 24, all of them the GST or MBP
   fusion partner. Exact matching named all 72 exact-tag inserts and none of the 72 with one
   residue changed. Section 5 table and figure 3.
4. In 12 of the 144 built constructs the reduced-alphabet check named the insert and exact
   matching did not; all 12 are GST or MBP with one residue changed, and local alignment had
   already flagged all 12 regions without naming them. Section 7.
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

- Ten of the twelve sequences are shorter than one k-mer across the whole range this alphabet
  is run at. That is arithmetic, not a measurement, and no amount of data changes it. Anything
  short enough to need a different method (His6, FLAG, HA, Myc, Strep, the short linkers) is
  also short enough for an exact search to find reliably.
- Where the reduced alphabet does reach, it matched a clean protein to a fusion partner on 3
  identical amino acids out of 19. On real submissions that is the behaviour threshold 2 is
  there to catch.

One thing points the other way, and it is recorded here rather than left out: for an insert
longer than k that is not an exact copy of the list entry, the reduced alphabet named it and
exact matching did not, in 12 of 144 built constructs. Whether that matters depends on how
often a real discrepancy is a long, inexact insert rather than a short exact tag, which is a
question about the 200 submissions and cannot be answered without them.

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
| CPU-hours | 0.05 or less. Measured with `/usr/bin/time -p`: the mechanics check 1.64 s user time, the 31 unit checks 1.50 s, the notebook about 25 s to execute. The rest is downloading 10 UniProt records and 8 Crossref records |
| Elapsed hours | about 3.5, one session, all on the laptop |
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

A second agent, in a fresh session, with no part of the work's reasoning, was asked to try to
disprove each claim against the code and data. It checked ten of them, re-ran the mechanics
check, resolved every DOI against Crossref, wrote 24 malformed submissions files against the
loader, and built its own boundary cases for the scorer. Five claims failed. Everything below
was fixed before this notebook was called done, and the numbers in sections 1 to 8 are the
corrected ones.

| Claim | Its verdict | What it found, and what was done |
| --- | --- | --- |
| 1. No facility submissions exist here | HELD | Grepped the whole tree and all history; the only hits are this notebook's own files and a sibling branch whose submission templates are headers with no rows. It also found the sentence "construct appears only as the ordinary verb" to be false against `git log --all`, and that a sibling pull request asks the same question. Section 2 now says both. |
| 2. Eight sequences shorter than 19 aa | FAILED | It is ten of twelve, which this notebook's own cell prints; the prose said eight in three places. It also showed 19 is not the lowest k this repository runs a hydrophobic/polar alphabet at: the k-mer spectra pipeline starts at 15 and the Pfam region benchmark at 16. Both corrected, and figure 2's band is now the real k=15 to 30 range. |
| 3. 24 named right, 72 and 0 for exact matching | HELD | Recomputed from the table and by re-running the script; the three tables came back byte-identical, so the run is deterministic. 432 rows, 144 per method, 144 distinct tag, host and insert-kind groups. |
| 4. Nothing the reduced alphabet named that exact matching did not | FAILED | There are 12, which this notebook's own cell prints: GST and MBP with one residue changed, in all six hosts. The claim contradicted the cell it cited. Section 7 and summary claim 4 rewritten; this is now recorded as the one result pointing the favourable way. |
| 5. The Ced9/MBP false flag | HELD | Verified the 1-based coordinates against the frozen FASTA, the 3 identical amino acids, the identical hydrophobic/polar strings, and that MBP 12-30 is inside the frozen mature chain. |
| 6. Every tag source and citation is correct | HELD, two errors | Every UniProt subsequence is at its recorded accession and coordinates, and all eight DOIs resolve to the paper claimed, with matching title, journal, volume, year and pages. No invented citation. Two given names were wrong: Evan 1985 (George K. Lewis, Gary Ramsay, not Gerald and Gregor) and Hopp 1988 (Douglas Pat Cerretti, David L. Urdal, not Dale and Douglas). Both corrected against Crossref and re-frozen. |
| 7. The scorer enforces the frozen rule | FAILED | The thresholds and boundary cases were right (19 of 20 holds, 18 of 20 fails, exactly 1 per 100 holds, and an unmeasured threshold 4 does block a PASS), and `Flag.overlaps` has no off-by-one. But the scorer computed only two of the three frozen drop clauses: `cannot_separate_intended` was never evaluated, so a method that cannot tell an ordered insert from a mistaken one could still return PASS. It also hardcoded the three thresholds instead of reading them from the frozen rule. Both fixed, with three new tests, including one that fires the clause and one that shows it staying quiet when the flags do separate. |
| 8. The loader refuses a file that breaks the contract | FAILED | Of 24 malformed files, 11 were correctly refused and 5 kinds got through: a header-only file returned zero rows; the contract's split rule, that one intended protein stays in one split, was documented but never checked; `strip()` dropped a blank final field; a short row raised `KeyError` instead of being reported; and a repeated header column was silently accepted. All five fixed, each with its own test. |
| 9. The checksums and the freeze | HELD | All 16 files verify, and re-running the freeze left `git status` clean. It noted that `frozen_date` is today's date, so re-running on another day changes four files; that is intended, since a re-freeze is a new freeze. |
| 10. The figures do not mislead | FAILED | Every number checked out and no point falls outside an axis. But blue, vermillion and green each meant three different things across the four figures, and the grey band meant a k range in one and a length range in another. The colours are now reserved for the three methods, tag category moved to marker shape, present and missing to black and purple, and the 19 aa marker is one dashed line used the same way everywhere. |

Other things it raised, and what was done:

- The `cannot_separate_intended` clause being unenforceable was counted as a real error. Fixed,
  as above.
- The composition-matched control takes a stretch of EGFR, so for the EGFR host it inserts EGFR
  into EGFR. It still controls for composition, but that one case is not a neutral stretch. Left
  as it is, and said here, because it affects only the mechanics check.
- `insert_was_intended` is only checked on insert rows, so a nonsense value on a clean row
  passes. Left as it is: the column is only read for inserts.
- The compute numbers were prose with no cell behind them. They are now measured values from
  `/usr/bin/time -p`, stated in section 8.
- It could not check whether the 200 submissions exist outside this repository, which is the one
  thing that would change the verdict. Only Olga can answer that.
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
