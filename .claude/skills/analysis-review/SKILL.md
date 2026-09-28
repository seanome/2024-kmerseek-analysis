---
name: analysis-review
description: Review a Jupyter notebook, a notebook helper module, or a PR in 2024-kmerseek-analysis for analysis mistakes a linter cannot see. Trigger when asked to review, check or "look over" a notebook, a figure notebook, a results PR or a claim in one; before opening a PR that adds or changes a notebook; and when finishing a notebook. It checks every number in the text against a printed output, ties, nulls, joins, sort order, which arm and query set a number comes from, and the project's naming rules. When the review is of a PR, it posts the review as one PR comment, in the same layout as /rustacean-review. /rustacean-review covers Rust; this one covers the analysis.
---

# Analysis review for 2024-kmerseek-analysis notebooks

Added 2026-09-25 at Olga's request. The rules below came from mistakes already made in this
repo, each recorded in the project memory when it happened. The mechanical checks (error
outputs, run order, collapsed cells, `plt.show()`, secrets, undefined names) are in
`scripts/check_notebooks.py` and the pre-commit hooks. This skill covers what needs reading.

## Step 1: run the mechanical checks first

```bash
pre-commit run --files <changed notebooks>
```

If pre-commit is not installed, run `python scripts/check_notebooks.py <notebooks>`. Report
its failures as they are. Do not repeat them below.

## Step 2: read every cell against these rules

Work through the notebook top to bottom. For each finding, give the cell index, the rule,
the text or code, and what goes wrong. Check each finding against the saved outputs before
reporting it. If you cannot confirm one, label it "unverified".

### Numbers and claims

1. **Every number in a markdown cell, figure title or conclusion appears in a printed output
   of the same notebook.** Search the outputs for it. A number with no printed source is a
   finding, even if it is probably right. Every number also comes from committed code, not
   from a chat session or a file in `~/Downloads`.
2. **Every biological claim is backed by data queried in the notebook**: a family identity,
   a function, a domain name. Nothing from memory. For SCOPe-labelled hits, load the
   `scope-benchmark-rules` skill before calling a hit interesting.
3. **Every plotting cell prints the table it draws**, or says the table is in the next cell.
   Every summary statistic has a figure.

### Rankings and statistics

4. **A reported rank prints `n_tied` beside it.** A metric that is constant per query
   (`query_tfidf`) or zero for a whole arm (`region_ka_bits` without a lambda) ties the
   whole proteome at rank 1. Nb 241 had 18_302 proteins tied.
5. **Compare ranks as a percent of the hit list**, not as raw ranks: 213 of 18_064 and 173 of
   1_161 are different results.
6. **A best result over many arms or metrics has a random-protein null**: the same
   statistic for a protein drawn from the same hit lists over the same arms.
   `au.random_protein_null()` in `notebooks/alphabet_ranking_utils.py` does this (added in
   PR #46; not on main as of 2026-09-25, so check it exists first). In nb 241,
   BCL2's best rank of 213 was beaten by a random protein in 97% of draws.
7. **An ordered read sorts on a total key**: score, then an id column. Sorting on score alone
   leaves ties in polars' arbitrary row order. `sens_first_fp` moved 56% between two
   identical runs.

### polars traps

8. **After a left join, `pl.min_horizontal` / `pl.max_horizontal` skip nulls.** Guard with
   `pl.when(col.is_null())`. This once turned every false positive into IoU 1.0.
9. **No `with_row_index` followed by an anti-join in a lazy plan to take a sample.** It kept
   a different random subset on each run.
10. **No `scan_csv` on a `.csv.zst`.** It inflates the whole file in memory. Use the streaming
    converter.

### Which data a number comes from

11. **Query counts show the total and the FoldSeek-intersected set separately**
    (`n_queries_all` and `n_queries_ref`).
12. **The dedup setting is stated.** The MultiQC region-benchmark tables use
    `dedup_transfers=False`, and nb 236 uses `True`. On the dedup arm the mammal window is a
    tie with phmmer. A comparison across the two is a finding.
13. **kmerseek region coordinates are 0-based and end-exclusive**; the Pfam and Swiss-Prot
    truth tables are 1-based and inclusive. Look for `seq[start:end]` on kmerseek regions
    and a conversion wherever the two are compared.
14. **Domain recall at IoU ≥ 0.5 measures where the call lands, not whether the domain was
    found.** A sentence that reads it as detection is a finding.
15. **The HP arm is `hp_pbotc_1st_ed` k=19** unless the notebook says why it uses another
    one. Notebooks 075–079 still use `thomas_dill` k=26.
16. **A sequence pair shown in the text prints its residues**: both sequences, coordinates,
    a match line and the count. kmerseek alignments are ungapped, so no gaps.

### Reuse and pipeline boundaries

17. **New AUC, confidence-interval, ranking or plotting code is checked against the shared
    modules** in `notebooks/` (`ortholog_analysis_utils.py`, `scope_kmerseek_utils.py`,
    `mhc_region_utils.py`, `hp_conservation_utils.py`, and the others). Use
    `ou.load_all_alphabet_ksize_combos()` for the list of alphabet and k-mer size
    combinations.
18. **No Nextflow pipeline reads a file a notebook wrote.** Pipelines run before notebooks.

### Words

19. Run the `writing-style` skill on every markdown cell and the `clear-figures` checklist on
    every figure. In this repo, also:
    - FoldSeek missed a protein; it is not "outside the benchmark".
    - No "twilight zone" framing (dropped 2026-09-13).
    - A grid entry is a "square" or "entry", never a "cell".
    - "Sweep" means only the alphabet × k-mer size parameter sweep.
    - Integers are written `14_873` in code (the script catches `= 14,873`).

## Step 3: write the review

Use the same layout as a `/rustacean-review` comment. Write it to a file in the scratchpad,
not straight into a command.

```markdown
> [!NOTE]
> Written by Claude Code (<model>) at olgabot's request. Posted from her account, so replies here are to Claude, not to her.

## /analysis-review, run <YYYY-MM-DD HH:MM> UTC

**Commit:** `<short sha>`, the PR's head when the review was read.

## Verdict
2 or 3 sentences: can this merge, and the one thing that matters most.

## Tooling
The commit reviewed, and what ran: `pre-commit run --files ...` per hook, and any cell or
number you re-computed. What could not run and why (for example, the input is on Sherlock).

## Blocking
Wrong numbers, claims with no printed source, a statistic that does not hold. Often empty.

## Should fix
Missing n_tied or null, unstated dedup setting or query set, a reimplemented helper.

## Consider
Wording and judgement calls the author may decline.

## Verified clean
What you checked and found right, named specifically ("all 14 numbers in cell 7 match the
table printed in cell 6").
```

Each finding gives: the notebook and cell (`notebooks/241_x.ipynb`, cell 7), what is wrong in
one sentence, what goes wrong because of it, the fix, and how sure you are. If you
re-computed a number, say so and give both values.

Do not edit the notebook unless asked.

## Step 4: post it on the PR

When the review is of a PR, end the review file with this line, filled in with the head
commit you reviewed and today's date:

```markdown
<!-- analysis-review sha=<head sha> date=<YYYY-MM-DD> -->
```

The heading, time and commit at the top are what Olga reads on the PR to see that a review
happened and which commit it covered; this hidden line only feeds the weekly table. Never post
the line without the visible heading and the review.

The weekly PR status (`~/.claude/skills/weekly-pr-status`) finds reviews by this line. It
marks the PR reviewed when the SHA is the PR's head commit, and out of date when commits came
after it. Get the SHA with
`gh api repos/seanome/2024-kmerseek-analysis/pulls/<n> --jq .head.sha`.

Post it as one comment on the PR. Use `gh api`, never `gh pr comment` or `gh pr edit`:

```bash
gh api -X POST repos/seanome/2024-kmerseek-analysis/issues/<n>/comments -F body=@<review.md> --jq .html_url
```

Then read it back and check the Verdict came through:

```bash
gh api repos/seanome/2024-kmerseek-analysis/issues/<n>/comments --jq '.[-1].body' | head -8
```

Give Olga the link. If the review is of local changes with no PR, show it in the chat
instead.

When a review finds a mistake that no rule here covers, add the rule to this file with the
date and the notebook it came from.
