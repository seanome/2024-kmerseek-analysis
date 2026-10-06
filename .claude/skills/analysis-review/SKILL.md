---
name: analysis-review
description: Review a Jupyter notebook, a notebook helper module, or a PR in 2024-kmerseek-analysis for analysis mistakes a linter cannot see. Trigger when asked to review, check or "look over" a notebook, a figure notebook, a results PR or a claim in one; before opening a PR that adds or changes a notebook; and when finishing a notebook. It tries to prove the headline result wrong, checks every number in the text against a printed output and every sentence against what its number supports, re-computes the main number a second way, and checks ties, nulls, joins, sort order, stale outputs, and which arm and query set a number comes from. Before posting, it runs the are-you-sure pass on its own findings. When the review is of a PR, it posts the review as one PR comment, in the same layout as /rustacean-review. /rustacean-review covers Rust; this one covers the analysis.
---

# Analysis review for 2024-kmerseek-analysis notebooks

Added 2026-09-25 at Olga's request. The rules below came from mistakes already made in this
repo, each recorded in the project memory when it happened. The mechanical checks (error
outputs, printed tracebacks, run order, collapsed cells, `plt.show()`, integers written with
commas, secrets, undefined names) are in `scripts/check_notebooks.py` and the pre-commit
hooks. This skill covers what needs reading.

Changed 2026-10-01 after an adversarial review of PR #56 found that the skill checked that
each number existed but not whether the sentence around it was true, trusted saved outputs
as current, and never checked its own findings. Steps 0, 2, 3 and 5, rules 2, 9–11 and
21–26, and the section "Preparing for an Adversarial Reviewer 2: Stress-testing the
findings" were added then. A hook now refuses to post a review comment without that
section (see Step 6).

## How to read the work

Assume the headline result is wrong and try to show it. A review that only confirms what
the author wrote has not been done. On 2026-09-29 one "Are you sure?" pass on the
experiments page found 7 sentences that claimed more than their numbers showed, after a
570-number audit had passed every one of them.

## Step 0: review the right commit

For a PR, review its head commit, not whatever is checked out:

```bash
GH_PAGER=cat gh api repos/seanome/2024-kmerseek-analysis/pulls/<n> --jq '.head.sha, .head.ref'
cd /Users/olga/code/2024-kmerseek-analysis && git fetch origin <head ref> && git worktree add /tmp/review-<n> <head sha>
```

List every changed file with `git diff --stat origin/main...<head sha>`. Review the changed
helper modules (`notebooks/*_utils.py`, `ortholog_analysis_utils.py`, ...) as well as the
notebooks: a notebook number can move when only a helper changed. For each changed helper,
list the notebooks that import it.

Then grep the project memory for the notebook's topic before reading, so known traps are
in mind:

```bash
grep -il '<topic word>' ~/.claude/projects/-Users-olga-code-2024-kmerseek-analysis/memory/*.md
```

## Step 1: run the mechanical checks

```bash
python -m pytest scripts/tests -q
pre-commit run --files <changed notebooks>
```

If pre-commit is not installed, run `python scripts/check_notebooks.py <notebooks>`. Report
the failures as they are. Do not repeat them below.

## Step 2: write down the claims, then attack them

Before reading rule by rule, list the notebook's claims: every sentence in a markdown cell,
figure title or conclusion that states a result. For each one, write the weakest
statement its printed number supports. Then list the three cheapest ways the headline
could be wrong (a wrong denominator, a null that was never run, a stale input, a different
query set, ties) and test each one in Steps 3 and 4.

## Step 3: re-compute the headline number

Re-compute the notebook's main number a second way, from the input file, without the
notebook's helper: a polars query written differently, `wc -l`, a row picked by hand. Give
both values. If the input is not reachable (on Sherlock or S3), say so and re-compute the
largest number you can reach.

Check that the saved outputs are current. Compare the date the notebook was last run
(the newest output, or the commit that last changed its outputs) with the last commit of
each helper it imports and the date of each input file. An output older than a change to
its code or data is stale: a finding, and every number from it is unverified.

## Step 4: read every cell against these rules

Work through the notebook top to bottom. For each finding, give the cell index, the rule,
the text or code, and what goes wrong.

### Numbers and claims

1. **Every number in a markdown cell, figure title or conclusion appears in a printed output
   of the same notebook.** Search the outputs for it. A number with no printed source is a
   finding, even if it is probably right. Every number also comes from committed code, not
   from a chat session or a file in `~/Downloads`. Rounding, a percent printed as a
   fraction, and a ratio of two printed numbers ("4x") count as found only after you do the
   arithmetic.
2. **Every sentence claims no more than its number shows.** Compare each claim with the
   weakest statement from Step 2. "Better" needs a difference larger than the spread;
   "all" needs a count equal to the total; "detects" needs a detection metric, not a
   boundary metric (rule 14). This is the rule the 2026-09-29 audit missed.
3. **Every biological claim is backed by data queried in the notebook**: a family identity,
   a function, a domain name. Nothing from memory. For SCOPe-labelled hits, load the
   `scope-benchmark-rules` skill before calling a hit interesting.
4. **Every plotting cell prints the table it draws**, or says the table is in the next cell.
   Every summary statistic has a figure.

### Rankings and statistics

5. **A reported rank prints `n_tied` beside it.** A metric that is constant per query
   (`query_tfidf`) or zero for a whole arm (`region_ka_bits` without a lambda) ties the
   whole proteome at rank 1. Nb 241 had 18_302 proteins tied.
6. **Compare ranks as a percent of the hit list**, not as raw ranks: 213 of 18_064 and 173 of
   1_161 are different results.
7. **A best result over many arms or metrics has a random-protein null**: the same
   statistic for a protein drawn from the same hit lists over the same arms.
   `au.random_protein_null()` in `notebooks/alphabet_ranking_utils.py` does this (added in
   PR #46; check it exists on the branch you review). In nb 241, BCL2's best rank of 213
   was beaten by a random protein in 97% of draws.
8. **An ordered read sorts on a total key**: score, then an id column. Sorting on score alone
   leaves ties in polars' arbitrary row order. `sens_first_fp` moved 56% between two
   identical runs.
9. **Every comparison prints its sample size and its spread** (a confidence interval, a
   bootstrap range, or the per-species values). A difference smaller than the spread is not
   a result.
10. **Related proteins are not independent samples.** Orthologs, paralogs and members of one
    Pfam family share sequence. A test or interval that counts each one as a separate draw
    is too narrow; group by family or by orthogroup.
11. **Nothing chosen after seeing the result is reported as if chosen before.** An example
    protein, a threshold, an alphabet or a k-mer size picked because it looked best needs
    held-out data or the null in rule 7. Cross-validation groups related proteins in the
    same fold (GroupKFold), and a tie in the group key can leak a family across folds.

### polars traps

12. **After a left join, `pl.min_horizontal` / `pl.max_horizontal` skip nulls.** Guard with
    `pl.when(col.is_null())`. This once turned every false positive into IoU 1.0.
13. **No `with_row_index` followed by an anti-join in a lazy plan to take a sample.** It kept
    a different random subset on each run.
14. **No `scan_csv` on a `.csv.zst`.** It inflates the whole file in memory. Use the
    streaming converter.

### Which data a number comes from

15. **Query counts show the total and the FoldSeek-intersected set separately**
    (`n_queries_all` and `n_queries_ref`).
16. **The dedup setting is stated.** The MultiQC region-benchmark tables use
    `dedup_transfers=False`, and nb 236 uses `True`. On the dedup arm the mammal window is a
    tie with phmmer. A comparison across the two is a finding.
17. **kmerseek region coordinates are 0-based and end-exclusive**; the Pfam and Swiss-Prot
    truth tables are 1-based and inclusive. Look for `seq[start:end]` on kmerseek regions
    and a conversion wherever the two are compared.
18. **Domain recall at IoU ≥ 0.5 measures where the call lands, not whether the domain was
    found.** A sentence that reads it as detection is a finding.
19. **The HP arm is `hp_pbotc_1st_ed` k=19** unless the notebook says why it uses another
    one. Notebooks 075–079 still use `thomas_dill` k=26.
20. **A sequence pair shown in the text prints its residues**: both sequences, coordinates,
    a match line and the count. kmerseek alignments are ungapped, so no gaps.
21. **Every arm searched every query.** 17 pipeline combinations searched only about 1_000
    of the 19_696 human genes, and their missing hits look like real negatives. Check the
    query count per arm before comparing arms.
22. **No Swiss-Prot truth arm as a species ranking.** Ciona has 28 reviewed entries, so the
    arm ranks species by how well curated they are.
23. **Filtering a clade out of the hits is not the same as leaving it out of the index.**
    kmerseek computes IDF when the index is built. Filtering afterwards is close enough when
    the clade is small (Bivalvia, 0.05% of the index) and not when it is large
    (Gammaproteobacteria, 21.56%).
24. **An empty download is checked before it is cached.** BioMart serves an outage as an
    HTML page with HTTP 200, so a cached empty result can read as a real zero.
25. **Input paths are the current files.** `ou.py` once read a five-month-old MGI snapshot by
    default. Check the path and date of every input file the notebook names.
26. **Pfam truth favours methods that find Pfam-like domains.** A sentence that reads a
    Pfam-recall gap as a sensitivity gap, without that caveat, is a finding.

### Reuse and pipeline boundaries

27. **New AUC, confidence-interval, ranking or plotting code is checked against the shared
    modules** in `notebooks/` (`ortholog_analysis_utils.py`, `scope_kmerseek_utils.py`,
    `mhc_region_utils.py`, `hp_conservation_utils.py`, and the others). Use
    `ou.load_all_alphabet_ksize_combos()` for the list of alphabet and k-mer size
    combinations.
28. **No Nextflow pipeline reads a file a notebook wrote.** Pipelines run before notebooks.

### Words

29. Run the `writing-style` skill on every markdown cell and the `clear-figures` checklist on
    every figure. In this repo, also:
    - FoldSeek missed a protein; it is not "outside the benchmark".
    - No "twilight zone" framing (dropped 2026-09-13).
    - A grid entry is a "square" or "entry", never a "cell".
    - "Sweep" means only the alphabet × k-mer size parameter sweep.
    - Integers are written `14_873` in code (the script catches `14,873`).

## Step 5: check the review before posting it

Invoke the `are-you-sure` skill on the draft review. Then, for every finding:

- **Try to show the finding is wrong.** Re-read the cell and its saved output, and re-run
  what you can. Drop a finding you cannot back.
- **Label it** CONFIRMED (you re-ran or re-computed it and saw the problem) or PLAUSIBLE
  (you read the code and the output but did not reproduce it). A PLAUSIBLE finding is never
  Blocking.
- **Every "Verified clean" line names the check behind it.** "Numbers in cell 7 match"
  without saying how is not a check.

If Olga asked for a second reviewer, give each Blocking finding to a fresh subagent with
only the notebook path and the finding, and ask it to prove the finding wrong. Report where
it disagreed.

## Step 6: write the review

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
Wrong numbers, claims with no printed source or stronger than their number, a statistic
that does not hold, stale outputs. CONFIRMED findings only. Often empty.

## Should fix
Missing n_tied, null, sample size or spread; unstated dedup setting or query set; a
reimplemented helper.

## Consider
Wording and judgement calls the author may decline.

## Verified clean
What you checked and found right, each with the check behind it ("all 14 numbers in cell
7 match the table printed in cell 6, searched by value").

## Preparing for an Adversarial Reviewer 2: Stress-testing the findings
- Headline re-computed: <the number> from <input> by <second method>: <value> vs <value>.
- Outputs current: last run <date>; helpers last changed <date>; inputs dated <date>.
- Finding <n>: tried to show it wrong by <what>; <result>.
- Not tested: <what could not be run, and why>.
```

Each finding gives: the notebook and cell (`notebooks/241_x.ipynb`, cell 7), what is wrong in
one sentence, what goes wrong because of it, the fix, and CONFIRMED or PLAUSIBLE. If you
re-computed a number, give both values.

`.claude/hooks/require_review_check.py` refuses to post an `/analysis-review` comment that
has no "## Preparing for an Adversarial Reviewer 2: Stress-testing the findings" heading,
no bullet under it, or no "Not tested" line. If nothing went untested, write
"Not tested: nothing" and say why.

Do not edit the notebook unless asked.

## Step 7: post it on the PR

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
after it.

Post it as one comment on the PR. Use `gh api`, never `gh pr comment` or `gh pr edit`, and
keep the comment id it returns:

```bash
GH_PAGER=cat gh api -X POST repos/seanome/2024-kmerseek-analysis/issues/<n>/comments -F body=@<review.md> --jq '.id, .html_url'
```

Read that comment back by its id and check the Verdict came through. Do not read the last
item of the comment list: the list returns 30 comments per page, so on a longer PR the last
item is not the new comment.

```bash
GH_PAGER=cat gh api repos/seanome/2024-kmerseek-analysis/issues/comments/<id> --jq .body | head -8
```

Give Olga the link. If the review is of local changes with no PR, show it in the chat
instead, with the same sections. Remove the review worktree when done:
`git worktree remove /tmp/review-<n>`.

When a review finds a mistake that no rule here covers, add the rule to this file with the
date and the notebook it came from.
