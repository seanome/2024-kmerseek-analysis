# Notebook 346 task brief, as given on 2026-10-18

The brief below was given to Claude Code by olgabot on 2026-10-18, before any database scan of notebook 346 ran. Section 2 of the notebook copies its predictions and rule. It was written when this notebook was to be numbered by the next free number (it became 346); numbers in it come from earlier PRs and notes and are not reproduced unless `tables/346_inputs.tsv` says so.

---

Claude Code prompt: linked H/P word pairs (strobemers) at database scale, on pairs where the answer is already known
Repo: `seanome/2024-kmerseek-analysis`. Branch off `main`: `olgabot/linked-pairs-known-pairs`. Analysis only. No change to `seanome/kmerseek`, no Rust, one alphabet (`hp_thomas_dill2`), one notebook, one fixed rule.
Why this notebook exists
PR 143 (notebook 294) tested a strobemer over the H/P alphabet: two short exact H/P words a little apart, hashed as one search key, against one longer exact word. Verdict KILL. At 20–30% identity the linked key gained +6.7 points of recall (95% interval 3.2 to 8.9) over the 8-letter seed, and control B3 (one 8-letter seed plus the best second gap-free segment within 32 residues) gained +8.4 (4.0 to 10.9). Linked minus B3 was −1.7 (−3.0 to 0.5). The gain came from scoring a second extended segment, not from the second exact word.
Every number in PR 143 was per pair, on each pair's own alignment. Its "Not tested" list includes three things this notebook tests:

1. database-scale false positives (0.001 per comparison is about 573 false pairs per query against Swiss-Prot);
2. chance key matches off the alignment's diagonal (PR 143 counted 3,896 for the linked key, 70 for the 8-letter seed and 18 for the 10-letter seed, per pair of unrelated 131-residue proteins, on the diagonal only);
3. kmerseek's own chaining (`--chain-max-shift`), which may already give B3's gain.

The test runs on protein pairs where the right answer is known before the search, so the result reads as a check with a fixed rule, not as a discovery.
The pairs
Pair 1, expected negative: human BCL-2 and C. elegans CED-9. A true homolog pair (SCOP fold f.1) that 20-letter methods find. phmmer finds BCL-2 at E = 7.9 × 10⁻⁵ and jackhmmer at E = 7.7 × 10⁻⁴¹ against the human proteome (notebook 241). Smith–Waterman with BLOSUM62 ranks BCL-2 second of 14,280 SCOPe40 domains, behind Bcl-xL (the SCOPe rescue test of 2026-10-06). In H/P the whole 154-position structure alignment carries 22.9 bits, and E = 1 on the human proteome needs 25.7 bits (notebook 245). Gapped H/P Smith–Waterman left 2 to 8 unrelated SCOPe40 domains above BCL-2, depending on gap cost. The best kmerseek rank for BCL-2 from CED-9 was 213 of 18,064 (notebook 241, mean IDF, polarity4, k = 9). By E-value it was 1,352 of 2,175 (wwmj5, k = 9). So this pair says where any H/P key stops. A linked key that ranks it in the top 10 would be a surprise worth a second look.
Pair 2, negative control: Borrelia burgdorferi P66 and human CD47. The paper reporting the SIRPα mimicry says the two sequences share no region of significant similarity. P66 is a β-barrel porin; CD47 is an Ig domain with five transmembrane helices. The SIRPα-binding loop of P66, residues 181–187 in mature numbering (202–208 in the precursor), is QENDKDT: seven polar residues in H/P. In the SCOPe rescue test, 44% of length-matched unrelated domains scored at least as high as CD47 against P66 with 20 letters, and 96% in H/P. CD47's best kmerseek rank from P66 was 166 of 18,775 (notebook 241). Any key that ranks CD47 high for P66 is measuring chance. This pair cannot pass anything. It calibrates the chance-match rate of each key at database scale.
Pairs 3 to 6, expected positives: four pairs from notebook 294. Choose them from 294's report half at 20–30% identity, before any database search. Three are the pairs with the largest (linked key − B2) score gain where B2 missed and the linked key hit. The fourth is the pair with the highest linked score among the 139 that the linked key finds and neither phmmer nor MMseqs2 finds. Name their Pfam families. These show on real residues what a strobemer gain is. The database search then asks whether that gain survives a proteome of decoys.
Every number above comes from PR text and chat notes. It may be rounded or stale. Re-read each from its notebook before quoting it, and read it programmatically into `tables/NNN_inputs.tsv` (the notebook 245 pattern: value, kind, source notebook, cell). Never type a number from this prompt into a notebook cell. If the SCOPe rescue scripts are not in the repo, say so and label those numbers "from notes, not reproduced here".
Read first

* `CLAUDE.md`; the `writing-style`, `clear-figures` and `nextflow-notebook-workflow` skills. Load them with the Skill tool; do not work from memory of them.
* PR 143's files: `notebooks/294_linked_hp_word_pairs_kill_test_pfam_seed.ipynb`, `notebooks/linked_pairs_utils.py`, `scripts/linked_pairs_on_alignments.py`, `scripts/linked_pairs_key_spectrum.py`, `tests/test_linked_pairs_utils.py`, `scripts/make_nb294.py`, `notebooks/294_prior_art.md`, and the PR body and review comment at https://github.com/seanome/2024-kmerseek-analysis/pull/143. Reuse the scorers in `linked_pairs_utils.py`. Do not rewrite them. If the database-scale search needs a new function (the same scoring on every diagonal, with no alignment given), add it to that module with a brute-force test beside the existing ones.
* Notebook 245 (branch `olgabot/bits-ceiling-245` if not merged) and its tables: `tables/245_bcl2_ced9.fasta`, `tables/245_bcl2_ced9_usalign_pairs.tsv` (the BCL-2/CED-9 structure alignment, 1-based). Notebook 241 (PR 46): the ranks, the human proteome file it searched, its `arms.csv` (the code calls a kmerseek setting an arm; say so once in the notebook, then write "kmerseek setting"). `notebooks/test-data/cd47.fasta`. Whatever the repo holds on P66/CD47 (grep `P66`, `Oms66`, `CD47`); if there is no notebook, say so.
* Sequences: check BCL-2 = UniProt P10415, CED-9 = P41958, CD47 = Q08722 against the FASTA headers you use. Find the P66 sequence the repo already uses; if none, fetch it from UniProt by accession, and record accession, length and fetch date in the inputs table.
* The shuffle utilities already in the repo: the dark-set chance control (notebook 279) and the dipeptide-preserving shuffle discussed in PR 30's review. Reuse, do not reimplement.
* Notebook 294's inputs: `~/data/pfam/230_pfam_seed_pair_alignments.parquet`, `~/data/pfam/232_seed_extend_scores.parquet`, `~/data/pfam/294_linked_pairs_scores.parquet`, `~/data/pfam/233_aligner_check/233_aligner_pair_evalues.parquet`.

Notebook number
Take the next free number across every remote branch and every local branch (`git ls-tree` on each; PR 143's review found a clash this way, and 294 and 295 are taken). Use the same number NNN for the notebook, `scripts/make_nbNNN.py`, `figures/NNN_*`, `tables/NNN_*` and `~/data/.../NNN_*`. The notebook is built by the script, as 245 and 294 are.
Section 1: the pairs, with their residues
First cell: define each of these in a biologist's words, once, and use the same word everywhere after. H/P alphabet (which residues are H), seed, extension, strobemer (the field's word; put "linked key" beside it once, then use one of the two), anchor, gap, B2, B3, randstrobe, decoy, rank, false-positive rate.
For every pair, before any search, print a fixed-width block. It holds both sequences, a match line, the coordinates, the identity count ("x of 37 identical"), the H/P class strings under each sequence, a class match line, and the longest class-identical run marked with its positions. On that block mark every pair of exact shared 6-letter H/P words at the same aligned positions whose starts are at most 32 apart (the linked key's anchors). Mark B3's two segments too. Say how many linked keys the alignment holds and the longest exact shared H/P word.

* BCL-2/CED-9: the whole USalign alignment, then the BH1 block (BCL-2 130–166 against CED-9 154–190) on its own.
* P66/CD47: no alignment exists. Print the SIRPα loop (both numberings) with its class string. Then count the shared exact 6-letter and 8-letter H/P words and the shared linked keys between the two full sequences, on every diagonal. Count the same between P66 and 1,000 human proteins of CD47's length ± 10%, chosen with a fixed seed. Print CD47's percentile among them.
* Pairs 3 to 6: the same block from notebook 230's alignments, with Pfam family names and identity.

Figure 1, one panel per pair. The two sequences are rows of squares, one colour per H/P class, legend before the marks. Identical residues get one outline. Exact shared words of 6 or more are bars. The best linked key is two bars joined by a bracket labelled with the gap. B3's second segment gets its own mark. Run the `clear-figures` checklist before `savefig`.
Section 2: predictions and the rule, written before section 3 runs
Read notebook 245's constants from its outputs: κ, the hydrophobic share, and the h–h, p–p and h–p bits. From them compute three numbers for BCL-2/CED-9: the bits carried by the best linked key plus its extension on the structure alignment; the 22.9-bit whole-alignment total; the 25.7-bit E = 1 line (log₂(K·m·n), every factor named). Print all three in one sentence.
Then a markdown cell, dated, with these predictions and this rule. Do not edit it after section 3 runs. If the data argue for a different rule, add a second cell labelled "rule considered after the run" and still report under the first.
Predictions:

* P1. CED-9 → human: BCL-2's rank under the linked key is worse than 10, under every H/P scorer.
* P2. BCL-2's rank under the linked key is no better than under B3, in the same database with the same decoys.
* P3. The share of decoy proteins scoring at or above BCL-2 under the linked key is above 0.001.
* P4. P66 → human: CD47 is outside the top 1%, or, if inside, the share of decoys at or above it is above 0.01.
* P5. Of pairs 3 to 6, at most two keep a decoy rate at or below 0.001 at database scale under the linked key.

Rule, for "is a linked key worth anything at database scale that B3 is not":

* PASS: in at least one of the two directions (CED-9 → human, BCL-2 → C. elegans) the linked key ranks the true partner 10 or better with a decoy rate of 0.001 or below, and B3 does not.
* KILL: in both directions the linked key's rank is no better than B3's, or its decoy rate at the true partner's score is 0.01 or above.
* Unclear: anything else. Say which number would settle it.

The P66/CD47 result cannot PASS. It only flags a key whose chance-match rate is too high.
Section 3: the search
Databases, each with its n and source recorded in the inputs table:

* CED-9 (full length, and the SCOPe domain d1ty4a_ as a second query) → the human proteome notebook 241 searched, same file, so ranks compare with 241's.
* BCL-2 → the C. elegans reference proteome from UniProt; record which and its release.
* P66 → the same human proteome.
* CD47 → the Borrelia burgdorferi B31 reference proteome (about 1,300 proteins).
* Pairs 3 to 6: one member as query against Swiss-Prot (`~/data/uniprot/uniprot_sprot.fasta.gz`, as `linked_pairs_key_spectrum.py` uses), partner's rank.

Decoys: one shuffled copy of every database protein, two shuffles with fixed seeds: (a) uniform residue shuffle (notebook 232's null) and (b) the dipeptide-preserving shuffle, which keeps H/P runs. Report both; (b) is the headline. "Decoy rate" below means the share of decoy proteins scoring at or above the true partner.
Scorers, all `hp_thomas_dill2`, all from `linked_pairs_utils.py`, now on every diagonal with no alignment given:

* B2: one exact 8-letter H/P seed plus ungapped extension (mismatch −2, as 232 and 294).
* Linked key: every pair of 6-letter anchors with starts at most 32 apart, gap length not stored (PR 143's chosen setting).
* Randstrobe: one key per position, same anchors and gap.
* B3: the 8-letter seed plus the best second gap-free segment within 32 residues.

The score of a protein pair is the best over all key matches, scored as 294 scores them. Record per pair the number of key matches too; that is the off-diagonal chance-match count PR 143 could not measure.
kmerseek itself: search CED-9 against the human index with `hp_thomas_dill2` at the smallest k the tool accepts and at k = 10, with extension on and chaining on. Those are the flags named in PR 143 and in kmerseek PR 54; read `kmerseek search --help` for the current names and record the kmerseek version. Report BCL-2's rank by `region_evalue` (that column name only for kmerseek's own output) and whether the output saturates (every protein hit). This answers PR 143's "kmerseek's own chaining was not tested" on this one pair.
Gates, each an `assert` with the numbers printed:

* Gate 1: restricting the database-scale scorer to the BCL-2/CED-9 USalign diagonal gives the same value as 294's per-pair scorer, for all four scorers.
* Gate 2: pairs 3 to 6, restricted to their notebook 230 alignment, get the scores stored in `294_linked_pairs_scores.parquet`.

One table of outputs per query × database × scorer. Columns: rank of the true partner (ties count against it), decoy rate under each shuffle, number of real proteins at or above it, key matches with the true partner, and the median key matches over the database. A second table for kmerseek's own ranks.
Figure 2, one panel per query direction, one row per scorer. The score of every real database protein in one colour, every decoy in one colour per shuffle, and the true partner's score as one marked line with its rank printed beside it. Legend before the marks. One meaning per colour across the whole notebook.
Figure 3: chance key matches per protein pair against target length, one mark per key type, measured on the human proteome. Draw PR 143's three per-131-residue numbers as labelled points, "from notebook 294, on alignments".
Section 4: verdict
Apply the section 2 rule as written, and report each prediction as held or failed with its number. State what the data say; no evaluative words. List what was not tested (other alphabets, three anchors, SCOPe pairs, FoldSeek).
Checks before the PR

* `pytest tests/test_linked_pairs_utils.py` plus every new brute-force test; `pre-commit run --files` on the PR's files; `prose_check.py` from the `writing-style` skill on the notebook's markdown and on the PR body.
* Every number quoted from an earlier notebook is in `tables/NNN_inputs.tsv` with its source cell.
* A fresh-eyes reader agent (Agent tool, `isolation: "worktree"`, has not seen the build) reads the notebook as a biologist and lists what is missing or unclear. Fix everything it lists.
* An adversarial agent (Agent tool, `isolation: "worktree"`, has not seen the build) gets a numbered list of claims, each with the code path or computation that would refute it. It returns HOLDS / WRONG / UNVERIFIABLE per claim with the `file:line` or recomputed number. It recomputes the headline ranks and decoy rates with its own numpy from the saved score tables. Tell me which claims held, which failed, and what changed.
* Citations: fetch every DOI from PubMed with the search tool, never from memory. The strobemer paper is already in `notebooks/294_prior_art.md`. Find the P66/CD47 mimicry paper and the BCL-2/CED-9 similarity paper (Hengartner and Horvitz, 1994) through PubMed. Chicago style, full journal name, italics.

Budget and stop rules
Laptop. If any step is projected over two hours of wall time, stop and tell me the projection before running it. The step to watch is the all-diagonal scan of 18,064 human proteins and 18,064 decoys by four scorers; use numba or multiprocessing as `linked_pairs_on_alignments.py` does. Write the per-protein score tables to `~/data/.../NNN_*.parquet` so the notebook re-runs from them.
Deliverables
Commits in the repo's style. A PR title and body in the repo's format, with this notebook's verdict in the title. The body has the note that it was written by Claude Code at olgabot's request, "In plain English", the rule as written before the run, the results tables, gates, not tested, and files and inputs. Show me the PR body and wait for my go before pushing or opening the PR. Figures through `finish_figure` (notebook figures); no `pubfig` export unless I ask.
Do not

* Change kmerseek, write Rust, add an alphabet, add a third anchor, or search any pair beyond the six listed. More BCL-2/CED-9 and P66/CD47 searching is parked by the triage page; this is one notebook with one rule.
* Type a number from this prompt into the notebook. Read it from its source.
* Write "arm" in prose (write "kmerseek setting"), or any word coined in the session. One name per thing, defined in the first cell.
* Call a null result a failure. Report what held and what did not.
