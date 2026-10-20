# Notebook 299 rules, fixed 2026-10-18 before any Sherlock run

Question: can hashed reduced-alphabet sketches (sourmash FracMinHash, which keeps only the
k-mers whose hash falls below a cut-off set by `scaled`) find the same catalog genes as
DIAMOND blastp, for a protein query, without also hitting genes for shuffled copies of the
query? If yes, the next step is the kmerseek layer and a talk with the Branchwater and Logan
groups. If no, searching every sequencing run this way needs a different index.

Decision the reader makes: whether to spend any more time on board idea 12 with sourmash
sketches as the index.

## Data

- Targets: OM-RGC v2 genes (BioStudies S-BSST297, released 2019-11-12, file
  `OM-RGC_v2.tsv.gz`, 10_320_406_416 bytes, the gene table whose column 14 is each gene's
  DNA; changed 2026-10-19 from `OM-RGC_v2_genes.fna.gz`, whose two downloads differed and
  failed gzip's check, before any result existed). Gene count is counted on Sherlock; the
  Ocean Gene Atlas v2.0 paper gives 46.8 million. sourmash reads the DNA in six frames
  (`sketch translate`). DIAMOND reads the same genes translated in frame 1 with genetic
  code 11 (seqkit). In the laptop test of the catalog's first 39_000 genes, frame 1 gave
  0 proteins with an internal stop; the first 40_000 genes of the table did the same on
  Sherlock.
- Real queries (109): 100 arctic clam proteins drawn with `random.Random(299)` from the
  69_055 in `Glanz_Clam_AllModels.proteins.fasta`, and 9 BHF regions copied from PR 80's
  `hits241.BHF.json`: for each Figure 1 gene, its lowest-E region over notebook 241's
  settings.
- Shuffled copies: 20 per real query (2_180), dipeptide-shuffled (same length, same counts
  of every adjacent residue pair). Every method searches them: DIAMOND and every sourmash
  setting.

## Step 1 rule: does the whole catalog fit?

Time `sourmash sketch translate` on one CPU on the catalog's first 1 GB, for each of the 18
settings (protein, dayhoff, hp x k 7, 10, 13 x scaled 100, 1000). Multiply each setting's
CPU-seconds per GB by the catalog's size (arithmetic, labelled so). The catalog fits at a
scaled value if all nine alphabet and k settings at that scaled need at most 1_536 CPU-hours
together. Dead if it fits at neither scaled.

The prompt says "two compute days". Olga set it on 2026-10-19, before the step 1 job ran:
two days of a 32-CPU node, 48 h x 32 CPUs = 1_536 CPU-hours. (The first version of this
file, 2026-10-18, read it as 48 CPU-hours.)

### Step 1 run 1 (jobs 48551687, finished 2026-10-20) and run 2

Run 1 wrote sketches straight to `.zip` and reported "dead": ten settings stopped at the
40_000 s limit (all nine at scaled 1000, and hp k = 7 at scaled 100). The slowness is the
`.zip` writer slowing down on identical sketches (all empty sketches are identical), not
sketching: 132 s to `.zip` vs 2.5 s to `.sig.gz` for the same 20_000 genes. Run 1's tables
are kept as `tables/299_step1_run1_*`. Run 2 uses the same rule and budget and times
`sketch translate` to `.sig.gz` plus writing a `.zip` of the non-empty sketches (what step 3
reads). Dropping empty sketches changes no hit.

The first attempt at run 2 (jobs 48632260, 2026-10-20 06:01) was cancelled at 09:19: four
settings with many identical non-empty sketches (hp k = 7 at scaled 100; dayhoff k = 7 and
hp k = 10 and 13 at scaled 1000) sat for over 3 h writing the `.zip`, the same slowdown for
sketches that are identical but not empty. The `.zip` is now written with each sketch under
its own position number (3.2 s vs 12.5 s on 39_000 genes, identical manysearch hits), and
the `.zip` step has its own time limit counted as a lower bound. Run 2 was resubmitted with
all 18 settings (jobs 48647076).

## Step 3 rule: the kill test

- A true hit is a (real query, catalog gene) pair DIAMOND reports at E < 1e-5.
- A sourmash hit is a (query, gene) pair sharing at least one hash
  (`sourmash scripts manysearch`, default threshold containment 0.01; every query sketch
  here has under 100 hashes, so this means one shared hash). A query with no hash at a
  setting hits nothing; its DIAMOND hits stay in the denominator.
- Recall of a setting: true hits it finds, divided by all true hits, pooled over the 109
  real queries.
- Shuffled-hit rate of a setting: genes hit by all shuffled copies, divided by 20 times the
  genes hit by the real queries. It reads "per copy, a shuffled query hits this share of
  what a real query hits". The same number is computed for DIAMOND at E < 1e-5.
- How each number is counted. Recall is exact: the real queries are searched against only
  the genes DIAMOND hit, since whether a query and a gene share a hash does not depend on
  any other gene. Hit counts come from the full catalog when that search would write at
  most 2 billion rows; otherwise from a fixed 1% random sample of genes (seed 299), and the
  shuffled-hit rate is then marked as estimated. Reason: in the laptop test on the
  catalog's first 39_000 genes, hp at k = 10 and scaled 100 gave 52.7 million hit rows
  (an hp word of 10 letters has only 1_024 possible values), so on the whole catalog such a
  setting would write tens of billions.
- Pass: at least one setting with recall >= 80% and shuffled-hit rate < 1%.
- Dead: no setting passes.
- BHF regions also go through `sourmash scripts fastmultigather`; that table is reported
  beside the rule, not used by it.

## Step 4 runs only if step 3 passes

## Not changed after the run

If a rule above turns out to be the wrong rule, the notebook says so and still reports the
result under it.
