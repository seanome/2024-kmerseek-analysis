# Query strata: rules written before any result was read

Written 2026-10-16, before any of the jobs in `scripts/query_strata/` had run. What had
been seen at that point: the column list and stratum axes of the midi-plus
`all_domain_metrics.parquet` (sha256 `8ba26616...`), its 15 target species, and the
resource rows of its Nextflow traces. No per-query value of any stratum below had been
computed or looked at.

Every stratum is computed from the query proteins alone, with no truth label, so an
annotator can compute the same bins for their own proteome.

## Bins

| Axis (column `stratum_axis`) | Unit | Value | Bins (`stratum_bin`) |
|---|---|---|---|
| `nearest_identity` | protein, per target proteome | MMseqs2 `easy-search -s 7.0 -e 1e-3` of the query against that target proteome; best hit by lowest E-value (ties: higher bit score, then target name); identity = column 3 (`fident`) x 100 | `no hit` (nothing at E < 1e-3) · `<30%` · `30-50%` · `50-90%` · `>=90%`. Edges: [0, 30), [30, 50), [50, 90), [90, 100] |
| `membrane` | protein | DeepTMHMM predicted type | `alpha-helical TM` (TM, SP+TM) · `beta-barrel` (BETA) · `neither` (SP, GLOB) |
| `low_complexity` | protein | residues masked by SEG (NCBI `segmasker`, SEG defaults: window 12, locut 2.2, hicut 2.5) / protein length | `<5%` · `5-20%` · `>20%`. Edges: [0, 0.05), [0.05, 0.20], (0.20, 1] |
| `protein_length` | protein | residues in the FASTA sequence | `<100` · `100-300` · `300-1000` · `>1000`. Edges: [0, 100), [100, 300), [300, 1000], (1000, inf) |
| `region_length` | truth region | `domain_end - domain_start + 1` (truth coordinates are 1-based and inclusive) | `<30` · `30-100` · `>100`. Edges: [1, 30), [30, 100], (100, inf) |
| `family_superkingdoms` | truth region (Pfam-A and Pfam-N only) | superkingdoms (archaea, bacteria, eukaryota) in which the region's Pfam family has at least one sequence in Pfam 38's reference proteomes | `1` · `2` · `3` |
| `family_phyla` | truth region (Pfam-A and Pfam-N only) | distinct NCBI phyla in which the family has a sequence | `1` · `2-5` · `6-20` · `>20` |
| `busco` | protein | BUSCO 6.1.0, protein mode, `primates_odb12`, on the whole QfO 2020_04 human proteome (20,600 proteins); the query's rows in `full_table.tsv` | `complete` (any row Complete or Duplicated) · `fragmented` (no complete row, at least one Fragmented row) · `not a BUSCO gene` (no row) |

`plddt` (mean AlphaFold pLDDT, bins 0-50 · 50-70 · 70-90 · 90-100) already exists and is kept.

Notes on the rules:

- Identity to Swiss-Prot is not used: every one of the 998 human queries is in Swiss-Prot,
  so its nearest Swiss-Prot protein is itself. The target proteome being searched is the
  reference instead. For an annotator this is "how close is the proteome I transfer from".
- `nearest_identity` differs from the existing `identity` axis. `identity` is per truth
  region (identity of the domain to its closest same-family domain in the target); this
  one is per protein (the whole query's best hit in the target, any family).
- A protein listed by BUSCO as both Complete for one BUSCO gene and Fragmented for another
  counts as complete: complete wins, fragmented is only "fragmented and never complete".
- Family breadth uses the Pfam family of the truth region, not a family a tool called.
  Swiss-Prot feature regions have no Pfam family and get no row on these two axes.
- Swiss-Prot point features (one residue) are widened by one residue in the truth table,
  so their region length is 2 and they sit in `<30`.
- A sequence whose UniProt species code is a "virtual" code above phylum rank (such as
  9BACT = Bacteria) counts for its superkingdom but adds no phylum.

## How rows are made

The scoring step of the QfO region pipeline (`evaluate_domain_calls.py`) cuts every
(tool, setting, target, truth set) by each axis, as it already does for `plddt`. Each new
row is one (experiment, tool, setting, query set, target, truth set, metric) on one bin.
Protein-level axes put every query in exactly one bin; region-level axes put every truth
region in exactly one bin. So for every axis, `n_instances` summed over its bins equals the
whole-set `n_instances` (`family_*` axes: of the Pfam truth sets only). That equality is a
check, not an assumption (`scripts/query_strata/check_query_strata.py`).

A bin with fewer than 30 truth instances is shown grey on the site.

## Predictions, written before any result

1. Every sequence-search tool scores lower Fmax in the `<30%` and `no hit` bins of
   `nearest_identity` than in the `>=90%` bin, on every target species where all three bins
   hold at least 30 truth instances.
2. Proteins with more than 20% of residues masked by SEG have lower Fmax than proteins under
   5%, for most tools.
3. Regions shorter than 30 residues have lower Fmax than regions over 100 residues, for every
   tool.
4. Fewer than 5% of the 998 queries are `fragmented` BUSCO genes: they come from the
   reference human proteome, whose gene models are among the best there are.
