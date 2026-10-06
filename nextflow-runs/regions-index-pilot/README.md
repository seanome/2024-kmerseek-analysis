# Regions-index pilot

Does an index of Swiss-Prot's annotated regions alone let kmerseek call short and
disordered regions that it cannot call against whole proteins? And does it help kmerseek
more than MMseqs2?

A hit needs a score of log2(K·m·n / E) bits to reach E-value E, where m is the query length
and n the number of residues in the index. A smaller index lowers that bar by
log2(n_whole / n_regions) bits. The two indexes are therefore compared at the same decoy
error rate, never at the same E-value.

## What is built

- **Queries:** the 975 reviewed human Swiss-Prot entries on chromosome 6 (proteome
  UP000005640, component "Chromosome 6", release 2026_03), in
  `assets/chr6_queries.2026_03.tsv`.
- **Truth:** each query's own features of type DOMAIN, REGION, MOTIF, REPEAT, ZN_FING,
  COMPBIAS, COILED, TRANSMEM and INTRAMEM, with their evidence codes
  (`bin/build_query_truth.py`). A REGION "Disordered" counts as disordered only when its
  evidence is experimental (ECO:0000269), or when at least half its residues lie inside
  DisProt consensus disorder (`assets/disprot_human_disorder.tsv`).
- **The whole-protein index:** reviewed Swiss-Prot with every Mammalia entry removed
  (NCBI taxon 40674, read from the OC lineage lines). Built by
  `bin/build_clade_excluded_reference.py`, copied unchanged from
  `nextflow-runs/invertebrate-dark-set` on olgabot/dark-set-kmerseek-0.4 (7913404).
- **The regions index:** every feature of the nine types above in those same entries, cut
  out with (k_max − 1) = 18 residues on each side, clipped at the protein ends. Each entry
  is named `accession|feature type|description|start-end`, with the feature's own
  coordinates. Cut-outs of features of 30 aa or more are clustered with
  `mmseqs easy-cluster --min-seq-id 0.5 -c 0.8 --cov-mode 0`, and the representatives are
  kept. Shorter ones are deduplicated by exact sequence. `cluster_members.parquet` lists
  every cut-out, its representative, and whether the cluster's feature types agree.
- **Decoys:** each index gets a copy with every entry's residues shuffled within
  non-overlapping 10-residue windows (seed 20261006), named `DECOY_<name>`, searched
  together with the targets.

k_max = 19 is the largest k-size among the proposed kmerseek settings (hp_pbotc_1st_ed2
k = 19; see `BUILD_REPORT.md`).

The kmerseek and MMseqs2 searches are not in `main.nf` yet. They are added after the
build is reviewed.

## Run

```bash
make build-local SWISSPROT_DAT=/path/to/uniprot_sprot.2026_03.dat.gz
```

Results of the build, its checksums and the numbers it printed are in `build_summary/` and
`BUILD_REPORT.md`. The predictions, written before any search, are in `PREDICTIONS.md`.

## Words

A "kmerseek setting" is one combination of alphabet, k-size, scaled, extension penalty and
low-complexity mask. The invertebrate-dark-set code calls a setting an "arm"
(`arm_label`).
