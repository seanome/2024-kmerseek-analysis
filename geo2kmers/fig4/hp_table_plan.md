# HP k-mer table plan

Written on 2026-11-08, before any HP k-mer was counted.

**Goal.** A table of 30-letter hydrophobic-polar (HP) k-mers from the proteins behind each probe set
(`probe_protein_plan.md`), with each k-mer's value in each sample.

**Alphabets.** Two, as kmerseek defines them (`src/rust/alphabets.rs`):

- `hp_lehninger2`, kmerseek's default `hp`: hydrophobic `AFGILMPVWY`, polar `CDEHKNQRST`.
- `hp_thomas_dill2`: hydrophobic `ACFILMVWY`, polar `DEGHKNPQRST`.

**k-mers.** Every 30-residue window of every protein of 30 amino acids or more. A window holding a letter outside the
20 amino acids is skipped. A probe set's k-mers are the union over its proteins, so isoforms sharing a stretch count it
once. ESTs and genomic clones have no protein and give no HP k-mers.

**Values.** The series matrix values GEO holds for each probe set (the authors' RMA, log2), as deposited. HTA 2.0 series
matrices name probe sets by `probeset_id`, which the platform table maps to the transcript cluster. A k-mer's value in a
sample is the mean over the probe sets whose proteins hold it, as `kamrat_table.py` does for a k-mer on two probes.

**Series.** GSE28998 and GSE27536 (U133 Plus 2) and GSE111555 (HTA 2.0). GSE83578 is on U219, which has no protein
table yet.

**Files.** Most k-mers sit in one probe set, so a full sample matrix would repeat each probe set's row hundreds of
times. The table is written as three parts, which together give every k-mer's value:

- `<array>.<alphabet>.kmers.npz`: each k-mer, packed as 30 bits (hydrophobic = 1, first residue highest), and the
  probe sets holding it;
- `<series>.probe_set_values.tsv.gz`: each probe set's value per sample;
- `<series>.<alphabet>.shared_kmers.tsv.gz`: the full sample matrix for k-mers held by two or more probe sets, whose
  value is a mean.

`<array>.<alphabet>.summary.tsv` counts proteins, probe sets with a k-mer, distinct k-mers and how many probe sets
share each one.
