# Probe-set protein plan

Written on 2026-11-08, before any protein was fetched.

**Why.** Array probes are 25 letters and mostly outside coding sequence, too short for a hydrophobic-polar (HP) k-mer of
30 amino acids (90 nt). A probe set measures how much of its transcript is in the sample, wherever the probes sit on it,
so the protein of that transcript can carry the probe set's value. This uses the array annotation, unlike the landing,
which uses none; it does not change any landing.

**Source for each probe set, first rule that applies** (`probe_set_sources.py`):

1. `refseq_mrna`: every RefSeq mRNA (NM_) the annotation lists for the probe set. U133 Plus 2: the RefSeq Transcript ID
   column. HTA 2.0: NM_ entries in mrna_assignment.
2. `ensembl_coding` (HTA 2.0 only): Ensembl transcripts whose biotype is protein_coding.
3. `refseq_model`: RefSeq model mRNAs (XM_), when no NM_ is listed.
4. `genbank_mrna`: the GenBank mRNA the probe set was designed on (U133 exemplar sequences; HTA 2.0 GenBank entries).
5. `est`: U133 probe sets designed on an EST consensus, with no RefSeq transcript: the EST sequence itself.
6. `other_only`, `none`: no protein-coding source named (mostly non-coding RNAs on HTA 2.0).

**Proteins.** For rules 1, 3 and 4, the coding regions NCBI annotates on each accession, translated by NCBI
(`fetch_probe_sequences.py`, efetch `fasta_cds_aa`). A probe set with several transcripts keeps every distinct protein,
each with the probe set's value. ESTs are fetched as nucleotide sequence and not translated here; whether to use
them, and how, is decided after counting how many reach 90 nt.

**Changed after the first fetch, before any k-mer was made.** 2,161 BC, 1,328 AK and 690 AF accessions, among others,
sat in the `est` group because U133 calls their probe sets consensus sequences, but they are full-length cDNAs. So each
`est` accession is also asked for its annotated protein, and one with a protein of 30 amino acids or more counts as
`cdna_with_protein`. Accessions over 5000 nucleotides are genomic clones (`genomic_clone`).

**What this cannot show.** The value is the RNA's, not the protein's, and a probe set pools the isoforms that share its
probes, so a protein's k-mers carry a gene-level value.
