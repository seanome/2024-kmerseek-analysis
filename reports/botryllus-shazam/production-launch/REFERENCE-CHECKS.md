# Reference checks before the 64-library launch

The genome FASTA comes from the Botryllus Genome Project download page:
https://botryllus.stanford.edu/botryllusgenome/download/ . The original project GTF,
transcript FASTA, downloaded genome, and earlier outputs remain unchanged.

The original GTF does not reconstruct the reference FASTA at its declared coordinates.
Moving both interval endpoints one base earlier reconstructs 71,802 of 72,113 coding
sequences when CDS and stop-codon intervals are combined. The remaining 311 sequences
are unresolved. Full exons are a different comparison: some include untranslated regions
that are absent from the reference FASTA. No corrected GTF was written because no tested
coordinate convention reconstructs every sequence.

This finding blocks treating this annotation as validated for an exact STAR/HTSeq
reproduction of Figure 4. It does not change the frozen Kallisto reference sequences.
The current Nextflow GENE_COUNTS process reads only transcript and gene identifiers from
the GTF. It does not use its interval coordinates. Genome PAF alignments are separate
sequence evidence and are not projected through this GTF into gene assignments.

Direct genome/Pfam process checks use 500 reference transcripts and 500 reference
proteins, plus empty-input cases. They test executable versions, command generation,
output query identifiers, checksums, and the distinction between an empty query set
and an unmeasured analysis. Their results do not establish query function or exact
paper-reference equivalence.

The supplied supplementary workbook's fingerprint matches the earlier extraction.
The six-page appendix contains supplementary figure, table, movie, and dataset legends;
its extracted text did not provide genome/HTSeq methods or a Botryllus score implementation.
