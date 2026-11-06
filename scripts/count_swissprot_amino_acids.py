"""Count every residue in a Swiss-Prot FASTA and write the amino acid frequencies.

The output is the source of ortholog_analysis_utils.SWISSPROT_AA_FREQUENCIES. Only the 20
standard amino acids count towards the frequencies, the same as AA_FREQUENCIES; the other
letters (X, U, B, Z, O) are reported and left out.
"""

import argparse
import gzip
from collections import Counter
from pathlib import Path

STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def count_residues(fasta_gz: Path) -> tuple[Counter, int]:
    """Return the count of each letter in *fasta_gz* and its number of sequences."""
    counts: Counter = Counter()
    n_sequences = 0
    with gzip.open(fasta_gz, "rt") as handle:
        for line in handle:
            if line.startswith(">"):
                n_sequences += 1
            else:
                counts.update(line.strip())
    return counts, n_sequences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fasta_gz", type=Path)
    parser.add_argument("output_tsv", type=Path)
    args = parser.parse_args()

    counts, n_sequences = count_residues(args.fasta_gz)
    n_standard = sum(counts[aa] for aa in STANDARD_AMINO_ACIDS)
    other = {aa: n for aa, n in counts.items() if aa not in STANDARD_AMINO_ACIDS}
    print(f"{n_sequences} sequences, {sum(counts.values())} residues")
    print(f"{n_standard} standard; left out: {dict(sorted(other.items()))}")

    with args.output_tsv.open("w") as out:
        out.write("amino_acid\tcount\tfrequency\n")
        for aa in sorted(STANDARD_AMINO_ACIDS, key=lambda a: -counts[a]):
            out.write(f"{aa}\t{counts[aa]}\t{counts[aa] / n_standard:.6f}\n")


if __name__ == "__main__":
    main()
