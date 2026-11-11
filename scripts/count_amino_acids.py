"""Count every residue in one or more protein FASTA files and write the amino acid frequencies.

The outputs are the source of ortholog_analysis_utils.AA_FREQUENCIES (GENCODE human + mouse) and
SWISSPROT_AA_FREQUENCIES (Swiss-Prot). Only the 20 standard amino acids count towards the
frequencies; the other letters (X, U, B, Z, O, *) are reported and left out. A second file holds
the totals: sequences, all residues, standard residues and each left-out letter.
"""

import argparse
import gzip
from collections import Counter
from pathlib import Path

STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def open_fasta(path: Path):
    """Open *path* as text, gzipped or not."""
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open()


def count_residues(fasta_paths: list[Path]) -> tuple[Counter, int]:
    """Return the count of each letter across *fasta_paths* and their number of sequences."""
    counts: Counter = Counter()
    n_sequences = 0
    for path in fasta_paths:
        with open_fasta(path) as handle:
            for line in handle:
                if line.startswith(">"):
                    n_sequences += 1
                else:
                    counts.update(line.strip())
    return counts, n_sequences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_tsv", type=Path)
    parser.add_argument("totals_tsv", type=Path)
    parser.add_argument("fasta", type=Path, nargs="+")
    args = parser.parse_args()

    counts, n_sequences = count_residues(args.fasta)
    n_standard = sum(counts[aa] for aa in STANDARD_AMINO_ACIDS)
    other = {aa: n for aa, n in sorted(counts.items()) if aa not in STANDARD_AMINO_ACIDS}
    print(f"{n_sequences} sequences, {sum(counts.values())} residues")
    print(f"{n_standard} standard; left out: {other}")

    with args.output_tsv.open("w") as out:
        out.write("amino_acid\tcount\tfrequency\n")
        for aa in sorted(STANDARD_AMINO_ACIDS, key=lambda a: -counts[a]):
            out.write(f"{aa}\t{counts[aa]}\t{counts[aa] / n_standard:.6f}\n")

    with args.totals_tsv.open("w") as out:
        out.write("quantity\tcount\n")
        out.write(f"sequences\t{n_sequences}\n")
        out.write(f"all_residues\t{sum(counts.values())}\n")
        out.write(f"standard_residues\t{n_standard}\n")
        out.write(f"left_out_residues\t{sum(other.values())}\n")
        for aa, n in other.items():
            out.write(f"left_out_{aa}\t{n}\n")


if __name__ == "__main__":
    main()
