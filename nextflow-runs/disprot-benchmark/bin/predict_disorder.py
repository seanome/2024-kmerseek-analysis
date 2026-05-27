#!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
"""
predict_disorder.py

Run metapredict V2/V3 on a query FASTA to get per-residue disorder scores.
Outputs per-protein summary statistics used to stratify benchmark results.

Usage:
    predict_disorder.py <query.fasta> <output.tsv>

Output TSV columns:
    uniprot_acc, protein_length, mean_disorder, max_window_disorder,
    fraction_disordered, disorder_category
    + per-DisProt-region columns added by build_disprot_ground_truth

disorder_category thresholds:
    ordered    : mean_disorder < 0.2
    partial    : 0.2 <= mean_disorder < 0.5
    disordered : mean_disorder >= 0.5
"""

import sys
from pathlib import Path


def read_fasta(path: str) -> dict[str, str]:
    seqs = {}
    current_acc, current_seq = None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if current_acc:
                    seqs[current_acc] = "".join(current_seq)
                header = line.strip()[1:]
                parts = header.split("|")
                current_acc = parts[1] if len(parts) >= 2 else parts[0].split()[0]
                current_seq = []
            else:
                current_seq.append(line.strip())
        if current_acc:
            seqs[current_acc] = "".join(current_seq)
    return seqs


def disorder_category(mean_d: float) -> str:
    if mean_d < 0.2:
        return "ordered"
    if mean_d < 0.5:
        return "partial"
    return "disordered"


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    fasta_path, output_tsv = sys.argv[1], sys.argv[2]

    # ------------------------------------------------------------------
    # Import metapredict (graceful fallback if not installed)
    # ------------------------------------------------------------------
    try:
        import metapredict as meta
        has_metapredict = True
        print("Using metapredict for disorder prediction", file=sys.stderr)
    except ImportError:
        has_metapredict = False
        print("WARNING: metapredict not installed — using uniform 0.5 scores", file=sys.stderr)
        print("Install with: pip install metapredict", file=sys.stderr)

    seqs = read_fasta(fasta_path)
    print(f"Predicting disorder for {len(seqs)} sequences", file=sys.stderr)

    cols = [
        "uniprot_acc", "protein_length", "mean_disorder",
        "max_window_disorder", "fraction_disordered", "disorder_category",
    ]

    with open(output_tsv, "w") as out:
        out.write("\t".join(cols) + "\n")

        for i, (acc, seq) in enumerate(seqs.items()):
            if i % 100 == 0:
                print(f"  {i}/{len(seqs)}", file=sys.stderr)

            if has_metapredict:
                try:
                    scores = meta.predict_disorder_domains(seq).disorder
                except Exception as e:
                    print(f"WARNING: metapredict failed for {acc}: {e}", file=sys.stderr)
                    scores = [0.5] * len(seq)
            else:
                scores = [0.5] * len(seq)

            if not scores:
                scores = [0.5]

            mean_d    = sum(scores) / len(scores)
            frac_d    = sum(1 for s in scores if s > 0.5) / len(scores)

            # Max mean disorder in any 30-residue sliding window
            w = 30
            if len(scores) >= w:
                max_win = max(
                    sum(scores[i:i+w]) / w
                    for i in range(len(scores) - w + 1)
                )
            else:
                max_win = mean_d

            cat = disorder_category(mean_d)

            out.write(
                f"{acc}\t{len(seq)}\t{mean_d:.4f}\t"
                f"{max_win:.4f}\t{frac_d:.4f}\t{cat}\n"
            )

    print(f"Wrote {output_tsv}", file=sys.stderr)


if __name__ == "__main__":
    main()
