"""Write the proteins to query: the dark proteins, the six controls, and 200 background proteins.

Background: 200 proteins drawn at random (seed 2026) from those that are neither in
dark.txt nor controls. A sample's hit threshold is the 99th percentile of these 200
proteins' containments in that sample.

    python prepare_queries.py PROTEOME_FASTA dark.txt controls.tsv OUT_FASTA OUT_TSV
"""

import csv
import random
import sys

N_BACKGROUND = 200
SEED = 2026


def read_fasta(path):
    seqs, name = {}, None
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                name = line[1:].split()[0]
                seqs[name] = []
            else:
                seqs[name].append(line.strip())
    return {k: "".join(v) for k, v in seqs.items()}


def main(fasta, dark_txt, controls_tsv, out_fasta, out_tsv):
    seqs = read_fasta(fasta)
    dark = [l.strip() for l in open(dark_txt) if l.strip() and not l.startswith("#")]
    controls = [r["protein"] for r in csv.DictReader(open(controls_tsv), delimiter="\t")]
    missing = [p for p in dark + controls if p not in seqs]
    if missing:
        raise SystemExit(f"{len(missing)} names not in the FASTA, e.g. {missing[:3]}")
    pool = sorted(set(seqs) - set(dark) - set(controls))
    background = random.Random(SEED).sample(pool, N_BACKGROUND)
    roles = [(p, "control") for p in controls] + [(p, "dark") for p in dark if p not in controls] \
        + [(p, "background") for p in background]
    with open(out_fasta, "w") as fa, open(out_tsv, "w") as tsv:
        tsv.write("protein\trole\tlength\n")
        for p, role in roles:
            fa.write(f">{p}\n{seqs[p]}\n")
            tsv.write(f"{p}\t{role}\t{len(seqs[p])}\n")
    print(f"controls {len(controls)}, dark {len(dark)}, background {len(background)} "
          f"(from {len(pool)} proteins neither dark nor a control)")


if __name__ == "__main__":
    main(*sys.argv[1:6])
