"""Native mergeTags context from adjusted significant seeds, separately by contrast."""

import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

COMP = str.maketrans("ACGT", "TGCA")


def assemble(seeds, seed_ids, binary, out):
    out = Path(out)
    out.mkdir()
    dna, origins, records = {}, {}, []
    contrasts = sorted({t["contrast"] for tests in seeds.values() for t in tests})
    for contrast in contrasts:
        selected = {}
        for seq, tests in seeds.items():
            candidates = [t for t in tests if t["contrast"] == contrast]
            if candidates:
                best = min(candidates, key=lambda t: t["padj"])
                if not math.isfinite(float(best["log2foldchange"])):
                    raise ValueError("Nonfinite selected seed fold change")
                selected[seq] = best
        source = out / (contrast + ".selected.tsv")
        with source.open("w") as f:
            writer = csv.writer(f, delimiter="\t", lineterminator="\n")
            # Native positional schema: q occupies the representative-ranking column.
            # The two mean placeholders are explicitly unused, never estimated counts.
            writer.writerow(
                ["kmer", "member_padj", "unused_meanA", "unused_meanB", "log2FC"]
            )
            for seq, test in sorted(selected.items()):
                writer.writerow([seq, test["padj"], 0, 0, test["log2foldchange"]])
        target = out / (contrast + ".merged.tsv")
        argv = [str(binary), "-k", "31", "-m", "15", "-n", str(source)]
        with target.open("w") as f, (out / (contrast + ".log")).open("w") as log:
            subprocess.run(argv, stdout=f, stderr=log, check=True)
        represented = 0
        covered = set()
        with target.open() as f:
            for row in csv.DictReader(f, delimiter="\t"):
                seq = row["contig"]
                representative = row["kmer"]
                if representative not in selected or set(seq) - set("ACGT"):
                    raise ValueError("Invalid mergeTags representative or sequence")
                reverse = representative.translate(COMP)[::-1]
                if representative not in seq and reverse not in seq:
                    raise ValueError("Contig lost its representative seed")
                for offset in range(len(seq) - 30):
                    kmer = seq[offset : offset + 31]
                    reverse_kmer = kmer.translate(COMP)[::-1]
                    for member in (kmer, reverse_kmer):
                        if member in selected:
                            covered.add(member)
                count = int(row["nb_merged_kmers"])
                represented += count
                qid = (
                    "DMC"
                    + hashlib.sha256(
                        (contrast + ":" + representative + ":" + seq).encode()
                    ).hexdigest()[:24]
                )
                if qid in dna:
                    raise ValueError("Contig ID collision")
                dna[qid], origins[qid] = seq, seed_ids[representative]
                records.append(
                    dict(
                        query=qid,
                        contrast=contrast,
                        representative_seed=seed_ids[representative],
                        assembled_kmers=count,
                        sequence=seq,
                        representative_member_padj=float(row["member_padj"]),
                        representative_log2FC=float(row["log2FC"]),
                    )
                )
        if represented != len(selected) or covered != set(selected):
            raise ValueError("mergeTags did not account for every selected k-mer")
    (out / "contigs.fasta").write_text("".join(f">{k}\n{v}\n" for k, v in dna.items()))
    (out / "contig-metadata.json").write_text(
        json.dumps(
            dict(
                contigs=records,
                binary_sha256=hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
                interpretation="Representative member adjusted P is not a contig-level test. Native up/down assemblies are separate within each contrast. Representative links are not complete assembly membership.",
            ),
            indent=2,
        )
        + "\n"
    )
    return dna, origins
