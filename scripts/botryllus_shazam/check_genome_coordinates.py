"""Check GTF exon coordinates against every transcript before deriving an annotation."""

import json
from pathlib import Path
import re
import sys
from Bio import SeqIO
from Bio.Seq import Seq

genome_file, transcript_file, annotation_file, out = map(Path, sys.argv[1:])
out.mkdir(exist_ok=False)
genome = {r.id: str(r.seq).upper() for r in SeqIO.parse(genome_file, "fasta")}
transcripts = {r.id: str(r.seq).upper() for r in SeqIO.parse(transcript_file, "fasta")}
exons = {}
coding = {}
for line in annotation_file.open():
    if line.startswith("#"):
        continue
    f = line.rstrip().split("\t")
    if f[2] not in ("exon", "CDS", "stop_codon"):
        continue
    attrs = dict(re.findall(r'(\S+) "([^"]*)"', f[8]))
    (exons if f[2] == "exon" else coding).setdefault(attrs["gene_id"], []).append(
        (f[0], int(f[3]), int(f[4]), f[6])
    )
checks = {}
for mode, intervals in [("exons", exons), ("CDS-plus-stop", coding)]:
    for shift in [0, -1, 1]:
        bad = []
        for gene, expected in transcripts.items():
            spans = sorted(intervals.get(gene, []), key=lambda r: r[1])
            if not spans or any(
                c not in genome or a + shift < 1 or b + shift > len(genome[c])
                for c, a, b, strand in spans
            ):
                bad.append(gene)
                continue
            merged = []
            for c, a, b, strand in spans:
                if merged and c == merged[-1][0] and a <= merged[-1][2] + 1:
                    merged[-1] = (c, merged[-1][1], max(b, merged[-1][2]), strand)
                else:
                    merged.append((c, a, b, strand))
            actual = "".join(
                genome[c][a + shift - 1 : b + shift] for c, a, b, strand in merged
            )
            if spans[0][3] == "-":
                actual = str(Seq(actual).reverse_complement())
            if actual != expected:
                bad.append(gene)
        checks[f"{mode}:{shift}"] = dict(
            exact=len(transcripts) - len(bad), mismatch_ids=bad
        )
report = dict(transcripts=len(transcripts), checks=checks)
(out / "coordinate-check.json").write_text(json.dumps(report, indent=2) + "\n")
valid = {int(k.split(":")[-1]) for k, v in checks.items() if not v["mismatch_ids"]}
if len(valid) != 1:
    raise ValueError(
        "No unique coordinate offset reproduces every transcript; inspect coordinate-check.json"
    )
shift = valid.pop()
with annotation_file.open() as src, (out / "checked.gtf").open("w") as dest:
    dest.write(
        f"# Derived from {annotation_file}; shift both coordinates by {shift}; all {len(transcripts)} transcripts independently reconstructed.\n"
    )
    for line in src:
        if line.startswith("#"):
            dest.write(line)
            continue
        f = line.rstrip().split("\t")
        f[3] = str(int(f[3]) + shift)
        f[4] = str(int(f[4]) + shift)
        if int(f[3]) < 1:
            raise ValueError("Non-positive derived coordinate")
        dest.write("\t".join(f) + "\n")
print(
    json.dumps(
        {
            "transcripts": len(transcripts),
            "validated_shift": shift,
            "exact_matches": len(transcripts),
        }
    )
)
