"""Mask unsupported characters in a separate search FASTA, retaining coordinates."""

import csv
import hashlib
import sys
from pathlib import Path
from Bio import SeqIO

source, output, changes = map(Path, sys.argv[1:])
if output.exists() or changes.exists():
    raise ValueError("Search copy or mask map exists; compare before reuse")
allowed = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")
with output.open("w") as fasta, changes.open("w") as table:
    writer = csv.writer(table, delimiter="\t")
    writer.writerow(
        [
            "record_id",
            "position_1based",
            "original_character",
            "search_character",
            "original_sequence_sha256",
        ]
    )
    for record in SeqIO.parse(source, "fasta"):
        sequence = str(record.seq)
        checksum = hashlib.sha256(sequence.encode()).hexdigest()
        search = []
        for position, character in enumerate(sequence, 1):
            if character.upper() not in allowed:
                writer.writerow([record.id, position, character, "X", checksum])
                search.append("X")
            else:
                search.append(character.upper())
        fasta.write(f'>{record.id}\n{"".join(search)}\n')
