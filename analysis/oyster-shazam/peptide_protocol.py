"""Six-frame translation with stops as the only boundaries; preserve internal X."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq

PROTOCOL = "six-frame-standard-stop-only-X-preserved-min30-v1"


@dataclass(frozen=True)
class Segment:
    frame: int
    aa_start: int
    aa_end: int
    dna_start: int
    dna_end: int
    peptide: str


def segments(dna):
    """Yield >=30aa segments; coordinates are zero-based, half-open.

    Amino-acid coordinates refer to the translated frame. DNA coordinates refer
    to the original forward sequence, including for reverse frames -1/-2/-3.
    """
    sequence = Seq(dna)
    for direction, strand in ((1, sequence), (-1, sequence.reverse_complement())):
        for offset in range(3):
            length = (len(strand) - offset) // 3 * 3
            if length <= 0:
                continue
            translated = str(strand[offset : offset + length].translate(table=1))
            start = 0
            for peptide in translated.split("*"):
                end = start + len(peptide)
                if len(peptide) >= 30:
                    low, high = offset + start * 3, offset + end * 3
                    if direction == -1:
                        low, high = len(dna) - high, len(dna) - low
                    yield Segment(
                        direction * (offset + 1), start, end, low, high, peptide
                    )
                start = end + 1


def export(source, destination):
    """Keep every original record, including those with no retained segments."""
    destination.mkdir()  # Refuse to merge outputs, including partial previous attempts.
    seen = set()
    with source.open() as source_handle, (destination / "origins.jsonl").open(
        "w"
    ) as origins, (destination / "segments.jsonl").open("w") as mapping, (
        destination / "peptides.fa"
    ).open(
        "w"
    ) as fasta:
        for index, record in enumerate(SeqIO.parse(source_handle, "fasta")):
            if record.id in seen:
                raise ValueError(f"Duplicate original ID: {record.id}")
            seen.add(record.id)
            dna = str(record.seq)
            count = 0
            for count, segment in enumerate(segments(dna), 1):
                segment_id = f"segment-{index}-{count}"
                fasta.write(f">{segment_id}\n{segment.peptide}\n")
                mapping.write(
                    json.dumps(
                        dict(
                            segment_id=segment_id,
                            origin_id=record.id,
                            **asdict(segment),
                        )
                    )
                    + "\n"
                )
            origins.write(
                json.dumps(
                    dict(
                        origin_id=record.id,
                        description=record.description,
                        dna=dna,
                        segments=count,
                    )
                )
                + "\n"
            )
    if not seen:
        raise ValueError("No original records")
    with source.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    outputs = {}
    for name in ("origins.jsonl", "segments.jsonl", "peptides.fa"):
        with (destination / name).open("rb") as handle:
            outputs[name] = hashlib.file_digest(handle, "sha256").hexdigest()
    (destination / "complete.json").write_text(
        json.dumps(
            dict(
                protocol=PROTOCOL,
                source_sha256=digest,
                origins=len(seen),
                outputs=outputs,
            )
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    export(args.source, args.destination)
