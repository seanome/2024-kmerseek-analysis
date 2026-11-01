"""Make synthetic construct records, to check the pipeline's mechanics only.

Nothing here is evidence about real constructs. The sequences are drawn from a background
composition, and the inserts are put in by this script, so recovery on these files says
the code runs, not that the method works.

    python src/synthetic.py --output synthetic --constructs 60 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

# Published tag, linker and protease-site sequences.
TAGS = [
    ("His6", "HHHHHH", "affinity tag"),
    ("His10", "HHHHHHHHHH", "affinity tag"),
    ("FLAG", "DYKDDDDK", "affinity tag"),
    ("Strep-tag II", "WSHPQFEK", "affinity tag"),
    ("Myc", "EQKLISEEDL", "epitope tag"),
    ("HA", "YPYDVPDYA", "epitope tag"),
    ("V5", "GKPIPNPLLGLDST", "epitope tag"),
    ("S-tag", "KETAAAKFERQHMDS", "affinity tag"),
    ("TEV site", "ENLYFQG", "protease site"),
    ("thrombin site", "LVPRGS", "protease site"),
    ("enterokinase site", "DDDDK", "protease site"),
    ("GS linker", "GGGGSGGGGSGGGGS", "linker"),
    ("GSA linker", "GSAGSAAGSGEF", "linker"),
    ("rigid linker", "AEAAAKEAAAKA", "linker"),
    ("SUMO tail", "MSDSEVNQEAKPEVKPEVKPETHINLKVSDGSSEI", "solubility tag"),
    ("AviTag", "GLNDIFEAQKIEWHE", "modification tag"),
    ("SpyTag", "AHIVMVDAYKPTK", "conjugation tag"),
    ("Sortase site", "LPETGG", "conjugation site"),
]

#: Rough amino acid frequencies, so the generated proteins are not uniform noise.
BACKGROUND = (
    "AAAAAAAAALLLLLLLLLLLLGGGGGGGGSSSSSSSSVVVVVVVEEEEEEEKKKKKKKTTTTTTT"
    "IIIIIIDDDDDDRRRRRRPPPPPNNNNNFFFFFQQQQQYYYYHHHCCCMMMWW"
)
CODONS = {
    "A": "GCT", "C": "TGT", "D": "GAT", "E": "GAA", "F": "TTT", "G": "GGT",
    "H": "CAT", "I": "ATT", "K": "AAA", "L": "CTT", "M": "ATG", "N": "AAT",
    "P": "CCT", "Q": "CAA", "R": "CGT", "S": "AGT", "T": "ACT", "V": "GTT",
    "W": "TGG", "Y": "TAT",
}


def protein(rng: random.Random, length: int) -> str:
    return "".join(rng.choice(BACKGROUND) for _ in range(length))


def back_translate(seq: str) -> str:
    return "".join(CODONS[r] for r in seq)


def mutate(seq: str, rng: random.Random, changes: int) -> str:
    letters = list(seq)
    for _ in range(changes):
        position = rng.randrange(len(letters))
        letters[position] = rng.choice(BACKGROUND)
    return "".join(letters)


def build(output: Path, n_constructs: int, seed: int, dev: bool) -> None:
    rng = random.Random(seed)
    output.mkdir(parents=True, exist_ok=True)

    constructs, expected, events = [], [], []
    dictionary_names = [t[0] for t in TAGS[:12]]

    for index in range(n_constructs):
        construct_id = f"{'DEV' if dev else 'C'}{index + 1:04d}"
        group_id = f"G{(index // 2) + 1:04d}" if not dev else f"DG{index + 1:04d}"
        version = "v1"
        core = protein(rng, rng.randrange(120, 320))

        permitted_name, permitted_seq, _ = TAGS[rng.randrange(len(TAGS))]
        intended = permitted_seq + "GS" + core
        received = intended
        status = "clean"

        roll = rng.random()
        if dev:
            roll = 1.0  # the development collection is all clean
        if roll < 0.28:
            status = "error"
        elif roll < 0.36:
            status = "unresolved"

        expected.append(
            {
                "construct_id": construct_id,
                "version": version,
                "insert_name": permitted_name,
                "insert_seq": permitted_seq,
                "expected_start": 1,
                "expected_end": len(permitted_seq),
            }
        )

        if status in ("error", "unresolved"):
            name, seq, _ = TAGS[rng.randrange(len(TAGS))]
            while name == permitted_name:
                name, seq, _ = TAGS[rng.randrange(len(TAGS))]
            # Most unexpected inserts carry a substitution or two, as a real one would.
            inserted = seq if rng.random() < 0.4 else mutate(seq, rng, 1)
            at = rng.randrange(len(permitted_seq) + 2, len(intended))
            received = intended[:at] + inserted + intended[at:]
            events.append(
                {
                    "construct_id": construct_id,
                    "version": version,
                    "event_id": f"E{len(events) + 1:04d}",
                    "insert_name": name,
                    "insert_seq": inserted,
                    "start": at + 1,
                    "end": at + len(inserted),
                    "decision": (
                        "unexpected_insert" if status == "error" else "unresolved"
                    ),
                    "evidence": "synthetic record" if status == "error" else "",
                    "decided_date": "2024-01-15",
                }
            )

        has_dna = rng.random() < 0.5
        constructs.append(
            {
                "construct_id": construct_id,
                "version": version,
                "group_id": group_id,
                "submitted_date": "2025-06-01",
                "status": status,
                "intended_protein_seq": intended,
                "received_protein_seq": received,
                "received_seq_provenance": "synthetic",
                "original_dna_seq": back_translate(received) if has_dna else "",
                "dna_provenance": "synthetic back-translation" if has_dna else "",
            }
        )

    dictionary = [
        {
            "insert_name": name,
            "insert_seq": seq,
            "category": category,
            "source": "published tag sequence",
            "frozen_date": "2024-01-01",
        }
        for name, seq, category in TAGS
        if name in dictionary_names or not dev
    ]

    def write(name: str, rows: list[dict], columns: tuple[str, ...]) -> None:
        with (output / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns))
            writer.writeheader()
            writer.writerows(rows)

    from records import (
        CONSTRUCT_COLUMNS,
        DICTIONARY_COLUMNS,
        EVENT_COLUMNS,
        EXPECTED_INSERT_COLUMNS,
        REVIEW_COLUMNS,
    )

    write("constructs.csv", constructs, CONSTRUCT_COLUMNS)
    write("expected_inserts.csv", expected, EXPECTED_INSERT_COLUMNS)
    write("discrepancy_events.csv", events, EVENT_COLUMNS)
    write("insert_dictionary.csv", dictionary, DICTIONARY_COLUMNS)
    # No review sessions: synthetic records cannot carry recorded human review time.
    write("review_sessions.csv", [], REVIEW_COLUMNS)

    documented = sum(1 for e in events if e["decision"] == "unexpected_insert")
    print(
        f"{output}: {len(constructs)} constructs, {documented} documented events, "
        f"{len(events) - documented} unresolved, {len(dictionary)} dictionary inserts"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="synthetic")
    parser.add_argument("--constructs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = Path(args.output)
    build(root / "dev", max(8, args.constructs // 4), args.seed + 1, dev=True)
    build(root / "eval", args.constructs, args.seed, dev=False)
    return 0


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
