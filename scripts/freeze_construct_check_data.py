"""Freeze the reference data for the construct-check kill test (notebook 355).

Writes ``data/kill_tests/construct_check/``:

- ``tags_linkers.tsv``: the standard tag and linker list, one row per sequence, with where
  each sequence came from.
- ``clean_host_proteins_swissprot.fasta``: Swiss-Prot proteins used as hosts in the
  mechanics check. These are not construct submissions and do not count toward the verdict.
- ``uniprot_cache/``: the UniProt FASTA records the two files above were built from, so the
  build can be repeated without the network.
- ``references.md``: the full citation for every sequence whose source is a publication.
- ``SHA256SUMS``: checksum of every frozen file.

The 200 historical facility construct submissions the kill test needs are NOT here and
cannot be fetched: nobody outside the facility has them. ``README.md`` in the same
directory states the format they must arrive in. Nothing in this script invents a
submission.

Run with the project interpreter:

    /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 \
        scripts/freeze_construct_check_data.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import time
import urllib.error
import urllib.request

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
FROZEN_DIR = REPO_ROOT / "data" / "kill_tests" / "construct_check"
CACHE_DIR = FROZEN_DIR / "uniprot_cache"

USER_AGENT = "2024-kmerseek-analysis construct-check freeze (mailto:olga@seanome.org)"

# UniProt accessions this script downloads. Parents are proteins a tag sequence is taken
# from; hosts are the clean proteins the mechanics check inserts tags into.
TAG_PARENTS = {
    "P03437": "Influenza A virus A/Aichi/2/1968 H3N2 hemagglutinin, source of the HA tag",
    "P01106": "Human MYC, source of the Myc (9E10) tag",
    "P08515": "Schistosoma japonicum GST (sj26), the pGEX fusion partner",
    "P0AEX9": "Escherichia coli K-12 MalE, the maltose-binding protein fusion partner",
}
CLEAN_HOSTS = {
    "P10415": "BCL2_HUMAN",
    "P41958": "CED9_CAEEL",
    "Q07820": "MCL1_HUMAN",
    "Q07812": "BAX_HUMAN",
    "P04637": "P53_HUMAN",
    "P00533": "EGFR_HUMAN",
}

# Citations resolved against the Crossref API (api.crossref.org) on 2026-10-20, not written
# from memory. Each key is used in the `source_id` column of tags_linkers.tsv.
REFERENCES = {
    "hochuli1988": {
        "authors": "Hochuli, E., W. Bannwarth, H. Döbeli, R. Gentz, and D. Stüber",
        "title": "Genetic Approach to Facilitate Purification of Recombinant Proteins "
        "with a Novel Metal Chelate Adsorbent",
        "journal": "Bio/Technology",
        "year": 1988,
        "volume": "6",
        "pages": "1321-1325",
        "doi": "10.1038/nbt1188-1321",
    },
    "hopp1988": {
        "authors": "Hopp, Thomas P., Kathryn S. Prickett, Virginia L. Price, "
        "Randell T. Libby, Carl J. March, Dale Pat Cerretti, Douglas L. Urdal, "
        "and Paul J. Conlon",
        "title": "A Short Polypeptide Marker Sequence Useful for Recombinant Protein "
        "Identification and Purification",
        "journal": "Bio/Technology",
        "year": 1988,
        "volume": "6",
        "pages": "1204-1210",
        "doi": "10.1038/nbt1088-1204",
    },
    "field1988": {
        "authors": "Field, J., J. Nikawa, D. Broek, B. MacDonald, L. Rodgers, "
        "I. A. Wilson, R. A. Lerner, and M. Wigler",
        "title": "Purification of a RAS-Responsive Adenylyl Cyclase Complex from "
        "Saccharomyces cerevisiae by Use of an Epitope Addition Method",
        "journal": "Molecular and Cellular Biology",
        "year": 1988,
        "volume": "8",
        "pages": "2159-2165",
        "doi": "10.1128/mcb.8.5.2159",
    },
    "evan1985": {
        "authors": "Evan, Gerard I., Gerald K. Lewis, Gregor Ramsay, and J. Michael Bishop",
        "title": "Isolation of Monoclonal Antibodies Specific for Human c-myc "
        "Proto-Oncogene Product",
        "journal": "Molecular and Cellular Biology",
        "year": 1985,
        "volume": "5",
        "pages": "3610-3616",
        "doi": "10.1128/mcb.5.12.3610-3616.1985",
    },
    "schmidt1993": {
        "authors": "Schmidt, Thomas G. M., and Arne Skerra",
        "title": "The Random Peptide Library-Assisted Engineering of a C-Terminal "
        "Affinity Peptide, Useful for the Detection and Purification of a Functional "
        "Ig Fv Fragment",
        "journal": "Protein Engineering, Design and Selection",
        "year": 1993,
        "volume": "6",
        "pages": "109-122",
        "doi": "10.1093/protein/6.1.109",
    },
    "smith1988": {
        "authors": "Smith, Donald B., and Kevin S. Johnson",
        "title": "Single-Step Purification of Polypeptides Expressed in Escherichia coli "
        "as Fusions with Glutathione S-Transferase",
        "journal": "Gene",
        "year": 1988,
        "volume": "67",
        "pages": "31-40",
        "doi": "10.1016/0378-1119(88)90005-4",
    },
    "diguan1988": {
        "authors": "di Guan, Chu, Panzhi Li, Paul D. Riggs, and Hiroshi Inouye",
        "title": "Vectors That Facilitate the Expression and Purification of Foreign "
        "Peptides in Escherichia coli by Fusion to Maltose-Binding Protein",
        "journal": "Gene",
        "year": 1988,
        "volume": "67",
        "pages": "21-30",
        "doi": "10.1016/0378-1119(88)90004-2",
    },
    "huston1988": {
        "authors": "Huston, James S., David Levinson, Michael Mudgett-Hunter, "
        "Meng-Sheng Tai, Jiri Novotný, Michael N. Margolies, Robert J. Ridge, "
        "Rosanne E. Bruccoleri, Edward Haber, Richard Crea, and Hermann Oppermann",
        "title": "Protein Engineering of Antibody Binding Sites: Recovery of Specific "
        "Activity in an Anti-Digoxin Single-Chain Fv Analogue Produced in "
        "Escherichia coli",
        "journal": "Proceedings of the National Academy of Sciences",
        "year": 1988,
        "volume": "85",
        "pages": "5879-5883",
        "doi": "10.1073/pnas.85.16.5879",
    },
}

# Short tags and linkers given as literal sequences. `parent` is set when the sequence is a
# subsequence of a real protein; the script then checks it really is one and records the
# coordinates it found, so a typo here fails the build instead of entering the frozen file.
LITERAL_ENTRIES = [
    {
        "name": "His6",
        "category": "affinity tag",
        "sequence": "HHHHHH",
        "parent": None,
        "source_id": "hochuli1988",
        "provenance": "synthetic; polyhistidine affinity tag",
    },
    {
        "name": "His8",
        "category": "affinity tag",
        "sequence": "HHHHHHHH",
        "parent": None,
        "source_id": "hochuli1988",
        "provenance": "synthetic; length variant of the polyhistidine tag, same principle "
        "as His6 in the cited paper, which used His6",
    },
    {
        "name": "FLAG",
        "category": "epitope tag",
        "sequence": "DYKDDDDK",
        "parent": None,
        "source_id": "hopp1988",
        "provenance": "synthetic; designed as a marker peptide, not taken from a protein",
    },
    {
        "name": "HA",
        "category": "epitope tag",
        "sequence": "YPYDVPDYA",
        "parent": "P03437",
        "source_id": "field1988",
        "provenance": "subsequence of influenza A hemagglutinin; the cited paper is its "
        "use as an epitope tag",
    },
    {
        "name": "Myc",
        "category": "epitope tag",
        "sequence": "EQKLISEEDL",
        "parent": "P01106",
        "source_id": "evan1985",
        "provenance": "subsequence of human MYC; the cited paper is the 9E10 antibody "
        "this epitope is detected with",
    },
    {
        "name": "Strep-tag II",
        "category": "affinity tag",
        "sequence": "WSHPQFEK",
        "parent": None,
        "source_id": "schmidt1993",
        "provenance": "synthetic; selected from a random peptide library",
    },
    {
        "name": "GS linker G4S",
        "category": "linker",
        "sequence": "GGGGS",
        "parent": None,
        "source_id": "huston1988",
        "provenance": "synthetic; one repeat of the glycine-serine linker used in the "
        "cited single-chain Fv",
    },
    {
        "name": "GS linker (G4S)x3",
        "category": "linker",
        "sequence": "GGGGSGGGGSGGGGS",
        "parent": None,
        "source_id": "huston1988",
        "provenance": "synthetic; the 15-residue glycine-serine linker of the cited "
        "single-chain Fv",
    },
    {
        "name": "GS linker GSG",
        "category": "linker",
        "sequence": "GSG",
        "parent": None,
        "source_id": None,
        "provenance": "synthetic; short glycine-serine linker in common use, with no "
        "single originating publication to cite",
    },
    {
        "name": "GS linker GGSGGS",
        "category": "linker",
        "sequence": "GGSGGS",
        "parent": None,
        "source_id": None,
        "provenance": "synthetic; short glycine-serine linker in common use, with no "
        "single originating publication to cite",
    },
]

# Fusion partners taken whole (or as their mature chain) from a UniProt record.
FUSION_ENTRIES = [
    {
        "name": "GST (sj26)",
        "category": "fusion partner",
        "parent": "P08515",
        "region": None,
        "source_id": "smith1988",
        "provenance": "full Schistosoma japonicum GST sequence, the pGEX fusion partner",
    },
    {
        "name": "MBP (MalE mature chain)",
        "category": "fusion partner",
        "parent": "P0AEX9",
        # UniProt P0AEX9 annotates Signal 1-26 and Chain 27-396; a fusion carries the
        # mature chain, so the signal peptide is cut here. Checked against the UniProt
        # feature table on 2026-10-20.
        "region": (27, 396),
        "source_id": "diguan1988",
        "provenance": "mature chain (UniProt Chain 27-396) of E. coli MalE, the pMAL "
        "fusion partner; the signal peptide is not part of the fusion",
    },
]


def fetch_uniprot_fasta(accession: str, offline: bool) -> str:
    """Return the FASTA text for *accession*, caching it under ``uniprot_cache``."""
    cached = CACHE_DIR / f"{accession}.fasta"
    if cached.exists():
        return cached.read_text()
    if offline:
        raise SystemExit(f"--offline given but {cached} is missing")
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=120) as handle:
                text = handle.read().decode()
            break
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503) or attempt == 4:
                raise
            time.sleep(10 * (attempt + 1))
    else:  # pragma: no cover - the loop either breaks or raises
        raise SystemExit(f"could not fetch {accession}")
    if not text.startswith(">"):
        raise SystemExit(
            f"{accession}: UniProt did not return FASTA, got {text[:80]!r}"
        )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached.write_text(text)
    time.sleep(1)
    return text


def parse_fasta(text: str) -> tuple[str, str]:
    """Split one-record FASTA text into (header without '>', sequence)."""
    lines = text.strip().splitlines()
    return lines[0][1:], "".join(line.strip() for line in lines[1:])


def sha256(path: pathlib.Path) -> str:
    """Hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_rows(sequences: dict[str, tuple[str, str]]) -> list[dict[str, str]]:
    """Build the tag/linker table, checking every claimed parent subsequence."""
    rows: list[dict[str, str]] = []
    for entry in LITERAL_ENTRIES:
        coords = ""
        parent = entry["parent"]
        if parent is not None:
            _, parent_sequence = sequences[parent]
            start = parent_sequence.find(entry["sequence"])
            if start < 0:
                raise SystemExit(
                    f"{entry['name']}: {entry['sequence']} is not in {parent}. "
                    "Fix the sequence or the parent; do not freeze an unchecked one."
                )
            if parent_sequence.find(entry["sequence"], start + 1) >= 0:
                raise SystemExit(
                    f"{entry['name']}: {entry['sequence']} occurs more than once in "
                    f"{parent}, so the coordinates would be ambiguous."
                )
            coords = f"{start + 1}-{start + len(entry['sequence'])}"
        rows.append(
            {
                "name": entry["name"],
                "category": entry["category"],
                "sequence": entry["sequence"],
                "length_aa": str(len(entry["sequence"])),
                "source_kind": "UniProt subsequence" if parent else "synthetic",
                "source_accession": parent or "",
                "source_coords_1based": coords,
                "source_id": entry["source_id"] or "",
                "provenance": entry["provenance"],
            }
        )
    for entry in FUSION_ENTRIES:
        _, parent_sequence = sequences[entry["parent"]]
        if entry["region"] is None:
            sequence = parent_sequence
            coords = f"1-{len(parent_sequence)}"
        else:
            start, end = entry["region"]
            if end > len(parent_sequence):
                raise SystemExit(
                    f"{entry['name']}: {entry['parent']} is only "
                    f"{len(parent_sequence)} aa, cannot take {start}-{end}"
                )
            sequence = parent_sequence[start - 1 : end]
            coords = f"{start}-{end}"
        rows.append(
            {
                "name": entry["name"],
                "category": entry["category"],
                "sequence": sequence,
                "length_aa": str(len(sequence)),
                "source_kind": "UniProt record",
                "source_accession": entry["parent"],
                "source_coords_1based": coords,
                "source_id": entry["source_id"],
                "provenance": entry["provenance"],
            }
        )
    return rows


def write_references(path: pathlib.Path, retrieved: str) -> None:
    """Write the citation list, Chicago style, with the DOI for each entry."""
    lines = [
        "# Sources for the construct-check tag and linker list",
        "",
        f"Resolved against the Crossref API (api.crossref.org) on {retrieved}. The DOI of "
        "every entry came back from that query, not from memory. `source_id` in "
        "`tags_linkers.tsv` points at a key below.",
        "",
    ]
    for key, reference in sorted(REFERENCES.items()):
        lines += [
            f"## {key}",
            "",
            f"{reference['authors']}. \"{reference['title']}.\" *{reference['journal']}* "
            f"{reference['volume']} ({reference['year']}): {reference['pages']}. "
            f"https://doi.org/{reference['doi']}",
            "",
        ]
    lines += [
        "## No citation",
        "",
        "`GS linker GSG` and `GS linker GGSGGS` have an empty `source_id`. Short "
        "glycine-serine linkers of these lengths are used widely and no single paper "
        "introduced them, so the table records them as synthetic with no citation rather "
        "than attaching a paper that does not describe that exact sequence.",
        "",
    ]
    path.write_text("\n".join(lines))


def write_readme(path: pathlib.Path, today: str, n_tags: int) -> None:
    """Write the directory's README, including the submission contract.

    The column list is imported from ``notebooks/construct_check_utils.py``, the module that
    validates an arriving file, so the documented contract and the enforced one cannot
    drift apart.
    """
    import sys

    sys.path.insert(0, str(REPO_ROOT / "notebooks"))
    import construct_check_utils as cc

    lines = [
        "# Construct-check kill test: frozen data",
        "",
        f"Frozen {today} by `scripts/freeze_construct_check_data.py`. Checksums for every "
        "file are in `SHA256SUMS`. Used by notebook 355.",
        "",
        "## What the kill test asks",
        "",
        "When a protein-production facility gets a construct back and the sequenced, "
        "translated protein is not what was ordered, does writing the protein in a reduced "
        "alphabet (grouping the 20 amino acids into a handful of letters) find the "
        "unexpected piece of sequence when exact matching and local alignment do not?",
        "",
        "## What is missing",
        "",
        "The 200 historical facility construct submissions. They are not in this "
        "directory, not anywhere else in this repository, and not obtainable from a public "
        "database: they are a facility's own records of what was ordered, what came back, "
        "and what the discrepancy turned out to be. Without them the kill test cannot run. "
        "No substitute was used. A single published construct, or a construct this analysis "
        "built by putting a tag into a clean protein, is not a historical submission: it "
        "has no documented discrepancy record, and the mistakes it contains are the ones "
        "that were put there on purpose, which is the one thing the test must not assume.",
        "",
        "## The format the submissions must arrive in",
        "",
        "One tab-separated file, UTF-8, one row per submission, a header row with these "
        "column names in any order. `notebooks/construct_check_utils.py` "
        "(`load_submissions`) refuses a file that breaks any of this, so a partial export "
        "fails instead of quietly shrinking the test.",
        "",
        "### Required columns",
        "",
        "| Column | What goes in it |",
        "| --- | --- |",
    ]
    for column, description in cc.SUBMISSION_COLUMNS.items():
        lines.append(f"| `{column}` | {description} |")
    lines += [
        "",
        "### Optional columns",
        "",
        "| Column | What goes in it |",
        "| --- | --- |",
    ]
    for column, description in cc.OPTIONAL_SUBMISSION_COLUMNS.items():
        lines.append(f"| `{column}` | {description} |")
    lines += [
        "",
        "### Rules the file must satisfy",
        "",
        "1. `submission_id` is unique.",
        f"2. `discrepancy_class` is one of: "
        f"{', '.join(sorted(cc.DISCREPANCY_CLASSES))}.",
        "3. `intended_protein` and `received_protein` hold one-letter amino-acid code with "
        "no gaps, spaces or line breaks. `X` for an unknown residue and `*` for a stop are "
        "allowed; nothing else is.",
        "4. When `discrepancy_class` is `unexpected_insert`, all four `resolved_insert_*` "
        "columns are filled, and `received_protein[resolved_insert_start:"
        "resolved_insert_end]` is exactly `resolved_insert_sequence`, counting from 1 and "
        "including both ends. This is checked on load, because a coordinate convention that "
        "is off by one would move every number in the result.",
        "5. `insert_was_intended` is `true` or `false`, never empty, for every insert. The "
        "test is about telling a tag the customer ordered from one nobody ordered, so a "
        "blank here removes the row from both sides of that comparison.",
        "6. `split` is `dev` or `test`, assigned so that every submission sharing an "
        "intended protein is in the same split. The alphabet and k are chosen on `dev` "
        "only, then frozen.",
        "",
        "### What else is needed, beyond the file",
        "",
        "- **Review times.** `review_minutes` per batch, as the facility recorded them, "
        "before and after adding the new flags. Threshold 4 of the pass rule is about "
        "review time. Without these numbers it is reported as not assessable, which under "
        "the fixed rule is not a pass. A count of flagged regions is not a substitute.",
        "- **The DNA, where it was kept.** `received_dna`, plus the facility's own "
        "DNA-level call file (VecScreen or equivalent) with `submission_id`, "
        "`protein_start`, `protein_end` and `hit_name`. The DNA check is part of the "
        "comparison the protein method is measured against and is not left out.",
        "- **The insert sequences from the discrepancy records.** The list in this "
        "directory holds only the standard tags and linkers. Any other insert that the "
        "records resolved (a vector fragment, a fragment of another gene) comes with the "
        "submissions, with its own source recorded.",
        "",
        "## What is here",
        "",
        f"- `tags_linkers.tsv`: {n_tags} tag, linker and fusion-partner sequences, with "
        "where each one came from. Every sequence is either a subsequence of a named "
        "UniProt entry at recorded coordinates, or synthetic with the publication that "
        "introduced it. The build fails if a sequence claimed to come from a UniProt entry "
        "is not in it.",
        "- `clean_host_proteins_swissprot.fasta`: six Swiss-Prot proteins "
        "(BCL2, Ced9, MCL1, BAX, p53, EGFR). These are hosts for the mechanics check only. "
        "They are not construct submissions and no number from them counts toward the "
        "verdict.",
        "- `references.md`: the full citation for every sequence whose source is a paper, "
        "resolved against Crossref.",
        "- `uniprot_cache/`: the UniProt records the two data files were built from, so the "
        "build repeats without the network (`--offline`).",
        "- `pass_rule.json`: the four thresholds and the three fail clauses, written before "
        "any result existed.",
        "- `manifest.json`, `SHA256SUMS`: what was frozen, when, and its checksums.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="build only from the cached UniProt records, do not touch the network",
    )
    args = parser.parse_args()

    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()

    sequences: dict[str, tuple[str, str]] = {}
    for accession in list(TAG_PARENTS) + list(CLEAN_HOSTS):
        sequences[accession] = parse_fasta(fetch_uniprot_fasta(accession, args.offline))

    rows = build_rows(sequences)
    columns = [
        "name",
        "category",
        "sequence",
        "length_aa",
        "source_kind",
        "source_accession",
        "source_coords_1based",
        "source_id",
        "provenance",
    ]
    tsv_path = FROZEN_DIR / "tags_linkers.tsv"
    with tsv_path.open("w") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join(row[column] for column in columns) + "\n")
    print(f"wrote {tsv_path} with {len(rows)} sequences")

    hosts_path = FROZEN_DIR / "clean_host_proteins_swissprot.fasta"
    with hosts_path.open("w") as handle:
        for accession, label in CLEAN_HOSTS.items():
            header, sequence = sequences[accession]
            handle.write(f">{header}\n")
            for start in range(0, len(sequence), 60):
                handle.write(sequence[start : start + 60] + "\n")
            print(f"  host {accession} {label} {len(sequence)} aa")

    # A host that already contains a tag sequence would make a false flag that is not the
    # method's fault. Report any such case instead of hiding it.
    print("tag sequences already present in a clean host protein:")
    found_any = False
    for accession in CLEAN_HOSTS:
        _, host_sequence = sequences[accession]
        for row in rows:
            if row["sequence"] in host_sequence:
                position = host_sequence.find(row["sequence"]) + 1
                print(f"  {accession} contains {row['name']} at {position}")
                found_any = True
    if not found_any:
        print("  none")

    write_references(FROZEN_DIR / "references.md", today)
    write_readme(FROZEN_DIR / "README.md", today, len(rows))

    # The pass rule is written from the module that scores it, before any result exists, so
    # the committed thresholds can be diffed against the ones any later run used.
    import sys

    sys.path.insert(0, str(REPO_ROOT / "notebooks"))
    import construct_check_utils as cc

    cc.write_pass_rule(FROZEN_DIR / "pass_rule.json")

    manifest = {
        "frozen_date": today,
        "built_by": "scripts/freeze_construct_check_data.py",
        "uniprot_accessions": {**TAG_PARENTS, **CLEAN_HOSTS},
        "n_tag_sequences": len(rows),
        "submissions_present": False,
        "submissions_note": "The 200 historical facility construct submissions are not in "
        "this directory and were never available to the analysis. See README.md for the "
        "format they must arrive in.",
    }
    (FROZEN_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # Written last so that it covers every other frozen file, manifest.json included.
    sums_path = FROZEN_DIR / "SHA256SUMS"
    frozen_files = sorted(
        path
        for path in FROZEN_DIR.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS"}
    )
    with sums_path.open("w") as handle:
        handle.write(f"# frozen {today} by scripts/freeze_construct_check_data.py\n")
        for path in frozen_files:
            handle.write(f"{sha256(path)}  {path.relative_to(FROZEN_DIR)}\n")
    print(f"wrote {sums_path} covering {len(frozen_files)} files")


if __name__ == "__main__":
    main()
