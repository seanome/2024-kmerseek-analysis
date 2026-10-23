"""Mechanics check for the construct-check kill test (notebook 355).

This does NOT run the kill test. The 200 historical facility construct submissions the test
needs do not exist in this repository, so there is nothing here to measure recall or false
flags on. What this script does is check that the three methods work on constructs it builds
itself, by putting each frozen tag into each frozen clean Swiss-Prot host protein. Every
number it writes is about sequences this script made up, so no number counts toward the
verdict.

Writes three tables under ``tables/``:

- ``355-mechanics_tag_kmers_by_ksize.tsv``: for each tag and each k, whether the tag is long
  enough to have even one k-mer. This is a fact about the frozen tag list and does not
  depend on any construct.
- ``355-mechanics_recovery_by_method.tsv``: one row per built construct and method, saying
  whether the method flagged the inserted tag.
- ``355-mechanics_false_flags_on_controls.tsv``: flags raised on clean hosts, on hosts
  carrying a shuffled insert, and on hosts carrying a composition-matched natural stretch.

Run with the project interpreter:

    /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 \
        scripts/run_nb355_mechanics_check.py
"""

from __future__ import annotations

import pathlib
import zlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "notebooks"))

import construct_check_utils as cc  # noqa: E402

FROZEN_DIR = REPO_ROOT / "data" / "kill_tests" / "construct_check"
TABLES_DIR = REPO_ROOT / "tables"

# The two alphabets the mechanics check runs: the 2-letter hydrophobic/polar split kmerseek
# uses for distant homology, and the full 20-letter alphabet as the reference point.
ALPHABETS = ["hp_thomas_dill", "protein"]
# k from 3 to 30. The house rule is that an HP alphabet is only run at k of 19 or more, so
# the whole range is computed and that part of it is marked, rather than showing one k.
KSIZES = list(range(3, 31))
REALISTIC_HP_KSIZES = (19, 30)
# One substitution in the middle of a tag: the case exact matching is expected to lose.
SUBSTITUTION_RESIDUE = "E"


def load_hosts() -> dict[str, str]:
    """The frozen clean host proteins, as {short name: sequence}."""
    hosts: dict[str, list[str]] = {}
    name = None
    text = (FROZEN_DIR / "clean_host_proteins_swissprot.fasta").read_text()
    for line in text.split("\n"):
        if line.startswith(">"):
            name = line[1:].split("|")[2].split(" ")[0]
            hosts[name] = []
        elif line.strip() and name:
            hosts[name].append(line.strip())
    return {name: "".join(parts) for name, parts in hosts.items()}


def substitute_one_residue(sequence: str) -> str:
    """Change the middle residue of *sequence*, so an exact search cannot match it."""
    middle = len(sequence) // 2
    replacement = (
        SUBSTITUTION_RESIDUE if sequence[middle] != SUBSTITUTION_RESIDUE else "K"
    )
    return sequence[:middle] + replacement + sequence[middle + 1 :]


def write_tsv(path: pathlib.Path, columns: list[str], rows: list[dict]) -> None:
    """Write *rows* as a tab-separated file with a header."""
    with path.open("w") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join(str(row[column]) for column in columns) + "\n")
    print(f"wrote {path} with {len(rows)} rows")


def tag_kmers_by_ksize(tags) -> list[dict]:
    """For each tag and k, the number of k-mers the tag has."""
    rows = []
    for tag in tags:
        for ksize in KSIZES:
            n_kmers = max(0, len(tag.sequence) - ksize + 1)
            rows.append(
                {
                    "tag_name": tag.name,
                    "category": tag.category,
                    "tag_length_aa": len(tag.sequence),
                    "ksize": ksize,
                    "n_kmers": n_kmers,
                    "has_any_kmer": "true" if n_kmers > 0 else "false",
                }
            )
    return rows


def recovery_by_method(tags, hosts, alphabet: str, ksize: int) -> list[dict]:
    """Run the three protein methods on one built construct per tag and host."""
    rows = []
    for tag in tags:
        for host_name, host in hosts.items():
            for insert_kind in ("exact tag", "tag with one substitution"):
                insert = (
                    tag.sequence
                    if insert_kind == "exact tag"
                    else substitute_one_residue(tag.sequence)
                )
                # crc32, not hash(): Python randomises str hashing per process, so hash()
                # would put the insert somewhere different on every run.
                seed = zlib.crc32(f"{tag.name}|{host_name}|{insert_kind}".encode())
                received, start, end = cc.insert_at_random_position(host, insert, seed)
                methods = {
                    "exact tag match (protein20)": cc.exact_tag_flags(
                        "built", received, tags
                    ),
                    "local alignment (BLOSUM62)": cc.alignment_flags(
                        "built", host, received
                    ),
                    f"kmerseek reduced alphabet ({alphabet}, k={ksize})": (
                        cc.reduced_kmer_flags("built", received, tags, alphabet, ksize)
                    ),
                }
                for method, flags in methods.items():
                    hits = [flag for flag in flags if flag.overlaps(start, end) > 0]
                    names = {flag.insert_name for flag in hits if flag.insert_name}
                    rows.append(
                        {
                            "tag_name": tag.name,
                            "category": tag.category,
                            "tag_length_aa": len(tag.sequence),
                            "host": host_name,
                            "insert_kind": insert_kind,
                            "method": method,
                            "insert_start": start,
                            "insert_end": end,
                            # "flagged" is the loose measure: a flag touching the insert,
                            # whatever it names. "flagged_named_right" also requires the
                            # flag to name the tag that was actually put in, because a flag
                            # that lands on the insert while naming a different tag does
                            # not tell a reviewer what is there. Local alignment names no
                            # tag at all, so it is scored loose only and its
                            # flagged_named_right column is "not applicable".
                            "flagged": "true" if hits else "false",
                            "flagged_named_right": (
                                "not applicable"
                                if method.startswith("local alignment")
                                else "true" if tag.name in names else "false"
                            ),
                            "flag_names_on_insert": ";".join(sorted(names)) or "none",
                            "n_flags_on_insert": len(hits),
                            "n_flags_total": len(flags),
                        }
                    )
    return rows


def false_flags_on_controls(tags, hosts, alphabet: str, ksize: int) -> list[dict]:
    """Flags on clean hosts and on the two mechanics controls.

    A clean host carries no insert at all, so every flag on it is a false one. The shuffled
    and composition-matched controls carry a stretch that is not a tag, so a flag on the
    inserted region is also a false one.
    """
    rows = []
    donor = max(hosts.values(), key=len)
    for host_name, host in hosts.items():
        constructs = {"clean": (host, None, None)}
        for tag in tags:
            if len(tag.sequence) < ksize:
                continue
            shuffled = cc.shuffle_sequence(tag.sequence, seed=11)
            constructs[f"shuffled {tag.name}"] = cc.insert_at_random_position(
                host, shuffled, seed=12
            )
            natural = cc.composition_matched_stretch(
                donor, len(tag.sequence), tag.sequence, seed=13
            )
            constructs[f"composition-matched stretch for {tag.name}"] = (
                cc.insert_at_random_position(host, natural, seed=14)
            )
        for construct_kind, (received, start, end) in constructs.items():
            methods = {
                "exact tag match (protein20)": cc.exact_tag_flags(
                    "control", received, tags
                ),
                "local alignment (BLOSUM62)": cc.alignment_flags(
                    "control", host, received
                ),
                f"kmerseek reduced alphabet ({alphabet}, k={ksize})": (
                    cc.reduced_kmer_flags("control", received, tags, alphabet, ksize)
                ),
            }
            for method, flags in methods.items():
                on_insert = (
                    0
                    if start is None
                    else sum(1 for flag in flags if flag.overlaps(start, end) > 0)
                )
                rows.append(
                    {
                        "host": host_name,
                        "construct_kind": construct_kind,
                        "method": method,
                        "n_flags_total": len(flags),
                        "n_flags_on_inserted_region": on_insert,
                        "flag_names": ";".join(
                            sorted({flag.insert_name or "unnamed" for flag in flags})
                        )
                        or "none",
                    }
                )
    return rows


def main() -> None:
    TABLES_DIR.mkdir(exist_ok=True)
    tags = cc.load_tag_list()
    hosts = load_hosts()
    print(f"{len(tags)} frozen tag sequences, {len(hosts)} clean host proteins")

    # k for the mechanics check: the smallest k in the realistic HP range, which gives the
    # reduced-alphabet check its best chance of reaching a short tag. A larger k can only
    # reach fewer tags, so this is the favourable end of the range, not a cherry-picked one.
    ksize = REALISTIC_HP_KSIZES[0]
    alphabet = "hp_thomas_dill"
    print(f"mechanics check at {alphabet}, k={ksize}")

    write_tsv(
        TABLES_DIR / "355-mechanics_tag_kmers_by_ksize.tsv",
        ["tag_name", "category", "tag_length_aa", "ksize", "n_kmers", "has_any_kmer"],
        tag_kmers_by_ksize(tags),
    )
    write_tsv(
        TABLES_DIR / "355-mechanics_recovery_by_method.tsv",
        [
            "tag_name",
            "category",
            "tag_length_aa",
            "host",
            "insert_kind",
            "method",
            "insert_start",
            "insert_end",
            "flagged",
            "flagged_named_right",
            "flag_names_on_insert",
            "n_flags_on_insert",
            "n_flags_total",
        ],
        recovery_by_method(tags, hosts, alphabet, ksize),
    )
    write_tsv(
        TABLES_DIR / "355-mechanics_false_flags_on_controls.tsv",
        [
            "host",
            "construct_kind",
            "method",
            "n_flags_total",
            "n_flags_on_inserted_region",
            "flag_names",
        ],
        false_flags_on_controls(tags, hosts, alphabet, ksize),
    )

    n_reachable = sum(1 for tag in tags if len(tag.sequence) >= ksize)
    print(
        f"\n{n_reachable} of {len(tags)} frozen tags are at least {ksize} aa, so only "
        f"{n_reachable} can produce a single k-mer at k={ksize}"
    )


if __name__ == "__main__":
    main()
