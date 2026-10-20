"""Tests for notebooks/construct_check_utils.py, the construct-check kill test workflow.

The 200 facility submissions the kill test needs do not exist in the repository, so every
test here runs on the frozen tag list and on constructs this file builds from Swiss-Prot
host proteins. These tests check the mechanics: that each method finds an insert it is given,
that the coordinates it reports are the right residues, that the loader refuses a file that
breaks the contract, and that the pass rule scores a case whose answer is known by hand.

None of these tests is evidence that the method works on real submissions.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "notebooks"))

import construct_check_utils as cc  # noqa: E402

FROZEN_DIR = REPO_ROOT / "data" / "kill_tests" / "construct_check"


@pytest.fixture(scope="module")
def tags():
    return cc.load_tag_list()


@pytest.fixture(scope="module")
def hosts():
    """The frozen clean host proteins, as {name: sequence}."""
    sequences = {}
    name = None
    for line in (
        (FROZEN_DIR / "clean_host_proteins_swissprot.fasta").read_text().split("\n")
    ):
        if line.startswith(">"):
            name = line[1:].split("|")[2].split(" ")[0]
            sequences[name] = []
        elif line.strip() and name:
            sequences[name].append(line.strip())
    return {name: "".join(parts) for name, parts in sequences.items()}


def test_tag_list_is_frozen_and_complete(tags):
    """Every tag the task named is present, with a source recorded for each sequence."""
    names = {tag.name for tag in tags}
    for wanted in ("His6", "FLAG", "HA", "Myc", "Strep-tag II"):
        assert wanted in names, f"{wanted} missing from the frozen tag list"
    assert any(tag.name.startswith("GST") for tag in tags)
    assert any(tag.name.startswith("MBP") for tag in tags)
    assert any(tag.category == "linker" for tag in tags)
    for tag in tags:
        assert tag.sequence, f"{tag.name} has no sequence"
        assert set(tag.sequence) <= set(cc.AA_FREQUENCIES), tag.name
        assert tag.provenance, f"{tag.name} has no provenance recorded"
        if tag.source_kind in {"UniProt subsequence", "UniProt record"}:
            assert tag.source_accession, f"{tag.name} claims UniProt but names no entry"
            assert tag.source_coords_1based, f"{tag.name} names no coordinates"


def test_tag_subsequences_really_are_in_their_named_parent(tags):
    """A tag whose source is a UniProt subsequence is at the coordinates recorded."""
    for tag in tags:
        if tag.source_kind != "UniProt subsequence":
            continue
        cached = FROZEN_DIR / "uniprot_cache" / f"{tag.source_accession}.fasta"
        parent = "".join(cached.read_text().split("\n")[1:])
        start, end = (int(part) for part in tag.source_coords_1based.split("-"))
        assert parent[start - 1 : end] == tag.sequence, (
            f"{tag.name}: {tag.source_accession} {tag.source_coords_1based} is "
            f"{parent[start - 1:end]!r}, not {tag.sequence!r}"
        )


def test_checksums_match_the_frozen_files():
    """Every file listed in SHA256SUMS still has the checksum recorded."""
    import hashlib

    lines = [
        line
        for line in (FROZEN_DIR / "SHA256SUMS").read_text().split("\n")
        if line and not line.startswith("#")
    ]
    assert lines, "SHA256SUMS is empty"
    for line in lines:
        digest, relative = line.split("  ", 1)
        path = FROZEN_DIR / relative
        assert path.exists(), f"{relative} is in SHA256SUMS but missing"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, relative


def test_encode_maps_every_residue_and_marks_unknown():
    """The reduced alphabet covers all 20 residues; X cannot match anything."""
    all_residues = "".join(sorted(cc.AA_FREQUENCIES))
    encoded = cc.encode(all_residues, "hp_thomas_dill")
    assert len(encoded) == len(all_residues)
    assert set(encoded) == {"h", "p"}
    assert cc.encode("ACX", "hp_thomas_dill") == "hh?"
    assert "?" not in cc.encode("ACDEFG", "hp_thomas_dill")


def test_exact_tag_match_finds_an_inserted_tag_at_the_right_residues(tags, hosts):
    """A FLAG tag put into BCL2 is found, and the coordinates are the tag's residues."""
    flag_tag = next(tag for tag in tags if tag.name == "FLAG")
    host = hosts["BCL2_HUMAN"]
    received, start, end = cc.insert_at_random_position(host, flag_tag.sequence, seed=1)
    flags = cc.exact_tag_flags("s1", received, tags)
    matching = [flag for flag in flags if flag.insert_name == "FLAG"]
    assert len(matching) == 1
    assert (matching[0].received_start, matching[0].received_end) == (start, end)
    assert received[start - 1 : end] == flag_tag.sequence


def test_exact_tag_match_misses_a_tag_with_one_substitution(tags, hosts):
    """One changed residue is enough to lose an exact match. This is the gap under test."""
    flag_tag = next(tag for tag in tags if tag.name == "FLAG")
    altered = flag_tag.sequence[:3] + "E" + flag_tag.sequence[4:]
    assert altered != flag_tag.sequence
    received, _, _ = cc.insert_at_random_position(hosts["BCL2_HUMAN"], altered, seed=2)
    names = {flag.insert_name for flag in cc.exact_tag_flags("s1", received, tags)}
    assert "FLAG" not in names


def test_reduced_kmer_check_finds_a_long_inserted_tag(tags, hosts):
    """The MBP fusion partner put into Ced9 is flagged by the reduced-alphabet check."""
    mbp = next(tag for tag in tags if tag.name.startswith("MBP"))
    received, start, end = cc.insert_at_random_position(
        hosts["CED9_CAEEL"], mbp.sequence, seed=3
    )
    flags = cc.reduced_kmer_flags(
        "s1", received, [mbp], alphabet="hp_thomas_dill", ksize=20
    )
    assert flags, "no flag on a 370 aa insert"
    assert any(flag.overlaps(start, end) > 0 for flag in flags)


def test_reduced_kmer_check_skips_a_tag_shorter_than_k(tags):
    """A tag shorter than k has no k-mers, so it can never be flagged at that k."""
    short = next(tag for tag in tags if tag.name == "GS linker GSG")
    assert len(short.sequence) < 20
    assert cc.reduced_kmer_flags("s1", "A" * 300, [short], "hp_thomas_dill", 20) == []


def test_reduced_kmer_flag_coordinates_cover_the_matching_window(tags):
    """The flag's coordinates are the residues whose k-mers matched, inclusive."""
    mbp = next(tag for tag in tags if tag.name.startswith("MBP"))
    ksize = 20
    flags = cc.reduced_kmer_flags("s1", mbp.sequence, [mbp], "hp_thomas_dill", ksize)
    assert len(flags) == 1
    assert flags[0].received_start == 1
    assert flags[0].received_end == len(mbp.sequence)


def test_alignment_flags_find_an_insert_and_name_its_length(hosts):
    """Local alignment reports the inserted stretch of the received protein."""
    intended = hosts["BCL2_HUMAN"]
    insert = "WSHPQFEKGGGGSWSHPQFEK"
    received, start, end = cc.insert_at_random_position(intended, insert, seed=4)
    flags = cc.alignment_flags("s1", intended, received)
    assert flags, "local alignment covered an insert it should have left out"
    assert any(flag.overlaps(start, end) >= len(insert) - 2 for flag in flags)


def test_alignment_flags_leave_a_clean_construct_alone(hosts):
    """Received identical to intended produces no flag."""
    host = hosts["BAX_HUMAN"]
    assert cc.alignment_flags("s1", host, host) == []


def test_match_line_and_identical_count(hosts):
    """The printout's match line and its count agree with the residues above it."""
    host = hosts["BCL2_HUMAN"]
    altered = host[:50] + "W" + host[51:]
    text = cc.format_flagged_region(host, altered, 51, 51, alphabet="hp_thomas_dill")
    assert "intended" in text and "received" in text
    window = text.split("\n")[1]
    identical, _, compared = window.split()[0], None, window.split()[2]
    assert int(compared) - int(identical) == 1, window
    assert "^" in text


def test_format_flagged_region_shows_the_insert_residues(tags, hosts):
    """Every residue of the inserted tag appears on the received lines of the printout."""
    mbp = next(tag for tag in tags if tag.name.startswith("MBP"))
    intended = hosts["CED9_CAEEL"]
    received, start, end = cc.insert_at_random_position(intended, mbp.sequence, seed=5)
    text = cc.format_flagged_region(intended, received, start, end, context=0)
    printed = "".join(
        line.split()[-1]
        for line in text.split("\n")
        if line.strip().startswith("received ")
    )
    assert (
        printed == mbp.sequence
    ), f"the printout shows {len(printed)} residues, the insert is {len(mbp.sequence)}"


def test_shuffled_insert_keeps_composition_and_length(tags):
    """The shuffled-insert control changes the order, not the composition."""
    mbp = next(tag for tag in tags if tag.name.startswith("MBP"))
    shuffled = cc.shuffle_sequence(mbp.sequence, seed=6)
    assert shuffled != mbp.sequence
    assert sorted(shuffled) == sorted(mbp.sequence)


def test_composition_matched_stretch_is_from_the_donor(hosts):
    """The composition-matched control is a real stretch of a real protein."""
    stretch = cc.composition_matched_stretch(
        hosts["EGFR_HUMAN"], length=40, target_composition="GGGGSGGGGS", seed=7
    )
    assert len(stretch) == 40
    assert stretch in hosts["EGFR_HUMAN"]


def test_insert_at_random_position_keeps_away_from_the_termini(hosts):
    """The control insert is not at either end, where a real tag would sit."""
    host = hosts["MCL1_HUMAN"]
    received, start, end = cc.insert_at_random_position(
        host, "HHHHHH", seed=8, margin=20
    )
    assert start > 20
    assert end < len(received) - 20
    assert received[start - 1 : end] == "HHHHHH"


def test_loader_refuses_a_file_missing_a_contract_column(tmp_path):
    """A submissions export without the resolved coordinates fails loudly."""
    path = tmp_path / "submissions.tsv"
    path.write_text(
        "submission_id\tintended_protein\treceived_protein\n" "s1\tAAA\tAAA\n"
    )
    with pytest.raises(ValueError, match="missing required columns"):
        cc.load_submissions(path)


def test_loader_refuses_an_insert_whose_coordinates_do_not_hold_it(tmp_path, hosts):
    """A resolved insert that is not at the coordinates given is rejected."""
    host = hosts["BAX_HUMAN"]
    received, start, end = cc.insert_at_random_position(host, "DYKDDDDK", seed=9)
    columns = list(cc.SUBMISSION_COLUMNS)
    row = {
        "submission_id": "s1",
        "intended_protein": host,
        "received_protein": received,
        "discrepancy_class": "unexpected_insert",
        "resolved_insert_name": "FLAG",
        "resolved_insert_sequence": "DYKDDDDK",
        "resolved_insert_start": str(start + 3),  # wrong on purpose
        "resolved_insert_end": str(end + 3),
        "insert_was_intended": "false",
        "split": "test",
    }
    path = tmp_path / "submissions.tsv"
    path.write_text(
        "\t".join(columns) + "\n" + "\t".join(row[column] for column in columns) + "\n"
    )
    with pytest.raises(ValueError, match="not the resolved insert"):
        cc.load_submissions(path)

    row["resolved_insert_start"] = str(start)
    row["resolved_insert_end"] = str(end)
    path.write_text(
        "\t".join(columns) + "\n" + "\t".join(row[column] for column in columns) + "\n"
    )
    assert len(cc.load_submissions(path)) == 1


def test_pass_rule_fails_a_case_built_to_fail(tags, hosts):
    """A worked case whose answer is known by hand: nothing recovered, so threshold 1 fails."""
    host = hosts["BAX_HUMAN"]
    rows = [
        {
            "submission_id": "clean1",
            "intended_protein": host,
            "received_protein": host,
            "discrepancy_class": "clean",
            "resolved_insert_name": "",
            "resolved_insert_sequence": "",
            "resolved_insert_start": "",
            "resolved_insert_end": "",
            "insert_was_intended": "false",
            "split": "test",
        },
        {
            "submission_id": "insert1",
            "intended_protein": host,
            "received_protein": host[:50] + "DYKDDDDK" + host[50:],
            "discrepancy_class": "unexpected_insert",
            "resolved_insert_name": "FLAG",
            "resolved_insert_sequence": "DYKDDDDK",
            "resolved_insert_start": "51",
            "resolved_insert_end": "58",
            "insert_was_intended": "false",
            "split": "test",
        },
    ]
    reduced_method = "kmerseek reduced alphabet (hp_thomas_dill, k=20)"
    result = cc.evaluate_pass_rule(
        rows,
        {
            reduced_method: [],  # finds nothing: an 8 aa tag has no 20-mer
            "exact tag match (protein20)": cc.exact_tag_flags(
                "insert1", rows[1]["received_protein"], tags
            ),
        },
        reduced_method=reduced_method,
    )
    assert result["1_recall_of_documented_inserts"]["value"] == 0.0
    assert result["1_recall_of_documented_inserts"]["holds"] is False
    assert result["2_false_flags_per_100_clean"]["value"] == 0.0
    assert result["2_false_flags_per_100_clean"]["holds"] is True
    assert result["3_comparison_missed_and_found"]["value"] == 0
    assert result["4_median_review_time_per_batch"]["holds"] is False
    assert "not assessable" in result["4_median_review_time_per_batch"]["counts"]
    assert result["fail_clauses"]["exact_already_catches"] is True
    assert result["verdict"] == "FAIL"


def test_pass_rule_passes_a_case_built_to_pass(hosts):
    """The same scorer returns PASS when every threshold is met, so FAIL is not hardwired."""
    host = hosts["MCL1_HUMAN"]
    rows = []
    for index in range(100):
        rows.append(
            {
                "submission_id": f"clean{index}",
                "intended_protein": host,
                "received_protein": host,
                "discrepancy_class": "clean",
                "resolved_insert_name": "",
                "resolved_insert_sequence": "",
                "resolved_insert_start": "",
                "resolved_insert_end": "",
                "insert_was_intended": "false",
                "split": "test",
            }
        )
    reduced_method = "kmerseek reduced alphabet (hp_thomas_dill, k=20)"
    reduced_flags = []
    for index in range(6):
        received = host[:60] + "W" * 30 + host[60:]
        rows.append(
            {
                "submission_id": f"insert{index}",
                "intended_protein": host,
                "received_protein": received,
                "discrepancy_class": "unexpected_insert",
                "resolved_insert_name": "unnamed",
                "resolved_insert_sequence": "W" * 30,
                "resolved_insert_start": "61",
                "resolved_insert_end": "90",
                "insert_was_intended": "false",
                "split": "test",
            }
        )
        reduced_flags.append(
            cc.Flag(f"insert{index}", reduced_method, 61, 90, "unnamed", "test case")
        )
    result = cc.evaluate_pass_rule(
        rows,
        {reduced_method: reduced_flags, "exact tag match (protein20)": []},
        reduced_method=reduced_method,
        review_minutes_before=[30.0, 32.0, 31.0],
        review_minutes_after=[29.0, 31.0, 30.0],
    )
    assert result["1_recall_of_documented_inserts"]["value"] == 1.0
    assert result["3_comparison_missed_and_found"]["value"] == 6
    assert result["4_median_review_time_per_batch"]["holds"] is True
    assert result["verdict"] == "PASS"


def test_choose_alphabet_and_k_refuses_when_nothing_fits_the_budget(tags, hosts):
    """The selection rule raises rather than quietly widening the false-flag budget."""
    host = hosts["BAX_HUMAN"]
    dev_rows = [
        {
            "submission_id": "clean1",
            "intended_protein": host,
            "received_protein": host,
            "discrepancy_class": "clean",
            "resolved_insert_name": "",
            "resolved_insert_sequence": "",
            "resolved_insert_start": "",
            "resolved_insert_end": "",
            "insert_was_intended": "false",
            "split": "dev",
        }
    ]
    with pytest.raises(ValueError, match="false-flag budget"):
        cc.choose_alphabet_and_k(
            dev_rows,
            tags,
            alphabets=["hp_thomas_dill"],
            ksizes=[4],
            false_flag_budget_per_100=0.0,
        )
