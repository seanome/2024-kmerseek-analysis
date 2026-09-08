"""The recall-ceiling section must not claim a ceiling the numbers do not support.

Its caption said "no search can transfer a family the target does not have, so every
recall_reachable in this report divides by the reachable bar only". On the midi run the
bar returned exactly 7_000 reachable / 0 unreachable for eight of nine species, and
6_991 / 9 for the ninth. E. coli does not share every human Pfam family with mouse; the
denominator was not measuring anything.

Two separate defects, both visible in that one figure:

1. Degenerate vocabulary. The default primary truth set is Swiss-Prot, whose `pfam_id`
   column holds one of ~15 curated feature types rather than a Pfam accession. Every
   proteome carries nearly all of them, so the reachability join matches everything and
   recall_reachable is plain recall. On the Pfam truth set the same run gives real
   variation -- 298 / 2_358 reachable for E. coli against 2_334 / 2_358 for mouse.

2. The bar counts labels, not annotated proteins, so a species with 28 annotated target
   proteins still scores 99.9% reachable. That is why it could not detect the Ciona
   collapse it was the natural place to detect.
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))
import build_multiqc_inputs as bmi  # noqa: E402


def per(rows):
    return pl.DataFrame(rows)


SWISSPROT_SHAPE = per([
    {"species": "mouse", "n_truth_families": 12, "n_target_map_proteins": 17_228},
    {"species": "chicken", "n_truth_families": 12, "n_target_map_proteins": 2_309},
    {"species": "ecoli", "n_truth_families": 12, "n_target_map_proteins": 4_531},
    {"species": "ciona", "n_truth_families": 12, "n_target_map_proteins": 28},
])

PFAM_SHAPE = per([
    {"species": "mouse", "n_truth_families": 912, "n_target_map_proteins": 17_228},
    {"species": "chicken", "n_truth_families": 912, "n_target_map_proteins": 12_010},
    {"species": "ecoli", "n_truth_families": 912, "n_target_map_proteins": 4_531},
    {"species": "ciona", "n_truth_families": 912, "n_target_map_proteins": 9_800},
])


def test_a_feature_type_vocabulary_withdraws_the_ceiling_claim():
    note = bmi.reachability_caveat(SWISSPROT_SHAPE, "swissprot")
    assert "not a ceiling" in note
    assert "plain recall" in note


def test_a_thin_species_is_named_in_the_caption():
    note = bmi.reachability_caveat(SWISSPROT_SHAPE, "swissprot")
    assert "ciona" in note
    assert "28 annotated" in note


def test_a_family_vocabulary_with_healthy_targets_says_nothing_extra():
    assert bmi.reachability_caveat(PFAM_SHAPE, "pfam") == ""


def test_a_table_without_the_new_columns_says_nothing_extra():
    old = PFAM_SHAPE.drop("n_truth_families", "n_target_map_proteins")
    assert bmi.reachability_caveat(old, "pfam") == ""
