"""Tests for the venom-peptide retrieval benchmark.

None of these read the downloaded data. The two sequences the regression test is built on
are written out in full, so the test keeps working after a UniProt or ConoServer release
changes what is in the database.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from venom_retrieval import encodings, evaluate, frameworks, manifest, methods  # noqa: E402
from venom_retrieval.align import normalised_local_score  # noqa: E402

# --------------------------------------------------------------------------------------
# The regression case: two peptides a two-class cysteine alphabet cannot tell apart.
# --------------------------------------------------------------------------------------

#: Conotoxin Mr6.14 from Conus marmoreus (ConoServer P05381, O2 superfamily).
MR614 = "SCDQTGEPCVLNEQCCYGWCTNHGTCY"

#: Nemertide alpha-8 from the ribbon worm Riseriellus occultus (UniProt P0DQT0), a sodium
#: channel toxin. A different phylum and a different family from Mr6.14.
NEMERTIDE_ALPHA8 = "GCIATGSFCTLSKGCCTKNCGWNFACN"


def test_cys2_cannot_tell_the_two_toxins_apart():
    """The two-class representation must stay unable to separate this pair.

    Both peptides are 27 residues with six cysteines in the same places, so cys2 writes
    them as one string. Keeping this as a test means a later change to the alphabet, or to
    the encoder, cannot quietly make the pair separable and leave the collision report
    saying something that is no longer true.
    """
    assert encodings.encode(MR614, "cys2") == encodings.encode(
        NEMERTIDE_ALPHA8, "cys2"
    ) == "xcxxxxxxcxxxxxccxxxcxxxxxcx"


def test_the_pair_is_not_two_copies_of_one_peptide():
    """The pair is only interesting if the two peptides are genuinely different."""
    assert MR614 != NEMERTIDE_ALPHA8
    # A quarter of the alignment score of the weaker peptide against itself: related at
    # best distantly, and nothing a reader would call the same peptide.
    assert normalised_local_score(MR614, NEMERTIDE_ALPHA8) < 0.35


def test_every_framework_level_method_ties_the_pair():
    """Framework, loop lengths and loop composition all score the pair as a match.

    This is the point of separating the endpoints. The pair shares a framework and even
    the exact loop lengths, so no method working at that level can rank one above the
    other, whatever their measured activities turn out to be.
    """
    features = methods.build_features(
        ["mr614", "alpha8"], [MR614, NEMERTIDE_ALPHA8], kmer_settings=[]
    )
    query, target = features
    assert query.framework == target.framework == "C-C-CC-C-C"
    assert query.loop_lengths == target.loop_lengths
    assert methods.framework_only(query, target) == 1.0
    assert methods.loop_lengths_only(query, target) == 1.0


def test_residue_level_alphabets_do_separate_the_pair():
    """Giving the residues between the cysteines any classes at all separates them.

    cys2 is the only alphabet here that cannot. This is what says the collision is a
    property of the two-class reading rather than of the two peptides.
    """
    for alphabet in ("cys_hp3", "hp_pbotc_1st_ed2", "dayhoff6", "protein20"):
        assert encodings.encode(MR614, alphabet) != encodings.encode(
            NEMERTIDE_ALPHA8, alphabet
        ), alphabet


# --------------------------------------------------------------------------------------
# Frameworks, loops and connectivity
# --------------------------------------------------------------------------------------


def test_loop_layout_keeps_the_cysteines_out_of_the_segments():
    layout = frameworks.loop_layout("AACGGCTT")
    assert layout.nterm == "AA"
    assert layout.loops == ("GG",)
    assert layout.cterm == "TT"
    assert "C" not in "".join(layout.loops)


def test_compatibility_needs_the_same_cysteine_count():
    assert frameworks.frameworks_compatible("CACAC", "CAACAAC")
    assert not frameworks.frameworks_compatible("CAC", "CACAC")
    assert not frameworks.frameworks_compatible(
        "CACAC", "CAACAAC", max_loop_difference=0
    )


def test_compatibility_says_nothing_about_bonding():
    """Two peptides can be compatible and still be bonded differently.

    Nothing in this benchmark may read connectivity off cysteine order, so compatibility
    is defined on the ordering alone and the two 1-4,2-5,3-6 and 1-3,2-5,4-6 patterns are
    both allowed for the same pair of frameworks.
    """
    assert frameworks.frameworks_compatible(
        "CAACAACAACAACAAC"[:16], "CAACAACAACAACAAC"[:16]
    )
    assert frameworks.connectivity_is_complete("1-4,2-5,3-6", 6)
    assert frameworks.connectivity_is_complete("1-3,2-5,4-6", 6)


def test_connectivity_drops_bonds_outside_the_mature_peptide():
    """A bond to a cysteine that processing removes is not this peptide's connectivity."""
    precursor = "MKCAACDEFCGGC"
    # Cysteines at 1-based 3, 6, 10, 13. Take the mature peptide as residues 6 to 13, so
    # the bond 3..10 has one end in the signal region and must be dropped.
    assert frameworks.connectivity_from_pairs(
        "DISULFID 3..10; DISULFID 6..13", precursor, offset=5, length=8
    ) == "1-3"


def test_evidence_separates_a_measured_bond_from_an_inferred_one():
    assert (
        frameworks.disulfide_evidence('DISULFID 1..4; /evidence="ECO:0000269|PubMed:1"')
        == "experimental"
    )
    assert (
        frameworks.disulfide_evidence(
            'DISULFID 1..4; /evidence="ECO:0000250|UniProtKB:P1"'
        )
        == "inferred"
    )
    assert frameworks.disulfide_evidence("") == "none"


def test_partial_connectivity_is_not_a_fold_label():
    assert not frameworks.connectivity_is_complete("1-4,2-5", 6)
    assert not frameworks.connectivity_is_complete(None, 6)


# --------------------------------------------------------------------------------------
# Encodings and feature counts
# --------------------------------------------------------------------------------------


def test_no_subsampling_every_kmer_is_kept():
    """A sequence whose k-mers are all distinct contributes len - k + 1 of them."""
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    for k in (4, 6, 10):
        assert len(encodings.kmer_set(sequence, "protein20", k)) == len(sequence) - k + 1


def test_encode_refuses_a_residue_it_has_no_class_for():
    """A modified residue written into the sequence must fail, not be skipped.

    Skipping it would shorten the sequence and shift every k-mer after it, which would be
    a wrong answer rather than an error.
    """
    with pytest.raises(KeyError):
        encodings.encode("ACXDE", "cys2")


def test_collision_report_counts_peptides_not_pairs():
    report = encodings.collision_report(
        {"a": "CKGKC", "b": "CRSRC", "c": "CKGKCA"}, "cys2", 3
    )
    assert report.n_peptides_in_collision == 2
    assert report.n_distinct_encodings == 2
    assert report.largest_collision_group == 2


# --------------------------------------------------------------------------------------
# Methods: what a method refuses to answer
# --------------------------------------------------------------------------------------


def test_loop_methods_refuse_incompatible_frameworks():
    """Different cysteine counts give no score, rather than a bad one."""
    features = methods.build_features(
        ["four", "six"], ["CAACAACAAC", "CACACACACACA"], kmer_settings=[]
    )
    query, target = features
    assert query.n_cysteines != target.n_cysteines
    for method in (
        methods.loop_lengths_only,
        methods.loop_composition_only,
        methods.loop_anchored,
    ):
        assert method(query, target) == methods.NOT_COMPARABLE, method.__name__


def test_rank_leaves_out_what_a_method_could_not_compare():
    features = methods.build_features(
        ["q", "same_count", "other_count"],
        ["CAACAACAAC", "CDDCDDCDDC", "CACACACACACA"],
        kmer_settings=[],
    )
    query, *targets = features
    order, scores = methods.rank(query, targets, methods.loop_anchored)
    assert list(order) == [0], "only the peptide with a matching cysteine count is ranked"
    assert len(scores) == 1


def test_terminal_segments_are_included():
    """The stretches outside the cysteines are part of the loop-wise comparison."""
    features = methods.build_features(["q"], ["AACGGCTT"], kmer_settings=[])
    assert features[0].segments == ("AA", "GG", "TT")
    without = methods.build_features(
        ["q"], ["AACGGCTT"], kmer_settings=[], include_termini=False
    )
    assert without[0].segments == ("GG",)


# --------------------------------------------------------------------------------------
# Evaluation: unknown is not negative, and a lost candidate is a miss
# --------------------------------------------------------------------------------------


def _labels(**kwargs) -> evaluate.Labels:
    row = {
        "framework": "C-C-C-C",
        "n_cysteines": 4,
        "connectivity": "1-3,2-4",
        "connectivity_complete": True,
        "connectivity_evidence": "experimental",
        "knottin": False,
        "target_labels": "uniprot:sodium_channel",
    }
    row.update(kwargs)
    return evaluate.Labels.from_row(row)


def test_an_unlabelled_peptide_is_neither_positive_nor_negative():
    query = _labels()
    unknown = _labels(target_labels="")
    assert not evaluate.is_true_positive(query, unknown, "target")
    assert not evaluate.truth_is_known(query, unknown, "target")


def test_a_different_label_is_a_known_negative():
    query = _labels()
    other = _labels(target_labels="uniprot:potassium_channel")
    assert not evaluate.is_true_positive(query, other, "target")
    assert evaluate.truth_is_known(query, other, "target")


def test_multilabel_any_overlap_counts():
    query = _labels(target_labels="uniprot:sodium_channel;uniprot:antimicrobial")
    overlapping = _labels(target_labels="uniprot:antimicrobial;uniprot:hemolysis")
    assert evaluate.is_true_positive(query, overlapping, "target")


def test_inferred_connectivity_is_not_a_fold_label():
    inferred = _labels(connectivity_evidence="inferred")
    assert not evaluate.has_label(inferred, "fold_connectivity")
    assert evaluate.has_label(_labels(), "fold_connectivity")


def test_a_true_positive_lost_at_candidate_generation_is_an_end_to_end_miss():
    """End-to-end recall counts every true positive, including ones never ranked.

    Two true positives, one of which the candidate filter dropped. The conditional recall
    sees only the one that survived and reports 1.0; the end-to-end recall reports 0.5.
    Reporting only the conditional number would hide the loss.
    """
    truth = np.array([True, True, False, False])
    hard_negative = np.array([False, False, True, False])
    same_framework_truth = np.array([True, False, False, False])
    # The filter kept targets 0 and 2 only, and ranked 0 first.
    result = evaluate.evaluate_ranking(
        query_id="q",
        truth=truth,
        hard_negative=hard_negative,
        same_framework_truth=same_framework_truth,
        n_reference=4,
        order=np.array([0, 2]),
        seconds=0.0,
        method_name="filtered",
        endpoint="target",
        reference_mode="with_family",
        budget=10,
    )
    assert result is not None
    assert result.n_true_positives == 2
    assert result.n_true_positives_in_candidates == 1
    assert result.candidate_recall == 0.5
    assert result.recall_at_budget == 0.5
    assert result.recall_at_budget_conditional == 1.0


def test_a_query_with_no_true_positive_is_not_scored():
    result = evaluate.evaluate_ranking(
        query_id="q",
        truth=np.array([False, False]),
        hard_negative=np.array([False, False]),
        same_framework_truth=np.array([False, False]),
        n_reference=2,
        order=np.array([0, 1]),
        seconds=0.0,
        method_name="any",
        endpoint="target",
        reference_mode="with_family",
        budget=10,
    )
    assert result is None


def test_ties_count_half_in_the_hard_negative_auc():
    """A method that cannot order the same-framework block scores 0.5, not 1.0."""
    assert evaluate._auc(np.array([0]), np.array([0])) == 0.5
    assert evaluate._auc(np.array([0]), np.array([1])) == 1.0
    assert evaluate._auc(np.array([1]), np.array([0])) == 0.0


# --------------------------------------------------------------------------------------
# Manifest parsing
# --------------------------------------------------------------------------------------


def test_a_modification_note_holding_a_semicolon_survives():
    """UniProt writes "Phosphoserine; alternate", and the list separator is not a semicolon."""
    parsed = manifest.parse_modifications(
        {"Modified residue": 'MOD_RES 12; /note="Phosphoserine; alternate"'}
    )
    assert parsed == ["12:Phosphoserine; alternate"]
    joined = manifest.MODIFICATION_SEPARATOR.join(parsed)
    assert joined.split(manifest.MODIFICATION_SEPARATOR) == parsed


def test_a_derived_mature_boundary_records_the_derivation():
    spans = manifest.parse_spans('SIGNAL 1..20; /evidence="ECO:0000255"')
    span, source, inference = manifest._mature_span(spans, "M" * 20 + "CAACAACAAC")
    assert source == "after_signal_propeptide"
    assert inference, "a derived boundary must say so in words"
    assert span.start == 21


def test_an_annotated_boundary_records_no_inference():
    spans = manifest.parse_spans('PEPTIDE 3..12; /evidence="ECO:0000269|PubMed:1"')
    span, source, inference = manifest._mature_span(spans, "MKCAACAACAACK")
    assert source == "annotated_peptide"
    assert inference == ""
    assert (span.start, span.end) == (3, 12)


def test_the_ion_channel_parent_keyword_only_fills_a_gap():
    """KW-0872 is a parent keyword, so it is not a label of its own next to a specific one."""
    specific, _ = manifest._uniprot_targets({"Keyword ID": "KW-0872;KW-0738"})
    assert specific == ["uniprot:sodium_channel"]
    alone, _ = manifest._uniprot_targets({"Keyword ID": "KW-0872"})
    assert alone == ["uniprot:ion_channel_unspecified"]


# --------------------------------------------------------------------------------------
# The examples in the docstrings are tests too
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module", [encodings, frameworks, manifest, methods], ids=lambda m: m.__name__
)
def test_docstring_examples(module):
    import doctest

    from venom_retrieval import align

    results = doctest.testmod(module, verbose=False)
    assert results.failed == 0, f"{results.failed} failing examples in {module.__name__}"
    assert doctest.testmod(align, verbose=False).failed == 0
