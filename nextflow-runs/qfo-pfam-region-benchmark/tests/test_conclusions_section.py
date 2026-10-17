"""The conclusions block has to state what the run supports and nothing further.

It sits first, right after the General Statistics table, so it is the only part of the
report most readers will finish. Three ways that goes wrong, one test each:

  overclaiming    "kmerseek wins the low-confidence band" when a baseline ties it there.
  speed           "orders of magnitude faster" stated without the comparison set, when one
                  baseline in the same run is several times faster.
  drift           numbers typed in by hand that no longer match the tables below them.

The third is why every sentence here is computed from the same frames the sections draw,
and why a point whose inputs are missing drops out instead of printing a stale one.
"""
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))
import build_multiqc_inputs as bmi  # noqa: E402

SPECIES = [("mouse", 90.0), ("chicken", 320.0), ("ecoli", 2000.0)]
# fmax, f1_reachable (default threshold), best_f1 (own best threshold), calls, IoU, P@IoU80.
# best_f1 >= f1_reachable always, which is the property the gain column depends on.
ARMS = [
    ("kmerseek", "polarity4_k17_lcFalse", 0.139, 0.123, 0.124, 4_552, 0.726, 0.239),
    ("kmerseek", "wwmj5_k11_lcFalse", 0.135, 0.136, 0.136, 3_209, 0.582, 0.035),
    ("hhblits", "default", 0.180, 0.032, 0.178, 1_213_773, 0.583, 0.144),
    ("hmmer3_phmmer", "default", 0.144, 0.072, 0.144, 27_163, 0.443, 0.101),
    ("mmseqs2_seqseq", "default", 0.110, 0.090, 0.098, 30_000, 0.400, 0.090),
]
# hhblits ties kmerseek in the middle band and beats it at the confident end; kmerseek
# peaks in the middle. Both facts are true of the real run and they pull opposite ways.
PLDDT = {
    "kmerseek": {"0-50": 0.10, "50-70": 0.15, "70-90": 0.173, "90-100": 0.12},
    "hhblits": {"0-50": 0.12, "50-70": 0.16, "70-90": 0.174, "90-100": 0.21},
    "hmmer3_phmmer": {"0-50": 0.05, "50-70": 0.08, "70-90": 0.10, "90-100": 0.15},
    "mmseqs2_seqseq": {"0-50": 0.04, "50-70": 0.06, "70-90": 0.08, "90-100": 0.12},
}
# Fmax by disorder bin (the four-bin axis evaluate_domain_calls writes), keyed by kmerseek
# variant and by baseline tool. Both kmerseek arms fall from the least to the most
# disordered bin (to x0.50 and x0.60 of where they start) while every baseline holds
# (x1.00 to x1.07), so the disorder caveat has to say kmerseek falls and the baselines
# stay flat. The caveat was a sentence typed in by hand until 2026-09-19; now it is read
# off these strata.
DISORDER = {
    "polarity4_k17_lcFalse": {"0.0-0.1": 0.20, "0.1-0.3": 0.16, "0.3-0.6": 0.12,
                              "0.6-1.01": 0.10},
    "wwmj5_k11_lcFalse": {"0.0-0.1": 0.15, "0.1-0.3": 0.13, "0.3-0.6": 0.11,
                          "0.6-1.01": 0.09},
    "hhblits": {"0.0-0.1": 0.18, "0.1-0.3": 0.18, "0.3-0.6": 0.17, "0.6-1.01": 0.19},
    "hmmer3_phmmer": {"0.0-0.1": 0.14, "0.1-0.3": 0.14, "0.3-0.6": 0.13, "0.6-1.01": 0.15},
    "mmseqs2_seqseq": {"0.0-0.1": 0.11, "0.1-0.3": 0.11, "0.3-0.6": 0.10, "0.6-1.01": 0.11},
}
# Whether the right family was named anywhere in the protein, for the two kmerseek arms.
# The 4-letter alphabet names the family far more often than it places it (0.30 vs 0.139),
# the 5-letter one less so (0.20 vs 0.135): recognition spans 1.5x across alphabets,
# placement about 1.0x, so reduction buys recognition and not placement.
FAMILY_FMAX = {"polarity4_k17_lcFalse": 0.30, "wwmj5_k11_lcFalse": 0.20}


def metrics() -> pl.DataFrame:
    rows = []
    for tool, variant, fmax, f1, best_f1, calls, iou, p80 in ARMS:
        for species, mya in SPECIES:
            base = dict(truth_set="pfam", tool=tool, variant=variant, species=species,
                        species_mya=mya, split="heldout", interval_semantics="alignment",
                        fmax=fmax if species != "ecoli" else fmax * 0.9,
                        f1_reachable=f1, best_f1=best_f1, n_calls=calls,
                        n_tp_calls=int(calls * 0.1), precision=p80,
                        residue_precision=0.1, residue_recall=0.041, residue_f1=0.06,
                        median_iou_tp=iou, precision_iou80=p80,
                        n_reachable=1_161 if species == "ecoli" else 7_063,
                        n_instances=7_185,
                        family_fmax=FAMILY_FMAX.get(variant))
            rows.append(dict(base, stratum_axis="all", stratum="all"))
            for band, v in PLDDT.get(tool, {}).items():
                rows.append(dict(base, stratum_axis="plddt", stratum=band, fmax=v))
            for band, v in DISORDER.get(variant if tool == "kmerseek" else tool, {}).items():
                rows.append(dict(base, stratum_axis="disorder", stratum=band, fmax=v))
    return pl.DataFrame(rows)


def curves() -> pl.DataFrame:
    rows = []
    # Precision starts above the floor and decays; kmerseek holds it further.
    for tool, variant, reach in (("kmerseek", "polarity4_k17_lcFalse", 0.087),
                                 ("kmerseek", "wwmj5_k11_lcFalse", 0.080),
                                 ("hhblits", "default", 0.049),
                                 ("hmmer3_phmmer", "default", 0.048),
                                 ("mmseqs2_seqseq", "default", 0.051)):
        for species, _ in SPECIES:
            for i, r in enumerate([0.0, reach / 2, reach, reach * 2]):
                rows.append({"truth_set": "pfam", "split": "heldout", "tool": tool,
                             "variant": variant, "species": species,
                             "score_threshold": float(10 - i),
                             "recall_reachable": r,
                             "precision": 0.9 if i < 3 else 0.2})
    return pl.DataFrame(rows)


def built(tmp_path) -> dict:
    bmi.section_conclusions(tmp_path, metrics(), curves(), pl.DataFrame(), 964, "pfam")
    p = tmp_path / "qfo_conclusions_mqc.json"
    return json.loads(p.read_text()) if p.exists() else {}


def test_it_is_written_and_says_which_truth_set(tmp_path):
    cfg = built(tmp_path)
    assert cfg, "the conclusions section was not written"
    assert cfg["id"] == "qfo_conclusions"
    assert "Pfam answer key" in cfg["description"]


def test_the_recall_at_precision_headline_names_the_arm_and_the_baseline(tmp_path):
    body = built(tmp_path)["data"]
    assert "Recall at a precision floor" in body
    assert "kmerseek polarity4_k17_lcFalse" in body
    assert "0.087" in body and "0.051" in body
    # The comparison has to be against the best BASELINE, not against the second kmerseek
    # arm, or the lead is a comparison of the method with itself.
    assert "mmseqs2_seqseq" in body or "hhblits" in body


def test_the_plddt_point_claims_a_peak_and_not_a_win(tmp_path):
    body = built(tmp_path)["data"]
    assert "peak" in body
    assert "tie, not a win" in body, "a 0.001 gap must not be reported as beating hhblits"


def test_the_divergence_point_uses_the_most_distant_species(tmp_path):
    body = built(tmp_path)["data"]
    assert "ecoli" in body and "2,000 Mya" in body


def test_the_gain_is_never_below_one(tmp_path):
    """It was, in the midi-plus report: fmax / f1 divides CAFA's protein-centric average
    by a call-level F1 with a different recall denominator, and one kmerseek arm printed
    0.99 -- the best threshold reading worse than the default, which cannot happen."""
    body = built(tmp_path)["data"]
    assert "threshold comparison is not the fight" in body
    assert "already at their own optimum" in body
    assert "-4%" not in body and "-0%" not in body


def test_the_negative_results_are_in_the_same_list(tmp_path):
    body = built(tmp_path)["data"]
    assert "does not show" in body
    assert "Disorder is not where kmerseek gains ground" in body
    assert "carries a sampling error" in body
    assert "Reduction buys recognition much more than placement" in body


def test_the_disorder_caveat_is_read_off_the_strata(tmp_path):
    # kmerseek's two arms end at x0.50 and x0.60 of their least-disordered value while
    # the baselines sit between x1.00 and x1.07; the words have to match those numbers.
    body = built(tmp_path)["data"]
    assert "Fmax falls for every kmerseek arm (x0.50 to x0.60" in body, body
    assert "stays flat for every baseline (x1.00 to x1.07)" in body, body
    assert "<code>0.0-0.1</code> to <code>0.6-1.01</code>" in body
    # The most-disordered bin names the best arm on each side with its number.
    assert "<code>kmerseek polarity4_k17_lcFalse</code>, is at 0.100 against 0.190 for " \
           "<code>hhblits</code>" in body, body


def test_the_disorder_caveat_drops_out_without_the_strata(tmp_path):
    # A run without the disorder axis must not print a sentence about it. This was the
    # failure mode before 2026-09-19: the sentence was typed in and printed regardless.
    bare = metrics().filter(pl.col("stratum_axis") != "disorder")
    assert bmi.conclusion_disorder(bare, "pfam") == ""


def test_the_reachability_census_is_stated_with_its_spread(tmp_path):
    body = built(tmp_path)["data"]
    assert "unreachable" in body
    # The floor species has to be named: a species far below its neighbours is the
    # signature of a target-annotation failure, not of a search result.
    assert "ecoli" in body and "16.2%" in body


def test_a_run_with_no_curves_still_writes_the_rest(tmp_path):
    bmi.section_conclusions(tmp_path, metrics(), pl.DataFrame(), pl.DataFrame(),
                            964, "pfam")
    body = json.loads((tmp_path / "qfo_conclusions_mqc.json").read_text())["data"]
    assert "Deliverable recall" not in body
    assert "largest divergence" in body


def test_the_speed_sentence_names_what_it_beats_and_what_beats_it(monkeypatch):
    """mmseqs2 is 6.5x faster than kmerseek in the real run. "Orders of magnitude faster"
    with no comparison set named is the claim a reviewer checks first and the one that
    would not survive."""
    # hhblits and phmmer as in the real run: kmerseek is ~1000x and ~2x faster
    # respectively, so one gap licenses "orders of magnitude" and the other does not.
    rates = {"kmerseek": 25.5, "mmseqs2_seqseq": 166.0, "hhblits": 0.023,
             "hmmer3_phmmer": 12.0}
    monkeypatch.setattr(
        bmi, "attach_throughput",
        lambda sel, trace, n: sel.with_columns(
            pl.col("tool").replace_strict(rates, default=None).alias("queries_per_s")))
    txt = bmi.conclusion_speed(metrics(), pl.DataFrame(), 964, "pfam")
    assert "hhblits" in txt, "the tools it does beat have to be named"
    assert "NOT the fastest arm" in txt
    assert "mmseqs2_seqseq" in txt and "6.5x" in txt
    assert "with the comparison set named" in txt
    # The 2x gap to phmmer must not be dressed up as orders of magnitude; only the
    # 1000x gap to hhblits licenses that phrase.
    lic = txt[txt.index("Against "):txt.index("the gap is two orders")]
    assert "hhblits" in lic and "hmmer3_phmmer" not in lic
    # And the slow arm's rate must not print as 0.0 q/s in the sentence comparing it.
    assert "0.023 q/s" in txt and "0.0 q/s" not in txt


def test_a_run_with_no_timings_drops_the_speed_point_rather_than_guessing(tmp_path):
    body = built(tmp_path)["data"]
    assert "query proteins/s" not in body


def test_nothing_is_written_when_the_truth_set_is_absent(tmp_path):
    bmi.section_conclusions(tmp_path, metrics(), curves(), pl.DataFrame(), 964, "mcsa")
    assert not (tmp_path / "qfo_conclusions_mqc.json").exists()
