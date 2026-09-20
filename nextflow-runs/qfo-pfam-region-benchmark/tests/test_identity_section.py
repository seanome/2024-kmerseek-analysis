"""The twilight-zone axis must plot a gradient or say it has no gradient to plot.

Same failure and same guard as the omega axis in test_covariate_sections.py, on the one
figure the paper's central claim is stated on. In the midi report the identity section
drew one bar per tool with no strata, under a caption about percent identity.

Identity was computed -- the Pfam truth set has all six bins populated in the same run
(15 / 125 / 261 / 635 / 2_197 instances plus 2_140 no_homolog). It cannot attach to the
Swiss-Prot truth set, which is the default primary: attach_identity joins on (accession,
pfam_id, domain_start, domain_end), and on that truth set `pfam_id` holds a curated
feature type rather than a Pfam accession, so nothing matches and all 7_000 instances land
in `no_homolog`. One bin, drawn as if it were an axis.
"""
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))
import build_multiqc_inputs as bmi  # noqa: E402

TOOLS = [("foldseek", "-"), ("kmerseek", "polarity4_k16_lcFalse"), ("hmmer3_phmmer", "-")]


def row(tool, variant, axis, stratum, fmax):
    return {"truth_set": "swissprot", "tool": tool, "variant": variant,
            "species": "mouse", "split": "all", "stratum_axis": axis,
            "stratum": stratum, "fmax": fmax, "auprc": fmax / 2,
            "n_stratum_proteins": 500}


def metrics(bins: dict[str, float]) -> pl.DataFrame:
    rows = []
    for tool, variant in TOOLS:
        rows.append(row(tool, variant, "all", "all", 0.3))
        for b, v in bins.items():
            rows.append(row(tool, variant, "identity", b, v))
    return pl.DataFrame(rows)


def section(tmp_path, bins) -> dict:
    bmi.NOT_IN_RUN.clear()
    bmi.section_identity(tmp_path, metrics(bins), "swissprot", max_tools=10)
    return json.loads((tmp_path / "qfo_identity_mqc.json").read_text())


def test_a_single_bin_is_not_drawn_as_a_bargraph(tmp_path):
    bmi.NOT_IN_RUN.clear()
    bmi.section_identity(tmp_path, metrics({"no_homolog": 0.13}), "swissprot", max_tools=10)
    # Not a section at all (until 2026-09-20 it was one saying "Not plotted"): a line in
    # the "Not in this run" list.
    assert not (tmp_path / "qfo_identity_mqc.json").exists()
    assert any("Percent identity" in what for what, _ in bmi.NOT_IN_RUN)


def test_it_names_the_reason_the_swissprot_join_cannot_match(tmp_path):
    bmi.NOT_IN_RUN.clear()
    bmi.section_identity(tmp_path, metrics({"no_homolog": 0.13}), "swissprot", max_tools=10)
    why = next(y for w, y in bmi.NOT_IN_RUN if "Percent identity" in w)
    assert "typed (DOMAIN, TRANSMEM" in why
    assert "Pfam key has all six identity bands" in why


def test_a_real_gradient_is_still_drawn(tmp_path):
    cfg = section(tmp_path, {"0-20%": 0.05, "20-30%": 0.11, "30-40%": 0.2,
                             "no_homolog": 0.01})
    assert cfg["plot_type"] == "linegraph"
    # Only the numeric bins, at their midpoints. `no_homolog` is not a point on a percent
    # identity axis -- it is the absence of one -- so putting it in the same category list
    # as "0-20%" was the confusion this split fixes.
    for series in cfg["data"].values():
        assert [float(x) for x in series] == [10.0, 25.0, 35.0]
        assert "no_homolog" not in series
    # It keeps its own categorical panel beside the axis, rather than being dropped.
    no_homolog = json.loads((tmp_path / "qfo_identity_no_homolog_mqc.json").read_text())
    assert no_homolog["plot_type"] == "bargraph"
