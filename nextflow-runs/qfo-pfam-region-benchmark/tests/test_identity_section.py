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
    bmi.section_identity(tmp_path, metrics(bins), "swissprot", max_tools=10)
    return json.loads((tmp_path / "qfo_identity_mqc.json").read_text())


def test_a_single_bin_is_not_drawn_as_a_bargraph(tmp_path):
    cfg = section(tmp_path, {"no_homolog": 0.13})
    assert cfg["plot_type"] == "html"
    assert "Not plotted" in cfg["data"]


def test_it_names_the_reason_the_swissprot_join_cannot_match(tmp_path):
    cfg = section(tmp_path, {"no_homolog": 0.13})
    assert "feature type" in cfg["data"]
    assert "Pfam truth set" in cfg["data"]


def test_a_real_gradient_is_still_drawn(tmp_path):
    cfg = section(tmp_path, {"0-20%": 0.05, "20-30%": 0.11, "30-40%": 0.2,
                             "no_homolog": 0.01})
    assert cfg["plot_type"] == "bargraph"
    assert list(cfg["categories"]) == ["0-20%", "20-30%", "30-40%", "no_homolog"]
