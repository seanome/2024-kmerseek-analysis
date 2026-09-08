"""A covariate section must plot a gradient or say it has no gradient to plot.

The regression this guards actually happened. On the chr6 (midi) query set, dN/dS covered
71 query proteins spread 17 / 31 / 23 / 0 over the four omega bins, and only the 31-protein
bin cleared MIN_STRATUM_PROTEINS in evaluate_domain_calls.strata_of. section_covariates
handed MultiQC a grouped bargraph with one category, which renders as one plain bar per
tool -- 19 bars of Fmax, no omega axis anywhere in the figure, under a caption about
selective pressure. The plot was not wrong so much as it was a different plot.
"""
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))
import build_multiqc_inputs as bmi  # noqa: E402

TOOLS = [("foldseek", "-"), ("kmerseek", "polarity4_k16_lcFalse"), ("hmmer3_phmmer", "-")]


def row(tool, variant, axis, stratum, fmax, n_proteins=500):
    return {"truth_set": "swissprot", "tool": tool, "variant": variant, "species": "mouse",
            "split": "all", "stratum_axis": axis, "stratum": stratum, "fmax": fmax,
            "auprc": fmax / 2, "n_stratum_proteins": n_proteins}


def metrics(omega_bins: dict[str, float]) -> pl.DataFrame:
    rows = []
    for tool, variant in TOOLS:
        rows.append(row(tool, variant, "all", "all", 0.30, 900))
        for stratum, fmax in omega_bins.items():
            rows.append(row(tool, variant, "omega", stratum, fmax, 31))
        for stratum, fmax in [("50-70", 0.1), ("70-90", 0.2), ("90-100", 0.3)]:
            rows.append(row(tool, variant, "plddt", stratum, fmax))
    return pl.DataFrame(rows)


def written(tmp_path, omega_bins) -> dict:
    bmi.section_covariates(tmp_path, metrics(omega_bins), "swissprot", 10)
    return {p.name.removesuffix("_mqc.json"): json.loads(p.read_text())
            for p in tmp_path.glob("*_mqc.json")}


def test_single_bin_axis_says_so_instead_of_drawing_a_bargraph(tmp_path):
    out = written(tmp_path, {"0.1-0.25": 0.2})
    omega = out["qfo_omega"]
    assert omega["plot_type"] == "html", "one bin must not become a bargraph"
    assert "0.1-0.25" in omega["data"]
    # The reason, not just the absence. A reader deciding whether to re-run needs to know
    # it is a coverage problem on this query set and not a broken axis.
    assert "chromosome 6" in omega["data"]


def test_a_real_gradient_still_plots(tmp_path):
    out = written(tmp_path, {"0.0-0.1": 0.1, "0.1-0.25": 0.2, "0.25-0.5": 0.3})
    omega = out["qfo_omega"]
    assert omega["plot_type"] == "bargraph"
    # Sorted by lower edge, not by whatever order the frame happened to hold.
    assert list(omega["categories"]) == ["0.0-0.1", "0.1-0.25", "0.25-0.5"]
    assert set(omega["data"]) == {"foldseek", "kmerseek polarity4_k16_lcFalse",
                                  "hmmer3_phmmer"}
    # The healthy axes in the same run must be untouched by the guard.
    assert out["qfo_plddt"]["plot_type"] == "bargraph"
    assert list(out["qfo_plddt"]["categories"]) == ["50-70", "70-90", "90-100"]


def test_axis_absent_from_the_run_writes_no_section(tmp_path):
    """No structures, no metapredict, no dN/dS file: nothing to report and nothing to
    explain. Only an axis that WAS scored and still cannot be read gets the note."""
    out = written(tmp_path, {})
    assert "qfo_omega" not in out
    assert "qfo_disorder" not in out
    assert "qfo_plddt" in out


def test_empty_bins_are_dropped_from_the_categories(tmp_path):
    """A bin no tool has a number in must not draw an empty slot beside the real ones,
    where a reader reads the gap as a zero."""
    m = metrics({"0.0-0.1": 0.1, "0.1-0.25": 0.2, "0.25-0.5": 0.3})
    m = m.with_columns(
        pl.when((pl.col("stratum_axis") == "omega") & (pl.col("stratum") == "0.25-0.5"))
        .then(None).otherwise(pl.col("fmax")).alias("fmax")
    )
    bmi.section_covariates(tmp_path, m, "swissprot", 10)
    omega = json.loads((tmp_path / "qfo_omega_mqc.json").read_text())
    assert list(omega["categories"]) == ["0.0-0.1", "0.1-0.25"]
    for series in omega["data"].values():
        assert "0.25-0.5" not in series


def test_omega_description_does_not_claim_species_specific_rows(tmp_path):
    """omega is a property of the human QUERY gene, joined onto every target species'
    rows alike. The old caption said rows only existed for mouse and chicken, which no
    code path ever produced."""
    blurb = bmi.COVARIATE_AXES["omega"][1]
    assert "chicken" not in blurb
    assert "QUERY" in blurb
