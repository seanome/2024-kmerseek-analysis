"""The disorder figure must be dots at measured disorder, and must not lie about Fmax.

Two things this guards, and neither is cosmetic.

The x axis. The section used to be four bars at four bin midpoints, and the widest bin's
midpoint (0.805) sits about 0.11 above the mean disorder of the proteins actually in it
(0.697 on the midi-plus query set). evaluate_domain_calls now writes `stratum_value_mean`
per cell and this section plots it, so a point is drawn where its proteins are.

The y axis. Fmax is a threshold-optimised, protein-macro-averaged F -- it exists over a
population of queries and has no per-protein value. The obvious way to make this figure
denser is one dot per query protein, and that would be a fabricated statistic: measured on
the real foldseek/mouse calls at that arm's own Fmax threshold, a protein's own F is 0 on
472 of 998 queries and 1.0 on another 110. So the y axis label and the description have to
say the dot is a cut of the query set, and this file holds them to it.
"""
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))
import build_multiqc_inputs as bmi  # noqa: E402
import evaluate_domain_calls as edc  # noqa: E402

TOOLS = [("foldseek", "3di_aa"), ("hmmer3_phmmer", "default"),
         ("kmerseek", "gbmr4_k21_lcTrue"), ("kmerseek", "gbmr4_k21_lcFalse"),
         ("kmerseek", "polarity4_k16_lcFalse"), ("kmerseek", "dayhoff6_k12_lcTrue"),
         ("kmerseek", "hp2_k24_lcFalse")]

# The real edges, so the test breaks if they are retuned without a second thought.
FINE = edc.DISORDER_FINE_EDGES
COARSE = [0.0, 0.1, 0.3, 0.6, 1.01]


def _bins(edges):
    return [f"{lo}-{hi}" for lo, hi in zip(edges[:-1], edges[1:])]


def row(tool, variant, axis, stratum, fmax, value_mean=None):
    return {"truth_set": "pfam", "tool": tool, "variant": variant, "species": "mouse",
            "split": "all", "stratum_axis": axis, "stratum": stratum, "fmax": fmax,
            "sens_first_fp_mean": fmax / 2, "auprc": fmax / 2,
            "n_stratum_proteins": 44, "n_truth_instances": 70,
            "stratum_value_mean": value_mean}


def metrics(*, axis="disorder_fine", edges=None, measured=True) -> pl.DataFrame:
    """A sweep where the structure arm decays with disorder and kmerseek does not."""
    edges = edges or FINE
    bins = _bins(edges)
    rows = []
    for i, (tool, variant) in enumerate(TOOLS):
        rows.append(row(tool, variant, "all", "all", 0.30 - 0.01 * i))
        for j, (stratum, lo, hi) in enumerate(zip(bins, edges[:-1], edges[1:])):
            # Measured mean deliberately below the midpoint, as it is in the real data.
            mean = (lo + hi) / 2 - 0.02 * (hi - lo) * 10 if measured else None
            fmax = (0.50 - 0.03 * j) if tool != "kmerseek" else 0.42
            rows.append(row(tool, variant, axis, stratum, fmax, mean))
    return pl.DataFrame(rows)


def written(tmp_path, m) -> dict:
    bmi.section_disorder_scatter(tmp_path, m, "pfam")
    return {p.name.removesuffix("_mqc.json"): json.loads(p.read_text())
            for p in tmp_path.glob("*_mqc.json")}


def test_the_fine_axis_is_drawn_as_dots_at_measured_disorder(tmp_path):
    out = written(tmp_path, metrics())
    plot = out["qfo_disorder_scatter"]
    assert plot["plot_type"] == "scatter"

    xs = sorted({p["x"] for series in plot["data"].values() for p in series})
    assert len(xs) == len(FINE) - 1, "one dot per cut, not four"
    assert len(xs) > 4, "the whole point is more than the four coarse bins"

    # Measured, not the midpoint. Every x must be the stratum_value_mean the metrics row
    # carried, which this fixture set below the midpoint on purpose.
    midpoints = [round((lo + hi) / 2, 4) for lo, hi in zip(FINE[:-1], FINE[1:])]
    assert not set(xs) & set(midpoints), "midpoints must not survive as x"
    assert plot["pconfig"]["xlab"].startswith("mean disorder of the proteins in the cut")


def test_the_y_axis_never_claims_to_be_per_protein(tmp_path):
    """Fmax has no value for a single protein. The labels must not imply it does."""
    plot = written(tmp_path, metrics())["qfo_disorder_scatter"]
    assert plot["pconfig"]["ylab"] == "Fmax over the proteins in the cut"
    desc = plot["description"]
    assert "One dot is one cut of the query set, not one protein." in desc
    assert "population statistic" in desc
    # And the point names must say which cut, not pretend to name a protein.
    a_point = next(iter(plot["data"].values()))[0]
    assert "@ disorder" in a_point["name"]


def test_the_legend_is_trimmed_to_a_readable_set_of_arms(tmp_path):
    """--max-tools puts a dozen-plus same-green kmerseek arms under one legend."""
    plot = written(tmp_path, metrics())["qfo_disorder_scatter"]
    drawn = set(plot["data"])
    kmerseek = [lb for lb in drawn if lb.startswith("kmerseek")]
    assert len(kmerseek) <= bmi.DISORDER_SCATTER_TOP_KMERSEEK, kmerseek
    # Every baseline still gets its one variant; the trim only narrows the sweep block.
    assert "foldseek" in drawn and "hmmer3_phmmer" in drawn


def test_the_two_low_complexity_arms_of_one_combo_count_as_one(tmp_path):
    """On the real midi run the top two kmerseek arms by Fmax are the same alphabet and
    ksize with the low-complexity filter on and off, 0.0005 apart and the same colour.
    Spending both legend slots on them trims nothing a reader can see."""
    drawn = set(written(tmp_path, metrics())["qfo_disorder_scatter"]["data"])
    pair = {"kmerseek gbmr4_k21_lcTrue", "kmerseek gbmr4_k21_lcFalse"}
    assert len(drawn & pair) <= 1, drawn & pair


def test_a_tree_without_the_fine_axis_falls_back_and_says_so(tmp_path):
    """Scored before disorder_fine existed: draw the coarse axis, do not claim otherwise."""
    m = metrics(axis="disorder", edges=COARSE)
    plot = written(tmp_path, m)["qfo_disorder_scatter"]
    assert plot["plot_type"] == "scatter"
    assert len({p["x"] for s in plot["data"].values() for p in s}) == 4
    assert "no <code>disorder_fine</code> rows" in plot["description"]


def test_a_tree_without_stratum_value_mean_uses_midpoints_and_says_so(tmp_path):
    m = metrics(measured=True).drop("stratum_value_mean")
    plot = written(tmp_path, m)["qfo_disorder_scatter"]
    xs = sorted({p["x"] for s in plot["data"].values() for p in s})
    assert xs == sorted(round((lo + hi) / 2, 4)
                        for lo, hi in zip(FINE[:-1], FINE[1:]))
    assert plot["pconfig"]["xlab"] == "disorder (cut midpoint)"
    assert "scored before <code>stratum_value_mean</code> existed" in plot["description"]


def test_a_partly_rescored_tree_draws_midpoints_throughout(tmp_path):
    """Half the cells measured and half not would put part of the axis at measured
    disorder and part at midpoints, under one label naming one of them."""
    m = metrics()
    bins = _bins(FINE)
    m = m.with_columns(
        pl.when(pl.col("stratum").is_in(bins[:5])).then(None)
          .otherwise(pl.col("stratum_value_mean")).alias("stratum_value_mean"))
    plot = written(tmp_path, m)["qfo_disorder_scatter"]
    xs = sorted({p["x"] for s in plot["data"].values() for p in s})
    assert xs == sorted(round((lo + hi) / 2, 4)
                        for lo, hi in zip(FINE[:-1], FINE[1:]))
    assert plot["pconfig"]["xlab"] == "disorder (cut midpoint)"


def test_no_disorder_rows_writes_no_section(tmp_path):
    m = metrics().filter(pl.col("stratum_axis") == "all")
    assert written(tmp_path, m) == {}


def test_the_table_carries_the_numbers_the_scatter_is_drawn_from(tmp_path):
    out = written(tmp_path, metrics())
    table = out["qfo_disorder_scatter_table"]
    assert table["plot_type"] == "table"
    assert len(table["data"]) == len(FINE) - 1
    first = table["data"][_bins(FINE)[0]]
    assert first["proteins"] == 44 and first["instances"] == 70
    # Every drawn arm is a column, so the figure and the table cannot disagree.
    for label in out["qfo_disorder_scatter"]["data"]:
        assert label in first


def test_the_description_reports_which_arms_fall_and_which_do_not(tmp_path):
    """The fixture decays the baselines and holds kmerseek flat; the prose must match.

    Half against half, not first cut against last. On the real data the most-ordered cut
    runs high for every arm, and a first-minus-last reading therefore called kmerseek
    falling when its middle cuts are its best -- the opposite of what the figure shows.
    """
    out = written(tmp_path, metrics())
    desc = out["qfo_disorder_scatter"]["description"]
    falling, _, rising = desc.partition("Level or higher")
    assert "<code>foldseek</code>" in falling
    assert "<code>hmmer3_phmmer</code>" in falling
    assert "kmerseek" in rising and "<code>foldseek</code>" not in rising


def test_the_trend_bullet_is_not_hostage_to_the_end_cuts(tmp_path):
    """An arm whose two end cuts are low but whose disordered half is high must not be
    reported as falling. This is the real shape: kmerseek gbmr4_k21 peaks in the middle."""
    m = metrics()
    bins = _bins(FINE)
    # Low at both ends, high across the disordered middle.
    shape = {b: (0.2 if b in (bins[0], bins[-1]) else 0.3 + 0.02 * i)
             for i, b in enumerate(bins)}
    m = m.with_columns(
        pl.when((pl.col("tool") == "kmerseek")
                & (pl.col("stratum_axis") == "disorder_fine"))
          .then(pl.col("stratum").replace_strict(shape, default=None))
          .otherwise(pl.col("fmax")).alias("fmax"))
    desc = written(tmp_path, m)["qfo_disorder_scatter"]["description"]
    _, _, rising = desc.partition("Level or higher")
    assert "kmerseek" in rising


def test_stratum_value_mean_averages_proteins_not_truth_rows():
    """A twelve-domain protein must not count twelve times toward the cell's x."""
    truth = pl.DataFrame({
        "accession": ["A"] * 9 + ["B"],
        "pfam_id": [f"PF{i:05d}" for i in range(10)],
        "disorder_fraction_plddt": [0.9] * 9 + [0.1],
    })
    got = edc.stratum_value_mean(truth, "disorder_fine")
    assert abs(got - 0.5) < 1e-9, got
    # Axes with no number behind them get null, not zero.
    assert edc.stratum_value_mean(truth, "hgnc") is None
