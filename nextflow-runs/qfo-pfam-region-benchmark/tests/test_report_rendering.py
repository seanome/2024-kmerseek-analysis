"""Rendering contracts a reviewer had to catch by reading the report instead of the code.

Four regressions, all of them shipped and all of them invisible to the section-content
tests, because every one produced a well-formed plot that could not be read:

  colour range   Four heatmaps pinned min=0 max=1 while their largest cell was 0.28. Every
                 value landed in the first two stops of the ramp and the figures rendered
                 as uniform blue slabs with no ordering in them.
  category order A MultiQC line plot with `categories: True` takes its axis order from the
                 traces, first trace first and later values appended. A series that skipped
                 a bin the next series had reordered the axis, and a log2 grid came out
                 reading 1/2, 1, 2, 4, 8, 32, 1/4, 16.
  row order      MultiQC sorts bar plot samples by name unless told not to, which put an
                 alphabetical axis on the one figure whose entire content is a ranking.
  arm selection  The boundary table picked its rows with a RECOGNITION rank, so the figure
                 reporting where a call LANDS contained no 2-letter HP arm at all -- the
                 hypothesis's own alphabets, absent from the half of the report that tests
                 placement.
"""
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))
import build_multiqc_inputs as bmi  # noqa: E402


def written(tmp_path) -> dict:
    return {p.name.removesuffix("_mqc.json"): json.loads(p.read_text())
            for p in tmp_path.glob("*_mqc.json")}


# --- colour range ----------------------------------------------------------------------

def test_heat_max_ignores_blanks_and_spans_every_grid_passed():
    assert bmi.heat_max([[0.1, None], [0.28, 0.02]]) == 0.28
    assert bmi.heat_max([[0.1]], [[0.84]]) == 0.84
    assert bmi.heat_max([[None, None]]) is None


def test_heat_range_note_names_the_range_so_panels_are_not_cross_read():
    note = bmi.heat_range_note(0.28)
    assert "0 to 0.28" in note
    assert "Do not compare colours with a panel on a different range" in note
    assert bmi.heat_range_note(None) == ""


def feature_type_metrics() -> pl.DataFrame:
    """Two alphabets x two feature types, best F1 low and coverage high, as in the run."""
    rows = []
    for alpha, f1 in (("hp_thomas_dill2", 0.20), ("protein20", 0.05)):
        for stratum, point_fraction in (("TRANSMEM", 0.0), ("ACT_SITE", 1.0)):
            rows.append({
                "truth_set": "swissprot", "tool": "kmerseek",
                "variant": f"{alpha}_k19_lcFalse", "species": "mouse", "split": "all",
                "stratum_axis": "feature_type", "stratum": stratum,
                "best_f1": f1, "coverage": 0.84, "fmax": f1,
                "n_truth_instances": 400, "point_fraction": point_fraction,
                "median_feature_length": 21.0,
            })
    return pl.DataFrame(rows)


def test_feature_type_heatmaps_use_their_own_data_range_not_zero_to_one(tmp_path):
    bmi.section_ceiling_feature_type(tmp_path, feature_type_metrics())
    out = written(tmp_path)
    best = out["qfo_ceiling_feature_type"]["pconfig"]
    assert best["max"] == 0.20, "best F1 must not be drawn on a range it never reaches"
    # Coverage is a different quantity on the same grid. Sharing 0..1 with it is what made
    # the best-F1 panel unreadable, so the two are on separate ranges and both say so.
    cov = out["qfo_ceiling_feature_type_coverage"]["pconfig"]
    assert cov["max"] == 0.84
    for sec in ("qfo_ceiling_feature_type", "qfo_ceiling_feature_type_coverage"):
        text = out[sec]["description"]
        assert "own range" in text and "0.20" in text and "0.84" in text


def test_sequential_quantities_do_not_get_a_diverging_ramp(tmp_path):
    bmi.section_ceiling_feature_type(tmp_path, feature_type_metrics())
    out = written(tmp_path)
    stops = out["qfo_ceiling_feature_type"]["pconfig"]["colstops"]
    assert stops == bmi.SEQUENTIAL_COLSTOPS
    # A sequential ramp has to run one direction in lightness. The default RdYlBu-rev does
    # not: it passes through pale yellow at the midpoint, which on an Fmax grid reads as
    # "nothing here" over the middle of the range.
    assert stops[0][1] != stops[-1][1]
    assert bmi.DIVERGING_COLSTOPS[2][0] == 0.5, "the diverging ramp has a midpoint"


# --- category order --------------------------------------------------------------------

def test_ratio_labels_read_as_fractions_below_one():
    assert bmi._ratio_label(0.25) == "1/4"
    assert bmi._ratio_label(0.5) == "1/2"
    assert bmi._ratio_label(1.0) == "1"
    assert bmi._ratio_label(32.0) == "32"
    assert bmi._ratio_categories([2.0, 0.25, 1.0, 0.25]) == ["1/4", "1", "2"]


def test_every_series_spans_the_full_category_list_in_one_order():
    cats = ["1/4", "1/2", "1", "2"]
    early = bmi.on_categories({"1/2": 0.1, "1": 0.2}, cats)
    late = bmi.on_categories({"1/4": 0.05, "2": 0.3}, cats)
    # Same keys, same order, in both. Without this the second series introduces "1/4" after
    # the first has already fixed the axis, and Plotly appends it at the right-hand end.
    assert list(early) == cats
    assert list(late) == cats
    assert early["1/4"] is None, "a gap stays a gap rather than becoming a zero"


def length_metrics() -> pl.DataFrame:
    """One HP alphabet at two k, whose ratio bins deliberately do not fully overlap."""
    rows = []
    for k, lengths in ((18, [9.0, 18.0, 36.0]), (24, [24.0, 48.0, 96.0, 192.0])):
        for length in lengths:
            rows.append({
                "truth_set": "swissprot", "tool": "kmerseek",
                "variant": f"hp_thomas_dill2_k{k}_lcFalse", "species": "mouse",
                "split": "all", "stratum_axis": "feature_length_bin",
                "stratum": f"{length:.0f}", "median_feature_length": length,
                "best_f1": 0.1, "coverage": 0.5, "fmax": 0.1, "point_fraction": 0.0,
            })
    return pl.DataFrame(rows)


def test_length_panels_are_categorical_and_share_one_axis_order(tmp_path):
    bmi.section_ceiling_length(tmp_path, length_metrics(), "swissprot")
    bmi.section_ceiling_length_by_k(tmp_path, length_metrics(), "swissprot")
    out = written(tmp_path)
    for sec in ("qfo_ceiling_length", "qfo_ceiling_length_by_k"):
        cfg = out[sec]["pconfig"]
        assert cfg.get("categories") is True
        # The log axis is what drew log10 minor ticks on a log2 grid:
        # 2, 3, ... 9, 1, 2, ... 9, 10.
        assert not cfg.get("xlog")
    by_k = out["qfo_ceiling_length_by_k"]
    datasets = by_k["data"] if isinstance(by_k["data"], list) else [by_k["data"]]
    orders = {tuple(series) for ds in datasets for series in ds.values()}
    assert len(orders) == 1, f"series disagree on the axis order: {orders}"
    assert list(orders)[0] == ("1/2", "1", "2", "4", "8")


def test_the_pooled_panel_comes_first_so_seven_look_alike_panels_need_not_be_opened(
        tmp_path):
    bmi.section_ceiling_length_by_k(tmp_path, length_metrics(), "swissprot")
    labels = written(tmp_path)["qfo_ceiling_length_by_k"]["pconfig"]["data_labels"]
    assert labels[0]["name"] == "all HP alphabets"


# --- row order and arm selection --------------------------------------------------------

def boundary_metrics() -> pl.DataFrame:
    """A high-Fmax 4-letter arm, two 2-letter arms below it, and a baseline.

    The 2-letter arms place well and recognise badly, which is the whole shape of the
    problem: ranked on Fmax they never reach the table, and the table is the one that
    reports placement.
    """
    rows = []
    arms = [
        ("kmerseek", "polarity4_k17_lcFalse", 0.30, 0.72, 858),
        ("kmerseek", "hp_thomas_dill2_k23_lcFalse", 0.12, 0.66, 24),
        ("kmerseek", "hp_pbotc_1st_ed2_k22_lcFalse", 0.11, 0.64, 23),
        ("kmerseek", "protein20_k11_lcFalse", 0.05, 0.30, 20),
        ("hmmer3_phmmer", "-", 0.14, 0.44, 2567),
    ]
    for tool, variant, fmax, iou, n_tp in arms:
        rows.append({
            "truth_set": "swissprot", "tool": tool, "variant": variant,
            "species": "mouse", "split": "all", "stratum_axis": "all", "stratum": "all",
            "fmax": fmax, "family_fmax": fmax + 0.2, "median_iou_tp": iou,
            "n_tp_strict": n_tp, "residue_f1": 0.1, "coverage": 0.5,
            "interval_semantics": "alignment",
        })
    return pl.DataFrame(rows)


def test_boundary_dots_show_every_hp_arm_even_when_fmax_rank_excludes_it(tmp_path):
    # max_tools=2 is the truncation that hid them: the table keeps polarity4 and phmmer.
    bmi.section_boundary_dots(tmp_path, boundary_metrics(), "swissprot", 2)
    sec = written(tmp_path)["qfo_boundary_dots"]
    labels = " ".join(sec["data"][0])
    for alpha in ("hp_thomas_dill2", "hp_pbotc_1st_ed2"):
        assert alpha in labels, f"{alpha} was measured and must be drawn"
    assert "protein20" in labels, "the reference the HP numbers are read against"
    assert "pbotc" in sec["description"], "pinned arms are named, not silently added"


def test_boundary_dots_print_n_beside_every_arm_and_sort_by_iou(tmp_path):
    bmi.section_boundary_dots(tmp_path, boundary_metrics(), "swissprot", 5)
    sec = written(tmp_path)["qfo_boundary_dots"]
    rows = list(sec["data"][0])
    assert all("n=" in r for r in rows), "a median IoU without its denominator misleads"
    ious = [next(v for v in cell.values() if v is not None)
            for cell in sec["data"][0].values()]
    assert ious == sorted(ious, reverse=True)
    # Sorted rows are only sorted if MultiQC is told not to re-sort them by name.
    assert sec["pconfig"]["sort_samples"] is False


def test_recognition_bars_are_ordered_by_the_gap_not_alphabetically(tmp_path):
    rows = []
    for alpha, fmax, family in (("dayhoff6", 0.10, 0.31),
                                ("hp_kyte_doolittle2", 0.09, 0.39),
                                ("protein20", 0.08, 0.16)):
        rows.append({
            "truth_set": "pfam", "tool": "kmerseek", "variant": f"{alpha}_k19_lcFalse",
            "species": "mouse", "split": "all", "stratum_axis": "all", "stratum": "all",
            "fmax": fmax, "family_fmax": family, "coverage": 0.5,
            "n_family_truth": 100, "n_family_calls": 90,
        })
    bmi.section_ceiling_recognition(tmp_path, pl.DataFrame(rows), "pfam")
    sec = written(tmp_path)["qfo_ceiling_recognition"]
    assert sec["pconfig"]["sort_samples"] is False
    order = list(sec["data"][1])
    # hp_kyte_doolittle2 has the widest gap (0.30) and sorts last alphabetically, which is
    # exactly the case an alphabetical axis hides.
    assert order[0] == "hp_kyte_doolittle2"
    assert order[-1] == "protein20"
    assert "ordered by the gap" in sec["description"]


# --- log axes --------------------------------------------------------------------------

def test_throughput_axis_ends_on_decades_so_plotly_draws_no_minor_ticks():
    """The frontier's y ticks read 2, 100, 5, 10, 5, 2, 1 -- log10 minors on a rate axis.

    Plotly adds minor ticks to a log axis whose range covers less than roughly a decade and
    a half, and there is no tick control in a MultiQC pconfig. Snapping the range out to
    whole powers of ten is what removes them.
    """
    _, ymin, ymax = bmi.throughput_reference_lines([0.9, 4.2, 19.0, 124.0], 19_696)
    assert ymin == 0.1 and ymax == 1000.0
    # Still only ever widened past the data, never narrowed onto it: the scatter DROPS
    # points outside the range rather than clipping the axis.
    assert ymin < 0.9 and ymax > 124.0


def test_the_frontier_bar_is_not_set_by_a_tool_the_report_disqualifies(tmp_path):
    """A motif arm sets no reference line on the frontier.

    section_boundary says a motif arm reports the envelope of a discontinuous residue set
    and must not be ranked against the alignment arms. Letting one set the "best incumbent"
    line is that ranking by the back door. It used to get a named dotted line of its own,
    which put it back into the comparison; now it is drawn as an open marker and no line.
    """
    rows = []
    for tool, acc, sem in (("folddisco", 0.30, "motif"), ("hhblits", 0.178, "alignment"),
                           ("kmerseek", 0.138, "alignment")):
        rows.append({
            "truth_set": "swissprot", "tool": tool,
            "variant": "polarity4_k17_lcFalse" if tool == "kmerseek" else "-",
            "species": "mouse", "split": "all", "stratum_axis": "all", "stratum": "all",
            "fmax": acc, "residue_f1": acc, "coverage": 0.5, "interval_semantics": sem,
        })
    import mqc_trace as mt
    trace = pl.DataFrame(
        [{**{c: None for c in mt.TRACE_SCHEMA},
          "process": "searchHhblits", "tag": f"{tool}:mouse", "status": "COMPLETED",
          "tool": tool, "is_search": True, "realtime_s": 100.0, "cpus": 1,
          "cpu_hours": 0.03}
         for tool in ("folddisco", "hhblits", "kmerseek")],
        schema=mt.TRACE_SCHEMA,
    )
    bmi.section_frontier(tmp_path, pl.DataFrame(rows), trace, 60, "swissprot")
    sec = written(tmp_path)["qfo_frontier"]
    assert sec["plot_type"] == "scatter", sec.get("data")
    # Accuracy is on y now and cost on x, so the incumbent bar is a horizontal.
    values = {ln["label"]: ln["value"] for ln in sec["pconfig"]["y_lines"]}
    best = next(v for k, v in values.items() if k.startswith("best incumbent"))
    assert best == 0.178, "the bar is the best ALIGNMENT arm, not the motif one"
    assert not any("motif" in k for k in values), "the motif arm sets no line of its own"
    assert sec["data"]["folddisco"]["marker_symbol"].endswith("-open"), "still drawn"


def test_the_frontier_puts_cost_on_a_log_axis_rather_than_accuracy(tmp_path):
    """Every arm's accuracy landed in a band 0.08 wide, so labels printed over each other.
    Cost spans decades; accuracy does not."""
    rows = [{
        "truth_set": "swissprot", "tool": tool, "variant": "-", "species": "mouse",
        "split": "all", "stratum_axis": "all", "stratum": "all",
        "fmax": acc, "residue_f1": acc, "coverage": 0.5,
        "interval_semantics": "alignment",
    } for tool, acc in (("hhblits", 0.178), ("hmmer3_phmmer", 0.138))]
    import mqc_trace as mt
    trace = pl.DataFrame(
        [{**{c: None for c in mt.TRACE_SCHEMA},
          "process": "searchHhblits", "tag": f"{tool}:mouse", "status": "COMPLETED",
          "tool": tool, "is_search": True, "realtime_s": 100.0, "cpus": 1,
          "cpu_hours": hours}
         for tool, hours in (("hhblits", 200.0), ("hmmer3_phmmer", 0.4))],
        schema=mt.TRACE_SCHEMA,
    )
    bmi.section_frontier(tmp_path, pl.DataFrame(rows), trace, 60, "swissprot")
    pconfig = written(tmp_path)["qfo_frontier"]["pconfig"]
    assert pconfig["xlog"] is True
    assert "CPU-hours" in pconfig["xlab"]
    assert not pconfig.get("ylog")


# --- denominators ----------------------------------------------------------------------

def test_grayzone_prints_n_because_the_percentage_view_ranks_on_nothing_else(tmp_path):
    """The share and the count point opposite ways, so the panel carries both.

    In the midi run the highest true-positive share belongs to the arm with the smallest
    denominator in the report -- 50% of 3_894 calls against reseek's 0.09% of 39_669_449 --
    and on a percentage view those bars are the same width. So there is no percentage view:
    the counts are grouped bars on a log axis and every arm's tick label carries its own
    total and its own share.
    """
    rows = []
    arms = (("kmerseek", "protein20_k11_lcFalse", 1950, 1345, 605),
            ("reseek", "-", 34040, 21926984, 17708425))
    for tool, variant, tp, fp, gray in arms:
        rows.append({
            "truth_set": "swissprot", "tool": tool, "variant": variant, "species": "mouse",
            "split": "all", "stratum_axis": "all", "stratum": "all", "fmax": 0.1,
            "n_tp_calls": tp, "n_fp_calls": fp, "n_gray_calls": gray, "coverage": 0.5,
        })
    bmi.section_grayzone(tmp_path, pl.DataFrame(rows), "swissprot", 10)
    sec = written(tmp_path)["qfo_grayzone"]
    keys = list(sec["data"])
    # Ordered by total, largest first, with the denominator and the share in the label.
    assert keys[0].startswith("reseek") and "39,669,449 calls" in keys[0]
    assert "0.1% TP" in keys[0]
    assert "3,900 calls" in keys[1] and "50.0% TP" in keys[1]
    # No HTML entities in a tick label: they render in the browser and then break the PNG
    # and PDF export, which is what goes in a figure.
    assert not any("&" in k for k in keys)
    assert sec["pconfig"]["xlog"] is True
    assert sec["pconfig"]["stacking"] == "group"
    assert sec["pconfig"]["cpswitch"] is False, "no percentage view to rank on"



def test_low_complexity_collapses_to_one_panel_stating_the_negative_result(tmp_path):
    """One alphabet whose toggle is dramatic at low k and inert at the k it is run at.

    That is gbmr7's real shape in the midi run: 0.0133 -> 0.0764 at k=13, and +0.00017 at
    k=15, where its unfiltered arm is best. A per-alphabet bar can only carry one of those
    and the k grid carries both, which is why it replaced two figures rather than one.

    The headline number has to be the PAIRED one at a fixed k. Comparing gbmr7's best
    filtered combo (k=13, rescued) with its best unfiltered combo (k=15) credits the filter
    with a k change and reports +0.0016 for an alphabet the filter does nothing for.
    """
    rows = []
    for k, off, on in ((13, 0.013, 0.076), (15, 0.0744, 0.0746)):
        for lc, fmax in (("False", off), ("True", on)):
            rows.append({
                "truth_set": "swissprot", "tool": "kmerseek",
                "variant": f"gbmr7_k{k}_lc{lc}", "species": "mouse", "split": "all",
                "stratum_axis": "all", "stratum": "all", "fmax": fmax, "coverage": 0.5,
            })
    bmi.section_alphabet_matrix(tmp_path, pl.DataFrame(rows), "swissprot")
    out = written(tmp_path)
    assert "qfo_lowcomplexity_bars" not in out, "the 19-panel switcher is gone"
    lc = out["qfo_lowcomplexity"]
    assert lc["plot_type"] == "heatmap", "one alphabet x k grid, not a bar per alphabet"
    assert lc["pconfig"]["colstops"] == bmi.DIVERGING_COLSTOPS, "the sign is the reading"
    assert lc["pconfig"]["tt_decimals"] >= 4, "0.0008 must not render as 0.00"
    # The verdict is computed from the data rather than written down, so a run where the
    # toggle does matter cannot inherit a sentence saying it does not.
    assert "+0.0002" in lc["description"], "paired delta at k=15, not the k=13 jump"
    assert "+0.0016" not in lc["description"], "best-vs-best credits the filter with a k"
    # The low-k jump is not hidden either; it is named as the grid's largest cell.
    assert "0.063" in lc["description"]
