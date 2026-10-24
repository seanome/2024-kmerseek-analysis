"""scripts/build_no_search_baseline_table.py: not-applicable cells hold no number, the
checks compare against every kmerseek row, and cost comes from the trace."""

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_no_search_baseline_table as b  # noqa: E402


def metrics():
    base = {"split": "all", "variant": None, "tool": None, "truth_set": None,
            "stratum_axis": None, "stratum": None, "n_truth_instances": 0, "fmax": 0.0}
    rows = [
        ("kyte_doolittle", "w19_t1.6_transmem", "swissprot", "feature_type", "TRANSMEM", 1124, 0.778),
        ("kyte_doolittle", "w19_t1.6_transmem", "swissprot", "feature_type", "DOMAIN", 1107, 0.0),
        ("kyte_doolittle", "w19_t1.6_intramem", "swissprot", "feature_type", "INTRAMEM", 9, 0.333),
        ("kyte_doolittle", "w19_t1.6_transmem", "swissprot", "all", "all", 7185, 0.196),
        ("pfam_family_frequency", "length_bin", "pfam", "all", "all", 2435, 0.035),
    ]
    return pl.DataFrame([{**base, "tool": t, "variant": v, "truth_set": ts,
                          "stratum_axis": ax, "stratum": s, "n_truth_instances": n,
                          "fmax": f} for t, v, ts, ax, s, n, f in rows])


def results():
    rows = []
    for setting, v in [("a", 0.5), ("b", 0.2)]:
        rows.append({"experiment": b.EXPERIMENT, "tool": "kmerseek", "setting": setting,
                     "target_db": "QfO mouse", "metric": "fmax",
                     "stratum_axis": "feature_type", "stratum_bin": "TRANSMEM",
                     "value": v, "n_instances": 1124,
                     "truth_source": "Swiss-Prot features", "truth_version": None})
        rows.append({**rows[-1], "stratum_bin": "INTRAMEM", "value": v, "n_instances": 9})
        rows.append({**rows[-1], "stratum_axis": "all", "stratum_bin": "all",
                     "truth_source": "Pfam-A domain instances", "truth_version": "Pfam 38.2",
                     "value": v / 10, "n_instances": 2435})
    return pl.DataFrame(rows)


def test_not_applicable_cells_have_no_value():
    m = b.nsb.mark_applicable(metrics(), b.nsb.read_applies(b.APPLIES))
    rows = b.fill_not_applicable(b.long_rows(m, "x", "abc"), results())
    fmax = rows.filter(pl.col("metric") == "fmax")
    na = fmax.filter(~pl.col("applicable"))
    assert na["value"].null_count() == na.height > 0
    ok = fmax.filter(pl.col("applicable"))
    assert sorted(zip(ok["tool"], ok["stratum_bin"], ok["value"])) == [
        ("kyte_doolittle", "INTRAMEM", 0.333), ("kyte_doolittle", "TRANSMEM", 0.778),
        ("pfam_family_frequency", "all", 0.035)]
    # KD on Pfam gets a not-applicable row, filled from the search tools' cells.
    assert na.filter((pl.col("tool") == "kyte_doolittle")
                     & (pl.col("truth_source") == "Pfam-A domain instances")).height == 2
    assert list(rows.columns) == list(b.SCHEMA)


def test_check_kd_counts_every_kmerseek_row():
    m = b.nsb.mark_applicable(metrics(), b.nsb.read_applies(b.APPLIES))
    c = b.check_kd(m, results())
    tm = c.filter(pl.col("feature_type") == "TRANSMEM").row(0, named=True)
    assert (tm["kmerseek_rows"], tm["kmerseek_rows_at_or_above"]) == (2, 0)
    im = c.filter(pl.col("feature_type") == "INTRAMEM").row(0, named=True)
    assert im["grey_below_30"] and im["kmerseek_rows_at_or_above"] == 1


def test_check_frequency_floor():
    m = b.nsb.mark_applicable(metrics(), b.nsb.read_applies(b.APPLIES))
    c = b.check_frequency(m, results()).row(0, named=True)
    assert (c["tool"], c["rows_below_floor"], c["frequency_fmax"]) == ("kmerseek", 1, 0.035)


def test_compute_rows_from_trace(tmp_path):
    header = "task_id\tnative_id\tprocess\ttag\tstatus\tcpus\trealtime\tpeak_rss\tcomplete\n"
    lines = [
        "1\t1\tnoSearchBaselines:kdScan\tkyte_doolittle\tCOMPLETED\t1\t2.5s\t100 MB\t1\n",
        "2\t1\tnoSearchBaselines:iupredLong\tiupred long\tCOMPLETED\t1\t1m 30s\t50 MB\t2\n",
        "3\t1\tnoSearchBaselines:baselineCalls\tiupred\tCOMPLETED\t1\t500ms\t1 GB\t3\n",
        "4\t1\tnoSearchBaselines:scoreBaselines\tpfam (1 baselines)\tCOMPLETED\t1\t9s\t1 GB\t4\n",
        "5\t1\tnoSearchBaselines:kdScan\tkyte_doolittle\tFAILED\t1\t9s\t-\t0\n",
    ]
    t = tmp_path / "trace.txt"
    t.write_text(header + "".join(lines))
    c = b.compute_rows(t, 998)
    iu = c.filter(pl.col("tool") == "iupred").row(0, named=True)
    assert iu["wall_s"] == 90.5 and iu["n_tasks"] == 2 and iu["peak_ram_gb"] == 1.0
    kd = c.filter(pl.col("tool") == "kyte_doolittle").row(0, named=True)
    assert abs(kd["seconds_per_1000_queries"] - 2.5 / 998 * 1000) < 1e-9
    assert set(c["tool"]) == {"iupred", "kyte_doolittle"}   # scoring is not a tool cost
