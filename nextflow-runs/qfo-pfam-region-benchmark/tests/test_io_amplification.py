"""read_bytes without rchar cannot say which of two opposite fixes a hot reader needs.

scoreDomainCalls has been the pipeline's largest reader in three consecutive reports --
906 GB, then 1_052 GB -- and has been diagnosed twice off that column alone. The two
candidate causes look identical in it:

  page re-faulting    polars memory-maps a parquet scan; re-collecting a plan over a mapped
                      file whose pages the task keeps evicting brings them back from disk
                      each time. Counts in read_bytes, never in rchar. Fix: materialise the
                      frame once.
  duplicate reading   the same staged file read once per truth set, or once per dedup mode.
                      Counts in BOTH. Fix: fewer passes, or fewer tasks over one input.

The ratio between the two columns separates them, so the report has to carry both and say
which side the number falls on.
"""
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))
import build_multiqc_inputs as bmi  # noqa: E402
import mqc_trace as mt  # noqa: E402


def trace(rows) -> pl.DataFrame:
    base = {"status": "COMPLETED", "tag": "t", "exit": "0", "attempt": 1, "cpus": 1,
            "realtime_s": 10.0, "duration_s": 10.0, "peak_rss_b": 1e9,
            "requested_mem_b": 2e9, "pct_cpu": 90.0, "tool": "overhead",
            "is_search": False, "cpu_hours": 0.01, "mem_used_frac": 0.5,
            "rchar_b": None, "wchar_b": None}
    return pl.DataFrame([dict(base, **r) for r in rows],
                        schema_overrides=mt.TRACE_SCHEMA)


def io_section(tmp_path, rows):
    bmi.section_resources(tmp_path, trace(rows), 964, "this run")
    p = tmp_path / "qfo_res_io_mqc.json"
    return json.loads(p.read_text()) if p.exists() else {}


GB = 1024 ** 3


def test_both_columns_are_drawn_when_the_trace_has_them(tmp_path):
    cfg = io_section(tmp_path, [
        {"process": "scoreDomainCalls", "read_b": 1052 * GB, "write_b": 58 * GB,
         "rchar_b": 20 * GB, "wchar_b": 55 * GB},
    ])
    assert set(cfg["categories"]) == {"read_gb", "rchar_gb", "write_gb", "wchar_gb"}
    row = cfg["data"]["scoreDomainCalls"]
    assert round(row["read_gb"]) == 1052 and round(row["rchar_gb"]) == 20


def test_a_large_gap_is_named_as_refaulting_not_as_reading(tmp_path):
    cfg = io_section(tmp_path, [
        {"process": "scoreDomainCalls", "read_b": 1052 * GB, "write_b": 58 * GB,
         "rchar_b": 20 * GB, "wchar_b": 55 * GB},
        {"process": "folddiscoQuery", "read_b": 521 * GB, "write_b": 79 * GB,
         "rchar_b": 500 * GB, "wchar_b": 79 * GB},
    ])
    text = cfg["description"]
    assert "scoreDomainCalls" in text, "the worst offender has to be named"
    assert "factor of 53" in text
    assert "re-faulting rather than reading" in text
    assert "materialise the frame once" in text


def test_a_ratio_near_one_points_at_the_opposite_fix(tmp_path):
    cfg = io_section(tmp_path, [
        {"process": "folddiscoQuery", "read_b": 521 * GB, "write_b": 79 * GB,
         "rchar_b": 480 * GB, "wchar_b": 79 * GB},
    ])
    text = cfg["description"]
    assert "read the file fewer times" in text
    assert "materialising frames will not help" in text


def test_a_trace_without_rchar_says_so_instead_of_guessing(tmp_path):
    cfg = io_section(tmp_path, [
        {"process": "scoreDomainCalls", "read_b": 1052 * GB, "write_b": 58 * GB},
    ])
    assert set(cfg["categories"]) == {"read_gb", "write_gb"}
    assert "rchar / wchar are missing from this trace" in cfg["description"]
    assert "trace.fields" in cfg["description"]


def test_the_trace_loader_parses_rchar_from_a_real_field_list(tmp_path):
    p = tmp_path / "trace.txt"
    p.write_text(
        "process\tstatus\trealtime\tread_bytes\twrite_bytes\trchar\twchar\n"
        "scoreDomainCalls\tCOMPLETED\t10s\t1.5 GB\t100 MB\t50 MB\t99 MB\n")
    df = mt.load_trace(p)
    assert df["rchar_b"][0] == 50 * 1024 ** 2
    assert df["wchar_b"][0] == 99 * 1024 ** 2
    # And a trace that predates the field keeps loading, with nulls rather than a crash.
    q = tmp_path / "old.txt"
    q.write_text("process\tstatus\trealtime\tread_bytes\n"
                 "scoreDomainCalls\tCOMPLETED\t10s\t1.5 GB\n")
    old = mt.load_trace(q)
    assert old["rchar_b"].null_count() == old.height
