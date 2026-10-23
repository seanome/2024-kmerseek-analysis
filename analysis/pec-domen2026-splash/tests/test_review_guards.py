"""Review regressions using tiny local files, with no downloads or cluster jobs."""

import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import openpyxl
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "download_run", ROOT / "scripts/download_run.py"
)
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


def test_stale_markers_are_replaced_after_real_validation(tmp_path, monkeypatch):
    reads = tmp_path / "reads"
    reads.mkdir()
    blob = gzip.compress(b"@a\nACGT\n+\nIIII\n")
    for mate in (1, 2):
        (reads / f"RUN_{mate}.fastq.gz").write_bytes(blob)
        (reads / f"RUN_{mate}.json").write_text(json.dumps({"records": 999}))
    (reads / "RUN.done").write_text("stale")
    report = tmp_path / "ena.tsv"
    row = {
        "run_accession": "RUN",
        "read_count": "1",
        "fastq_ftp": ";".join(f"example/RUN_{m}.fastq.gz" for m in (1, 2)),
        "fastq_md5": ";".join([hashlib.md5(blob).hexdigest()] * 2),
        "fastq_bytes": ";".join([str(len(blob))] * 2),
    }
    with report.open("w") as handle:
        w = csv.DictWriter(handle, row, delimiter="\t")
        w.writeheader()
        w.writerow(row)
    monkeypatch.setattr(
        sys, "argv", ["download_run.py", str(report), "RUN", str(reads)]
    )
    download.main()
    assert json.loads((reads / "RUN_1.json").read_text())["records"] == 1
    assert (reads / "RUN.done").read_text() == "ok\n"


@pytest.mark.parametrize(
    "text", [b"@a\nACGT\n+\nIII\n", b"@a\nACGT\n+\n", b"bad\nACGT\n+\nIIII\n"]
)
def test_invalid_fastq_is_not_counted_as_success(tmp_path, text):
    path = tmp_path / "reads.gz"
    path.write_bytes(gzip.compress(text))
    with pytest.raises(subprocess.CalledProcessError):
        download.count_records(path)


def design_fixture(tmp_path, missing_last=False):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "DEseq group annotation"
    rows = []
    for group in range(1, 7):
        if missing_last and group == 6:
            continue
        ws.append([f"Group {group}"])
        for treatment in ["Con", "Tre"]:
            number = group * 2 + (treatment == "Tre")
            alias = f"PaceMaker{treatment}_JA{number}_6996m_SYS_24d"
            ws.append([alias])
            rows.append(
                {
                    "run_accession": f"RUN{number}",
                    "sample_alias": alias,
                    "library_name": alias,
                    "experiment_alias": alias,
                    "read_count": "1",
                }
            )
    xlsx = tmp_path / "design.xlsx"
    wb.save(xlsx)
    ena = tmp_path / "ena.tsv"
    with ena.open("w") as handle:
        w = csv.DictWriter(handle, rows[0], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    out = tmp_path / "out"
    return (
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/freeze_design.py"),
                str(xlsx),
                str(ena),
                str(out),
            ],
            capture_output=True,
        ),
        out,
    )


def test_freeze_records_input_hashes(tmp_path):
    result, out = design_fixture(tmp_path)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((out / "freeze_manifest.json").read_text())
    assert manifest["runs"] == 12 and manifest["assigned_runs"] == 12
    assert len(manifest["sources"]) == 2


def test_freeze_rejects_missing_contrast_before_writing_samples(tmp_path):
    result, out = design_fixture(tmp_path, missing_last=True)
    assert result.returncode != 0
    assert not (out / "samples.tsv").exists()


def test_bh_keeps_missing_fitted_anchors_in_declared_test_family():
    import numpy as np

    spec = importlib.util.spec_from_file_location(
        "pec_analysis_utils", ROOT / "scripts/pec_analysis_utils.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    q = module.bh_over_fitted_anchors([0.01, float("nan"), 0.04])
    assert np.allclose(q[[0, 2]], [0.03, 0.06])
    assert np.isnan(q[1])


@pytest.mark.parametrize("mode", ["mismatch", "missing", "duplicate"])
def test_ena_comparison_fails_on_changed_or_missing_runs(tmp_path, mode):
    handoff = tmp_path / "handoff.tsv"
    handoff.write_text("run\tmd5_1\tmd5_2\tread_pairs\nR1\ta\tb\t1\n")
    row = "R1\tu_1.fastq.gz;u_2.fastq.gz\ta;b\t1\n"
    if mode == "mismatch":
        row = row.replace("a;b", "a;c")
    if mode == "missing":
        row = row.replace("R1", "R2")
    if mode == "duplicate":
        row += row
    report = tmp_path / "ena.tsv"
    report.write_text("run_accession\tfastq_ftp\tfastq_md5\tread_count\n" + row)
    shell = (ROOT / "scripts/fetch_ena_report.sh").read_text()
    code = shell.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    result = subprocess.run(
        [sys.executable, "-", str(handoff), str(report)],
        input=code,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0


def test_control_truth_uses_same_all_tested_anchor_bh_family(monkeypatch):
    import numpy as np

    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "control_truth", ROOT / "scripts/control_truth.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    q = module.bh([0.01, float("nan"), 0.04])
    assert np.allclose(q[[0, 2]], [0.03, 0.06])
    assert np.isnan(q[1])
