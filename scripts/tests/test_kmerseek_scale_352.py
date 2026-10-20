"""Tests for scripts/kmerseek_scale_352.py (notebook 352). Small made-up inputs only."""

import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import kmerseek_scale_352 as ks  # noqa: E402


def test_hash_unit_is_in_unit_interval_repeatable_and_keyed():
    a = ks.hash_unit("UPI0000000001", ks.SUBSET_HASH_KEY)
    assert 0.0 <= a < 1.0
    assert a == ks.hash_unit("UPI0000000001", ks.SUBSET_HASH_KEY)
    assert a != ks.hash_unit("UPI0000000001", ks.QUERY_HASH_KEY)


def test_hash_bin_is_the_top_bits_of_hash_unit():
    for acc in ("UPI0000000001", "UPI00000ABCDE", "UPI0029FFFFFF"):
        assert ks.hash_bin(acc) == int(ks.hash_unit(acc, ks.SUBSET_HASH_KEY) * ks.N_BINS)


def test_cutoffs_reach_each_size_with_the_first_bin_that_does():
    hist = np.zeros(ks.N_BINS, dtype=np.int64)
    hist[[3, 10, 20]] = [5, 7, 9]  # cumulative 5, 12, 21
    rows = ks.cutoffs_from_histogram(hist, sizes=(5, 6, 12, 21, 22))
    assert [r["last_bin"] for r in rows] == [3, 10, 10, 20, None]
    assert [r["residues"] for r in rows] == [5, 12, 12, 21, None]


def test_slice_index_puts_a_bin_in_the_smallest_size_holding_it():
    last_bins = [3, 10, 20]
    assert [ks.slice_index(b, last_bins) for b in (0, 3, 4, 10, 11, 20, 21)] == [0, 0, 1, 1, 2, 2, None]


def make_parts(part_dir: Path, n_parts=3, per_part=200, seed=352):
    rng = np.random.default_rng(seed)
    part_dir.mkdir()
    records = {}
    for p in range(n_parts):
        with gzip.open(part_dir / f"uniparc_active_p{p + 1}.fasta.gz", "wt") as out:
            for i in range(per_part):
                acc = f"UPI{p:03d}{i:07d}"
                seq = "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=int(rng.integers(5, 400))))
                records[acc] = seq
                ks.write_record(out, acc, seq)
    return records


def test_two_passes_give_nested_subsets_with_exact_residue_counts(tmp_path, monkeypatch):
    records = make_parts(tmp_path / "parts")
    total = sum(len(s) for s in records.values())
    sizes = (total // 10, total // 3, total // 2)
    monkeypatch.setattr(ks, "SIZES_RESIDUES", sizes)
    monkeypatch.setattr(ks, "N_QUERIES", 5)
    ks.run_histogram(tmp_path / "parts", tmp_path / "hist", workers=2)
    rows = ks.cutoffs_from_histogram(np.load(tmp_path / "hist" / "histogram.npy"), sizes)
    (tmp_path / "cutoffs.json").write_text(json.dumps(rows))
    ks.run_split(tmp_path / "parts", tmp_path / "cutoffs.json", tmp_path / "slices", workers=2)
    subsets = json.loads((tmp_path / "slices" / "subsets.json").read_text())

    members = []
    for s in subsets:
        accs = set()
        for i in s["slices"]:
            for part in (tmp_path / "slices" / f"slice{i}").glob("*.fasta.gz"):
                accs |= {a for a, _ in ks.read_fasta(part)}
        members.append(accs)
        assert sum(len(records[a]) for a in accs) == s["n_residues"] >= s["size_residues"]
        # Every sequence with a lower hash bin than the subset's last one is in it.
        last = max(ks.hash_bin(a) for a in accs)
        assert accs == {a for a in records if ks.hash_bin(a) <= last}
    assert members[0] < members[1] < members[2]

    ks.pick_queries(tmp_path / "slices" / "candidates", tmp_path / "q.fasta", tmp_path / "q.tsv")
    queries = [a for a, _ in ks.read_fasta(tmp_path / "q.fasta")]
    eligible = sorted(
        (ks.hash_unit(a, ks.QUERY_HASH_KEY), a) for a in members[0] if len(records[a]) >= ks.KSIZE
    )
    assert queries == [a for _, a in eligible[:5]]


def test_loglog_fit_on_three_points_by_hand():
    # log10 points (0, 0), (1, 1), (2, 3): slope 1.5, residuals 1/6, -1/3, 1/6, SE sqrt(1/12).
    fit = ks.loglog_fit([1, 10, 100], [1, 10, 1000])
    assert fit["slope"] == pytest.approx(1.5)
    assert fit["intercept"] == pytest.approx(-1 / 6)
    assert fit["se"] == pytest.approx(math.sqrt(1 / 12))
    assert fit["df"] == 1
    assert fit["t_975"] == pytest.approx(12.7062, abs=1e-4)
    assert fit["ci_low"] == pytest.approx(1.5 - 12.706205 * math.sqrt(1 / 12), abs=1e-5)
    assert not ks.rule_fires(fit)


def test_rule_fires_only_when_the_whole_interval_is_above_one():
    x = np.array([1e10, 1e11, 1e12])
    near_quadratic = ks.loglog_fit(x, x**2 * np.array([1.0, 1.001, 1.0]))
    assert near_quadratic["ci_low"] > 1 and ks.rule_fires(near_quadratic)
    linear = ks.loglog_fit(x, x * np.array([1.0, 1.001, 1.0]))
    assert not ks.rule_fires(linear)


def test_loglog_fit_refuses_two_points():
    with pytest.raises(ValueError):
        ks.loglog_fit([1, 10], [1, 10])


def test_gate_picks_the_largest_size_that_fits():
    measured = {
        "index_build_peak_rss_gb": 30.0,
        "query_peak_rss_gb_max": 2.0,
        "index_bytes_allocated": 200e9,
        "database_fasta_bytes": 11e9,
    }
    avail = [1_000_000_000, 10_000_000_000, 20_000_000_000, 30_000_000_000, 50_000_000_000, 100_000_000_000]
    # 10 h measured: 100 billion needs 120 h, 50 needs 60 h, 30 needs 36 h; 12 h used.
    d = ks.choose_third_size(measured, job_h=10.0, used_h=12.0, available=avail)
    assert d["chosen_size_residues"] == 30_000_000_000
    # Nothing fits when the budget is nearly gone.
    d = ks.choose_third_size(measured, job_h=10.0, used_h=40.0, available=avail)
    assert d["chosen_size_residues"] is None
    # A size the source cannot fill is never chosen.
    d = ks.choose_third_size(measured, job_h=1.0, used_h=0.0, available=avail[:4])
    assert d["chosen_size_residues"] == 30_000_000_000


def test_budget_file_running_total(tmp_path):
    b = tmp_path / "budget.tsv"
    assert ks.budget_total_h(b) == 0
    ks.budget_add(b, "setup", "1", 0, 3600)
    ks.budget_add(b, "size-10000000000", "2", 0, 7200)
    assert ks.budget_total_h(b) == pytest.approx(3.0)
    assert ks.job_hours(b, "size-10000000000") == pytest.approx(2.0)


def test_timed_run_counts_output_bytes_and_lines(tmp_path):
    cmd = [sys.executable, "-c", "import sys; sys.stdout.write('a,b\\n' * 1000)"]
    r = ks.timed_run(cmd, env=None, stderr_path=tmp_path / "err", count_stdout=True)
    assert r["exit"] == 0 and r["stdout_bytes"] == 4000 and r["stdout_lines"] == 1000
    assert r["wall_s"] > 0


def test_check_cpu_only_refuses_a_visible_gpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(SystemExit):
        ks.check_cpu_only()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    ks.check_cpu_only()


def test_parse_metalink_reads_the_committed_uniparc_list():
    path = Path(__file__).resolve().parents[2] / "notebooks" / "352_data" / "uniparc_active_RELEASE.metalink"
    parts = {n: v for n, v in ks.parse_metalink(path).items() if n.endswith(".fasta.gz")}
    assert len(parts) == 200
    assert sum(size for size, _ in parts.values()) == 248_676_531_083


STUB_KMERSEEK = """#!/usr/bin/env python3
# Stands in for kmerseek in the test: `index` writes a directory, `search` prints rows.
import sys
from pathlib import Path
args = sys.argv[1:]
if args[0] == "index":
    out = Path(args[args.index("--output") + 1])
    out.mkdir()
    (out / "000001.sst").write_bytes(b"x" * 5000)
elif args[0] == "search":
    query = Path(args[args.index("--query") + 1]).read_text().split()[1]
    if query.startswith("W"):
        sys.exit(3)
    sys.stdout.write("query_name,target_name\\n" + "q,t\\n" * len(query))
"""


def size_args(tmp_path, database, queries, budget_left_h):
    """Inputs for run_size: one slice with `database`, a query file with `queries`."""
    stub = tmp_path / "kmerseek"
    stub.write_text(STUB_KMERSEEK)
    stub.chmod(0o755)
    slice_dir = tmp_path / "slices" / "slice0"
    slice_dir.mkdir(parents=True)
    with gzip.open(slice_dir / "p1.fasta.gz", "wt") as out:
        for acc, seq in database:
            ks.write_record(out, acc, seq)
    n_res = sum(len(s) for _, s in database)
    (tmp_path / "subsets.json").write_text(
        json.dumps([{"size_residues": n_res, "slices": [0], "n_sequences": len(database), "n_residues": n_res}])
    )
    with open(tmp_path / "q.fasta", "w") as out:
        for acc, seq in queries:
            ks.write_record(out, acc, seq)
    (tmp_path / "local").mkdir()
    args = type("Args", (), {})()
    args.local_dir, args.out_dir = str(tmp_path / "local"), str(tmp_path / "out")
    args.size_residues, args.subsets_json = n_res, str(tmp_path / "subsets.json")
    args.slice_root, args.queries_fasta = str(tmp_path / "slices"), str(tmp_path / "q.fasta")
    args.kmerseek, args.budget_left_h = str(stub), budget_left_h
    return args


def test_run_size_with_a_stand_in_kmerseek(tmp_path, monkeypatch):
    monkeypatch.setattr(ks, "N_QUERIES", 2)
    args = size_args(
        tmp_path,
        database=[("UPI1", "ACDEFGHIKLMNPQRSTVWYACDEF")],
        # 19 residues gives 20 output lines; the stub fails on the W query.
        queries=[("UPI1", "ACDEFGHIKLMNPQRSTVW"), ("UPI2", "WWWWWWWWWWWWWWWWWWWWW")],
        budget_left_h=1.0,
    )
    ks.run_size(args)
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["index_build_exit"] == 0
    assert summary["index_bytes_apparent"] == 5000
    assert summary["database_fasta_bytes"] == len(">UPI1\nACDEFGHIKLMNPQRSTVWYACDEF\n")
    assert summary["n_queries_ok"] == 1 and summary["n_queries_failed"] == 1
    assert not summary["complete"]
    rows = [r.split("\t") for r in (tmp_path / "out" / "queries.tsv").read_text().splitlines()]
    assert rows[1][:3] == ["0", "UPI1", "0"] and rows[1][-1] == "20"
    assert rows[2][2] == "3"


def test_run_size_stops_starting_queries_when_the_budget_is_gone(tmp_path):
    args = size_args(
        tmp_path,
        database=[("UPI1", "ACDEFGHIKLMNPQRSTVWY")],
        queries=[("UPI1", "ACDEFGHIKLMNPQRSTVW")],
        budget_left_h=0.0,
    )
    ks.run_size(args)
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["n_queries_not_run"] == 1 and summary["n_queries_ok"] == 0
