"""The DisProt report builder: a failed arm is named and left out, blank cells say why,
tables carry three decimals and the pair counts, and the head carries the answer."""
import json
import subprocess
import sys
from pathlib import Path

import polars as pl

BIN = Path(__file__).resolve().parent.parent / "bin"
SHARED = Path(__file__).resolve().parents[2] / "shared"
SPECIES = [("mouse", 1561, 662), ("ecoli", 647, 347)]


def metrics(tmp: Path) -> Path:
    rows = []
    for tool, found, pr, rec in (("kmerseek_k26", 200, 0.57, 0.05), ("mmseqs2", 300, 0.72, 0.40),
                                 ("foldseek", 0, float("nan"), float("nan"))):
        for sp, n, pos in SPECIES:
            for cat, np_, npos in (("all", n, pos), ("ordered", 171, 171), ("disordered", 300, 140)):
                v_pr = float("nan") if (found == 0 or np_ == npos) else pr
                v_rec = float("nan") if (found == 0 or np_ == npos) else rec
                rows.append({"tool": tool, "species": sp, "disorder_category": cat,
                             "auc_roc": 0.8, "auc_pr": v_pr, "sens_at_99prec": 0.1,
                             "recall_at_fdr01": 0.1, "recall_at_fdr05": v_rec,
                             "n_pairs": np_, "n_positives": npos,
                             "n_found": found if cat == "all" else found // 3})
    p = tmp / "m.parquet"
    pl.DataFrame(rows).write_parquet(p)
    return p


def build(tmp: Path) -> Path:
    stats = tmp / "benchmark_stats.txt"
    stats.write_text("Total DisProt query proteins: 271\n")
    out = tmp / "in"
    subprocess.run([sys.executable, str(BIN / "make_multiqc_input.py"), str(metrics(tmp)),
                    str(out), str(stats)], check=True, capture_output=True, text=True,
                   env={"PYTHONPATH": str(SHARED), "PATH": "/usr/bin:/bin"})
    return out


def test_a_failed_arm_is_named_dashed_and_left_out(tmp_path):
    out = build(tmp_path)
    cfg = (out / "multiqc_config.yaml").read_text()
    assert 'subtitle: "kmerseek, MMseqs2: homology' in cfg, "the failed arm is not in the subtitle"
    table = (out / "auc_pr_all_mqc.tsv").read_text()
    assert "foldseek" not in table
    assert "foldseek" not in (out / "auc_pr_vs_mya_mqc.yaml").read_text()
    overview = json.loads((out / "overview_mqc.json").read_text())
    assert "a failed arm" in overview["data"]
    assert '"dashed": true' in json.dumps(overview["data"]) or "dashed" in overview["data"]
    nir = json.loads((out / "not_in_run_mqc.json").read_text())
    assert "Foldseek" in nir["data"] and "zero AlphaFold structures" in nir["data"]
    assert "not_in_run" in cfg


def test_tables_carry_three_decimals_and_the_pair_counts(tmp_path):
    out = build(tmp_path)
    table = (out / "auc_pr_all_mqc.tsv").read_text().splitlines()
    header = [l for l in table if l.startswith("Sample")][0].split("\t")
    assert header == ["Sample", "kmerseek_k26", "mmseqs2", "pairs", "positive"]
    mouse = [l for l in table if l.startswith("mouse")][0].split("\t")
    assert mouse == ["mouse", "0.570", "0.720", "1561", "662"]
    assert "format: '{:,.3f}'" in "\n".join(table)


def test_a_blank_cell_says_why(tmp_path):
    out = build(tmp_path)
    ordered = (out / "auc_pr_ordered_mqc.tsv").read_text()
    assert ("Blank cells: mouse (171 pairs), ecoli (171 pairs): every pair in this bin is "
            "positive, and with no negative pair there is no precision to compute." in ordered)
    ecoli = [l for l in ordered.splitlines() if l.startswith("ecoli")][0].split("\t")
    assert ecoli == ["ecoli", "", "", "171", "171"]


def test_the_head_carries_the_answer_and_the_words(tmp_path):
    cfg = (build(tmp_path) / "multiqc_config.yaml").read_text()
    assert cfg.count("intro_text: ") == 1
    assert "mean AUC-PR of 0.57 over 2 proteomes against 0.72 for MMseqs2" in cfg
    assert "so its ranking of pairs is worse overall" in cfg
    assert "kmerseek recovers 0.05 of the true pairs against 0.40 for MMseqs2" in cfg
    assert "Foldseek ran but reported no pair on any proteome, so it is a failed arm" in cfg
    assert "Words used on every page" in cfg and "<b>Recall at 5% FDR.</b>" in cfg
