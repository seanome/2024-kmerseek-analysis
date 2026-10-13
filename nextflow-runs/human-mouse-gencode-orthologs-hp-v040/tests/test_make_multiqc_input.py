"""The sweep report builder: the column is "kept after correction" and never "recall",
the head says when the Poisson filter is not filtering, and the first screen carries the
answer."""
import json
import subprocess
import sys
from pathlib import Path

BIN = Path(__file__).resolve().parent.parent / "bin"
SHARED = Path(__file__).resolve().parents[2] / "shared"


def sweep(tmp: Path, *, below_alpha: bool = True) -> Path:
    results = []
    for enc, ks in (("hp", [15, 16]), ("hp_thomas_dill", [15, 16, 17]), ("protein", [8])):
        for k in ks:
            complete = not (enc == "hp_thomas_dill" and k == 17)
            r = {"encoding": enc, "ksize": k}
            if complete:
                m = 425_144_632
                bh = {"rejected": 30_000, "TP": 18_000, "precision": 0.6, "recall": 0.7 if enc == "hp" else 0.5}
                r.update({"total_hits": m, "n_ortholog": 20_000, "n_non_ortholog": m - 20_000,
                          "mht": {"bonferroni": dict(bh, recall=0.2), "bh": bh, "by": bh,
                                  "two_stage_bh": bh}})
                if below_alpha:
                    r["mht"].update({"n_pairs_below_alpha": 397_343_989,
                                     "n_ortholog_below_alpha": 19_000, "alpha": 0.05})
            results.append(r)
    p = tmp / "kmer_sweep_summary.json"
    p.write_text(json.dumps({"results": results}))
    return p


def build(tmp: Path, **kw) -> Path:
    out = tmp / "in"
    stats = tmp / "ortholog_stats.txt"
    stats.write_text("Total human-mouse pairs: 21000\nNumber of human genes with mouse orthologs: 17000\n"
                     "Human genes with exactly one mouse ortholog: 16000\n")
    subprocess.run([sys.executable, str(BIN / "make_multiqc_input.py"), str(sweep(tmp, **kw)),
                    str(out), str(stats)], check=True, capture_output=True, text=True,
                   env={"PYTHONPATH": str(SHARED), "PATH": "/usr/bin:/bin"})
    return out


def test_the_word_recall_is_gone_and_the_column_is_kept(tmp_path):
    out = build(tmp_path)
    table = (out / "summary_table_mqc.tsv").read_text()
    assert "bh_kept" in table and "bh_recall" not in table
    assert "not recall of the search" in table
    assert (out / "bh_kept_vs_ksize_mqc.yaml").exists()
    assert not (out / "bh_recall_vs_ksize_mqc.yaml").exists()
    cfg = (out / "multiqc_config.yaml").read_text()
    assert "bh_kept_vs_ksize_mqc" in cfg
    overview = json.loads((out / "overview_mqc.json").read_text())["data"]
    assert "Kept after correction" in overview
    # The only "recall" left on the page says what the column is not.
    for txt in (overview, cfg, table):
        for line in txt.split("recall")[1:]:
            pass
    assert "is recall among what" not in overview


def test_the_head_says_the_filter_is_not_filtering(tmp_path):
    cfg = (build(tmp_path) / "multiqc_config.yaml").read_text()
    assert "intro_text: " in cfg
    assert ("397_343_989 of the 425_144_632 possible pairs (93%) pass p &le; 0.05" in cfg)
    assert "the Poisson filter is not filtering" in cfg
    assert "<code>hp</code> at k=16: 0.700 kept after correction" in cfg
    assert "1 of 6 combos did not complete" in cfg
    assert "Words used on every page" in cfg


def test_a_run_without_the_count_says_nothing_about_the_filter(tmp_path):
    cfg = (build(tmp_path, below_alpha=False) / "multiqc_config.yaml").read_text()
    assert "Poisson filter" not in cfg
    assert "kept after correction" in cfg
