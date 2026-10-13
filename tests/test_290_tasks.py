"""Tests for scripts/290_shuffle_control/tasks.py: penalties and the command check."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tasks290", ROOT / "scripts/290_shuffle_control/tasks.py")
tasks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tasks)

# kappa values of assets/kappa_by_alphabet.tsv on the dark-set branch (PR 57)
KAPPA = {"hp_lehninger2": 0.3944, "hp_lehninger_c_nonpolar2": 0.4014, "hp_pbotc_1st_ed2": 0.4389}

# The kmerseek part of the run's .command.sh for worm chunk_0000, hp_lehninger2 k17
# (work/88/a53049c04403eb430b242872cc975b), as written by main.nf.
RUN_SCRIPT = """#!/bin/bash -ue
set -euo pipefail
set +e
kmerseek search \\
    --alphabet hp_lehninger2 --ksize 17 \\
    --query  chunk_0000.fasta --target minus_Chromadorea.hp_lehninger2.k17.lcfalse.kmerseek.rocksdb  --extend-mismatch-penalty 1.51 --extend-xdrop 6.04 --ka-queries 0 --chain-max-gap 30 --chain-max-shift 10 \\
    --threshold        0.0 \\
    --min-shared-kmers 2 \\
    --max-query-pvalue 0.05 \\
    --min-region-score 1.3 \\
    2> chunk_0000.hp_lehninger2.k17.s1.lcfalse.extend-c1.51.log \\
  | awk -F, -v max=10000 'NR==1 { print; next } ($c+0) <= max' \\
  | zstd -T2 -o x.regions.csv.zst
"""


def test_penalties_match_main_nf():
    assert tasks.optimal_penalty("hp_lehninger2", KAPPA) == "1.51"
    assert tasks.optimal_penalty("hp_lehninger_c_nonpolar2", KAPPA) == "1.52"
    assert tasks.optimal_penalty("hp_pbotc_1st_ed2", KAPPA) == "1.59"
    # X-drop is 4 x the two-decimal penalty, not the table's xdrop_4c (6.03, 6.35)
    assert tasks.xdrop_for("1.51") == "6.04"
    assert tasks.xdrop_for("1.59") == "6.36"


def test_penalty_string_is_java_half_up():
    assert tasks.penalty_string(2.0) == "2"
    assert tasks.penalty_string(1.505) == "1.51"  # Python's round() gives 1.5 here
    assert tasks.penalty_string(0.14) == "0.14"


def test_command_matches_run_and_catches_a_change():
    ran = tasks.command_from_script(RUN_SCRIPT)
    ours = tasks.search_command("hp_lehninger2", 17, "1.51", "chunk_0000.fasta",
                                "minus_Chromadorea.hp_lehninger2.k17.lcfalse.kmerseek.rocksdb")
    assert ran == ours
    assert ran != tasks.search_command("hp_lehninger2", 17, "1.52", "chunk_0000.fasta",
                                       "minus_Chromadorea.hp_lehninger2.k17.lcfalse.kmerseek.rocksdb")
    assert "max=10000" in tasks.awk_filter_from_script(RUN_SCRIPT)
