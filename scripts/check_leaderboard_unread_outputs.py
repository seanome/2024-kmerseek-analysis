#!/usr/bin/env python3
"""Print the evidence behind results/leaderboard/inventory.md's "could not read" entries.

Each check reads a file or a folder where an output should be and prints what is there,
so every number the inventory gives for an output it leaves out comes from this script.
Run on Olga's laptop (the outputs it checks were written there); Sherlock paths are
checked with `ssh sherlock test -e`.

    python scripts/check_leaderboard_unread_outputs.py > results/leaderboard/unread_outputs_check.txt
"""

import subprocess
from pathlib import Path

import polars as pl

HOME = Path("/Users/olga")
REPO_MAIN = HOME / "code/2024-kmerseek-analysis"
PFAM_BENCH = REPO_MAIN / "results/pfam_benchmark"


def exists_laptop(path: Path) -> str:
    return "present" if path.exists() else "absent"


def exists_sherlock(path: str) -> str:
    r = subprocess.run(
        ["ssh", "sherlock", f"test -e {path} && echo present || echo absent"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "ssh failed"


def section(title: str) -> None:
    print(f"\n## {title}\n")


def main() -> None:
    section("1. human_mouse_orthologs (nextflow-runs/human-mouse-gencode-orthologs*)")
    for name in ("results-human-mouse-orthologs", "results-human-mouse-orthologs-hp-v040"):
        print(f"laptop ~/data/gencode/{name}: {exists_laptop(HOME / 'data/gencode' / name)}")
    for rel in (
        "data/gencode/results-human-mouse-orthologs",
        "data/gencode/results-human-mouse-orthologs-hp-v040",
    ):
        print(f"sherlock $SCRATCH/{rel}: {exists_sherlock('$SCRATCH/' + rel)}")
    traces = sorted(
        (REPO_MAIN / "nextflow-runs/human-mouse-gencode-orthologs").glob("*.trace.txt")
    )
    print(f"traces left in the laptop clone: {len(traces)}")

    section("2. throughput-benchmark and vs-other-orthology-tools-resource-benchmark")
    p = HOME / "data/scope/results-throughput-benchmark"
    print(f"laptop ~/data/scope/results-throughput-benchmark: {exists_laptop(p)}")
    t = REPO_MAIN / (
        "nextflow-runs/vs-other-orthology-tools-resource-benchmark/results/"
        "pipeline_trace.txt"
    )
    tr = pl.read_csv(t, separator="\t", infer_schema=False)
    print(f"vs-other trace rows: {tr.height}")
    for r in tr.select("name", "status", "submit").iter_rows():
        print("  ", " | ".join(r))

    section("3. scope40-lowcomplexity-null (notebook 086)")
    p = HOME / "data/scope/results-lowcomplexity-null"
    print(f"laptop ~/data/scope/results-lowcomplexity-null: {exists_laptop(p)}")
    print(
        "sherlock $SCRATCH/data/scope/results-lowcomplexity-null: "
        + exists_sherlock("$SCRATCH/data/scope/results-lowcomplexity-null")
    )

    section("4. qfo-pfam-benchmark kmerseek pair AUC (notebook 123)")
    auc = pl.read_parquet(PFAM_BENCH / "merged/auc_by_species_ksize_score.parquet")
    print(f"auc_by_species_ksize_score.parquet rows: {auc.height}")
    flat = auc.filter(pl.col("roc_auc") == 0.5).group_by("ksize").len().sort("ksize")
    print("rows with ROC AUC exactly 0.5, by k:", dict(flat.iter_rows()))
    for sp, k in [("mouse", 24), ("mouse", 10), ("ecoli", 20)]:
        m = pl.read_parquet(PFAM_BENCH / f"merged/human_vs_{sp}.hp.k{k}.merged.parquet")
        pairs = pl.read_parquet(PFAM_BENCH / f"pairs/human_vs_{sp}_ground_truth.parquet")
        n_pos_rows = int(m["label"].sum())
        n_pos_pairs = int(pairs["label"].sum())
        uniq = m.select("human_accession", "species_accession").n_unique()
        hit = int((m["containment"].fill_null(0) > 0).sum())
        print(
            f"{sp} k{k}: merged rows {m.height:,}, distinct pairs {uniq:,}, "
            f"positive rows {n_pos_rows:,} vs positive pairs in the pair file "
            f"{n_pos_pairs:,}; rows with containment > 0: {hit:,}"
        )
    print(
        "laptop ~/data/qfo-pfam-benchmark (raw kmerseek results): "
        + exists_laptop(HOME / "data/qfo-pfam-benchmark")
    )

    section("4b. pfam-benchmark-tools hit files")
    res = HOME / "data/pfam-benchmark-tools/results"
    for tool in ("diamond", "hhblits", "hmmer3_phmmer"):
        files = sorted((res / tool).glob("*.tsv.gz"))
        empty = [f.name for f in files if f.stat().st_size <= 32]
        print(f"{tool}: {len(files)} hit files, {len(empty)} empty (<= 32 bytes)")
    metrics = sorted((res / "metrics").glob("*.metrics.parquet"))
    print(f"metrics files: {len(metrics)}")
    for f in metrics:
        m = pl.read_parquet(f)
        if m["auc_roc"].is_nan().any():
            print(f"  NaN ROC AUC: {f.name} (n_found {m['n_found'][0]})")

    section("5. Pfam subdomain DIAMOND-hard/easy split (PR 66) and E3 (notebook 340)")
    strat = sorted((PFAM_BENCH / "pairs_stratified").glob("*_hard_easy.parquet"))
    print(f"pairs_stratified hard/easy files: {len(strat)}")
    fresh = sorted((PFAM_BENCH / "diamond_fresh").glob("*.diamond.tsv.gz"))
    print(f"diamond_fresh hit files: {len(fresh)}")
    print(f"laptop ~/data/e3-detection-gap: {exists_laptop(HOME / 'data/e3-detection-gap')}")

    section("6. DisProt transfer run trace")
    t = HOME / (
        "data/leaderboard-inputs/2024-kmerseek-disprot-region-transfer/nextflow-runs/"
        "qfo-pfam-region-benchmark/run-disprot/qfo_pfam_region.2026-09-29.trace.txt"
    )
    tr = pl.read_csv(t, separator="\t", infer_schema=False)
    print(dict(tr.group_by("status").len().sort("status").iter_rows()))
    step = pl.col("process").str.split(":").list.last()
    done = tr.filter(pl.col("status") == "COMPLETED").group_by(step).len().sort("process")
    print("completed by step:", dict(done.iter_rows()))
    truth = tr.filter(step.is_in(["buildDomainTruth", "buildSwissprotTruth"]))
    print("truth-building tasks:", truth.select(step, "status").rows())


if __name__ == "__main__":
    main()
