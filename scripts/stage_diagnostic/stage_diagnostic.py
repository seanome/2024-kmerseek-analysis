#!/usr/bin/env python3
"""kmerseek stage diagnostic: where does a true hit get lost?

Subcommands, run in this order (paths are relative to the repository root):

  freeze    build the three datasets' panels, labels and splits from the config
  align     stages A and B (seedless alignment) on the laptop
  job       write the Sherlock job for stages C, D, E and the fixtures
  collect   read the job's kmerseek outputs back into one table per dataset
  evaluate  every metric, the fixture checks, the self-hit check, the decision rules
  plot      one figure per endpoint
  report    the decision report (markdown) from evaluate's tables

Example:
  python scripts/stage_diagnostic/stage_diagnostic.py freeze
  python scripts/stage_diagnostic/stage_diagnostic.py align
  python scripts/stage_diagnostic/stage_diagnostic.py job --job-dir /tmp/nb350_job
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
DATASETS = ["ice_binding", "toxin_scaffold", "toxin_target"]


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(open(path))
    cfg["inputs"] = {
        k: (
            str(Path(v).expanduser())
            if k.endswith(("dat", "truth", "swissprot", "peptides"))
            else v
        )
        for k, v in cfg["inputs"].items()
    }
    for k in ("ice_truth", "random_swissprot", "cys_peptides"):
        cfg["inputs"][k] = str(ROOT / cfg["inputs"][k])
    return cfg


def data_dir(cfg) -> Path:
    return ROOT / "data" / "350_stage_diagnostic"


def results_dir(cfg) -> Path:
    return ROOT / "tables" / "350_stage_diagnostic" / f"v{cfg['version']}"


def cmd_freeze(cfg, args):
    import hashlib

    import panels
    from swissprot import load_slim

    dat = Path(cfg["inputs"]["swissprot_dat"])
    md5 = hashlib.md5(open(dat, "rb").read()).hexdigest()
    if md5 != cfg["inputs"]["swissprot_dat_md5"]:
        sys.exit(f"{dat} md5 {md5} is not the config's {cfg['inputs']['swissprot_dat_md5']}")
    sp = load_slim(dat)
    sp_idx = {r["accession"]: r for r in sp.iter_rows(named=True)}
    random_ids = [x.strip() for x in open(cfg["inputs"]["random_swissprot"]) if x.strip()]
    d = data_dir(cfg)
    panels.freeze_ice(cfg, sp, sp_idx, random_ids, d / "ice_binding")
    cys = panels.cys_rich_nontoxins(cfg, sp, cfg["panels"]["toxin_scaffold"]["n_cys_rich_nontoxin"])
    panels.freeze_scaffold(cfg, sp, sp_idx, random_ids, cys, d / "toxin_scaffold")
    panels.freeze_target(cfg, sp, sp_idx, random_ids, cys, d / "toxin_target")
    write_manifest(d)


def write_manifest(d: Path):
    import hashlib

    lines = ["file\tbytes\tsha1"]
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.tsv" and "inputs" not in p.parts:
            lines.append(
                f"{p.relative_to(d)}\t{p.stat().st_size}\t{hashlib.sha1(p.read_bytes()).hexdigest()}"
            )
    (d / "MANIFEST.tsv").write_text("\n".join(lines) + "\n")


def check_manifest(d: Path):
    import hashlib

    for line in (d / "MANIFEST.tsv").read_text().splitlines()[1:]:
        f, _, h = line.split("\t")
        if hashlib.sha1((d / f).read_bytes()).hexdigest() != h:
            sys.exit(
                f"{d / f} changed since freeze: the panels are frozen, re-freeze as a new version"
            )


def cmd_align(cfg, args):
    import align
    import labels

    check_manifest(data_dir(cfg))
    for ds in args.datasets:
        alphabets = sorted({s["alphabet"] for s in cfg["kmerseek"]["settings"][ds]})
        pos = labels.positive_sets(data_dir(cfg) / ds, ds)
        wall = align.run(
            data_dir(cfg) / ds, results_dir(cfg) / ds, ["full"] + alphabets, pos, args.workers
        )
        print(f"{ds}: A and B ({', '.join(alphabets)}) in {wall:.0f} s")


def cmd_job(cfg, args):
    import shutil

    import kmerseek_job

    check_manifest(data_dir(cfg))
    job = Path(args.job_dir)
    for ds in DATASETS:
        (job / "data" / ds).mkdir(parents=True, exist_ok=True)
        for f in ("queries.fasta", "references.fasta"):
            shutil.copyfile(data_dir(cfg) / ds / f, job / "data" / ds / f)
    kmerseek_job.write_job(cfg, data_dir(cfg), job, DATASETS)
    print(f"wrote {job}/run.sh and submit.sbatch")


def cmd_collect(cfg, args):
    import collect

    collect.run(Path(args.job_dir), results_dir(cfg), cfg)


def cmd_evaluate(cfg, args):
    import evaluate

    evaluate.run(cfg, data_dir(cfg), results_dir(cfg))


def cmd_plot(cfg, args):
    import plots

    plots.run(cfg, results_dir(cfg), ROOT / "figures" / "350")


def cmd_report(cfg, args):
    import report

    report.run(cfg, results_dir(cfg), ROOT / "reports" / "350_stage_diagnostic.md")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, default=HERE / "config" / "stage_diagnostic_v1.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("freeze")
    p = sub.add_parser("align")
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--workers", type=int, default=None)
    for name in ("job", "collect"):
        p = sub.add_parser(name)
        p.add_argument("--job-dir", required=True)
    sub.add_parser("evaluate")
    sub.add_parser("plot")
    sub.add_parser("report")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.cmd == "align" and args.workers is None:
        import align

        args.workers = align.default_workers()
    globals()[f"cmd_{args.cmd}"](cfg, args)


if __name__ == "__main__":
    main()
