#!/usr/bin/env python3
"""Cut the midi-plus call tables down to the 1:1 ortholog pairs of notebook 356.

Runs on Sherlock, inside the kmerseek apptainer image, as a SLURM array. The kmerseek call
tables for chicken and zebrafish are 70 GB and 89 GB; the rows notebook 356 needs are the
calls between a human query and its 1:1 ortholog (notebooks/356_data/orthologs_1to1.tsv),
which fit on the laptop.

Each table is read with the pipeline's own loader, bin/evaluate_domain_calls.load_regions,
with the settings notebook 244 used: kmerseek Bonferroni p < 0.05 and region_enrichment as
the score; the aligners' TSVs parsed as the pipeline parses them. Foldseek and Reseek calls
then go through dedup_fragment_regions at IoU 0.9, as score_one does. Coordinates are left
as each tool wrote them (kmerseek 0-based end-exclusive, the others 1-based inclusive);
notebook 356 converts them.

Writes, under --out-dir:
  calls/<tool>.<arm>.<species>.parquet   query_acc, target_acc, qstart, qend, tstart, tend,
                                         score (empty file = the arm made no call on any pair)
  status/<same name>.json                rows before and after the cut, or why it is empty
  plddt.parquet                          task 0 only: per-residue pLDDT of every pair's
                                         target and query AlphaFold model in the run

A task skips an output that already exists, so resubmitting the array only redoes what is
missing. Task i of n takes every n-th input, sorted.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import polars as pl

SPECIES = ("chicken", "zebrafish")
#: Every comparison tool in the run that writes a per-region TSV, with the one variant
#: each ran in midi-plus (names as in notebook 244's reducer).
BASELINES = {
    "foldseek": "3di_aa",
    "prostt5": "3di_from_seq",
    "reseek": "verysensitive",
    "hmmer3_phmmer": "default",
    "hmmer3_jackhmmer": "n3",
    "mmseqs2_seqseq": "s7",
    "mmseqs2_iterative": "s7",
    "hhblits": "default",
}
FRAGMENT_TOOLS = {"foldseek", "reseek"}
KM_RE = re.compile(
    r"^human_vs_(?P<sp>[a-z]+)\.(?P<alpha>.+)\.k(?P<k>\d+)\.lc(?P<lc>true|false)"
    r"\.regions\.parquet$"
)
BL_RE = re.compile(r"^human_vs_(?P<sp>[a-z]+)\.(?P<tool>[a-z0-9_]+)\.tsv\.gz$")


def list_jobs(results: Path) -> list[dict]:
    jobs = []
    for p in sorted((results / "kmerseek").glob("human_vs_*.regions.parquet")):
        m = KM_RE.match(p.name)
        if m and m["sp"] in SPECIES:
            lc = "True" if m["lc"] == "true" else "False"
            jobs.append(
                dict(path=p, species=m["sp"], tool="kmerseek",
                     arm=f"{m['alpha']}_k{m['k']}_lc{lc}")
            )
    for tool, variant in BASELINES.items():
        for p in sorted((results / "regions" / tool).glob("human_vs_*.tsv.gz")):
            m = BL_RE.match(p.name)
            if m and m["sp"] in SPECIES and m["tool"] == tool:
                jobs.append(dict(path=p, species=m["sp"], tool=tool, arm=variant))
    return jobs


def read_cif_plddt(path: Path) -> list[tuple[int, float]]:
    """(1-based residue number, pLDDT) per CA atom, from an AlphaFold mmCIF's atom_site."""
    cols, out, in_loop = [], [], False
    with open(path) as fh:
        for line in fh:
            if line.startswith("_atom_site."):
                cols.append(line.strip().split(".", 1)[1])
                in_loop = True
                continue
            if in_loop and cols and line.startswith(("ATOM", "HETATM")):
                f = line.split()
                rec = dict(zip(cols, f))
                if rec.get("label_atom_id") == "CA":
                    out.append((int(rec["label_seq_id"]), float(rec["B_iso_or_equiv"])))
            elif in_loop and out and line.startswith("#"):
                break
    return out


def write_plddt(pairs: pl.DataFrame, structures: Path, out: Path) -> None:
    rows = []
    for side, acc_col, sp_col in (("target", "target_acc", "species"), ("query", "query_acc", None)):
        seen = set()
        for r in pairs.iter_rows(named=True):
            sp = r[sp_col] if sp_col else "human"
            key = (sp, r[acc_col])
            if key in seen:
                continue
            seen.add(key)
            cif = structures / sp / f"AF-{r[acc_col]}-F1.cif"
            if not cif.exists():
                rows.append(dict(side=side, species=sp, accession=r[acc_col],
                                 residue=None, plddt=None, has_model=False))
                continue
            for resnum, b in read_cif_plddt(cif):
                rows.append(dict(side=side, species=sp, accession=r[acc_col],
                                 residue=resnum, plddt=b, has_model=True))
    pl.DataFrame(rows, schema={"side": pl.String, "species": pl.String,
                               "accession": pl.String, "residue": pl.Int64,
                               "plddt": pl.Float64, "has_model": pl.Boolean}
                 ).write_parquet(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True, help="midi-plus results/")
    ap.add_argument("--structures", type=Path, required=True, help="midi-plus structures/")
    ap.add_argument("--qfo-bin", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, required=True, help="orthologs_1to1.tsv")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--n-tasks", type=int,
                    default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    ap.add_argument("--only", default=None, help="regex on the input file name, for testing")
    ap.add_argument("--list", action="store_true", help="print the job count and exit")
    args = ap.parse_args()

    sys.path.insert(0, str(args.qfo_bin))
    import evaluate_domain_calls as ev  # noqa: E402

    jobs = list_jobs(args.results)
    if args.only:
        jobs = [j for j in jobs if re.search(args.only, j["path"].name)]
    if args.list:
        print(len(jobs))
        return
    pairs = pl.read_csv(args.pairs, separator="\t")
    (args.out_dir / "calls").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "status").mkdir(parents=True, exist_ok=True)
    if args.task == 0 and not (args.out_dir / "plddt.parquet").exists():
        write_plddt(pairs, args.structures, args.out_dir / "plddt.parquet")

    # Work files (inflated TSVs) go in the task's own directory.
    work = args.out_dir / "work" / str(args.task)
    work.mkdir(parents=True, exist_ok=True)
    os.chdir(work)
    for job in jobs[args.task :: args.n_tasks]:
        name = f"{job['tool']}.{job['arm']}.{job['species']}"
        out = args.out_dir / "calls" / f"{name}.parquet"
        status = args.out_dir / "status" / f"{name}.json"
        if out.exists() and status.exists():
            continue
        keep = pairs.filter(pl.col("species") == job["species"]).select(
            "query_acc", "target_acc")
        if job["tool"] == "kmerseek":
            lf = ev.load_regions(job["path"], False, rank_by="region_enrichment",
                                 max_bonferroni_p=0.05)
        else:
            lf = ev.load_regions(job["path"], False)
        info = dict(job, path=str(job["path"]))
        if lf is None:
            df = pl.DataFrame(schema={c: pl.Int64 for c in ("qstart", "qend", "tstart", "tend")})
            info["status"] = "empty table"
        else:
            df = (lf.join(keep.lazy(), on=["query_acc", "target_acc"], how="semi")
                  .collect(engine="streaming"))
            if job["tool"] in FRAGMENT_TOOLS and df.height:
                n0 = df.height
                df = ev.dedup_fragment_regions(df, 0.9)
                info["n_dropped_fragment_dups"] = n0 - df.height
            info["status"] = "ok"
        ev.release_inflated(job["path"])
        info["n_rows_on_pairs"] = df.height
        df.write_parquet(out)
        status.write_text(json.dumps(info, indent=2) + "\n")
        print(f"{name}: {df.height} rows on pairs", flush=True)


if __name__ == "__main__":
    main()
