"""Table of the read sets to search: the 64 Botryllus runs of PRJNA1219309 and the human RNA-seq runs
of the GEO series the paper names that are small enough to include.

Botryllus runs use the fastp-trimmed pairs already on Sherlock ($S/trimmed/<sample>_{1,2}.fastq.gz).
Human runs are single-end and are downloaded untrimmed from ENA into $S/sra_search/reads/.

Usage: python sra_runs_table.py <design/samples.tsv> <paper_accession_runs.tsv> <S> <out.tsv> <GSE,...>
"""

import csv
import sys

samples_tsv, acc_runs_tsv, S, out_tsv, include = sys.argv[1:6]
include = set(include.split(","))
rows = []
for r in csv.DictReader(open(samples_tsv), delimiter="\t"):
    rows.append(
        {
            "run": r["run"],
            "sample": r["sample"],
            "dataset": "PRJNA1219309",
            "organism": "Botryllus schlosseri",
            "condition": r["treatment"],
            "contrast": r["contrast"],
            "read_pairs": r["read_pairs"],
            "base_count": "",
            "files": f"{S}/trimmed/{r['sample']}_1.fastq.gz;{S}/trimmed/{r['sample']}_2.fastq.gz",
            "fastq_ftp": "",
            "fastq_md5": "",
        }
    )
base = {
    r["run_accession"]: r for r in csv.DictReader(open(acc_runs_tsv), delimiter="\t")
}
for r in base.values():
    if r["paper_accession"] == "PRJNA1219309":
        rows_by_run = {x["run"]: x for x in rows}
        rows_by_run[r["run_accession"]]["base_count"] = r["base_count"]
    elif r["paper_accession"] in include:
        rows.append(
            {
                "run": r["run_accession"],
                "sample": r["sample_accession"],
                "dataset": r["paper_accession"],
                "organism": r["scientific_name"],
                "condition": "not parsed",
                "contrast": "human_exercise_series",
                "read_pairs": r["read_count"],
                "base_count": r["base_count"],
                "files": f"{S}/sra_search/reads/{r['run_accession']}.fastq.gz",
                "fastq_ftp": r["fastq_ftp"],
                "fastq_md5": r["fastq_md5"],
            }
        )
with open(out_tsv, "w") as f:
    w = csv.DictWriter(f, list(rows[0]), delimiter="\t", lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
gbp = {}
for r in rows:
    gbp[r["dataset"]] = gbp.get(r["dataset"], 0) + int(r["base_count"] or 0) / 1e9
print(len(rows), "runs;", {k: round(v, 1) for k, v in gbp.items()})
