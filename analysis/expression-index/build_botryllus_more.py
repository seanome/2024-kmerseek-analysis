"""Build runs_botryllus_more.tsv: the Botryllus bulk RNA-seq runs beyond the paper's 64.

Input is Olga's list of every Botryllus run in SRA (sources/botryllus_runs.tsv, 785 runs).
Kept: library_source TRANSCRIPTOMIC and library_strategy RNA-Seq (516 runs); then the 64
runs of PRJNA1219309, already in runs_to_index.tsv, are removed. RAD-seq, WGS, Hi-C,
bisulfite and the 10 single-cell runs are not expression data and are left out.
File links and sizes come from ENA's filereport for each study. Sorted smallest first,
with the columns sketch_run.sh reads.

    python build_botryllus_more.py
"""

import io
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
FIELDS = "run_accession,study_accession,fastq_bytes,fastq_ftp,read_count,library_layout,instrument_model,sample_title,first_public"


def ena(study):
    url = (f"https://www.ebi.ac.uk/ena/portal/api/filereport?accession={study}"
           f"&result=read_run&fields={FIELDS}&format=tsv")
    with urllib.request.urlopen(url, timeout=120) as r:
        return pd.read_csv(io.StringIO(r.read().decode()), sep="\t")


def main():
    src = pd.read_csv(HERE / "sources/botryllus_runs.tsv", sep="\t")
    rna = src[(src.library_source == "TRANSCRIPTOMIC") & (src.library_strategy == "RNA-Seq")]
    paper = set(pd.read_csv(HERE / "runs_to_index.tsv", sep="\t").accession)
    rna = rna[~rna.run_accession.isin(paper)]
    e = pd.concat([ena(s) for s in sorted(rna.study_accession.unique())])
    e = e[e.run_accession.isin(rna.run_accession)]
    if len(e) != len(rna):
        raise SystemExit(f"ENA returned {len(e)} of {len(rna)} runs")
    files = e.fastq_ftp.fillna("").str.split(";")
    e["size_gb"] = e.fastq_bytes.fillna("").astype(str).map(
        lambda s: round(sum(int(x) for x in s.split(";") if x.isdigit()) / 1e9, 4))
    e["ena_fastq_http_1"] = [next((f"http://{f}" for f in fs if f.endswith("_1.fastq.gz")), "") for fs in files]
    e["ena_fastq_http_2"] = [next((f"http://{f}" for f in fs if f.endswith("_2.fastq.gz")), "") for fs in files]
    e["ena_fastq_http"] = [next((f"http://{f}" for f in fs if f and not f.endswith(("_1.fastq.gz", "_2.fastq.gz"))), "") for fs in files]
    e = e.rename(columns={"run_accession": "accession"}).merge(
        rna[["run_accession", "study_title"]].rename(columns={"run_accession": "accession"}), on="accession")
    e.insert(1, "organism", "Botryllus schlosseri")
    e["built_from"] = "reads"
    e.sort_values(["size_gb", "accession"]).to_csv(HERE / "runs_botryllus_more.tsv", sep="\t", index=False)
    print(e.groupby("study_accession").agg(runs=("accession", "size"), gb=("size_gb", "sum")).round(1))
    print(f"{len(e)} runs, {e.size_gb.sum():.1f} GB")


if __name__ == "__main__":
    main()
