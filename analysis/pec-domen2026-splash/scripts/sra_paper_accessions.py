"""List every sequence-data accession Domen et al. 2026 (PNAS) names, and resolve each to runs.

The accessions were read from the paper's PMC full text (`$S/si/fulltext.xml`), Data, Materials,
and Software Availability statement and reference list (refs 42, 61-68). For each GEO series we
ask NCBI what kind of data it is (array or sequencing) and whether it links to an SRA study; for
each SRA study we ask ENA's filereport API for its runs and base counts.

Usage: python sra_paper_accessions.py <out_dir>
Writes paper_accessions.tsv and paper_accession_runs.tsv into <out_dir>.
"""

import csv
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ENA = "https://www.ebi.ac.uk/ena/portal/api/filereport"

WHERE_DATA = "Data, Materials, and Software Availability statement"
# accession, what the paper says it is, where it is named
PAPER = [
    (
        "PRJNA1219309",
        "this study's Botryllus RNA-seq (PEC and control)",
        WHERE_DATA + "; ref 42",
    ),
    (
        "GSE104079",
        "human and mouse exercise data: Marosi et al., intermittent fasting and endurance",
        WHERE_DATA + "; ref 61",
    ),
    (
        "GSE27536",
        "human and mouse exercise data: Turan et al., vastus lateralis before/after 8 weeks training (healthy and COPD)",
        WHERE_DATA + "; ref 62; Methods (Fig. 4C scores)",
    ),
    (
        "GSE120862",
        "human and mouse exercise data: Popov et al., acute endurance exercise and training, skeletal muscle",
        WHERE_DATA + "; ref 63",
    ),
    (
        "GSE83578",
        "human and mouse exercise data: Abbasi et al., leukocytes after endotoxin and exercise",
        WHERE_DATA + "; ref 64",
    ),
    (
        "GSE111555",
        "human and mouse exercise data: da Silva et al., blood and muscle signatures of training",
        WHERE_DATA + "; ref 65",
    ),
    (
        "GSE99963",
        "human and mouse exercise data: Laker et al., short-term nutrient-exercise stress [RNA-seq]",
        WHERE_DATA + "; ref 66",
    ),
    (
        "GSE144304",
        "human and mouse exercise data: van der Hoek et al., muscle in fit and (pre-)frail adults",
        WHERE_DATA + "; ref 67",
    ),
    (
        "GSE28998",
        "human and mouse exercise data: Gordon et al., muscle response to exercise",
        WHERE_DATA + "; ref 68",
    ),
]
RUN_FIELDS = "run_accession,study_accession,sample_accession,scientific_name,library_strategy,library_layout,read_count,base_count,fastq_bytes,fastq_ftp,fastq_md5"


def get(url: str) -> str:
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read().decode()
        except Exception:  # noqa: BLE001 - network retry
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(f"failed: {url}")


def geo_summary(gse: str) -> dict:
    ids = json.loads(
        get(
            f"{EUTILS}/esearch.fcgi?db=gds&term={gse}%5BACCN%5D+AND+gse%5BETYP%5D&retmode=json"
        )
    )["esearchresult"]["idlist"]
    if not ids:
        return {}
    d = json.loads(get(f"{EUTILS}/esummary.fcgi?db=gds&id={ids[0]}&retmode=json"))[
        "result"
    ][ids[0]]
    sra = [
        x["targetobject"]
        for x in d.get("extrelations", [])
        if x["relationtype"] == "SRA"
    ]
    return {
        "data_type": d["gdstype"],
        "organism": d["taxon"],
        "n_samples": d["n_samples"],
        "sra_study": ";".join(sra),
        "geo_title": d["title"],
    }


def ena_runs(study: str) -> list[dict]:
    text = get(
        f"{ENA}?accession={study}&result=read_run&fields={RUN_FIELDS}&format=tsv"
    )
    return list(csv.DictReader(io.StringIO(text), delimiter="\t"))


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    rows, runs = [], []
    for acc, what, where in PAPER:
        info = {
            "data_type": "sequencing (SRA BioProject)",
            "organism": "Botryllus schlosseri",
            "n_samples": "",
            "sra_study": acc,
            "geo_title": "",
        }
        if acc.startswith("GSE"):
            info = geo_summary(acc)
            time.sleep(0.4)
        study_runs = []
        for study in filter(None, info.get("sra_study", "").split(";")):
            for r in ena_runs(study):
                r["paper_accession"] = acc
                study_runs.append(r)
        runs += study_runs
        gbp = sum(int(r["base_count"] or 0) for r in study_runs) / 1e9
        rows.append(
            {
                "accession": acc,
                "what_it_is": what,
                "where_named": where,
                "geo_data_type": info.get("data_type", ""),
                "organism": info.get("organism", ""),
                "geo_samples": info.get("n_samples", ""),
                "sra_study": info.get("sra_study", ""),
                "n_runs": len(study_runs),
                "gbp": round(gbp, 2),
            }
        )
        print(
            acc,
            info.get("data_type"),
            info.get("organism"),
            info.get("sra_study"),
            len(study_runs),
            f"{gbp:.1f} Gbp",
            flush=True,
        )
    with (out / "paper_accessions.tsv").open("w") as f:
        w = csv.DictWriter(f, list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    with (out / "paper_accession_runs.tsv").open("w") as f:
        cols = ["paper_accession"] + RUN_FIELDS.split(",")
        w = csv.DictWriter(
            f, cols, delimiter="\t", lineterminator="\n", extrasaction="ignore"
        )
        w.writeheader()
        w.writerows(runs)


if __name__ == "__main__":
    main()
