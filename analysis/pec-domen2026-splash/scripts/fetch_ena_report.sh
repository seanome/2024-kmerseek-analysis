#!/bin/bash
# Fresh ENA file report for PRJNA1219309, compared run by run with the handoff table.
set -euo pipefail
: "${PEC_ROOT:?}" "${PEC_CODE:?}"
mkdir -p "$PEC_ROOT/meta" && cd "$PEC_ROOT/meta"
F="run_accession,sample_accession,experiment_alias,sample_alias,library_name,read_count,base_count,fastq_ftp,fastq_md5,fastq_bytes,instrument_model,library_layout,library_strategy,first_public"
curl -fsSL -o ena_filereport.tsv "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=PRJNA1219309&result=read_run&fields=$F&format=tsv"
date -u +%FT%TZ > ena_filereport.fetched_at
sha256sum ena_filereport.tsv > ena_filereport.sha256
tail -n +2 ena_filereport.tsv | cut -f1 | sort > runs.txt
echo "runs: $(wc -l < runs.txt)"
# compare MD5s with the handoff table
python3 - "$PEC_CODE/handoff/samples_all64.tsv" ena_filereport.tsv <<'PY'
import sys, csv
h = {r["run"]: r for r in csv.DictReader(open(sys.argv[1]), delimiter="\t")}
e = {r["run_accession"]: r for r in csv.DictReader(open(sys.argv[2]), delimiter="\t")}
bad = 0
for run, r in e.items():
    m = dict(zip([u.rsplit("_", 1)[1] for u in r["fastq_ftp"].split(";")], r["fastq_md5"].split(";")))
    hh = h.get(run)
    if hh is None or hh["md5_1"] != m.get("1.fastq.gz") or hh["md5_2"] != m.get("2.fastq.gz") or int(hh["read_pairs"]) != int(r["read_count"]):
        bad += 1; print("MISMATCH", run)
print(f"ENA runs {len(e)}, handoff runs {len(h)}, mismatches {bad}")
if set(e) != set(h):
    raise SystemExit("ENA and handoff run sets differ")
if bad:
    raise SystemExit("ENA metadata differs from the handoff")
PY
