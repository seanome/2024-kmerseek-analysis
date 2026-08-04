#!/usr/bin/env python3
"""Fetch human + mouse canonical-transcript CDS sequences from the Ensembl REST API.

Input: a gene-pairs CSV with at least `human_gene, human_ensg, mouse_gene, mouse_ensmusg`
columns (produced by notebook 206's "Preparing gene pairs for the omega pipeline" section).

Output: one two-record CDS FASTA per gene pair (`{outdir}/{human_gene}__{mouse_gene}.cds.fa`),
named `human` and `mouse`, plus a `fetch_manifest.tsv` recording per-pair status so a failed
or interrupted run can be resumed without re-fetching pairs that already succeeded.

Single serial process (not one Nextflow task per gene pair) deliberately, so the ~15
req/s Ensembl REST rate limit is enforced from one place rather than fanned out across
parallel tasks that would each retry independently and hammer the API.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import requests

REST_BASE = "https://rest.ensembl.org"
HEADERS = {"Content-Type": "application/json"}


def get_with_retry(url, headers=None, max_retries=5, sleep_s=0.15):
    for attempt in range(max_retries):
        resp = requests.get(url, headers=headers or {})
        if resp.status_code == 200:
            time.sleep(sleep_s)
            return resp
        if resp.status_code == 429:
            wait = float(resp.headers.get("Retry-After", 2.0))
            print(f"  429 rate-limited, sleeping {wait:.1f}s (attempt {attempt + 1}/{max_retries})", file=sys.stderr)
            time.sleep(wait)
            continue
        if resp.status_code == 404:
            return None
        print(f"  HTTP {resp.status_code} for {url}, retrying (attempt {attempt + 1}/{max_retries})", file=sys.stderr)
        time.sleep(sleep_s * (attempt + 1))
    return None


def canonical_transcript_id(gene_id: str) -> str | None:
    resp = get_with_retry(f"{REST_BASE}/lookup/id/{gene_id}?expand=0", headers=HEADERS)
    if resp is None:
        return None
    data = resp.json()
    return data.get("canonical_transcript", "").split(".")[0] or None


def fetch_cds(transcript_id: str) -> str | None:
    resp = get_with_retry(
        f"{REST_BASE}/sequence/id/{transcript_id}?type=cds",
        headers={"Content-Type": "text/x-fasta"},
    )
    if resp is None:
        return None
    lines = resp.text.strip().splitlines()
    return "".join(line for line in lines if not line.startswith(">"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gene_pairs", required=True, help="CSV with human_gene,human_ensg,mouse_gene,mouse_ensmusg")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--sleep", type=float, default=0.15, help="seconds between REST calls")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit (debug/test mode)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = outdir / "fetch_manifest.tsv"

    already_done = set()
    if manifest_path.exists():
        with open(manifest_path) as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if row["status"] == "ok":
                    already_done.add((row["human_gene"], row["mouse_gene"]))

    with open(args.gene_pairs) as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    manifest_exists = manifest_path.exists()
    manifest_f = open(manifest_path, "a", newline="")
    writer = csv.DictWriter(manifest_f, fieldnames=["human_gene", "mouse_gene", "status", "detail"], delimiter="\t")
    if not manifest_exists:
        writer.writeheader()

    n_ok = n_skip = n_fail = 0
    for i, row in enumerate(rows):
        hgene, mgene = row["human_gene"], row["mouse_gene"]
        if (hgene, mgene) in already_done:
            n_skip += 1
            continue

        out_fa = outdir / f"{hgene}__{mgene}.cds.fa"
        try:
            human_tx = canonical_transcript_id(row["human_ensg"])
            mouse_tx = canonical_transcript_id(row["mouse_ensmusg"])
            if not human_tx or not mouse_tx:
                raise ValueError(f"no canonical transcript (human={human_tx}, mouse={mouse_tx})")

            human_cds = fetch_cds(human_tx)
            mouse_cds = fetch_cds(mouse_tx)
            if not human_cds or not mouse_cds:
                raise ValueError("CDS fetch returned empty")
            if len(human_cds) % 3 or len(mouse_cds) % 3:
                raise ValueError(f"CDS length not divisible by 3 (human={len(human_cds)}, mouse={len(mouse_cds)})")

            with open(out_fa, "w") as fh:
                fh.write(f">human\n{human_cds}\n>mouse\n{mouse_cds}\n")
            writer.writerow({"human_gene": hgene, "mouse_gene": mgene, "status": "ok", "detail": ""})
            n_ok += 1
        except Exception as e:
            writer.writerow({"human_gene": hgene, "mouse_gene": mgene, "status": "fail", "detail": str(e)})
            n_fail += 1

        if (i + 1) % 25 == 0:
            manifest_f.flush()
            print(f"[{i + 1}/{len(rows)}] ok={n_ok} fail={n_fail} skip={n_skip}", flush=True)

    manifest_f.close()
    print(f"Done. ok={n_ok} fail={n_fail} skipped(already cached)={n_skip} total={len(rows)}")


if __name__ == "__main__":
    main()
