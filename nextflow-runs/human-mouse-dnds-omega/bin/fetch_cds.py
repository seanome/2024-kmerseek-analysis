#!/usr/bin/env python3
"""Fetch human + mouse canonical-transcript CDS sequences from the Ensembl REST API.

Input: a gene-pairs CSV with at least `human_gene, human_ensg, mouse_gene, mouse_ensmusg`
columns (produced by notebook 206's "Preparing gene pairs for the omega pipeline" section).

Output: one two-record CDS FASTA per gene pair (`{outdir}/{human_gene}__{mouse_gene}.cds.fa`),
named `human` and `mouse`, plus a `fetch_manifest.tsv` recording per-pair status so a failed
or interrupted run can be resumed without re-fetching pairs that already succeeded.

Uses Ensembl's batch POST endpoints (POST /lookup/id, POST /sequence/id) instead of one GET
per gene/transcript. A per-id GET run for ~1300 pairs means ~5300 individual requests, and at
Ensembl's observed 500/503 error rate that means thousands of retries with backoff -- that's
what made the original per-id version take >6h and still only get 350/1338 done. Batching
(500 ids/lookup call, 50 ids/sequence call, per Ensembl's documented limits) collapses that to
~60 requests total.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import requests

REST_BASE = "https://rest.ensembl.org"
LOOKUP_BATCH_SIZE = 500
SEQUENCE_BATCH_SIZE = 50


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def post_with_retry(url, json_body, headers, max_retries=5, sleep_s=1.0):
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=json_body, headers=headers)
        except requests.RequestException as e:
            print(f"  {type(e).__name__} for {url}, retrying (attempt {attempt + 1}/{max_retries})", file=sys.stderr)
            time.sleep(sleep_s * (attempt + 1))
            continue
        if resp.status_code == 200:
            return resp
        if resp.status_code == 429:
            wait = float(resp.headers.get("Retry-After", 2.0))
            print(f"  429 rate-limited, sleeping {wait:.1f}s (attempt {attempt + 1}/{max_retries})", file=sys.stderr)
            time.sleep(wait)
            continue
        print(f"  HTTP {resp.status_code} for {url}, retrying (attempt {attempt + 1}/{max_retries})", file=sys.stderr)
        time.sleep(sleep_s * (attempt + 1))
    return None


def batch_lookup(gene_ids, sleep_s):
    """gene_id -> canonical_transcript (version-stripped), missing ids simply absent."""
    result = {}
    for batch in chunked(sorted(gene_ids), LOOKUP_BATCH_SIZE):
        resp = post_with_retry(
            f"{REST_BASE}/lookup/id",
            {"ids": batch},
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            sleep_s=sleep_s,
        )
        if resp is not None:
            for gene_id, data in resp.json().items():
                if data and data.get("canonical_transcript"):
                    result[gene_id] = data["canonical_transcript"].split(".")[0]
        time.sleep(sleep_s)
    return result


def batch_sequences(transcript_ids, sleep_s):
    """transcript_id -> CDS sequence string, missing ids simply absent."""
    result = {}
    for batch in chunked(sorted(transcript_ids), SEQUENCE_BATCH_SIZE):
        resp = post_with_retry(
            f"{REST_BASE}/sequence/id?type=cds",
            {"ids": batch},
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            sleep_s=sleep_s,
        )
        if resp is not None:
            for record in resp.json():
                if isinstance(record, dict) and record.get("id") and record.get("seq"):
                    result[record["id"]] = record["seq"]
        time.sleep(sleep_s)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gene_pairs", required=True, help="CSV with human_gene,human_ensg,mouse_gene,mouse_ensmusg")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between batch REST calls")
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

    pending = [row for row in rows if (row["human_gene"], row["mouse_gene"]) not in already_done]
    n_skip = len(rows) - len(pending)

    gene_ids = {gid for row in pending for gid in (row["human_ensg"], row["mouse_ensmusg"])}
    print(f"Looking up canonical transcripts for {len(gene_ids)} genes ({len(pending)} pending pairs)...", flush=True)
    canonical_tx = batch_lookup(gene_ids, args.sleep)

    transcript_ids = set()
    for row in pending:
        human_tx = canonical_tx.get(row["human_ensg"])
        mouse_tx = canonical_tx.get(row["mouse_ensmusg"])
        if human_tx:
            transcript_ids.add(human_tx)
        if mouse_tx:
            transcript_ids.add(mouse_tx)

    print(f"Fetching CDS for {len(transcript_ids)} transcripts...", flush=True)
    cds_by_tx = batch_sequences(transcript_ids, args.sleep)

    manifest_exists = manifest_path.exists()
    manifest_f = open(manifest_path, "a", newline="")
    writer = csv.DictWriter(manifest_f, fieldnames=["human_gene", "mouse_gene", "status", "detail"], delimiter="\t")
    if not manifest_exists:
        writer.writeheader()

    n_ok = n_fail = 0
    for row in pending:
        hgene, mgene = row["human_gene"], row["mouse_gene"]
        human_tx = canonical_tx.get(row["human_ensg"])
        mouse_tx = canonical_tx.get(row["mouse_ensmusg"])
        human_cds = cds_by_tx.get(human_tx) if human_tx else None
        mouse_cds = cds_by_tx.get(mouse_tx) if mouse_tx else None

        if not human_tx or not mouse_tx:
            writer.writerow({"human_gene": hgene, "mouse_gene": mgene, "status": "fail",
                              "detail": f"no canonical transcript (human={human_tx}, mouse={mouse_tx})"})
            n_fail += 1
            continue
        if not human_cds or not mouse_cds:
            writer.writerow({"human_gene": hgene, "mouse_gene": mgene, "status": "fail", "detail": "CDS fetch returned empty"})
            n_fail += 1
            continue
        if len(human_cds) % 3 or len(mouse_cds) % 3:
            writer.writerow({"human_gene": hgene, "mouse_gene": mgene, "status": "fail",
                              "detail": f"CDS length not divisible by 3 (human={len(human_cds)}, mouse={len(mouse_cds)})"})
            n_fail += 1
            continue

        out_fa = outdir / f"{hgene}__{mgene}.cds.fa"
        with open(out_fa, "w") as fh:
            fh.write(f">human\n{human_cds}\n>mouse\n{mouse_cds}\n")
        writer.writerow({"human_gene": hgene, "mouse_gene": mgene, "status": "ok", "detail": ""})
        n_ok += 1

    manifest_f.close()
    print(f"Done. ok={n_ok} fail={n_fail} skipped(already cached)={n_skip} total={len(rows)}")


if __name__ == "__main__":
    main()
