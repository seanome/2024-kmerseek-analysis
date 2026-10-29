#!/usr/bin/env python3
"""Notebook 301, steps 2 and 4: UniRef90 cluster of every query, and Swiss-Prot without them.

Standard library only (runs on Sherlock with `module load python/3.12.1`).

For every accession in --accessions (real queries, chimera hosts and chimera insert
sources, from build_301_step2_queries.py), ask rest.uniprot.org which UniRef90 cluster
holds it. Every Swiss-Prot member of those clusters is then left out of the search
database, so no query can find itself or a sequence at 90% identity or more to itself.
One database serves every query, so each query also loses the other queries' clusters.

The cluster's member list comes from the UniRef record. When the record lists fewer
members than its size, the Swiss-Prot members are fetched from UniProtKB instead
(query `uniref_cluster_90:<id> AND reviewed:true`).

Outputs in --out
  uniref90_clusters.tsv        accession, cluster_id, cluster_size, n_swissprot_members
  removed_from_swissprot.txt   primary accessions left out of the database
  swissprot_minus_query_clusters.fasta
  uniref90_counts.tsv          entries before and after, removed, UniProt release
  uniref90_responses.jsonl     every server answer, so a rerun reads the cache
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://rest.uniprot.org"
BATCH = 50
RETRIES = 5


class Cache:
    """Server answers keyed by URL, appended to a JSON-lines file."""

    def __init__(self, path: Path):
        self.path, self.data = path, {}
        if path.exists():
            for line in path.read_text().splitlines():
                r = json.loads(line)
                self.data[r["url"]] = r
        self.release = next(
            (r["release"] for r in self.data.values() if r.get("release")), ""
        )

    def get(self, url: str) -> str:
        if url in self.data:
            return self.data[url]["body"]
        for attempt in range(RETRIES):
            try:
                with urllib.request.urlopen(url, timeout=120) as resp:
                    body = resp.read().decode()
                    release = resp.headers.get("X-UniProt-Release", "")
                break
            except Exception as exc:  # network errors and 5xx answers both retry
                if attempt == RETRIES - 1:
                    raise
                print(f"retry {attempt + 1} after {exc}: {url}", file=sys.stderr)
                time.sleep(2**attempt * 5)
        if body.lstrip().startswith("<"):
            raise RuntimeError(f"HTML instead of data from {url}")
        self.release = self.release or release
        rec = {"url": url, "body": body, "release": release}
        self.data[url] = rec
        with open(self.path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        return body


def uniref_url(accs: list[str]) -> str:
    q = "(" + " OR ".join(f"uniprot_id:{a}" for a in accs) + ") AND identity:0.9"
    return f"{API}/uniref/search?" + urllib.parse.urlencode(
        {"query": q, "fields": "id,count,members", "format": "tsv", "size": 500}
    )


def parse_uniref_tsv(body: str) -> list[tuple[str, int, list[str]]]:
    rows = []
    for line in body.splitlines()[1:]:
        if not line.strip():
            continue
        cid, count, members = line.split("\t")
        rows.append((cid, int(count), [m.strip() for m in members.split(";")]))
    return rows


def reviewed_members(cache: Cache, cid: str) -> list[str]:
    out, url = [], f"{API}/uniprotkb/search?" + urllib.parse.urlencode(
        {
            "query": f"uniref_cluster_90:{cid} AND reviewed:true",
            "format": "list",
            "size": 500,
        }
    )
    body = cache.get(url)
    out += [a for a in body.split() if a]
    if len(out) == 500:
        raise RuntimeError(f"{cid}: 500 or more Swiss-Prot members; add paging")
    return out


def read_dat_accessions(path: Path):
    """(primary accession, all accessions, sequence) per Swiss-Prot entry."""
    opener = gzip.open if str(path).endswith(".gz") else open
    accs, seq, in_seq = [], [], False
    with opener(path, "rt") as fh:
        for line in fh:
            tag = line[:2]
            if tag == "ID":
                accs, seq, in_seq = [], [], False
            elif tag == "AC":
                accs += [a for a in line[5:].replace(";", " ").split() if a]
            elif tag == "SQ":
                in_seq = True
            elif tag == "//":
                yield accs[0], accs, "".join(seq)
            elif in_seq and line.startswith("     "):
                seq.append(line.replace(" ", "").strip())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--accessions", type=Path, required=True)
    ap.add_argument("--sprot-dat", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cache = Cache(a.out / "uniref90_responses.jsonl")
    wanted = [x for x in a.accessions.read_text().split() if x]

    entries = list(read_dat_accessions(a.sprot_dat))
    to_primary = {acc: p for p, accs, _ in entries for acc in accs}

    cluster_of: dict[str, tuple[str, int, list[str]]] = {}
    for i in range(0, len(wanted), BATCH):
        batch = wanted[i : i + BATCH]
        for cid, count, members in parse_uniref_tsv(cache.get(uniref_url(batch))):
            for acc in batch:
                if acc in members:
                    cluster_of[acc] = (cid, count, members)
    for acc in wanted:  # a long member list can leave a query out of its own record
        if acc not in cluster_of:
            rows = parse_uniref_tsv(cache.get(uniref_url([acc])))
            if len(rows) != 1:
                raise RuntimeError(f"{acc}: {len(rows)} UniRef90 clusters")
            cluster_of[acc] = rows[0]

    removed: set[str] = set()
    out_rows = []
    sp_members: dict[str, set[str]] = {}
    for acc in wanted:
        cid, count, members = cluster_of[acc]
        if cid not in sp_members:
            if len(members) < count:
                listed = reviewed_members(cache, cid)
            else:
                listed = members
            sp_members[cid] = {to_primary[m] for m in listed if m in to_primary}
        # the query itself is always removed, whatever the member list says
        sp = sp_members[cid] | ({to_primary[acc]} if acc in to_primary else set())
        removed |= sp
        out_rows.append((acc, cid, count, len(sp)))

    with open(a.out / "uniref90_clusters.tsv", "w") as fh:
        fh.write("accession\tcluster_id\tcluster_size\tn_swissprot_members\n")
        for r in out_rows:
            fh.write("\t".join(map(str, r)) + "\n")
    (a.out / "removed_from_swissprot.txt").write_text("\n".join(sorted(removed)) + "\n")
    kept = 0
    with open(a.out / "swissprot_minus_query_clusters.fasta", "w") as fh:
        for p, _, s in entries:
            if p in removed:
                continue
            kept += 1
            fh.write(f">{p}\n")
            for j in range(0, len(s), 60):
                fh.write(s[j : j + 60] + "\n")
    counts = {
        "UniProt release (X-UniProt-Release header)": cache.release,
        "accessions looked up": len(wanted),
        "distinct UniRef90 clusters": len({r[1] for r in out_rows}),
        "Swiss-Prot entries": len(entries),
        "Swiss-Prot entries removed": len(removed),
        "Swiss-Prot entries in the search database": kept,
    }
    with open(a.out / "uniref90_counts.tsv", "w") as fh:
        fh.write("step\tvalue\n")
        for k, v in counts.items():
            fh.write(f"{k}\t{v}\n")
            print(f"{k}\t{v}")


if __name__ == "__main__":
    main()
