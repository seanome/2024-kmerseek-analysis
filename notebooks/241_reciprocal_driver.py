#!/usr/bin/env python3
"""The other direction of notebook 241: the human partner as the query.

The forward search asks where BCL2 ranks among human proteins when Ced9 is the query,
and where CD47 ranks when P66 is the query. This script asks the reverse:

  BCL2 (human) against the C. elegans proteome      -> where does CED-9 rank?
  CD47 (human) against the B. burgdorferi proteome  -> where does P66 rank?

A pair of proteins that find each other first in both directions is a reciprocal best
hit, the usual test for orthology. Each database is indexed on its own, so the IDF of
a k-mer and the Karlin-Altschul fit come from that proteome, as they would for anyone
searching it. The alphabets, k values, mismatch penalties and X-drops are the
forward run's (plan.json), so every arm has a reverse twin.

Databases: UniProt's reference proteome files, one protein per gene, downloaded on the
first run (the release and the sha256 of each file go into provenance.json):
  C. elegans       UP000001940, 19_792 proteins in 2026_03, CED-9 is P41958
  B. burgdorferi   UP000001807 (strain B31), 1_291 proteins, P66 is H7C7N8
Not the REST API's proteome query: for C. elegans it returns 26_629 entries, several
per gene (hecd-1 eight times), and copies of one gene push the partner down the ranks.
Headers are rewritten as accession|gene|length so the collector reads the gene the
same way it reads GENCODE's.

Usage:
  241_reciprocal_driver.py [--workers 2] [--dry-run]   # index, search (stdlib only)
  241_reciprocal_driver.py --collect                    # tables (needs polars)

Writes under $NB241_DIR/reciprocal/: reciprocal_ranks.csv (one row per database,
arm and metric) and reciprocal_top_hits.csv (the 20 best targets of each).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import hashlib
import re
import importlib.util
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


drv = load("nb241_driver", "241_alphabet_ranking_driver.py")
OUT = drv.OUT / "reciprocal"

# database label -> (path under reference_proteomes/, query gene in the human FASTA,
# partner gene in the database as UniProt's GN= gives it)
CASES = {
    "celegans": ("Eukaryota/UP000001940/UP000001940_6239.fasta.gz", "BCL2", "ced-9"),
    "bburgdorferi": ("Bacteria/UP000001807/UP000001807_224326.fasta.gz", "CD47", "p66"),
}
UNIPROT = (
    "https://ftp.uniprot.org/pub/databases/uniprot/current_release/"
    "knowledgebase/reference_proteomes/"
)


def parse_fasta(path: Path):
    name, seq = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(seq)
                name, seq = line[1:], []
            else:
                seq.append(line.strip())
    if name is not None:
        yield name, "".join(seq)


def uniprot_header(h: str, seq: str) -> str:
    """'sp|P41958|CED9_CAEEL Apoptosis regulator ced-9 OS=... GN=ced-9 PE=1' ->
    'P41958|ced-9|280'. Entries with no GN= keep the accession as the gene."""
    acc = h.split("|")[1] if "|" in h else h.split()[0]
    gene = acc
    for tok in h.split():
        if tok.startswith("GN="):
            gene = tok[3:]
    return f"{acc}|{gene}|{len(seq)}"


def fetch_database(label: str, proteome: str) -> dict:
    """Download the proteome once, rewrite its headers, return its provenance."""
    fa = OUT / f"{label}.fa"
    raw = OUT / f"{label}.uniprot.fa"
    prov = {"proteome": proteome.split("/")[1], "url": UNIPROT + proteome}
    if not raw.exists():
        with urllib.request.urlopen(prov["url"], timeout=600) as r:
            body = gzip.decompress(r.read())
        with urllib.request.urlopen(UNIPROT + "RELEASE.metalink", timeout=60) as r:
            m = re.search(r"<version>([^<]+)</version>", r.read().decode())
            prov["uniprot_release"] = m.group(1) if m else None
        if not body.startswith(b">"):
            raise SystemExit(
                f"241 reciprocal: {proteome} download is not FASTA: {body[:200]!r}"
            )
        raw.write_bytes(body)
    prov["sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
    n = 0
    with open(fa, "w") as fh:
        for h, s in parse_fasta(raw):
            fh.write(f">{uniprot_header(h, s)}\n{s}\n")
            n += 1
    prov["n_proteins"] = n
    return prov


def partner_header(label: str, gene: str) -> str:
    hits = [h for h, _ in parse_fasta(OUT / f"{label}.fa") if h.split("|")[1] == gene]
    if len(hits) != 1:
        raise SystemExit(
            f"241 reciprocal: {len(hits)} entries for {gene} in {label}: {hits}"
        )
    return hits[0]


def write_query(label: str, gene: str) -> Path:
    """The human partner's sequence from the forward run's database, named by its gene."""
    seqs = [s for h, s in parse_fasta(drv.HUMAN) if h.split("|")[6] == gene]
    if len(seqs) != 1:
        raise SystemExit(f"241 reciprocal: {len(seqs)} human entries for {gene}")
    q = OUT / f"{label}.query.fa"
    q.write_text(f">{gene}\n{seqs[0]}\n")
    return q


def run(args) -> None:
    for d in ("idx", "search", "ka_survival", "pair", "logs"):
        for label in CASES:
            (OUT / label / d).mkdir(parents=True, exist_ok=True)
    prov = {label: fetch_database(label, p) for label, (p, _, _) in CASES.items()}
    old = (
        json.loads((OUT / "provenance.json").read_text())
        if (OUT / "provenance.json").exists()
        else {}
    )
    for label in prov:  # keep the release the first download recorded
        for key in ("uniprot_release",):
            if prov[label].get(key) is None and old.get(label, {}).get(key):
                prov[label][key] = old[label][key]
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=1))
    if args.fetch_only:
        print(json.dumps(prov, indent=1))
        return

    plan = json.loads((drv.OUT / "plan.json").read_text())
    if args.only:
        keep = set(args.only.split(","))
        plan = [a for a in plan if f"{a['alphabet']}.k{a['ksize']}" in keep]
    jobs = []
    for label, (_, query_gene, partner_gene) in CASES.items():
        q = write_query(label, query_gene)
        pairs = {query_gene: partner_header(label, partner_gene)}
        for arm in plan:
            jobs.append(
                (
                    arm["alphabet"],
                    arm["ksize"],
                    arm["penalty"],
                    arm["xdrop"],
                    OUT / f"{label}.fa",
                    q,
                    OUT / label,
                    pairs,
                    label,
                )
            )
    print(
        f"{len(jobs)} index+search jobs ({len(plan)} arms x {len(CASES)} databases)",
        file=sys.stderr,
    )
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(
                drv.one_arm,
                a,
                k,
                c,
                x,
                args.dry_run,
                db=db,
                queries=q,
                out=out,
                pairs=pairs,
                db_label=label,
            )
            for a, k, c, x, db, q, out, pairs, label in jobs
        ]
        for fut in cf.as_completed(futs):
            rec = fut.result()
            print(json.dumps(rec), file=sys.stderr, flush=True)
            with open(OUT / "runs.jsonl", "a") as fh:
                fh.write(json.dumps(rec) + "\n")


def collect() -> None:
    import polars as pl

    col = load("nb241_collect", "241_alphabet_ranking_collect.py")
    plan = json.loads((drv.OUT / "plan.json").read_text())
    ranks, tops = [], []
    for label, (_, query_gene, partner_gene) in CASES.items():
        for arm in plan:
            a, k, bits = arm["alphabet"], arm["ksize"], arm["bits"]
            df = col.read_search(f"{a}.k{k}", out=OUT / label)
            if df is None:
                continue
            surv = OUT / label / "ka_survival" / f"{a}.k{k}.retry.csv"
            if not surv.exists():
                surv = OUT / label / "ka_survival" / f"{a}.k{k}.csv"
            fitted = drv.fitted(surv)
            for r in col.rank_rows(
                df,
                a,
                k,
                bits,
                queries=[query_gene],
                partner_of={query_gene: partner_gene},
            ):
                ranks.append({"database": label, "fitted": fitted, **r})
            for r in col.top_rows(
                df, a, k, bits, queries=[query_gene], partner={query_gene: partner_gene}
            ):
                tops.append({"database": label, **r})
    pl.DataFrame(ranks, infer_schema_length=None).write_csv(
        OUT / "reciprocal_ranks.csv"
    )
    pl.DataFrame(tops, schema={"database": pl.Utf8, **col.TOP_SCHEMA}).write_csv(
        OUT / "reciprocal_top_hits.csv"
    )
    print(f"rank rows: {len(ranks)}  top-hit rows: {len(tops)}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument(
        "--fetch-only",
        action="store_true",
        help="download the two proteomes and stop (for a login node, if compute nodes "
        "have no internet)",
    )
    ap.add_argument(
        "--only",
        default="",
        help="comma-separated arms, e.g. dayhoff6.k15 (for a test)",
    )
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    collect() if args.collect else run(args)


if __name__ == "__main__":
    main()
