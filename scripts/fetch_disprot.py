#!/usr/bin/env python3
"""Download one DisProt release and write the tables notebook 251 and the pipeline read.

DisProt annotates regions of proteins that were shown by experiment to be disordered. Each
region carries one ontology term from one namespace. The namespace this benchmark needs is
"Disorder function": what the disordered region does (flexible linker, phosphorylation
display site, molecular adaptor activity, ...). A region is kept when

  1. its namespace is "Disorder function",
  2. the evidence code for that function is experimental (GO codes EXP, IDA, IPI, IMP, IGI,
     IEP; IC, curator inference, and no code at all are left out), and
  3. it overlaps a "Structural state" region with the term "disorder" (IDPO:0000002) that
     itself has experimental evidence, so the region is shown disordered, not assumed.

Coordinates are DisProt's own: 1-based and inclusive, on the sequence DisProt stores. A
protein is only used where the QfO 2020_04 proteome holds the identical sequence under the
same accession, because a coordinate on a different sequence version points at different
residues. Every row records whether that holds.

Outputs, under --out-dir:

  raw/disprot_<release>.json         the API response, unchanged
  disprot_proteins.parquet           one row per DisProt protein
  disprot_function_regions.parquet   one row per kept (protein, interval, term), with the
                                     number of DisProt assertions behind it
  disprot_function_regions_all.parquet  every functional-region row, kept or not, with
                                     `keep` and the reason a dropped row was dropped
  human_query_accessions.txt         human proteins with a kept region and a QfO match
  disprot_nonhuman.fasta             every non-human DisProt protein that has a kept
                                     region, in QfO-style headers: the pooled target
  fetch_summary.json                 counts at every filter, and the URL fetched

The pooled FASTA uses DisProt's sequences. Where a protein is also in a QfO proteome with
the same sequence, the pooled copy and the QfO copy are the same protein; the pool is a
separate target, searched on its own, so this does not double-count anything.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------------
# Config. Every external identifier this script uses, in one place. Printed at start-up
# so it can be checked against https://disprot.org/api before the numbers are trusted.
# ---------------------------------------------------------------------------
DISPROT_RELEASE = "2026_06"  # "current" on 2026-09-24; the newest region release date
DISPROT_URL = (
    "https://disprot.org/api/search?release={release}&show_ambiguous=true"
    "&show_obsolete=false&format=json&page_size=100000"
)
FUNCTION_NAMESPACE = "Disorder function"
STATE_NAMESPACE = "Structural state"
DISORDER_TERM = (
    "IDPO:0000002"  # "disorder" in the Intrinsically Disordered Proteins Ontology
)
# GO experimental evidence codes (geneontology.org/docs/guide-go-evidence-codes).
EXPERIMENTAL_CODES = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP"}
HUMAN_TAXON = 9606

REPO = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = (
    REPO / "nextflow-runs" / "qfo-pfam-region-benchmark" / "assets" / "qfo_species.tsv"
)
DEFAULT_QFO = (
    Path.home()
    / "data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143"
)
DEFAULT_OUT = Path.home() / "data/disprot-region-transfer"
#: The nine targets the region benchmark has always run. Other registry rows are ignored.
QFO_TARGETS = [
    "mouse",
    "chicken",
    "zebrafish",
    "ciona",
    "fly",
    "worm",
    "yeast",
    "arabidopsis",
    "ecoli",
]


def read_fasta(path: Path) -> dict[str, str]:
    """accession -> sequence for a QfO FASTA (>sp|ACC|NAME ... or >tr|ACC|NAME ...)."""
    seqs: dict[str, list[str]] = {}
    acc = None
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                tok = line[1:].split()[0]
                acc = tok.split("|")[1] if "|" in tok else tok
                seqs[acc] = []
            elif acc is not None:
                seqs[acc].append(line.strip())
    return {a: "".join(s) for a, s in seqs.items()}


def fetch(url: str, dest: Path) -> dict:
    if dest.exists():
        print(f"  using cached {dest}")
        return json.loads(dest.read_text())
    print(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=600) as resp:
        body = resp.read()
    payload = json.loads(body)
    # DisProt answers a bad release name with a small JSON error, not an HTTP error.
    if "data" not in payload:
        raise SystemExit(f"DisProt returned no 'data' for {url}: {body[:300]!r}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)
    return payload


def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return a0 <= b1 and b0 <= a1


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--release", default=DISPROT_RELEASE)
    ap.add_argument("--qfo-dir", type=Path, default=DEFAULT_QFO)
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    url = DISPROT_URL.format(release=args.release)
    print("External sources (check these):")
    print(f"  DisProt release {args.release}: {url}")
    print(f"  QfO proteomes:  {args.qfo_dir}")
    print(f"  species table:  {args.registry}")
    print(
        f"  kept: namespace '{FUNCTION_NAMESPACE}', evidence in "
        f"{sorted(EXPERIMENTAL_CODES)}, overlapping '{STATE_NAMESPACE}' {DISORDER_TERM} "
        f"with experimental evidence"
    )

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    payload = fetch(url, out / "raw" / f"disprot_{args.release}.json")
    entries = payload["data"]

    with open(args.registry, newline="") as fh:
        registry = {r["label"]: r for r in csv.DictReader(fh, delimiter="\t")}
    taxon_to_label = {
        int(registry[l]["taxon"]): l for l in ["human", *QFO_TARGETS] if l in registry
    }
    qfo_seqs: dict[str, dict[str, str]] = {}
    for label in taxon_to_label.values():
        r = registry[label]
        qfo_seqs[label] = read_fasta(
            args.qfo_dir / r["subdir"] / f"{r['proteome']}_{r['taxon']}.fasta"
        )

    proteins, regions = [], []
    for e in entries:
        acc = e["acc"]
        taxon = int(e["ncbi_taxon_id"])
        seq = e["sequence"]
        label = taxon_to_label.get(taxon)
        qfo_seq = qfo_seqs[label].get(acc) if label else None
        genes = e.get("genes") or []
        gene = (genes[0].get("name") or {}).get("value", "") if genes else ""
        proteins.append(
            dict(
                accession=acc,
                disprot_id=e["disprot_id"],
                taxon=taxon,
                organism=e.get("organism", ""),
                gene=gene,
                protein_name=e.get("name", ""),
                length=len(seq),
                sequence=seq,
                seq_md5=hashlib.md5(seq.encode()).hexdigest(),
                qfo_label=label,
                in_qfo=qfo_seq is not None,
                qfo_same_sequence=qfo_seq == seq,
            )
        )
        disorder = [
            (r["start"], r["end"])
            for r in e["regions"]
            if r.get("disprot_namespace") == STATE_NAMESPACE
            and r.get("term_id") == DISORDER_TERM
            and r.get("ec_go") in EXPERIMENTAL_CODES
        ]
        for r in e["regions"]:
            if r.get("disprot_namespace") != FUNCTION_NAMESPACE:
                continue
            s, t = int(r["start"]), int(r["end"])
            exp = r.get("ec_go") in EXPERIMENTAL_CODES
            in_dis = any(overlaps(s, t, a, b) for a, b in disorder)
            why = []
            if not exp:
                why.append(f"evidence {r.get('ec_go') or 'none'}")
            if not in_dis:
                why.append("no experimental disorder region overlaps it")
            regions.append(
                dict(
                    accession=acc,
                    taxon=taxon,
                    region_id=r.get("region_id", ""),
                    start=s,
                    end=t,
                    length=t - s + 1,
                    term_id=r.get("term_id", ""),
                    term_name=r.get("term_name", ""),
                    ec_go=r.get("ec_go"),
                    ec_id=r.get("ec_id"),
                    ec_name=r.get("ec_name"),
                    reference_id=r.get("reference_id"),
                    region_released=r.get("released"),
                    experimental_function=exp,
                    overlaps_experimental_disorder=in_dis,
                    keep=exp and in_dis,
                    drop_reason="; ".join(why) or None,
                )
            )

    prot = pl.DataFrame(proteins)
    reg = (
        pl.DataFrame(regions).join(
            prot.select("accession", "qfo_label", "in_qfo", "qfo_same_sequence"),
            on="accession",
        )
        # One functional term can be asserted twice on one interval by two papers. The
        # benchmark unit is (protein, interval, term); the evidence of the first row is kept
        # and the number of assertions recorded.
        .sort("accession", "start", "end", "term_id", "region_id")
    )
    kept = (
        reg.filter("keep")
        .group_by("accession", "start", "end", "term_id", maintain_order=True)
        .agg(pl.all().first(), n_assertions=pl.len())
        .select(reg.columns + ["n_assertions"])
    )
    prot.write_parquet(out / "disprot_proteins.parquet")
    reg.write_parquet(out / "disprot_function_regions_all.parquet")
    kept.write_parquet(out / "disprot_function_regions.parquet")

    human_q = (
        kept.filter((pl.col("taxon") == HUMAN_TAXON) & pl.col("qfo_same_sequence"))[
            "accession"
        ]
        .unique()
        .sort()
    )
    (out / "human_query_accessions.txt").write_text("\n".join(human_q) + "\n")

    pool = prot.filter(
        (pl.col("taxon") != HUMAN_TAXON)
        & pl.col("accession").is_in(kept["accession"].unique().implode())
    ).sort("accession")
    with open(out / "disprot_nonhuman.fasta", "w") as fh:
        for r in pool.iter_rows(named=True):
            # QfO style, so every tool's accession parsing treats it like the proteomes.
            fh.write(
                f">dp|{r['accession']}|{r['disprot_id']} OS={r['organism']} "
                f"OX={r['taxon']}\n{r['sequence']}\n"
            )

    per_label = (
        kept.with_columns(pl.col("qfo_label").fill_null("other organism"))
        .group_by("qfo_label")
        .agg(
            n_regions=pl.len(),
            n_proteins=pl.col("accession").n_unique(),
            n_regions_same_seq_in_qfo=pl.col("qfo_same_sequence").sum(),
        )
        .sort("n_regions", descending=True)
    )
    summary = dict(
        release=args.release,
        url=url,
        n_proteins=prot.height,
        n_function_regions=reg.height,
        n_dropped_evidence=int((~reg["experimental_function"]).sum()),
        n_dropped_no_disorder=int(
            (
                reg["experimental_function"] & ~reg["overlaps_experimental_disorder"]
            ).sum()
        ),
        n_kept_rows=int(reg["keep"].sum()),
        n_kept_unique=kept.height,
        n_terms=kept["term_id"].n_unique(),
        n_human_queries=len(human_q),
        n_pool_proteins=pool.height,
        per_label=per_label.to_dicts(),
    )
    (out / "fetch_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_label"}, indent=2))
    print(per_label)


if __name__ == "__main__":
    sys.exit(main())
