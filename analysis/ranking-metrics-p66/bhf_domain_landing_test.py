#!/usr/bin/env python3
"""Do BHF's matched regions land on real human protein domains?

BHF has no Pfam annotation of its own, so the 20% rule is applied to the human
side: a region counts as on a domain when it overlaps a Pfam domain of the human
target by at least 20% of the region's own length.

The number on its own says nothing, because proteins are substantially covered by
Pfam domains anyway. The comparison is an exact placement null: slide a window of
the same length to every position in that same human protein and count how often
it would pass. That gives a per-region expectation, and the sum of those is what
BHF has to beat.

Two things this checks before trusting a coordinate:

  * Gene symbols are mapped to UniProt through HGNC, then Pfam domains are pulled
    from InterPro, which is keyed on UniProt.
  * UniProt and GENCODE do not always agree on the sequence, and a coordinate
    cannot transfer when they differ. Genes whose two lengths disagree are
    dropped rather than silently mis-labelled.
"""
import argparse
import json
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

HGNC = Path("/Users/olga/data/qfo-pfam-region-midi-plus/hgnc_complete_set.txt")
API = "https://www.ebi.ac.uk/interpro/api/entry/pfam/protein/uniprot/{}/?page_size=200"


def fetch_pfam(accessions: dict, pause: float = 0.12) -> pl.DataFrame:
    """Pfam domains with coordinates, one row per interval. 204 means no Pfam entry."""
    rows, failed = [], []
    for symbol, acc in accessions.items():
        payload = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(API.format(acc), timeout=45) as fh:
                    payload = json.loads(fh.read().decode())
                break
            except urllib.error.HTTPError as e:
                if e.code == 204:
                    payload = {"results": []}
                    break
                if attempt == 2:
                    break
                time.sleep(2)
            except Exception:
                if attempt == 2:
                    break
                time.sleep(2)
        if payload is None:
            failed.append(symbol)
            continue
        for res in payload.get("results", []):
            pfam = res["metadata"]["accession"]
            for prot in res.get("proteins", []):
                for loc in prot.get("entry_protein_locations", []):
                    for frag in loc.get("fragments", []):
                        rows.append({"symbol": symbol, "accession": acc, "pfam_id": pfam,
                                     "uniprot_length": prot.get("protein_length"),
                                     "domain_start": frag["start"], "domain_end": frag["end"]})
        time.sleep(pause)
    if failed:
        print(f"  no Pfam answer for {len(failed)}: {failed}")
    return pl.DataFrame(rows)


def overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hits", required=True, help="BHF query_subset CSV for one arm")
    p.add_argument("--min-fraction", type=float, default=0.20)
    p.add_argument("--outdir", default=".", type=Path)
    p.add_argument("--n-sim", type=int, default=20_000)
    args = p.parse_args()

    d = (pl.read_csv(args.hits)
           .with_columns(pl.col("target_name").str.split("|").list.get(-2).alias("gene"),
                         pl.col("target_name").str.split("|").list.get(-1)
                           .cast(pl.Int64).alias("gencode_length")))
    print(f"{len(d)} regions across {d['gene'].n_unique()} human genes")

    hgnc = pl.read_csv(HGNC, separator="\t", infer_schema_length=0, ignore_errors=True)
    genes = set(d["gene"].to_list())
    mapping = {r["symbol"]: r["uniprot_ids"].split("|")[0].strip()
               for r in hgnc.select(["symbol", "uniprot_ids"]).drop_nulls().iter_rows(named=True)
               if r["symbol"] in genes}
    print(f"mapped to UniProt through HGNC: {len(mapping)} of {len(genes)}")

    pfam = fetch_pfam(mapping)
    pfam.write_csv(args.outdir / "bhf_targets_pfam.csv")

    # A coordinate only transfers when the two databases agree on the sequence.
    lens = pfam.select(["symbol", "uniprot_length"]).unique()
    check = (d.select(["gene", "gencode_length"]).unique()
               .join(lens, left_on="gene", right_on="symbol", how="inner")
               .with_columns((pl.col("gencode_length") == pl.col("uniprot_length")).alias("same")))
    usable = set(check.filter(pl.col("same"))["gene"].to_list())
    print(f"lengths agree, so coordinates transfer: {len(usable)} of {len(check)} genes")

    dom = defaultdict(list)
    for r in pfam.iter_rows(named=True):
        if r["symbol"] in usable:
            dom[r["symbol"]].append((r["domain_start"] - 1, r["domain_end"]))

    rows = []
    for r in d.iter_rows(named=True):
        g = r["gene"]
        if g not in usable:
            continue
        ts, te, plen = r["target_start"], r["target_end"], r["gencode_length"]
        rlen = te - ts
        doms = dom.get(g, [])
        best = max((overlap(ts, te, s, e) for s, e in doms), default=0)
        total = max(1, plen - rlen + 1)
        hits = sum(1 for pos in range(total)
                   if any(overlap(pos, pos + rlen, s, e) >= args.min_fraction * rlen
                          for s, e in doms))
        rows.append({"gene": g, "region_length": rlen,
                     "region_evalue": r["region_evalue"],
                     "region_ka_lambda": r["region_ka_lambda"],
                     "best_overlap": best,
                     "passes": best >= args.min_fraction * rlen,
                     "expected_rate": hits / total})
    res = pl.DataFrame(rows)
    res.write_csv(args.outdir / "bhf_regions_vs_human_pfam.csv")

    n, obs = len(res), int(res["passes"].sum())
    exp = float(res["expected_rate"].sum())
    print(f"\n{n} regions scored")
    print(f"  land on a human Pfam domain      : {obs} ({100 * obs / n:.0f}%)")
    print(f"  expected from random placement   : {exp:.1f} ({100 * exp / n:.0f}%)")
    rng = np.random.default_rng(0)
    sim = (rng.random((args.n_sim, n)) < res["expected_rate"].to_numpy()).sum(axis=1)
    print(f"  P(random placement >= {obs})        : {(sim >= obs).mean():.3f}")


if __name__ == "__main__":
    main()
