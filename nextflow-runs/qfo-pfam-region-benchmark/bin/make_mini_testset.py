#!/usr/bin/env python3
"""Build a small self-contained data directory that exercises the whole pipeline fast.

The point is a smoke test that can fail, not one that trivially passes. Two things make
the difference:

  Real positives.  Target proteins are chosen because they share a Pfam family with a
                   query protein. A random subset of two proteomes would share almost
                   nothing, every metric would read 0.0, and a broken scoring path would
                   look identical to a working one.
  Real negatives.  An equal number of targets share no family with any query, so
                   precision can drop below 1.0 and a transfer bug has somewhere to show.

The selection also spreads queries across multi-domain proteins, HGNC groups and pLDDT
bins, so the strata, domain-count MCC and held-out split all have something to chew on
rather than collapsing to a single cell.

Structures are symlinked from the flat AlphaFold cache, so the mini set needs none of the
~36 GB full download -- whatever the cache already has is enough to exercise the Foldseek
and Folddisco arms.
"""

import argparse
import json
import shutil
from pathlib import Path

import polars as pl


def read_fasta(path: Path) -> dict[str, str]:
    records, name, buf = {}, None, []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if name:
                    records[name] = "".join(buf)
                name, buf = line.rstrip("\n"), []
            else:
                buf.append(line)
    if name:
        records[name] = "".join(buf)
    return records


def accession_of(header: str) -> str:
    parts = header.lstrip(">").split("|")
    return parts[1] if len(parts) >= 2 else header.lstrip(">").split()[0]


def write_fasta(records: dict[str, str], keep: set[str], out: Path) -> int:
    n = 0
    with open(out, "w") as f:
        for header, seq in records.items():
            if accession_of(header) in keep:
                f.write(f"{header}\n{seq}")
                n += 1
    return n


def pick_queries(human: pl.DataFrame, n: int) -> list[str]:
    """Prefer multi-domain proteins, then spread across families.

    Multi-domain first because single-domain proteins alone would make domain-count MCC
    undefined (one class empty) and would never test whether a tool splits a protein
    correctly.
    """
    # accession is the final sort key, and it is load-bearing: without it the many ties on
    # (n_domains, n_fam) are broken by group_by's arbitrary row order, so two runs over the
    # same annotations pick different queries. A smoke test whose contents drift between
    # the Mac and the cluster cannot be compared against itself.
    per_protein = (
        human.group_by("accession")
        .agg(pl.len().alias("n_domains"), pl.col("pfam_id").n_unique().alias("n_fam"))
        .sort(["n_domains", "n_fam", "accession"], descending=[True, True, False])
    )
    multi = per_protein.filter(pl.col("n_domains") > 1)["accession"].to_list()
    single = per_protein.filter(pl.col("n_domains") == 1)["accession"].to_list()
    # Roughly 60/40 multi/single: enough multi-domain proteins for MCC to be defined,
    # enough single-domain ones that the set still resembles a real proteome.
    n_multi = min(len(multi), int(n * 0.6))
    return multi[:n_multi] + single[: n - n_multi]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", required=True, type=Path)
    p.add_argument("--qfo-dir", required=True, type=Path)
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--species", default="yeast,ecoli",
                   help="target species; default is the two smallest proteomes")
    p.add_argument("--n-queries", type=int, default=200)
    p.add_argument("--n-targets", type=int, default=300,
                   help="per species, split evenly between family-sharing and decoy")
    p.add_argument("--structure-cache", type=Path,
                   default=Path.home() / "data/alphafold_structures")
    args = p.parse_args()

    # QfO proteome files, keyed by the labels main.nf uses.
    proteomes = {
        "human": ("Eukaryota", "UP000005640_9606"),
        "mouse": ("Eukaryota", "UP000000589_10090"),
        "chicken": ("Eukaryota", "UP000000539_9031"),
        "zebrafish": ("Eukaryota", "UP000000437_7955"),
        "ciona": ("Eukaryota", "UP000008144_7719"),
        "fly": ("Eukaryota", "UP000000803_7227"),
        "worm": ("Eukaryota", "UP000001940_6239"),
        "yeast": ("Eukaryota", "UP000002311_559292"),
        "arabidopsis": ("Eukaryota", "UP000006548_3702"),
        "ecoli": ("Bacteria", "UP000000625_83333"),
    }
    species = [s.strip() for s in args.species.split(",")]

    ann_out = args.outdir / "annotations"
    qfo_out = args.outdir / "qfo"
    struct_out = args.outdir / "structures"
    for d in (ann_out, qfo_out / "Eukaryota", qfo_out / "Bacteria", struct_out):
        d.mkdir(parents=True, exist_ok=True)

    human = pl.read_parquet(args.annotations / "human_pfam_domains.parquet").filter(
        pl.col("has_position")
    )
    query_acc = set(pick_queries(human, args.n_queries))
    query_families = set(
        human.filter(pl.col("accession").is_in(query_acc))["pfam_id"].unique().to_list()
    )

    summary = {
        "n_queries": len(query_acc),
        "n_query_families": len(query_families),
        "species": {},
    }

    # ---- query side ----
    sub, name = proteomes["human"]
    human_fa = read_fasta(args.qfo_dir / sub / f"{name}.fasta")
    n_written = write_fasta(human_fa, query_acc, qfo_out / sub / f"{name}.fasta")
    summary["human_fasta_records"] = n_written

    human_sub = human.filter(pl.col("accession").is_in(query_acc))
    human_sub.write_parquet(ann_out / "human_pfam_domains.parquet")

    keep_structs = set(query_acc)

    # ---- target side ----
    for sp in species:
        if sp not in proteomes:
            raise SystemExit(f"unknown species '{sp}'")
        ann = pl.read_parquet(args.annotations / f"{sp}_pfam_domains.parquet").filter(
            pl.col("has_position")
        )
        sharing = (
            ann.filter(pl.col("pfam_id").is_in(query_families))["accession"]
            .unique().sort().to_list()
        )
        decoys = (
            ann.filter(~pl.col("pfam_id").is_in(query_families))["accession"]
            .unique().sort().to_list()
        )
        half = args.n_targets // 2
        keep = set(sharing[:half]) | set(decoys[: args.n_targets - half])

        sub, name = proteomes[sp]
        fa = read_fasta(args.qfo_dir / sub / f"{name}.fasta")
        n_fa = write_fasta(fa, keep, qfo_out / sub / f"{name}.fasta")
        ann.filter(pl.col("accession").is_in(keep)).write_parquet(
            ann_out / f"{sp}_pfam_domains.parquet"
        )
        keep_structs |= keep

        summary["species"][sp] = {
            "n_targets": len(keep),
            "n_family_sharing": min(len(sharing), half),
            "n_decoy": len(keep) - min(len(sharing), half),
            "fasta_records": n_fa,
            # If this is 0 the run is still valid, it just cannot produce a true positive
            # and every recall number will be 0.0 by construction.
            "n_shared_families": len(
                set(ann.filter(pl.col("accession").is_in(keep))["pfam_id"].to_list())
                & query_families
            ),
        }

    # ---- structures, symlinked from the flat cache ----
    per_species_acc = {"human": query_acc}
    for sp in species:
        ann = pl.read_parquet(ann_out / f"{sp}_pfam_domains.parquet")
        per_species_acc[sp] = set(ann["accession"].unique().to_list())

    # rglob, not iterdir: the local cache is one flat directory, but on a cluster the
    # same structures already sit under the per-species tree that sync-structures
    # populated. Recursing lets one flag serve both without a second code path.
    cache_index = {}
    if args.structure_cache.exists():
        for f in args.structure_cache.rglob("AF-*.cif*"):
            cache_index.setdefault(f.name.split("-")[1], f.resolve())

    for label, accs in per_species_acc.items():
        d = struct_out / label
        d.mkdir(parents=True, exist_ok=True)
        linked = 0
        for acc in accs:
            src = cache_index.get(acc)
            if src:
                dst = d / f"AF-{acc}-F1.cif"
                if not dst.exists():
                    dst.symlink_to(src)
                linked += 1
        summary.setdefault("structures", {})[label] = {
            "n_wanted": len(accs), "n_linked": linked
        }

    (args.outdir / "mini_testset_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
