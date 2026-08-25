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
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gene_sets as gs  # noqa: E402


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
    # Three spellings for one thing. Nextflow params use underscores
    # (--target_species), argparse conventionally uses dashes (--target-species), and
    # --species is what this script shipped with. Accepting all three means a Makefile
    # recipe cannot pass the wrong one, which is what broke mini-testset-sherlock
    # when the Nextflow param was renamed and the rename leaked onto this python call.
    #
    # As everywhere else in this pipeline: human is the query, these are the TARGETS.
    p.add_argument("--target-species", "--target_species", "--species",
                   dest="species", default="yeast,ecoli",
                   help="TARGET proteomes to search human against; "
                        "default is the two smallest")
    p.add_argument("--n-queries", type=int, default=200)
    p.add_argument("--n-targets", type=int, default=300,
                   help="per species, split evenly between family-sharing and decoy")
    p.add_argument("--structure-cache", type=Path,
                   default=Path.home() / "data/alphafold_structures")
    p.add_argument("--gene-set", choices=["default", "mhc", "chr6"], default="default",
                   help="mhc restricts queries to the MHC region genes of notebooks 210-216; "
                        "chr6 takes every HGNC gene on chromosome 6 (the MHC's chromosome), "
                        "which is the 'midi' set -- a real query load, not a smoke test")
    p.add_argument("--full-targets", action="store_true",
                   help="do not subset the target proteomes: symlink the real FASTAs, "
                        "annotations and structure directories straight through. Only the "
                        "QUERY side is cut down. This is what makes a midi run comparable "
                        "to the full one -- the targets are literally the same files.")
    p.add_argument("--hgnc", type=Path,
                   help="HGNC table, required by --gene-set mhc to map symbols to accessions")
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

    if args.gene_set == "mhc":
        # The 25 genes notebooks 210-216 score, mapped symbol -> accession through HGNC's
        # own uniprot_ids column. Deliberately NOT size-matched to the default set: this is
        # a vignette for the MHC figures and a pipeline smoke test, and an n=25 stratum
        # must not be made to look like a population statistic by padding it.
        if not args.hgnc or not args.hgnc.exists():
            raise SystemExit("--gene-set mhc requires --hgnc")
        hgnc = pl.read_csv(args.hgnc, separator="\t", infer_schema_length=0,
                           quote_char=None, null_values=[""])
        hgnc = (
            hgnc.select("symbol", "uniprot_ids")
            .filter(pl.col("uniprot_ids").is_not_null())
            .with_columns(pl.col("uniprot_ids").str.split("|").alias("accession"))
            .explode("accession")
            .with_columns(pl.col("accession").str.strip_chars())
        )
        wanted = set(gs.MHC_CLASSES)
        mhc_acc = set(
            hgnc.filter(pl.col("symbol").is_in(wanted))["accession"].to_list()
        )
        query_acc = mhc_acc & set(human["accession"].to_list())
        missing = sorted(
            wanted - set(hgnc.filter(pl.col("accession").is_in(query_acc))["symbol"].to_list())
        )
        if missing:
            # Reported dropped: a gene absent from the Pfam annotation subset
            # cannot contribute a true positive, and which ones are missing changes what an
            # MHC claim covers.
            print(f"NOTE: {len(missing)} MHC genes absent from the Pfam annotation set: "
                  f"{', '.join(missing)}")
    elif args.gene_set == "chr6":
        # Every HGNC gene whose cytogenetic location starts 6p or 6q. Anchored on purpose:
        # a bare startswith("6") also catches nothing useful, but a substring match would
        # pull in 16q and 6 is not a prefix of 16 only by luck of ordering.
        if not args.hgnc or not args.hgnc.exists():
            raise SystemExit("--gene-set chr6 requires --hgnc")
        hgnc = pl.read_csv(args.hgnc, separator="\t", infer_schema_length=0,
                           quote_char=None, null_values=[""])
        chr6 = (
            hgnc.filter(pl.col("location").is_not_null()
                        & pl.col("location").str.contains(r"^6[pq]"))
            .select("symbol", "uniprot_ids")
            .filter(pl.col("uniprot_ids").is_not_null())
            .with_columns(pl.col("uniprot_ids").str.split("|").alias("accession"))
            .explode("accession")
            .with_columns(pl.col("accession").str.strip_chars())
        )
        query_acc = set(chr6["accession"].to_list()) & set(human["accession"].to_list())
        print(f"NOTE: chr6 query set: {len(query_acc)} proteins carrying "
              f"{human.filter(pl.col('accession').is_in(list(query_acc))).height} domain "
              f"instances")
    else:
        query_acc = set(pick_queries(human, args.n_queries))
    query_families = set(
        human.filter(pl.col("accession").is_in(query_acc))["pfam_id"].unique().to_list()
    )

    summary = {
        "gene_set": args.gene_set,
        "full_targets": args.full_targets,
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
    def link(src: Path, dst: Path) -> None:
        """Symlink src to dst, replacing whatever is there so a rebuild is idempotent."""
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        elif dst.is_dir():
            # unlink() raises on a real directory, so a second build over a first one died
            # here rather than replacing it.
            shutil.rmtree(dst)
        dst.symlink_to(src.resolve())

    for sp in species:
        if sp not in proteomes:
            raise SystemExit(f"unknown species '{sp}'")

        if args.full_targets:
            # Nothing is copied or filtered: the pipeline reads the same target files the
            # full run reads. Every target-side database the pipeline caches under storeDir
            # -- prostt5Db, foldseekDb, mmseqsDb, reseekConvert, folddiscoIndex -- is
            # therefore built from identical input. Pass --db_cache pointing at the full
            # run's outdir and the full run reuses them directly; without it each run keeps
            # its own copies and ProstT5 pays for nine proteomes twice.
            sub, name = proteomes[sp]
            link(args.qfo_dir / sub / f"{name}.fasta", qfo_out / sub / f"{name}.fasta")
            link(args.annotations / f"{sp}_pfam_domains.parquet",
                 ann_out / f"{sp}_pfam_domains.parquet")
            n_ann = pl.read_parquet(args.annotations / f"{sp}_pfam_domains.parquet").filter(
                pl.col("has_position")
            )
            summary["species"][sp] = {
                "full_target": True,
                "n_targets": n_ann["accession"].n_unique(),
                "n_domain_instances": n_ann.height,
                "n_shared_families": len(
                    set(n_ann["pfam_id"].to_list()) & query_families
                ),
            }
            continue

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
    # With --full-targets the species structure directories are passed through whole rather
    # than rebuilt link by link: there are ~20k files per proteome and the pipeline wants
    # all of them. Only the human side is curated down to the query accessions.
    if args.full_targets:
        # A real directory of per-file symlinks, NOT one symlink to the species directory.
        # Two reasons. Nextflow's file(...).list() does not follow a directory symlink, so
        # the pipeline's own has_structs check reports an empty directory (fixed there too,
        # but not worth depending on). And this is exactly how the full run's
        # data/structures/<sp> is built by fetch_alphafold_structures.sh, so the structure
        # arms stage identical input in both runs instead of two different shapes.
        for sp in species:
            src = args.structure_cache / sp
            if not src.is_dir():
                print(f"NOTE: no structure directory at {src} -- the structure arms will "
                      f"skip {sp}. Fetch it with `make fetch-structures SPECIES_ONE={sp}`.")
                continue
            d = struct_out / sp
            d.mkdir(parents=True, exist_ok=True)
            n = 0
            for f in src.iterdir():
                if not f.name.startswith("AF-"):
                    continue
                dst = d / f.name
                if not dst.exists() and not dst.is_symlink():
                    dst.symlink_to(f.resolve())
                n += 1
            summary.setdefault("structures", {})[sp] = {"n_linked": n, "full_target": True}
            print(f"NOTE: {sp}: linked {n} structures from {src}")
        per_species_acc = {"human": query_acc}
    else:
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

    (args.outdir / f"{'midi' if args.full_targets else 'mini'}_testset_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
