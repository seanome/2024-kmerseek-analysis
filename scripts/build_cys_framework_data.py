#!/usr/bin/env python3
"""Build the query and reference sets for the cysteine-framework experiment.

Reads the four downloads (ConoServer protein XML, Swiss-Prot FASTA, the Swiss-Prot query
taxa TSV, and a taxid-to-genus-and-phylum table built from the NCBI taxonomy dump) and
writes the tables the notebook and the pipeline read. Every count it prints is also
written to a table, so a number in the notebook comes from a committed file.

Usage:
    build_cys_framework_data.py <download-dir> <out-dir>

where <download-dir> holds conoserver_raw/, swissprot_raw/ and taxid_genus_phylum.tsv.
"""

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parents[1] / "notebooks"))

import cys_framework_utils as cfu  # noqa: E402

#: The lineages the Swiss-Prot queries come from, named for the notebook's figures.
QUERY_LINEAGES = {
    "Araneae": "spiders",
    "Scorpiones": "scorpions",
    "Actiniaria": "sea anemones",
    "Conus": "cone snails",
}


def main(download_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    taxonomy = cfu.read_taxonomy(download_dir / "taxid_genus_phylum.tsv")
    print(f"taxonomy: {len(taxonomy)} taxids")

    # --- ConoServer -------------------------------------------------------------------
    entries = cfu.read_conoserver(download_dir / "conoserver_raw" / "protein.xml")
    fasta = cfu.read_fasta(download_dir / "conoserver_raw" / "conoserver_protein.fa.gz")
    assert entries.height == len(fasta), (
        f"ConoServer XML has {entries.height} entries but its FASTA has {len(fasta)}; "
        "the XML parse lost entries"
    )
    print(f"ConoServer: {entries.height} entries, matching its FASTA")

    cono_queries, cono_dropped = cfu.conoserver_queries(entries)
    cono_queries = cono_queries.with_columns(
        phylum=pl.lit("Mollusca"),
        framework_from_sequence=pl.col("peptide").map_elements(
            cfu.framework_string, return_dtype=pl.Utf8
        ),
    )
    print(f"ConoServer queries: {cono_queries.height}")
    print(cono_dropped)

    # --- Swiss-Prot queries ------------------------------------------------------------
    table = cfu.read_swissprot_table(download_dir / "swissprot_raw" / "querytaxa.tsv")
    sprot_queries, sprot_dropped = cfu.swissprot_queries(table, taxonomy)
    sprot_queries = sprot_queries.with_columns(
        framework_from_sequence=pl.col("framework"),
    )
    print(
        f"Swiss-Prot query entries read: {table.height}, queries kept: {sprot_queries.height}"
    )
    print(sprot_dropped)

    # --- One query table ---------------------------------------------------------------
    shared = [
        "query_id",
        "source",
        "organism",
        "genus",
        "phylum",
        "peptide",
        "peptide_length",
        "framework_from_sequence",
    ]

    # peptide_length comes back as UInt32 from one source and Int64 from the other, so
    # both are cast before concat rather than letting the schemas clash.
    def shaped(frame: pl.DataFrame, label: pl.Expr) -> pl.DataFrame:
        return frame.with_columns(
            framework_label=label,
            peptide_length=pl.col("peptide_length").cast(pl.Int64),
        ).select(*shared, "framework_label")

    queries = pl.concat(
        [
            shaped(cono_queries, pl.col("framework")),
            shaped(sprot_queries, pl.lit(None, dtype=pl.Utf8)),
        ]
    )
    # The framework read off the residues is the one every method is scored on, so it is
    # the column the rest of the analysis uses. ConoServer's own label is kept beside it
    # for the cross-check in the notebook.
    queries = queries.rename({"framework_from_sequence": "framework"}).with_columns(
        n_cysteines=pl.col("peptide").str.count_matches("C"),
    )

    # ConoServer numbers its own entries P00001, P00002 and so on, which is exactly the
    # shape of a UniProt accession, and three of those numbers are in use on both sides
    # for unrelated peptides: P05484 is omega-conotoxin MVIIA in Swiss-Prot and an
    # 11-residue Conus marmoreus peptide in ConoServer. Joining results on a bare id would
    # merge the two. Every query id therefore carries its source, and `source_id` keeps
    # the original for looking an entry up in the database it came from.
    queries = queries.with_columns(
        source_id=pl.col("query_id"),
        query_id=pl.when(pl.col("source") == "ConoServer")
        .then("cono:" + pl.col("query_id"))
        .otherwise("sp:" + pl.col("query_id")),
    )
    collisions = queries.group_by("source_id").len().filter(pl.col("len") > 1)
    print(f"source ids used by both databases: {collisions.height}")
    assert queries["query_id"].n_unique() == queries.height, "query_id is not unique"

    # Keep only short disulfide-rich peptides, which is what the experiment asks about.
    all_queries = queries
    queries = queries.filter(
        (pl.col("peptide_length") >= cfu.MIN_PEPTIDE_LENGTH)
        & (pl.col("peptide_length") <= cfu.MAX_PEPTIDE_LENGTH)
        & (pl.col("n_cysteines") >= cfu.MIN_CYSTEINES)
    )
    too_long = all_queries.filter(
        pl.col("peptide_length") > cfu.MAX_PEPTIDE_LENGTH
    ).height
    too_short = all_queries.filter(
        pl.col("peptide_length") < cfu.MIN_PEPTIDE_LENGTH
    ).height
    too_few = all_queries.filter(
        (pl.col("peptide_length") >= cfu.MIN_PEPTIDE_LENGTH)
        & (pl.col("peptide_length") <= cfu.MAX_PEPTIDE_LENGTH)
        & (pl.col("n_cysteines") < cfu.MIN_CYSTEINES)
    ).height
    sprot_dropped = pl.concat(
        [
            sprot_dropped,
            pl.DataFrame(
                [
                    {
                        "reason": f"longer than {cfu.MAX_PEPTIDE_LENGTH} aa",
                        "n": too_long,
                    },
                    {
                        "reason": f"shorter than {cfu.MIN_PEPTIDE_LENGTH} aa",
                        "n": too_short,
                    },
                    {
                        "reason": f"fewer than {cfu.MIN_CYSTEINES} cysteines",
                        "n": too_few,
                    },
                ]
            ).filter(pl.col("n") > 0),
        ]
    )
    print(
        f"\nmature peptides built: {all_queries.height}; short disulfide-rich peptides "
        f"({cfu.MIN_PEPTIDE_LENGTH}-{cfu.MAX_PEPTIDE_LENGTH} aa, >= {cfu.MIN_CYSTEINES} C): "
        f"{queries.height}"
    )
    queries.write_csv(out_dir / "queries.csv")
    all_queries.write_csv(out_dir / "queries_all_mature_peptides.csv")
    print(f"queries.csv: {queries.height} rows")
    print(queries.group_by("source").len().sort("source"))

    dropped = pl.concat(
        [
            cono_dropped.with_columns(source=pl.lit("ConoServer")),
            sprot_dropped.with_columns(source=pl.lit("Swiss-Prot")),
        ]
    )
    dropped.write_csv(out_dir / "queries_dropped.csv")

    # --- Reference --------------------------------------------------------------------
    sprot = cfu.read_fasta(download_dir / "swissprot_raw" / "uniprot_sprot.fasta.gz")
    accession_taxid, by_accession = {}, {}
    for header, sequence in sprot.items():
        accession, taxid = cfu.swissprot_header_fields(header)
        accession_taxid[header] = taxid
        by_accession[accession] = header
    print(
        f"\nSwiss-Prot: {len(sprot)} entries, {len(by_accession)} distinct accessions"
    )

    toxin_accessions = set()
    with open(download_dir / "swissprot_raw" / "kw0800.tsv") as handle:
        next(handle)
        for line in handle:
            toxin_accessions.add(line.split("\t")[0])
    in_fasta = {a for a in toxin_accessions if a in by_accession}
    print(
        f"Toxin keyword (KW-0800): {len(toxin_accessions)} accessions, "
        f"{len(in_fasta)} of them in the FASTA"
    )
    assert len(in_fasta) == len(toxin_accessions), (
        f"{len(toxin_accessions) - len(in_fasta)} toxin accessions are missing from the "
        "Swiss-Prot FASTA; the two downloads are from different releases"
    )

    # One row per Swiss-Prot entry, with the genus and phylum resolved once here so that
    # the pipeline never re-reads the taxonomy dump, and the Toxin flag that decides a
    # true positive. An entry whose taxid has no genus or phylum rank gets an empty cell,
    # and an empty cell matches no excluded clade, so such an entry stays in every
    # reference.
    pl.DataFrame(
        {
            "header": list(sprot),
            "accession": [cfu.swissprot_header_fields(h)[0] for h in sprot],
            "taxid": [accession_taxid[h] for h in sprot],
            "genus": [
                (taxonomy.get(accession_taxid[h], (None, None))[0] or "") for h in sprot
            ],
            "phylum": [
                (taxonomy.get(accession_taxid[h], (None, None))[1] or "") for h in sprot
            ],
            "length": [len(s) for s in sprot.values()],
            "is_toxin": [
                cfu.swissprot_header_fields(h)[0] in toxin_accessions for h in sprot
            ],
        }
    ).write_csv(out_dir / "reference_entries.csv")

    # How big each held-out reference is, and how many toxins it leaves to find.
    rows = []
    for level in ("genus", "phylum"):
        for value in sorted(set(queries[level].to_list()) - {""}):
            kept = cfu.held_out_reference(
                sprot, accession_taxid, taxonomy, value, level
            )
            toxins = sum(
                1 for h in kept if cfu.swissprot_header_fields(h)[0] in toxin_accessions
            )
            rows.append(
                {
                    "level": level,
                    "excluded": value,
                    "n_reference": len(kept),
                    "n_removed": len(sprot) - len(kept),
                    "n_toxins_left": toxins,
                    "n_queries": queries.filter(pl.col(level) == value).height,
                }
            )
    references = pl.DataFrame(rows).sort(["level", "excluded"])
    references.write_csv(out_dir / "held_out_references.csv")
    print("\nheld-out references:")
    print(references)

    settings = cfu.settings_table(queries)
    settings.write_csv(out_dir / "settings.csv")
    print("\nsettings:")
    print(settings)

    counts = (
        queries.group_by(["source", "framework"])
        .len()
        .rename({"len": "n_queries"})
        .sort(["source", "n_queries"], descending=[False, True])
    )
    counts.write_csv(out_dir / "queries_per_framework.csv")
    print(f"\ndistinct frameworks: {counts.height} source-framework pairs")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
