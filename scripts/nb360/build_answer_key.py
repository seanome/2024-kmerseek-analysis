#!/usr/bin/env python3
"""Build notebook 360's search database and answer key for one organism.

Rules are in notebooks/360_data/preregistered.md, sections "Answer key" and choice 3.

Inputs (all paths on the command line):
  --pep, --cdna       Ensembl peptide and cDNA FASTA (gzipped) of the organism
  --regions           Pfam 38.2 regions, already cut down to the MD5s we need (TSV with
                      the header of Pfam-A.regions.tsv.gz)
  --clans             Pfam-A.clans.tsv.gz (family, clan, ...)
  --sprot             Swiss-Prot table written by parse_swissprot.py (parquet: accession,
                      md5, genera, sequence)
  --exclude-genus     genus whose proteins leave the search database (Ciona, Danio)

Outputs under --out:
  search_db.fasta     Swiss-Prot proteins with at least one Pfam row, minus the genus
  search_db.pfam.tsv  their Pfam domains, 0-based start, end exclusive (kmerseek's
                      coordinates), with the clan
  answer_key.tsv      the 200 genes, their family, status and fold change
  transcripts.fa      the 200 cDNAs, named by transcript id, for the simulator
  proteins.fa         the 200 peptides
  provenance.txt      counts at every filter step
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import re
from pathlib import Path

import numpy as np
import polars as pl

SEED = 360
N_GENES = 200
N_UP = 50
N_DOWN = 50
FOLD_UP = 4.0
FOLD_DOWN = 0.25
CDNA_MIN = 500
CDNA_MAX = 5000


def read_fasta(path: Path) -> list[tuple[str, str]]:
    """(header line without '>', sequence) for every record of a gzipped FASTA."""
    records, header, chunks = [], None, []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks)))
                header, chunks = line[1:], []
            else:
                chunks.append(line.strip())
    if header is not None:
        records.append((header, "".join(chunks)))
    return records


def header_field(header: str, key: str) -> str | None:
    m = re.search(rf"\b{key}:(\S+)", header)
    return m.group(1) if m else None


def md5(seq: str) -> str:
    return hashlib.md5(seq.upper().encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--organism", required=True)
    ap.add_argument("--pep", type=Path, required=True)
    ap.add_argument("--cdna", type=Path, required=True)
    ap.add_argument("--regions", type=Path, required=True)
    ap.add_argument("--clans", type=Path, required=True)
    ap.add_argument("--sprot", type=Path, required=True)
    ap.add_argument("--exclude-genus", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    log: list[str] = [f"organism\t{args.organism}", f"seed\t{SEED}"]

    regions = pl.read_csv(args.regions, separator="\t", schema_overrides={"md5": pl.Utf8})
    clans = pl.read_csv(
        args.clans, separator="\t", has_header=False, new_columns=["pfam_acc", "clan", "clan_id", "pfam_id", "pfam_desc"]
    ).select("pfam_acc", pl.col("clan").fill_null(""))
    # A family with no clan is its own clan.
    clans = clans.with_columns(pl.when(pl.col("clan") == "").then(pl.col("pfam_acc")).otherwise(pl.col("clan")).alias("clan"))
    regions = regions.rename({"pfamA_acc": "pfam_acc"}).join(clans, on="pfam_acc", how="left").with_columns(
        pl.col("clan").fill_null(pl.col("pfam_acc"))
    )

    # --- Search database: Swiss-Prot with Pfam rows (same accession and same sequence), minus the genus.
    sprot = pl.read_parquet(args.sprot)
    log.append(f"swissprot_proteins\t{sprot.height}")
    sp_regions = regions.join(sprot.select(pl.col("accession").alias("pfamseq_acc"), "md5"), on=["pfamseq_acc", "md5"], how="inner")
    with_pfam = sprot.filter(pl.col("accession").is_in(sp_regions["pfamseq_acc"].unique().implode()))
    log.append(f"swissprot_with_pfam_rows\t{with_pfam.height}")
    excluded = with_pfam.filter(pl.col("genera").list.contains(args.exclude_genus))
    db = with_pfam.filter(~pl.col("genera").list.contains(args.exclude_genus)).sort("accession")
    log.append(f"swissprot_excluded_genus_{args.exclude_genus}\t{excluded.height}")
    log.append(f"search_db_proteins\t{db.height}")
    with open(args.out / "search_db.fasta", "w") as fh:
        for acc, seq in db.select("accession", "sequence").iter_rows():
            fh.write(f">{acc}\n{seq}\n")
    db_domains = (
        sp_regions.filter(pl.col("pfamseq_acc").is_in(db["accession"].implode()))
        .select(
            pl.col("pfamseq_acc").alias("target"),
            "pfam_acc",
            "clan",
            (pl.col("seq_start") - 1).alias("start0"),
            pl.col("seq_end").alias("end"),
        )
        .sort("target", "start0", "pfam_acc")
    )
    db_domains.write_csv(args.out / "search_db.pfam.tsv", separator="\t")
    db_families = set(db_domains["pfam_acc"].unique().to_list())
    log.append(f"search_db_domains\t{db_domains.height}")
    log.append(f"search_db_families\t{len(db_families)}")

    # --- Candidate genes.
    cdna = {h.split()[0]: s for h, s in read_fasta(args.cdna)}
    rows = []
    for h, s in read_fasta(args.pep):
        rows.append(
            dict(
                peptide_id=h.split()[0],
                gene_id=header_field(h, "gene"),
                transcript_id=header_field(h, "transcript"),
                gene_symbol=header_field(h, "gene_symbol") or "",
                md5=md5(s.rstrip("*")),
                protein=s.rstrip("*"),
            )
        )
    pep = pl.DataFrame(rows)
    log.append(f"ensembl_peptides\t{pep.height}")
    log.append(f"ensembl_genes\t{pep['gene_id'].n_unique()}")
    pep = pep.with_columns(
        pl.col("transcript_id").map_elements(lambda t: len(cdna.get(t, "")), return_dtype=pl.Int64).alias("cdna_length")
    )
    org_regions = regions.join(pep.select("md5").unique(), on="md5", how="inner")
    per_md5 = org_regions.group_by("md5").agg(
        pl.col("pfam_acc").n_unique().alias("n_families"),
        pl.col("pfam_acc").first(),
        pl.col("clan").first(),
        pl.len().alias("n_domain_rows"),
        pl.col("seq_start").min().alias("domain_start1"),
        pl.col("seq_end").max().alias("domain_end1"),
    )
    cand = pep.join(per_md5, on="md5", how="inner")
    log.append(f"genes_with_pfam_md5_match\t{cand['gene_id'].n_unique()}")
    cand = cand.filter(pl.col("n_families") == 1)
    log.append(f"genes_one_family\t{cand['gene_id'].n_unique()}")
    cand = cand.filter(pl.col("cdna_length").is_between(CDNA_MIN, CDNA_MAX))
    log.append(f"genes_cdna_{CDNA_MIN}_{CDNA_MAX}\t{cand['gene_id'].n_unique()}")
    cand = cand.filter(pl.col("pfam_acc").is_in(list(db_families)))
    log.append(f"genes_family_in_search_db\t{cand['gene_id'].n_unique()}")
    # One transcript per gene: longest cDNA, then transcript id, so the pick is total.
    cand = cand.sort(["gene_id", "cdna_length", "transcript_id"], descending=[False, True, False]).unique(
        "gene_id", keep="first", maintain_order=True
    )

    # --- Draw 200 genes, one per clan, then the statuses.
    rng = np.random.default_rng(SEED)
    order = rng.permutation(cand.height)
    picked, used_clans = [], set()
    for i in order:
        clan = cand["clan"][int(i)]
        if clan in used_clans:
            continue
        used_clans.add(clan)
        picked.append(int(i))
        if len(picked) == N_GENES:
            break
    log.append(f"clans_available\t{cand['clan'].n_unique()}")
    if len(picked) < N_GENES:
        raise SystemExit(f"only {len(picked)} genes with distinct clans; need {N_GENES}")
    key = cand[picked]
    status = np.array(["unchanged"] * N_GENES, dtype=object)
    s = rng.permutation(N_GENES)
    status[s[:N_UP]] = "up"
    status[s[N_UP : N_UP + N_DOWN]] = "down"
    fold = np.where(status == "up", FOLD_UP, np.where(status == "down", FOLD_DOWN, 1.0))
    key = key.with_columns(pl.Series("status", status.tolist()), pl.Series("fold_change", fold.tolist()))
    key = key.select(
        "gene_id", "transcript_id", "peptide_id", "gene_symbol", "pfam_acc", "clan", "status", "fold_change",
        "cdna_length", pl.col("protein").str.len_chars().alias("protein_length"), "n_domain_rows",
        "domain_start1", "domain_end1", "md5",
    ).sort("transcript_id")
    key.write_csv(args.out / "answer_key.tsv", separator="\t")
    with open(args.out / "transcripts.fa", "w") as fh:
        for t in key["transcript_id"]:
            fh.write(f">{t}\n{cdna[t]}\n")
    prot = dict(zip(pep["transcript_id"], pep["protein"]))
    with open(args.out / "proteins.fa", "w") as fh:
        for t, g in key.select("transcript_id", "gene_id").iter_rows():
            fh.write(f">{t}\n{prot[t]}\n")
    for st in ("up", "down", "unchanged"):
        log.append(f"key_{st}\t{(key['status'] == st).sum()}")
    log.append(f"key_distinct_families\t{key['pfam_acc'].n_unique()}")
    log.append(f"key_distinct_clans\t{key['clan'].n_unique()}")
    (args.out / "provenance.txt").write_text("\n".join(log) + "\n")
    print("\n".join(log))


if __name__ == "__main__":
    main()
